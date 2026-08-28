//! Local CTC forced alignment: refine transcript word timings on this machine.
//!
//! Opt-in per transcription provider (`ProviderConfig::align_locally`, off by
//! default). When enabled and the model files are installed, `run_transcription`
//! passes each transcribed chunk's 16 kHz mono audio plus its words through
//! [`Engine::refine`], which re-times every word against the acoustics via CTC
//! Viterbi forced alignment. Only `start`/`end` change — never the word list
//! itself, whose indices are load-bearing downstream (`Span::first_word`,
//! `cut_ranges_json` neighbour lookups, editor payload positions).
//!
//! Why this exists (measured, not assumed — see the Phase-0 probe tests below):
//! some transcription endpoints already serve aligner-grade timestamps, but
//! vanilla Whisper endpoints are 50-200 ms off, which is the difference between
//! a clean cut and a clipped consonant. CTC emissions are "peaky" — the first
//! emission of a fricative-initial word lags its acoustic onset by 30-80 ms —
//! so this layer deliberately stops at word-level refinement and leaves final
//! boundary placement to the DSP layer (`SpeechAnalysis::word_extent` /
//! `place_removal`), which extends from these hints to the physical onset.
//!
//! Runtime pieces, all loaded at run time from `<resource path>/ReaLackey/models`
//! (nothing is linked into the shipped binary):
//!   * ONNX Runtime as a dynamic library, loaded by ABSOLUTE path via `ort`'s
//!     `load-dynamic` feature. Never by name: inside reaper.exe the DLL search
//!     order finds the stale `onnxruntime.dll` Windows ships in System32.
//!   * The MMS-300m forced-aligner model (int8 ONNX, ~303 MB). Its weights are
//!     CC-BY-NC — fine for this non-commercial project. Vocabulary is lowercase
//!     a-z + apostrophe (blank id 0), so any Latin-romanizable language aligns
//!     without a G2P step.
//!
//! Both files are fetched on demand (see `download_items`) or installed from the
//! "with-models" release bundle for machines where a 300 MB download is not an
//! option. Downloads are integrity-checked against pinned SHA-256 digests —
//! the runtime library is executable code.

#![allow(clippy::needless_range_loop)] // trellis code reads clearer indexed

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use crate::providers::transcription::Word;

/// wav2vec2 stride: 320 samples at 16 kHz per emission frame.
const FRAME: f64 = 0.02;
/// The model's fixed input rate; `__transcribe_chunk` renders at this rate.
const SR: f64 = 16_000.0;
/// CTC blank id in the model's vocabulary.
const BLANK: usize = 0;
/// Seconds of words per aligned window. Whole-chunk alignment is quadratic in
/// memory (the backtrack matrix is frames x states); 30-second windows keep it
/// around a megabyte while giving CTC ample context.
const MAX_WINDOW: f64 = 28.0;
/// Acoustic context added around each window so edge words aren't clipped.
const PAD: f64 = 1.0;
/// A boundary that moves further than this from the transcript's is distrusted
/// (a mis-transcribed word drags its neighbours; keep the transcript's timing).
const MAX_SHIFT: f64 = 0.6;

/// `Engine::refine`'s error message when stopped via its cancel flag.
pub const CANCELLED: &str = "cancelled";

// ---- installed files ---------------------------------------------------------

/// Model file name under `models_dir()`. Named for what it is, not the upstream
/// "model_int8.onnx", so the directory stays legible when more models arrive.
pub const MODEL_FILE: &str = "mms300m-align-int8.onnx";
// Pinned to a COMMIT revision, not `main`: HF refs are mutable, and an upstream
// re-quantization would otherwise turn every download into a full 317 MB fetch
// followed by a checksum rejection — hostile to exactly the limited-data users
// this feature caters to. This revision serves the bytes MODEL_SHA256 pins.
const MODEL_URL: &str = "https://huggingface.co/onnx-community/mms-300m-1130-forced-aligner-ONNX/resolve/2100fb247d8e43962eef24491597fbeb8b469531/onnx/model_int8.onnx";
const MODEL_SIZE: u64 = 317_341_664;
const MODEL_SHA256: &str = "2eb5c3d2f6db2ef476aa7a7e1e5800145973e9064eb5292b1b9b8ada1207712a";

// The ONNX Runtime dynamic library, version-matched to the `ort` crate (rc.13
// targets 1.28.0). Downloaded from this repo's fixed `align-deps-1` release —
// flat files repackaged from Microsoft's official archives by the
// `align-deps.yml` workflow — because the official downloads are zip/tgz
// archives and shipping an unzipper for one file is not worth it.
#[cfg(target_os = "windows")]
pub const RUNTIME_FILE: &str = "onnxruntime.dll";
#[cfg(target_os = "windows")]
const RUNTIME_URL: &str =
    "https://github.com/Timtam/ReaLackey/releases/download/align-deps-1/onnxruntime-win-x64-1.28.0.dll";
#[cfg(target_os = "windows")]
const RUNTIME_SIZE: u64 = 15_809_848;
#[cfg(target_os = "windows")]
const RUNTIME_SHA256: &str = "18370c375f07357fa5874344a9d9ac17e6b6fe1eb18b1dd209d79483b4470257";

#[cfg(target_os = "macos")]
pub const RUNTIME_FILE: &str = "libonnxruntime.1.28.0.dylib";
#[cfg(target_os = "macos")]
const RUNTIME_URL: &str =
    "https://github.com/Timtam/ReaLackey/releases/download/align-deps-1/libonnxruntime-osx-arm64-1.28.0.dylib";
#[cfg(target_os = "macos")]
const RUNTIME_SIZE: u64 = 39_312_136;
#[cfg(target_os = "macos")]
const RUNTIME_SHA256: &str = "dc19bbcb2f5c9fb3c68b4f9248aa0a35065ff702c5dbeae75eac54a74da97b6d";

// Linux builds compile but have no packaged runtime (we don't ship Linux).
#[cfg(not(any(target_os = "windows", target_os = "macos")))]
pub const RUNTIME_FILE: &str = "libonnxruntime.so";
#[cfg(not(any(target_os = "windows", target_os = "macos")))]
const RUNTIME_URL: &str = "";
#[cfg(not(any(target_os = "windows", target_os = "macos")))]
const RUNTIME_SIZE: u64 = 0;
#[cfg(not(any(target_os = "windows", target_os = "macos")))]
const RUNTIME_SHA256: &str = "";

/// Whether local alignment can run on this machine at all. ONNX Runtime stopped
/// shipping Intel-mac builds, so on macOS the answer is Apple Silicon only —
/// checked at run time because the mac binary is universal.
pub fn platform_support() -> Result<(), &'static str> {
    #[cfg(target_os = "windows")]
    {
        Ok(())
    }
    #[cfg(target_os = "macos")]
    {
        if std::env::consts::ARCH == "aarch64" {
            Ok(())
        } else {
            Err("local alignment needs an Apple-Silicon Mac (ONNX Runtime no longer provides Intel-mac builds)")
        }
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        Err("local alignment is not packaged for this platform")
    }
}

static MODELS_DIR: OnceLock<PathBuf> = OnceLock::new();

/// Resolve and cache the models directory. MUST be called once on the main
/// thread at startup (after the REAPER handle is published): the resource path
/// comes from the main-thread-only REAPER API, and resolving it lazily from the
/// worker would silently fall back to the per-user app dir, splitting the model
/// location from the config location on portable installs.
pub fn init() {
    if let Some(dir) = crate::providers::registry::config_dir() {
        let _ = MODELS_DIR.set(dir.join("models"));
    }
}

/// `<REAPER resource path>/ReaLackey/models`, as cached by [`init`].
pub fn models_dir() -> Option<PathBuf> {
    MODELS_DIR.get().cloned()
}

/// The two files local alignment needs, verified present.
pub struct AlignFiles {
    pub model: PathBuf,
    pub runtime: PathBuf,
}

fn file_ok(path: &PathBuf, size: u64) -> bool {
    std::fs::metadata(path).map(|m| m.len() == size).unwrap_or(false)
}

/// Both files installed (exact expected sizes — cheap corruption guard; full
/// SHA-256 verification happens once, at download time)? `None` if anything is
/// missing, truncated, or the dir was never resolved.
pub fn installed() -> Option<AlignFiles> {
    let dir = models_dir()?;
    let model = dir.join(MODEL_FILE);
    let runtime = dir.join(RUNTIME_FILE);
    (file_ok(&model, MODEL_SIZE) && file_ok(&runtime, RUNTIME_SIZE))
        .then_some(AlignFiles { model, runtime })
}

/// One file the downloader must fetch.
pub struct DownloadItem {
    pub url: &'static str,
    pub file: &'static str,
    pub size: u64,
    pub sha256: &'static str,
    /// Short human name for progress/errors ("the alignment model").
    pub label: &'static str,
}

/// What still needs downloading (already-valid files are skipped, so a bundled
/// install downloads nothing and a broken/partial install fetches only the rest).
pub fn download_items() -> Vec<DownloadItem> {
    let Some(dir) = models_dir() else {
        return Vec::new();
    };
    let mut items = Vec::new();
    if !file_ok(&dir.join(RUNTIME_FILE), RUNTIME_SIZE) {
        items.push(DownloadItem {
            url: RUNTIME_URL,
            file: RUNTIME_FILE,
            size: RUNTIME_SIZE,
            sha256: RUNTIME_SHA256,
            label: "the ONNX Runtime library",
        });
    }
    if !file_ok(&dir.join(MODEL_FILE), MODEL_SIZE) {
        items.push(DownloadItem {
            url: MODEL_URL,
            file: MODEL_FILE,
            size: MODEL_SIZE,
            sha256: MODEL_SHA256,
            label: "the alignment model",
        });
    }
    items
}

/// Rough total download size, for the consent prompt ("about 318 MB").
pub fn download_megabytes() -> u64 {
    download_items().iter().map(|i| i.size).sum::<u64>() / 1_000_000
}

// ---- the engine --------------------------------------------------------------

/// A loaded alignment session. Creation costs ~1 s (the 300 MB model is read and
/// prepared); `run_transcription` creates one per transcription and drops it
/// after, so the memory is only held while alignment actually runs.
pub struct Engine {
    session: ort::session::Session,
}

/// ONNX Runtime's dylib can be loaded into the process exactly once; remember
/// success so a second Engine doesn't re-init (and a FAILED attempt stays
/// retryable — e.g. the user installs the files and tries again this session).
static ORT_READY: OnceLock<()> = OnceLock::new();

fn ensure_ort(runtime: &std::path::Path) -> Result<(), String> {
    if ORT_READY.get().is_some() {
        return Ok(());
    }
    let builder = ort::init_from(runtime).map_err(|e| {
        format!("could not load ONNX Runtime from {}: {e}", runtime.display())
    })?;
    builder.commit();
    let _ = ORT_READY.set(());
    Ok(())
}

/// Outcome stats for one `refine` pass, for a quiet status line.
pub struct Refined {
    /// Words whose timing was updated from the alignment.
    pub aligned: usize,
    /// Words kept on their transcript timing (unalignable text, failed window,
    /// or an implausibly large shift).
    pub skipped: usize,
    /// Largest boundary movement applied, seconds.
    pub max_shift: f64,
}

impl Engine {
    /// Load the runtime + model. Fails cleanly (never panics): a missing or
    /// wrong-versioned dylib, or an unreadable model, must degrade to
    /// "transcription without refinement", not take the worker down.
    pub fn load(files: &AlignFiles) -> Result<Self, String> {
        ensure_ort(&files.runtime)?;
        let session = ort::session::Session::builder()
            .and_then(|mut b| b.commit_from_file(&files.model))
            .map_err(|e| format!("could not load the alignment model: {e}"))?;
        Ok(Self { session })
    }

    /// Re-time `words` against `samples` (mono, 16 kHz, same clock as the word
    /// times). Only `start`/`end` are written; text, count and order are
    /// untouched. `cancel` is checked between windows (a window is ~10 s of CPU
    /// on an older machine); `progress(done, total)` is called after each.
    pub fn refine(
        &mut self,
        samples: &[f32],
        sr: f64,
        words: &mut [Word],
        cancel: &AtomicBool,
        progress: &mut dyn FnMut(usize, usize),
    ) -> Result<Refined, String> {
        if (sr - SR).abs() > 0.5 {
            return Err(format!("alignment expects 16 kHz audio, got {sr} Hz"));
        }
        if words.is_empty() || samples.is_empty() {
            return Ok(Refined { aligned: 0, skipped: 0, max_shift: 0.0 });
        }
        let groups = group_words(words, MAX_WINDOW);
        let total = groups.len();
        let clip_end = samples.len() as f64 / SR;
        let mut stats = Refined { aligned: 0, skipped: 0, max_shift: 0.0 };
        for (gi, &(a, b)) in groups.iter().enumerate() {
            if cancel.load(Ordering::Relaxed) {
                return Err(CANCELLED.into());
            }
            let t0 = (words[a].start - PAD).max(0.0).min(clip_end);
            let t1 = (words[b - 1].end.max(words[b - 1].start) + PAD).clamp(t0, clip_end);
            let s0 = (t0 * SR) as usize;
            let s1 = ((t1 * SR) as usize).min(samples.len());
            // Below ~0.1 s the conv frontend has nothing to work with.
            if s1.saturating_sub(s0) < 1600 {
                stats.skipped += b - a;
                continue;
            }
            // Above the design bound the window can only be a single runaway
            // word (Whisper's known pathology: a final word stretched across
            // trailing silence to the chunk end — group_words bounds multi-word
            // groups, but a lone word's own length is unbounded). The encoder's
            // attention is quadratic in frames, so a 600 s window means gigabytes
            // and minutes of uncancellable inference — and MAX_SHIFT would
            // reject the result anyway. Keep the transcript's timing instead.
            if s1 - s0 > ((MAX_WINDOW + 2.0 * PAD) * SR) as usize {
                stats.skipped += b - a;
                continue;
            }
            self.refine_window(&samples[s0..s1], s0 as f64 / SR, &mut words[a..b], &mut stats);
            progress(gi + 1, total);
        }
        Ok(stats)
    }

    /// Align one window's words against its audio; update `stats` and the words.
    /// A window that fails (emission error, no alignable text, no Viterbi path)
    /// skips its words rather than failing the pass — a bad stretch of audio
    /// must not cost the rest of the clip its refinement.
    fn refine_window(&mut self, chunk: &[f32], origin: f64, words: &mut [Word], stats: &mut Refined) {
        let vocab = vocab();
        let mut tokens: Vec<usize> = Vec::new();
        let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(words.len());
        for w in words.iter() {
            let start = tokens.len();
            for c in romanize(&w.text).chars() {
                if let Some(id) = vocab.iter().find_map(|&(vc, id)| (vc == c).then_some(id)) {
                    tokens.push(id);
                }
            }
            ranges.push((start, tokens.len()));
        }
        if tokens.is_empty() {
            stats.skipped += words.len();
            return;
        }
        let emis = match self.emissions(chunk) {
            Ok(e) => e,
            Err(_) => {
                stats.skipped += words.len();
                return;
            }
        };
        let Some(spans) = ctc_align(&emis, &tokens, BLANK) else {
            stats.skipped += words.len();
            return;
        };
        for (wi, w) in words.iter_mut().enumerate() {
            let (ra, rb) = ranges[wi];
            if ra == rb {
                stats.skipped += 1; // nothing alignable (digits, punctuation)
                continue;
            }
            let new_start = origin + spans[ra].0 as f64 * FRAME;
            // One past the last emission frame: a boundary ON it hands 20 ms of
            // the final consonant to the next word.
            let new_end = origin + (spans[rb - 1].1 + 1) as f64 * FRAME;
            let shift = (new_start - w.start).abs().max((new_end - w.end).abs());
            if shift > MAX_SHIFT || new_end <= new_start {
                stats.skipped += 1;
                continue;
            }
            w.start = new_start;
            w.end = new_end;
            stats.aligned += 1;
            stats.max_shift = stats.max_shift.max(shift);
        }
    }

    /// Run the encoder on one window, returning per-frame log-probabilities.
    fn emissions(&mut self, chunk: &[f32]) -> Result<Vec<Vec<f32>>, String> {
        let input = normalize(chunk);
        let len = input.len();
        let tensor = ort::value::Tensor::from_array(([1usize, len], input))
            .map_err(|e| e.to_string())?;
        let outputs = self
            .session
            .run(ort::inputs!["input_values" => tensor])
            .map_err(|e| e.to_string())?;
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;
        let dims: Vec<i64> = shape.iter().copied().collect();
        if dims.len() != 3 {
            return Err(format!("unexpected logits shape {dims:?}"));
        }
        let (frames, vocab) = (dims[1] as usize, dims[2] as usize);
        Ok((0..frames)
            .map(|t| {
                let row = &data[t * vocab..(t + 1) * vocab];
                // log-softmax
                let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let z: f32 = row.iter().map(|v| (v - m).exp()).sum();
                let lz = z.ln() + m;
                row.iter().map(|v| v - lz).collect()
            })
            .collect())
    }
}

/// Split `words` into contiguous index ranges whose transcript span stays under
/// `max_window` seconds. Boundaries always fall between words, and every group
/// gets `PAD` of acoustic context on both sides when aligned.
fn group_words(words: &[Word], max_window: f64) -> Vec<(usize, usize)> {
    let mut groups = Vec::new();
    let mut g0 = 0;
    for i in 0..words.len() {
        if i > g0 && words[i].end.max(words[i].start) - words[g0].start > max_window {
            groups.push((g0, i));
            g0 = i;
        }
    }
    groups.push((g0, words.len()));
    groups
}

/// The MMS forced-aligner vocabulary (from its vocab.json): lowercase a-z +
/// apostrophe; id 0 is the CTC blank.
fn vocab() -> &'static [(char, usize)] {
    &[
        ('a', 4), ('b', 20), ('c', 23), ('d', 16), ('e', 6), ('f', 27), ('g', 17),
        ('h', 18), ('i', 5), ('j', 25), ('k', 14), ('l', 15), ('m', 13), ('n', 7),
        ('o', 8), ('p', 21), ('q', 29), ('r', 12), ('s', 11), ('t', 10), ('u', 9),
        ('v', 24), ('w', 22), ('x', 30), ('y', 19), ('z', 26), ('\'', 28),
    ]
}

/// Uroman-style romanization into the aligner's a-z vocabulary. German gets the
/// umlaut/ß folding; anything else keeps its a-z letters. Words with nothing
/// left (digits, punctuation-only) return an empty string and keep their
/// transcript timing — CTC cannot align a token that has no symbol.
fn romanize(word: &str) -> String {
    word.to_lowercase()
        .chars()
        .flat_map(|c| match c {
            'ä' => vec!['a'],
            'ö' => vec!['o'],
            'ü' => vec!['u'],
            'ß' => vec!['s', 's'],
            'a'..='z' | '\'' => vec![c],
            _ => vec![],
        })
        .collect()
}

/// Zero-mean unit-variance, as the model's preprocessor demands.
fn normalize(x: &[f32]) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let s = (var + 1e-7).sqrt();
    x.iter().map(|v| (v - mean) / s).collect()
}

/// CTC Viterbi forced alignment: per-token frame spans through the emission
/// matrix. Standard 2N+1-state trellis (blanks interleaved), log domain.
/// Returns for each token its (first_frame, last_frame) on the best path.
fn ctc_align(logp: &[Vec<f32>], tokens: &[usize], blank: usize) -> Option<Vec<(usize, usize)>> {
    let t_max = logp.len();
    let n = tokens.len();
    if n == 0 || t_max < n {
        return None;
    }
    let s_max = 2 * n + 1;
    let label = |s: usize| if s % 2 == 0 { blank } else { tokens[(s - 1) / 2] };
    const NEG: f32 = f32::NEG_INFINITY;
    let mut dp = vec![NEG; s_max];
    let mut bp = vec![vec![0u8; s_max]; t_max]; // 0 = stay, 1 = from s-1, 2 = from s-2
    dp[0] = logp[0][blank];
    if s_max > 1 {
        dp[1] = logp[0][label(1)];
    }
    for t in 1..t_max {
        let prev = dp.clone();
        for s in 0..s_max {
            let mut best = prev[s];
            let mut from = 0u8;
            if s >= 1 && prev[s - 1] > best {
                best = prev[s - 1];
                from = 1;
            }
            // Skipping a blank between two DIFFERENT tokens is legal in CTC.
            if s >= 2 && s % 2 == 1 && label(s) != label(s - 2) && prev[s - 2] > best {
                best = prev[s - 2];
                from = 2;
            }
            dp[s] = if best == NEG { NEG } else { best + logp[t][label(s)] };
            bp[t][s] = from;
        }
    }
    // End in the final token or the final blank, whichever scored better.
    let mut s = if s_max >= 2 && dp[s_max - 2] > dp[s_max - 1] { s_max - 2 } else { s_max - 1 };
    if dp[s] == NEG {
        return None;
    }
    let mut spans = vec![(usize::MAX, 0usize); n];
    for t in (0..t_max).rev() {
        if s % 2 == 1 {
            let tok = (s - 1) / 2;
            spans[tok].0 = spans[tok].0.min(t);
            spans[tok].1 = spans[tok].1.max(t);
        }
        if t > 0 {
            s -= bp[t][s] as usize;
        }
    }
    spans.iter().all(|&(a, _)| a != usize::MAX).then_some(spans)
}

// ---- tests -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn romanize_folds_german_and_drops_the_rest() {
        assert_eq!(romanize("Schoß."), "schoss");
        assert_eq!(romanize("zärtlich"), "zartlich");
        assert_eq!(romanize("Über"), "uber");
        assert_eq!(romanize("1993"), "");
        assert_eq!(romanize("don't"), "don't");
    }

    #[test]
    fn group_words_splits_at_word_gaps() {
        let w = |s: f64, e: f64| Word { start: s, end: e, text: "x".into() };
        let words: Vec<Word> = (0..100).map(|i| w(i as f64, i as f64 + 0.8)).collect();
        let groups = group_words(&words, 28.0);
        assert!(groups.len() >= 3);
        assert_eq!(groups.first().unwrap().0, 0);
        assert_eq!(groups.last().unwrap().1, 100);
        for pair in groups.windows(2) {
            assert_eq!(pair[0].1, pair[1].0); // contiguous, no gaps or overlaps
        }
        for &(a, b) in &groups {
            assert!(words[b - 1].end - words[a].start <= 28.0 + 0.8);
        }
    }

    #[test]
    fn ctc_align_picks_the_obvious_path() {
        // 6 frames, vocab {blank=0, a=1, b=2}; "ab" with a in frames 1-2, b in 4-5.
        let quiet = vec![0.0f32, -8.0, -8.0];
        let a = vec![-8.0f32, 0.0, -8.0];
        let b = vec![-8.0f32, -8.0, 0.0];
        let logp = vec![quiet.clone(), a.clone(), a, quiet.clone(), b.clone(), b];
        let spans = ctc_align(&logp, &[1, 2], 0).expect("path");
        assert_eq!(spans, vec![(1, 2), (4, 5)]);
    }

    // ---- the Phase-0 probe: RTF + junction quality on real audio -------------
    //
    // Inert unless the env vars point at the real artefacts:
    //   ALIGN_ORT_DYLIB=<onnxruntime dylib> ALIGN_MODEL=<model_int8.onnx>
    //   PARO16=<paro16.wav> PARO_WORDS=<words file: "start|end|text" lines>
    //   cargo test --lib align -- --nocapture
    // See the alignment-research memory note for where those live.

    struct Clip {
        samples: Vec<f32>,
        words: Vec<(f64, f64, String)>,
    }

    fn load() -> Option<(Engine, Clip)> {
        let dylib = std::env::var("ALIGN_ORT_DYLIB").ok()?;
        let model = std::env::var("ALIGN_MODEL").ok()?;
        let wav = std::env::var("PARO16").ok()?;
        let words_path = std::env::var("PARO_WORDS").ok()?;
        let t0 = Instant::now();
        let engine = Engine::load(&AlignFiles {
            model: model.into(),
            runtime: dylib.into(),
        })
        .expect("load engine");
        eprintln!("model load: {:.1}s", t0.elapsed().as_secs_f64());
        let bytes = std::fs::read(&wav).expect("read wav");
        let (samples, ch, sr) = crate::dsp::parse_wav(&bytes).expect("parse wav");
        assert_eq!(ch, 1, "probe expects mono");
        assert_eq!(sr, SR, "probe expects 16 kHz");
        let samples: Vec<f32> = samples.iter().map(|&x| x as f32).collect();
        let words = std::fs::read_to_string(&words_path)
            .expect("words file")
            .lines()
            .filter_map(|l| {
                let mut p = l.split('|');
                Some((
                    p.next()?.parse().ok()?,
                    p.next()?.parse().ok()?,
                    p.next()?.to_string(),
                ))
            })
            .collect();
        Some((engine, Clip { samples, words }))
    }

    /// End-to-end over the production path: refine the full clip's words,
    /// report RTF and the boundaries at the labelled junctions.
    #[test]
    fn align_probe_rtf_and_junctions() {
        let Some((mut engine, clip)) = load() else {
            eprintln!("set ALIGN_ORT_DYLIB / ALIGN_MODEL / PARO16 / PARO_WORDS to run the probe");
            return;
        };
        let mut words: Vec<Word> = clip
            .words
            .iter()
            .map(|(s, e, t)| Word { start: *s, end: *e, text: t.clone() })
            .collect();
        let originals = words.clone();
        let total_secs = clip.samples.len() as f64 / SR;
        let cancel = AtomicBool::new(false);
        let t0 = Instant::now();
        let stats = engine
            .refine(&clip.samples, SR, &mut words, &cancel, &mut |done, total| {
                if done % 5 == 0 {
                    eprintln!("  window {done}/{total}");
                }
            })
            .expect("refine");
        let infer = t0.elapsed().as_secs_f64();
        eprintln!(
            "RTF: {total_secs:.0}s audio -> {infer:.1}s alignment (RTF {:.3}); aligned {} skipped {} max shift {:.3}s",
            infer / total_secs,
            stats.aligned,
            stats.skipped,
            stats.max_shift
        );
        // Boundaries near the labelled junctions, refined vs transcript.
        let interest: &[f64] = &[2.9, 3.8, 4.8, 5.9, 6.3, 7.0, 8.2, 13.2, 16.7, 107.7, 130.9];
        for (w, o) in words.iter().zip(&originals) {
            let near = interest
                .iter()
                .any(|&j| (o.start - j).abs() < 1.2 || (o.end - j).abs() < 1.2);
            if near {
                eprintln!(
                    "CTC {:>16} transcript {:>8.3}-{:<8.3} ctc {:>8.3}-{:<8.3} drift {:+.3}/{:+.3}",
                    w.text,
                    o.start,
                    o.end,
                    w.start,
                    w.end,
                    w.start - o.start,
                    w.end - o.end
                );
            }
        }
        // The refiner's contract: text/count/order untouched.
        assert_eq!(words.len(), originals.len());
        for (w, o) in words.iter().zip(&originals) {
            assert_eq!(w.text, o.text);
        }
    }
}
