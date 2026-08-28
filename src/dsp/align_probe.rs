//! Phase-0 probe: CTC forced alignment, in-process, measured.
//!
//! Answers two questions before any shipping code is written: what does a
//! 300M-parameter CTC aligner cost on plain CPU inside this crate, and do its
//! boundaries settle the junction disputes the DSP layer fights? Everything here
//! is `#[cfg(test)]` and env-gated; `ort` is a dev-dependency, so the shipped
//! cdylib is byte-identical until the probe graduates.
//!
//! Run:
//!   ALIGN_MODEL=<model_int8.onnx> PARO16=<paro16.wav> PARO_WORDS=<words.json> \
//!     cargo test --lib align_probe -- --nocapture
//!
//! Model: onnx-community/mms-300m-1130-forced-aligner-ONNX (CC-BY-NC weights —
//! fine for this non-commercial project). Vocab is lowercase a-z + apostrophe,
//! blank id 0; German romanizes with a five-line mapping, no G2P needed.

#![allow(clippy::needless_range_loop)] // trellis code reads clearer indexed

use std::time::Instant;

/// Uroman-style romanization for German into the aligner's a-z vocabulary.
/// Words with nothing left (digits, punctuation-only) return an empty string
/// and are skipped by the caller — CTC cannot align a token that has no symbol.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// wav2vec2 stride: 320 samples at 16 kHz per emission frame.
    const FRAME: f64 = 0.02;
    const SR: f64 = 16_000.0;

    struct Clip {
        samples: Vec<f32>,
        words: Vec<(f64, f64, String)>,
    }

    fn load() -> Option<(ort::session::Session, Clip)> {
        let model = std::env::var("ALIGN_MODEL").ok()?;
        let wav = std::env::var("PARO16").ok()?;
        let words_path = std::env::var("PARO_WORDS").ok()?;
        let t0 = Instant::now();
        let session = ort::session::Session::builder()
            .and_then(|mut b| b.commit_from_file(&model))
            .expect("load onnx model");
        eprintln!("model load: {:.1}s", t0.elapsed().as_secs_f64());
        for i in session.inputs() {
            eprintln!("  input:  {}", i.name());
        }
        for o in session.outputs() {
            eprintln!("  output: {}", o.name());
        }
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
        Some((session, Clip { samples, words }))
    }

    /// Zero-mean unit-variance, as the model's preprocessor demands.
    fn normalize(x: &[f32]) -> Vec<f32> {
        let n = x.len() as f32;
        let mean = x.iter().sum::<f32>() / n;
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let s = (var + 1e-7).sqrt();
        x.iter().map(|v| (v - mean) / s).collect()
    }

    /// Run the encoder on one chunk, returning per-frame log-probs.
    fn emissions(session: &mut ort::session::Session, chunk: &[f32]) -> Vec<Vec<f32>> {
        let input = normalize(chunk);
        let len = input.len();
        let tensor = ort::value::Tensor::from_array(([1usize, len], input)).expect("tensor");
        let outputs = session
            .run(ort::inputs!["input_values" => tensor])
            .expect("run");
        let (shape, data) = outputs[0].try_extract_tensor::<f32>().expect("logits");
        let dims: Vec<i64> = shape.iter().copied().collect();
        let (frames, vocab) = (dims[1] as usize, dims[2] as usize);
        (0..frames)
            .map(|t| {
                let row = &data[t * vocab..(t + 1) * vocab];
                // log-softmax
                let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let z: f32 = row.iter().map(|v| (v - m).exp()).sum();
                let lz = z.ln() + m;
                row.iter().map(|v| v - lz).collect()
            })
            .collect()
    }

    /// The full probe: RTF over the whole clip, then boundary quality on the
    /// chunks that contain the labelled junctions.
    #[test]
    fn align_probe_rtf_and_junctions() {
        let Some((mut session, clip)) = load() else {
            eprintln!("set ALIGN_MODEL / PARO16 / PARO_WORDS to run the probe");
            return;
        };
        let vocab: std::collections::HashMap<char, usize> =
            [('a', 4), ('b', 20), ('c', 23), ('d', 16), ('e', 6), ('f', 27), ('g', 17),
             ('h', 18), ('i', 5), ('j', 25), ('k', 14), ('l', 15), ('m', 13), ('n', 7),
             ('o', 8), ('p', 21), ('q', 29), ('r', 12), ('s', 11), ('t', 10), ('u', 9),
             ('v', 24), ('w', 22), ('x', 30), ('y', 19), ('z', 26), ('\'', 28)]
            .into_iter()
            .collect();
        let total_secs = clip.samples.len() as f64 / SR;
        let chunk_secs = 30.0;
        let chunk_len = (chunk_secs * SR) as usize;

        // --- RTF: encoder over the whole clip, chunked ---
        let t0 = Instant::now();
        let mut n_chunks = 0usize;
        let mut per_chunk = Vec::new();
        let mut all_emissions: Vec<Vec<Vec<f32>>> = Vec::new();
        for chunk in clip.samples.chunks(chunk_len) {
            let c0 = Instant::now();
            all_emissions.push(emissions(&mut session, chunk));
            per_chunk.push(c0.elapsed().as_secs_f64());
            n_chunks += 1;
        }
        let infer = t0.elapsed().as_secs_f64();
        eprintln!(
            "RTF: {total_secs:.0}s audio, {n_chunks} chunks of {chunk_secs:.0}s -> {infer:.1}s inference  (RTF {:.3}, {:.1}s per 30s chunk)",
            infer / total_secs,
            per_chunk.iter().sum::<f64>() / per_chunk.len() as f64
        );

        // --- Quality: align each chunk's words, print the labelled junctions ---
        let interest: &[f64] = &[2.9, 3.8, 4.8, 5.9, 6.3, 7.0, 8.2, 13.2, 16.7, 107.7, 130.9];
        for (ci, emis) in all_emissions.iter().enumerate() {
            let (t0c, t1c) = (ci as f64 * chunk_secs, (ci as f64 + 1.0) * chunk_secs);
            if !interest.iter().any(|&j| j > t0c && j < t1c) {
                continue;
            }
            // Words fully inside this chunk (0.5s margins).
            let inside: Vec<(usize, &(f64, f64, String))> = clip
                .words
                .iter()
                .enumerate()
                .filter(|(_, w)| w.0 > t0c + 0.5 && w.1 < t1c - 0.5)
                .collect();
            let mut tokens = Vec::new();
            let mut ranges = Vec::new(); // per word: token index range
            for (_, w) in &inside {
                let start = tokens.len();
                for c in romanize(&w.2).chars() {
                    if let Some(&id) = vocab.get(&c) {
                        tokens.push(id);
                    }
                }
                ranges.push((start, tokens.len()));
            }
            let Some(spans) = ctc_align(emis, &tokens, 0) else {
                eprintln!("chunk {ci}: alignment failed");
                continue;
            };
            eprintln!("--- chunk {ci} ({t0c:.0}-{t1c:.0}s): {} words ---", inside.len());
            for (wi, (_, w)) in inside.iter().enumerate() {
                let (a, b) = ranges[wi];
                if a == b {
                    continue; // unalignable (digits)
                }
                let start = t0c + spans[a].0 as f64 * FRAME;
                let end = t0c + (spans[b - 1].1 + 1) as f64 * FRAME;
                let near = interest.iter().any(|&j| (w.0 - j).abs() < 1.2 || (w.1 - j).abs() < 1.2);
                if near {
                    eprintln!(
                        "CTC {:>16} transcript {:>8.3}-{:<8.3} ctc {:>8.3}-{:<8.3} drift {:+.3}/{:+.3}",
                        w.2, w.0, w.1, start, end, start - w.0, end - w.1
                    );
                    // Char-level emission spans: the ownership evidence a word-level
                    // transcript cannot give (which word the /s/, /t/, /ch/ frames
                    // belong to at the disputed junctions).
                    let chars: Vec<char> = romanize(&w.2).chars().collect();
                    let detail: Vec<String> = (a..b)
                        .map(|ti| {
                            let (fa, fb) = spans[ti];
                            format!(
                                "{}:{:.2}-{:.2}",
                                chars[ti - a],
                                t0c + fa as f64 * FRAME,
                                t0c + (fb + 1) as f64 * FRAME
                            )
                        })
                        .collect();
                    eprintln!("      chars {}", detail.join(" "));
                }
            }
        }
    }
}
