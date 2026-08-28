//! Cut preview: the editor's Play button must audition EXACTLY what Confirm will
//! write, so it asks Rust for the real cut boundaries instead of guessing from the
//! transcript.
//!
//! Without this the preview played transcript times with a fixed 30 ms lead, while
//! the cut used measured boundaries — so the two disagreed by design, and every
//! judgement made from the preview was made against something that never got written.

use std::sync::Mutex;

use crate::dsp::speech::{Edge, EdgeCause, SpeechAnalysis};
use crate::providers::transcription::Word;

/// Everything needed to re-derive the cut for the item currently open in the editor.
struct Ctx {
    analysis: SpeechAnalysis,
    words: Vec<Word>,
    /// Word times went through local forced alignment: they are trustworthy to a
    /// frame or two, and measured extents may only REFINE near them (see the
    /// corridor in `word_bounds_explained`), never wander.
    refined: bool,
}

static CTX: Mutex<Option<Ctx>> = Mutex::new(None);

/// Cache the analysis for the item the editor is about to open. Cheap: it reuses the
/// WAV already rendered for in-editor playback, so no extra audio read.
/// `refined` = the word times went through local forced alignment.
pub fn arm(wav: &[u8], words: &[Word], refined: bool) {
    let analysis = crate::dsp::parse_wav(wav)
        .ok()
        .and_then(|(samples, ch, sr)| {
            let mono: Vec<f64> = if ch <= 1 {
                samples
            } else {
                samples.chunks(ch).map(|f| f.iter().sum::<f64>() / ch as f64).collect()
            };
            SpeechAnalysis::new(&mono, sr)
        });
    if let (Some(analysis), Ok(mut g)) = (analysis, CTX.lock()) {
        *g = Some(Ctx { analysis, words: words.to_vec(), refined });
    }
}

/// A copy of the cached analysis, for the CUT executor.
///
/// The executor used to build its OWN SpeechAnalysis over a few seconds around the
/// edit. Noise floor, Otsu split and syllable period are whole-clip statistics, so a
/// single-word deletion calibrated them on ~4 s of audio while the preview used the
/// whole take — the preview therefore auditioned a different calibration than the one
/// that wrote the cut, which is the exact disagreement this module exists to prevent.
/// Cloning costs one allocation per cut (a user-initiated action, not per keystroke).
pub fn analysis_clone() -> Option<SpeechAnalysis> {
    CTX.lock().ok()?.as_ref().map(|c| c.analysis.clone())
}

/// Forget it when the editor closes.
pub fn disarm() {
    if let Ok(mut g) = CTX.lock() {
        *g = None;
    }
}

/// Measured [onset, offset] for EVERY word, in item time.
///
/// Each is obtained by asking where the cut would go if that word alone were removed,
/// so what you hear when you land on a word is exactly what would disappear if you
/// deleted it. Falls back to the transcript's own times per word when unmeasurable.
/// How far a measured word edge may sit past the neighbour's transcript boundary
/// before it is treated as having crossed into the neighbour rather than measured
/// this word — as a fraction of the speaker's OWN syllable period.
///
/// It was a fixed 40 ms, which is both too tight and a breadth violation: syllable
/// rate varies about 2x across languages and as much again between talkers, and the
/// transcript side of the comparison is a Whisper time that drifts by up to ~0.2 s.
/// At 40 ms this was the single largest cause of unmeasured edges, and the rejections
/// were overshooting by only 15-30 ms — correct measurements hitting a bound too
/// tight to hold them. Half a syllable is the natural scale: an edge further than
/// that past the neighbour's start is not this word's boundary in anyone's speech.
const NEIGHBOUR_BLEED_SYLLABLES: f64 = 0.5;

/// Refinement corridor around ALIGNER-GRADE word times, seconds. When the words
/// went through forced alignment, the DSP's job shrinks to sub-frame refinement
/// NEAR the aligned boundary — free-range re-measurement made things worse than
/// the times it was handed ("langen Wimpern und hebt den": aligned times were
/// correct to a frame, measured extents shifted ~100 ms and midpoint
/// reconciliation shredded "und" down to its final /n/). The corridor is
/// asymmetric on purpose, from measured CTC behaviour:
/// * a word's real onset starts BEFORE its first CTC emission (peakiness — the
///   /f/ of "Fell" 82 ms early), so starts may move earlier generously but
///   later barely;
/// * a word's real end runs past its last emission (codas, releases —
///   "graue"+108 ms), so ends get the mirror.
pub(crate) const REFINED_START_EARLY: f64 = 0.10;
pub(crate) const REFINED_START_LATE: f64 = 0.02;
pub(crate) const REFINED_END_EARLY: f64 = 0.02;
pub(crate) const REFINED_END_LATE: f64 = 0.12;

/// One word's measured extent, with the reason for each edge that is a fallback.
pub struct Bound {
    pub start: f64,
    pub end: f64,
    pub start_cause: EdgeCause,
    pub end_cause: EdgeCause,
    /// By how much a crossing crossed; 0.0 otherwise.
    pub detail: f64,
    /// Length in FRAMES of the quiet run behind each edge, -1 when none was found.
    pub runs: (f64, f64),
    /// Peak high-band-over-voice-bar tilt, dB, in the window each edge searched.
    pub tilt: (f64, f64),
}

/// Frames, or `-` when no run was found.
fn fmt_run(r: f64) -> String {
    if r < 0.0 {
        "-".to_string()
    } else {
        format!("{r:.0}")
    }
}

pub fn word_bounds() -> Vec<(f64, f64)> {
    word_bounds_explained().into_iter().map(|b| (b.start, b.end)).collect()
}

pub fn word_bounds_explained() -> Vec<Bound> {
    let g = match CTX.lock() {
        Ok(g) => g,
        Err(_) => return Vec::new(),
    };
    let Some(ctx) = g.as_ref() else {
        return Vec::new();
    };
    let (clip_a, clip_b) = ctx.analysis.span();
    let bleed = ctx.analysis.syllable_period * NEIGHBOUR_BLEED_SYLLABLES;
    let mut bounds: Vec<Bound> = (0..ctx.words.len())
        .map(|i| {
            let w = &ctx.words[i];
            let prev = i.checked_sub(1).and_then(|j| ctx.words.get(j)).map(|p| (p.start, p.end));
            let next = ctx.words.get(i + 1).map(|n| (n.start, n.end));
            let (ap, an) = ctx.analysis.anchors(prev, (w.start, w.end), next);
            // The two edges are independent, and placement has always treated them so.
            // Collapsing to (None, None) when EITHER anchor was missing threw away a
            // perfectly good edge on the other side — and since anchors() has no
            // previous word for word 0 and no next word for word N-1, it made the
            // first and last word of every clip unmeasurable for no acoustic reason.
            // The clip's own edge is the honest substitute there: nothing lies beyond
            // it, so the nearest quiet run inward is still the word's real boundary.
            // The word's own extent, NOT the cut boundaries: those sit inside the
            // surrounding pause by design.
            let (o, f) = ctx.analysis.word_extent(
                ap.unwrap_or(clip_a),
                an.unwrap_or(clip_b),
                (w.start, w.end),
                next.map(|n| n.0),
            );
            // A measured edge may bleed a little past the neighbour's transcript
            // boundary, but not INTO the neighbour. Without this, a gap too short to
            // register as a quiet run makes the search skip over the next word and take
            // the run after it — the report showed ~10 pairs resolving to the identical
            // run, worst "Robot" running 0.68 s past its end and swallowing the word
            // after it. `.min`/`.max` against the word's own hint so an overlapping
            // transcript (numbers, hyphenated tokens) can never tighten the bound past
            // where the word itself claims to be.
            let lo = prev.map(|(_, pe)| pe.min(w.start) - bleed);
            let hi = next.map(|(ns, _)| ns.max(w.end) + bleed);
            let bleed = |e: Edge, out: bool| if out { e.reject(EdgeCause::BleedReject) } else { e };
            let o = bleed(o, o.time.is_some_and(|t| lo.is_some_and(|l| t < l)));
            let f = bleed(f, f.time.is_some_and(|t| hi.is_some_and(|h| t > h)));
            // Aligner-grade times: measurements may only refine near them (the
            // corridor constants above); outside it the hint wins via the
            // rejected edge's fallback below.
            let corridor = |e: Edge, hint: f64, early: f64, late: f64| match e.time {
                Some(t) if t < hint - early || t > hint + late => {
                    e.reject(EdgeCause::CorridorReject)
                }
                _ => e,
            };
            let (o, f) = if ctx.refined {
                (
                    corridor(o, w.start, REFINED_START_EARLY, REFINED_START_LATE),
                    corridor(f, w.end, REFINED_END_EARLY, REFINED_END_LATE),
                )
            } else {
                (o, f)
            };
            Bound {
                start: o.time.unwrap_or(w.start),
                end: f.time.unwrap_or(w.end),
                start_cause: o.cause,
                end_cause: f.cause,
                detail: o.detail.max(f.detail),
                runs: (o.run, f.run),
                // Measured over the SAME windows the edges searched, so a fricative
                // that should have been found there cannot hide from this.
                tilt: (
                    ctx.analysis.band_tilt(ap.unwrap_or(clip_a), w.start),
                    ctx.analysis.band_tilt(w.end, an.unwrap_or(clip_b)),
                ),
            }
        })
        .collect();
    // The two sides of a junction are measured INDEPENDENTLY (word i's end edge,
    // word i+1's start edge), and at a handful of junctions with no acoustic cue
    // between two fused consonants they disagree by 20-80 ms. Auditioning both
    // sides of that disagreement plays the blend twice, so adjacent overlaps are
    // reconciled HERE, at the consumer -- the measurement, the report and its
    // overlap health metric stay raw. Which way to reconcile depends on how much
    // the times are worth: with aligner-grade words the ONSET side wins (a word
    // missing its first consonant is the audible failure -- "grossen" playing as
    // "rossen"; a coda trimmed a few hundredths early is not), while unrefined
    // times get the midpoint, since neither side deserves more trust.
    for i in 1..bounds.len() {
        let (a, b) = bounds.split_at_mut(i);
        let (prev, next) = (a.last_mut().unwrap(), b.first_mut().unwrap());
        if next.start < prev.end {
            if ctx.refined {
                prev.end = next.start.max(prev.start);
            } else {
                let mid = (0.5 * (next.start + prev.end)).max(prev.start).min(next.end);
                prev.end = mid;
                next.start = mid;
            }
        }
    }
    bounds
}

/// A whole-clip diagnostic: every word with its transcript times, its MEASURED
/// extent, the drift between them, and the measured gap to the next word.
///
/// One paste diagnoses a whole take. Without it every symptom costs a re-cut, and
/// only the words that happened to be cut are visible at all.
pub fn report() -> String {
    // Computed first: it takes the lock itself.
    let bounds = word_bounds_explained();
    let g = match CTX.lock() {
        Ok(g) => g,
        Err(_) => return String::new(),
    };
    let Some(ctx) = g.as_ref() else {
        return "No clip is loaded in the editor.".into();
    };

    // Tally per cause, so "94 words fail" becomes a breakdown that says how much is
    // engineering-recoverable and how much is audio that has no boundary to find.
    let mut tally: Vec<(EdgeCause, usize)> = Vec::new();
    let mut bump = |c: EdgeCause| match tally.iter_mut().find(|(k, _)| *k == c) {
        Some((_, n)) => *n += 1,
        None => tally.push((c, 1)),
    };
    for b in &bounds {
        bump(b.start_cause);
        bump(b.end_cause);
    }
    tally.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
    let unmeasured = |b: &Bound| {
        b.start_cause != EdgeCause::Measured || b.end_cause != EdgeCause::Measured
    };
    let edges_bad: usize = tally
        .iter()
        .filter(|(c, _)| *c != EdgeCause::Measured)
        .map(|(_, n)| n)
        .sum();

    // Contiguous runs matter: one word the detector cannot see also breaks its
    // neighbours, because their search bands are derived from its nucleus. Runs are
    // the signature of that propagation, so they are worth naming explicitly.
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < bounds.len() {
        if unmeasured(&bounds[i]) {
            let a = i;
            while i + 1 < bounds.len() && unmeasured(&bounds[i + 1]) {
                i += 1;
            }
            if i > a {
                runs.push((a, i));
            }
        }
        i += 1;
    }

    let mut out = format!(
        "Cut-by-text clip report — {} words, {} nuclei, syllable period {:.3}s, hop {:.3}s
         Unmeasured edges: {} of {} ({} words affected)
         By cause: {}
         Runs of consecutive unmeasured words: {}
         All times in seconds from the item start. drift = measured minus transcript;
         an edge that could not be measured shows -- and falls back to the transcript.
         run = length in FRAMES of the quiet run each edge came from (- = none).
         tilt = peak high-band-over-voice-bar dominance, dB, in that edge's search
         window. A fricative should stand out here; if a /sch/ junction reads no
         higher than a vowel-to-vowel one, it cannot be found by band balance at all.
         A run of 1-2 frames is not a pause, it is the flattest point of continuous
         speech, so an edge taken from it carries no information about the boundary.
         idx word            transcript        measured         drift start/end     gap  why
",
        ctx.words.len(),
        ctx.analysis.nuclei.len(),
        ctx.analysis.syllable_period,
        ctx.analysis.hop(),
        edges_bad,
        bounds.len() * 2,
        bounds.iter().filter(|b| unmeasured(b)).count(),
        tally
            .iter()
            .map(|(c, n)| format!("{} {}", c.token(), n))
            .collect::<Vec<_>>()
            .join(" | "),
        if runs.is_empty() {
            "none".to_string()
        } else {
            let shown: Vec<String> = runs.iter().take(40).map(|(a, b)| format!("{a}-{b}")).collect();
            let more = runs.len().saturating_sub(shown.len());
            let mut t = shown.join(", ");
            if more > 0 {
                t.push_str(&format!(" (+{more} more)"));
            }
            t
        },
    );
    for (i, w) in ctx.words.iter().enumerate() {
        let Some(b) = bounds.get(i) else { continue };
        let gap = bounds.get(i + 1).map(|n| n.start - b.end);
        let txt: String = w.text.chars().take(14).collect();
        // A fallback used to print +0.000, which is exactly what a perfect
        // measurement prints. Show -- instead so the two can never be confused.
        let drift = |c: EdgeCause, d: f64| {
            if c == EdgeCause::Measured {
                format!("{d:+6.3}")
            } else {
                "    --".to_string()
            }
        };
        out.push_str(&format!(
            "{:>3} {:<15} {:>7.3}-{:<7.3} {:>7.3}-{:<7.3} {}/{} {}  {}/{} run {}/{} tilt {:>5.1}/{:>5.1}{}
",
            i,
            txt,
            w.start,
            w.end,
            b.start,
            b.end,
            drift(b.start_cause, b.start - w.start),
            drift(b.end_cause, b.end - w.end),
            gap.map_or_else(|| "      -".to_string(), |g| format!("{g:7.3}")),
            b.start_cause.token(),
            b.end_cause.token(),
            fmt_run(b.runs.0),
            fmt_run(b.runs.1),
            b.tilt.0,
            b.tilt.1,
            if b.detail > 0.0 { format!(" by {:.3}", b.detail) } else { String::new() },
        ));
    }
    out
}

/// The KEPT segments, in item time, that a cut with these flags would leave —
/// computed by the same placement the cut itself uses. Empty when unavailable, in
/// which case the caller falls back to transcript times.
pub fn kept_segments(keep: &[bool]) -> Vec<(f64, f64)> {
    let g = match CTX.lock() {
        Ok(g) => g,
        Err(_) => return Vec::new(),
    };
    let Some(ctx) = g.as_ref() else {
        return Vec::new();
    };
    let spans = crate::edit::remove_spans_from_kept(&ctx.words, keep);
    if spans.is_empty() {
        return Vec::new();
    }
    let last = ctx.words.last().map(|w| w.end).unwrap_or(0.0);
    let mut removes: Vec<(f64, f64)> = Vec::new();
    for s in &spans {
        let prev = (s.first_word > 0)
            .then(|| ctx.words.get(s.first_word - 1))
            .flatten()
            .map(|w| (w.start, w.end));
        let next = ctx.words.get(s.last_word + 1).map(|w| (w.start, w.end));
        let (ap, an) = ctx.analysis.anchors(prev, (s.start, s.end), next);
        let placed = match (ap, an) {
            (Some(a), Some(b)) => ctx
                .analysis
                .place_removal(a, b, (s.start, s.end), (prev.map(|p| p.1), next.map(|n| n.0)))
                .ok(),
            _ => None,
        };
        // Same fallbacks as the cut: an unmeasured edge keeps the transcript time.
        let (st, en) = match placed {
            Some(p) => (p.start.unwrap_or(s.start), p.end.unwrap_or(s.end)),
            None => (s.start, s.end),
        };
        removes.push((st.min(s.start), en.max(s.end)));
    }
    removes.sort_by(|a, b| a.0.total_cmp(&b.0));
    // Invert: what is left after those removals.
    let mut kept_out = Vec::new();
    let mut cursor = 0.0f64;
    for (a, b) in removes {
        if a > cursor + 0.005 {
            kept_out.push((cursor, a));
        }
        cursor = cursor.max(b);
    }
    if last > cursor + 0.005 {
        kept_out.push((cursor, last));
    }
    kept_out
}
