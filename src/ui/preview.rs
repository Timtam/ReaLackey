//! Cut preview: the editor's Play button must audition EXACTLY what Confirm will
//! write, so it asks Rust for the real cut boundaries instead of guessing from the
//! transcript.
//!
//! Without this the preview played transcript times with a fixed 30 ms lead, while
//! the cut used measured boundaries — so the two disagreed by design, and every
//! judgement made from the preview was made against something that never got written.

use std::sync::Mutex;

use crate::dsp::speech::SpeechAnalysis;
use crate::providers::transcription::Word;

/// Everything needed to re-derive the cut for the item currently open in the editor.
struct Ctx {
    analysis: SpeechAnalysis,
    words: Vec<Word>,
}

static CTX: Mutex<Option<Ctx>> = Mutex::new(None);

/// Cache the analysis for the item the editor is about to open. Cheap: it reuses the
/// WAV already rendered for in-editor playback, so no extra audio read.
pub fn arm(wav: &[u8], words: &[Word]) {
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
        *g = Some(Ctx { analysis, words: words.to_vec() });
    }
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
/// this word. Matches the tolerance the nucleus search already allows.
const NEIGHBOUR_BLEED: f64 = 0.040;

pub fn word_bounds() -> Vec<(f64, f64)> {
    let g = match CTX.lock() {
        Ok(g) => g,
        Err(_) => return Vec::new(),
    };
    let Some(ctx) = g.as_ref() else {
        return Vec::new();
    };
    (0..ctx.words.len())
        .map(|i| {
            let w = &ctx.words[i];
            let prev = i.checked_sub(1).and_then(|j| ctx.words.get(j)).map(|p| (p.start, p.end));
            let next = ctx.words.get(i + 1).map(|n| (n.start, n.end));
            let (ap, an) = ctx.analysis.anchors(prev, (w.start, w.end), next);
            // The word's own extent, NOT the cut boundaries: those sit inside the
            // surrounding pause by design.
            let (o, f) = match (ap, an) {
                (Some(a), Some(b)) => ctx.analysis.word_extent(a, b, (w.start, w.end)),
                _ => (None, None),
            };
            // A measured edge may bleed a little past the neighbour's transcript
            // boundary, but not INTO the neighbour. Without this, a gap too short to
            // register as a quiet run makes the search skip over the next word and take
            // the run after it — the report showed ~10 pairs resolving to the identical
            // run, worst "Robot" running 0.68 s past its end and swallowing the word
            // after it. `.min`/`.max` against the word's own hint so an overlapping
            // transcript (numbers, hyphenated tokens) can never tighten the bound past
            // where the word itself claims to be.
            let lo = prev.map(|(_, pe)| pe.min(w.start) - NEIGHBOUR_BLEED);
            let hi = next.map(|(ns, _)| ns.max(w.end) + NEIGHBOUR_BLEED);
            let o = o.filter(|&t| lo.map_or(true, |l| t >= l));
            let f = f.filter(|&t| hi.map_or(true, |h| t <= h));
            (o.unwrap_or(w.start), f.unwrap_or(w.end))
        })
        .collect()
}

/// A whole-clip diagnostic: every word with its transcript times, its MEASURED
/// extent, the drift between them, and the measured gap to the next word.
///
/// One paste diagnoses a whole take. Without it every symptom costs a re-cut, and
/// only the words that happened to be cut are visible at all.
pub fn report() -> String {
    // Computed first: it takes the lock itself.
    let bounds = word_bounds();
    let g = match CTX.lock() {
        Ok(g) => g,
        Err(_) => return String::new(),
    };
    let Some(ctx) = g.as_ref() else {
        return "No clip is loaded in the editor.".into();
    };
    let mut out = format!(
        "Cut-by-text clip report — {} words, {} nuclei, syllable period {:.3}s
         All times in seconds from the item start. drift = measured minus transcript;
         large drift, or gap 0.000, is where to look.
         idx word            transcript        measured         drift start/end     gap
",
        ctx.words.len(),
        ctx.analysis.nuclei.len(),
        ctx.analysis.syllable_period,
    );
    for (i, w) in ctx.words.iter().enumerate() {
        let (o, f) = bounds.get(i).copied().unwrap_or((w.start, w.end));
        let gap = bounds.get(i + 1).map(|n| n.0 - f);
        let txt: String = w.text.chars().take(14).collect();
        out.push_str(&format!(
            "{:>3} {:<15} {:>7.3}-{:<7.3} {:>7.3}-{:<7.3} {:>+6.3}/{:>+6.3} {}
",
            i,
            txt,
            w.start,
            w.end,
            o,
            f,
            o - w.start,
            f - w.end,
            gap.map_or_else(|| "      -".to_string(), |g| format!("{g:7.3}"))
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
