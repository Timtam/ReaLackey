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
