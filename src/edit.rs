//! Cut-by-text planning: align an edited transcript back to word-level timings
//! to decide which spans of the original audio to keep and which to cut.
//!
//! This is the deterministic, host-independent core of the cut-by-text workflow
//! (transcribe an item -> the user deletes the words they want gone -> cut the
//! removed audio). Like [`crate::dsp`], it touches no REAPER/FFI state, so it is
//! fully unit-testable in CI. The REAPER side — splitting the item and
//! ripple-deleting the removed spans — is a thin executor built on this plan.
//!
//! ## How alignment works
//!
//! The user starts from the transcript and *deletes* the words they want gone.
//! We get back the remaining text and must map it onto the original words (each
//! carrying a start/end time). Tokens are matched on an alphanumeric-normalised
//! form (case- and punctuation-insensitive) via a greedy left-to-right
//! subsequence walk: every kept token claims the earliest not-yet-passed
//! original word. Original words no kept token lands on are what we cut.
//!
//! When the edit is pure deletion — the intended workflow — the kept text is a
//! subsequence of the original, so every token matches and the result is exact.
//! Text the user *added* (typed anew) matches nothing and is counted in
//! `unmatched_tokens`, so the caller can warn that it will not be spoken.
//!
//! ## Limits (deliberate, for now)
//!
//! - **Moved text is not reordered.** If the user cut a sentence and pasted it
//!   elsewhere, its words no longer line up in order: they read as removed at the
//!   origin and unmatched at the destination. The plan will *cut* the moved audio
//!   and raise `unmatched_tokens`. True move support is deferred until the editor
//!   exists — it can capture a move as an explicit operation rather than leaving
//!   us to infer one from text.
//! - **Repeated phrases match their earliest occurrence.** Deleting one of two
//!   identical sentences is ambiguous from text alone, so the plan makes a
//!   consistent, earliest-preserving choice. The editor will later disambiguate
//!   by handing us the kept word indices directly instead of reconstructed text —
//!   [`spans_from_matched`] is factored out for exactly that path.
//!
//! Times are on the transcript's own timeline (seconds from the start of the
//! transcribed audio); mapping them onto the project/take timeline is the
//! executor's job.

use crate::providers::transcription::Word;

/// A run of consecutive original words treated as one continuous span of the
/// original audio timeline. Word indices are into the slice passed to
/// [`plan_cut`].
#[derive(Debug, Clone, PartialEq)]
pub struct Span {
    /// Start on the transcript timeline (seconds).
    pub start: f64,
    /// End on the transcript timeline (seconds); always `>= start`.
    pub end: f64,
    /// First word in the run.
    pub first_word: usize,
    /// Last word in the run (inclusive).
    pub last_word: usize,
}

impl Span {
    /// Duration in seconds (never negative).
    pub fn seconds(&self) -> f64 {
        (self.end - self.start).max(0.0)
    }

    /// Number of original words in the run.
    pub fn words(&self) -> usize {
        self.last_word - self.first_word + 1
    }
}

/// The result of aligning an edited transcript against the original word timings.
#[derive(Debug, Clone, PartialEq)]
pub struct CutPlan {
    /// Audio to keep, ascending on the timeline.
    pub keep: Vec<Span>,
    /// Audio to cut, ascending on the timeline — the ripple-delete list.
    pub remove: Vec<Span>,
    /// Kept tokens that matched no original word (text typed rather than kept).
    /// They produce no audio; surfaced so the caller can warn the user.
    pub unmatched_tokens: usize,
}

impl CutPlan {
    /// Total audio removed, in seconds.
    pub fn removed_seconds(&self) -> f64 {
        self.remove.iter().map(Span::seconds).sum()
    }

    /// Total audio kept, in seconds.
    pub fn kept_seconds(&self) -> f64 {
        self.keep.iter().map(Span::seconds).sum()
    }

    /// True when the edit removes nothing (the user deleted no transcript text).
    pub fn is_noop(&self) -> bool {
        self.remove.is_empty()
    }
}

/// Normalise a token for matching: keep only alphanumerics, lower-cased. This
/// makes matching insensitive to case, surrounding punctuation, and the leading
/// space Whisper attaches to word tokens (`" Don't,"` and `"don't"` both become
/// `"dont"`).
fn normalize(token: &str) -> String {
    token
        .chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// Plan a cut from the original word timings and the edited (kept) transcript
/// text. See the module docs for the matching model and its limits.
pub fn plan_cut(words: &[Word], edited: &str) -> CutPlan {
    let orig: Vec<String> = words.iter().map(|w| normalize(&w.text)).collect();
    let kept: Vec<String> = edited
        .split_whitespace()
        .map(normalize)
        .filter(|t| !t.is_empty())
        .collect();

    // Greedy left-to-right subsequence match: each kept token claims the earliest
    // original word at or after the previous claim. Exact when the edit is pure
    // deletion (kept is a subsequence of orig); a token that finds no match ahead
    // is counted as unmatched and does NOT advance the cursor, so a stray typed
    // word can't derail the words that follow it.
    let mut matched = vec![false; orig.len()];
    let mut cursor = 0usize;
    let mut unmatched_tokens = 0usize;
    for token in &kept {
        match orig[cursor..].iter().position(|o| o == token) {
            Some(rel) => {
                let idx = cursor + rel;
                matched[idx] = true;
                cursor = idx + 1;
            }
            None => unmatched_tokens += 1,
        }
    }

    let (keep, remove) = spans_from_matched(words, &matched);
    CutPlan {
        keep,
        remove,
        unmatched_tokens,
    }
}

/// Group consecutive original words into keep/remove spans from a per-word
/// "matched" flag. Factored out of [`plan_cut`] so a future editor path — which
/// can hand us kept-word flags directly, with no text to reconstruct — reuses
/// the same span construction.
fn spans_from_matched(words: &[Word], matched: &[bool]) -> (Vec<Span>, Vec<Span>) {
    debug_assert_eq!(words.len(), matched.len());
    let mut keep = Vec::new();
    let mut remove = Vec::new();
    let mut i = 0;
    while i < words.len() {
        let flag = matched[i];
        let start = i;
        while i < words.len() && matched[i] == flag {
            i += 1;
        }
        let span = span_of(words, start, i - 1);
        if flag {
            keep.push(span);
        } else {
            remove.push(span);
        }
    }
    (keep, remove)
}

/// Build the audio span covering original words `first..=last`. Uses the min
/// start / max end across the run so a non-monotonic word timing (Whisper can
/// occasionally emit an `end` past the next word's `start`, or even `end < start`)
/// can never produce a negative-length or backwards span.
fn span_of(words: &[Word], first: usize, last: usize) -> Span {
    let run = &words[first..=last];
    let start = run.iter().map(|w| w.start).fold(f64::INFINITY, f64::min);
    let end = run.iter().map(|w| w.end).fold(start, f64::max);
    Span {
        start,
        end,
        first_word: first,
        last_word: last,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a word list from `(start, end, text)` triples.
    fn words(spec: &[(f64, f64, &str)]) -> Vec<Word> {
        spec.iter()
            .map(|&(start, end, text)| Word {
                start,
                end,
                text: text.to_string(),
            })
            .collect()
    }

    fn sample() -> Vec<Word> {
        words(&[
            (0.0, 0.5, "Hello"),
            (0.6, 1.0, "brave"),
            (1.1, 1.4, "new"),
            (1.5, 2.0, "world"),
        ])
    }

    #[test]
    fn pure_deletion_cuts_the_removed_span() {
        let plan = plan_cut(&sample(), "hello world");
        // Two kept runs (either side of the hole), one removed run between them.
        assert_eq!(plan.keep.len(), 2);
        assert_eq!(plan.keep[0].first_word, 0);
        assert_eq!(plan.keep[0].last_word, 0);
        assert_eq!(plan.keep[1].first_word, 3);
        assert_eq!(plan.remove.len(), 1);
        assert_eq!(plan.remove[0].first_word, 1);
        assert_eq!(plan.remove[0].last_word, 2);
        assert_eq!(plan.remove[0].start, 0.6);
        assert_eq!(plan.remove[0].end, 1.4);
        assert!((plan.removed_seconds() - 0.8).abs() < 1e-9);
        assert_eq!(plan.unmatched_tokens, 0);
        assert!(!plan.is_noop());
    }

    #[test]
    fn contiguous_kept_words_merge_into_one_span() {
        // Keeping two adjacent original words = one continuous audio range, no
        // cut between them.
        let plan = plan_cut(&sample(), "hello brave");
        assert_eq!(plan.keep.len(), 1);
        assert_eq!(plan.keep[0].first_word, 0);
        assert_eq!(plan.keep[0].last_word, 1);
        assert_eq!(plan.keep[0].start, 0.0);
        assert_eq!(plan.keep[0].end, 1.0);
        assert_eq!(plan.keep[0].words(), 2);
        assert_eq!(plan.remove.len(), 1);
        assert_eq!(plan.remove[0].first_word, 2);
        assert_eq!(plan.remove[0].last_word, 3);
    }

    #[test]
    fn matching_ignores_case_and_punctuation() {
        // Original tokens carry capitals and trailing punctuation; the edited text
        // is lower-case and bare. They must still line up.
        let ws = words(&[
            (0.0, 0.5, "Hello,"),
            (0.6, 1.0, "World."),
            (1.1, 1.5, "Bye!"),
        ]);
        let plan = plan_cut(&ws, "hello world");
        assert_eq!(plan.unmatched_tokens, 0);
        assert_eq!(plan.keep.len(), 1);
        assert_eq!(plan.keep[0].last_word, 1);
        assert_eq!(plan.remove.len(), 1);
        assert_eq!(plan.remove[0].first_word, 2); // "Bye!" cut
    }

    #[test]
    fn typed_words_are_counted_but_dont_derail_matching() {
        // "zzz" exists nowhere in the original: it's counted unmatched and does
        // not consume the cursor, so "world" still matches correctly afterwards.
        let plan = plan_cut(&sample(), "hello zzz world");
        assert_eq!(plan.unmatched_tokens, 1);
        assert_eq!(plan.keep.len(), 2);
        assert_eq!(plan.keep[1].first_word, 3);
        assert_eq!(plan.remove.len(), 1);
        assert_eq!(plan.remove[0].first_word, 1);
        assert_eq!(plan.remove[0].last_word, 2);
    }

    #[test]
    fn deleting_all_text_removes_everything() {
        let plan = plan_cut(&sample(), "   ");
        assert!(plan.keep.is_empty());
        assert_eq!(plan.remove.len(), 1);
        assert_eq!(plan.remove[0].first_word, 0);
        assert_eq!(plan.remove[0].last_word, 3);
        assert!((plan.removed_seconds() - 2.0).abs() < 1e-9);
        assert!(!plan.is_noop());
    }

    #[test]
    fn keeping_everything_is_a_noop() {
        let plan = plan_cut(&sample(), "Hello brave new world");
        assert!(plan.is_noop());
        assert!(plan.remove.is_empty());
        assert_eq!(plan.keep.len(), 1);
        assert_eq!(plan.keep[0].first_word, 0);
        assert_eq!(plan.keep[0].last_word, 3);
        assert!((plan.kept_seconds() - 2.0).abs() < 1e-9);
        assert_eq!(plan.unmatched_tokens, 0);
    }

    #[test]
    fn repeated_phrase_matches_earliest_occurrence() {
        // "go left go right" -> "go right": earliest-preserving greedy keeps the
        // first "go" and the "right", cutting the middle "left go". The choice is
        // ambiguous from text alone, but it is deterministic.
        let ws = words(&[
            (0.0, 0.4, "go"),
            (0.5, 0.9, "left"),
            (1.0, 1.4, "go"),
            (1.5, 1.9, "right"),
        ]);
        let plan = plan_cut(&ws, "go right");
        assert_eq!(plan.unmatched_tokens, 0);
        assert_eq!(plan.keep.len(), 2);
        assert_eq!(plan.keep[0].first_word, 0);
        assert_eq!(plan.keep[1].first_word, 3);
        assert_eq!(plan.remove.len(), 1);
        assert_eq!(plan.remove[0].first_word, 1);
        assert_eq!(plan.remove[0].last_word, 2);
    }

    #[test]
    fn non_monotonic_word_timings_never_make_a_backwards_span() {
        // A kept run whose second word ends before the first does: span end is the
        // max end, and is clamped to be >= start.
        let ws = words(&[
            (0.0, 1.0, "one"),
            (0.5, 0.8, "two"), // end 0.8 < previous end 1.0
        ]);
        let plan = plan_cut(&ws, "one two");
        assert_eq!(plan.keep.len(), 1);
        assert_eq!(plan.keep[0].start, 0.0);
        assert_eq!(plan.keep[0].end, 1.0);
        assert!(plan.keep[0].end >= plan.keep[0].start);

        // A lone word whose end precedes its start collapses to zero length, not a
        // negative one.
        let ws = words(&[(2.0, 1.5, "oops")]);
        let plan = plan_cut(&ws, "oops");
        assert_eq!(plan.keep.len(), 1);
        assert_eq!(plan.keep[0].start, 2.0);
        assert_eq!(plan.keep[0].end, 2.0);
        assert_eq!(plan.keep[0].seconds(), 0.0);
    }

    #[test]
    fn empty_transcript_yields_an_empty_plan() {
        let plan = plan_cut(&[], "anything typed here");
        assert!(plan.keep.is_empty());
        assert!(plan.remove.is_empty());
        assert!(plan.is_noop());
        assert_eq!(plan.unmatched_tokens, 3);
    }
}
