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

use crate::providers::transcription::{Segment, Word};

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
/// "matched" flag. Factored out of [`plan_cut`] so the editor path — which hands
/// us kept-word flags directly, with no text to reconstruct — reuses the same
/// span construction.
pub(crate) fn spans_from_matched(words: &[Word], matched: &[bool]) -> (Vec<Span>, Vec<Span>) {
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

/// One resulting fragment of an item after a cut, in timeline order.
#[derive(Debug, Clone, PartialEq)]
pub struct CutPiece {
    /// Original project-time start of the fragment.
    pub start: f64,
    /// Original project-time end of the fragment.
    pub end: f64,
    /// True if the fragment is cut (deleted); false if kept.
    pub removed: bool,
    /// Where a *kept* fragment moves to once earlier removed spans close up.
    /// Equal to `start` for removed fragments (unused).
    pub new_start: f64,
}

/// A concrete, host-independent split/compact plan for cutting spans out of one
/// media item and closing the gaps. Produced by [`plan_item_cut`]; consumed by
/// the REAPER executor, which splits the item at [`ItemCut::splits`], deletes the
/// removed fragments and slides the kept ones to their `new_start`.
#[derive(Debug, Clone, PartialEq)]
pub struct ItemCut {
    /// Interior project-time positions to split the item at, strictly ascending.
    /// Splitting there (left to right) yields `pieces.len()` fragments in order.
    pub splits: Vec<f64>,
    /// The fragments in timeline order (kept and removed interleaved).
    pub pieces: Vec<CutPiece>,
    /// Total removed duration — how much the item (and, if the caller ripples the
    /// track, everything after it) shortens.
    pub removed_seconds: f64,
}

impl ItemCut {
    /// True when nothing is cut.
    pub fn is_noop(&self) -> bool {
        !self.pieces.iter().any(|p| p.removed)
    }

    /// True when every fragment is removed — the cut would delete the whole item.
    pub fn removes_everything(&self) -> bool {
        !self.pieces.is_empty() && self.pieces.iter().all(|p| p.removed)
    }

    /// The item's right edge after kept fragments are compacted (its new length is
    /// `kept_end() - item_start`).
    pub fn kept_end(&self, item_start: f64) -> f64 {
        item_start + (self.item_span() - self.removed_seconds).max(0.0)
    }

    fn item_span(&self) -> f64 {
        self.pieces.last().map_or(0.0, |p| p.end) - self.pieces.first().map_or(0.0, |p| p.start)
    }
}

/// Resolve a set of remove ranges (project time) against one item's project span
/// `[item_start, item_end]` into a concrete split/compact plan. Ranges are clamped
/// to the item, dropped when empty after clamping, sorted, and merged when they
/// overlap or sit within `epsilon` of each other. `epsilon` also keeps splits away
/// from an edge (or each other) so REAPER never gets asked to carve a zero-length
/// sliver.
///
/// The plan keeps everything the ranges don't cover — including the leading and
/// trailing audio and the pauses between words — and slides the kept fragments
/// together so they play back-to-back.
pub fn plan_item_cut(
    item_start: f64,
    item_end: f64,
    removes: &[(f64, f64)],
    epsilon: f64,
) -> ItemCut {
    // Clamp to the item, drop empties, sort by start.
    let mut norm: Vec<(f64, f64)> = removes
        .iter()
        .map(|&(a, b)| (a.max(item_start), b.min(item_end)))
        .filter(|&(a, b)| b - a > epsilon)
        .collect();
    norm.sort_by(|x, y| x.0.total_cmp(&y.0));

    // Merge overlapping / near-touching ranges so the gaps between them can't
    // become sub-epsilon kept slivers. (This keeps the pieces tiling the item
    // exactly. Protecting a real kept word from two converging removals is done
    // upstream, before the ranges get here — see the snap guard in the executor.)
    let mut merged: Vec<(f64, f64)> = Vec::new();
    for (a, b) in norm {
        match merged.last_mut() {
            Some(last) if a <= last.1 + epsilon => last.1 = last.1.max(b),
            _ => merged.push((a, b)),
        }
    }

    // Walk the item start to end, emitting a kept fragment before each removed
    // range and a removed fragment for the range itself, then the trailing keep.
    let mut pieces: Vec<CutPiece> = Vec::new();
    let mut cursor = item_start;
    let mut kept_cursor = item_start; // compaction target for kept fragments
    let mut removed_seconds = 0.0;
    for (a, b) in &merged {
        let (a, b) = (*a, *b);
        if a - cursor > epsilon {
            pieces.push(CutPiece {
                start: cursor,
                end: a,
                removed: false,
                new_start: kept_cursor,
            });
            kept_cursor += a - cursor;
        }
        pieces.push(CutPiece {
            start: a,
            end: b,
            removed: true,
            new_start: a,
        });
        removed_seconds += b - a;
        cursor = b;
    }
    if item_end - cursor > epsilon {
        pieces.push(CutPiece {
            start: cursor,
            end: item_end,
            removed: false,
            new_start: kept_cursor,
        });
    }

    // Splits are the interior fragment boundaries — every fragment start but the
    // first (which is the item's own left edge).
    let splits: Vec<f64> = pieces.iter().skip(1).map(|p| p.start).collect();

    ItemCut {
        splits,
        pieces,
        removed_seconds,
    }
}

/// The remove spans (audio to cut) implied by a per-word KEPT flag array — the
/// editor's Save path. `kept[i] == true` keeps word `i`. A flag array shorter or
/// longer than `words` is tolerated (a missing flag counts as kept), so a stale
/// payload can never cut audio the editor didn't mark.
pub(crate) fn remove_spans_from_kept(words: &[Word], kept: &[bool]) -> Vec<Span> {
    let matched: Vec<bool> = (0..words.len())
        .map(|i| kept.get(i).copied().unwrap_or(true))
        .collect();
    spans_from_matched(words, &matched).1
}

/// Whether a word token ends a sentence: its text, after trailing quotes/brackets
/// are stripped, ends in `.`, `!`, `?`, or `…`.
fn ends_sentence(text: &str) -> bool {
    let t = text.trim_end_matches(|c: char| {
        c.is_whitespace() || matches!(c, '"' | '\'' | ')' | ']' | '»' | '”' | '’')
    });
    matches!(t.chars().last(), Some('.' | '!' | '?' | '…'))
}

/// Renumber a non-decreasing id list to contiguous ids starting at 0 (drops ids
/// that no element uses).
fn contiguous(ids: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(ids.len());
    let mut cur = 0usize;
    let mut prev: Option<usize> = None;
    for &x in ids {
        if let Some(p) = prev {
            if x != p {
                cur += 1;
            }
        }
        out.push(cur);
        prev = Some(x);
    }
    out
}

/// Assign every word a sentence id, for the editor's Up/Down navigation.
///
/// Primary: split on sentence-ending punctuation. If the transcript carries no
/// such punctuation (some ASR emits none), fall back to the transcript's own
/// segments as sentence units. If neither yields a split, it's one sentence.
pub(crate) fn sentence_ids(words: &[Word], segments: &[Segment]) -> Vec<usize> {
    if words.is_empty() {
        return Vec::new();
    }
    // Primary: punctuation. A word ending a sentence closes the current one.
    let mut ids = vec![0usize; words.len()];
    let mut cur = 0usize;
    for (i, w) in words.iter().enumerate() {
        ids[i] = cur;
        if ends_sentence(&w.text) {
            cur += 1;
        }
    }
    let sentences = ids.last().map_or(0, |&x| x) + 1;
    if sentences >= 2 {
        return ids;
    }
    // Fallback: use segments as sentences (map each word to the segment covering
    // its start), when punctuation gave us nothing to split on.
    if segments.len() >= 2 {
        let mut seg = vec![0usize; words.len()];
        let mut si = 0usize;
        for (i, w) in words.iter().enumerate() {
            while si + 1 < segments.len() && w.start >= segments[si + 1].start {
                si += 1;
            }
            seg[i] = si;
        }
        let out = contiguous(&seg);
        if out.last().map_or(0, |&x| x) >= 1 {
            return out;
        }
    }
    ids // one sentence
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

    const EPS: f64 = 1e-6;

    #[test]
    fn item_cut_removes_middle_and_compacts() {
        let cut = plan_item_cut(0.0, 10.0, &[(3.0, 5.0)], EPS);
        assert_eq!(cut.splits, vec![3.0, 5.0]);
        assert_eq!(cut.pieces.len(), 3);
        assert_eq!(cut.pieces[0], keep(0.0, 3.0, 0.0));
        assert!(cut.pieces[1].removed);
        assert_eq!((cut.pieces[1].start, cut.pieces[1].end), (3.0, 5.0));
        // The trailing keep slides left by the 2 s hole: 5.0 -> 3.0.
        assert_eq!(cut.pieces[2], keep(5.0, 10.0, 3.0));
        assert_eq!(cut.removed_seconds, 2.0);
        assert_eq!(cut.kept_end(0.0), 8.0);
        assert!(!cut.is_noop());
        assert!(!cut.removes_everything());
    }

    #[test]
    fn item_cut_compacts_multiple_holes() {
        let cut = plan_item_cut(0.0, 20.0, &[(2.0, 4.0), (10.0, 12.0)], EPS);
        assert_eq!(cut.splits, vec![2.0, 4.0, 10.0, 12.0]);
        let kept: Vec<_> = cut.pieces.iter().filter(|p| !p.removed).collect();
        assert_eq!(kept.len(), 3);
        assert_eq!(kept[0].new_start, 0.0); // [0,2)
        assert_eq!(kept[1].new_start, 2.0); // [4,10) slides back 2 s
        assert_eq!(kept[2].new_start, 8.0); // [12,20) slides back 4 s
        assert_eq!(cut.removed_seconds, 4.0);
        assert_eq!(cut.kept_end(0.0), 16.0);
    }

    #[test]
    fn item_cut_merges_near_touching_removes() {
        // A 5 ms gap under a 10 ms epsilon fuses the two removes into one.
        let cut = plan_item_cut(0.0, 10.0, &[(2.0, 4.0), (4.005, 6.0)], 0.01);
        assert_eq!(cut.splits, vec![2.0, 6.0]);
        assert_eq!(cut.removed_seconds, 4.0);
        let kept: Vec<_> = cut.pieces.iter().filter(|p| !p.removed).collect();
        assert_eq!(kept.len(), 2); // no sub-epsilon sliver between the removes
    }

    #[test]
    fn item_cut_at_the_start_has_no_leading_keep() {
        let cut = plan_item_cut(0.0, 10.0, &[(0.0, 3.0)], EPS);
        assert_eq!(cut.splits, vec![3.0]);
        assert_eq!(cut.pieces.len(), 2);
        assert!(cut.pieces[0].removed);
        assert_eq!(cut.pieces[1], keep(3.0, 10.0, 0.0)); // slides to the item start
    }

    #[test]
    fn item_cut_at_the_end_has_no_trailing_keep() {
        let cut = plan_item_cut(0.0, 10.0, &[(7.0, 10.0)], EPS);
        assert_eq!(cut.splits, vec![7.0]);
        assert_eq!(cut.pieces.len(), 2);
        assert_eq!(cut.pieces[0], keep(0.0, 7.0, 0.0));
        assert!(cut.pieces[1].removed);
    }

    #[test]
    fn item_cut_whole_item() {
        let cut = plan_item_cut(0.0, 10.0, &[(0.0, 10.0)], EPS);
        assert!(cut.splits.is_empty());
        assert_eq!(cut.pieces.len(), 1);
        assert!(cut.removes_everything());
        assert_eq!(cut.removed_seconds, 10.0);
        assert_eq!(cut.kept_end(0.0), 0.0);
    }

    #[test]
    fn item_cut_is_a_noop_when_nothing_qualifies() {
        // Empty list, and a sub-epsilon range, both leave the item whole.
        for removes in [vec![], vec![(3.0, 3.0000001)]] {
            let cut = plan_item_cut(0.0, 10.0, &removes, EPS);
            assert!(cut.is_noop(), "{removes:?}");
            assert!(cut.splits.is_empty());
            assert_eq!(cut.pieces.len(), 1);
            assert_eq!(cut.removed_seconds, 0.0);
        }
    }

    #[test]
    fn item_cut_clamps_ranges_to_the_item() {
        // Ranges reaching past either edge are clipped to the item's span.
        let cut = plan_item_cut(0.0, 10.0, &[(-5.0, 2.0), (8.0, 20.0)], EPS);
        assert_eq!(cut.splits, vec![2.0, 8.0]);
        assert_eq!(cut.removed_seconds, 4.0); // (0,2) + (8,10)
        let kept: Vec<_> = cut.pieces.iter().filter(|p| !p.removed).collect();
        assert_eq!(kept.len(), 1);
        assert_eq!(*kept[0], keep(2.0, 8.0, 0.0));
    }

    /// A kept [`CutPiece`] with the given original span and compacted position.
    fn keep(start: f64, end: f64, new_start: f64) -> CutPiece {
        CutPiece {
            start,
            end,
            removed: false,
            new_start,
        }
    }

    fn segs(spec: &[(f64, f64, &str)]) -> Vec<Segment> {
        spec.iter()
            .map(|&(start, end, text)| Segment {
                start,
                end,
                text: text.to_string(),
            })
            .collect()
    }

    #[test]
    fn remove_spans_follow_the_kept_flags() {
        let ws = sample(); // Hello brave new world
        // Keep Hello + world; drop brave + new.
        let rm = remove_spans_from_kept(&ws, &[true, false, false, true]);
        assert_eq!(rm.len(), 1);
        assert_eq!((rm[0].first_word, rm[0].last_word), (1, 2));
        assert_eq!((rm[0].start, rm[0].end), (0.6, 1.4));
        // A short flag array counts the missing tail as kept (never over-cuts).
        let rm = remove_spans_from_kept(&ws, &[false]);
        assert_eq!(rm.len(), 1);
        assert_eq!(rm[0].first_word, 0);
        assert_eq!(rm[0].last_word, 0);
    }

    #[test]
    fn sentences_split_on_punctuation() {
        let ws = words(&[
            (0.0, 0.4, "Hello"),
            (0.5, 0.9, "world."),
            (1.0, 1.4, "How"),
            (1.5, 1.8, "are"),
            (1.9, 2.3, "you?"),
        ]);
        // Punctuation wins; segments (if any) are ignored.
        let ids = sentence_ids(&ws, &segs(&[(0.0, 2.3, "everything in one segment")]));
        assert_eq!(ids, vec![0, 0, 1, 1, 1]);
    }

    #[test]
    fn sentences_fall_back_to_segments_without_punctuation() {
        let ws = words(&[
            (0.0, 0.4, "hello"),
            (0.5, 0.9, "world"),
            (2.1, 2.5, "second"),
            (2.6, 3.0, "part"),
        ]);
        let ids = sentence_ids(&ws, &segs(&[(0.0, 2.0, "hello world"), (2.0, 3.0, "second part")]));
        assert_eq!(ids, vec![0, 0, 1, 1]);
    }

    #[test]
    fn one_sentence_when_no_punctuation_and_no_segments() {
        let ws = words(&[(0.0, 0.4, "just"), (0.5, 0.9, "words")]);
        assert_eq!(sentence_ids(&ws, &[]), vec![0, 0]);
        assert!(sentence_ids(&[], &[]).is_empty());
    }

    #[test]
    fn sentence_end_survives_a_trailing_quote() {
        assert!(ends_sentence("world.\""));
        assert!(ends_sentence("really?”"));
        assert!(!ends_sentence("mid-word"));
    }
}
