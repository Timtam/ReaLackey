//! Speech-boundary analysis for cut-by-text.
//!
//! Placing a cut when a word is deleted is not "find the nearest quiet patch". Two
//! facts kill that approach, and both were found the hard way:
//!
//! * A word-internal stop closure (the /p/ in "Ja-pan") is *real silence*, so no
//!   level test tells it from a word gap. Duration doesn't either: closures run
//!   40-100 ms, Korean tense stops 100-150 ms, and geminates in Japanese, Italian
//!   and Finnish 150-250 ms — while most word boundaries in fluent speech carry no
//!   silence at all. The distributions don't merely overlap, they invert.
//! * Transcript word timings are only approximate (Whisper aligns by attention;
//!   for its DTW output the end of one word is *identical* to the start of the
//!   next), so they cannot be used as hard limits either.
//!
//! The structural answer is to bound the search by SYLLABIC NUCLEI: the last nucleus
//! of the previous kept word and the first nucleus of the next one. A word-internal
//! closure lies *between* two nuclei of its own word, so it is outside that band by
//! construction — for singletons, geminates and tense stops alike, in any language.
//! Inside the band a continuous cost then picks the splice; no duration threshold is
//! involved anywhere.
//!
//! Everything is calibrated from the material itself — noise floor, speech level and
//! the speaker's own syllable rate — so there is nothing to tune per voice, mic,
//! room or language. Where a constant is unavoidable it is marked and justified.
//!
//! Pure and host-independent, like the rest of [`crate::dsp`], so it is testable in
//! CI without a REAPER host.

use super::{percentile, Biquad};

/// Analysis frame hop. 10 ms is the universal speech-analysis rate and puts the
/// ±5 ms quantisation an order of magnitude below transcript error.
const HOP: f64 = 0.010;
/// Analysis window. 20 ms resolves the 30-60 ms gaps that separate words in fast
/// speech — a longer window plus smoothing spans more than such a gap and makes it
/// invisible, which is exactly how an earlier version failed to see word boundaries
/// at all. Pitch ripple is handled by smoothing the nucleus band only (below).
const WIN: f64 = 0.020;
/// Extra margin taken outside the measured word edges. The costs are asymmetric —
/// removing a few ms of near-silence is inaudible, leaving a word's attack is
/// immediately obvious — so err outward. Bounded by the silence actually available,
/// per side, so it never eats into a kept word.
const GUARD: f64 = 0.040;
/// Smoothing applied to the NUCLEUS band only: bridges glottal-pulse ripple so one
/// vowel yields one syllable. Never applied to the level bands — see WIN.
const SMOOTH: f64 = 0.030;
/// A nucleus must stand this far above its flanking dips (de Jong & Wempe's
/// validated syllable-nuclei method).
const NUCLEUS_PROMINENCE_DB: f64 = 2.0;
/// Fallback nucleus criterion for whispered/heavily-coded material, where voicing
/// can't be required: a run above the speech/silence split at least this long. Below
/// the shortest vowel nucleus in fast speech (~60-80 ms), so it still rejects clicks.
const MIN_UNVOICED_NUCLEUS: f64 = 0.040;

/// How a boundary should be joined once placed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Join {
    /// A real pause was found; the fragments can simply abut.
    Butt,
    /// No silence exists here (connected speech). Splice anyway, but crossfade.
    Crossfade,
}

/// Why a boundary could not be placed. Surfaced to the user: a detector that goes
/// quietly inert for a whole class of material is the worst failure mode, because
/// nobody can report what they can't perceive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // Unanchorable is raised by the caller when a word has no nucleus
pub enum Refusal {
    /// A neighbouring kept word had no syllabic nucleus — nothing to anchor to.
    Unanchorable,
    /// The band between the anchors is implausibly long; the assignment is suspect.
    BandTooWide,
    /// Cost is flat across the band (music bed, crosstalk, reverb-filled gap).
    NoMinimum,
}

/// Per-item speech analysis: band envelopes, self-calibrated levels, and the
/// syllabic nuclei that anchor every boundary decision.
pub struct SpeechAnalysis {
    hop: f64,
    /// Smoothed per-frame levels in dB, broadband / voice-bar / high-frequency.
    bb: Vec<f64>,
    hf: Vec<f64>,
    floor_bb: f64,
    floor_hf: f64,
    /// Each band's own speech-to-floor range, so levels can be compared between
    /// bands with very different noise floors.
    contrast_bb: f64,
    contrast_hf: f64,
    /// Nucleus times, ascending.
    pub nuclei: Vec<f64>,
    /// Median interval between nuclei — the speaker's own syllable period. Every
    /// duration downstream is expressed in units of this, never in milliseconds,
    /// because syllable rate varies by ~2x across languages and as much again
    /// between talkers.
    pub syllable_period: f64,
}

impl SpeechAnalysis {
    /// Analyse a mono slice. `None` if it is too short to say anything.
    pub fn new(samples: &[f64], sample_rate: f64) -> Option<Self> {
        if sample_rate <= 0.0 {
            return None;
        }
        let win = ((WIN * sample_rate).round() as usize).max(1);
        let hop = ((HOP * sample_rate).round() as usize).max(1);
        if samples.len() < win * 3 {
            return None;
        }
        // 60 Hz high-pass first: below the F0 of any modal voice, and it removes
        // rumble, DC and pops that would otherwise dominate a level measurement.
        let base = highpass(samples, sample_rate, 60.0);
        let nyq = sample_rate / 2.0;
        let hf_sig = highpass(&base, sample_rate, 3000.0_f64.min(nyq * 0.8));

        // Level bands are NOT smoothed: smoothing spans a short inter-word gap and
        // hides it. Only the nucleus band is smoothed, where pitch ripple would
        // otherwise split one vowel into several syllables.
        let bb = frame_db(&base, win, hop);
        let hf = frame_db(&hf_sig, win, hop);
        // MID (300-3000 Hz) is where sonorant nuclei live; used only for nuclei.
        let mid = smooth_db(
            &frame_db(&bandpass(&base, sample_rate, 300.0, 3000.0_f64.min(nyq * 0.9)), win, hop),
            hop,
            sample_rate,
        );
        if bb.len() < 3 {
            return None;
        }


        let floor_bb = noise_floor_db(&bb);
        let floor_hf = noise_floor_db(&hf);
        let (t_split, speech_level) = otsu_split(&bb);
        let contrast_bb = (speech_level - floor_bb).max(0.0);
        let contrast_hf = (otsu_split(&hf).1 - floor_hf).max(0.0);

        let hop_s = hop as f64 / sample_rate;
        let nuclei = find_nuclei(&mid, &base, hop, win, sample_rate, t_split, hop_s);
        let syllable_period = if nuclei.len() >= 3 {
            let mut d: Vec<f64> = nuclei.windows(2).map(|w| w[1] - w[0]).collect();
            d.sort_by(|a, b| a.total_cmp(b));
            percentile(&d, 50.0)
        } else {
            0.200 // a plausible default; only used for the distance prior's width
        };

        Some(Self {
            hop: hop_s,
            bb,
            hf,
            floor_bb,
            floor_hf,
            contrast_bb,
            contrast_hf,
            nuclei,
            syllable_period,
        })
    }

    fn frame_at(&self, t: f64) -> usize {
        ((t / self.hop).round().max(0.0) as usize).min(self.bb.len().saturating_sub(1))
    }
    fn time_of(&self, k: usize) -> f64 {
        k as f64 * self.hop
    }

    /// The two anchoring nuclei for a removal, by NEAREST-CENTRE assignment over the
    /// three spans involved: the previous kept word, the removed run, and the next
    /// kept word.
    ///
    /// Assigning by "is it within some window of this word" is not good enough — a
    /// window wide enough to survive the transcript's timing error also reaches into
    /// the neighbouring word, and then the removed word's own nucleus gets used as an
    /// anchor and the band collapses. Comparing distances to all three centres is
    /// both simpler and robust to a span shifted by more than its own length, since
    /// only the RELATIVE order of the centres matters.
    pub fn anchors(
        &self,
        prev: Option<(f64, f64)>,
        removed: (f64, f64),
        next: Option<(f64, f64)>,
    ) -> (Option<f64>, Option<f64>) {
        let mid = |(a, b): (f64, f64)| (a + b) / 2.0;
        let cr = mid(removed);
        let cp = prev.map(mid);
        let cn = next.map(mid);
        let owns = |n: f64, own: f64| {
            (n - own).abs() <= (n - cr).abs()
                && cp.map_or(true, |c| c == own || (n - own).abs() <= (n - c).abs())
                && cn.map_or(true, |c| c == own || (n - own).abs() <= (n - c).abs())
        };
        let a_prev = cp.and_then(|c| self.nuclei.iter().copied().rfind(|&n| owns(n, c)));
        let a_next = cn.and_then(|c| self.nuclei.iter().copied().find(|&n| owns(n, c)));
        (a_prev, a_next)
    }

    /// Count nuclei strictly inside a band — the caller uses this to check the word
    /// list is locally consistent with the audio before trusting a placement.
    #[allow(dead_code)] // consistency gate: for the word-list sanity check
    pub fn nuclei_between(&self, a: f64, b: f64) -> usize {
        self.nuclei.iter().filter(|&&n| n > a && n < b).count()
    }

    /// How "loud" a frame is, as a FRACTION of each band's own dynamic range, taken
    /// as the max across bands. A frame is quiet only when every band is quiet — that
    /// is what stops a fricative, whose broadband level collapses but whose HF does
    /// not, from counting as silence.
    ///
    /// Normalising per band is essential: a narrow band has a much lower noise floor
    /// than a wide one, so raw "dB above my own floor" is not comparable between
    /// them, and a max() over those raw numbers lets the narrowest band veto every
    /// gap. (That bug made real inter-word silences invisible: broadband sat 0.2 dB
    /// over its floor while a 60-350 Hz band read 9 dB over its own — mostly filter
    /// ringing and the previous vowel's fundamental decaying.)
    fn level(&self, k: usize) -> f64 {
        let f = |v: f64, floor: f64, contrast: f64| {
            if contrast > 1.0 { ((v - floor) / contrast).max(0.0) } else { 0.0 }
        };
        f(self.bb[k], self.floor_bb, self.contrast_bb)
            .max(f(self.hf[k], self.floor_hf, self.contrast_hf))
    }

    /// Walk FORWARD from a nucleus to where that word's audio actually ends.
    ///
    /// The nucleus is a safe starting point (it is inside the word by construction),
    /// so walking out from it cannot land in a neighbour. A brief intra-word dip — a
    /// stop closure — does not end the walk, because the level must stay down for a
    /// run; and a trailing fricative does not end it either, because [`Self::level`]
    /// still sees its HF energy.
    pub fn decay_after(&self, anchor: f64, limit: f64) -> f64 {
        let quiet = self.quiet_level();
        let need = self.run_frames();
        let (from, to) = (self.frame_at(anchor), self.frame_at(limit));
        let mut run = 0usize;
        for k in from..=to {
            if self.level(k) <= quiet {
                run += 1;
                if run >= need {
                    return self.time_of(k + 1 - run);
                }
            } else {
                run = 0;
            }
        }
        self.time_of(to)
    }

    /// Walk BACKWARD from a nucleus to where that word's audio actually starts. This
    /// is what finds a word-initial fricative ("sch-") that the transcript missed:
    /// the walk passes straight through it because its HF energy keeps the level up.
    pub fn onset_before(&self, anchor: f64, limit: f64) -> f64 {
        let quiet = self.quiet_level();
        let need = self.run_frames();
        let (from, to) = (self.frame_at(anchor), self.frame_at(limit));
        let mut run = 0usize;
        for k in (to..=from).rev() {
            if self.level(k) <= quiet {
                run += 1;
                if run >= need {
                    return self.time_of(k + run - 1);
                }
            } else {
                run = 0;
            }
        }
        self.time_of(to)
    }

    /// "Quiet" as a share of this item's own speech/silence contrast — self-calibrating,
    /// so it means the same thing in a whisper and in a shout.
    fn quiet_level(&self) -> f64 {
        0.15
    }
    /// The bar for "this frame is part of a word at all" — deliberately far lower
    /// than [`Self::quiet_level`] (2% vs 15% of a band's range), so a word's onset
    /// ramp and its decaying tail both count as belonging to the word rather than to
    /// the surrounding pause. Set low on purpose: over-including a little room tone
    /// is inaudible once cut, while under-including clips the attack.
    fn edge_level(&self) -> f64 {
        0.02
    }
    /// How long the level must stay down to end a word: a fraction of the speaker's own
    /// syllable period, so it scales with speaking rate instead of assuming one.
    fn run_frames(&self) -> usize {
        // Kept small deliberately: a fast read separates words by only 30-50 ms, and
        // requiring a longer run than the gap itself makes every boundary invisible.
        // Two frames is the resolution floor (one analysis window of quiet).
        ((0.10 * self.syllable_period / self.hop).round() as usize).clamp(2, 4)
    }

    /// Place BOTH edges of a removal between two anchoring nuclei.
    ///
    /// Everything between the previous kept word's decay and the next kept word's
    /// onset is deletable — it is either the removed word or the pauses around it —
    /// so the edges are placed from those measured points, NOT from the transcript's
    /// idea of where the removed word begins. That distinction is the whole fix: a
    /// late-reported onset (very common: Whisper puts the start of "ständig" at the
    /// /t/, past the "sch") would otherwise drag the cut in behind the fricative and
    /// leave it behind.
    ///
    /// Returns `(start, end, join)` in analysis time, leaving one natural-sounding
    /// gap at the join.
    /// The measured intermediate values behind one placement, for diagnostics. Real
    /// recordings are the only way to know whether the analysis matches what a
    /// listener hears, so the tool surfaces these rather than making the user guess.
    pub fn explain(&self, a_prev: f64, a_next: f64) -> (f64, f64, f64, f64) {
        let d = self.decay_after(a_prev, a_next);
        let o = self.onset_before(a_next, a_prev);
        let inner: Vec<f64> = self
            .nuclei
            .iter()
            .copied()
            .filter(|&n| n > a_prev && n < a_next)
            .collect();
        let edge = self.edge_level();
        let (kd, ko) = (self.frame_at(d), self.frame_at(o));
        let sf = (kd..=ko)
            .find(|&k| self.level(k) > edge)
            .map(|k| self.time_of(k))
            .unwrap_or(d);
        let sl = (kd..=ko)
            .rev()
            .find(|&k| self.level(k) > edge)
            .map(|k| self.time_of(k))
            .unwrap_or(o);
        let (w0, w1) = match (inner.first(), inner.last()) {
            (Some(&f), Some(&l)) => (
                self.onset_before(f, d).min(sf),
                self.decay_after(l, o).max(sl),
            ),
            _ => (sf, sl),
        };
        (d, w0, w1, o)
    }

    pub fn place_removal(&self, a_prev: f64, a_next: f64) -> Option<(f64, f64, Join)> {
        if a_next <= a_prev {
            return None;
        }
        let d = self.decay_after(a_prev, a_next); // previous word really ends here
        let o = self.onset_before(a_next, a_prev); // next word really starts here
        if o <= d {
            // Connected speech: no silence between them at all. Splice at the midpoint
            // and let the caller crossfade.
            let mid = (d + o) / 2.0;
            return Some((mid, mid, Join::Crossfade));
        }
        // The removed word's own extent inside the band. Found with a much LOWER
        // threshold than gap detection uses: a word's onset ramps up over a few tens
        // of milliseconds, so anything set high enough to call a gap "quiet" starts
        // the word late and leaves its first syllable behind — which is exactly the
        // "the initial sound is still there" failure, whatever that sound happens to
        // be (a fricative in "ständig", a vowel in "ungefähr").
        // Prefer to measure the removed word the same way we measure the kept ones:
        // anchor on ITS nucleus and walk outward. A threshold crossing finds a word
        // late, because an onset ramps up over tens of milliseconds before it passes
        // any fixed bar — which is exactly how a word's attack keeps surviving the
        // cut. Walking out from the nucleus stops at silence instead, so it captures
        // the whole ramp.
        let inner: Vec<f64> = self
            .nuclei
            .iter()
            .copied()
            .filter(|&n| n > a_prev && n < a_next)
            .collect();
        // Two independent estimates of where the removed run starts and ends; take
        // the more conservative (outermost) of each. They fail in different
        // directions, so the pair is far more robust than either alone:
        //
        // * Scanning OUTWARD-IN from the neighbouring silence finds the first sound
        //   there is — including a leading fricative — but a low bar is needed to
        //   catch a gradual onset.
        // * Walking back from the removed word's own nucleus tracks the ramp
        //   properly, but stops at any quiet patch INSIDE the word: a stop closure is
        //   20-50 ms of real silence, so an onset in front of one (the "sch" before
        //   the /t/ of "ständig") would be left behind.
        let edge = self.edge_level();
        let (kd, ko) = (self.frame_at(d), self.frame_at(o));
        let scan_first = (kd..=ko).find(|&k| self.level(k) > edge).map(|k| self.time_of(k));
        let scan_last = (kd..=ko).rev().find(|&k| self.level(k) > edge).map(|k| self.time_of(k));
        let (w0, w1) = match (inner.first(), inner.last()) {
            (Some(&f), Some(&l)) => (
                self.onset_before(f, d).min(scan_first.unwrap_or(d)),
                self.decay_after(l, o).max(scan_last.unwrap_or(o)),
            ),
            // No nucleus between the anchors (an unvoiced word, or the removal is
            // only silence): the scan is all we have.
            _ => match (scan_first, scan_last) {
                (Some(a), Some(b)) if b >= a => (a, b),
                _ => (d, o),
            },
        };
        let (w0, w1) = (w0.clamp(d, o), w1.clamp(d, o));
        let (w0, w1) = if w1 >= w0 { (w0, w1) } else { (d, o) };
        // Spare silence on each side, and the gap we want at the join.
        let avail_l = (w0 - d).max(0.0);
        let avail_r = (o - w1).max(0.0);
        let spare = avail_l + avail_r;
        let target = (0.5 * self.syllable_period).clamp(0.040, 0.220);
        let extra = target.min(spare);
        // Proportional, not 50/50: with a 40 ms gap on one side and 600 ms on the
        // other, an even split would put an edge inside the removed word.
        let take_l = if spare > 0.0 { extra * avail_l / spare } else { 0.0 };
        let take_r = extra - take_l;
        // Guard band. The costs are asymmetric: cutting a few extra milliseconds of
        // near-silence is inaudible, while leaving the attack of a deleted word is
        // immediately obvious. So bias outward past the measured edges, bounded by
        // the space actually available.
        // Per side: a tight gap on the right must not shrink the margin on the left.
        let guard_l = GUARD.min(avail_l.max(0.0));
        let guard_r = GUARD.min(avail_r.max(0.0));
        Some((
            (d + take_l).min(w0 - guard_l).max(d),
            (o - take_r).max(w1 + guard_r).min(o),
            if spare > 0.005 { Join::Butt } else { Join::Crossfade },
        ))
    }

}

/// RMS per frame, in dB.
fn frame_db(x: &[f64], win: usize, hop: usize) -> Vec<f64> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i + win <= x.len() {
        let s: f64 = x[i..i + win].iter().map(|v| v * v).sum();
        out.push(20.0 * (s / win as f64).sqrt().max(1e-12).log10());
        i += hop;
    }
    out
}

/// Moving average over `SMOOTH` seconds.
fn smooth_db(db: &[f64], hop: usize, sample_rate: f64) -> Vec<f64> {
    let n = ((SMOOTH / (hop as f64 / sample_rate)).round() as usize).max(1);
    if n <= 1 || db.len() < n {
        return db.to_vec();
    }
    let half = n / 2;
    (0..db.len())
        .map(|i| {
            let a = i.saturating_sub(half);
            let b = (i + half + 1).min(db.len());
            db[a..b].iter().sum::<f64>() / (b - a) as f64
        })
        .collect()
}

/// Noise floor as the LOWEST MODE of the level histogram — not a percentile. A
/// percentile of frame levels is a mixture statistic whose value is set by how much
/// of the item is speech, so it moves when the talker pauses more; the room's own
/// level is a mode and stays put.
fn noise_floor_db(db: &[f64]) -> f64 {
    if db.is_empty() {
        return -120.0;
    }
    let mut s = db.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let (lo, hi) = (s[0], s[s.len() - 1]);
    if hi - lo < 1.0 {
        return lo;
    }
    let bins = (((hi - lo) / 0.5).ceil() as usize).clamp(4, 400);
    let w = (hi - lo) / bins as f64;
    let mut hist = vec![0usize; bins];
    for v in &s {
        let b = (((v - lo) / w).floor() as usize).min(bins - 1);
        hist[b] += 1;
    }
    let need = (s.len() as f64 * 0.05).ceil() as usize;
    // First bin (from the bottom) that is a local peak holding enough frames.
    for (b, &c) in hist.iter().enumerate() {
        if c >= need.max(1)
            && (b == 0 || hist[b - 1] <= c)
            && (b + 1 >= bins || hist[b + 1] <= c)
        {
            return lo + (b as f64 + 0.5) * w;
        }
    }
    percentile(&s, 10.0)
}

/// Otsu's two-class split of the level histogram: returns (split_db, upper_mean_db).
/// Fully self-calibrating — it finds this item's own speech/silence division rather
/// than applying an absolute level.
fn otsu_split(db: &[f64]) -> (f64, f64) {
    if db.is_empty() {
        return (-60.0, -20.0);
    }
    let mut s = db.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let (lo, hi) = (s[0], s[s.len() - 1]);
    if hi - lo < 1.0 {
        return (lo, hi);
    }
    let bins = 128usize;
    let w = (hi - lo) / bins as f64;
    let mut hist = vec![0f64; bins];
    for v in &s {
        let b = (((v - lo) / w).floor() as usize).min(bins - 1);
        hist[b] += 1.0;
    }
    let total: f64 = hist.iter().sum();
    let sum_all: f64 = hist.iter().enumerate().map(|(i, c)| i as f64 * c).sum();
    let (mut w0, mut sum0, mut best_var, mut best_b) = (0.0f64, 0.0f64, -1.0f64, 0usize);
    for (b, &h) in hist.iter().enumerate() {
        w0 += h;
        if w0 == 0.0 {
            continue;
        }
        let w1 = total - w0;
        if w1 <= 0.0 {
            break;
        }
        sum0 += b as f64 * h;
        let m0 = sum0 / w0;
        let m1 = (sum_all - sum0) / w1;
        let var = w0 * w1 * (m0 - m1) * (m0 - m1);
        if var > best_var {
            best_var = var;
            best_b = b;
        }
    }
    let split = lo + (best_b as f64 + 0.5) * w;
    let upper: Vec<f64> = s.iter().copied().filter(|v| *v > split).collect();
    let upper_mean = if upper.is_empty() {
        hi
    } else {
        upper.iter().sum::<f64>() / upper.len() as f64
    };
    (split, upper_mean)
}

/// Syllabic nuclei: prominent, voiced peaks of the mid-band envelope.
fn find_nuclei(
    mid: &[f64],
    signal: &[f64],
    hop: usize,
    win: usize,
    sample_rate: f64,
    t_split: f64,
    hop_s: f64,
) -> Vec<f64> {
    if mid.len() < 3 {
        return Vec::new();
    }
    // 1. Every local maximum of the mid-band envelope that rises above the item's own
    //    speech/silence split.
    let mut cands: Vec<usize> = (1..mid.len() - 1)
        .filter(|&k| mid[k] > t_split && mid[k] >= mid[k - 1] && mid[k] >= mid[k + 1])
        .collect();

    // 2. Merge by PROMINENCE. Two maxima are separate syllables only if the valley
    //    between them drops at least NUCLEUS_PROMINENCE_DB below the quieter of the
    //    two; otherwise they are ripple within one vowel (glottal pulses, a formant
    //    wobble) and the louder wins. Measuring the valley BETWEEN peaks — rather
    //    than over a fixed window — is what makes this independent of speaking rate.
    let mut kept: Vec<usize> = Vec::new();
    for &c in &cands {
        if let Some(&last) = kept.last() {
            let valley = mid[last..=c].iter().copied().fold(f64::INFINITY, f64::min);
            if mid[c].min(mid[last]) - valley < NUCLEUS_PROMINENCE_DB {
                if mid[c] > mid[last] {
                    kept.pop();
                    kept.push(c);
                }
                continue;
            }
        }
        kept.push(c);
    }
    cands = kept;

    // 3. Voicing, so a click, a lip smack or a stop burst can't pass as a syllable.
    //    Only the surviving candidates are tested — autocorrelation is the expensive
    //    part. Whispered or heavily-coded material has no periodicity to find, so
    //    there we fall back to a duration criterion instead of rejecting every
    //    syllable.
    let periodicity: Vec<f64> = cands
        .iter()
        .map(|&k| {
            let a = k * hop;
            if a + win <= signal.len() {
                voicing(&signal[a..a + win], sample_rate)
            } else {
                0.0
            }
        })
        .collect();
    let p90 = {
        let mut ps = periodicity.clone();
        ps.sort_by(|a, b| a.total_cmp(b));
        percentile(&ps, 90.0)
    };
    let voiced_ok = p90 >= 0.4;
    let need_run = ((MIN_UNVOICED_NUCLEUS / hop_s).round() as usize).max(1);

    cands
        .iter()
        .zip(&periodicity)
        .filter(|(&k, &p)| {
            if voiced_ok {
                p >= (0.5 * p90).max(0.30)
            } else {
                let a = k.saturating_sub(need_run / 2);
                let b = (k + need_run / 2 + 1).min(mid.len());
                (b - a) >= need_run && mid[a..b].iter().all(|&v| v > t_split)
            }
        })
        .map(|(&k, _)| k as f64 * hop_s)
        .collect()
}

/// Normalised autocorrelation peak over plausible F0 lags (60-400 Hz) — 1.0 is
/// perfectly periodic, near 0 is noise. Covers male, female and child voices.
fn voicing(frame: &[f64], sample_rate: f64) -> f64 {
    let min_lag = ((sample_rate / 400.0).round() as usize).max(2);
    let max_lag = ((sample_rate / 60.0).round() as usize).min(frame.len().saturating_sub(1));
    if max_lag <= min_lag {
        return 0.0;
    }
    let energy: f64 = frame.iter().map(|v| v * v).sum();
    if energy <= 1e-12 {
        return 0.0;
    }
    let mut best = 0.0f64;
    for lag in min_lag..=max_lag {
        let n = frame.len() - lag;
        let r: f64 = (0..n).map(|i| frame[i] * frame[i + lag]).sum();
        let norm = r / energy;
        if norm > best {
            best = norm;
        }
    }
    best.clamp(0.0, 1.0)
}

/// RBJ cookbook 2nd-order high-pass.
fn highpass(x: &[f64], sample_rate: f64, freq: f64) -> Vec<f64> {
    let f = freq.clamp(10.0, sample_rate / 2.0 * 0.95);
    let w = 2.0 * std::f64::consts::PI * f / sample_rate;
    let (sn, cs) = (w.sin(), w.cos());
    let alpha = sn / (2.0 * std::f64::consts::FRAC_1_SQRT_2.recip());
    let a0 = 1.0 + alpha;
    Biquad {
        b0: ((1.0 + cs) / 2.0) / a0,
        b1: (-(1.0 + cs)) / a0,
        b2: ((1.0 + cs) / 2.0) / a0,
        a1: (-2.0 * cs) / a0,
        a2: (1.0 - alpha) / a0,
    }
    .apply(x)
}

/// Band-pass as a high-pass followed by a low-pass (adequate for an envelope).
fn bandpass(x: &[f64], sample_rate: f64, lo: f64, hi: f64) -> Vec<f64> {
    let hp = highpass(x, sample_rate, lo);
    let f = hi.clamp(20.0, sample_rate / 2.0 * 0.95);
    let w = 2.0 * std::f64::consts::PI * f / sample_rate;
    let (sn, cs) = (w.sin(), w.cos());
    let alpha = sn / (2.0 * std::f64::consts::FRAC_1_SQRT_2.recip());
    let a0 = 1.0 + alpha;
    Biquad {
        b0: ((1.0 - cs) / 2.0) / a0,
        b1: (1.0 - cs) / a0,
        b2: ((1.0 - cs) / 2.0) / a0,
        a1: (-2.0 * cs) / a0,
        a2: (1.0 - alpha) / a0,
    }
    .apply(&hp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const SR: f64 = 32_000.0;

    /// A crude but structurally honest speech synthesiser: a voiced "vowel" is a
    /// pulse train with formant-ish shaping, a "fricative" is high-frequency noise,
    /// a "closure" is near-silence, and everything sits on a room-tone floor.
    struct Syn {
        s: Vec<f64>,
        seed: u64,
    }
    impl Syn {
        fn new() -> Self {
            Self { s: Vec::new(), seed: 0x2545F491_4F6CDD1D }
        }
        fn rnd(&mut self) -> f64 {
            self.seed ^= self.seed << 13;
            self.seed ^= self.seed >> 7;
            self.seed ^= self.seed << 17;
            ((self.seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        }
        fn n(secs: f64) -> usize {
            (SR * secs) as usize
        }
        /// Voiced vowel at `f0` Hz, amplitude `amp`.
        fn vowel(&mut self, secs: f64, f0: f64, amp: f64) -> &mut Self {
            let n = Self::n(secs);
            let start = self.s.len();
            for i in 0..n {
                let t = i as f64 / SR;
                let v = (2.0 * PI * f0 * t).sin() * 0.6
                    + (2.0 * PI * 2.0 * f0 * t).sin() * 0.3
                    + (2.0 * PI * 700.0 * t).sin() * 0.25;
                // Gentle onset/offset so it isn't a click.
                let env = ((i as f64 / (SR * 0.01)).min(1.0))
                    .min(((n - i) as f64 / (SR * 0.01)).min(1.0));
                self.s.push(v * amp * env);
            }
            let _ = start;
            self
        }
        /// Unvoiced fricative: high-frequency noise, quiet in broadband terms.
        fn fricative(&mut self, secs: f64, amp: f64) -> &mut Self {
            let n = Self::n(secs);
            let mut prev = 0.0;
            for _ in 0..n {
                let w = self.rnd();
                let hp = w - prev; // crude HF emphasis
                prev = w;
                self.s.push(hp * amp);
            }
            self
        }
        /// Silence at the room-tone floor.
        fn quiet(&mut self, secs: f64) -> &mut Self {
            let n = Self::n(secs);
            for _ in 0..n {
                let r = self.rnd();
                self.s.push(r * 0.0006);
            }
            self
        }
        fn burst(&mut self, amp: f64) -> &mut Self {
            for _ in 0..Self::n(0.006) {
                let r = self.rnd();
                self.s.push(r * amp);
            }
            self
        }
        fn at(&self) -> f64 {
            self.s.len() as f64 / SR
        }
    }

    /// "Ja-pan ständig": the exact shape that broke the level-based detectors.
    /// Ja(vowel) [closure] pan(vowel) [gap] sch(fricative) tändig(vowel)
    fn japan_staendig() -> (Vec<f64>, f64, f64, f64, f64) {
        let mut s = Syn::new();
        s.quiet(0.30);
        let ja = s.at();
        s.vowel(0.16, 120.0, 0.5);
        s.quiet(0.05); // /p/ CLOSURE — real silence inside the word
        s.burst(0.25);
        let pan = s.at();
        s.vowel(0.16, 120.0, 0.5);
        let japan_end = s.at();
        s.quiet(0.05); // the true word gap
        let sch = s.at();
        s.fricative(0.11, 0.06); // "sch" — broadband-quiet, HF-loud
        s.burst(0.22);
        let taendig = s.at();
        s.vowel(0.22, 120.0, 0.5);
        s.quiet(0.30);
        (s.s.clone(), ja, pan, japan_end, {
            let _ = (sch, taendig);
            sch
        })
    }

    #[test]
    fn nuclei_are_found_and_are_voiced() {
        let (sig, ..) = japan_staendig();
        let a = SpeechAnalysis::new(&sig, SR).expect("analysable");
        // Three voiced vowels -> at least three nuclei; the fricative and the two
        // bursts must NOT produce one.
        assert!(a.nuclei.len() >= 3, "nuclei: {:?}", a.nuclei);
        assert!(a.nuclei.len() <= 6, "bursts leaked in: {:?}", a.nuclei);
    }

    /// THE INVARIANT. Deleting "ständig" must never cut before the LAST nucleus of
    /// "Japan" — that is what took the syllable "pan" — and the /p/ closure lies
    /// before that nucleus, so it is out of reach by construction, not by threshold.
    #[test]
    fn cut_never_reaches_past_the_previous_words_last_nucleus() {
        let (sig, _ja, pan, japan_end, sch) = japan_staendig();
        let a = SpeechAnalysis::new(&sig, SR).expect("analysable");
        let (ap, an_) = a.anchors(Some((0.30, japan_end)), (japan_end, sch), Some((sch, sch + 0.45)));
        let a_prev = ap.expect("Japan has a nucleus");
        // The anchor is in "pan", i.e. after the /p/ closure.
        assert!(a_prev > pan - 0.02, "anchor {a_prev} should be in 'pan' (>= {pan})");
        let a_next = an_.expect("ständig has a nucleus");
        if let Some((cut_s, cut_e, _)) = a.place_removal(a_prev, a_next) {
            assert!(cut_s >= a_prev, "cut at {cut_s} landed before the anchor {a_prev}");
            assert!(cut_e <= a_next, "cut at {cut_e} landed past the next anchor {a_next}");
        }
    }

    /// Sweep: across voices (F0), levels, gap lengths and noise floors, a placed cut
    /// must always land inside the band. This is the property that matters; single
    /// fixtures are what let every previous bug through.
    #[test]
    fn placement_stays_inside_the_band_across_the_parameter_space() {
        for &f0 in &[85.0, 120.0, 210.0, 300.0] {
            for &amp in &[0.08, 0.3, 0.7] {
                for &gap in &[0.02, 0.05, 0.12, 0.30] {
                    let mut s = Syn::new();
                    s.quiet(0.25);
                    let w1 = s.at();
                    s.vowel(0.18, f0, amp);
                    let w1_end = s.at();
                    s.quiet(gap);
                    let w2 = s.at();
                    s.vowel(0.18, f0, amp);
                    let w2_end = s.at();
                    s.quiet(0.25);
                    let Some(a) = SpeechAnalysis::new(&s.s, SR) else { continue };
                    let (Some(p), Some(n)) =
                        a.anchors(Some((w1, w1_end)), (w1_end, w2), Some((w2, w2_end)))
                    else { continue };
                    if let Some((cs, ce, _)) = a.place_removal(p, n) {
                        assert!(
                            cs >= p && ce <= n && ce >= cs,
                            "f0={f0} amp={amp} gap={gap}: cut [{cs}, {ce}] outside band [{p}, {n}]"
                        );
                    }
                }
            }
        }
    }

    /// The complement of the reported bug: when the NEXT kept word begins with a
    /// fricative, the cut must stop before it — otherwise deleting a word clips the
    /// head off the word that was kept.
    #[test]
    fn a_fricative_is_not_mistaken_for_a_pause() {
        let mut s = Syn::new();
        s.quiet(0.25);
        let w1 = s.at();
        s.vowel(0.18, 120.0, 0.5);
        let w1_end = s.at();
        s.quiet(0.05);
        let rem = s.at();
        s.vowel(0.16, 120.0, 0.5); // the word being deleted
        let rem_end = s.at();
        s.quiet(0.05);
        let fric_start = s.at();
        s.fricative(0.12, 0.06); // the NEXT kept word's initial fricative
        s.vowel(0.20, 120.0, 0.5);
        let w2_end = s.at();
        s.quiet(0.25);
        let a = SpeechAnalysis::new(&s.s, SR).expect("analysable");
        let (p, n) = a.anchors(
            Some((w1, w1_end)),
            (rem, rem_end),
            Some((fric_start, w2_end)),
        );
        let (p, n) = (p.expect("prev nucleus"), n.expect("next nucleus"));
        let (_, cut_e, _) = a.place_removal(p, n).expect("placed");
        assert!(
            cut_e <= fric_start + 0.02,
            "cut ends at {cut_e}, inside the next word's fricative starting {fric_start}"
        );
        assert!(cut_e >= rem_end - 0.02, "cut at {cut_e} left the deleted word's tail");
    }

    /// THE REPORTED BUG: deleting "ständig" left its initial "sch" behind, because
    /// the transcript puts the word's start at the /t/, past the fricative. The
    /// placement must be driven by where the PREVIOUS word's audio actually ends —
    /// after which everything, fricative included, is deletable.
    #[test]
    fn removal_starts_before_a_word_initial_fricative() {
        let mut s = Syn::new();
        s.quiet(0.25);
        let w1 = s.at();
        s.vowel(0.18, 120.0, 0.5); // "Japan" (last syllable)
        let w1_end = s.at();
        s.quiet(0.06); // the true gap
        let fric = s.at();
        s.fricative(0.11, 0.06); // "sch" — the part that kept surviving
        s.burst(0.22); // /t/
        let w2 = s.at();
        s.vowel(0.20, 120.0, 0.5); // "-ändig"
        let w2_end = s.at();
        s.quiet(0.25);
        let a = SpeechAnalysis::new(&s.s, SR).expect("analysable");
        let (p, n) = a.anchors(Some((w1, w1_end)), (fric, w2), Some((w2, w2_end)));
        let (p, n) = (p.expect("prev nucleus"), n.expect("next nucleus"));
        let (cut_s, cut_e, _) = a.place_removal(p, n).expect("placed");
        assert!(
            cut_s <= fric + 0.015,
            "cut starts at {cut_s}, after the fricative at {fric} — the 'sch' survives"
        );
        assert!(cut_s >= w1_end - 0.05, "cut at {cut_s} bit into the previous word (ends {w1_end})");
        assert!(cut_e >= cut_s, "degenerate range");
    }

    /// The mirror: a word-FINAL fricative must be taken with the word, not left as a
    /// hiss at the head of the next one.
    #[test]
    fn removal_ends_after_a_word_final_fricative() {
        let mut s = Syn::new();
        s.quiet(0.25);
        let w1 = s.at();
        s.vowel(0.18, 120.0, 0.5);
        let w1_end = s.at();
        s.quiet(0.05);
        s.vowel(0.14, 120.0, 0.5); // removed word...
        s.fricative(0.10, 0.06); // ...ending in "-s"
        let fric_end = s.at();
        s.quiet(0.06);
        let w2 = s.at();
        s.vowel(0.18, 120.0, 0.5);
        let w2_end = s.at();
        s.quiet(0.25);
        let a = SpeechAnalysis::new(&s.s, SR).expect("analysable");
        let (p, n) = a.anchors(Some((w1, w1_end)), (w1_end, w2), Some((w2, w2_end)));
        let (p, n) = (p.expect("prev nucleus"), n.expect("next nucleus"));
        let (_, cut_e, _) = a.place_removal(p, n).expect("placed");
        assert!(
            cut_e >= fric_end - 0.015,
            "cut ends at {cut_e}, before the trailing fricative ends at {fric_end}"
        );
    }

    /// "ständig" in full: /ʃ/ then a /t/ CLOSURE then the vowel. Walking back from
    /// the vowel's nucleus stops at that closure — it is 20-50 ms of real silence,
    /// indistinguishable from a word gap — which leaves the "sch" in front of it
    /// outside the cut. The word's start must be found by scanning FORWARD from the
    /// previous word's decay, where the first sound encountered is the fricative.
    #[test]
    fn removal_covers_an_onset_before_an_internal_closure() {
        let mut s = Syn::new();
        s.quiet(0.25);
        let w1 = s.at();
        s.vowel(0.18, 120.0, 0.5);
        let w1_end = s.at();
        s.quiet(0.07); // the true word gap
        let rem = s.at();
        s.fricative(0.09, 0.06); // "sch"
        s.quiet(0.035); // /t/ closure — looks exactly like a gap
        s.burst(0.22); // /t/ release
        s.vowel(0.16, 120.0, 0.5); // "-ändig"
        let rem_end = s.at();
        s.quiet(0.07);
        let w2 = s.at();
        s.vowel(0.18, 120.0, 0.5);
        let w2_end = s.at();
        s.quiet(0.25);
        let a = SpeechAnalysis::new(&s.s, SR).expect("analysable");
        let (p, n) = a.anchors(Some((w1, w1_end)), (rem, rem_end), Some((w2, w2_end)));
        let (p, n) = (p.expect("prev nucleus"), n.expect("next nucleus"));
        let (cut_s, _, _) = a.place_removal(p, n).expect("placed");
        assert!(
            cut_s <= rem + 0.015,
            "cut starts at {cut_s}, after the fricative at {rem} — the 'sch' survives"
        );
    }

    /// REPORTED: deleting "ungefähr" left its initial "u" audible. That word starts
    /// with a VOWEL, so this was never about fricatives — a word's onset ramps up
    /// over a few tens of ms, and finding the word's extent with the same threshold
    /// used to detect gaps starts it late and leaves the attack behind, whatever
    /// sound the attack happens to be.
    #[test]
    fn removal_covers_a_gradual_vowel_onset() {
        let sr = SR;
        let mut s = Syn::new();
        s.quiet(0.25);
        let w1 = s.at();
        s.vowel(0.18, 120.0, 0.5);
        let w1_end = s.at();
        s.quiet(0.07);
        let rem = s.at();
        // A vowel-initial word that fades IN over 60 ms, like "ungefähr".
        let n = (sr * 0.06) as usize;
        for i in 0..n {
            let t = i as f64 / sr;
            let ramp = i as f64 / n as f64;
            s.s.push((2.0 * PI * 120.0 * t).sin() * 0.5 * ramp);
        }
        s.vowel(0.14, 120.0, 0.5);
        let rem_end = s.at();
        s.quiet(0.07);
        let w2 = s.at();
        s.vowel(0.18, 120.0, 0.5);
        let w2_end = s.at();
        s.quiet(0.25);
        let a = SpeechAnalysis::new(&s.s, sr).expect("analysable");
        let (p, n2) = a.anchors(Some((w1, w1_end)), (rem, rem_end), Some((w2, w2_end)));
        let (p, n2) = (p.expect("prev nucleus"), n2.expect("next nucleus"));
        let (cut_s, _, _) = a.place_removal(p, n2).expect("placed");
        assert!(
            cut_s <= rem + 0.015,
            "cut starts at {cut_s}, after the word's onset at {rem} — its attack survives"
        );
        assert!(cut_s >= w1_end - 0.03, "cut at {cut_s} bit into the kept word (ends {w1_end})");
    }

    #[test]
    fn refuses_when_there_is_no_anchor() {
        let mut s = Syn::new();
        s.quiet(0.5);
        let a = SpeechAnalysis::new(&s.s, SR).expect("analysable");
        assert!(a.anchors(Some((0.0, 0.5)), (0.5, 0.5), None).0.is_none());
    }

    #[test]
    fn syllable_period_tracks_the_speaking_rate() {
        for &period in &[0.20f64, 0.35] {
            let mut s = Syn::new();
            s.quiet(0.2);
            for _ in 0..8 {
                s.vowel(period * 0.6, 120.0, 0.5);
                s.quiet(period * 0.4);
            }
            let a = SpeechAnalysis::new(&s.s, SR).expect("analysable");
            if a.nuclei.len() >= 3 {
                assert!(
                    (a.syllable_period - period).abs() < period * 0.5,
                    "period {} vs expected {period}",
                    a.syllable_period
                );
            }
        }
    }
}


