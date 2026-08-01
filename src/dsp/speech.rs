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
/// Analysis window. Needs ≥4 pitch periods for a stable level estimate; 32 ms covers
/// a 125 Hz male voice. (A lower voice gets a slightly ripplier envelope, which the
/// smoothing below absorbs.)
const WIN: f64 = 0.032;
/// Envelope smoothing, from auditory temporal integration — the criterion is whether
/// a splice is *audible*, not the raw RMS. Also bridges vocal-fry inter-pulse gaps.
const SMOOTH: f64 = 0.030;
/// A nucleus must stand this far above its flanking dips (de Jong & Wempe's
/// validated syllable-nuclei method).
const NUCLEUS_PROMINENCE_DB: f64 = 2.0;
/// Fallback nucleus criterion for whispered/heavily-coded material, where voicing
/// can't be required: a run above the speech/silence split at least this long. Below
/// the shortest vowel nucleus in fast speech (~60-80 ms), so it still rejects clicks.
const MIN_UNVOICED_NUCLEUS: f64 = 0.040;
/// A quiet run terminated by a burst within this long is a stop release, not a pause.
const BURST_WINDOW: f64 = 0.040;

/// How a boundary should be joined once placed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Join {
    /// A real pause was found; the fragments can simply abut.
    Butt,
    /// No silence exists here (connected speech). Splice anyway, but crossfade.
    Crossfade,
}

/// Result of placing one cut edge.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Splice {
    /// Move the edge to this time (seconds from the analysed slice's start).
    At(f64, Join),
    /// The material gives no defensible answer — keep the caller's original time.
    /// Reported, never silently swallowed.
    Refused(Refusal),
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
    vb: Vec<f64>,
    hf: Vec<f64>,
    /// Positive frame-to-frame rise, a cheap transient measure (burst detection).
    flux: Vec<f64>,
    flux_p90: f64,
    floor_bb: f64,
    floor_vb: f64,
    floor_hf: f64,
    /// Otsu split between the silence and speech classes, in dB.
    t_split: f64,
    contrast: f64,
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
        let vb_sig = bandpass(&base, sample_rate, 60.0, 350.0);
        let hf_sig = highpass(&base, sample_rate, 3000.0_f64.min(nyq * 0.8));

        let bb = smooth_db(&frame_db(&base, win, hop), hop, sample_rate);
        let vb = smooth_db(&frame_db(&vb_sig, win, hop), hop, sample_rate);
        let hf = smooth_db(&frame_db(&hf_sig, win, hop), hop, sample_rate);
        // MID (300-3000 Hz) is where sonorant nuclei live; used only for nuclei.
        let mid = smooth_db(
            &frame_db(&bandpass(&base, sample_rate, 300.0, 3000.0_f64.min(nyq * 0.9)), win, hop),
            hop,
            sample_rate,
        );
        if bb.len() < 3 {
            return None;
        }

        let flux: Vec<f64> = std::iter::once(0.0)
            .chain(bb.windows(2).map(|w| (w[1] - w[0]).max(0.0)))
            .collect();
        let flux_p90 = {
            let mut s = flux.clone();
            s.sort_by(|a, b| a.total_cmp(b));
            percentile(&s, 90.0).max(1e-6)
        };

        let floor_bb = noise_floor_db(&bb);
        let floor_vb = noise_floor_db(&vb);
        let floor_hf = noise_floor_db(&hf);
        let (t_split, speech_level) = otsu_split(&bb);
        let contrast = (speech_level - floor_bb).max(0.0);

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
            vb,
            hf,
            flux,
            flux_p90,
            floor_bb,
            floor_vb,
            floor_hf,
            t_split,
            contrast,
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

    /// The LAST nucleus belonging to the word spanning `[start, end]`, by nearest
    /// centre — robust to a span that is shifted by more than its own length, which
    /// "the loudest frame inside the span" is not.
    pub fn last_nucleus_of(&self, start: f64, end: f64) -> Option<f64> {
        self.nuclei_of(start, end).last().copied()
    }
    /// The FIRST nucleus belonging to that word.
    pub fn first_nucleus_of(&self, start: f64, end: f64) -> Option<f64> {
        self.nuclei_of(start, end).first().copied()
    }

    fn nuclei_of(&self, start: f64, end: f64) -> Vec<f64> {
        let centre = (start + end) / 2.0;
        let half = ((end - start).abs() / 2.0).max(self.syllable_period);
        self.nuclei
            .iter()
            .copied()
            .filter(|n| (n - centre).abs() <= half)
            .collect()
    }

    /// Count nuclei strictly inside a band — the caller uses this to check the word
    /// list is locally consistent with the audio before trusting a placement.
    #[allow(dead_code)] // consistency gate: for the word-list sanity check
    pub fn nuclei_between(&self, a: f64, b: f64) -> usize {
        self.nuclei.iter().filter(|&&n| n > a && n < b).count()
    }

    /// Place one cut edge inside the band `[a_prev, a_next]` (the two anchoring
    /// nuclei), preferring a time near `want`.
    ///
    /// The cost is continuous — there is no threshold that decides "pause" vs "not
    /// pause", precisely because no such threshold generalises. A frame is cheap only
    /// when EVERY band is quiet (`max` over the three, never a broadband sum): that
    /// is what stops a weak fricative or a devoiced mora, whose broadband level
    /// collapses but whose HF does not, from being mistaken for silence.
    pub fn splice(&self, a_prev: f64, a_next: f64, want: f64) -> Splice {
        // A band far longer than the speaker's own rhythm means the nuclei were
        // mis-assigned; guessing inside it would be worse than declining.
        let max_band = (1.5 * self.syllable_period).clamp(0.150, 0.400) * 2.0;
        if a_next <= a_prev || (a_next - a_prev) > max_band.max(0.400) {
            return Splice::Refused(Refusal::BandTooWide);
        }
        let (lo, hi) = (self.frame_at(a_prev), self.frame_at(a_next));
        if hi <= lo + 1 {
            return Splice::Refused(Refusal::NoMinimum);
        }
        let sigma = (0.5 * self.syllable_period).max(0.020);
        let mut best = lo;
        let mut best_cost = f64::INFINITY;
        let mut costs = Vec::with_capacity(hi - lo + 1);
        for k in lo..=hi {
            let level = (self.bb[k] - self.floor_bb)
                .max(self.vb[k] - self.floor_vb)
                .max(self.hf[k] - self.floor_hf)
                .max(0.0);
            let dt = (self.time_of(k) - want) / sigma;
            let c = level
                + 6.0 * (self.flux[k] / self.flux_p90).clamp(0.0, 2.0)
                + 24.0 * f64::from(u8::from(self.is_closure(k)))
                + 0.5 * dt * dt;
            costs.push(level); // acceptance is judged on LEVEL, not the priors
            if c < best_cost {
                best_cost = c;
                best = k;
            }
        }
        let level_at_best = costs[best - lo];
        let t = self.time_of(best);
        // A real pause: quiet relative to this item's own speech/silence contrast.
        if level_at_best <= 0.25 * self.contrast {
            return Splice::At(t, Join::Butt);
        }
        // No silence anywhere (Spanish resyllabification, French liaison, fast
        // Japanese — the common case in syllable-timed languages). Splice anyway if
        // there is a genuine local minimum, but crossfade it.
        let mut sorted = costs.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let median = percentile(&sorted, 50.0);
        if median - level_at_best >= 3.0 {
            return Splice::At(t, Join::Crossfade);
        }
        Splice::Refused(Refusal::NoMinimum)
    }

    /// Is this frame part of a stop closure rather than a pause? Two universal cues:
    /// a release burst just after it, or a voice bar (LF energy with no HF).
    fn is_closure(&self, k: usize) -> bool {
        if self.bb[k] > self.t_split {
            return false;
        }
        let ahead = ((BURST_WINDOW / self.hop).round() as usize).max(1);
        let burst = ((k + 1)..=(k + ahead).min(self.flux.len() - 1))
            .any(|j| self.flux[j] > self.flux_p90);
        let voice_bar =
            self.vb[k] > self.floor_vb + 10.0 && self.hf[k] <= self.floor_hf + 3.0;
        burst || voice_bar
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
        let a_prev = a
            .last_nucleus_of(0.30, japan_end)
            .expect("Japan has a nucleus");
        // The anchor is in "pan", i.e. after the /p/ closure.
        assert!(a_prev > pan - 0.02, "anchor {a_prev} should be in 'pan' (>= {pan})");
        let a_next = a
            .first_nucleus_of(sch, sch + 0.40)
            .expect("ständig has a nucleus");
        match a.splice(a_prev, a_next, japan_end) {
            Splice::At(t, _) => {
                assert!(t >= a_prev, "cut at {t} landed before the anchor {a_prev}");
                assert!(t <= a_next, "cut at {t} landed past the next anchor {a_next}");
            }
            Splice::Refused(_) => { /* declining is always safe */ }
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
                    let (Some(p), Some(n)) = (
                        a.last_nucleus_of(w1, w1_end),
                        a.first_nucleus_of(w2, w2_end),
                    ) else { continue };
                    if let Splice::At(t, _) = a.splice(p, n, w1_end) {
                        assert!(
                            t >= p && t <= n,
                            "f0={f0} amp={amp} gap={gap}: cut {t} outside band [{p}, {n}]"
                        );
                    }
                }
            }
        }
    }

    /// A fricative must not read as silence: with "sch" between the anchors, the
    /// chosen splice must not sit inside it (that is what left the "sch" behind).
    #[test]
    fn a_fricative_is_not_mistaken_for_a_pause() {
        let mut s = Syn::new();
        s.quiet(0.25);
        let w1 = s.at();
        s.vowel(0.18, 120.0, 0.5);
        let w1_end = s.at();
        s.quiet(0.05);
        let fric_start = s.at();
        s.fricative(0.12, 0.06);
        let fric_end = s.at();
        let w2 = s.at();
        s.vowel(0.20, 120.0, 0.5);
        let w2_end = s.at();
        s.quiet(0.25);
        let a = SpeechAnalysis::new(&s.s, SR).expect("analysable");
        let (p, n) = (
            a.last_nucleus_of(w1, w1_end).expect("w1 nucleus"),
            a.first_nucleus_of(w2, w2_end).expect("w2 nucleus"),
        );
        if let Splice::At(t, _) = a.splice(p, n, fric_start) {
            assert!(
                !(t > fric_start + 0.02 && t < fric_end - 0.02),
                "cut at {t} landed inside the fricative [{fric_start}, {fric_end}]"
            );
        }
    }

    #[test]
    fn refuses_when_there_is_no_anchor() {
        let mut s = Syn::new();
        s.quiet(0.5);
        let a = SpeechAnalysis::new(&s.s, SR).expect("analysable");
        assert!(a.last_nucleus_of(0.0, 0.5).is_none());
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
