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
/// Smoothing applied to the NUCLEUS band only: bridges glottal-pulse ripple so one
/// vowel yields one syllable. Never applied to the level bands — see WIN.
const SMOOTH: f64 = 0.030;
/// A nucleus must stand this far above its flanking dips (de Jong & Wempe's
/// validated syllable-nuclei method).
const NUCLEUS_PROMINENCE_DB: f64 = 2.0;
/// Two envelope peaks closer than this are treated as one syllable when the valley
/// between them is shallow. Below the shortest vowel nucleus in fast speech, so
/// genuine adjacent syllables are never merged.
const MIN_SYLLABLE_SEP: f64 = 0.080;
/// How far outside the transcript's span a nucleus may still belong to the removed
/// word.
///
/// 40 ms, checked against a real case rather than picked: a measured cut counted a
/// nucleus at 24.423 for a word ending at 24.368 — 55 ms past — and cut into the
/// following word. At 120 ms that nucleus passed; at 40 ms it does not. Erring small
/// only shrinks the removed extent, which under-cuts and is recoverable by ear, while
/// erring large cuts into a word the user kept.
const HINT_TOL: f64 = 0.040;
/// How far past a NEIGHBOUR's reported boundary a nucleus may still be counted as the
/// removed word's. Tight on purpose: erring small only shrinks the removed extent,
/// which under-cuts (recoverable), while erring large over-cuts into a kept word.
const NEIGHBOUR_TOL: f64 = 0.030;
/// Fallback nucleus criterion for whispered/heavily-coded material, where voicing
/// can't be required: a run above the speech/silence split at least this long. Below
/// the shortest vowel nucleus in fast speech (~60-80 ms), so it still rejects clicks.
const MIN_UNVOICED_NUCLEUS: f64 = 0.040;
/// A band with less range than this says nothing useful and is excluded.
const MIN_BAND_RANGE: f64 = 6.0;
/// A search stretch with less dynamic range than this has no measurable boundary —
/// a music bed, a long reverb tail, crosstalk. Refuse rather than guess. Kept low:
/// at 6 dB most short unstressed words refused, and a refusal falls back to bare
/// transcript times, which is strictly less information than a shallow measurement.
const MIN_STRETCH_RANGE: f64 = 3.5;
/// Floor on the "equally quiet" tolerance, so it can never be finer than the
/// measurement's own jitter.
const MIN_TOL_DB: f64 = 1.0;
/// Ceiling on the same tolerance, so a noisy estimate can never widen "equally
/// quiet" far enough to swallow a word's decay.
const MAX_TOL_DB: f64 = 6.0;

/// Where a removal's two edges go, and how they should be joined.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Placement {
    /// `None` when THIS edge could not be measured — the caller keeps the transcript
    /// time for it. Per edge, because the two sides are independent: a nucleus close
    /// to the removed word leaves no measurable stretch on that side while the other
    /// side is perfectly good, and refusing both threw away a real measurement.
    pub start: Option<f64>,
    pub end: Option<f64>,
    /// Equal-power fade length for the join, in seconds.
    pub fade: f64,
    /// The pause left at the join — the single number that exposes a placement which
    /// swallowed the silence, so it is reported.
    pub gap: f64,
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

/// Why ONE edge of a word could not be measured. Per-edge and specific, because
/// "unmeasured" lumps together a word the detector never saw, a rounding rejection,
/// and a boundary that genuinely is not in the audio — and those need opposite fixes.
/// Without this the report showed a fallback as `+0.000` drift, typographically
/// identical to a perfect measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeCause {
    /// Found in the audio.
    Measured,
    /// The anchors were degenerate (out-of-order or zero-length transcript span).
    NoAnchor,
    /// No band has enough range to measure anything — a whole-clip property.
    BandsUnusable,
    /// The word has no syllabic nucleus of its own inside the search band, so the
    /// search fell back to the transcript hint. The detector never saw this word.
    NoNucleusInBand,
    /// A nucleus was found but no quiet run flanks it on this side.
    NoQuietRun,
    /// A run was found but did not contain the word's own nucleus.
    ContainmentReject,
    /// The two edges crossed, and the word has no nucleus of its own.
    CrossedNoNucleus,
    /// The two edges crossed around a SINGLE nucleus — the left and right search
    /// windows meet at that one frame, so both runs can touch it.
    CrossedOneNucleus,
    /// The two edges crossed despite the word spanning several nuclei, which the
    /// window geometry should make impossible.
    CrossedMulti,
    /// The edge landed inside a neighbouring word.
    BleedReject,
}

impl EdgeCause {
    /// Short token for the diagnostic report.
    pub fn token(self) -> &'static str {
        match self {
            EdgeCause::Measured => "ok",
            EdgeCause::NoAnchor => "no-anchor",
            EdgeCause::BandsUnusable => "bands",
            EdgeCause::NoNucleusInBand => "no-nucleus",
            EdgeCause::NoQuietRun => "no-run",
            EdgeCause::ContainmentReject => "contain",
            EdgeCause::CrossedNoNucleus => "crossed-nonuc",
            EdgeCause::CrossedOneNucleus => "crossed-1nuc",
            EdgeCause::CrossedMulti => "crossed-multi",
            EdgeCause::BleedReject => "bleed",
        }
    }
}

/// One measured edge, carrying WHY when there is no time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Edge {
    pub time: Option<f64>,
    pub cause: EdgeCause,
    /// LENGTH IN FRAMES of the quiet run this edge came from, or -1 when no run was
    /// found. The measurement four fixes at the "ihren Schoss" junction were aimed at
    /// and none of them ever took: the report's `gap` column is
    /// next.measured_start - this.measured_end, which is NOT a run length, so whether
    /// a marginal-run gate would even fire there has never actually been observed.
    pub run: f64,
    /// For a crossing, BY HOW MUCH. The single number that says whether this is
    /// frame quantisation (<= one hop) or a real inversion.
    pub detail: f64,
}

impl Edge {
    fn ok(t: f64, run: f64) -> Self {
        Edge { time: Some(t), cause: EdgeCause::Measured, detail: 0.0, run }
    }
    fn fail(cause: EdgeCause) -> Self {
        Edge { time: None, cause, detail: 0.0, run: -1.0 }
    }
    /// Drop a measured time that failed a caller-side check, keeping the reason and
    /// the run it came from — a rejected edge still says how much silence was there.
    pub fn reject(self, cause: EdgeCause) -> Self {
        Edge { time: None, cause, detail: 0.0, run: self.run }
    }
}

/// Per-item speech analysis: band envelopes, self-calibrated levels, and the
/// syllabic nuclei that anchor every boundary decision.
#[derive(Clone)]
pub struct SpeechAnalysis {
    hop: f64,
    /// Smoothed per-frame levels in dB, broadband / voice-bar / high-frequency.
    bb: Vec<f64>,
    hf: Vec<f64>,
    /// Voice bar, 300-3000 Hz, unsmoothed.
    vb: Vec<f64>,
    floor_bb: f64,
    floor_hf: f64,
    floor_vb: f64,
    /// Each band's own speech-to-floor range, so levels can be compared between
    /// bands with very different noise floors.
    contrast_bb: f64,
    contrast_hf: f64,
    contrast_vb: f64,
    /// Nucleus times, ascending.
    pub nuclei: Vec<f64>,
    /// Median interval between nuclei — the speaker's own syllable period. Every
    /// duration downstream is expressed in units of this, never in milliseconds,
    /// because syllable rate varies by ~2x across languages and as much again
    /// between talkers.
    #[allow(dead_code)] // reported by tests; kept as the material's own rhythm
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
        // The VOICE BAR (300-3000 Hz) is where sonorant nuclei live. It is a level
        // band like the other two: quiet_runs has to be able to see the band nuclei
        // are DETECTED in, or a word can be a nucleus and a silence at the same time.
        // It was, for 178 edges on a 271 s take — an unstressed function word is a
        // vowel, so its energy sits here, it is weak broadband and has nothing above
        // 3 kHz, while the content words around it have fricatives that do. Measured
        // on the only two bands level() could see, the whole word read as silence,
        // one continuous quiet run spanning it, split in two by the search-window
        // boundary at its own nucleus — which is why both edges landed on the same
        // frame, to the sample.
        let vb = frame_db(&bandpass(&base, sample_rate, 300.0, 3000.0_f64.min(nyq * 0.9)), win, hop);
        // Nuclei use a SMOOTHED copy, where pitch ripple would otherwise split one
        // vowel into several syllables. The smoothing must not reach level(), where
        // it would span a short inter-word gap and hide it.
        let mid = smooth_db(&vb, hop, sample_rate);
        if bb.len() < 3 {
            return None;
        }


        let floor_bb = noise_floor_db(&bb);
        let floor_hf = noise_floor_db(&hf);
        let floor_vb = noise_floor_db(&vb);
        let (t_split, speech_level) = otsu_split(&bb);
        let contrast_bb = (speech_level - floor_bb).max(0.0);
        let contrast_hf = (otsu_split(&hf).1 - floor_hf).max(0.0);
        let contrast_vb = (otsu_split(&vb).1 - floor_vb).max(0.0);

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
            vb,
            floor_bb,
            floor_hf,
            floor_vb,
            contrast_bb,
            contrast_hf,
            contrast_vb,
            nuclei,
            syllable_period,
        })
    }

    /// Analysis frame period. Callers need it to reason about quantisation: a time
    /// derived from a frame index is only ever accurate to half of this.
    pub fn hop(&self) -> f64 {
        self.hop
    }
    /// The analysed span, in item time. Used as a substitute anchor at the clip
    /// edges, where there is no neighbouring word to anchor to.
    pub fn span(&self) -> (f64, f64) {
        (0.0, self.time_of(self.bb.len().saturating_sub(1)))
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
        // Ownership by distance to the word's SPAN (zero inside it), not to its
        // midpoint. A long word's midpoint sits far from its own first nucleus,
        // and by midpoints "ihre" owned the first nucleus OF "grossen" (9.78:
        // 147 ms from ihre's centre, 173 ms from grossen's) — the start-edge
        // search band then began at the next word's own nucleus, skipped the
        // real closure dip at 9.72-9.75, and measured the word as starting at
        // its VOWEL ("roßen"). A nucleus inside a word's transcript span
        // belongs to that word, however long the word is.
        let d = |n: f64, (a, b): (f64, f64)| {
            if n < a {
                a - n
            } else if n > b {
                n - b
            } else {
                0.0
            }
        };
        let owns = |n: f64, own: (f64, f64)| {
            d(n, own) <= d(n, removed)
                && prev.map_or(true, |s| s == own || d(n, own) <= d(n, s))
                && next.map_or(true, |s| s == own || d(n, own) <= d(n, s))
        };
        let a_prev = prev.and_then(|s| self.nuclei.iter().copied().rfind(|&n| owns(n, s)));
        let a_next = next.and_then(|s| self.nuclei.iter().copied().find(|&n| owns(n, s)));
        (a_prev, a_next)
    }

    /// Count nuclei strictly inside a band — the caller uses this to check the word
    /// list is locally consistent with the audio before trusting a placement.
    #[allow(dead_code)] // consistency gate: for the word-list sanity check
    pub fn nuclei_between(&self, a: f64, b: f64) -> usize {
        self.nuclei.iter().filter(|&&n| n > a && n < b).count()
    }

    /// How loud a frame is, in dB above the noise floor, as the max over the bands
    /// that have enough range to be meaningful.
    ///
    /// dB, not a fraction of the range: `tol` downstream is then a physical quantity
    /// rather than "3% of whatever this item's contrast happened to be", which was
    /// 1.7 dB on a clean recording and 0.36 dB on a phone call — below the noise of
    /// the estimate itself, so the comparison degenerated into picking estimator
    /// artefacts. A band with under MIN_BAND_RANGE of range says nothing and is
    /// excluded rather than contributing a near-zero denominator.
    fn level(&self, k: usize) -> f64 {
        let mut m = 0.0f64;
        if self.contrast_bb >= MIN_BAND_RANGE {
            m = m.max((self.bb[k] - self.floor_bb).clamp(0.0, self.contrast_bb));
        }
        if self.contrast_hf >= MIN_BAND_RANGE {
            m = m.max((self.hf[k] - self.floor_hf).clamp(0.0, self.contrast_hf));
        }
        if self.contrast_vb >= MIN_BAND_RANGE {
            m = m.max((self.vb[k] - self.floor_vb).clamp(0.0, self.contrast_vb));
        }
        m
    }

    /// Peak high-band-over-voice-bar dominance in [from, to], in dB.
    ///
    /// The premise under seven failed attempts at "ihren Schoss", never once
    /// measured: that a fricative between two words is high-band dominant enough to
    /// be found. Every version assumed it and differed only in how to threshold or
    /// attribute it; none moved that junction by a frame while all of them moved
    /// dozens of others. If this reads no higher between "ihren" and "Schoss" than it
    /// does across junctions with no fricative at all, the whole approach is dead and
    /// no threshold saves it.
    pub fn band_tilt(&self, from: f64, to: f64) -> f64 {
        let (lo, hi) = (self.frame_at(from.min(to)), self.frame_at(from.max(to)));
        (lo..=hi)
            .map(|k| (self.hf[k] - self.floor_hf) - (self.vb[k] - self.floor_vb))
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Frame where voicing gives way to friction — the high band overtaking the voice
    /// bar — searched in [from, to] and only where no silence precedes it.
    ///
    /// Measured on the real recording at "ihren Schoss": the /sch/ runs 8.150-8.230
    /// and is loud in the VOICE BAR too (36-38 dB), so level() rightly calls it
    /// speech. The only amplitude dip in the whole junction is 8.240-8.260, between
    /// the /sch/ and the following vowel -- inside "Schoss". quiet_runs found the only
    /// dip there is; that dip simply is not the word boundary, so every attempt to
    /// re-rank or re-threshold runs was looking in the wrong place.
    ///
    /// The boundary is the tilt SIGN CHANGE at 8.150: -14.8 dB to +1.2 dB in one
    /// frame. No magnitude threshold is needed and none is used -- "the high band now
    /// dominates and did not a moment ago" is self-calibrating by construction.
    ///
    /// `before` is the first frame of the quiet run that would otherwise be used, and
    /// it is what separates this from a normal pause: at "streichelt dabei" the level
    /// really does fall to 4 dB, and the tilt rise that follows is the /d/ burst AFTER
    /// that silence, so it is correctly ignored. Friction that begins while the
    /// speaker is still going is a word boundary; friction after a pause is just the
    /// next word starting where the pause already said it would.
    /// Boundary at a friction junction, or `None` when the run already answers.
    ///
    /// `edge_hint` is this word's transcript edge nearest the junction, and
    /// `start_edge` says which edge is being placed. Both are needed because
    /// friction-dip-vowel is the same acoustic shape whether the friction is the
    /// next word's ONSET ("ihren | Schoss") or the previous word's CODA
    /// ("das | graue", "bist | du") — and the two need OPPOSITE boundaries: an
    /// onset starts its word, a coda stays with the word before. The first
    /// version of this ignored codas and split "das" in the middle of its /s/.
    ///
    /// `junction` is the transcript start of the word FOLLOWING the friction —
    /// this word's own start when placing a start edge, the NEXT word's start
    /// when placing an end edge. `place_after_coda` is true for start edges,
    /// where a coda-classified junction still needs a boundary placed after the
    /// neighbour's consonant cluster; an end edge falls back to the run instead.
    fn friction_boundary(
        &self,
        from: f64,
        to: f64,
        run: (usize, usize),
        junction: f64,
        place_after_coda: bool,
    ) -> Option<f64> {
        let (lo, hi) = (self.frame_at(from.min(to)), self.frame_at(from.max(to)));
        let need = ((0.040 / self.hop).round() as usize).max(2);
        let tilt = |k: usize| (self.hf[k] - self.floor_hf) - (self.vb[k] - self.floor_vb);
        // quiet_runs measures relative FLATNESS, not silence -- and a sustained
        // fricative is the flattest thing in a junction with no pause, so the "run"
        // is sometimes the fricative itself: Fell's /f/ read as a 13-frame "pause"
        // at level 29 dB, "zaertlich"'s /ch/ as 6 frames at 29.3 dB, and treating
        // them as pauses parked the boundary inside the consonant ("weichef | ell",
        // "zaertli | chueber"). A real pause is flat AND QUIET; friction is flat,
        // loud, and high-band dominant. Measured: real pauses here average 2-11 dB
        // above floor, friction-as-run 29 dB, against a speech contrast of ~30.
        let frames = (run.1 - run.0 + 1) as f64;
        let run_level = (run.0..=run.1).map(|k| self.level(k)).sum::<f64>() / frames;
        let run_tilt = (run.0..=run.1).map(&tilt).sum::<f64>() / frames;
        let loud = 0.5 * self.contrast_bb.max(self.contrast_vb).max(self.contrast_hf);
        let run_is_friction = run_level > loud && run_tilt > 0.0;
        // A LONG run that is not friction is a real pause, and the boundary belongs
        // inside it -- "Kopf. Sie" has a 190 ms silence and its /f/ runs right into
        // it, so contiguity alone would drag "Sie" back into "Kopf".
        if run.1 - run.0 >= need && !run_is_friction {
            return None;
        }
        // A friction FRAME is tilt-positive AND loud. The level test is not
        // optional: silence on real recordings can carry positive tilt (the
        // high-band noise floor sits relatively above the voice bar's — measured
        // +13..+17 dB in the dead pause before "fragt"), and a walk on tilt alone
        // marched through entire pauses, extending "Sattelrobbe." 430 ms to the
        // brink of the next word.
        let fric = |k: usize| tilt(k) > 0.0 && self.level(k) > loud;
        // Friction contiguous BEFORE the run. Friction that stops short of the dip
        // is a coda with a real boundary after it, handled by the run itself.
        let mut k = hi.min(run.0);
        let stop = lo + 1;
        while k > stop && fric(k - 1) {
            k -= 1;
        }
        // Friction contiguous AFTER the run: a stop release. "fragt Irmgard" has a
        // 30 ms /kt/ closure dip followed by a 50-60 ms aspirated release and NO
        // friction before the dip -- the release is as real as any pre-run friction
        // and used to be invisible here, so Irmgard's start fell back to the run and
        // captured it ("tirmgart").
        let mut f = run.1;
        while f < hi && fric(f + 1) {
            f += 1;
        }
        // First frame AFTER all friction on the run's far side. One past the last
        // friction frame, not the frame itself: a boundary placed ON it hands 10 ms
        // of the consonant to the wrong word ("zaertlich"'s /ch/ tail played at the
        // start of "ueber").
        let after = if f > run.1 {
            f + 1
        } else if run_is_friction {
            run.1 + 1
        } else {
            run.1
        };
        let after_t = self.time_of(after);
        // The run's own frames count toward the friction when they ARE the friction
        // (Fell: 1 frame before the run + the 13-frame /f/-as-run).
        let pre_len = hi.min(run.0).saturating_sub(k)
            + if run_is_friction { run.1 - run.0 } else { 0 };
        let has_pre = pre_len >= need && k > stop;
        let post_len = f.saturating_sub(run.1);
        if !has_pre && post_len < need && !run_is_friction {
            return None; // no meaningful friction on either side of the run
        }
        let onset = self.time_of(k);
        let near = self.syllable_period * 0.75;
        // ONE discriminator for both edges: word-initial friction reaches past the
        // transcript start of the word it begins ("Schoss": friction ends 8.240,
        // word starts 8.183; "Stimmen": ~87.71 vs 87.698), while a coda ends short
        // of the next word ("das|graue": 6.300 vs 6.342; "bist|du": 2.920 vs
        // 2.961). An earlier version thresholded the end edge on where the
        // friction STARTS relative to this word's transcript end instead, and the
        // margin does not exist: "die Stimmen"'s onset began 58 ms before a
        // drifted "die" end, "das"'s coda 62 ms before its own.
        // Ownership: does the friction reach past the junction? Judged at the far
        // side of the RUN, not any forward walk -- a bright voiced onset ("graue")
        // keeps the tilt positive well into the next word and would drag the far
        // side over the junction for true codas too.
        if has_pre && self.time_of(run.1) > junction {
            // Word-initial: the word begins where its friction does. The onset must
            // START no later than the junction — every genuine onset measured begins
            // 26-82 ms before its word's transcript start, while friction starting
            // AFTER it is the next word's own interior ("mehr | als": the /s/ inside
            // "als" at +75 ms; "diese | oft" at +50 ms) and accepting it dragged this
            // word's end over the top of its neighbour.
            return (onset >= junction - near && onset <= junction + self.hop)
                .then_some(onset);
        }
        if !place_after_coda {
            // End edge. The word keeps friction on the run's far side only when it
            // is plainly its own: a coda-as-run ("zaertlich"'s /ch/), or a stop
            // release that finishes by the next word's start ("fragt"'s /kt/ ends
            // at 3.88, "Irmgard" starts 3.901). A tail that runs on INTO the next
            // word is that word's own onset ramp ("graue" after "das") and claiming
            // it would eat the neighbour -- there the run fallback stands.
            if run_is_friction && after_t <= junction {
                // Coda-as-run ("zaertlich"): end after the friction. Only when the
                // walk finished BEFORE the junction: a marginally bright vowel can
                // carry the walk deep into the next word ("an," after "freundlich":
                // tilt +1..+4 at full level), and capping that at the junction
                // manufactured ends glued to raw transcript times. Past the
                // junction, the run itself is the honest fallback.
                return Some(self.time_of(run.1).min(junction).max(after_t));
            }
            if !run_is_friction && f > run.1 && after_t <= junction {
                // A release that finishes clearly before the next word: keep it.
                // NOT "within a hop of the junction" — a release ending AT the
                // junction produces an end snapped to the next word's raw
                // transcript time, which is no measurement, and the neighbour's
                // real audio usually starts earlier than its drifted transcript
                // says ("freundlich" ended at exactly "an,"s transcript start,
                // 45 ms inside its measured audio).
                return Some(after_t.max(self.time_of(run.1)));
            }
            // A coda stays with its word; the run's start is the dip after it.
            return None;
        }
        // The friction is the PREVIOUS word's coda or release. Anything contiguous
        // after the run is the tail of the same cluster -- at "bist du" the /t/
        // release sits between the closure dip and the vowel -- and belongs to the
        // neighbour too, so this word starts after it. But ONLY when the tail
        // finishes by the word's own transcript start (the mirror of the end-edge
        // rule): a tail that runs past it is this word's own onset material -- KI's
        // aspiration, Webseite's /v/ -- and the earlier version's cap then snapped
        // the start to the raw transcript time, which is no measurement at all and
        // planted it under the neighbour's honestly-measured end.
        if after > run.1 && after_t > junction + self.hop {
            return None;
        }
        Some(self.time_of(run.1).min(junction).max(after_t.min(junction)))
    }

    /// True when no band has enough range to measure anything (a music bed, a very
    /// reverberant room, heavy compression). The caller must refuse rather than cut.
    fn bands_unusable(&self) -> bool {
        self.contrast_bb < MIN_BAND_RANGE
            && self.contrast_hf < MIN_BAND_RANGE
            && self.contrast_vb < MIN_BAND_RANGE
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
    /// Contiguous runs of "as quiet as it gets here" within `[from, to]`, as frame
    /// index ranges, with stop closures removed.
    ///
    /// The baseline is the 10th percentile, NOT the sample minimum: over ~50 frames a
    /// minimum is biased low by two or three standard deviations, so selecting an
    /// extreme relative to it is extreme-value statistics on measurement noise. The
    /// tolerance scales with the material's own frame-to-frame jitter (MAD), so it can
    /// never be finer than the measurement.
    fn quiet_runs(&self, from: f64, to: f64) -> Vec<(usize, usize)> {
        let (lo, hi) = (self.frame_at(from.min(to)), self.frame_at(from.max(to)));
        if hi < lo + 2 {
            return Vec::new();
        }
        // Rank on a 3-frame MEDIAN of the level, and take its MINIMUM as the
        // baseline. A percentile is only a floor estimate when the stretch actually
        // contains a floor: between two nuclei it is mostly speech, so p10 lands ON
        // the speech (measured: 42.9 dB, right on a fricative) and everything quieter
        // than that became "equally quiet". The median filter removes the single-frame
        // estimator dips that make a raw minimum untrustworthy, so the minimum of the
        // filtered series is both robust and actually near the floor.
        let raw: Vec<f64> = (lo..=hi).map(|k| self.level(k)).collect();
        let med3: Vec<f64> = (0..raw.len())
            .map(|i| {
                let a = raw[i.saturating_sub(1)];
                let b = raw[i];
                let c = raw[(i + 1).min(raw.len() - 1)];
                a.max(b).min(a.min(b).max(c))
            })
            .collect();
        let base = med3.iter().copied().fold(f64::INFINITY, f64::min);
        let mut sorted = med3.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let q = &sorted[..(sorted.len() / 4).max(2)];
        let med = percentile(q, 50.0);
        let mut dev: Vec<f64> = q.iter().map(|v| (v - med).abs()).collect();
        dev.sort_by(|a, b| a.total_cmp(b));
        let tol = (2.0 * 1.4826 * percentile(&dev, 50.0)).clamp(MIN_TOL_DB, MAX_TOL_DB);
        let bar = base + tol;

        let mut runs: Vec<(usize, usize)> = Vec::new();
        let mut cur: Option<usize> = None;
        for k in lo..=hi {
            if med3[k - lo] <= bar {
                cur.get_or_insert(k);
            } else if let Some(s) = cur.take() {
                runs.push((s, k - 1));
            }
        }
        if let Some(s) = cur {
            runs.push((s, hi));
        }
        // NOTE: a burst test to separate a stop CLOSURE from a word gap belongs here
        // (a closure is followed by its release). It is not implemented: the obvious
        // form — "followed by a large rise" — fires on every gap too, since a gap is
        // followed by the next word starting, and separating a transient release from
        // a sustained onset needs validation on real speech that has not been done.
        // Without it, run granularity plus earliest/latest still handles the common
        // cases; a word-final released stop on a kept monosyllable is the known gap.
        runs.retain(|&(a, b)| b > a);
        runs
    }

    /// Dynamic range of a search stretch, in dB — reported on a refusal so the log
    /// says WHICH side had nothing to measure and how close it came, instead of the
    /// bare word "NoMinimum".
    pub fn stretch_range(&self, from: f64, to: f64) -> f64 {
        let (lo, hi) = (self.frame_at(from.min(to)), self.frame_at(from.max(to)));
        if hi < lo + 2 {
            return 0.0;
        }
        let mut lv: Vec<f64> = (lo..=hi).map(|k| self.level(k)).collect();
        lv.sort_by(|a, b| a.total_cmp(b));
        percentile(&lv, 90.0) - percentile(&lv, 10.0)
    }

    /// Is there a measurable boundary in this stretch at all? A continuous background
    /// (music bed, long reverb tail, crosstalk) has no dynamic range to work with, and
    /// guessing inside it is worse than saying so.
    fn stretch_measurable(&self, from: f64, to: f64) -> bool {
        let (lo, hi) = (self.frame_at(from.min(to)), self.frame_at(from.max(to)));
        if hi < lo + 2 {
            return false;
        }
        let mut lv: Vec<f64> = (lo..=hi).map(|k| self.level(k)).collect();
        lv.sort_by(|a, b| a.total_cmp(b));
        percentile(&lv, 90.0) - percentile(&lv, 10.0) >= MIN_STRETCH_RANGE
    }


    /// The measured intermediate values behind one placement, for diagnostics: the
    /// removed word's own syllable nuclei, and the two chosen cut points. Real
    /// recordings are the only way to know whether the analysis matches what a
    /// listener hears, so the tool surfaces these rather than making the user guess.
    pub fn explain(&self, a_prev: f64, a_next: f64) -> (f64, f64, usize) {
        let inner: Vec<f64> = self
            .nuclei
            .iter()
            .copied()
            .filter(|&n| n > a_prev && n < a_next)
            .collect();
        let first = inner.first().copied().unwrap_or((a_prev + a_next) / 2.0);
        let last = inner.last().copied().unwrap_or(first);
        (first, last, inner.len())
    }

    /// Place both edges of a removal between two anchoring nuclei.
    ///
    /// Three separable steps, deliberately: min-seeking LOCATES a region, an existence
    /// test decides whether that region is real, and only then is a pause ALLOCATED
    /// out of it. Collapsing those (choosing a point directly from the minimum) is
    /// what made every pause vanish: at the floor all frames tie exactly, so the
    /// extremes of the tie set are the frames touching the neighbouring words, and the
    /// cut swallowed the whole silence on both sides.
    /// Where a word's audio actually STARTS and ENDS — the inner edges of the quiet
    /// runs flanking it.
    ///
    /// Distinct from [`Self::place_removal`], which returns CUT boundaries: those sit
    /// inside the surrounding pause on purpose, so the join keeps a natural gap.
    /// Using them to audition a word made a sentence-initial word play the whole
    /// pause before it.
    pub fn word_extent(
        &self,
        a_prev: f64,
        a_next: f64,
        hint: (f64, f64),
        // The NEXT word's transcript start, when known: it is the junction the end
        // edge's friction decision needs. Without it, friction there is left to
        // the run (the conservative pre-friction behaviour).
        next_start: Option<f64>,
    ) -> (Edge, Edge) {
        if self.bands_unusable() {
            let e = Edge::fail(EdgeCause::BandsUnusable);
            return (e, e);
        }
        if a_next <= a_prev {
            let e = Edge::fail(EdgeCause::NoAnchor);
            return (e, e);
        }
        let inner: Vec<f64> = self
            .nuclei
            .iter()
            .copied()
            .filter(|&n| n > a_prev && n < a_next)
            .filter(|&n| n >= hint.0 - HINT_TOL && n <= hint.1 + HINT_TOL)
            .collect();
        // Whether the word has a nucleus of its OWN. When it does not, everything
        // below is measured against the transcript hint instead, which is worth
        // reporting separately: it is the detector missing the word, not the audio
        // lacking a boundary.
        let has_nucleus = !inner.is_empty();
        let (first_in, last_in) = match (inner.first(), inner.last()) {
            (Some(&f), Some(&l)) => (f, l),
            _ => (hint.0.clamp(a_prev, a_next), hint.1.clamp(a_prev, a_next)),
        };
        // The word begins where the quiet run before it ENDS, and ends where the run
        // after it BEGINS. Take the run nearest the word on each side, so a pause
        // further out (a sentence break) is not swallowed.
        // Keep the run itself, not just the edge time: its LENGTH is what decides
        // whether a junction has a real pause or only the flattest point of continuous
        // speech, and it has never been observed on real audio.
        let lrun = self.quiet_runs(a_prev, first_in).last().copied();
        let rrun = self.quiet_runs(last_in, a_next).first().copied();
        // Friction that starts while the speaker is still voicing marks the boundary;
        // the amplitude dip after it belongs to a word, not between two. Which word
        // is friction_boundary's decision.
        let raw_start = lrun.map(|(a, b)| {
            self.friction_boundary(a_prev, first_in, (a, b), hint.0, true)
                .unwrap_or(self.time_of(b))
        });
        let raw_end = rrun.map(|(a, b)| {
            next_start
                .and_then(|ns| self.friction_boundary(last_in, a_next, (a, b), ns, false))
                .unwrap_or(self.time_of(a))
        });
        let llen = lrun.map_or(-1.0, |(a, b)| (b - a) as f64);
        let rlen = rrun.map_or(-1.0, |(a, b)| (b - a) as f64);
        // The word's own nucleus MUST lie inside its extent. Nothing forced that
        // before, and the report showed both ways it fails: ~9% of words (nearly all
        // short function words) came back zero-length because the two runs resolved to
        // the same place, and others overshot by up to 0.68 s into the next word
        // because the right-hand run was found far too late. A side that fails this
        // is discarded so the caller falls back to the transcript for that edge alone.
        //
        // Tolerated by HALF A HOP. Nuclei sit on the frame grid, so when the word has
        // its own nucleus this can never reject by rounding. But when it does not,
        // first_in/last_in are the CLAMPED TRANSCRIPT HINT — an arbitrary float that
        // frame_at rounds — so a run touching the window edge overshoots by up to
        // hop/2 and a correct measurement was being thrown away as if it had crossed
        // into the neighbour. That is frame quantisation, not measurement error, and
        // it fired hardest on short words, whose flanking windows are only a few
        // frames wide so the run always touches the edge.
        let q = self.hop / 2.0;
        let start = raw_start.filter(|&t| t <= first_in + q);
        let end = raw_end.filter(|&t| t >= last_in - q);
        let cause = |raw: Option<f64>| {
            if raw.is_some() {
                EdgeCause::ContainmentReject
            } else if has_nucleus {
                EdgeCause::NoQuietRun
            } else {
                EdgeCause::NoNucleusInBand
            }
        };
        let mk = |kept: Option<f64>, raw: Option<f64>, run: f64| match kept {
            Some(t) => Edge::ok(t, run),
            None => Edge { run, ..Edge::fail(cause(raw)) },
        };
        match (start, end) {
            (Some(a), Some(b)) if b <= a => {
                let (lr, rr) = (llen, rlen);
                // Split by nucleus geometry, because these need opposite fixes: with a
                // single nucleus the two search windows MEET at that frame, so both
                // runs can legitimately touch it and the "crossing" may be nothing but
                // quantisation; with several, the windows are disjoint and a crossing
                // should be impossible.
                let cause = if !has_nucleus {
                    EdgeCause::CrossedNoNucleus
                } else if first_in >= last_in {
                    EdgeCause::CrossedOneNucleus
                } else {
                    EdgeCause::CrossedMulti
                };
                let base = Edge { time: None, cause, detail: a - b, run: -1.0 };
                (Edge { run: lr, ..base }, Edge { run: rr, ..base })
            }
            _ => (mk(start, raw_start, llen), mk(end, raw_end, rlen)),
        }
    }

    pub fn place_removal(
        &self,
        a_prev: f64,
        a_next: f64,
        hint: (f64, f64),
        // (previous kept word's end, next kept word's start), when known.
        neighbours: (Option<f64>, Option<f64>),
    ) -> Result<Placement, Refusal> {
        if a_next <= a_prev || self.bands_unusable() {
            return Err(Refusal::NoMinimum);
        }
        // Nuclei belonging to the removed run. Everything between the anchors is NOT
        // automatically part of it: a multi-syllable neighbour contributes nuclei
        // there too, and counting one inflates the removed extent, pushing the end
        // placement past where the word really finishes (measured: syllables reported
        // at 24.183-24.423 for a word the transcript ends at 24.368). The transcript
        // is loose but not that loose, so require membership within a tolerance of it.
        let inner: Vec<f64> = self
            .nuclei
            .iter()
            .copied()
            .filter(|&n| n > a_prev && n < a_next)
            // Bounded by the NEIGHBOURING words' own boundaries, not by a window
            // around the removed word. A ±120 ms window was too loose to exclude the
            // very case it was written for: a nucleus 55 ms past the word's end, which
            // belongs to the next word, still passed and inflated the removed extent.
            .filter(|&n| {
                neighbours.0.map_or(true, |e| n >= e - NEIGHBOUR_TOL)
                    && neighbours.1.map_or(true, |st| n <= st + NEIGHBOUR_TOL)
            })
            .filter(|&n| n >= hint.0 - HINT_TOL && n <= hint.1 + HINT_TOL)
            .collect();
        let (first_in, last_in) = match (inner.first(), inner.last()) {
            (Some(&f), Some(&l)) => (f, l),
            // No nucleus inside the removal — an unstressed function word, or one
            // whose syllable got merged into a louder neighbour by the prominence
            // test. Fall back to the TRANSCRIPT's idea of where the word is rather
            // than to the band midpoint: the midpoint collapses both search stretches
            // and the placement then refuses outright, which is what pushed most
            // short words onto the no-measurement path.
            _ => {
                let h = (hint.0.clamp(a_prev, a_next), hint.1.clamp(a_prev, a_next));
                // Use the transcript span if it yields measurable stretches on both
                // sides; otherwise fall back to the band midpoint. The hint is the
                // better guess, but a removal butted straight against the next word
                // leaves no measurable stretch beside it.
                if self.stretch_measurable(a_prev, h.0) && self.stretch_measurable(h.1, a_next) {
                    h
                } else {
                    let m = (a_prev + a_next) / 2.0;
                    (m, m)
                }
            }
        };
        // Far more syllables between the anchors than the removal can account for
        // means the anchors are mis-assigned (a second speaker, a dropped word).
        let expected = ((last_in - first_in) / self.syllable_period).ceil().max(1.0) as usize + 1;
        if inner.len() > expected + 1 {
            return Err(Refusal::BandTooWide);
        }
        let left = if self.stretch_measurable(a_prev, first_in) {
            self.quiet_runs(a_prev, first_in)
        } else {
            Vec::new()
        };
        let right = if self.stretch_measurable(last_in, a_next) {
            self.quiet_runs(last_in, a_next)
        } else {
            Vec::new()
        };
        if left.is_empty() && right.is_empty() {
            return Err(Refusal::NoMinimum);
        }
        // Earliest surviving run on the left, latest on the right — at RUN
        // granularity, where the preference is meaningful, rather than at frame
        // granularity, where it just picks whatever touches the neighbouring speech.
        // Same correction as word_extent, and this is the path the CUT takes: where a
        // word begins with friction there is no pause before it, only the dip AFTER
        // the friction, which lies inside the word. Cutting at that dip leaves the
        // "sch" of "Schoss" attached to the word before. A zero-width run, because
        // there genuinely is no silence here for the allocation below to hand out.
        let lrun = left.first().map(|&(a, b)| {
            match self.friction_boundary(a_prev, first_in, (a, b), hint.0, true) {
                Some(t) => (t, t),
                None => (self.time_of(a), self.time_of(b)),
            }
        });
        let rrun = right.last().map(|&(a, b)| {
            match neighbours
                .1
                .and_then(|ns| self.friction_boundary(last_in, a_next, (a, b), ns, false))
            {
                Some(t) => (t, t),
                None => (self.time_of(a), self.time_of(b)),
            }
        });
        // Each side falls back to the removed word's own edge when unmeasurable, so
        // the allocation below still has a sane span to work with.
        let (lt0, lt1) = lrun.unwrap_or((first_in, first_in));
        let (rt0, rt1) = rrun.unwrap_or((last_in, last_in));

        // Allocate a pause out of the silence the runs actually revealed.
        let s_l = (lt1 - lt0).max(0.0);
        let s_r = (rt1 - rt0).max(0.0);
        let mut target = s_l.min(s_r);
        // A markedly long pause on either side is a prosodic boundary — a phrase or
        // sentence break. Keep it, rather than butting two sentences together.
        if s_l.max(s_r) > 1.5 * self.syllable_period {
            target = s_l.max(s_r);
        }
        target = target.min(2.0 * self.syllable_period);
        let total = s_l + s_r;
        let (g_l, g_r) = if total > 0.0 {
            let g = target * s_l / total;
            (g, target - g)
        } else {
            (0.0, 0.0)
        };
        let start = lrun.map(|_| (lt0 + g_l).clamp(lt0, lt1));
        let end = rrun.map(|_| (rt1 - g_r).clamp(rt0, rt1));
        // Fade length on a continuum rather than a butt/crossfade branch that can pick
        // the wrong side: over true silence a 5 ms equal-power fade is indistinguishable
        // from an abutment, so nothing is lost and the discrete failure mode goes away.
        let lvl = |t: Option<f64>| t.map(|x| self.level(self.frame_at(x))).unwrap_or(0.0);
        let q = (lvl(start).max(lvl(end)) / self.contrast_bb.max(1.0)).clamp(0.0, 1.0);
        Ok(Placement {
            start,
            end,
            fade: 0.005 + 0.020 * q,
            gap: (start.unwrap_or(lt0) - lt0) + (rt1 - end.unwrap_or(rt1)),
        })
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
            let apart = (c - last) as f64 * hop_s;
            // Merge ONLY when the peaks are also close in time. The dip test alone
            // absorbs a quiet word's syllable into a louder neighbour whenever the
            // level barely dips between them — which is normal in connected speech —
            // and that word then has no nucleus, so its removal cannot be measured at
            // all. Ripple within one vowel occurs over a few tens of ms; two peaks
            // further apart than the shortest syllable are separate syllables however
            // shallow the valley.
            if apart < MIN_SYLLABLE_SEP && mid[c].min(mid[last]) - valley < NUCLEUS_PROMINENCE_DB {
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
    /// A word's edges are independent, and each carries its own reason. Regression
    /// for the report printing a fallback as `+0.000` — indistinguishable from a
    /// perfect measurement — and for one bad side discarding a good one.
    #[test]
    fn word_extent_reports_a_cause_per_edge() {
        let mut syn = Syn::new();
        syn.quiet(0.30)
            .vowel(0.20, 120.0, 0.5) // previous word
            .quiet(0.15)
            .vowel(0.20, 130.0, 0.5) // the word under test
            .quiet(0.15)
            .vowel(0.20, 140.0, 0.5) // next word
            .quiet(0.30);
        let a = SpeechAnalysis::new(&syn.s, SR).expect("analysable");
        let (o, f) = a.word_extent(0.40, 1.10, (0.65, 0.85), None);
        assert_eq!(o.cause, EdgeCause::Measured, "left edge: {o:?}");
        assert_eq!(f.cause, EdgeCause::Measured, "right edge: {f:?}");
        assert!(o.time.unwrap() < f.time.unwrap());

        // Degenerate anchors must name themselves, not masquerade as a measurement.
        let (o, f) = a.word_extent(1.0, 0.5, (0.65, 0.85), None);
        assert_eq!(o.cause, EdgeCause::NoAnchor);
        assert_eq!(f.cause, EdgeCause::NoAnchor);
        assert!(o.time.is_none() && f.time.is_none());
    }

    /// The containment filter tolerates half a hop. When a word has no nucleus of its
    /// own the window edge is the CLAMPED TRANSCRIPT HINT, an arbitrary float that
    /// frame_at rounds, so a correct run overshoots by up to hop/2 — that is frame
    /// quantisation, and rejecting it threw away exactly the short-word cohort.
    #[test]
    fn containment_tolerates_frame_quantisation() {
        let mut syn = Syn::new();
        syn.quiet(0.30)
            .vowel(0.20, 120.0, 0.5)
            .quiet(0.15)
            .vowel(0.20, 130.0, 0.5)
            .quiet(0.15)
            .vowel(0.20, 140.0, 0.5)
            .quiet(0.30);
        let a = SpeechAnalysis::new(&syn.s, SR).expect("analysable");
        // Sweep the hint off the frame grid by sub-hop amounts: none of these may
        // flip an edge from measured to rejected.
        for k in 0..10 {
            let d = k as f64 * a.hop() / 10.0;
            let (o, f) = a.word_extent(0.40, 1.10, (0.65 + d, 0.85 + d), None);
            assert_ne!(o.cause, EdgeCause::ContainmentReject, "offset {d:.4}: {o:?}");
            assert_ne!(f.cause, EdgeCause::ContainmentReject, "offset {d:.4}: {f:?}");
        }
    }

    /// A quiet VOICED word between two fricative-rich loud ones measures.
    ///
    /// NOT a regression guard for the voice-bar change: it passes with the voice band
    /// excluded from level() too, so it does not discriminate. Synthetic speech has a
    /// near-digital-silence floor, which keeps a faint vowel far above baseline in
    /// every band; the real failure needs a search window containing no true pause.
    /// Kept as a property test, and as a marker that the fixture set cannot yet
    /// reproduce the crossing — real audio is needed for that.
    #[test]
    fn a_quiet_voiced_word_is_not_silence() {
        let mut syn = Syn::new();
        syn.quiet(0.30)
            .vowel(0.20, 120.0, 0.6)
            .fricative(0.10, 0.6) // loud, HF-rich neighbour
            .quiet(0.12)
            .vowel(0.10, 130.0, 0.12) // the unstressed function word: voiced, faint
            .quiet(0.12)
            .fricative(0.10, 0.6)
            .vowel(0.20, 140.0, 0.6)
            .quiet(0.30);
        let a = SpeechAnalysis::new(&syn.s, SR).expect("analysable");
        let (o, f) = a.word_extent(0.55, 1.00, (0.72, 0.82), None);
        assert_eq!(o.cause, EdgeCause::Measured, "start: {o:?}");
        assert_eq!(f.cause, EdgeCause::Measured, "end: {f:?}");
        let (st, en) = (o.time.unwrap(), f.time.unwrap());
        assert!(en > st, "extent collapsed: {st:.3}-{en:.3}");
        assert!(
            en - st > 0.04,
            "extent implausibly short for a 100 ms word: {st:.3}-{en:.3}"
        );
    }

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
        if let Ok(pl) = a.place_removal(a_prev, a_next, (japan_end, sch), (None, None)) {
            // Only a PLACED edge carries the invariant; an unmeasured one falls back
            // to the transcript in the caller.
            let (cut_s, cut_e) = (pl.start.unwrap_or(a_prev), pl.end.unwrap_or(a_next));
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
                    if let Ok(pl) = a.place_removal(p, n, (w1_end, w2), (None, None)) {
                        let (cs, ce) = (pl.start.unwrap_or(p), pl.end.unwrap_or(n));
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
        let pl = a.place_removal(p, n, (rem, rem_end), (None, None)).expect("placed");
        let cut_e = pl.end.expect("end measured");
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
        let pl = a.place_removal(p, n, (fric, w2), (None, None)).expect("placed");
        let (cut_s, cut_e) = (pl.start.expect("start measured"), pl.end.expect("end measured"));
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
        let pl = a.place_removal(p, n, (w1_end, w2), (None, None)).expect("placed");
        let cut_e = pl.end.expect("end measured");
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
        let pl = a.place_removal(p, n, (rem, rem_end), (None, None)).expect("placed");
        let cut_s = pl.start.expect("start measured");
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
        let pl = a.place_removal(p, n2, (rem, rem_end), (None, None)).expect("placed");
        let cut_s = pl.start.expect("start measured");
        assert!(
            cut_s <= rem + 0.015,
            "cut starts at {cut_s}, after the word's onset at {rem} — its attack survives"
        );
        assert!(cut_s >= w1_end - 0.03, "cut at {cut_s} bit into the kept word (ends {w1_end})");
    }

    /// THE REGRESSION THE MAINTAINER CAUGHT. Deleting a word beside a long pause must
    /// LEAVE a pause. Picking a point straight from the level minimum consumed the
    /// silence on both sides — at the floor every frame ties exactly, so the extremes
    /// of the tie set are the frames touching the neighbouring words — and two
    /// sentences slammed together. No previous test asserted surviving pause length,
    /// which is why this reached real audio.
    #[test]
    fn a_long_pause_survives_the_cut() {
        let mut s = Syn::new();
        s.quiet(0.25);
        let w1 = s.at();
        s.vowel(0.18, 120.0, 0.5);
        let w1_end = s.at();
        s.quiet(0.45); // a sentence-boundary pause
        let rem = s.at();
        s.vowel(0.16, 120.0, 0.5);
        let rem_end = s.at();
        s.quiet(0.45);
        let w2 = s.at();
        s.vowel(0.18, 120.0, 0.5);
        let w2_end = s.at();
        s.quiet(0.25);
        let a = SpeechAnalysis::new(&s.s, SR).expect("analysable");
        let (p, n) = a.anchors(Some((w1, w1_end)), (rem, rem_end), Some((w2, w2_end)));
        let pl = a
            .place_removal(p.expect("prev"), n.expect("next"), (rem, rem_end), (None, None))
            .expect("placed");
        assert!(
            pl.gap >= 0.150,
            "only {:.3}s of pause left at the join — a 0.45s pause was available on              each side; the cut swallowed it",
            pl.gap
        );
        assert!(pl.gap <= 0.700, "left {:.3}s, more than the material had", pl.gap);
    }

    /// Real speech does not fall to the noise floor between words — room tone, breath
    /// and reverb keep it up. A threshold-based search finds nothing here and fails
    /// silently; every earlier fixture had gaps that reached the floor, which is why
    /// they passed while real audio did not.
    #[test]
    fn a_gap_that_never_reaches_the_floor_is_still_found() {
        let sr = SR;
        let mut s = Syn::new();
        s.quiet(0.25);
        let w1 = s.at();
        s.vowel(0.18, 120.0, 0.5);
        let w1_end = s.at();
        // "Silence" 18 dB below speech but ~30 dB ABOVE the room floor.
        let n = (sr * 0.12) as usize;
        for i in 0..n {
            let t = i as f64 / sr;
            s.s.push((2.0 * PI * 150.0 * t).sin() * 0.06);
        }
        let rem = s.at();
        s.vowel(0.16, 120.0, 0.5);
        let rem_end = s.at();
        for i in 0..n {
            let t = i as f64 / sr;
            s.s.push((2.0 * PI * 150.0 * t).sin() * 0.06);
        }
        let w2 = s.at();
        s.vowel(0.18, 120.0, 0.5);
        let w2_end = s.at();
        s.quiet(0.25);
        let a = SpeechAnalysis::new(&s.s, sr).expect("analysable");
        let (p, n2) = a.anchors(Some((w1, w1_end)), (rem, rem_end), Some((w2, w2_end)));
        let pl = a
            .place_removal(p.expect("prev"), n2.expect("next"), (rem, rem_end), (None, None))
            .expect("a boundary exists even though nothing reaches the floor");
        let cs = pl.start.expect("start measured");
        assert!(cs <= rem + 0.02, "cut starts {cs:.3} after the word at {rem:.3}");
        let ce = pl.end.expect("end measured");
        assert!(ce >= rem_end - 0.02, "cut ends {ce:.3} before the word ends {rem_end:.3}");
    }

    /// A quiet unstressed word beside a loud one must keep its OWN nucleus. The
    /// prominence merge previously absorbed it into the louder neighbour whenever the
    /// level barely dipped between them — normal in connected speech — and the word
    /// then had no nucleus, so its removal could not be measured and fell back to bare
    /// transcript times. That is what left two of three real cuts unmeasured.
    #[test]
    fn a_quiet_word_keeps_its_own_nucleus() {
        let mut s = Syn::new();
        s.quiet(0.25);
        let loud = s.at();
        s.vowel(0.20, 120.0, 0.6);
        let loud_end = s.at();
        s.quiet(0.02); // barely a dip — connected speech
        let soft = s.at();
        s.vowel(0.14, 120.0, 0.12); // ~14 dB down: an unstressed function word
        let soft_end = s.at();
        s.quiet(0.02);
        s.vowel(0.20, 120.0, 0.6);
        s.quiet(0.25);
        let a = SpeechAnalysis::new(&s.s, SR).expect("analysable");
        let own = a
            .nuclei
            .iter()
            .filter(|&&n| n > soft - 0.02 && n < soft_end + 0.02)
            .count();
        assert!(
            own >= 1,
            "the quiet word [{soft:.3}, {soft_end:.3}] has no nucleus of its own: {:?}",
            a.nuclei
        );
        assert!(loud < loud_end, "fixture sanity");
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




/// Diagnostics against a real recording, off by default.
///
/// Seven attempts at the "ihren Schoss" junction each cost a full round trip
/// through a human ear, so every one was a guess between measurements. With the
/// source file these become local: `PARO_WAV=<path> cargo test --lib real_ -- --nocapture`.
#[cfg(test)]
mod real_audio {
    use super::*;

    /// (label, previous word, word under test, next word) — all transcript spans.
    type Case = (&'static str, (f64, f64), (f64, f64), (f64, f64));

    fn load() -> Option<SpeechAnalysis> {
        let path = std::env::var("PARO_WAV").ok()?;
        let bytes = std::fs::read(path).ok()?;
        let (samples, ch, sr) = crate::dsp::parse_wav(&bytes).ok()?;
        let mono: Vec<f64> = if ch <= 1 {
            samples
        } else {
            samples.chunks(ch).map(|f| f.iter().sum::<f64>() / ch as f64).collect()
        };
        SpeechAnalysis::new(&mono, sr)
    }

    /// Fetch the real transcript once and cache it beside the WAV, so the whole-clip
    /// regression below runs offline. Uses the configured local Whisper endpoint and
    /// the key from the credential store; the key is never printed.
    /// `PARO_WAV=<path> PARO_FETCH=1 cargo test --lib real_fetch -- --nocapture`
    #[test]
    fn real_fetch_transcript() {
        if std::env::var("PARO_FETCH").is_err() {
            return;
        }
        let Ok(wav) = std::env::var("PARO_WAV") else { return };
        let out = std::path::Path::new(&wav).with_extension("words.json");
        let mp3 = std::env::var("PARO_MP3").unwrap_or_default();
        let bytes = std::fs::read(&mp3).expect("PARO_MP3");
        // Straight from the credential store into the request. Never printed, never
        // written to disk, never passed through the environment.
        let key = keyring::Entry::new("realackey", "apikey:whisper-local")
            .ok()
            .and_then(|e| e.get_password().ok())
            .filter(|k| !k.trim().is_empty());
        eprintln!("key from credential store: {}", if key.is_some() { "found" } else { "absent (keyless server?)" });
        let base = std::env::var("PARO_BASE").expect("PARO_BASE");
        let model = std::env::var("PARO_MODEL").expect("PARO_MODEL");
        let t = crate::providers::transcription::OpenAiTranscriber::new(base, key, model);
        let clip = crate::providers::transcription::AudioClip {
            bytes,
            filename: "paro.mp3".into(),
            mime: "audio/mpeg".into(),
        };
        let opts = crate::providers::transcription::TranscribeOptions {
            language: Some("de".into()),
            prompt: None,
        };
        let rt = tokio::runtime::Runtime::new().unwrap();
        let tr = rt
            .block_on(async {
                use crate::providers::transcription::TranscriptionProvider as _;
                t.transcribe(clip, &opts, tokio_util::sync::CancellationToken::new()).await
            })
            .expect("transcribe");
        let rows: Vec<String> = tr
            .words
            .iter()
            .map(|w| format!("{:.3}|{:.3}|{}", w.start, w.end, w.text.replace('|', " ")))
            .collect();
        eprintln!("{} words -> {}", rows.len(), out.display());
        std::fs::write(&out, rows.join("
")).unwrap();
    }

    /// The nine labelled junctions, measured end to end on the real recording.
    /// Transcript times are the Whisper output as printed by the clip report.
    #[test]
    fn real_labelled_junctions() {
        let Some(a) = load() else { return };
        // (label, prev, word, next) — the pair under test is `word`/`next`.
        let cases: &[Case] = &[
            ("BAD  ihren|Schoss", (7.843, 7.923), (7.983, 8.143), (8.183, 8.463)),
            ("BAD  Senioren|schmiegt", (16.665, 16.746), (16.766, 17.166), (17.186, 17.506)),
            ("BAD  grossen|schwarzen", (9.563, 9.703), (9.763, 10.143), (10.183, 10.603)),
            ("BAD  weiche|Fell", (6.342, 6.622), (6.742, 7.002), (7.102, 7.322)),
            ("BAD  Sie|schaut", (12.584, 12.824), (13.084, 13.184), (13.204, 13.384)),
            ("CODA das end~6.30", (5.982, 6.122), (6.162, 6.282), (6.342, 6.622)),
            ("CODA graue start~6.34", (6.162, 6.282), (6.342, 6.622), (6.742, 7.002)),
            ("CODA bist end~2.92", (2.401, 2.481), (2.781, 2.941), (2.961, 3.001)),
            ("CODA du start~2.96", (2.781, 2.941), (2.961, 3.001), (3.061, 3.301)),
            ("VSTOP ihre end", (9.223, 9.503), (9.563, 9.703), (9.763, 10.143)),
            ("RPT fragt start", (3.061, 3.301), (3.641, 3.881), (3.901, 4.181)),
            ("RPT Irmgard start", (3.641, 3.881), (3.901, 4.181), (4.201, 4.561)),
            ("RPT und end", (4.201, 4.561), (4.702, 4.782), (4.822, 5.182)),
            ("RPT streichelt start", (4.702, 4.782), (4.822, 5.182), (5.222, 5.422)),
            ("FRIC Fell start~7.02", (6.742, 7.002), (7.102, 7.322), (7.422, 7.522)),
            ("FRIC zaertlich end~5.93", (5.222, 5.422), (5.502, 5.922), (5.982, 6.122)),
            ("FRIC ueber start~5.94", (5.502, 5.922), (5.982, 6.122), (6.162, 6.282)),
            ("GOOD streichelt|dabei", (4.702, 4.782), (4.822, 5.182), (5.222, 5.422)),
            ("GOOD schon|seit", (107.486, 107.606), (107.606, 107.746), (107.786, 107.946)),
        ];
        for (name, prev, w, next) in cases {
            let (ap, an) = a.anchors(Some(*prev), *w, Some(*next));
            let (o, f) = match (ap, an) {
                (Some(x), Some(y)) => a.word_extent(x, y, *w, Some(next.0)),
                _ => (Edge::fail(EdgeCause::NoAnchor), Edge::fail(EdgeCause::NoAnchor)),
            };
            eprintln!(
                "{name:26} transcript {:.3}-{:.3}  measured {:?}({:?})-{:?}({:?})  end drift {:+.3}",
                w.0,
                w.1,
                o.time.map(|t| (t * 1000.0).round() / 1000.0),
                o.cause,
                f.time.map(|t| (t * 1000.0).round() / 1000.0),
                f.cause,
                f.time.unwrap_or(w.1) - w.1
            );
        }
    }

    /// The whole-clip tally, exactly as preview::word_bounds computes it, so a change
    /// can be scored against all 821 words in a second instead of a REAPER round trip.
    #[test]
    fn real_clip_tally() {
        let Some(a) = load() else { return };
        let Ok(wav) = std::env::var("PARO_WAV") else { return };
        let path = std::path::Path::new(&wav).with_extension("words.json");
        let Ok(text) = std::fs::read_to_string(&path) else {
            eprintln!("run real_fetch_transcript first");
            return;
        };
        let words: Vec<(f64, f64, String)> = text
            .lines()
            .filter_map(|l| {
                let mut p = l.split('|');
                Some((p.next()?.parse().ok()?, p.next()?.parse().ok()?, p.next()?.to_string()))
            })
            .collect();
        let bleed = a.syllable_period * 0.5;
        let (clip_a, clip_b) = a.span();
        let mut tally: std::collections::BTreeMap<&str, usize> = Default::default();
        let mut bounds = Vec::new();
        for i in 0..words.len() {
            let w = (words[i].0, words[i].1);
            let prev = i.checked_sub(1).map(|j| (words[j].0, words[j].1));
            let next = words.get(i + 1).map(|n| (n.0, n.1));
            let (ap, an) = a.anchors(prev, w, next);
            let (o, f) = a.word_extent(ap.unwrap_or(clip_a), an.unwrap_or(clip_b), w, next.map(|n| n.0));
            let lo = prev.map(|(_, pe)| pe.min(w.0) - bleed);
            let hi = next.map(|(ns, _)| ns.max(w.1) + bleed);
            let o = if o.time.is_some_and(|t| lo.is_some_and(|l| t < l)) {
                o.reject(EdgeCause::BleedReject)
            } else {
                o
            };
            let f = if f.time.is_some_and(|t| hi.is_some_and(|h| t > h)) {
                f.reject(EdgeCause::BleedReject)
            } else {
                f
            };
            *tally.entry(o.cause.token()).or_default() += 1;
            *tally.entry(f.cause.token()).or_default() += 1;
            bounds.push((o.time.unwrap_or(w.0), f.time.unwrap_or(w.1)));
        }
        let mut v: Vec<_> = tally.into_iter().collect();
        v.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
        eprintln!(
            "TALLY {}",
            v.iter().map(|(k, n)| format!("{k} {n}")).collect::<Vec<_>>().join(" | ")
        );
        let neg = bounds.windows(2).filter(|p| p[1].0 < p[0].1).count();
        eprintln!("overlapping neighbours: {neg} (raw; the editor reconciles them to midpoints)");
        for (i, w) in bounds.windows(2).enumerate() {
            if w[1].0 < w[0].1 {
                eprintln!(
                    "OVERLAP {:>3} {:<14} {:.3}-{:.3} | {:<14} {:.3}-{:.3}",
                    i, words[i].2, w[0].0, w[0].1, words[i + 1].2, w[1].0, w[1].1
                );
            }
        }
        if std::env::var("PARO_BLEED").is_ok() {
            for (i, w) in words.iter().enumerate() {
                let prev = i.checked_sub(1).map(|j| (words[j].0, words[j].1));
                let next = words.get(i + 1).map(|n| (n.0, n.1));
                let ww = (w.0, w.1);
                let (ap, an) = a.anchors(prev, ww, next);
                let (o, _) = a.word_extent(ap.unwrap_or(clip_a), an.unwrap_or(clip_b), ww, next.map(|n| n.0));
                let lo = prev.map(|(_, pe)| pe.min(w.0) - bleed);
                if o.time.is_some_and(|t| lo.is_some_and(|l| t < l)) {
                    eprintln!(
                        "BLEEDSTART {:>16} after {:<16} start {:.3} -> {:.3} (bound {:.3})",
                        w.2,
                        i.checked_sub(1).map(|j| words[j].2.clone()).unwrap_or_default(),
                        w.0,
                        o.time.unwrap(),
                        lo.unwrap()
                    );
                }
            }
        }
    }

    /// What the CUT would do when a fricative-initial word is deleted.
    #[test]
    fn real_cut_placement() {
        let Some(a) = load() else { return };
        // (label, prev kept, removed, next kept)
        let cases: [Case; 2] = [
            ("delete Schoss (/sch/ onset)", (7.983, 8.143), (8.183, 8.463), (8.823, 8.903)),
            ("delete dabei (/d/ onset)", (4.822, 5.182), (5.222, 5.422), (5.502, 5.922)),
        ];
        for (name, prev, rem, next) in cases {
            let (ap, an) = a.anchors(Some(prev), rem, Some(next));
            match (ap, an) {
                (Some(x), Some(y)) => {
                    match a.place_removal(x, y, rem, (Some(prev.1), Some(next.0))) {
                        Ok(p) => eprintln!(
                            "CUT {name:32} transcript {:.3}-{:.3}  cut {:?}-{:?}",
                            rem.0,
                            rem.1,
                            p.start.map(|t| (t * 1000.0).round() / 1000.0),
                            p.end.map(|t| (t * 1000.0).round() / 1000.0)
                        ),
                        Err(e) => eprintln!("CUT {name:32} declined: {e:?}"),
                    }
                }
                _ => eprintln!("CUT {name:32} unanchorable"),
            }
        }
    }

    /// Does a /sch/ stand out in ANY band measure? Prints the frame-by-frame picture
    /// across a known-bad junction and a known-good one, so the two can be compared
    /// directly instead of inferred from boundary positions.
    #[test]
    fn real_junction_profile() {
        let Some(a) = load() else {
            eprintln!("set PARO_WAV to run");
            return;
        };
        eprintln!(
            "hop {:.3}  floors bb {:.1} hf {:.1} vb {:.1}  contrast bb {:.1} hf {:.1} vb {:.1}",
            a.hop, a.floor_bb, a.floor_hf, a.floor_vb, a.contrast_bb, a.contrast_hf, a.contrast_vb
        );
        for (name, from, to) in [
            ("freundlich|an (/ch/ + vowel)", 14.68, 14.90),
            ("fragt|Irmgard (/t/ burst)", 3.72, 3.95),
            ("und|streichelt (/t/+/scht/)", 4.68, 4.95),
            ("bist|du     (coda /st/ stolen?)", 2.83, 3.06),
            ("ihren|Schoss  (BAD, /sch/)", 8.00, 8.40),
            ("streichelt|dabei (GOOD, /d/)", 5.10, 5.30),
            ("weiche|Fell   (BAD, /f/)", 6.95, 7.25),
            ("schon|seit    (GOOD, /z/)", 107.70, 107.90),
            ("ihre|grossen  (voiced /g/ stop)", 9.60, 10.00),
        ] {
            eprintln!("\n--- {name} ---");
            eprintln!("   time     bb     hf     vb   tilt  level  nucleus");
            let (lo, hi) = (a.frame_at(from), a.frame_at(to));
            for k in lo..=hi {
                let t = a.time_of(k);
                let tilt = (a.hf[k] - a.floor_hf) - (a.vb[k] - a.floor_vb);
                let nuc = a.nuclei.iter().any(|&n| (n - t).abs() < a.hop / 2.0);
                eprintln!(
                    "{:7.3} {:6.1} {:6.1} {:6.1} {:6.1} {:6.1}  {}",
                    t,
                    a.bb[k] - a.floor_bb,
                    a.hf[k] - a.floor_hf,
                    a.vb[k] - a.floor_vb,
                    tilt,
                    a.level(k),
                    if nuc { "*" } else { "" }
                );
            }
        }
    }
}
