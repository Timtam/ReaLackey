//! Audio-DSP feature extraction (design §kap-capabilities, F9).
//!
//! Pure, host-independent DSP: it takes a block of interleaved samples (as read
//! from a REAPER audio accessor on the main thread) and computes loudness and
//! spectral features. Keeping it free of any REAPER/FFI dependency means it is
//! deterministic and unit-testable without a live host — the one part of this
//! project that genuinely can be verified in CI.
//!
//! Everything is computed at a fixed 48 kHz analysis rate (the accessor
//! resamples for us) so the BS.1770 K-weighting coefficients below are exact.

use serde::Serialize;

/// Amplitude floor reported as a finite dBFS value instead of -inf.
const DBFS_FLOOR: f64 = -150.0;
/// FFT window size for the spectral analysis (power of two).
const FFT_SIZE: usize = 4096;

/// Extracted audio features. Serialized straight to the tool's JSON result.
#[derive(Debug, Clone, Serialize)]
pub struct AudioFeatures {
    pub sample_rate: f64,
    pub channels: usize,
    /// Samples per channel that were analysed.
    pub frames: usize,
    pub duration_seconds: f64,
    /// Peak sample magnitude in dBFS (sample peak, not oversampled true-peak).
    pub peak_dbfs: f64,
    pub rms_dbfs: f64,
    /// Peak-to-RMS ratio in dB (how "punchy" vs. compressed the signal is).
    pub crest_factor_db: f64,
    pub dc_offset: f64,
    /// Number of samples at/above ~full scale (|x| >= 0.999).
    pub clip_count: usize,
    pub clipping: bool,
    /// True when the peak is below -60 dBFS (effectively silent).
    pub silent: bool,
    /// Integrated loudness (BS.1770-4, gated). None if under ~400 ms of audio.
    pub integrated_lufs: Option<f64>,
    /// Loudness range (EBU Tech 3342), in LU. None under ~a few seconds of audio.
    pub loudness_range_lu: Option<f64>,
    /// Maximum momentary (400 ms) loudness, LUFS.
    pub momentary_lufs_max: Option<f64>,
    /// Maximum short-term (3 s) loudness, LUFS.
    pub short_term_lufs_max: Option<f64>,
    /// True peak in dBTP (4x-oversampled inter-sample peak; can exceed 0).
    pub true_peak_dbtp: Option<f64>,
    /// Spectral centre of mass in Hz (a rough "brightness" indicator).
    pub spectral_centroid_hz: Option<f64>,
    /// Loudest spectral bin in Hz.
    pub dominant_frequency_hz: Option<f64>,
    /// Share of spectral energy below 250 Hz, as a percentage.
    pub band_low_pct: Option<f64>,
    /// Share of spectral energy in 250 Hz .. 4 kHz, as a percentage.
    pub band_mid_pct: Option<f64>,
    /// Share of spectral energy above 4 kHz, as a percentage.
    pub band_high_pct: Option<f64>,
}

/// Analyse `samples` (interleaved, `channels` per frame) at `sample_rate`.
pub fn analyze(samples: &[f64], channels: usize, sample_rate: f64) -> AudioFeatures {
    let channels = channels.max(1);
    let sample_rate = if sample_rate > 0.0 {
        sample_rate
    } else {
        48_000.0
    };
    let frames = samples.len() / channels;

    // ---- time-domain: peak / RMS / DC / clipping ----------------------------
    // Non-finite input (a stray NaN/Inf from a float source) is treated as 0 so
    // it cannot poison the reductions — serde would render such a result as null.
    let mut peak = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut sum = 0.0f64;
    let mut clip_count = 0usize;
    for &x in &samples[..frames * channels] {
        let x = if x.is_finite() { x } else { 0.0 };
        let a = x.abs();
        if a > peak {
            peak = a;
        }
        if a >= 0.999 {
            clip_count += 1;
        }
        sum_sq += x * x;
        sum += x;
    }
    let n = (frames * channels).max(1) as f64;
    let rms = (sum_sq / n).sqrt();
    let dc_offset = sum / n;
    let peak_dbfs = to_dbfs(peak);
    let rms_dbfs = to_dbfs(rms);

    // ---- mono downmix for spectral work -------------------------------------
    let mono = downmix(samples, channels, frames);

    let sp = spectral(&mono, sample_rate);
    // Keep loudness finite: serde_json cannot represent -inf/NaN and would fail
    // the whole serialization, so drop a non-finite result to None instead.
    let integrated_lufs =
        integrated_loudness(samples, channels, frames, sample_rate).filter(|v| v.is_finite());
    let momentary_lufs_max =
        momentary_max(samples, channels, frames, sample_rate).filter(|v| v.is_finite());
    let (lra, st_max) = short_term_stats(samples, channels, frames, sample_rate);
    let loudness_range_lu = lra.filter(|v| v.is_finite());
    let short_term_lufs_max = st_max.filter(|v| v.is_finite());
    let true_peak_dbtp = true_peak_dbtp(samples, channels, frames).filter(|v| v.is_finite());

    AudioFeatures {
        sample_rate,
        channels,
        frames,
        duration_seconds: frames as f64 / sample_rate,
        peak_dbfs,
        rms_dbfs,
        crest_factor_db: (peak_dbfs - rms_dbfs).max(0.0),
        dc_offset,
        clip_count,
        clipping: clip_count > 0,
        silent: peak_dbfs < -60.0,
        integrated_lufs,
        loudness_range_lu,
        momentary_lufs_max,
        short_term_lufs_max,
        true_peak_dbtp,
        spectral_centroid_hz: sp.centroid_hz,
        dominant_frequency_hz: sp.dominant_hz,
        band_low_pct: sp.low_pct,
        band_mid_pct: sp.mid_pct,
        band_high_pct: sp.high_pct,
    }
}

fn to_dbfs(amp: f64) -> f64 {
    if amp <= 1e-12 {
        DBFS_FLOOR
    } else {
        20.0 * amp.log10()
    }
}

fn downmix(samples: &[f64], channels: usize, frames: usize) -> Vec<f64> {
    let mut mono = vec![0.0f64; frames];
    for (f, m) in mono.iter_mut().enumerate() {
        let mut acc = 0.0;
        for c in 0..channels {
            let s = samples[f * channels + c];
            acc += if s.is_finite() { s } else { 0.0 };
        }
        *m = acc / channels as f64;
    }
    mono
}

// ---- spectral analysis ------------------------------------------------------

/// Spectral summary; every field is None when there is less than one full
/// analysis window of audio.
struct Spectral {
    centroid_hz: Option<f64>,
    dominant_hz: Option<f64>,
    low_pct: Option<f64>,
    mid_pct: Option<f64>,
    high_pct: Option<f64>,
}

impl Spectral {
    fn empty() -> Self {
        Self {
            centroid_hz: None,
            dominant_hz: None,
            low_pct: None,
            mid_pct: None,
            high_pct: None,
        }
    }
}

fn spectral(mono: &[f64], sample_rate: f64) -> Spectral {
    if mono.len() < FFT_SIZE {
        return Spectral::empty();
    }
    let window = hann(FFT_SIZE);
    let hop = FFT_SIZE / 2;
    let bins = FFT_SIZE / 2; // usable bins 1..=bins (skip DC)
    let mut power = vec![0.0f64; bins + 1];
    let mut frame_count = 0usize;

    let mut start = 0;
    while start + FFT_SIZE <= mono.len() {
        let mut re = vec![0.0f64; FFT_SIZE];
        let mut im = vec![0.0f64; FFT_SIZE];
        for i in 0..FFT_SIZE {
            re[i] = mono[start + i] * window[i];
        }
        fft(&mut re, &mut im);
        for (k, p) in power.iter_mut().enumerate() {
            *p += re[k] * re[k] + im[k] * im[k];
        }
        frame_count += 1;
        start += hop;
    }
    if frame_count == 0 {
        return Spectral::empty();
    }

    let bin_hz = sample_rate / FFT_SIZE as f64;
    let nyquist = sample_rate / 2.0;
    let hi_edge = 20_000.0f64.min(nyquist);

    let mut num = 0.0f64; // sum f*power
    let mut den = 0.0f64; // sum power
    let mut low = 0.0f64;
    let mut mid = 0.0f64;
    let mut high = 0.0f64;
    let mut peak_power = 0.0f64;
    let mut peak_bin = 0usize;
    for (k, &p) in power.iter().enumerate().take(bins + 1).skip(1) {
        let f = k as f64 * bin_hz;
        if f < 20.0 || f > hi_edge {
            continue;
        }
        num += f * p;
        den += p;
        if f < 250.0 {
            low += p;
        } else if f < 4000.0 {
            mid += p;
        } else {
            high += p;
        }
        if p > peak_power {
            peak_power = p;
            peak_bin = k;
        }
    }
    if den <= 0.0 {
        return Spectral::empty();
    }
    let total = low + mid + high;
    let pct = |x: f64| if total > 0.0 { 100.0 * x / total } else { 0.0 };
    Spectral {
        centroid_hz: Some(num / den),
        dominant_hz: Some(peak_bin as f64 * bin_hz),
        low_pct: Some(pct(low)),
        mid_pct: Some(pct(mid)),
        high_pct: Some(pct(high)),
    }
}

fn hann(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (n as f64 - 1.0)).cos()))
        .collect()
}

/// In-place iterative radix-2 Cooley–Tukey FFT. `re`/`im` must be a power of two.
fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    assert!(n.is_power_of_two(), "fft length must be a power of two");
    // bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2usize;
    while len <= n {
        let ang = -2.0 * std::f64::consts::PI / len as f64;
        let (wl_re, wl_im) = (ang.cos(), ang.sin());
        let half = len / 2;
        let mut i = 0usize;
        while i < n {
            let (mut w_re, mut w_im) = (1.0f64, 0.0f64);
            for k in 0..half {
                let a = i + k;
                let b = i + k + half;
                let v_re = re[b] * w_re - im[b] * w_im;
                let v_im = re[b] * w_im + im[b] * w_re;
                re[b] = re[a] - v_re;
                im[b] = im[a] - v_im;
                re[a] += v_re;
                im[a] += v_im;
                let nw_re = w_re * wl_re - w_im * wl_im;
                let nw_im = w_re * wl_im + w_im * wl_re;
                w_re = nw_re;
                w_im = nw_im;
            }
            i += len;
        }
        len <<= 1;
    }
}

// ---- integrated loudness (ITU-R BS.1770-4) ----------------------------------

/// A single second-order section, applied via Direct-Form II transposed.
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

impl Biquad {
    fn apply(&self, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0f64; x.len()];
        let (mut z1, mut z2) = (0.0f64, 0.0f64);
        for (xi, yi) in x.iter().zip(y.iter_mut()) {
            let out = self.b0 * xi + z1;
            z1 = self.b1 * xi - self.a1 * out + z2;
            z2 = self.b2 * xi - self.a2 * out;
            *yi = out;
        }
        y
    }
}

/// The two BS.1770 K-weighting stages at 48 kHz: a high-shelf "pre-filter"
/// followed by a ~38 Hz high-pass (RLB).
fn k_weight(channel: &[f64]) -> Vec<f64> {
    let stage1 = Biquad {
        b0: 1.53512485958697,
        b1: -2.69169618940638,
        b2: 1.19839281085285,
        a1: -1.69065929318241,
        a2: 0.73248077421585,
    };
    let stage2 = Biquad {
        b0: 1.0,
        b1: -2.0,
        b2: 1.0,
        a1: -1.99004745483398,
        a2: 0.99007225036621,
    };
    stage2.apply(&stage1.apply(channel))
}

/// Gated integrated loudness in LUFS, or None if there is less than one 400 ms
/// gating block of audio.
fn integrated_loudness(
    samples: &[f64],
    channels: usize,
    frames: usize,
    sample_rate: f64,
) -> Option<f64> {
    let block = (0.4 * sample_rate).round() as usize; // 400 ms
    let step = (0.1 * sample_rate).round() as usize; // 100 ms (75% overlap)
    if block == 0 || frames < block {
        return None;
    }
    // De-interleave then K-weight each channel.
    let weighted: Vec<Vec<f64>> = (0..channels)
        .map(|c| {
            let ch: Vec<f64> = (0..frames)
                .map(|f| {
                    let s = samples[f * channels + c];
                    if s.is_finite() {
                        s
                    } else {
                        0.0
                    }
                })
                .collect();
            k_weight(&ch)
        })
        .collect();

    // Per-block mean-square per channel, and the block loudness.
    let mut block_ms: Vec<Vec<f64>> = Vec::new(); // [block][channel] mean-square
    let mut block_l: Vec<f64> = Vec::new();
    let mut start = 0usize;
    while start + block <= frames {
        let mut ms = vec![0.0f64; channels];
        for (c, m) in ms.iter_mut().enumerate() {
            let mut s = 0.0f64;
            for &v in &weighted[c][start..start + block] {
                s += v * v;
            }
            *m = s / block as f64;
        }
        let sum: f64 = ms.iter().sum();
        block_l.push(loudness_from_ms(sum));
        block_ms.push(ms);
        start += step;
    }
    if block_l.is_empty() {
        return None;
    }

    // Absolute gate at -70 LUFS.
    let abs_gated: Vec<usize> = (0..block_l.len()).filter(|&i| block_l[i] > -70.0).collect();
    if abs_gated.is_empty() {
        return None;
    }

    // Relative gate: mean loudness of abs-gated blocks minus 10 LU.
    let mean_ms_abs = mean_ms(&block_ms, &abs_gated, channels);
    let relative_gate = loudness_from_ms(mean_ms_abs.iter().sum()) - 10.0;
    let rel_gated: Vec<usize> = abs_gated
        .into_iter()
        .filter(|&i| block_l[i] > relative_gate)
        .collect();
    let gated = if rel_gated.is_empty() {
        return Some(loudness_from_ms(mean_ms_abs.iter().sum()));
    } else {
        rel_gated
    };

    let mean_ms_rel = mean_ms(&block_ms, &gated, channels);
    Some(loudness_from_ms(mean_ms_rel.iter().sum()))
}

/// BS.1770 block loudness from the (channel-weighted) sum of mean squares. All
/// channels use weight 1.0 (stereo/mono; no surround weighting).
fn loudness_from_ms(sum_mean_square: f64) -> f64 {
    if sum_mean_square <= 0.0 {
        return f64::NEG_INFINITY;
    }
    -0.691 + 10.0 * sum_mean_square.log10()
}

/// Mean of per-channel mean-squares across the given block indices.
fn mean_ms(block_ms: &[Vec<f64>], idx: &[usize], channels: usize) -> Vec<f64> {
    let mut acc = vec![0.0f64; channels];
    for &i in idx {
        for (c, a) in acc.iter_mut().enumerate() {
            *a += block_ms[i][c];
        }
    }
    let d = idx.len().max(1) as f64;
    for a in acc.iter_mut() {
        *a /= d;
    }
    acc
}

// ---- short-term loudness, loudness range (EBU Tech 3342), true peak ---------

/// Per-window K-weighted block loudness (LUFS) for a given window/hop in seconds.
/// Shared by the momentary (400 ms / 100 ms) and short-term (3 s / 1 s) metrics.
fn block_loudness_series(
    samples: &[f64],
    channels: usize,
    frames: usize,
    sample_rate: f64,
    window_s: f64,
    hop_s: f64,
) -> Vec<f64> {
    let block = (window_s * sample_rate).round() as usize;
    let step = (hop_s * sample_rate).round() as usize;
    if block == 0 || step == 0 || frames < block {
        return Vec::new();
    }
    let weighted: Vec<Vec<f64>> = (0..channels)
        .map(|c| {
            let ch: Vec<f64> = (0..frames)
                .map(|f| {
                    let s = samples[f * channels + c];
                    if s.is_finite() {
                        s
                    } else {
                        0.0
                    }
                })
                .collect();
            k_weight(&ch)
        })
        .collect();
    let mut out = Vec::new();
    let mut start = 0usize;
    while start + block <= frames {
        let mut sum_ms = 0.0f64;
        for w in &weighted {
            let mut s = 0.0f64;
            for &v in &w[start..start + block] {
                s += v * v;
            }
            sum_ms += s / block as f64;
        }
        out.push(loudness_from_ms(sum_ms));
        start += step;
    }
    out
}

/// Loudness for the linear-energy mean of the given block loudnesses (inverse of
/// `loudness_from_ms`, average, forward) — used for the EBU 3342 relative gate.
fn energy_mean_loudness(ls: &[f64]) -> f64 {
    if ls.is_empty() {
        return f64::NEG_INFINITY;
    }
    let sum: f64 = ls
        .iter()
        .map(|&l| 10f64.powf((l + 0.691) / 10.0))
        .sum();
    loudness_from_ms(sum / ls.len() as f64)
}

/// Percentile (0..100) of a pre-sorted ascending slice, linearly interpolated.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    match sorted.len() {
        0 => f64::NAN,
        1 => sorted[0],
        n => {
            let rank = (p / 100.0) * (n - 1) as f64;
            let lo = rank.floor() as usize;
            let hi = rank.ceil() as usize;
            sorted[lo] + (sorted[hi] - sorted[lo]) * (rank - lo as f64)
        }
    }
}

/// Maximum momentary (400 ms) loudness, LUFS.
fn momentary_max(samples: &[f64], channels: usize, frames: usize, sample_rate: f64) -> Option<f64> {
    block_loudness_series(samples, channels, frames, sample_rate, 0.4, 0.1)
        .into_iter()
        .filter(|v| v.is_finite())
        .fold(None, |m: Option<f64>, v| Some(m.map_or(v, |x| x.max(v))))
}

/// Loudness range (EBU Tech 3342) and the max short-term (3 s) loudness, from the
/// short-term loudness distribution. LRA = P95 − P10 of the gated short-term
/// values (absolute gate −70 LUFS, relative gate −20 LU below their energy mean).
fn short_term_stats(
    samples: &[f64],
    channels: usize,
    frames: usize,
    sample_rate: f64,
) -> (Option<f64>, Option<f64>) {
    let series = block_loudness_series(samples, channels, frames, sample_rate, 3.0, 1.0);
    let finite: Vec<f64> = series.into_iter().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return (None, None);
    }
    let st_max = finite.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Absolute gate at −70 LUFS, then a relative gate 20 LU below the energy mean.
    let abs_gated: Vec<f64> = finite.iter().cloned().filter(|&v| v > -70.0).collect();
    if abs_gated.len() < 2 {
        return (None, Some(st_max));
    }
    let rel_gate = energy_mean_loudness(&abs_gated) - 20.0;
    let mut gated: Vec<f64> = abs_gated.into_iter().filter(|&v| v > rel_gate).collect();
    if gated.len() < 2 {
        return (None, Some(st_max));
    }
    gated.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lra = percentile(&gated, 95.0) - percentile(&gated, 10.0);
    (Some(lra), Some(st_max))
}

/// True peak in dBTP via 4x oversampling (BS.1770-4 style): each channel is
/// interpolated with a windowed-sinc polyphase kernel and the largest
/// inter-sample magnitude across all channels is taken. Can exceed 0 dBTP.
fn true_peak_dbtp(samples: &[f64], channels: usize, frames: usize) -> Option<f64> {
    if frames == 0 {
        return None;
    }
    const OS: usize = 4; // oversample factor
    const HALF: usize = 6; // kernel half-width in input samples
    const TAPS: usize = 2 * HALF;

    // One kernel per fractional phase p/OS (p in 1..OS). Phase 0 is the exact
    // sample. Each kernel is a Blackman-windowed sinc, normalised to unity gain.
    let mut kernels = vec![vec![0.0f64; TAPS]; OS];
    for (p, ker) in kernels.iter_mut().enumerate() {
        let frac = p as f64 / OS as f64;
        let mut sum = 0.0f64;
        for (j, k) in ker.iter_mut().enumerate() {
            let x = (j as f64 - (HALF as f64 - 1.0)) - frac;
            let sinc = if x.abs() < 1e-9 {
                1.0
            } else {
                (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
            };
            let a = 2.0 * std::f64::consts::PI * j as f64 / (TAPS - 1) as f64;
            let w = 0.42 - 0.5 * a.cos() + 0.08 * (2.0 * a).cos(); // Blackman
            *k = sinc * w;
            sum += *k;
        }
        if sum.abs() > 1e-12 {
            for k in ker.iter_mut() {
                *k /= sum;
            }
        }
    }

    let mut peak = 0.0f64;
    for c in 0..channels {
        for i in 0..frames {
            // Phase 0: the sample itself.
            let s = samples[i * channels + c];
            let a = if s.is_finite() { s.abs() } else { 0.0 };
            if a > peak {
                peak = a;
            }
            // Phases 1..OS: interpolated inter-sample points.
            for ker in kernels.iter().skip(1) {
                let mut acc = 0.0f64;
                for (j, &k) in ker.iter().enumerate() {
                    let idx = i as isize - (HALF as isize - 1) + j as isize;
                    if idx >= 0 && (idx as usize) < frames {
                        let v = samples[idx as usize * channels + c];
                        if v.is_finite() {
                            acc += v * k;
                        }
                    }
                }
                let a = acc.abs();
                if a > peak {
                    peak = a;
                }
            }
        }
    }
    Some(to_dbfs(peak))
}

// ---- WAV decoding (for reading rendered/processed audio back) --------------

/// Decode a canonical PCM / IEEE-float WAV into interleaved f64 samples.
/// Returns `(interleaved_samples, channels, sample_rate)`. Used to read back the
/// temporary file produced by an offline render for post-FX analysis.
pub fn parse_wav(bytes: &[u8]) -> Result<(Vec<f64>, usize, f64), String> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".to_string());
    }
    let u16le = |o: usize| u16::from_le_bytes([bytes[o], bytes[o + 1]]);
    let u32le = |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);

    let mut fmt: Option<(u16, u16, u32, u16)> = None; // (format, channels, srate, bits)
    let mut data: Option<(usize, usize)> = None; // (offset, len)
    let mut pos = 12usize;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32le(pos + 4) as usize;
        let body = pos + 8;
        let avail = bytes.len().saturating_sub(body);
        if id == b"fmt " && size >= 16 && avail >= 16 {
            let mut format = u16le(body);
            let channels = u16le(body + 2);
            let srate = u32le(body + 4);
            let bits = u16le(body + 14);
            // WAVE_FORMAT_EXTENSIBLE: the real format code is the SubFormat GUID head.
            if format == 0xFFFE && size >= 40 && avail >= 26 {
                format = u16le(body + 24);
            }
            fmt = Some((format, channels, srate, bits));
        } else if id == b"data" {
            data = Some((body, size.min(avail)));
        }
        // Chunks are word-aligned (padded to an even length). Saturate so a
        // bogus size field can never overflow (and the loop always terminates).
        pos = body.saturating_add(size).saturating_add(size & 1);
    }

    let (format, channels, srate, bits) = fmt.ok_or("missing fmt chunk")?;
    if channels == 0 {
        return Err("WAV declares zero channels".to_string());
    }
    let (doff, dlen) = data.ok_or("missing data chunk")?;
    let samples = decode_pcm(&bytes[doff..doff + dlen], format, bits)?;
    Ok((samples, channels as usize, srate as f64))
}

fn decode_pcm(d: &[u8], format: u16, bits: u16) -> Result<Vec<f64>, String> {
    let mut out = Vec::new();
    match (format, bits) {
        (1, 8) => out.extend(d.iter().map(|&b| (b as f64 - 128.0) / 128.0)),
        (1, 16) => out.extend(
            d.chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]) as f64 / 32768.0),
        ),
        (1, 24) => out.extend(d.chunks_exact(3).map(|c| {
            let v = (c[0] as i32) | ((c[1] as i32) << 8) | ((c[2] as i32) << 16);
            ((v << 8) >> 8) as f64 / 8_388_608.0 // sign-extend 24 -> 32
        })),
        (1, 32) => out.extend(
            d.chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64 / 2_147_483_648.0),
        ),
        (3, 32) => out.extend(
            d.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64),
        ),
        (3, 64) => out.extend(
            d.chunks_exact(8)
                .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])),
        ),
        _ => return Err(format!("unsupported WAV format {format} / {bits}-bit")),
    }
    Ok(out)
}

/// Encode interleaved f64 samples as a canonical 16-bit PCM WAV. Used to hand a
/// portable audio clip to audio-capable models (they accept plain WAV, not the
/// 32-bit-float WAV our render produces). Samples are clamped to [-1, 1].
pub fn encode_pcm16_wav(samples: &[f64], channels: usize, sample_rate: f64) -> Vec<u8> {
    let channels = channels.max(1) as u16;
    let sr = if sample_rate > 0.0 {
        sample_rate as u32
    } else {
        48_000
    };
    let bits: u16 = 16;
    let block_align = channels * (bits / 8);
    let byte_rate = sr * block_align as u32;
    let data_len = (samples.len() * 2) as u32; // 2 bytes per sample
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sr.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        let v = if s.is_finite() { s.clamp(-1.0, 1.0) } else { 0.0 };
        let i = (v * 32767.0).round() as i16;
        out.extend_from_slice(&i.to_le_bytes());
    }
    out
}

// ---- time-series (over-time) analysis ---------------------------------------
// Aggregate `analyze()` collapses the whole take to scalars. `analyze_timeline`
// keeps the TIME axis: a per-window level envelope (for silence detection),
// transient onsets (spectral flux), and a single frequency's level over time.

/// One window of the level envelope. `t` is absolute timeline seconds.
#[derive(Debug, Clone, Serialize)]
pub struct LevelPoint {
    pub t: f64,
    pub rms_db: f64,
    pub peak_db: f64,
}

/// A stretch of audio below the silence threshold. Absolute timeline seconds.
#[derive(Debug, Clone, Serialize)]
pub struct SilentRegion {
    pub start: f64,
    pub end: f64,
    pub duration: f64,
}

/// One point of a single-frequency level track.
#[derive(Debug, Clone, Serialize)]
pub struct FreqPoint {
    pub t: f64,
    pub level_db: f64,
}

/// A single frequency's level over time.
#[derive(Debug, Clone, Serialize)]
pub struct FreqTrack {
    pub target_hz: f64,
    /// The actual FFT bin centre tracked (nearest bin to `target_hz`).
    pub bin_hz: f64,
    pub peak_level_db: f64,
    pub peak_time: Option<f64>,
    pub series: Vec<FreqPoint>,
}

/// Result of [`analyze_timeline`].
#[derive(Debug, Clone, Serialize)]
pub struct TimelineAnalysis {
    pub window_seconds: f64,
    pub hop_seconds: f64,
    /// Number of envelope windows analysed (before any decimation of `envelope`).
    pub windows: usize,
    pub envelope: Vec<LevelPoint>,
    pub silent_regions: Vec<SilentRegion>,
    pub loudest_time: Option<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub transients: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency: Option<FreqTrack>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// Parameters for [`analyze_timeline`].
pub struct TimelineOpts {
    pub window_ms: f64,
    pub silence_threshold_db: f64,
    pub min_silence_ms: f64,
    pub detect_transients: bool,
    pub target_hz: Option<f64>,
    /// Added to every reported time so results are absolute timeline seconds.
    pub time_offset: f64,
    /// Cap on each returned series length (envelope, frequency) — uniformly
    /// decimated past this so a long analysis doesn't bloat the JSON result.
    pub max_points: usize,
}

/// Time-series analysis of `samples` (interleaved, `channels` per frame). Pure
/// and host-independent, like [`analyze`].
pub fn analyze_timeline(
    samples: &[f64],
    channels: usize,
    sample_rate: f64,
    opts: &TimelineOpts,
) -> TimelineAnalysis {
    let channels = channels.max(1);
    let sr = if sample_rate > 0.0 { sample_rate } else { 48_000.0 };
    let frames = samples.len() / channels;
    let mono = downmix(samples, channels, frames);

    let win = ((opts.window_ms.max(1.0) * sr / 1000.0).round() as usize).max(1);
    let hop = (win / 2).max(1);

    // ---- per-window RMS/peak envelope + loudest window ----
    let mut env: Vec<LevelPoint> = Vec::new();
    let mut loudest: Option<(f64, f64)> = None; // (rms_db, t)
    let mut start = 0usize;
    while start < mono.len() {
        let end = (start + win).min(mono.len());
        let slice = &mono[start..end];
        let (mut ss, mut pk) = (0.0f64, 0.0f64);
        for &x in slice {
            let x = if x.is_finite() { x } else { 0.0 };
            ss += x * x;
            let a = x.abs();
            if a > pk {
                pk = a;
            }
        }
        let rms_db = to_dbfs((ss / slice.len() as f64).sqrt());
        let t = opts.time_offset + start as f64 / sr;
        if loudest.map_or(true, |(r, _)| rms_db > r) {
            loudest = Some((rms_db, t));
        }
        env.push(LevelPoint { t, rms_db, peak_db: to_dbfs(pk) });
        start += hop;
    }

    // ---- silent regions: runs of windows below the threshold ----
    let min_sil = opts.min_silence_ms.max(0.0) / 1000.0;
    let mut silent_regions: Vec<SilentRegion> = Vec::new();
    let mut run_start: Option<f64> = None;
    let push_region = |v: &mut Vec<SilentRegion>, s: f64, e: f64| {
        if e - s >= min_sil {
            v.push(SilentRegion { start: s, end: e, duration: e - s });
        }
    };
    for p in &env {
        if p.rms_db < opts.silence_threshold_db {
            run_start.get_or_insert(p.t);
        } else if let Some(s) = run_start.take() {
            push_region(&mut silent_regions, s, p.t);
        }
    }
    if let Some(s) = run_start.take() {
        push_region(&mut silent_regions, s, opts.time_offset + frames as f64 / sr);
    }

    // ---- STFT: transients (spectral flux) and/or single-frequency tracking ----
    let mut transients: Vec<f64> = Vec::new();
    let mut frequency: Option<FreqTrack> = None;
    let mut note: Option<String> = None;
    if opts.detect_transients || opts.target_hz.is_some() {
        if mono.len() >= FFT_SIZE {
            let window = hann(FFT_SIZE);
            let sh = (FFT_SIZE / 4).max(1); // ~21 ms hop at 48 kHz
            let bin_hz = sr / FFT_SIZE as f64;
            let half = FFT_SIZE / 2;
            let target_bin = opts
                .target_hz
                .map(|hz| ((hz / bin_hz).round() as usize).clamp(1, half));
            let mut freq_series: Vec<FreqPoint> = Vec::new();
            let mut freq_peak: Option<(f64, f64)> = None;
            let mut flux: Vec<(f64, f64)> = Vec::new();
            let mut prev_mag: Option<Vec<f64>> = None;
            let mut s = 0usize;
            while s + FFT_SIZE <= mono.len() {
                let mut re = vec![0.0f64; FFT_SIZE];
                let mut im = vec![0.0f64; FFT_SIZE];
                for i in 0..FFT_SIZE {
                    re[i] = mono[s + i] * window[i];
                }
                fft(&mut re, &mut im);
                let t = opts.time_offset + (s + FFT_SIZE / 2) as f64 / sr; // window centre
                let mag: Vec<f64> =
                    (0..=half).map(|k| (re[k] * re[k] + im[k] * im[k]).sqrt()).collect();
                if let Some(tb) = target_bin {
                    // Take the PEAK of a ±1-bin neighbourhood (so a tone landing
                    // between bins still reads its true level), scaled to amplitude
                    // (Hann coherent gain 0.5 -> factor 4/N): a bin-centred
                    // full-scale tone reads ~0 dBFS. (A mean/3 here underreads by
                    // ~3.5 dB.)
                    let lo = tb.saturating_sub(1);
                    let hi = (tb + 1).min(half);
                    let m = mag[lo..=hi].iter().copied().fold(0.0_f64, f64::max);
                    let level = to_dbfs(4.0 * m / FFT_SIZE as f64);
                    if freq_peak.map_or(true, |(l, _)| level > l) {
                        freq_peak = Some((level, t));
                    }
                    freq_series.push(FreqPoint { t, level_db: level });
                }
                if opts.detect_transients {
                    if let Some(pm) = &prev_mag {
                        let f: f64 = mag.iter().zip(pm).map(|(&m, &p)| (m - p).max(0.0)).sum();
                        flux.push((t, f));
                    }
                    prev_mag = Some(mag);
                }
                s += sh;
            }
            // Peak-pick the spectral flux above an adaptive threshold, min 50 ms apart.
            if opts.detect_transients && !flux.is_empty() {
                let vals: Vec<f64> = flux.iter().map(|&(_, f)| f).collect();
                let mean = vals.iter().sum::<f64>() / vals.len() as f64;
                let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
                let thresh = mean + 1.5 * var.sqrt();
                let mut last_t = f64::NEG_INFINITY;
                for i in 0..flux.len() {
                    let (t, f) = flux[i];
                    let is_peak = f > thresh
                        && (i == 0 || flux[i - 1].1 <= f)
                        && (i + 1 >= flux.len() || flux[i + 1].1 < f);
                    if is_peak && t - last_t >= 0.05 {
                        transients.push(t);
                        last_t = t;
                    }
                }
            }
            if let Some(tb) = target_bin {
                frequency = Some(FreqTrack {
                    target_hz: opts.target_hz.unwrap_or(0.0),
                    bin_hz: tb as f64 * bin_hz,
                    peak_level_db: freq_peak.map(|(l, _)| l).unwrap_or(DBFS_FLOOR),
                    peak_time: freq_peak.map(|(_, t)| t),
                    series: decimate(freq_series, opts.max_points),
                });
            }
        } else {
            note = Some(
                "audio too short for transient/frequency analysis (need at least one FFT window)"
                    .into(),
            );
        }
    }

    let windows = env.len();
    TimelineAnalysis {
        window_seconds: win as f64 / sr,
        hop_seconds: hop as f64 / sr,
        windows,
        envelope: decimate(env, opts.max_points),
        silent_regions,
        loudest_time: loudest.map(|(_, t)| t),
        transients,
        frequency,
        note,
    }
}

/// Uniformly downsample a series to at most `max` points (keeps the overall
/// shape). `max == 0` means no cap.
fn decimate<T>(v: Vec<T>, max: usize) -> Vec<T> {
    if max == 0 || v.len() <= max {
        return v;
    }
    let stride = v.len().div_ceil(max);
    v.into_iter().step_by(stride).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn sine(freq: f64, amp: f64, secs: f64, sr: f64) -> Vec<f64> {
        let n = (secs * sr) as usize;
        (0..n)
            .map(|i| amp * (2.0 * PI * freq * i as f64 / sr).sin())
            .collect()
    }

    #[test]
    fn silence_reads_as_silent() {
        let f = analyze(&vec![0.0; 48000], 1, 48000.0);
        assert!(f.silent);
        assert!(f.peak_dbfs <= DBFS_FLOOR + 1.0);
        assert_eq!(f.clip_count, 0);
        assert!(f.integrated_lufs.is_none() || f.integrated_lufs.unwrap() < -60.0);
    }

    fn timeline_opts() -> TimelineOpts {
        TimelineOpts {
            window_ms: 50.0,
            silence_threshold_db: -60.0,
            min_silence_ms: 200.0,
            detect_transients: false,
            target_hz: None,
            time_offset: 0.0,
            max_points: 512,
        }
    }

    #[test]
    fn timeline_detects_silent_gap() {
        let sr = 48000.0;
        // tone (0.5s) | silence (0.5s) | tone (0.5s)
        let mut sig = sine(440.0, 0.5, 0.5, sr);
        sig.extend(std::iter::repeat(0.0).take((0.5 * sr) as usize));
        sig.extend(sine(440.0, 0.5, 0.5, sr));
        let a = analyze_timeline(&sig, 1, sr, &timeline_opts());
        let gap = a
            .silent_regions
            .iter()
            .find(|r| r.start > 0.4 && r.start < 0.6)
            .expect("should find the silent gap near 0.5s");
        assert!(gap.end > 0.95 && gap.end < 1.05, "gap end ~1.0s, got {}", gap.end);
        assert!(gap.duration >= 0.2);
    }

    #[test]
    fn timeline_tracks_target_frequency() {
        let sr = 48000.0;
        let sig = sine(1000.0, 0.8, 1.0, sr);
        let opts = TimelineOpts { target_hz: Some(1000.0), ..timeline_opts() };
        let a = analyze_timeline(&sig, 1, sr, &opts);
        let f = a.frequency.expect("frequency track present");
        assert!((f.bin_hz - 1000.0).abs() < 15.0, "tracked bin ~1000Hz, got {}", f.bin_hz);
        // Peak-bin scaling: a 0.8-amplitude tone reads near -2 dBFS, not ~-20.
        assert!(f.peak_level_db > -6.0, "a strong tone should read near full, got {}", f.peak_level_db);
        assert!(!f.series.is_empty());
    }

    #[test]
    fn timeline_detects_transient_onset() {
        let sr = 48000.0;
        // 0.3s silence, then a sudden 1kHz tone onset.
        let mut sig = vec![0.0f64; (0.3 * sr) as usize];
        sig.extend(sine(1000.0, 1.0, 0.2, sr));
        let opts = TimelineOpts { detect_transients: true, ..timeline_opts() };
        let a = analyze_timeline(&sig, 1, sr, &opts);
        assert!(!a.transients.is_empty(), "should detect the onset");
        assert!(
            a.transients.iter().any(|&t| t > 0.25 && t < 0.45),
            "an onset near 0.3s, got {:?}",
            a.transients
        );
    }

    #[test]
    fn full_scale_sine_peaks_near_zero_dbfs() {
        let f = analyze(&sine(1000.0, 1.0, 1.0, 48000.0), 1, 48000.0);
        assert!((f.peak_dbfs - 0.0).abs() < 0.2, "peak {}", f.peak_dbfs);
        // A sine's RMS is -3.01 dB below its peak.
        assert!((f.rms_dbfs + 3.01).abs() < 0.3, "rms {}", f.rms_dbfs);
        assert!(!f.silent);
    }

    #[test]
    fn true_peak_at_or_above_sample_peak() {
        // A 1 kHz sine at -6 dBFS: the oversampled true peak is at/above the sample
        // peak and close to the sine's amplitude (~-6 dBTP).
        let f = analyze(&sine(1000.0, 0.5, 1.0, 48000.0), 1, 48000.0);
        let tp = f.true_peak_dbtp.expect("true peak");
        assert!(tp >= f.peak_dbfs - 0.05, "tp {tp} vs peak {}", f.peak_dbfs);
        assert!((tp + 6.02).abs() < 0.6, "tp {tp}");
    }

    #[test]
    fn steady_tone_has_near_zero_loudness_range() {
        // A constant-level tone has essentially no loudness range.
        let f = analyze(&sine(1000.0, 0.5, 10.0, 48000.0), 1, 48000.0);
        let lra = f.loudness_range_lu.expect("lra");
        assert!(lra.abs() < 1.0, "lra {lra}");
        assert!(f.short_term_lufs_max.is_some());
        assert!(f.momentary_lufs_max.is_some());
    }

    #[test]
    fn pcm16_wav_round_trips() {
        // Encode a stereo tone to PCM16 WAV and decode it back; values survive
        // within 16-bit quantisation.
        let mut interleaved = Vec::new();
        let l = sine(440.0, 0.5, 0.1, 48000.0);
        for &s in &l {
            interleaved.push(s);
            interleaved.push(s * 0.5);
        }
        let wav = encode_pcm16_wav(&interleaved, 2, 48000.0);
        let (back, ch, sr) = parse_wav(&wav).expect("decode");
        assert_eq!(ch, 2);
        assert_eq!(sr, 48000.0);
        assert_eq!(back.len(), interleaved.len());
        for (a, b) in interleaved.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1.0 / 32000.0, "{a} vs {b}");
        }
    }

    #[test]
    fn clipping_is_detected() {
        let mut s = sine(1000.0, 1.2, 0.2, 48000.0); // over full scale
        for x in s.iter_mut() {
            *x = x.clamp(-1.0, 1.0);
        }
        let f = analyze(&s, 1, 48000.0);
        assert!(f.clipping);
        assert!(f.clip_count > 0);
    }

    #[test]
    fn dc_offset_is_measured() {
        let f = analyze(&vec![0.25; 48000], 1, 48000.0);
        assert!((f.dc_offset - 0.25).abs() < 1e-9);
    }

    #[test]
    fn spectral_centroid_tracks_a_tone() {
        let f = analyze(&sine(1000.0, 0.5, 1.0, 48000.0), 1, 48000.0);
        let c = f.spectral_centroid_hz.expect("centroid");
        assert!((c - 1000.0).abs() < 60.0, "centroid {c}");
        let d = f.dominant_frequency_hz.expect("dominant");
        assert!((d - 1000.0).abs() < 20.0, "dominant {d}");
        // Almost all energy sits in the mid band for a 1 kHz tone.
        assert!(f.band_mid_pct.unwrap() > 80.0);
    }

    #[test]
    fn short_signal_has_no_loudness_or_spectrum() {
        let f = analyze(&sine(1000.0, 0.5, 0.05, 48000.0), 1, 48000.0); // 50 ms
        assert!(f.integrated_lufs.is_none());
        assert!(f.spectral_centroid_hz.is_none());
    }

    #[test]
    fn fft_matches_naive_dft() {
        let n = 64;
        let input: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 3.0 * i as f64 / n as f64).sin() + 0.5)
            .collect();
        let mut re = input.clone();
        let mut im = vec![0.0; n];
        fft(&mut re, &mut im);
        for k in 0..n {
            let (mut dr, mut di) = (0.0f64, 0.0f64);
            for (t, &x) in input.iter().enumerate() {
                let ang = -2.0 * PI * (k * t) as f64 / n as f64;
                dr += x * ang.cos();
                di += x * ang.sin();
            }
            assert!((re[k] - dr).abs() < 1e-6, "re[{k}]");
            assert!((im[k] - di).abs() < 1e-6, "im[{k}]");
        }
    }

    #[test]
    fn loudness_of_minus_20_dbfs_tone_is_about_minus_20_lufs() {
        // A 1 kHz tone at -20 dBFS peak (amp 0.1) sits near 0 dB K-weighting,
        // so its integrated loudness should be roughly its RMS level (~-23 LUFS).
        let f = analyze(&sine(1000.0, 0.1, 2.0, 48000.0), 1, 48000.0);
        let lufs = f.integrated_lufs.expect("lufs");
        assert!((-27.0..-19.0).contains(&lufs), "lufs {lufs}");
    }

    #[test]
    fn stereo_interleaving_is_handled() {
        // Left full-scale tone, right silent: peak still ~0 dBFS, mono downmix -6 dB.
        let left = sine(1000.0, 1.0, 1.0, 48000.0);
        let mut inter = Vec::with_capacity(left.len() * 2);
        for &x in &left {
            inter.push(x);
            inter.push(0.0);
        }
        let f = analyze(&inter, 2, 48000.0);
        assert_eq!(f.channels, 2);
        assert!((f.peak_dbfs - 0.0).abs() < 0.2, "peak {}", f.peak_dbfs);
    }

    #[test]
    fn parse_wav_pcm16_stereo_roundtrips() {
        let pcm: [i16; 4] = [16384, -16384, 32767, 0]; // interleaved L,R,L,R
        let sr: u32 = 48000;
        let (ch, bits): (u16, u16) = (2, 16);
        let block_align = ch * bits / 8;
        let data: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
        let mut w = Vec::new();
        w.extend_from_slice(b"RIFF");
        w.extend_from_slice(&((36 + data.len()) as u32).to_le_bytes());
        w.extend_from_slice(b"WAVE");
        w.extend_from_slice(b"fmt ");
        w.extend_from_slice(&16u32.to_le_bytes());
        w.extend_from_slice(&1u16.to_le_bytes()); // PCM
        w.extend_from_slice(&ch.to_le_bytes());
        w.extend_from_slice(&sr.to_le_bytes());
        w.extend_from_slice(&(sr * block_align as u32).to_le_bytes());
        w.extend_from_slice(&block_align.to_le_bytes());
        w.extend_from_slice(&bits.to_le_bytes());
        w.extend_from_slice(b"data");
        w.extend_from_slice(&(data.len() as u32).to_le_bytes());
        w.extend_from_slice(&data);

        let (s, c, r) = parse_wav(&w).expect("parse");
        assert_eq!(c, 2);
        assert_eq!(r, 48000.0);
        assert_eq!(s.len(), 4);
        assert!((s[0] - 0.5).abs() < 1e-3);
        assert!((s[1] + 0.5).abs() < 1e-3);
        assert!((s[2] - 1.0).abs() < 1e-3);
        assert!((s[3]).abs() < 1e-6);
    }

    #[test]
    fn parse_wav_rejects_garbage() {
        assert!(parse_wav(b"not a wav file at all").is_err());
    }
}
