/*
 * keydropper — browser port of the acoustic keystroke pipeline.
 *
 * Blue team: a compact keystroke recognizer used purely to measure leakage and to
 * prove the interference (masking) countermeasure. Faithful to the Python reference,
 * including the two hard-won fixes:
 *   - the keyboard identity is FIXED (independent of the recording), so a recognizer
 *     can generalize across recordings;
 *   - features are aligned to the transient peak so absolute-position offsets from the
 *     onset detector do not break cross-recording recognition.
 *
 * No external dependencies. Everything is a plain function on a global `KD` object so
 * the page also works when opened directly from disk (file://).
 */
(function (global) {
  "use strict";

  // ---- configuration (browser-fast defaults; see PLAN.md for the full-size rig) ----
  const CFG = {
    audio: { sr: 16000 },
    segment: {
      frameMs: 4, smoothMs: 8, thresholdRatio: 3,
      refractoryMs: 70, windowMs: 120, preOnsetMs: 10,
    },
    feature: { nFft: 256, hopMs: 4, winMs: 16, nMels: 24, fmin: 40, fmax: 8000, logFloor: 1e-10 },
    model: { decoysPerKey: 4, maskGain: 1.2, timingJitterMs: 12 },
    KEYBOARD_SEED: "20240517",
  };

  const DEFAULT_KEYS = "abcdefghijklmnopqrstuvwxyz".split("").concat(["<space>"]);

  // ---------------------------------------------------------------- seeded RNG ----
  function xmur3(str) {
    let h = 1779033703 ^ str.length;
    for (let i = 0; i < str.length; i++) {
      h = Math.imul(h ^ str.charCodeAt(i), 3432918353);
      h = (h << 13) | (h >>> 19);
    }
    return function () {
      h = Math.imul(h ^ (h >>> 16), 2246822507);
      h = Math.imul(h ^ (h >>> 13), 3266489909);
      return (h ^= h >>> 16) >>> 0;
    };
  }
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function Rng(seed) {
    const s = xmur3(String(seed));
    this._next = mulberry32(s());
    this._spare = null;
  }
  Rng.prototype.next = function () { return this._next(); };
  Rng.prototype.uniform = function (a, b) { return a + (b - a) * this._next(); };
  Rng.prototype.int = function (a, b) { return a + Math.floor(this._next() * (b - a + 1)); };
  Rng.prototype.choice = function (arr) { return arr[Math.floor(this._next() * arr.length)]; };
  Rng.prototype.gauss = function (mu, sigma) {
    mu = mu || 0; sigma = sigma === undefined ? 1 : sigma;
    if (this._spare !== null) { const v = this._spare; this._spare = null; return mu + sigma * v; }
    let u = 0, v = 0;
    while (u === 0) u = this._next();
    while (v === 0) v = this._next();
    const mag = Math.sqrt(-2 * Math.log(u));
    this._spare = mag * Math.sin(2 * Math.PI * v);
    return mu + sigma * mag * Math.cos(2 * Math.PI * v);
  };
  Rng.prototype.shuffle = function (arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(this._next() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  };

  // ------------------------------------------------------------------------ DSP ----
  const dsp = {};
  dsp.hann = function (n) {
    const w = new Float64Array(n);
    if (n === 1) { w[0] = 1; return w; }
    for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n);
    return w;
  };
  // In-place iterative radix-2 FFT on separate real/imag arrays. n must be power of 2.
  dsp.fft = function (re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) { [re[i], re[j]] = [re[j], re[i]];[im[i], im[j]] = [im[j], im[i]]; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = (-2 * Math.PI) / len;
      const wr = Math.cos(ang), wi = Math.sin(ang);
      for (let start = 0; start < n; start += len) {
        let cr = 1, ci = 0;
        for (let k = 0; k < len / 2; k++) {
          const i0 = start + k, i1 = i0 + len / 2;
          const tr = re[i1] * cr - im[i1] * ci;
          const ti = re[i1] * ci + im[i1] * cr;
          re[i1] = re[i0] - tr; im[i1] = im[i0] - ti;
          re[i0] += tr; im[i0] += ti;
          const ncr = cr * wr - ci * wi;
          ci = cr * wi + ci * wr; cr = ncr;
        }
      }
    }
  };
  dsp.rfftPower = function (frame, nFft) {
    const re = new Float64Array(nFft), im = new Float64Array(nFft);
    const m = Math.min(frame.length, nFft);
    for (let i = 0; i < m; i++) re[i] = frame[i];
    dsp.fft(re, im);
    const half = nFft / 2 + 1;
    const out = new Float64Array(half);
    for (let i = 0; i < half; i++) out[i] = re[i] * re[i] + im[i] * im[i];
    return out;
  };
  dsp.shortTimeEnergy = function (x, frameLen, hop) {
    const env = [], centers = [];
    const n = x.length;
    for (let start = 0; start < n; start += hop) {
      const end = Math.min(start + frameLen, n);
      let s = 0;
      for (let i = start; i < end; i++) s += x[i] * x[i];
      const count = end - start;
      env.push(count ? Math.sqrt(s / count) : 0);
      centers.push(start + (count >> 1));
    }
    return { env, centers };
  };
  dsp.movingAverage = function (x, k) {
    if (k <= 1) return x.slice();
    const n = x.length, half = k >> 1, out = new Array(n);
    const pref = new Float64Array(n + 1);
    for (let i = 0; i < n; i++) pref[i + 1] = pref[i] + x[i];
    for (let i = 0; i < n; i++) {
      const lo = Math.max(0, i - half), hi = Math.min(n, i + half + 1);
      out[i] = (pref[hi] - pref[lo]) / (hi - lo);
    }
    return out;
  };
  dsp.percentile = function (arr, q) {
    if (!arr.length) return 0;
    const xs = arr.slice().sort((a, b) => a - b);
    if (q <= 0) return xs[0];
    if (q >= 100) return xs[xs.length - 1];
    const pos = (q / 100) * (xs.length - 1);
    const lo = Math.floor(pos), hi = Math.ceil(pos);
    if (lo === hi) return xs[lo];
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo);
  };

  // --------------------------------------------------------------------- synth ----
  const synth = {};
  function keyResonances(key, keyboardSeed) {
    const rng = new Rng(keyboardSeed + ":" + key);
    const res = [];
    for (let i = 0; i < 5; i++) {
      res.push({ freq: rng.uniform(400, 7200), decay: rng.uniform(0.005, 0.03), amp: rng.uniform(0.4, 1.0) });
    }
    return res;
  }
  // One keystroke waveform. rng => per-hit jitter; without it, the canonical template.
  synth.keySignature = function (key, windowMs, rng, keyboardSeed) {
    keyboardSeed = keyboardSeed || CFG.KEYBOARD_SEED;
    const sr = CFG.audio.sr;
    const n = Math.max(1, Math.round((windowMs * sr) / 1000));
    const res = keyResonances(key, keyboardSeed);
    let ampScale = 1, detune = 1, noiseAmp = 0;
    if (rng) { ampScale = rng.uniform(0.9, 1.1); detune = rng.uniform(0.995, 1.005); noiseAmp = rng.uniform(0.002, 0.008); }
    const out = new Float64Array(n);
    const hitAt = Math.floor(0.1 * n), releaseAt = Math.floor(0.55 * n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      const t = (i - hitAt) / sr;
      if (t >= 0) for (const r of res) s += r.amp * Math.exp(-t / r.decay) * Math.sin(2 * Math.PI * r.freq * detune * t);
      const t2 = (i - releaseAt) / sr;
      if (t2 >= 0) for (const r of res) s += 0.4 * r.amp * Math.exp(-t2 / (0.6 * r.decay)) * Math.sin(2 * Math.PI * r.freq * detune * t2);
      out[i] = ampScale * s;
    }
    if (noiseAmp && rng) for (let i = 0; i < n; i++) out[i] += rng.gauss(0, noiseAmp);
    let peak = 0;
    for (let i = 0; i < n; i++) peak = Math.max(peak, Math.abs(out[i]));
    peak = peak || 1;
    for (let i = 0; i < n; i++) out[i] /= peak;
    return out;
  };
  // Playback profiles for the live masking control.  They retain the per-key
  // identity while changing the envelope and resonances enough to sound distinct.
  synth.SAMPLE_PROFILES = {
    mechanical: "Mechanical click",
    laptop: "Laptop tap",
    typewriter: "Typewriter",
    membrane: "Membrane thud",
  };
  synth.keyboardSample = function (key, windowMs, profile) {
    const base = synth.keySignature(key, windowMs);
    const sr = CFG.audio.sr, n = base.length;
    const out = new Float64Array(n);
    profile = synth.SAMPLE_PROFILES[profile] ? profile : "mechanical";
    let peak = 1e-8;
    for (let i = 0; i < n; i++) {
      const t = i / sr;
      // A deterministic noise burst gives a physical attack without making
      // repeated previews of the same sample unexpectedly change.
      const noise = Math.sin((i + 1) * 127.1 + key.charCodeAt(0) * 19.7) * 0.14;
      let v;
      if (profile === "laptop") {
        v = base[i] * Math.exp(-t / 0.014) * 0.55 + noise * Math.exp(-t / 0.004);
      } else if (profile === "typewriter") {
        const bell = Math.sin(2 * Math.PI * 2450 * t) * Math.exp(-t / 0.06) * 0.28;
        v = base[i] * 0.65 + noise * Math.exp(-t / 0.012) + bell;
      } else if (profile === "membrane") {
        const thud = Math.sin(2 * Math.PI * 180 * t) * Math.exp(-t / 0.022) * 0.55;
        v = base[i] * Math.exp(-t / 0.025) * 0.32 + thud + noise * Math.exp(-t / 0.008) * 0.35;
      } else {
        v = base[i] * 0.9 + noise * Math.exp(-t / 0.009) * 0.45;
      }
      out[i] = v;
      peak = Math.max(peak, Math.abs(v));
    }
    for (let i = 0; i < n; i++) out[i] /= peak;
    return out;
  };
  // Render a typing stream. Returns { samples, events:[{onset,key}] }.
  synth.renderStream = function (text, seed, windowMs, opts) {
    opts = opts || {};
    const kps = opts.keysPerSecond || 6;
    const bg = opts.backgroundNoise === undefined ? 0.01 : opts.backgroundNoise;
    const sr = CFG.audio.sr;
    const rng = new Rng(seed);
    const inter = 1 / kps;
    const positions = [];
    let t = 0.2;
    for (let c = 0; c < text.length; c++) { t += inter * rng.uniform(0.6, 1.4); positions.push(Math.floor(t * sr)); }
    const sigLen = Math.round((windowMs * sr) / 1000);
    const total = (positions.length ? positions[positions.length - 1] : Math.floor(0.4 * sr)) + sigLen + Math.floor(0.2 * sr);
    const samples = new Float64Array(total);
    for (let i = 0; i < total; i++) samples[i] = rng.gauss(0, bg);
    const events = [];
    for (let idx = 0; idx < text.length; idx++) {
      const ch = text[idx];
      const key = ch === " " ? "<space>" : ch;
      const sig = synth.keySignature(key, windowMs, rng);
      const pos = positions[idx];
      events.push({ onset: pos + Math.floor(0.1 * sig.length), key });
      for (let i = 0; i < sig.length; i++) { const j = pos + i; if (j >= 0 && j < total) samples[j] += sig[i]; }
    }
    return { samples, events };
  };
  synth.dataset = function (keys, perKey, windowMs, seed) {
    const rng = new Rng(seed);
    const clips = [], labels = [];
    for (const key of keys) for (let i = 0; i < perKey; i++) { clips.push(synth.keySignature(key, windowMs, rng)); labels.push(key); }
    const idx = rng.shuffle(clips.map((_, i) => i));
    return { clips: idx.map((i) => clips[i]), labels: idx.map((i) => labels[i]) };
  };

  // ------------------------------------------------------------------- segment ----
  const segment = {};
  function msToSamples(ms) { return Math.max(1, Math.round((ms * CFG.audio.sr) / 1000)); }
  segment.detectOnsets = function (x) {
    const seg = CFG.segment;
    const frameLen = msToSamples(seg.frameMs), hop = frameLen;
    const { env, centers } = dsp.shortTimeEnergy(x, frameLen, hop);
    if (!env.length) return [];
    let smoothFrames = Math.max(1, Math.round(seg.smoothMs / seg.frameMs));
    if (smoothFrames % 2 === 0) smoothFrames += 1;
    const sm = dsp.movingAverage(env, smoothFrames);
    const noiseFloor = dsp.percentile(sm, 20);
    let peak = 0; for (const v of sm) peak = Math.max(peak, v);
    const floor = Math.max(noiseFloor, 1e-6 * (peak > 0 ? peak : 1));
    const threshold = seg.thresholdRatio * floor;
    const refractory = msToSamples(seg.refractoryMs);
    const onsets = [];
    let last = -refractory - 1, armed = true;
    for (let i = 1; i < sm.length; i++) {
      const rising = sm[i] > sm[i - 1];
      if (armed && sm[i] >= threshold && rising) {
        if (centers[i] - last >= refractory) {
          let j = i, steps = 0;
          while (j + 1 < sm.length && sm[j + 1] >= sm[j] && steps < 8) { j++; steps++; }
          onsets.push(centers[j]); last = centers[j]; armed = false;
        }
      } else if (sm[i] < threshold) armed = true;
    }
    return onsets;
  };
  segment.extractWindows = function (x, onsets) {
    const seg = CFG.segment;
    const win = msToSamples(seg.windowMs), pre = msToSamples(seg.preOnsetMs), n = x.length;
    return onsets.map((onset) => {
      const start = onset - pre, clip = new Float64Array(win);
      for (let k = 0; k < win; k++) { const src = start + k; if (src >= 0 && src < n) clip[k] = x[src]; }
      return clip;
    });
  };
  segment.segment = function (x) { const onsets = segment.detectOnsets(x); return { onsets, clips: segment.extractWindows(x, onsets) }; };
  segment.matchOnsets = function (detected, truth, tol) {
    const used = new Array(truth.length).fill(false);
    let tp = 0;
    const det = detected.slice().sort((a, b) => a - b);
    const tru = truth.slice().sort((a, b) => a - b);
    for (const d of det) {
      let best = -1, bestDist = tol + 1;
      for (let i = 0; i < tru.length; i++) {
        if (used[i]) continue;
        const dist = Math.abs(d - tru[i]);
        if (dist <= tol && dist < bestDist) { bestDist = dist; best = i; }
      }
      if (best >= 0) { used[best] = true; tp++; }
    }
    return { tp, fp: det.length - tp, fn: tru.length - tp };
  };

  // ------------------------------------------------------------------ features ----
  const features = {};
  let _melCache = null, _melKey = "";
  function melFilterbank(sr, nFft, nMels, fmin, fmax) {
    const key = [sr, nFft, nMels, fmin, fmax].join(":");
    if (_melKey === key) return _melCache;
    const nBins = nFft / 2 + 1;
    fmax = Math.min(fmax, sr / 2);
    const hz2mel = (hz) => 2595 * Math.log10(1 + hz / 700);
    const mel2hz = (m) => 700 * (Math.pow(10, m / 2595) - 1);
    const melLo = hz2mel(fmin), melHi = hz2mel(fmax);
    const melPts = [], hzPts = [];
    for (let i = 0; i < nMels + 2; i++) melPts.push(melLo + ((melHi - melLo) * i) / (nMels + 1));
    for (const m of melPts) hzPts.push(mel2hz(m));
    const binFreq = []; for (let i = 0; i < nBins; i++) binFreq.push((i * sr) / nFft);
    const bank = [];
    for (let m = 1; m <= nMels; m++) {
      const left = hzPts[m - 1], center = hzPts[m], right = hzPts[m + 1];
      const row = new Float64Array(nBins);
      for (let b = 0; b < nBins; b++) {
        const f = binFreq[b];
        if (f >= left && f <= center && center > left) row[b] = (f - left) / (center - left);
        else if (f >= center && f <= right && right > center) row[b] = (right - f) / (right - center);
      }
      bank.push(row);
    }
    _melCache = bank; _melKey = key;
    return bank;
  }
  features.melFilterbank = melFilterbank;
  features.logmel = function (clip) {
    const f = CFG.feature, sr = CFG.audio.sr;
    let win = Math.max(1, Math.round((f.winMs * sr) / 1000));
    const hop = Math.max(1, Math.round((f.hopMs * sr) / 1000));
    const nFft = f.nFft;
    if (win > nFft) win = nFft;
    const window = dsp.hann(win);
    const bank = melFilterbank(sr, nFft, f.nMels, f.fmin, f.fmax);
    const out = [];
    for (let start = 0; start + win <= clip.length; start += hop) {
      const wf = new Float64Array(win);
      for (let i = 0; i < win; i++) wf[i] = clip[start + i] * window[i];
      const power = dsp.rfftPower(wf, nFft);
      const row = new Float64Array(f.nMels);
      for (let m = 0; m < f.nMels; m++) {
        const filt = bank[m]; let acc = 0;
        for (let b = 0; b < power.length; b++) { const w = filt[b]; if (w) acc += w * power[b]; }
        row[m] = Math.log(acc + f.logFloor);
      }
      out.push(row);
    }
    if (!out.length) { const r = new Float64Array(f.nMels).fill(Math.log(f.logFloor)); out.push(r); }
    return out;
  };
  features.inferNFrames = function (windowMs) {
    const f = CFG.feature, sr = CFG.audio.sr;
    const win = Math.min(Math.max(1, Math.round((f.winMs * sr) / 1000)), f.nFft);
    const hop = Math.max(1, Math.round((f.hopMs * sr) / 1000));
    const n = Math.max(1, Math.round((windowMs * sr) / 1000));
    if (n < win) return 1;
    return 1 + Math.floor((n - win) / hop);
  };
  features.alignPeak = function (mat, target, nFrames, nMels) {
    if (!mat.length) { const z = []; for (let i = 0; i < nFrames; i++) z.push(new Float64Array(nMels)); return z; }
    let peak = 0, peakE = -Infinity, gFloor = Infinity;
    for (let i = 0; i < mat.length; i++) {
      let s = 0; for (let j = 0; j < mat[i].length; j++) { s += mat[i][j]; if (mat[i][j] < gFloor) gFloor = mat[i][j]; }
      const e = s / mat[i].length; if (e > peakE) { peakE = e; peak = i; }
    }
    const shift = target - peak, out = [];
    for (let fr = 0; fr < nFrames; fr++) {
      const src = fr - shift;
      if (src >= 0 && src < mat.length) out.push(mat[src]);
      else { const row = new Float64Array(nMels).fill(gFloor); out.push(row); }
    }
    return out;
  };
  features.featurize = function (clips, nFrames) {
    const f = CFG.feature, target = Math.max(1, nFrames >> 2), out = [];
    for (const clip of clips) {
      const mat = features.alignPeak(features.logmel(clip), target, nFrames, f.nMels);
      const flat = new Float64Array(nFrames * f.nMels);
      let p = 0; for (const row of mat) for (let j = 0; j < f.nMels; j++) flat[p++] = row[j];
      out.push(flat);
    }
    return out;
  };

  // -------------------------------------------------------------------- models ----
  function KNN(k) { this.k = k || 3; }
  KNN.prototype.fit = function (X, y) {
    this.y = y.slice();
    const d = X[0].length, n = X.length;
    this.mean = new Float64Array(d); this.std = new Float64Array(d);
    for (const row of X) for (let j = 0; j < d; j++) this.mean[j] += row[j];
    for (let j = 0; j < d; j++) this.mean[j] /= n;
    for (const row of X) for (let j = 0; j < d; j++) { const v = row[j] - this.mean[j]; this.std[j] += v * v; }
    for (let j = 0; j < d; j++) this.std[j] = Math.sqrt(this.std[j] / n) + 1e-8;
    this.Xtr = X.map((row) => { const z = new Float64Array(d); for (let j = 0; j < d; j++) z[j] = (row[j] - this.mean[j]) / this.std[j]; return z; });
    this.trSq = this.Xtr.map((z) => { let s = 0; for (let j = 0; j < d; j++) s += z[j] * z[j]; return s; });
    return this;
  };
  KNN.prototype.predict = function (X) {
    const d = this.mean.length, out = [];
    for (const row of X) {
      const z = new Float64Array(d); for (let j = 0; j < d; j++) z[j] = (row[j] - this.mean[j]) / this.std[j];
      let zSq = 0; for (let j = 0; j < d; j++) zSq += z[j] * z[j];
      const dists = new Array(this.Xtr.length);
      for (let i = 0; i < this.Xtr.length; i++) {
        const tr = this.Xtr[i]; let dot = 0; for (let j = 0; j < d; j++) dot += z[j] * tr[j];
        dists[i] = { d: zSq + this.trSq[i] - 2 * dot, y: this.y[i] };
      }
      dists.sort((a, b) => a.d - b.d);
      const votes = {}; let best = null, bestC = 0;
      for (let i = 0; i < this.k; i++) { const lbl = dists[i].y; votes[lbl] = (votes[lbl] || 0) + 1; if (votes[lbl] > bestC) { bestC = votes[lbl]; best = lbl; } }
      out.push(best);
    }
    return out;
  };
  function accuracy(pred, truth) { if (!truth.length) return 0; let c = 0; for (let i = 0; i < truth.length; i++) if (pred[i] === truth[i]) c++; return c / truth.length; }
  const models = { KNN, accuracy };

  // ---------------------------------------------------------------------- data ----
  const data = {};
  data.labelDetectedClips = function (samples, truthEvents) {
    const { onsets, clips } = segment.segment(samples);
    const sr = CFG.audio.sr;
    const tol = Math.round((CFG.segment.refractoryMs * 0.5 * sr) / 1000);
    const truth = truthEvents.slice().sort((a, b) => a.onset - b.onset);
    const used = new Array(truth.length).fill(false);
    const outClips = [], outLabels = [];
    for (let c = 0; c < onsets.length; c++) {
      let best = -1, bestDist = tol + 1;
      for (let i = 0; i < truth.length; i++) {
        if (used[i]) continue;
        const dd = Math.abs(onsets[c] - truth[i].onset);
        if (dd <= tol && dd < bestDist) { bestDist = dd; best = i; }
      }
      if (best >= 0) { used[best] = true; outClips.push(clips[c]); outLabels.push(truth[best].key); }
    }
    return { clips: outClips, labels: outLabels };
  };
  // `transform(samples, events) -> samples` rewrites each stream before segmentation.
  // That is how an ADAPTIVE attacker collects data: with the defense already running,
  // so it learns the masked distribution rather than the clean one.
  data.buildTrainingSet = function (keys, perKey, seed, transform) {
    const rng = new Rng(1234 + (seed || 0));
    const seq = [];
    for (const k of keys) for (let i = 0; i < perKey; i++) seq.push(k);
    rng.shuffle(seq);
    const chunk = 40, clips = [], labels = [];
    for (let start = 0; start < seq.length; start += chunk) {
      const part = seq.slice(start, start + chunk);
      const text = part.map((k) => (k === "<space>" ? " " : k)).join("");
      const { samples, events } = synth.renderStream(text, 1234 + (seed || 0) + start, CFG.segment.windowMs);
      const used = transform ? transform(samples, events, start) : samples;
      const r = data.labelDetectedClips(used, events);
      for (let i = 0; i < r.clips.length; i++) { clips.push(r.clips[i]); labels.push(r.labels[i]); }
    }
    return { clips, labels };
  };

  // -------------------------------------------------------- interference/masker ----
  const masker = {};
  masker.decoyBank = function (keys, windowMs) { return keys.map((k) => synth.keySignature(k, windowMs)); };
  masker.applyMasking = function (samples, keyEvents, bank, seed) {
    const m = CFG.model, sr = CFG.audio.sr, rng = new Rng(seed || 4242);
    const out = Float64Array.from(samples), n = out.length;
    const jitter = Math.max(1, Math.round((m.timingJitterMs * sr) / 1000));
    for (const ev of keyEvents) {
      for (let d = 0; d < m.decoysPerKey; d++) {
        const tmpl = bank[Math.floor(rng.next() * bank.length)];
        const g = m.maskGain * rng.uniform(0.85, 1.15);
        const offset = ev.onset + rng.int(-jitter, jitter);
        const start = offset - Math.floor(0.1 * tmpl.length);
        for (let i = 0; i < tmpl.length; i++) { const j = start + i; if (j >= 0 && j < n) out[j] += tmpl[i] * g; }
      }
    }
    return out;
  };

  // ------------------------------------------------------------------- defense ----
  const defense = {};
  defense.keystrokeRecovery = function (samples, truthEvents, recognizer, nFrames) {
    const { onsets, clips } = segment.segment(samples);
    if (!onsets.length) return { recovery: 0, insertions: 0 };
    const feats = features.featurize(clips, nFrames);
    const preds = recognizer.predict(feats);
    const sr = CFG.audio.sr, tol = Math.round((CFG.segment.refractoryMs * 0.5 * sr) / 1000);
    const detected = onsets.map((o, i) => ({ onset: o, pred: preds[i] })).sort((a, b) => a.onset - b.onset);
    const used = new Array(detected.length).fill(false);
    let correct = 0;
    for (const t of truthEvents) {
      let best = -1, bestDist = tol + 1;
      for (let i = 0; i < detected.length; i++) {
        if (used[i]) continue;
        const dd = Math.abs(detected[i].onset - t.onset);
        if (dd <= tol && dd < bestDist) { bestDist = dd; best = i; }
      }
      if (best >= 0) { used[best] = true; if (detected[best].pred === t.key) correct++; }
    }
    const insertions = used.filter((u) => !u).length;
    return { recovery: truthEvents.length ? correct / truthEvents.length : 0, insertions };
  };
  defense.randomCorpus = function (keys, length, seed) {
    const rng = new Rng(seed);
    const chars = keys.map((k) => (k === "<space>" ? " " : k));
    let s = ""; for (let i = 0; i < length; i++) s += rng.choice(chars);
    return s;
  };

  global.KD = {
    CFG, DEFAULT_KEYS, Rng, dsp, synth, segment, features, models, data, masker, defense,
  };
})(typeof window !== "undefined" ? window : this);
