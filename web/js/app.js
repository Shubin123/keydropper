/* keydropper web app — UI, visualizations, synthetic lab, and live mic capture. */
(function () {
  "use strict";
  const K = window.KD;
  const $ = (id) => document.getElementById(id);
  const tick = () => new Promise((r) => setTimeout(r, 0));

  // -------------------------------------------------------------------- tabs ----
  const tabs = Array.from(document.querySelectorAll(".tab"));
  function activateTab(t, focus = false) {
      document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
      document.querySelectorAll(".panel").forEach((x) => x.classList.remove("active"));
      t.classList.add("active");
      tabs.forEach((x) => {
        const selected = x === t;
        x.setAttribute("aria-selected", String(selected));
        x.tabIndex = selected ? 0 : -1;
      });
      $(t.dataset.tab).classList.add("active");
      if (focus) t.focus();
  }
  tabs.forEach((t, index) => {
    t.addEventListener("click", () => activateTab(t));
    t.addEventListener("keydown", (event) => {
      let next = index;
      if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
      else if (event.key === "ArrowLeft") next = (index - 1 + tabs.length) % tabs.length;
      else if (event.key === "Home") next = 0;
      else if (event.key === "End") next = tabs.length - 1;
      else return;
      event.preventDefault(); activateTab(tabs[next], true);
    });
  });

  // slider value labels
  [["perkey", "perkeyVal"], ["evallen", "evallenVal"], ["maskgain", "maskgainVal"], ["decoys", "decoysVal"]]
    .forEach(([inp, out]) => { const el = $(inp); const set = () => ($(out).textContent = el.value); el.addEventListener("input", set); set(); });

  // ------------------------------------------------------------ canvas helpers ----
  function ctxOf(id) {
    const c = $(id);
    const ratio = window.devicePixelRatio || 1;
    if (c.width !== c.clientWidth * ratio) { c.width = c.clientWidth * ratio; c.height = c.height * ratio / (c._r || 1); }
    const g = c.getContext("2d");
    g.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { g, w: c.clientWidth, h: c.height / ratio };
  }
  function clear(g, w, h) { g.clearRect(0, 0, w, h); }
  // viridis-ish colormap for spectrograms
  function heat(t) {
    t = Math.max(0, Math.min(1, t));
    const r = Math.round(255 * Math.min(1, Math.max(0, -0.2 + 2.2 * t - 0.7 * t * t)));
    const gc = Math.round(255 * Math.min(1, Math.max(0, 0.1 + 1.1 * t)));
    const b = Math.round(255 * Math.min(1, Math.max(0, 0.6 - 0.9 * t + 0.6 * (1 - t) * (1 - t))));
    return `rgb(${r},${gc},${b})`;
  }

  function drawBars(clean, defended, chance) {
    const { g, w, h } = ctxOf("barChart"); clear(g, w, h);
    const pad = 40, base = h - 28, top = 16, bw = 90, gap = 90;
    const x0 = pad + 30;
    const scaleY = (v) => base - v * (base - top);
    g.strokeStyle = "#263258"; g.lineWidth = 1;
    g.beginPath(); g.moveTo(pad, base); g.lineTo(w - 10, base); g.stroke();
    g.fillStyle = "#9aa7c7"; g.font = "12px sans-serif";
    [0, 0.25, 0.5, 0.75, 1].forEach((v) => { const y = scaleY(v); g.fillText((v * 100).toFixed(0) + "%", 6, y + 4); g.strokeStyle = "#1b2543"; g.beginPath(); g.moveTo(pad, y); g.lineTo(w - 10, y); g.stroke(); });
    const bars = [["no defense", clean, "#ff6b6b"], ["with masking", defended, "#37d67a"]];
    bars.forEach(([label, v, col], i) => {
      const x = x0 + i * (bw + gap);
      g.fillStyle = col; const y = scaleY(v); g.fillRect(x, y, bw, base - y);
      g.fillStyle = "#e8edf9"; g.font = "bold 15px sans-serif"; g.textAlign = "center";
      g.fillText((v * 100).toFixed(1) + "%", x + bw / 2, y - 8);
      g.fillStyle = "#9aa7c7"; g.font = "12px sans-serif"; g.fillText(label, x + bw / 2, base + 18);
      g.textAlign = "left";
    });
    const yc = scaleY(chance);
    g.strokeStyle = "#4f8cff"; g.setLineDash([6, 5]); g.beginPath(); g.moveTo(pad, yc); g.lineTo(w - 10, yc); g.stroke(); g.setLineDash([]);
    g.fillStyle = "#4f8cff"; g.font = "12px sans-serif"; g.fillText("chance " + (chance * 100).toFixed(1) + "%", w - 120, yc - 6);
  }

  function drawWave(samples, onsets, id) {
    const { g, w, h } = ctxOf(id); clear(g, w, h);
    const n = samples.length, mid = h / 2;
    let peak = 1e-6; for (let i = 0; i < n; i++) peak = Math.max(peak, Math.abs(samples[i]));
    g.strokeStyle = "#38d6c8"; g.lineWidth = 1; g.beginPath();
    for (let x = 0; x < w; x++) {
      const i = Math.floor((x / w) * n);
      const v = (samples[i] / peak) * (mid - 4);
      if (x === 0) g.moveTo(x, mid - v); else g.lineTo(x, mid - v);
    }
    g.stroke();
    if (onsets) { g.strokeStyle = "#37d67a"; g.lineWidth = 1.5; onsets.forEach((o) => { const x = (o / n) * w; g.beginPath(); g.moveTo(x, 2); g.lineTo(x, 10); g.stroke(); }); }
  }

  function drawSpec(clip, id) {
    const { g, w, h } = ctxOf(id); clear(g, w, h);
    const mat = K.features.logmel(clip);
    if (!mat.length) return;
    const nF = mat.length, nM = mat[0].length;
    let mn = Infinity, mx = -Infinity;
    for (const row of mat) for (const v of row) { mn = Math.min(mn, v); mx = Math.max(mx, v); }
    const rng = mx - mn || 1;
    const cw = w / nF, ch = h / nM;
    for (let f = 0; f < nF; f++) for (let m = 0; m < nM; m++) {
      g.fillStyle = heat((mat[f][m] - mn) / rng);
      g.fillRect(f * cw, h - (m + 1) * ch, Math.ceil(cw) + 1, Math.ceil(ch) + 1);
    }
  }

  function drawOverlay(clean, masked, id) {
    const { g, w, h } = ctxOf(id); clear(g, w, h);
    const mid = h / 2, n = Math.min(clean.length, masked.length);
    let peak = 1e-6; for (let i = 0; i < n; i++) peak = Math.max(peak, Math.abs(masked[i]), Math.abs(clean[i]));
    const draw = (arr, col) => { g.strokeStyle = col; g.lineWidth = 1; g.beginPath(); for (let x = 0; x < w; x++) { const i = Math.floor((x / w) * n); const v = (arr[i] / peak) * (mid - 4); if (x === 0) g.moveTo(x, mid - v); else g.lineTo(x, mid - v); } g.stroke(); };
    draw(masked, "#ff6b6b"); draw(clean, "#4f8cff");
    g.fillStyle = "#4f8cff"; g.font = "12px sans-serif"; g.fillText("clean", 8, 14);
    g.fillStyle = "#ff6b6b"; g.fillText("masked", 56, 14);
  }

  // ----------------------------------------------------------- synthetic lab ----
  async function runBenchmark() {
    const btn = $("runBtn"); btn.disabled = true;
    const prog = $("progress"); prog.classList.remove("hidden");
    const setProg = (p, msg) => { prog.querySelector(".bar").style.width = p + "%"; prog.querySelector(".msg").textContent = msg; };
    try {
      const nKeys = parseInt($("keyset").value, 10);
      const keys = nKeys === 11 ? K.DEFAULT_KEYS.slice(0, 10).concat(["<space>"]) : K.DEFAULT_KEYS;
      const perKey = parseInt($("perkey").value, 10);
      const evalLen = parseInt($("evallen").value, 10);
      K.CFG.model.maskGain = parseFloat($("maskgain").value);
      K.CFG.model.decoysPerKey = parseInt($("decoys").value, 10);
      const windowMs = K.CFG.segment.windowMs;
      const nFrames = K.features.inferNFrames(windowMs);
      const chance = 1 / keys.length;

      setProg(10, "training the eavesdropper on clean audio…"); await tick();
      const train = K.data.buildTrainingSet(keys, perKey, 0);
      await tick();
      const Xtr = K.features.featurize(train.clips, nFrames);
      const clf = new K.models.KNN(3).fit(Xtr, train.labels);

      setProg(45, "measuring recognizer on an unseen recording…"); await tick();
      const corpusA = K.defense.randomCorpus(keys, Math.min(evalLen, 140), 321);
      const streamA = K.synth.renderStream(corpusA, 654, windowMs);
      const labA = K.data.labelDetectedClips(streamA.samples, streamA.events);
      const xaFeat = K.features.featurize(labA.clips, nFrames);
      const recogAcc = K.models.accuracy(clf.predict(xaFeat), labA.labels);
      // onset F1 on the same stream
      const onA = K.segment.detectOnsets(streamA.samples);
      const tol = Math.round((K.CFG.segment.refractoryMs * 0.5 * K.CFG.audio.sr) / 1000);
      const m = K.segment.matchOnsets(onA, streamA.events.map((e) => e.onset), tol);
      const prec = m.tp / (m.tp + m.fp || 1), rec = m.tp / (m.tp + m.fn || 1);
      const f1 = (2 * prec * rec) / (prec + rec || 1);

      setProg(70, "attacking a fresh stream (no defense)…"); await tick();
      const corpus = K.defense.randomCorpus(keys, evalLen, 11);
      const evalStream = K.synth.renderStream(corpus, 7, windowMs);
      const clean = K.defense.keystrokeRecovery(evalStream.samples, evalStream.events, clf, nFrames);

      setProg(80, "switching on the interference model…"); await tick();
      const bank = K.masker.decoyBank(keys, windowMs);
      const defended = K.masker.applyMasking(evalStream.samples, evalStream.events, bank, 3);
      const def = K.defense.keystrokeRecovery(defended, evalStream.events, clf, nFrames);

      // Adaptive attacker: retrain on masked audio (it knows the defense exists).
      // The defense's real guarantee is the strongest attacker, never the flattering one.
      setProg(90, "retraining an adaptive attacker on masked audio…"); await tick();
      const adaTrain = K.data.buildTrainingSet(keys, perKey, 17, (s, ev, off) =>
        K.masker.applyMasking(s, ev, bank, 500 + off));
      let adaRec = NaN;
      if (adaTrain.clips.length) {
        const adaClf = new K.models.KNN(3).fit(K.features.featurize(adaTrain.clips, nFrames), adaTrain.labels);
        adaRec = K.defense.keystrokeRecovery(defended, evalStream.events, adaClf, nFrames).recovery;
      }
      const best = isNaN(adaRec) ? def.recovery : Math.max(def.recovery, adaRec);

      // metrics
      $("mF1").textContent = f1.toFixed(2);
      $("mAcc").textContent = (recogAcc * 100).toFixed(0) + "%";
      $("mClean").textContent = (clean.recovery * 100).toFixed(0) + "%";
      $("mDef").textContent = (best * 100).toFixed(0) + "%";
      $("mAda").textContent = isNaN(adaRec) ? "–" : (adaRec * 100).toFixed(0) + "%";
      $("mRed").textContent = "-" + ((clean.recovery - best) * 100).toFixed(0) + " pts";

      // charts
      drawBars(clean.recovery, best, chance);
      const demo = K.synth.renderStream(K.defense.randomCorpus(keys, 20, 99), 5, windowMs);
      drawWave(demo.samples, K.segment.detectOnsets(demo.samples), "waveChart");
      drawSpec(K.synth.keySignature(keys[0], windowMs, new K.Rng(1)), "specChart");
      // clean vs masked: a window around the first onset
      const ev0 = evalStream.events[0].onset;
      const a = ev0 - 400, span = 2400;
      const cwin = Array.from(evalStream.samples.slice(Math.max(0, a), a + span));
      const mwin = Array.from(defended.slice(Math.max(0, a), a + span));
      drawOverlay(cwin, mwin, "maskChart");

      $("runBtn").textContent = "Run benchmark again";
      setProg(100, "done."); setTimeout(() => prog.classList.add("hidden"), 600);
    } catch (e) {
      prog.querySelector(".msg").textContent = "error: " + e.message;
      console.error(e);
    } finally { btn.disabled = false; }
  }
  $("runBtn").addEventListener("click", runBenchmark);

  // ------------------------------------------------------------- live capture ----
  const live = {
    ctx: null, ring: null, R: 0, writePos: 0, total: 0, node: null, stream: null,
    samples: [], labels: [], clf: null, testAcc: { n: 0, ok: 0 },
  };
  const CAP_PRE_MS = 40, CAP_SPAN_MS = 200, TARGET_SR = 16000;

  if (!window.isSecureContext && location.protocol === "file:") {
    $("secnote").textContent = "Microphone needs HTTPS or localhost. The synthetic lab works here; open this page on GitHub Pages or a local server for live capture.";
  }

  async function enableMic() {
    try {
      live.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
      live.ctx = new (window.AudioContext || window.webkitAudioContext)();
      live.R = Math.round(live.ctx.sampleRate * 3);
      live.ring = new Float32Array(live.R);
      const src = live.ctx.createMediaStreamSource(live.stream);
      live.node = live.ctx.createScriptProcessor(2048, 1, 1);
      live.node.onaudioprocess = (e) => {
        const inp = e.inputBuffer.getChannelData(0);
        for (let i = 0; i < inp.length; i++) { live.ring[live.writePos] = inp[i]; live.writePos = (live.writePos + 1) % live.R; live.total++; }
      };
      src.connect(live.node);
      const sink = live.ctx.createGain(); sink.gain.value = 0; live.node.connect(sink); sink.connect(live.ctx.destination);
      $("micStatus").textContent = "microphone on · " + live.ctx.sampleRate + " Hz";
      $("micBtn").disabled = true; $("micBtn").textContent = "microphone enabled";
    } catch (err) {
      $("micStatus").textContent = "mic denied: " + err.message;
    }
  }
  $("micBtn").addEventListener("click", enableMic);

  function ringGrab(markTotal) {
    // Extract a CAP_SPAN_MS window around markTotal, downsampled to TARGET_SR.
    const sr = live.ctx.sampleRate;
    const preS = Math.round((CAP_PRE_MS * sr) / 1000);
    const spanS = Math.round((CAP_SPAN_MS * sr) / 1000);
    const startAbs = markTotal - preS;
    const outN = Math.round((CAP_SPAN_MS * TARGET_SR) / 1000);
    const ratio = sr / TARGET_SR;
    const out = new Float64Array(outN);
    for (let i = 0; i < outN; i++) {
      const abs = startAbs + Math.round(i * ratio);
      if (abs < 0 || abs >= live.total || live.total - abs >= live.R) { out[i] = 0; continue; }
      const idx = ((live.writePos - (live.total - abs)) % live.R + live.R) % live.R;
      out[i] = live.ring[idx];
    }
    return out;
  }

  function keyFromEvent(e) {
    if (e.key === " ") return "<space>";
    if (e.key.length === 1 && /[a-zA-Z0-9]/.test(e.key)) return e.key.toLowerCase();
    return null;
  }

  function playMasking() {
    if (!live.ctx) return;
    const keys = K.DEFAULT_KEYS;
    const windowMs = K.CFG.segment.windowMs;
    for (let d = 0; d < 2; d++) {
      const key = keys[Math.floor(Math.random() * keys.length)];
      const sig = K.synth.keySignature(key, windowMs);
      const buf = live.ctx.createBuffer(1, sig.length, TARGET_SR);
      const ch = buf.getChannelData(0);
      for (let i = 0; i < sig.length; i++) ch[i] = sig[i] * 0.5;
      const s = live.ctx.createBufferSource(); s.buffer = buf;
      const g = live.ctx.createGain(); g.gain.value = 0.9;
      s.connect(g); g.connect(live.ctx.destination);
      s.start(live.ctx.currentTime + d * 0.008 + Math.random() * 0.006);
    }
  }

  function onTrainKey(e) {
    const key = keyFromEvent(e);
    if (!key || !live.ctx) return;
    if ($("maskToggle").checked) playMasking();
    const mark = live.total;
    setTimeout(() => {
      const clip = ringGrab(mark);
      live.samples.push(clip); live.labels.push(key);
      $("sampleCount").textContent = live.samples.length + " samples · " + new Set(live.labels).size + " keys";
      drawSpec(clip, "liveSpec");
      $("trainBtn").disabled = !(live.samples.length >= 12 && new Set(live.labels).size >= 2);
    }, CAP_SPAN_MS - CAP_PRE_MS + 30);
  }
  $("trainBox").addEventListener("keydown", onTrainKey);

  $("trainBtn").addEventListener("click", () => {
    const windowMs = K.CFG.segment.windowMs;
    const nFrames = K.features.inferNFrames(windowMs);
    const X = K.features.featurize(live.samples, nFrames);
    live.clf = new K.models.KNN(3).fit(X, live.labels);
    live._nFrames = nFrames;
    $("testBox").disabled = false; $("testBox").placeholder = "type here — it will guess each key from sound…";
    $("testBox").focus();
    live.testAcc = { n: 0, ok: 0 }; $("predStream").innerHTML = ""; $("liveAcc").textContent = "–"; $("liveN").textContent = "0";
  });

  $("clearBtn").addEventListener("click", () => {
    live.samples = []; live.labels = []; live.clf = null;
    $("sampleCount").textContent = "0 samples"; $("trainBtn").disabled = true;
    $("testBox").disabled = true; $("predStream").innerHTML = "";
  });

  function onTestKey(e) {
    const key = keyFromEvent(e);
    if (!key || !live.clf) return;
    if ($("maskToggle").checked) playMasking();
    const mark = live.total;
    setTimeout(() => {
      const clip = ringGrab(mark);
      const feat = K.features.featurize([clip], live._nFrames);
      const pred = live.clf.predict(feat)[0];
      const ok = pred === key;
      live.testAcc.n++; if (ok) live.testAcc.ok++;
      const span = document.createElement("span");
      span.className = ok ? "ok" : "no";
      span.textContent = pred === "<space>" ? "␣" : pred;
      $("predStream").appendChild(span);
      const ps = $("predStream"); while (ps.childNodes.length > 40) ps.removeChild(ps.firstChild);
      $("liveAcc").textContent = Math.round((live.testAcc.ok / live.testAcc.n) * 100) + "%";
      $("liveN").textContent = live.testAcc.n;
    }, CAP_SPAN_MS - CAP_PRE_MS + 30);
  }
  $("testBox").addEventListener("keydown", onTestKey);

})();
