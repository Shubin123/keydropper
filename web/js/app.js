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

  // -------------------------------------------------------- labeled WAV example ----
  function decodeExampleWav(buffer) {
    const view = new DataView(buffer);
    const text = (offset, length) => String.fromCharCode(...new Uint8Array(buffer, offset, length));
    if (buffer.byteLength < 44 || text(0, 4) !== "RIFF" || text(8, 4) !== "WAVE") {
      throw new Error("The example file is not a valid RIFF/WAVE file.");
    }
    let format = null, dataOffset = -1, dataLength = 0;
    for (let offset = 12; offset + 8 <= buffer.byteLength;) {
      const id = text(offset, 4), size = view.getUint32(offset + 4, true);
      const start = offset + 8;
      if (start + size > buffer.byteLength) throw new Error("The example WAV is truncated.");
      if (id === "fmt ") {
        if (size < 16) throw new Error("The example WAV format chunk is incomplete.");
        format = {
          codec: view.getUint16(start, true), channels: view.getUint16(start + 2, true),
          sampleRate: view.getUint32(start + 4, true), bits: view.getUint16(start + 14, true),
        };
      } else if (id === "data") { dataOffset = start; dataLength = size; }
      offset = start + size + (size % 2);
    }
    if (!format || dataOffset < 0) throw new Error("The example WAV is missing audio data.");
    if (format.codec !== 1 || format.channels !== 1 || format.bits !== 16 || format.sampleRate !== K.CFG.audio.sr || dataLength % 2) {
      throw new Error("The example WAV format is unsupported (expected 16 kHz mono PCM16).");
    }
    if (!dataLength) throw new Error("The WAV contains no audio samples.");
    const samples = new Float64Array(dataLength / 2);
    for (let i = 0; i < samples.length; i++) samples[i] = view.getInt16(dataOffset + i * 2, true) / 32768;
    if (samples.length / format.sampleRate > 60) throw new Error("WAV uploads must be 60 seconds or shorter.");
    return samples;
  }

  const sampleRecognizers = new Map();
  let sampleFrames = 0;

  function formatTimestamp(sample) {
    const seconds = sample / K.CFG.audio.sr;
    const minutes = Math.floor(seconds / 60);
    return `${String(minutes).padStart(2, "0")}:${(seconds % 60).toFixed(3).padStart(6, "0")}`;
  }

  const timelineState = { samples: null, onsets: [], aligned: [], duration: 0, width: 0, height: 0, ratio: 1, base: null };
  const timelineAudio = $("sampleAudio");
  let currentAudioUrl = null;

  function paintPlayhead() {
    const canvas = $("audioTimeline"), state = timelineState;
    if (!state.base) return;
    const ctx = canvas.getContext("2d"), ratio = state.ratio;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, state.width, state.height);
    ctx.drawImage(state.base, 0, 0, state.width, state.height);
    const now = Math.max(0, Math.min(state.duration, timelineAudio.currentTime || 0));
    const x = state.duration ? (now / state.duration) * state.width : 0;
    const waveTop = state.waveTop;
    ctx.strokeStyle = "#ffc857"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(x, 8); ctx.lineTo(x, state.height - 22); ctx.stroke();
    ctx.fillStyle = "#ffc857"; ctx.beginPath(); ctx.arc(x, waveTop - 7, 4, 0, Math.PI * 2); ctx.fill();
    $("playheadTime").textContent = formatTimestamp(now * K.CFG.audio.sr);
  }

  function renderAudioTimeline(samples, onsets, aligned) {
    const state = timelineState, canvas = $("audioTimeline");
    const sr = K.CFG.audio.sr;
    state.samples = samples; state.onsets = onsets; state.aligned = aligned;
    state.duration = samples.length / sr;
    state.width = Math.max(720, Math.min(6000, Math.ceil(state.duration * 150)));
    state.ratio = Math.min(window.devicePixelRatio || 1, 2);
    const cardW = 146, cardH = 22, laneGap = 5;
    const predictedRows = aligned.filter((item) => item.predIndex !== null);
    const laneEnds = [];
    const placements = predictedRows.map((item) => {
      const x = (onsets[item.predIndex] / sr) * state.width;
      const left = Math.max(2, Math.min(state.width - cardW - 2, x - cardW / 2));
      let lane = laneEnds.findIndex((end) => left > end + 5);
      if (lane < 0) lane = laneEnds.length;
      laneEnds[lane] = left + cardW;
      return { item, x, left, lane };
    });
    const laneCount = Math.max(1, laneEnds.length);
    state.waveTop = 18 + laneCount * (cardH + laneGap) + 5;
    state.height = state.waveTop + 70;
    canvas.style.width = `${state.width}px`;
    canvas.style.minWidth = `${state.width}px`;
    canvas.style.height = `${state.height}px`;
    canvas.width = Math.ceil(state.width * state.ratio);
    canvas.height = Math.ceil(state.height * state.ratio);
    const base = document.createElement("canvas");
    base.width = canvas.width; base.height = canvas.height;
    const g = base.getContext("2d");
    g.setTransform(state.ratio, 0, 0, state.ratio, 0, 0);
    g.clearRect(0, 0, state.width, state.height);

    // Render the audio envelope and a time ruler on the same horizontal scale as labels.
    const mid = state.waveTop + 28, half = 24;
    g.strokeStyle = "#263258"; g.lineWidth = 1;
    g.beginPath(); g.moveTo(0, mid); g.lineTo(state.width, mid); g.stroke();
    const peak = samples.reduce((v, s) => Math.max(v, Math.abs(s)), 1e-6);
    const bins = Math.max(1, Math.floor(state.width));
    g.strokeStyle = "#38d6c8"; g.globalAlpha = 0.86; g.beginPath();
    for (let px = 0; px < bins; px++) {
      const start = Math.floor((px / bins) * samples.length);
      const end = Math.max(start + 1, Math.floor(((px + 1) / bins) * samples.length));
      let lo = 0, hi = 0;
      for (let i = start; i < end && i < samples.length; i++) { lo = Math.min(lo, samples[i] / peak); hi = Math.max(hi, samples[i] / peak); }
      g.moveTo(px + 0.5, mid - hi * half); g.lineTo(px + 0.5, mid - lo * half);
    }
    g.stroke(); g.globalAlpha = 1;

    // Position each key label at its detected onset; lanes only prevent text overlap.
    const keyLabel = (key) => key === null ? "—" : key === "<space>" || key === " " ? "␣" : key;
    for (const place of placements) {
      const { item, x, left, lane } = place;
      const y = 9 + lane * (cardH + laneGap);
      const matched = item.expected !== null && item.expected === (item.predicted === "<space>" ? " " : item.predicted);
      const color = item.expected === null ? "#4f8cff" : matched ? "#37d67a" : "#ff6b6b";
      const label = item.expected === null
        ? `${formatTimestamp(onsets[item.predIndex])}  ${keyLabel(item.predicted)}`
        : `${formatTimestamp(onsets[item.predIndex])}  ${keyLabel(item.expected)} → ${keyLabel(item.predicted)}`;
      g.strokeStyle = color; g.fillStyle = "rgba(18,26,48,.96)"; g.lineWidth = 1;
      g.beginPath(); g.roundRect(left, y, cardW, cardH, 5); g.fill(); g.stroke();
      g.fillStyle = color; g.font = "11px ui-monospace, monospace"; g.textAlign = "center";
      g.fillText(label, left + cardW / 2, y + 15, cardW - 8);
      g.beginPath(); g.moveTo(x, y + cardH); g.lineTo(x, state.waveTop); g.stroke();
      g.beginPath(); g.arc(x, state.waveTop, 3, 0, Math.PI * 2); g.fill();
    }

    const tickStep = Math.max(0.5, Math.ceil(state.duration / 8 * 2) / 2);
    const rulerY = state.height - 21;
    g.strokeStyle = "#526083"; g.fillStyle = "#9aa7c7"; g.textAlign = "center";
    g.font = "11px ui-monospace, monospace"; g.beginPath(); g.moveTo(0, rulerY); g.lineTo(state.width, rulerY); g.stroke();
    for (let sec = 0; sec <= state.duration; sec += tickStep) {
      const x = (sec / state.duration) * state.width;
      g.beginPath(); g.moveTo(x, rulerY - 4); g.lineTo(x, rulerY + 4); g.stroke();
      g.fillText(`${sec.toFixed(sec % 1 ? 1 : 0)}s`, x, state.height - 5);
    }
    state.base = base;
    canvas.setAttribute("aria-label", `Waveform timeline, ${predictedRows.length} timestamped key detections over ${state.duration.toFixed(2)} seconds. Click or use arrow keys to seek.`);
    paintPlayhead();
  }

  ["timeupdate", "seeked", "loadedmetadata", "play", "pause", "ended"].forEach((eventName) => {
    timelineAudio.addEventListener(eventName, paintPlayhead);
  });
  let playheadFrame = 0;
  timelineAudio.addEventListener("play", () => {
    if (playheadFrame) return;
    const animate = () => {
      paintPlayhead();
      playheadFrame = timelineAudio.paused ? 0 : requestAnimationFrame(animate);
    };
    playheadFrame = requestAnimationFrame(animate);
  });
  timelineAudio.addEventListener("pause", () => {
    if (playheadFrame) cancelAnimationFrame(playheadFrame);
    playheadFrame = 0;
  });
  $("audioTimeline").addEventListener("click", (event) => {
    if (!timelineState.duration) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const fraction = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
    timelineAudio.currentTime = fraction * timelineState.duration;
    paintPlayhead();
  });
  $("audioTimeline").addEventListener("keydown", (event) => {
    if (!timelineState.duration) return;
    if (event.key === "ArrowLeft" || event.key === "ArrowRight" || event.key === "Home" || event.key === "End") {
      event.preventDefault();
      if (event.key === "Home") timelineAudio.currentTime = 0;
      else if (event.key === "End") timelineAudio.currentTime = timelineState.duration;
      else timelineAudio.currentTime = Math.max(0, Math.min(timelineState.duration, timelineAudio.currentTime + (event.key === "ArrowRight" ? 0.25 : -0.25)));
      paintPlayhead();
    }
  });

  function alignKeySequences(expected, predicted) {
    if (expected === null) return predicted.map((key, index) => ({ expected: null, predicted: key, predIndex: index }));
    const rows = expected.length + 1, cols = predicted.length + 1;
    const cost = Array.from({ length: rows }, () => new Uint32Array(cols));
    for (let i = expected.length; i >= 0; i--) {
      for (let j = predicted.length; j >= 0; j--) {
        if (i === expected.length) cost[i][j] = predicted.length - j;
        else if (j === predicted.length) cost[i][j] = expected.length - i;
        else if (expected[i] === (predicted[j] === "<space>" ? " " : predicted[j])) cost[i][j] = cost[i + 1][j + 1];
        else cost[i][j] = 1 + Math.min(cost[i + 1][j + 1], cost[i + 1][j], cost[i][j + 1]);
      }
    }
    const aligned = [];
    let i = 0, j = 0;
    while (i < expected.length || j < predicted.length) {
      if (i < expected.length && j < predicted.length &&
          expected[i] === (predicted[j] === "<space>" ? " " : predicted[j]) && cost[i][j] === cost[i + 1][j + 1]) {
        aligned.push({ expected: expected[i], predicted: predicted[j], predIndex: j }); i++; j++;
      } else if (i < expected.length && j < predicted.length && cost[i][j] === 1 + cost[i + 1][j + 1]) {
        aligned.push({ expected: expected[i], predicted: predicted[j], predIndex: j }); i++; j++;
      } else if (i < expected.length && cost[i][j] === 1 + cost[i + 1][j]) {
        aligned.push({ expected: expected[i], predicted: null, predIndex: null }); i++;
      } else {
        aligned.push({ expected: null, predicted: predicted[j], predIndex: j }); j++;
      }
    }
    return aligned;
  }

  function renderKeySequence(predicted, onsets, expected) {
    const root = $("sampleSequence"); root.replaceChildren();
    const aligned = alignKeySequences(expected, predicted);
    const keyLabel = (key) => key === null ? "—" : key === "<space>" || key === " " ? "␣" : key;
    const table = document.createElement("table");
    const head = document.createElement("thead"), headRow = document.createElement("tr");
    ["#", "Time", "Expected", "Predicted", "Result"].forEach((label) => {
      const th = document.createElement("th"); th.scope = "col"; th.textContent = label; headRow.appendChild(th);
    });
    head.appendChild(headRow); table.appendChild(head);
    const body = document.createElement("tbody");
    let matches = 0;
    aligned.forEach((item, index) => {
      const tr = document.createElement("tr");
      const cells = [String(index + 1), item.predIndex === null ? "—" : formatTimestamp(onsets[item.predIndex]), keyLabel(item.expected), keyLabel(item.predicted)];
      cells.forEach((value, cellIndex) => {
        const td = document.createElement("td"); td.textContent = value;
        if (cellIndex === 0) td.className = "sequence-index";
        if (cellIndex === 1 && item.predIndex !== null) {
          const sec = onsets[item.predIndex] / K.CFG.audio.sr;
          const time = document.createElement("time"); time.dateTime = `PT${sec.toFixed(3)}S`; time.textContent = cells[1];
          td.replaceChildren(time);
        }
        tr.appendChild(td);
      });
      const result = document.createElement("td");
      if (expected === null) { result.textContent = "unlabeled"; result.className = "sequence-unknown"; }
      else if (item.expected !== null && item.expected === (item.predicted === "<space>" ? " " : item.predicted)) {
        result.textContent = "match"; result.className = "sequence-match"; matches++;
      } else { result.textContent = "mismatch"; result.className = "sequence-mismatch"; }
      tr.appendChild(result); body.appendChild(tr);
    });
    table.appendChild(body); root.appendChild(table);
    return { aligned: aligned.length, matches, items: aligned };
  }

  function encodePcm16Wav(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2), view = new DataView(buffer);
    const writeTag = (offset, tag) => { for (let i = 0; i < tag.length; i++) view.setUint8(offset + i, tag.charCodeAt(i)); };
    const dataBytes = samples.length * 2;
    writeTag(0, "RIFF"); view.setUint32(4, 36 + dataBytes, true); writeTag(8, "WAVE");
    writeTag(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
    view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true); view.setUint16(34, 16, true); writeTag(36, "data"); view.setUint32(40, dataBytes, true);
    for (let i = 0; i < samples.length; i++) {
      const value = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(44 + i * 2, Math.round(value * (value < 0 ? 32768 : 32767)), true);
    }
    return buffer;
  }

  async function analyzeWav(buffer, expectedText, sourceLabel, keyboardType) {
    const status = $("sampleStatus"), result = $("sampleResult");
    const profile = K.SYNTH_KEYBOARDS[keyboardType];
    status.textContent = "Reading WAV locally…";
    const samples = decodeExampleWav(buffer);
    await tick();
    let recognizer = sampleRecognizers.get(keyboardType);
    if (!recognizer) {
      status.textContent = "Training the local recognizer…";
      await tick();
      sampleFrames = K.features.inferNFrames(K.CFG.segment.windowMs);
      const clips = [], labels = [];
      for (const key of K.DEFAULT_KEYS) for (let i = 0; i < 12; i++) {
        clips.push(K.synth.keySignature(key, K.CFG.segment.windowMs, new K.Rng(212 + i + key.charCodeAt(0)), profile.seed, keyboardType));
        labels.push(key);
      }
      recognizer = new K.models.KNN(3).fit(K.features.featurize(clips, sampleFrames), labels);
      sampleRecognizers.set(keyboardType, recognizer);
    }
    const detected = K.segment.segment(samples);
    if (!detected.clips.length) throw new Error("No keyboard onsets were detected in the WAV.");
    const predicted = recognizer.predict(K.features.featurize(detected.clips, sampleFrames));
    const expected = expectedText === null ? null : Array.from(expectedText.toLowerCase());
    const summary = renderKeySequence(predicted, detected.onsets, expected);
    if (currentAudioUrl) URL.revokeObjectURL(currentAudioUrl);
    currentAudioUrl = URL.createObjectURL(new Blob([buffer], { type: "audio/wav" }));
    timelineAudio.src = currentAudioUrl;
    timelineAudio.load();
    $("downloadCurrentWav").href = currentAudioUrl;
    $("downloadCurrentWav").download = `${sourceLabel.toLowerCase().replace(/[^a-z0-9]+/g, "-")}.wav`;
    renderAudioTimeline(samples, detected.onsets, summary.items);
    result.textContent = expected === null
      ? `${sourceLabel} · ${profile.label}: predicted ${predicted.length} keys in timestamp order. Add expected text to see label matches.`
      : `${sourceLabel} · ${profile.label}: ${summary.matches}/${summary.aligned} ordered labels matched · ${predicted.length} keys predicted.`;
    status.textContent = "Analyzed locally · file was not uploaded";
  }

  let analysisBusy = false;
  async function runModuleAnalysis(button, operation) {
    if (analysisBusy) return;
    analysisBusy = true;
    document.querySelectorAll(".analyze-phrase-btn, #analyzeUploadBtn").forEach((control) => { control.disabled = true; });
    try { await operation(); }
    catch (error) {
      $("sampleStatus").textContent = "Could not analyze audio";
      $("sampleResult").textContent = error.message;
      $("sampleSequence").replaceChildren();
    } finally {
      analysisBusy = false;
      document.querySelectorAll(".analyze-phrase-btn, #analyzeUploadBtn").forEach((control) => { control.disabled = false; });
    }
  }

  document.querySelectorAll(".analyze-phrase-btn").forEach((button) => {
    button.addEventListener("click", () => runModuleAnalysis(button, async () => {
      const module = button.closest(".audio-module");
      const title = module.querySelector("h3").textContent;
      const keyboardType = $("keyboardType").value;
      const profile = K.SYNTH_KEYBOARDS[keyboardType];
      $("reviewTrackTitle").textContent = title;
      $("sampleStatus").textContent = `Generating ${profile.label} audio…`;
      await tick();
      const stream = K.synth.renderStream(button.dataset.text, parseInt(button.dataset.seed, 10), K.CFG.segment.windowMs, {
        keysPerSecond: parseFloat(button.dataset.kps), backgroundNoise: 0.002,
        keyboardType, keyboardSeed: profile.seed,
      });
      await analyzeWav(encodePcm16Wav(stream.samples, K.CFG.audio.sr), button.dataset.text, title, keyboardType);
    }));
  });

  $("analyzeUploadBtn").addEventListener("click", async () => {
    const button = $("analyzeUploadBtn"), file = $("wavFileInput").files[0];
    await runModuleAnalysis(button, async () => {
      if (!file) throw new Error("Choose a WAV file first.");
      const rawExpected = $("expectedSequence").value.toLowerCase();
      if (rawExpected && !/^[a-z ]+$/.test(rawExpected)) throw new Error("Expected text can contain English letters and spaces only.");
      const keyboardType = $("keyboardType").value;
      $("reviewTrackTitle").textContent = file.name;
      await analyzeWav(await file.arrayBuffer(), rawExpected || null, file.name, keyboardType);
    });
  });

  // Reorder the module stack with pointer drag or the adjacent move buttons.
  const moduleStack = $("audioModuleStack");
  let draggingModule = null;
  function announceModuleOrder(module) {
    const modules = Array.from(moduleStack.children);
    const index = modules.indexOf(module);
    $("moduleOrderStatus").textContent = `${module.dataset.moduleTitle} moved to position ${index + 1} of ${modules.length}.`;
    modules.forEach((item, i) => {
      item.querySelector(".module-up").disabled = i === 0;
      item.querySelector(".module-down").disabled = i === modules.length - 1;
      const kicker = item.querySelector(".module-kicker");
      if (item.classList.contains("phrase-module")) kicker.textContent = `PHRASE ${String(modules.slice(0, i + 1).filter((x) => x.classList.contains("phrase-module")).length).padStart(2, "0")}`;
    });
  }
  Array.from(moduleStack.children).forEach((module) => {
    module.querySelector(".module-grab").addEventListener("dragstart", (event) => {
      draggingModule = module;
      module.classList.add("is-dragging");
      event.dataTransfer.effectAllowed = "move";
      event.dataTransfer.setData("text/plain", module.dataset.moduleTitle);
    });
    module.querySelector(".module-grab").addEventListener("dragend", () => {
      module.classList.remove("is-dragging"); draggingModule = null; announceModuleOrder(module);
    });
    module.querySelector(".module-up").addEventListener("click", () => {
      const previous = module.previousElementSibling;
      if (previous) { moduleStack.insertBefore(module, previous); announceModuleOrder(module); module.querySelector(".module-up").focus(); }
    });
    module.querySelector(".module-down").addEventListener("click", () => {
      const next = module.nextElementSibling;
      if (next) { moduleStack.insertBefore(next, module); announceModuleOrder(module); module.querySelector(".module-down").focus(); }
    });
  });
  moduleStack.addEventListener("dragover", (event) => {
    if (!draggingModule) return;
    event.preventDefault();
    const target = event.target.closest(".audio-module");
    if (!target || target === draggingModule) return;
    const rect = target.getBoundingClientRect();
    moduleStack.insertBefore(draggingModule, event.clientY < rect.top + rect.height / 2 ? target : target.nextSibling);
  });
  moduleStack.addEventListener("drop", (event) => { event.preventDefault(); if (draggingModule) announceModuleOrder(draggingModule); });
  announceModuleOrder(moduleStack.firstElementChild);

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
      // Reuse a context created by the sound-preview button; browsers limit the
      // number of AudioContexts a page may create.
      live.ctx = live.ctx || new (window.AudioContext || window.webkitAudioContext)();
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

  function selectedSound() { return $("soundSample").value; }

  function ensurePlaybackContext() {
    if (!live.ctx) live.ctx = new (window.AudioContext || window.webkitAudioContext)();
    if (live.ctx.state === "suspended") live.ctx.resume();
    return live.ctx;
  }

  function playSample(key, when, gain) {
    const ctx = ensurePlaybackContext();
    const sig = K.synth.keyboardSample(key, K.CFG.segment.windowMs, selectedSound());
    const buf = ctx.createBuffer(1, sig.length, TARGET_SR);
    const ch = buf.getChannelData(0);
    for (let i = 0; i < sig.length; i++) ch[i] = sig[i] * 0.5;
    const source = ctx.createBufferSource(); source.buffer = buf;
    const volume = ctx.createGain(); volume.gain.value = gain;
    source.connect(volume); volume.connect(ctx.destination);
    source.start(when === undefined ? ctx.currentTime : when);
  }

  function playMasking() {
    if (!live.ctx) return;
    const keys = K.DEFAULT_KEYS;
    for (let d = 0; d < 2; d++) {
      const key = keys[Math.floor(Math.random() * keys.length)];
      playSample(key, live.ctx.currentTime + d * 0.008 + Math.random() * 0.006, 0.9);
    }
  }

  $("previewSoundBtn").addEventListener("click", () => playSample("f", undefined, 0.75));

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
