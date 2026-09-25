#!/usr/bin/env node
// Exercise phrase generation, PCM WAV encoding, timestamp detection, and inference
// for every simulated keyboard profile.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const repo = path.resolve(__dirname, '..');
const examples = JSON.parse(fs.readFileSync(path.join(__dirname, 'example_wavs.json'), 'utf8'));
const browser = {};
browser.window = browser;
vm.runInNewContext(fs.readFileSync(path.join(repo, 'web/js/keydropper.js'), 'utf8'), browser);
const K = browser.KD;
function encodeWav(samples, sampleRate) {
  const buffer = Buffer.alloc(44 + samples.length * 2), dataBytes = samples.length * 2;
  buffer.write('RIFF', 0); buffer.writeUInt32LE(36 + dataBytes, 4); buffer.write('WAVE', 8);
  buffer.write('fmt ', 12); buffer.writeUInt32LE(16, 16); buffer.writeUInt16LE(1, 20);
  buffer.writeUInt16LE(1, 22); buffer.writeUInt32LE(sampleRate, 24); buffer.writeUInt32LE(sampleRate * 2, 28);
  buffer.writeUInt16LE(2, 32); buffer.writeUInt16LE(16, 34); buffer.write('data', 36); buffer.writeUInt32LE(dataBytes, 40);
  for (let i = 0; i < samples.length; i++) {
    const value = Math.max(-1, Math.min(1, samples[i]));
    buffer.writeInt16LE(Math.round(value * (value < 0 ? 32768 : 32767)), 44 + i * 2);
  }
  return buffer;
}

const sampleRate = K.CFG.audio.sr, nFrames = K.features.inferNFrames(K.CFG.segment.windowMs);
let reference = null, mechanicalDemo = null;
for (const [keyboardType, profile] of Object.entries(K.SYNTH_KEYBOARDS)) {
  const trainClips = [], trainLabels = [];
  for (const key of K.DEFAULT_KEYS) for (let i = 0; i < 12; i++) {
    trainClips.push(K.synth.keySignature(key, K.CFG.segment.windowMs, new K.Rng(212 + i + key.charCodeAt(0)), profile.seed, keyboardType));
    trainLabels.push(key);
  }
  const recognizer = new K.models.KNN(3).fit(K.features.featurize(trainClips, nFrames), trainLabels);
  for (const example of examples) {
    const { samples } = K.synth.renderStream(example.text, example.seed, K.CFG.segment.windowMs, {
      keysPerSecond: example.keysPerSecond, backgroundNoise: example.backgroundNoise,
      keyboardType, keyboardSeed: profile.seed,
    });
    const wav = encodeWav(samples, sampleRate);
    assert.equal(wav.toString('ascii', 0, 4), 'RIFF');
    assert.equal(wav.toString('ascii', 8, 12), 'WAVE');
    assert.equal(wav.readUInt32LE(24), sampleRate);
    assert.equal(wav.readUInt32LE(40) + 44, wav.length);
    if (example === examples[0]) {
      if (reference) assert.notDeepEqual(wav, reference, `${keyboardType} should produce a different keyboard sound`);
      reference = wav;
    }
    const decoded = new Float64Array((wav.length - 44) / 2);
    for (let i = 0; i < decoded.length; i++) decoded[i] = wav.readInt16LE(44 + i * 2) / 32768;
    const { clips, onsets } = K.segment.segment(decoded);
    assert.equal(onsets.length, Array.from(example.text).length, `${keyboardType}/${example.text} onset count`);
    for (let i = 1; i < onsets.length; i++) assert.ok(onsets[i] > onsets[i - 1], 'timestamps stay ordered');
    const prediction = recognizer.predict(K.features.featurize(clips, nFrames));
    assert.equal(prediction.length, onsets.length, `${keyboardType}/${example.text} prediction count`);
    const recognizedText = prediction.map((key) => key === '<space>' ? ' ' : key).join('');
    assert.equal(recognizedText, example.text, `${keyboardType}/${example.text} labeled sequence`);
    if (keyboardType === 'mechanical' && example.text === 'demo') {
      assert.equal(recognizedText, 'demo');
      mechanicalDemo = wav;
    }
    console.log(`${profile.label}: ${example.text} → ${prediction.length} predictions / ${onsets.length} ordered timestamps.`);
  }
}

const referenceWav = fs.readFileSync(path.join(repo, 'web/audio/synthetic-demo.wav'));
assert.equal(referenceWav.toString('ascii', 0, 4), 'RIFF');
assert.equal(referenceWav.readUInt32LE(24), sampleRate);
assert.deepEqual(referenceWav, mechanicalDemo, 'the downloadable reference should match the generated mechanical demo');
