#!/usr/bin/env node
// Functional check for the WAV decode -> onset -> features -> recognizer path.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const repo = path.resolve(__dirname, '..');
const wav = fs.readFileSync(path.join(repo, 'web/audio/synthetic-demo.wav'));
assert.equal(wav.toString('ascii', 0, 4), 'RIFF');
assert.equal(wav.toString('ascii', 8, 12), 'WAVE');
assert.equal(wav.readUInt16LE(20), 1, 'PCM format');
assert.equal(wav.readUInt16LE(22), 1, 'mono audio');
assert.equal(wav.readUInt32LE(24), 16000, 'sample rate');
assert.equal(wav.readUInt16LE(34), 16, '16-bit samples');

const browser = {};
browser.window = browser;
vm.runInNewContext(fs.readFileSync(path.join(repo, 'web/js/keydropper.js'), 'utf8'), browser);
const K = browser.KD;
const samples = new Float64Array((wav.length - 44) / 2);
for (let i = 0; i < samples.length; i++) samples[i] = wav.readInt16LE(44 + i * 2) / 32768;
const { clips } = K.segment.segment(samples);
const nFrames = K.features.inferNFrames(K.CFG.segment.windowMs);
const train = K.data.buildTrainingSet(K.DEFAULT_KEYS, 12, 212);
const recognizer = new K.models.KNN(3).fit(K.features.featurize(train.clips, nFrames), train.labels);
const prediction = recognizer.predict(K.features.featurize(clips, nFrames)).join('');
assert.equal(prediction, 'demo', 'the labeled sample should pass through the full recognizer');
console.log('WAV check passed: decoded and recognized "demo".');
