#!/usr/bin/env node
// Functional check for every WAV example through decode -> onset -> features -> recognizer.
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
const nFrames = K.features.inferNFrames(K.CFG.segment.windowMs);
const train = K.data.buildTrainingSet(K.DEFAULT_KEYS, 12, 212);
const recognizer = new K.models.KNN(3).fit(K.features.featurize(train.clips, nFrames), train.labels);

for (const example of examples) {
  const wav = fs.readFileSync(path.join(repo, 'web/audio', example.file));
  assert.equal(wav.toString('ascii', 0, 4), 'RIFF');
  assert.equal(wav.toString('ascii', 8, 12), 'WAVE');
  assert.equal(wav.readUInt16LE(20), 1, 'PCM format');
  assert.equal(wav.readUInt16LE(22), 1, 'mono audio');
  assert.equal(wav.readUInt32LE(24), 16000, 'sample rate');
  assert.equal(wav.readUInt16LE(34), 16, '16-bit samples');
  assert.equal(wav.readUInt32LE(40) + 44, wav.length, 'data chunk covers the file');

  const samples = new Float64Array((wav.length - 44) / 2);
  for (let i = 0; i < samples.length; i++) samples[i] = wav.readInt16LE(44 + i * 2) / 32768;
  const { clips, onsets } = K.segment.segment(samples);
  assert.equal(onsets.length, Array.from(example.text).length, `${example.file} onset count`);
  for (let i = 1; i < onsets.length; i++) assert.ok(onsets[i] > onsets[i - 1], 'timestamps stay ordered');
  const prediction = recognizer.predict(K.features.featurize(clips, nFrames))
    .map((key) => key === '<space>' ? ' ' : key).join('');
  assert.equal(prediction, example.text, `${example.file} labeled sequence`);
  console.log(`WAV check passed: ${example.file} → "${prediction}" (${onsets.length} ordered onsets).`);
}
