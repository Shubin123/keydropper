#!/usr/bin/env node
// Build deterministic, clearly synthetic WAV examples for the browser demo.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const repo = path.resolve(__dirname, '..');
const examples = JSON.parse(fs.readFileSync(path.join(__dirname, 'example_wavs.json'), 'utf8'));
const browser = { console };
browser.window = browser;
vm.runInNewContext(fs.readFileSync(path.join(repo, 'web/js/keydropper.js'), 'utf8'), browser);
const K = browser.KD;
const outputDir = path.join(repo, 'web/audio');
fs.mkdirSync(outputDir, { recursive: true });

for (const example of examples) {
  const { samples } = K.synth.renderStream(example.text, example.seed, K.CFG.segment.windowMs, {
    keysPerSecond: example.keysPerSecond,
    backgroundNoise: example.backgroundNoise,
  });
  const sampleRate = K.CFG.audio.sr;
  const dataBytes = samples.length * 2;
  const wav = Buffer.alloc(44 + dataBytes);
  const put = (offset, value) => wav.writeUInt32LE(value, offset);
  wav.write('RIFF', 0); put(4, 36 + dataBytes); wav.write('WAVE', 8);
  wav.write('fmt ', 12); put(16, 16); wav.writeUInt16LE(1, 20);
  wav.writeUInt16LE(1, 22); put(24, sampleRate); put(28, sampleRate * 2);
  wav.writeUInt16LE(2, 32); wav.writeUInt16LE(16, 34);
  wav.write('data', 36); put(40, dataBytes);
  for (let i = 0; i < samples.length; i++) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    wav.writeInt16LE(Math.round(clamped * (clamped < 0 ? 32768 : 32767)), 44 + i * 2);
  }
  fs.writeFileSync(path.join(outputDir, example.file), wav);
  console.log(`Wrote ${example.text} (${(wav.length / 1024).toFixed(1)} KiB, ${sampleRate} Hz mono PCM).`);
}
