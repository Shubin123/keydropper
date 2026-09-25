#!/usr/bin/env node
// Build a deterministic, clearly synthetic example clip for the browser demo.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const repo = path.resolve(__dirname, '..');
const browser = { console };
browser.window = browser;
vm.runInNewContext(fs.readFileSync(path.join(repo, 'web/js/keydropper.js'), 'utf8'), browser);
const K = browser.KD;
const text = 'demo';
const { samples } = K.synth.renderStream(text, 84219, K.CFG.segment.windowMs, {
  keysPerSecond: 4,
  backgroundNoise: 0.002,
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
fs.writeFileSync(path.join(repo, 'web/audio/synthetic-demo.wav'), wav);
console.log(`Wrote ${text} (${(wav.length / 1024).toFixed(1)} KiB, ${sampleRate} Hz mono PCM).`);
