// Driven by tools/spot_check.py. Reads {samples: [[6 floats], ...]} from
// stdin, runs the raw (network-native) forward pass for each spectrum,
// and prints {TT: [[...], ...], TE: [...], EE: [...]} on stdout.
//
// "Network-native" means: log10 C_ell for TT/EE, linear C_ell for TE --
// matching the python reference in spot_check.py.

import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const webDir = resolve(here, '..', 'web');

// emscripten ES6 module loader
const { CosmoEmulator } = await import(resolve(webDir, 'cosmo_emu.mjs'));

// In node, the module's `fetch()` of './weights/X.bin' won't work, so
// monkey-patch a tiny fetch shim that reads from disk relative to webDir.
globalThis.fetch = async (url) => {
  const path = resolve(webDir, url.startsWith('./') ? url.slice(2) : url);
  if (!existsSync(path)) throw new Error(`not found: ${path}`);
  const bytes = readFileSync(path);
  return {
    arrayBuffer: async () =>
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  };
};

const emu = new CosmoEmulator({ baseUrl: './' });
await emu.init();

const stdin = readFileSync(0, 'utf-8');
const { samples } = JSON.parse(stdin);

const out = { TT: [], TE: [], EE: [] };
for (const s of samples) {
  const arr = new Float32Array(s);
  for (const name of ['TT', 'TE', 'EE']) {
    const y = emu.predictRaw(name, arr);
    out[name].push(Array.from(y));
  }
}
process.stdout.write(JSON.stringify(out));
