// CosmoEmulator: loads the WASM kernel + 3 weight .bin files (TT, TE, EE)
// and exposes a single predict({omega_b, omega_m, H_0, tau, n_s, A_s}).
//
// The HTML page only ever talks to this class. WASM memory is allocated
// once at init time; predict() writes into those buffers in place.
//
// User parameterisation (sliders)   ->   cosmopower parameterisation (NN input)
//   omega_b   ->   omega_b   = omega_b                       (physical baryon density)
//   omega_m   ->   omega_cdm = omega_m - omega_b - omega_nu  (physical matter density)
//   H_0       ->   h         = H_0 / 100
//   tau       ->   tau_reio  = tau
//   n_s       ->   n_s       = n_s
//   A_s       ->   ln10^10A_s = ln(1e10 * A_s)
//
// omega_m is the total physical matter density; the emulator's CDM input is
// recovered by subtracting the baryons and the (fixed) single massive
// neutrino used in the CP_paper training, omega_nu = 0.06 eV / 93.14 eV.

import CosmoEmuModule from './cosmo_emu.js';

const SPECTRA = ['TT', 'TE', 'EE'];

export class CosmoEmulator {
  constructor(opts = {}) {
    this.baseUrl = opts.baseUrl ?? './';
    this.wasm = null;
    this.models = {};         // name -> { handle, nOut, outputKind, ellMin, ellMax }
    this.ell = null;          // shared multipole grid (Int32Array)
    this.inPtr = 0;
    this.outPtr = 0;
    this.outScratchLen = 0;
    this.predictions = null;  // pre-allocated output object reused across predict()
    // lazily-grown scratch buffers for synthGrid()
    this.shtClPtr = 0; this.shtClCap = 0;
    this.shtGridPtr = 0; this.shtGridCap = 0;
  }

  async init() {
    this.wasm = await CosmoEmuModule();

    let maxOut = 0;
    let sharedEllMin = null, sharedEllMax = null;

    for (const name of SPECTRA) {
      const url = `${this.baseUrl}weights/${name}.bin`;
      const blob = new Uint8Array(await (await fetch(url)).arrayBuffer());

      const ptr = this.wasm._cp_malloc(blob.byteLength);
      this.wasm.HEAPU8.set(blob, ptr);
      const handle = this.wasm._cp_load(ptr, blob.byteLength);
      this.wasm._cp_mfree(ptr);
      if (handle < 0) throw new Error(`cp_load failed for ${name}`);

      const nOut       = this.wasm._cp_n_outputs(handle);
      const outputKind = this.wasm._cp_output_kind(handle);
      const ellMin     = this.wasm._cp_ell_min(handle);
      const ellMax     = this.wasm._cp_ell_max(handle);

      if (sharedEllMin === null) {
        sharedEllMin = ellMin;
        sharedEllMax = ellMax;
      } else if (ellMin !== sharedEllMin || ellMax !== sharedEllMax) {
        throw new Error(`spectrum ${name} has different ell grid`);
      }

      this.models[name] = { handle, nOut, outputKind, ellMin, ellMax };
      if (nOut > maxOut) maxOut = nOut;
    }

    this.ell = new Int32Array(sharedEllMax - sharedEllMin + 1);
    for (let i = 0; i < this.ell.length; i++) this.ell[i] = sharedEllMin + i;

    this.inPtr  = this.wasm._cp_malloc(6 * 4);
    this.outPtr = this.wasm._cp_malloc(maxOut * 4);
    this.outScratchLen = maxOut;

    this.predictions = { ell: this.ell };
    for (const name of SPECTRA) {
      this.predictions[name] = new Float32Array(this.models[name].nOut);
    }
  }

  // Pack the 6 user-facing params into the cosmopower-native vector in
  // the WASM heap. Returns nothing.
  _packParams({ omega_b, omega_m, H_0, tau, n_s, A_s }) {
    const h = H_0 / 100;
    const OMEGA_NU = 0.000644;                       // fixed 0.06 eV neutrino
    const inView = new Float32Array(this.wasm.HEAPF32.buffer, this.inPtr, 6);
    inView[0] = omega_b;                             // omega_b
    inView[1] = omega_m - omega_b - OMEGA_NU;        // omega_cdm
    inView[2] = h;                                   // h
    inView[3] = tau;                                 // tau_reio
    inView[4] = n_s;                                 // n_s
    inView[5] = Math.log(1e10 * A_s);                // ln(1e10 A_s)
  }

  // Run all 3 spectra. Returns { ell, TT, TE, EE } where each spectrum
  // is a Float32Array of C_ell values (not log).
  // The arrays are reused between calls -- copy if you need to retain them.
  predict(params) {
    this._packParams(params);
    for (const name of SPECTRA) {
      const m = this.models[name];
      this.wasm._cp_predict(m.handle, this.inPtr, this.outPtr);
      const raw = new Float32Array(this.wasm.HEAPF32.buffer, this.outPtr, m.nOut);
      const dst = this.predictions[name];
      if (m.outputKind === 1) {
        // log10 C_ell -> C_ell
        for (let i = 0; i < m.nOut; i++) dst[i] = Math.pow(10, raw[i]);
      } else {
        dst.set(raw);
      }
    }
    return this.predictions;
  }

  // Synthesize a Gaussian random realization of a real field from an angular
  // power spectrum, returning an equirectangular (theta, phi) grid.
  //   clByEll : Float64Array/Array indexed by ell (physical units^2), ell<2 ignored
  //   lmax    : maximum multipole
  //   ntheta  : number of colatitude rings  (row i -> theta = pi*(i+0.5)/ntheta)
  //   nphi    : longitudes per ring (power of two, > 2*lmax)
  //   seed    : 32-bit RNG seed (different seed -> different realization)
  // Returns a Float32Array view (length ntheta*nphi, row-major) into WASM heap;
  // consume it before the next synthGrid()/predict() call.
  synthGrid(clByEll, lmax, ntheta, nphi, seed) {
    const w = this.wasm;
    const nCl = lmax + 1;
    if (this.shtClCap < nCl) {
      if (this.shtClPtr) w._cp_mfree(this.shtClPtr);
      this.shtClPtr = w._cp_malloc(nCl * 4);
      this.shtClCap = nCl;
    }
    const npix = ntheta * nphi;
    if (this.shtGridCap < npix) {
      if (this.shtGridPtr) w._cp_mfree(this.shtGridPtr);
      this.shtGridPtr = w._cp_malloc(npix * 4);
      this.shtGridCap = npix;
    }
    const cl = new Float32Array(w.HEAPF32.buffer, this.shtClPtr, nCl);
    for (let l = 0; l < nCl; l++) cl[l] = clByEll[l] || 0;
    w._sht_synth(this.shtClPtr, lmax, ntheta, nphi, seed | 0, this.shtGridPtr);
    // re-create the view: WASM memory may have grown (detaching the buffer)
    return new Float32Array(w.HEAPF32.buffer, this.shtGridPtr, npix);
  }

  // Raw forward pass for one spectrum, returning whatever the network
  // produces (log10 C_ell or C_ell). Used by spot_check.py.
  predictRaw(name, nativeParams /* Float32Array length 6 */) {
    const m = this.models[name];
    const inView = new Float32Array(this.wasm.HEAPF32.buffer, this.inPtr, 6);
    inView.set(nativeParams);
    this.wasm._cp_predict(m.handle, this.inPtr, this.outPtr);
    return new Float32Array(
      this.wasm.HEAPF32.buffer.slice(this.outPtr, this.outPtr + m.nOut * 4)
    );
  }
}
