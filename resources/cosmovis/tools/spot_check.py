#!/usr/bin/env python3
"""Spot-check the WASM kernel against a pure-numpy reference.

The reference forward pass is implemented straight from
cosmopower/cosmopower_NN.py::forward_pass_np and
cosmopower/cosmopower_PCAplusNN.py::forward_pass_np -- no cosmopower
import (which is heavy and currently fragile under TF 2.21). The WASM
side is driven through a small node harness that loads cosmo_emu.js
and prints the predictions as JSON.

Run with `make verify`.
"""
from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

# Shim for cosmopower pickles saved with tensorflow<=2.10
import tensorflow.python.trackable as _trk
import tensorflow.python.trackable.data_structures as _trk_ds
sys.modules['tensorflow.python.training.tracking'] = _trk
sys.modules['tensorflow.python.training.tracking.data_structures'] = _trk_ds

ROOT = Path(__file__).resolve().parent.parent
COSMOPOWER_DIR = Path('/home/lthiele/cosmopower/cosmopower/trained_models/CP_paper/CMB')

# Pick samples inside our chosen slider ranges (cosmopower-native units).
RNG = np.random.default_rng(0)
N_SAMPLES = 10


def native_param_box() -> dict[str, tuple[float, float]]:
    """Slider ranges expressed in cosmopower-native units, conservative.
    These are inside the training prior; see plan doc for derivation."""
    return {
        # h in [0.5, 0.85], so omega_b = Omega_b * h^2 with Omega_b in
        # [0.02, 0.055] => omega_b in [0.005, 0.0397]
        'omega_b':       (0.0080, 0.0300),
        'omega_cdm':     (0.05,   0.30),
        'h':             (0.50,   0.85),
        'tau_reio':      (0.02,   0.15),
        'n_s':           (0.85,   1.05),
        'ln10^{10}A_s':  (2.30,   3.70),
    }


def reference_forward_direct(L, theta):
    W, b, alphas, betas = L[0], L[1], L[2], L[3]
    pm, ps, fm, fs = L[4], L[5], L[6], L[7]
    n_layers = L[13]
    x = (theta - pm) / ps
    for i in range(n_layers - 1):
        act = x @ W[i] + b[i]
        x = (betas[i] + (1 - betas[i]) / (1 + np.exp(-alphas[i] * act))) * act
    y = x @ W[-1] + b[-1]
    return y * fs + fm


def reference_forward_pca(L, theta):
    W, b, alphas, betas = L[0], L[1], L[2], L[3]
    pm, ps = L[4], L[5]
    pca_mean, pca_std = L[6], L[7]
    fm, fs = L[8], L[9]
    basis = L[15]
    n_layers = L[17]
    x = (theta - pm) / ps
    for i in range(n_layers - 1):
        act = x @ W[i] + b[i]
        x = (betas[i] + (1 - betas[i]) / (1 + np.exp(-alphas[i] * act))) * act
    coeffs = x @ W[-1] + b[-1]
    return ((coeffs * pca_std + pca_mean) @ basis) * fs + fm


def load_pickles():
    out = {}
    out['TT'] = (
        pickle.loads((COSMOPOWER_DIR / 'cmb_TT_NN.pkl').read_bytes()),
        reference_forward_direct,
    )
    out['EE'] = (
        pickle.loads((COSMOPOWER_DIR / 'cmb_EE_NN.pkl').read_bytes()),
        reference_forward_direct,
    )
    out['TE'] = (
        pickle.loads((COSMOPOWER_DIR / 'cmb_TE_PCAplusNN.pkl').read_bytes()),
        reference_forward_pca,
    )
    return out


def sample_params(n):
    box = native_param_box()
    keys = ['omega_b', 'omega_cdm', 'h', 'tau_reio', 'n_s', 'ln10^{10}A_s']
    samples = np.empty((n, 6), dtype=np.float32)
    for j, k in enumerate(keys):
        lo, hi = box[k]
        samples[:, j] = RNG.uniform(lo, hi, size=n)
    return samples


def run_wasm(samples):
    """Drive cosmo_emu.js via node. Returns {name: ndarray (n, n_out)}."""
    harness = ROOT / 'tools' / '_node_harness.mjs'
    payload = json.dumps({'samples': samples.tolist()})
    proc = subprocess.run(
        ['node', '--experimental-vm-modules', str(harness)],
        input=payload, capture_output=True, text=True, check=True,
        cwd=ROOT / 'web',
    )
    result = json.loads(proc.stdout)
    return {k: np.asarray(v, dtype=np.float32) for k, v in result.items()}


def main():
    pickles = load_pickles()
    samples = sample_params(N_SAMPLES)
    wasm_out = run_wasm(samples)

    ok = True
    for name, (L, fwd) in pickles.items():
        ref = np.stack([fwd(L, samples[i]) for i in range(N_SAMPLES)], axis=0)
        got = wasm_out[name]
        diff = np.abs(ref - got)
        rel  = diff / np.maximum(np.abs(ref), 1e-30)
        print(f'[{name}] shape ref={ref.shape} got={got.shape}')
        print(f'       max |abs diff| = {diff.max():.3e}')
        print(f'       max |rel diff| = {rel.max():.3e}')
        # Network output is in the emulator's natural units (log10 for TT/EE,
        # linear for TE). The threshold reflects float32 round-off across
        # ~5 dense layers + (optionally) PCA reconstruction.
        threshold = 1e-3
        if diff.max() > threshold and rel.max() > threshold:
            print(f'       FAIL: exceeds {threshold:.0e}')
            ok = False
        else:
            print(f'       OK')
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
