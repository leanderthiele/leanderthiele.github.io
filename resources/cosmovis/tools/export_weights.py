#!/usr/bin/env python3
"""Convert cosmopower CMB pickles to flat .bin files for the WASM kernel.

The pickle layout (from cosmopower/cosmopower_NN.py save()/restore()):

    Direct NN (TT, EE) -- list of 15 entries:
       0  W_                list of n_layers weight matrices, shape (in,out)
       1  b_                list of n_layers biases, shape (out,)
       2  alphas_           list of n_layers-1 activation alphas, shape (out,)
       3  betas_            list of n_layers-1 activation betas,  shape (out,)
       4  parameters_mean_  (n_in,)
       5  parameters_std_   (n_in,)
       6  features_mean_    (n_out,)
       7  features_std_     (n_out,)
       8  n_parameters
       9  parameters        list of parameter names
      10  n_modes
      11  modes             (n_out,) int64
      12  n_hidden          list of hidden-layer widths
      13  n_layers          int (number of weight matrices)
      14  architecture      [n_in, *n_hidden, n_out]

    PCA+NN (TE) -- list of 19 entries:
       0..5  W_, b_, alphas_, betas_, parameters_mean_, parameters_std_
       6     pca_mean_       (N_PCA,)
       7     pca_std_        (N_PCA,)
       8     features_mean_  (n_out,)
       9     features_std_   (n_out,)
      10     parameters      names
      11     n_parameters
      12     modes           (n_out,) int64
      13     n_modes
      14     n_pcas
      15     pca_transform_matrix_  (N_PCA, n_out)
      16     n_hidden
      17     n_layers
      18     architecture

Both pickles were saved with a TF<=2.10 import path. Newer TF moved
tensorflow.python.training.tracking -> tensorflow.python.trackable;
we shim that before unpickling.
"""
from __future__ import annotations

import os
import pickle
import struct
import sys
from pathlib import Path

import numpy as np

# Shim for cosmopower pickles saved with tensorflow<=2.10
import tensorflow.python.trackable as _trk
import tensorflow.python.trackable.data_structures as _trk_ds
sys.modules['tensorflow.python.training.tracking'] = _trk
sys.modules['tensorflow.python.training.tracking.data_structures'] = _trk_ds


COSMOPOWER_DIR = Path('/home/lthiele/cosmopower/cosmopower/trained_models/CP_paper/CMB')

# spectrum -> (filename stem, output_kind, is_pca)
#   output_kind: 0 = linear C_ell, 1 = log10 C_ell
SPECTRA = {
    'TT': ('cmb_TT_NN',        1, False),
    'EE': ('cmb_EE_NN',        1, False),
    'TE': ('cmb_TE_PCAplusNN', 0, True),
}

MAGIC = b'CPNN'
VERSION = 1


def _f32(arr: np.ndarray) -> bytes:
    return np.ascontiguousarray(arr, dtype=np.float32).tobytes()


def _u32(*vals: int) -> bytes:
    return struct.pack('<' + 'I' * len(vals), *vals)


def export_direct_nn(L: list, out_path: Path, output_kind: int) -> None:
    W, b, alphas, betas = L[0], L[1], L[2], L[3]
    p_mean, p_std = L[4], L[5]
    f_mean, f_std = L[6], L[7]
    modes = L[11]
    n_layers = L[13]
    arch = list(L[14])

    assert len(W) == n_layers == len(b) == len(arch) - 1
    assert len(alphas) == len(betas) == n_layers - 1
    n_in, n_out = arch[0], arch[-1]
    assert np.all(np.diff(modes) == 1), 'expected contiguous ell grid'
    ell_min, ell_max = int(modes[0]), int(modes[-1])

    _write_bin(out_path,
               n_in=n_in, n_layers=n_layers, has_pca=0,
               n_outputs=n_out, output_kind=output_kind,
               ell_min=ell_min, ell_max=ell_max,
               arch=arch,
               p_mean=p_mean, p_std=p_std,
               W=W, b=b, alphas=alphas, betas=betas,
               pca_mean=None, pca_std=None, pca_basis=None,
               f_mean=f_mean, f_std=f_std)


def export_pca_nn(L: list, out_path: Path, output_kind: int) -> None:
    W, b, alphas, betas = L[0], L[1], L[2], L[3]
    p_mean, p_std = L[4], L[5]
    pca_mean, pca_std = L[6], L[7]
    f_mean, f_std = L[8], L[9]
    modes = L[12]
    n_pcas = L[14]
    basis = L[15]                       # (N_PCA, n_out)
    n_layers = L[17]
    arch = list(L[18])

    assert len(W) == n_layers == len(b) == len(arch) - 1
    assert len(alphas) == len(betas) == n_layers - 1
    assert arch[-1] == n_pcas == basis.shape[0]
    n_in = arch[0]
    n_out = basis.shape[1]
    assert np.all(np.diff(modes) == 1), 'expected contiguous ell grid'
    ell_min, ell_max = int(modes[0]), int(modes[-1])

    _write_bin(out_path,
               n_in=n_in, n_layers=n_layers, has_pca=1,
               n_outputs=n_out, output_kind=output_kind,
               ell_min=ell_min, ell_max=ell_max,
               arch=arch,
               p_mean=p_mean, p_std=p_std,
               W=W, b=b, alphas=alphas, betas=betas,
               pca_mean=pca_mean, pca_std=pca_std, pca_basis=basis,
               f_mean=f_mean, f_std=f_std)


def _write_bin(path: Path, *, n_in, n_layers, has_pca, n_outputs,
               output_kind, ell_min, ell_max, arch,
               p_mean, p_std, W, b, alphas, betas,
               pca_mean, pca_std, pca_basis, f_mean, f_std) -> None:
    buf = bytearray()
    buf += MAGIC
    buf += _u32(VERSION, n_in, n_layers, has_pca,
                n_outputs, output_kind, ell_min, ell_max)
    buf += _u32(*arch)                          # n_layers+1 dims
    buf += _f32(p_mean) + _f32(p_std)
    for i in range(n_layers):
        in_d, out_d = arch[i], arch[i + 1]
        assert W[i].shape == (in_d, out_d), f'W[{i}] shape {W[i].shape} vs {(in_d,out_d)}'
        assert b[i].shape == (out_d,)
        buf += _f32(W[i])                       # row-major (in, out)
        buf += _f32(b[i])
        if i < n_layers - 1:
            assert alphas[i].shape == (out_d,)
            assert betas[i].shape == (out_d,)
            buf += _f32(alphas[i]) + _f32(betas[i])
    if has_pca:
        assert pca_basis.shape == (arch[-1], n_outputs)
        buf += _f32(pca_mean) + _f32(pca_std)
        buf += _f32(pca_basis)                  # row-major (N_PCA, n_out)
    buf += _f32(f_mean) + _f32(f_std)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf)
    print(f'  wrote {path}  ({len(buf)/1024:.1f} KiB)')


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / 'web' / 'weights'
    for name, (stem, output_kind, is_pca) in SPECTRA.items():
        pkl_path = COSMOPOWER_DIR / f'{stem}.pkl'
        print(f'[{name}] loading {pkl_path.name}')
        with open(pkl_path, 'rb') as f:
            L = pickle.load(f)
        out_path = out_dir / f'{name}.bin'
        if is_pca:
            export_pca_nn(L, out_path, output_kind)
        else:
            export_direct_nn(L, out_path, output_kind)


if __name__ == '__main__':
    main()
