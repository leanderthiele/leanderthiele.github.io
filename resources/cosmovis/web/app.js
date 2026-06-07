// Top-level page logic: build sliders, drive the emulator, paint the plots.

import { CosmoEmulator } from './cosmo_emu.mjs';

// ---- parameter definitions ----------------------------------------------
// Slider ranges are chosen as a fixed Cartesian sub-box of the cosmopower
// training prior so any slider can be moved independently without bounds
// violations. A_s is deliberately restricted to +-10% of its default so the
// overall spectrum amplitude only varies modestly and the fixed y-axes stay
// nicely framed (the line moves, the axis does not).
const PARAMS = [
  { key: 'Omega_b',   label: 'Ω<sub>b</sub>',
                                    min: 0.020, max: 0.055, def: 0.0490, step: 0.0001, dps: 4 },
  { key: 'Omega_cdm', label: 'Ω<sub>cdm</sub>',
                                    min: 0.05,  max: 0.30,  def: 0.264,  step: 0.001,  dps: 3 },
  { key: 'H_0',       label: 'H<sub>0</sub>',
                                    min: 50,    max: 85,    def: 67.4,   step: 0.1,    dps: 1 },
  { key: 'tau',       label: 'τ',
                                    min: 0.02,  max: 0.15,  def: 0.054,  step: 0.001,  dps: 3 },
  { key: 'n_s',       label: 'n<sub>s</sub>',
                                    min: 0.85,  max: 1.05,  def: 0.965,  step: 0.001,  dps: 3 },
  { key: 'A_s',       label: 'A<sub>s</sub>',
                                    min: 1.89e-9, max: 2.31e-9, def: 2.1e-9, step: 5e-12, dps: 4 },
];

// ---- UI: build sliders + numeric inputs ---------------------------------
const slidersEl = document.getElementById('sliders');
const statusEl  = document.getElementById('status');
const ui = {};   // key -> {range, number, def, get(), param}

function buildSliders() {
  for (const p of PARAMS) {
    const row = document.createElement('div');
    row.className = 'slider-row';

    const label = document.createElement('label');
    label.innerHTML = p.label;
    label.classList.add('reset-label');
    label.title = 'click to reset to default';

    const range = document.createElement('input');
    range.type = 'range';
    range.min = String(p.min);
    range.max = String(p.max);
    range.step = String(p.step);
    range.value = String(p.def);

    const number = document.createElement('input');
    number.type = 'number';
    number.min  = String(p.min);
    number.max  = String(p.max);
    number.step = String(p.step);
    number.value = formatVal(p.def, p);

    range.addEventListener('input', () => {
      number.value = formatVal(parseFloat(range.value), p);
      schedule();
    });
    number.addEventListener('change', () => {
      let v = parseFloat(number.value);
      if (!isFinite(v)) v = p.def;
      v = Math.max(p.min, Math.min(p.max, v));
      number.value = formatVal(v, p);
      range.value = String(v);
      schedule();
    });

    label.addEventListener('click', () => {
      range.value = String(p.def);
      number.value = formatVal(p.def, p);
      schedule();
    });

    row.append(label, range, number);
    slidersEl.appendChild(row);

    ui[p.key] = { range, number, def: p.def, param: p,
                  get: () => parseFloat(range.value) };
  }
}

function formatVal(v, p) {
  if (p.key === 'A_s') return v.toExponential(2);
  return v.toFixed(p.dps);
}

// ---- emulator init + redraw loop ----------------------------------------
const emu = new CosmoEmulator({ baseUrl: './' });

let scheduled = false;
function schedule() {
  if (scheduled) return;
  scheduled = true;
  requestAnimationFrame(() => { scheduled = false; redraw(); });
}

const plots = {};

function currentParams() {
  const params = {};
  for (const p of PARAMS) params[p.key] = ui[p.key].get();
  return params;
}

// Push a parameter set onto the sliders + numeric inputs, then redraw.
function applyParams(params) {
  for (const p of PARAMS) {
    let v = params[p.key];
    if (!isFinite(v)) continue;
    v = Math.max(p.min, Math.min(p.max, v));
    ui[p.key].range.value = String(v);
    ui[p.key].number.value = formatVal(v, p);
  }
  redraw();
}

const SPECTRA = ['TT', 'TE', 'EE'];

// ---- saved comparison curves ---------------------------------------------
// The live (currently-adjusted) spectrum is always drawn in blue on top.
// "Saved" entries are frozen snapshots drawn underneath in their own colour,
// each with a legend chip at the bottom of the page.
const LIVE_COLOR = '#0a3d91';
// distinct, blue-free palette; black is reserved for the default entry
const PALETTE = ['#d62728', '#2ca02c', '#ff7f0e', '#9467bd',
                 '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#7f7f7f'];

let saved = [];      // [{ id, name, color, params, spectra:{ell,TT,TE,EE} }]
let entrySeq = 0;

function snapshotSpectra(params) {
  const out = emu.predict(params);
  return { ell: out.ell, TT: out.TT.slice(), TE: out.TE.slice(), EE: out.EE.slice() };
}

function nextColor() {
  const used = new Set(saved.map((e) => e.color));
  for (const c of PALETTE) if (!used.has(c)) return c;
  return `hsl(${(saved.length * 67) % 360}, 65%, 42%)`;
}

function addEntry(name, color, params) {
  saved.push({ id: ++entrySeq, name, color, params: { ...params },
               spectra: snapshotSpectra(params) });
  renderLegend();
  redraw();
}

function deleteEntry(id) {
  saved = saved.filter((e) => e.id !== id);
  renderLegend();
  redraw();
}

function redraw() {
  const out = emu.predict(currentParams());
  for (const name of SPECTRA) {
    const series = saved.map((e) => ({ ell: e.spectra.ell, C: e.spectra[name],
                                       color: e.color, width: 1.2 }));
    series.push({ ell: out.ell, C: out[name], color: LIVE_COLOR, width: 1.7 });
    plots[name].draw(series);
  }
}

// ---- plotting (vanilla canvas, no deps) ---------------------------------
// D_ell = ell*(ell+1)*C_ell/(2 pi) * T_CMB^2, in (microK)^2.
// T_CMB = 2.7255e6 microK ; T_CMB^2 = 7.4283e12 microK^2.
const TCMB2 = 7.4283e12;

function toD(ell, C) {
  const N = ell.length;
  const D = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const L = ell[i];
    D[i] = L * (L + 1) * C[i] / (2 * Math.PI) * TCMB2;
  }
  return D;
}

// Planck-style broken multipole axis: linear throughout, but with two
// different linear scales -- 2 <= ell <= L_SPLIT is stretched across the
// first FRAC of the width, L_SPLIT <= ell <= ell_max fills the remainder.
const L_SPLIT = 30;
const FRAC = 0.2;

// Top-axis angular-scale ticks: theta ~ 180 deg / ell.
const ANGULAR = [90, 18, 2, 1, 0.5, 0.2, 0.1];

class SpectrumPlot {
  constructor(canvas, opts) {
    this.canvas = canvas;
    this.opts = opts;       // { yLabel: [segments], signed, allowLog }
    this.ctx = canvas.getContext('2d');
    this.log = false;
    this.lowZoom = 1;       // magnification applied to the ell<L_SPLIT band
    // fixed data bounds, filled by observe()/finalize()
    this.dMin = Infinity; this.dMax = -Infinity; this.dPosMin = Infinity;
    this.dLowMax = 0;       // max |D| over the low-ell band, across the box
    this.resize();
    window.addEventListener('resize', () => { this.resize(); this.redraw(); });
    this._last = null;
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width  = Math.round(rect.width  * dpr);
    this.canvas.height = Math.round(rect.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.W = rect.width;
    this.H = rect.height;
  }

  setLog(on) { this.log = on; }

  // Accumulate data extremes while sampling the parameter box so the axis
  // can be fixed up front.
  observe(ell, D) {
    for (let i = 0; i < D.length; i++) {
      const v = D[i];
      if (v < this.dMin) this.dMin = v;
      if (v > this.dMax) this.dMax = v;
      if (v > 0 && v < this.dPosMin) this.dPosMin = v;
      if (ell[i] < L_SPLIT) {
        const a = Math.abs(v);
        if (a > this.dLowMax) this.dLowMax = a;
      }
    }
  }

  finalize() {
    // Tighten the corner-sampled extremes so the default spectra fill the
    // frame; extreme slider combos may clip slightly at the edge.
    // (0.7 once, then another 30% -> 0.7*0.7.)
    const SHRINK = 0.49;
    // linear range, padded; include 0 always so the baseline is visible
    let lo = Math.min(0, this.dMin) * SHRINK, hi = Math.max(0, this.dMax) * SHRINK;
    const pad = (hi - lo) * 0.05 || 1;
    this.yLo = lo - pad;
    this.yHi = hi + pad;
    // log range (positive data only), padded by a fixed factor
    this.yLoLog = this.dPosMin / 1.4;
    this.yHiLog = this.dMax * SHRINK * 1.4;

    // Pick a magnification for the low-ell band so its largest value over the
    // whole parameter box fills ~85% of the available half-axis, then round
    // down to a tidy 1/2/5 x 10^k so the live curve never clips.
    if (this.opts.lowZoom && this.dLowMax > 0) {
      const half = this.opts.signed ? Math.min(this.yHi, -this.yLo) : this.yHi;
      this.lowZoom = Math.max(1, niceFloor((half * 0.85) / this.dLowMax));
    }
  }

  redraw() { if (this._last) this.draw(this._last); }

  // series: [{ ell, C, color, width }] drawn back-to-front.
  draw(series) {
    this._last = series;
    const ctx = this.ctx, W = this.W, H = this.H;
    ctx.clearRect(0, 0, W, H);
    if (!series.length) return;

    const ell = series[0].ell;
    const N = ell.length;

    const padL = 64, padR = 14, padT = 40, padB = 34;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;
    const xMin = ell[0], xMax = ell[N - 1];

    // ---- x mapping (two linear scales meeting at L_SPLIT) ----
    const xFrac = (l) => l <= L_SPLIT
      ? (l - xMin) / (L_SPLIT - xMin) * FRAC
      : FRAC + (l - L_SPLIT) / (xMax - L_SPLIT) * (1 - FRAC);
    const x2px = (l) => padL + xFrac(l) * plotW;

    // ---- y mapping (linear or log) ----
    const useLog = this.log;
    const yLo = useLog ? this.yLoLog : this.yLo;
    const yHi = useLog ? this.yHiLog : this.yHi;
    const lLo = useLog ? Math.log10(yLo) : 0;
    const lHi = useLog ? Math.log10(yHi) : 0;
    const y2px = useLog
      ? (y) => padT + (1 - (Math.log10(Math.max(y, yLo)) - lLo) / (lHi - lLo)) * plotH
      : (y) => padT + (1 - (y - yLo) / (yHi - yLo)) * plotH;

    // ---- bounding box ----
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 1;
    ctx.strokeRect(padL, padT, plotW, plotH);

    // split indicator between the linear and log regions
    const xSplit = x2px(L_SPLIT);
    ctx.strokeStyle = '#ccc';
    ctx.beginPath();
    ctx.moveTo(xSplit, padT); ctx.lineTo(xSplit, padT + plotH);
    ctx.stroke();

    // y=0 reference line for signed spectra (linear only)
    if (this.opts.signed && !useLog && yLo < 0 && yHi > 0) {
      ctx.strokeStyle = '#888';
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(padL, y2px(0)); ctx.lineTo(padL + plotW, y2px(0));
      ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.fillStyle = '#000';
    ctx.strokeStyle = '#000';

    // ---- y ticks ----
    // In log mode label only the decades (10^k) and draw short unlabeled
    // minor ticks at 2,5; in linear mode every tick is labelled.
    const yTicks = useLog
      ? decadeTicks(yLo, yHi)
      : niceTicks(yLo, yHi, 5).map((v) => ({ v, major: true }));
    for (const { v, major } of yTicks) {
      const py = y2px(v);
      if (py < padT - 0.5 || py > padT + plotH + 0.5) continue;
      ctx.beginPath(); ctx.moveTo(padL - (major ? 4 : 2.5), py); ctx.lineTo(padL, py); ctx.stroke();
      if (major) drawRich(ctx, tickSegments(v), padL - 7, py, 'right');
    }

    // ---- bottom (multipole) ticks ----
    ctx.font = '11px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const xTicks = [2, 10, 30, 100, 500, 1000, 1500, 2000, 2500]
      .filter((l) => l >= xMin && l <= xMax);
    for (const l of xTicks) {
      const px = x2px(l);
      ctx.beginPath(); ctx.moveTo(px, padT + plotH); ctx.lineTo(px, padT + plotH + 4); ctx.stroke();
      ctx.fillText(String(l), px, padT + plotH + 6);
    }
    ctx.fillText('multipole ℓ', padL + plotW / 2, H - 13);

    // ---- top (angular scale) ticks ----
    ctx.textBaseline = 'bottom';
    for (const deg of ANGULAR) {
      const l = Math.round(180 / deg);
      if (l < xMin || l > xMax) continue;
      const px = x2px(l);
      ctx.beginPath(); ctx.moveTo(px, padT); ctx.lineTo(px, padT - 4); ctx.stroke();
      ctx.fillText(deg + '°', px, padT - 6);
    }
    ctx.textBaseline = 'top';
    ctx.fillText('angular scale θ ≈ 180°/ℓ', padL + plotW / 2, 2);

    // ---- y-axis title (rotated, with proper sub/superscripts) ----
    ctx.save();
    ctx.translate(16, padT + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    drawRich(ctx, this.opts.yLabel, 0, 0, 'center');
    ctx.restore();

    // ---- the spectra (saved curves first, live curve on top) ----
    // The low-ell band is magnified by this.lowZoom; the line is broken at
    // L_SPLIT so the two scales never join across the discontinuity.
    const zoom = this.lowZoom;
    for (const s of series) {
      const D = toD(s.ell, s.C);
      ctx.strokeStyle = s.color;
      ctx.lineWidth = s.width || 1.4;
      ctx.beginPath();
      let started = false, prevLow = null;
      for (let i = 0; i < N; i++) {
        const low = s.ell[i] < L_SPLIT;
        const dv = low ? D[i] * zoom : D[i];
        if (useLog && dv <= 0) { started = false; prevLow = low; continue; }
        const px = x2px(s.ell[i]);
        const py = y2px(dv);
        if (!started || low !== prevLow) { ctx.moveTo(px, py); started = true; }
        else ctx.lineTo(px, py);
        prevLow = low;
      }
      ctx.stroke();
    }

    // ---- low-ell magnification annotation ----
    if (zoom > 1) {
      ctx.fillStyle = '#444';
      ctx.font = 'italic 11px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText('ℓ<30  ×' + zoom, padL + (FRAC * plotW) / 2, padT + 6);
    }
  }
}

// Draw a sequence of text segments {t, kind: 'n'|'sub'|'sup'} starting at
// (x, y). align 'center' centres the whole run horizontally on x.
function drawRich(ctx, segs, x, y, align) {
  const base = '12px system-ui, sans-serif';
  const small = '9px system-ui, sans-serif';
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'left';
  let total = 0;
  for (const s of segs) {
    ctx.font = s.kind === 'n' ? base : small;
    total += ctx.measureText(s.t).width;
  }
  let cx = align === 'center' ? x - total / 2 : align === 'right' ? x - total : x;
  for (const s of segs) {
    ctx.font = s.kind === 'n' ? base : small;
    const dy = s.kind === 'sub' ? 3 : s.kind === 'sup' ? -4 : 0;
    ctx.fillText(s.t, cx, y + dy);
    cx += ctx.measureText(s.t).width;
  }
}

function yLabelSegments(kind) {
  return [
    { t: 'D', kind: 'n' },
    { t: 'ℓ', kind: 'sub' },
    { t: kind, kind: 'sup' },
    { t: ' [μK', kind: 'n' },
    { t: '2', kind: 'sup' },
    { t: ']', kind: 'n' },
  ];
}

function niceTicks(lo, hi, target) {
  const span = hi - lo;
  if (span <= 0) return [lo];
  const rough = span / target;
  const mag = Math.pow(10, Math.floor(Math.log10(rough)));
  const norm = rough / mag;
  let step;
  if      (norm < 1.5) step = 1 * mag;
  else if (norm < 3)   step = 2 * mag;
  else if (norm < 7)   step = 5 * mag;
  else                 step = 10 * mag;
  const first = Math.ceil(lo / step) * step;
  const out = [];
  for (let v = first; v <= hi + 1e-9 * Math.abs(hi); v += step) out.push(v);
  return out;
}

function decadeTicks(lo, hi) {
  const out = [];
  const k0 = Math.floor(Math.log10(lo));
  const k1 = Math.ceil(Math.log10(hi));
  for (let k = k0; k <= k1; k++) {
    const base = Math.pow(10, k);
    for (const m of [1, 2, 5]) {
      const v = m * base;
      if (v >= lo && v <= hi) out.push({ v, major: m === 1 });
    }
  }
  return out;
}

// Tick label as rich-text segments. Values needing scientific notation are
// rendered as m×10^e (or just 10^e) with a real superscript exponent.
function tickSegments(v) {
  if (v === 0) return [{ t: '0', kind: 'n' }];
  const a = Math.abs(v);
  if (a >= 1000 || a < 0.01) {
    const exp = Math.floor(Math.log10(a));
    const mant = Math.round((v / Math.pow(10, exp)) * 100) / 100;
    const expStr = String(exp).replace('-', '−');
    if (mant === 1)  return [{ t: '10', kind: 'n' },  { t: expStr, kind: 'sup' }];
    if (mant === -1) return [{ t: '−10', kind: 'n' }, { t: expStr, kind: 'sup' }];
    const mantStr = String(mant).replace('-', '−');
    return [{ t: mantStr + '×10', kind: 'n' }, { t: expStr, kind: 'sup' }];
  }
  return [{ t: Number.isInteger(v) ? String(v) : formatTick(v), kind: 'n' }];
}

// Largest tidy 1/2/5 x 10^k that is <= x (>=1 callers clamp separately).
function niceFloor(x) {
  if (!(x > 0)) return 1;
  const base = Math.pow(10, Math.floor(Math.log10(x)));
  for (const m of [5, 2, 1]) if (m * base <= x + 1e-9) return m * base;
  return base;
}

function formatTick(v) {
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 1000 || a < 0.01) return v.toExponential(0);
  if (a >= 100) return v.toFixed(0);
  if (a >= 10)  return v.toFixed(1);
  return v.toFixed(2);
}

// Sample the corners of the parameter box (plus the default) once so each
// plot can fix its y-range to comfortably contain every reachable spectrum.
function calibrateRanges() {
  const combos = [];
  const n = PARAMS.length;
  for (let mask = 0; mask < (1 << n); mask++) {
    const params = {};
    PARAMS.forEach((p, i) => { params[p.key] = (mask >> i & 1) ? p.max : p.min; });
    combos.push(params);
  }
  const def = {};
  for (const p of PARAMS) def[p.key] = p.def;
  combos.push(def);

  for (const params of combos) {
    const out = emu.predict(params);
    for (const name of SPECTRA) plots[name].observe(out.ell, toD(out.ell, out[name]));
  }
  for (const name of ['TT', 'TE', 'EE']) plots[name].finalize();
}

// ---- legend (fixed bottom bar) ------------------------------------------
const legendEl  = document.getElementById('legend');
const controlsEl = document.getElementById('controls');

function renderLegend() {
  legendEl.innerHTML = '';
  if (!saved.length) {
    const hint = document.createElement('span');
    hint.className = 'legend-empty';
    hint.textContent = 'no saved curves — enter a name above and press Save';
    legendEl.appendChild(hint);
  }
  for (const e of saved) {
    const item = document.createElement('span');
    item.className = 'legend-item';
    item.title = 'click to load these parameters';
    item.addEventListener('click', () => applyParams(e.params));

    const dot = document.createElement('span');
    dot.className = 'legend-dot';
    dot.style.background = e.color;

    const name = document.createElement('span');
    name.className = 'legend-name';
    name.textContent = e.name;

    const del = document.createElement('span');
    del.className = 'legend-del';
    del.textContent = '×';
    del.title = 'remove from plot';
    del.addEventListener('click', (ev) => { ev.stopPropagation(); deleteEntry(e.id); });

    const tip = document.createElement('div');
    tip.className = 'legend-tip';
    for (const p of PARAMS) {
      const row = document.createElement('div');
      row.innerHTML = `${p.label} = ${formatVal(e.params[p.key], p)}`;
      tip.appendChild(row);
    }

    item.append(dot, name, del, tip);
    legendEl.appendChild(item);
  }
  adjustPadding();
}

// Keep the page clear of the fixed top/bottom bars, whose heights vary with
// window width (slider grid reflow) and legend wrapping.
function adjustPadding() {
  document.body.style.paddingTop = controlsEl.offsetHeight + 'px';
  document.body.style.paddingBottom = legendEl.offsetHeight + 'px';
}

// ---- boot ---------------------------------------------------------------
async function main() {
  buildSliders();

  // save-current-parameters control
  const memoInput = document.getElementById('memo-name');
  const saveBtn   = document.getElementById('save-btn');
  const doSave = () => {
    const name = memoInput.value.trim() || `memo ${saved.length + 1}`;
    addEntry(name, nextColor(), currentParams());
    memoInput.value = '';
  };
  saveBtn.addEventListener('click', doSave);
  memoInput.addEventListener('keydown', (ev) => { if (ev.key === 'Enter') doSave(); });

  plots.TT = new SpectrumPlot(document.getElementById('canvas-TT'),
                              { yLabel: yLabelSegments('TT'), signed: false, allowLog: true });
  plots.TE = new SpectrumPlot(document.getElementById('canvas-TE'),
                              { yLabel: yLabelSegments('TE'), signed: true, lowZoom: true });
  plots.EE = new SpectrumPlot(document.getElementById('canvas-EE'),
                              { yLabel: yLabelSegments('EE'), signed: false, lowZoom: true });

  const ttLog = document.getElementById('tt-log');
  ttLog.addEventListener('change', () => { plots.TT.setLog(ttLog.checked); plots.TT.redraw(); });

  try {
    await emu.init();
  } catch (e) {
    statusEl.textContent = 'Failed to load emulator: ' + e.message;
    throw e;
  }
  calibrateRanges();

  // pre-populate with the default-parameter curve, drawn in black
  const defaults = {};
  for (const p of PARAMS) defaults[p.key] = p.def;
  addEntry('default', '#000000', defaults);   // also triggers first redraw

  statusEl.textContent = '';
  adjustPadding();
  window.addEventListener('resize', adjustPadding);
}

main();
