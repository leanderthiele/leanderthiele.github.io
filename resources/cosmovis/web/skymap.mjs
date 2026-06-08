// SkyMap: render a Gaussian realization of a sky map (Mollweide projection).
//
// The heavy lifting -- drawing a_lm and the spherical-harmonic synthesis --
// happens in WASM (emu.synthGrid), which returns an equirectangular
// (theta, phi) grid. Here we only reproject that grid onto the iconic
// Mollweide ellipse, apply a diverging colour map, and draw a colour bar.
// No external dependencies.

const TWO_PI = 2 * Math.PI;

// Diverging blue-white-red colour map (cool->warm), t in [-1, 1].
const C_NEG = [ 59,  76, 192];   // cold
const C_MID = [240, 240, 240];   // zero
const C_POS = [180,  24,  38];   // hot
function colormap(t, rgb, o) {
  if (t < -1) t = -1; else if (t > 1) t = 1;
  let a, b, f;
  if (t < 0) { a = C_MID; b = C_NEG; f = -t; }
  else       { a = C_MID; b = C_POS; f =  t; }
  rgb[o]     = (a[0] + (b[0] - a[0]) * f) | 0;
  rgb[o + 1] = (a[1] + (b[1] - a[1]) * f) | 0;
  rgb[o + 2] = (a[2] + (b[2] - a[2]) * f) | 0;
}

export class SkyMap {
  constructor(canvas, emu) {
    this.canvas = canvas;
    this.emu = emu;
    this.ctx = canvas.getContext('2d');

    // fixed internal raster (decoupled from on-screen CSS size); the ellipse
    // is 2:1, with a colour-bar strip beneath it.
    this.mapW = 760;
    this.mapH = 380;
    this.cbH  = 46;
    canvas.width  = this.mapW;
    canvas.height = this.mapH + this.cbH;

    this.halfH = this.mapH / 2 - 8;
    this.halfW = 2 * this.halfH;
    this.cx = this.mapW / 2;
    this.cy = this.mapH / 2;
  }

  // clByEll: array indexed by ell (microK^2). lmax: max multipole. seed: int.
  render(clByEll, lmax, seed) {
    const nphi = 1 << Math.ceil(Math.log2(2 * lmax + 2));
    const ntheta = nphi / 2;
    const grid = this.emu.synthGrid(clByEll, lmax, ntheta, nphi, seed);

    // area-weighted RMS for the colour scale (rings weighted by sin theta)
    let sw = 0, sw2 = 0;
    for (let i = 0; i < ntheta; i++) {
      const th = Math.PI * (i + 0.5) / ntheta, wi = Math.sin(th);
      let row2 = 0;
      const base = i * nphi;
      for (let j = 0; j < nphi; j++) { const v = grid[base + j]; row2 += v * v; }
      sw += wi * nphi; sw2 += wi * row2;
    }
    const rms = Math.sqrt(sw2 / sw);
    const vmax = 3 * rms || 1;

    const { mapW, mapH, halfW, halfH, cx, cy } = this;
    const img = this.ctx.createImageData(mapW, mapH + this.cbH);
    const data = img.data;
    data.fill(255);   // white background

    // Mollweide inverse, per raster row (each row is one iso-latitude).
    for (let py = 0; py < mapH; py++) {
      const Y = (cy - (py + 0.5)) / halfH;          // latitude axis, up positive
      if (Y < -1 || Y > 1) continue;
      const psi = Math.asin(Y);
      const cpsi = Math.sqrt(Math.max(0, 1 - Y * Y));
      const sinLat = (2 * psi + Math.sin(2 * psi)) / Math.PI;
      const lat = Math.asin(sinLat < -1 ? -1 : sinLat > 1 ? 1 : sinLat);
      const theta = Math.PI / 2 - lat;              // colatitude
      const fy = theta / Math.PI * ntheta - 0.5;    // grid row coord
      let iy = Math.floor(fy); let wy = fy - iy;
      if (iy < 0) { iy = 0; wy = 0; }
      if (iy >= ntheta - 1) { iy = ntheta - 2; wy = 1; }
      const r0 = iy * nphi, r1 = r0 + nphi;

      for (let px = 0; px < mapW; px++) {
        const X = ((px + 0.5) - cx) / halfW;
        if (X < -1 || X > 1) continue;
        const lon = Math.PI * X / cpsi;             // longitude in [-pi, pi]
        if (lon < -Math.PI || lon > Math.PI) continue;
        let phi = lon < 0 ? lon + TWO_PI : lon;
        const fx = phi / TWO_PI * nphi - 0.5;
        let jx = Math.floor(fx); let wx = fx - jx;
        jx = ((jx % nphi) + nphi) % nphi;
        const jx1 = (jx + 1) % nphi;

        // bilinear sample of the equirectangular grid
        const v = (grid[r0 + jx] * (1 - wx) + grid[r0 + jx1] * wx) * (1 - wy)
                + (grid[r1 + jx] * (1 - wx) + grid[r1 + jx1] * wx) * wy;

        const o = (py * mapW + px) * 4;
        colormap(v / vmax, data, o);
        data[o + 3] = 255;
      }
    }

    this._drawColorbar(data, vmax);
    this.ctx.putImageData(img, 0, 0);

    // ellipse outline (matches the black-frame theme)
    const c = this.ctx;
    c.strokeStyle = '#000'; c.lineWidth = 1;
    c.beginPath();
    c.ellipse(cx, cy, halfW, halfH, 0, 0, TWO_PI);
    c.stroke();

    // colour-bar frame + labels
    c.fillStyle = '#000';
    const cb = this._cb;
    c.strokeStyle = '#000'; c.lineWidth = 1;
    c.strokeRect(cb.x0, cb.barTop, cb.x1 - cb.x0, cb.barH);
    c.fillStyle = '#000';
    c.font = '11px system-ui, sans-serif';
    c.textBaseline = 'top';
    const ly = cb.barTop + cb.barH + 3;
    const v = Math.round(cb.vmax);
    c.textAlign = 'left';   c.fillText(`-${v}`, cb.x0, ly);
    c.textAlign = 'center'; c.fillText('0', (cb.x0 + cb.x1) / 2, ly);
    c.textAlign = 'right';  c.fillText(`+${v}`, cb.x1, ly);
    c.textAlign = 'left';
    c.fillText('ΔT [µK]', cb.x1 + 10, cb.barTop);
  }

  // Horizontal colour bar with -vmax / 0 / +vmax labels, in the bottom strip.
  _drawColorbar(data, vmax) {
    const { mapW, mapH, cbH } = this;
    const barH = 12, barTop = mapH + 6;
    const x0 = mapW * 0.30, x1 = mapW * 0.70, bw = x1 - x0;
    const tmp = [0, 0, 0];
    for (let bx = 0; bx < bw; bx++) {
      const t = (bx / (bw - 1)) * 2 - 1;     // -1 .. +1
      colormap(t, tmp, 0);
      for (let by = 0; by < barH; by++) {
        const o = ((barTop + by) * mapW + (x0 + bx | 0)) * 4;
        data[o] = tmp[0]; data[o + 1] = tmp[1]; data[o + 2] = tmp[2]; data[o + 3] = 255;
      }
    }
    // labels are drawn with the 2d context after putImageData
    this._cb = { x0, x1, barTop, barH, vmax };
  }
}
