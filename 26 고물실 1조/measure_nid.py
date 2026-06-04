"""Interactive .nid measurement tool (browser-based, rod height + FWHM).

Pick any .nid in any subfolder from the browser, read its Z-Axis image
directly via NSFopen, level it with the masked polyfit, view the raw and
leveled heatmaps side by side, and measure rod height AND FWHM (width)
by drawing a line across a feature.

Runs a tiny local HTTP server (Python stdlib only) that decodes .nid files
on request, so the browser can browse files and switch Forward/Backward
scans live. Metadata is printed to the console (and shown in the page)
whenever a file is loaded.

Usage:  python measure_nid.py    (opens the browser automatically; Ctrl+C to stop)

Reuses:
  extract_nid_info.info_text   -- Data Info-style metadata text
  export_leveled.masked_polyfit -- per-row degree-`deg` leveling (MAD reject + right-edge exclude)
"""
import json
import os
import socket
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs, unquote

import numpy as np
from NSFopen.read import read

from extract_nid_info import info_text
from export_leveled import masked_polyfit

HERE = os.path.dirname(os.path.abspath(__file__))

# The Windows console is often cp949 here, but metadata contains µ/° (UTF-8).
# Print without crashing on un-encodable characters.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


def list_nid_files():
    """All .nid files under HERE, as sorted repo-relative POSIX paths."""
    out = []
    for root, _dirs, files in os.walk(HERE):
        for fn in files:
            if fn.lower().endswith(".nid"):
                rel = os.path.relpath(os.path.join(root, fn), HERE)
                out.append(rel.replace(os.sep, "/"))
    out.sort()
    return out


def safe_abspath(rel):
    """Resolve a repo-relative path, refusing anything outside HERE."""
    rel = unquote(rel or "")
    ap = os.path.abspath(os.path.join(HERE, rel))
    if os.path.commonpath([ap, HERE]) != HERE:
        raise ValueError("path escapes repo root")
    if not ap.lower().endswith(".nid") or not os.path.isfile(ap):
        raise ValueError("not a .nid file")
    return ap


def available_channels(afm):
    """Directions that have a Z-Axis image, in a stable order."""
    chans = []
    for direction in ("Forward", "Backward"):
        try:
            afm.data[("Image", direction, "Z-Axis")]
            chans.append(direction)
        except KeyError:
            pass
    return chans


def pixel_nm(afm, axis, n):
    """Pixel size in nm along axis ('X'/'Y') for n pixels, from metadata.

    Falls back to image-size/256 assumption if the metadata is missing."""
    try:
        rng = afm.param[(axis, "range")]
        val = rng[0] if isinstance(rng, (list, tuple, np.ndarray)) else rng
        return float(val) * 1e9 / n
    except (KeyError, TypeError, IndexError):
        return 20000.0 / n


def build_payload(rel, channel, deg, k, right_exclude):
    ap = safe_abspath(rel)
    afm = read(ap)
    chans = available_channels(afm)
    if not chans:
        raise ValueError("no Z-Axis image in this file")
    if channel not in chans:
        channel = chans[0]
    z = np.asarray(afm.data[("Image", channel, "Z-Axis")], dtype=float)
    rows, cols = z.shape
    raw_nm = z * 1e9
    lev_nm = masked_polyfit(z, k=k, deg=deg, right_exclude=right_exclude) * 1e9
    px_x = pixel_nm(afm, "X", cols)
    px_y = pixel_nm(afm, "Y", rows)

    # one-line load indicator only; full metadata is shown in the browser
    print(f"loaded {rel} [{channel}]  {rows}x{cols}  "
          f"px=({px_x:.2f}, {px_y:.2f}) nm", flush=True)

    return {
        "name": rel,
        "channel": channel,
        "channels": chans,
        "rows": rows,
        "cols": cols,
        "pixel_nm_x": px_x,
        "pixel_nm_y": px_y,
        "deg": deg,
        "k": k,
        "right_exclude": right_exclude,
        "raw": np.round(raw_nm, 3).tolist(),
        "z": np.round(lev_nm, 3).tolist(),
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quieter console
        pass

    def _send(self, code, body, ctype):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, separators=(",", ":")), "application/json")

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/":
                self._send(200, PAGE, "text/html; charset=utf-8")
            elif u.path == "/api/files":
                self._json({"files": list_nid_files()})
            elif u.path == "/api/meta":
                ap = safe_abspath(q.get("path", [""])[0])
                self._json({"text": info_text(ap)})
            elif u.path == "/api/data":
                payload = build_payload(
                    q.get("path", [""])[0],
                    q.get("channel", ["Forward"])[0],
                    int(float(q.get("deg", ["2"])[0])),
                    float(q.get("k", ["2.5"])[0]),
                    float(q.get("right_exclude", ["0"])[0]),
                )
                self._json(payload)
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001 - surface any error to the browser
            self._json({"error": str(e)}, 400)


def find_port(start=8765):
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start


def main():
    port = find_port()
    srv = HTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"AFM .nid measurement tool serving at {url}")
    print("Pick a file in the browser. Metadata prints here on each load.")
    print("Press Ctrl+C to stop.")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping.")
        srv.shutdown()


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AFM .nid measurement tool</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 16px; color: #1a1a1a; }
  h1 { font-size: 18px; margin: 0 0 8px; }
  .row { display: flex; gap: 18px; align-items: flex-start; flex-wrap: wrap; }
  .panel { border: 1px solid #ccc; border-radius: 8px; padding: 12px; }
  canvas { border: 1px solid #999; cursor: crosshair; image-rendering: pixelated; }
  .maptitle { font-size: 12px; color: #555; margin: 0 0 4px; }
  label { font-size: 13px; }
  select, input[type=number] { font-size: 13px; }
  button { padding: 5px 10px; font-size: 13px; cursor: pointer; }
  table { border-collapse: collapse; font-size: 13px; margin-top: 8px; }
  th, td { border: 1px solid #ccc; padding: 3px 7px; text-align: right; }
  th { background: #f0f0f0; }
  .controls { margin: 8px 0; font-size: 13px; }
  .controls > * { margin-right: 12px; }
  .readout { font-size: 14px; margin: 8px 0; }
  .readout b { font-size: 18px; }
  pre { background: #f7f7f7; border: 1px solid #ddd; border-radius: 6px;
        padding: 8px; font-size: 12px; max-height: 320px; overflow: auto; }
  #status { color: #b00; font-size: 13px; }
</style>
</head>
<body>
<h1>AFM .nid measurement tool &mdash; rod height &amp; FWHM (leveled, masked polyfit)</h1>

<div class="controls">
  <label>File: <select id="file"></select></label>
  <label>Channel: <select id="channel"></select></label>
  <label>deg: <input type="number" id="deg" value="2" min="1" max="4" step="1" style="width:48px"></label>
  <label>k: <input type="number" id="k" value="2.5" min="0.5" max="6" step="0.5" style="width:54px"></label>
  <label>right exclude: <input type="number" id="rex" value="0" min="0" max="0.6" step="0.05" style="width:60px"></label>
  <label>Image size:
    <select id="imgsize">
      <option value="320">320 px</option>
      <option value="400">400 px</option>
      <option value="440" selected>440 px</option>
      <option value="520">520 px</option>
      <option value="640">640 px</option>
    </select>
  </label>
  <label>Measure on:
    <select id="measureon">
      <option value="z" selected>leveled</option>
      <option value="raw">raw</option>
    </select>
  </label>
  <span id="status"></span>
</div>
<div class="controls">
  <label>Swath width:
    <input type="range" id="swath" min="1" max="21" value="7" step="2">
    <span id="swathval">7 px</span>
  </label>
  <label title="Height uses the mean of the peak point plus this many points on each side">
    Peak average n:
    <input type="number" id="peakavg" value="1" min="0" max="20" step="1" style="width:48px">
  </label>
  <label title="FWHM uncertainty uses the x-range where the edge crosses 50% height plus/minus this many baseline-noise sigmas">
    FWHM band:
    <input type="number" id="fwhmband" value="1" min="0" max="5" step="0.25" style="width:54px"> sigma
  </label>
  <label title="align each cross-section to the center before averaging, so a tilted/curved rod keeps its flat top">
    <input type="checkbox" id="align" checked> align swath
  </label>
  <label>Baseline:
    <select id="baseline">
      <option value="auto" selected>auto (outer %)</option>
      <option value="manual">manual (drag handles)</option>
    </select>
  </label>
  <label>Baseline window (each end):
    <input type="range" id="basefrac" min="5" max="45" value="25" step="5">
    <span id="basefracval">25%</span>
  </label>
</div>
<div class="controls">
  <span>Click two points across a feature on the <b>leveled</b> map; drag an endpoint to refine.
  In <b>manual</b> baseline mode, drag the <span style="color:#c47f00">rod start/end</span> handles in the profile.</span>
</div>

<div class="row">
  <div class="panel">
    <p class="maptitle">leveled (measure here)</p>
    <canvas id="map"></canvas>
  </div>
  <div class="panel">
    <p class="maptitle">raw (un-leveled)</p>
    <canvas id="rawmap"></canvas>
  </div>
  <div class="panel">
    <p class="maptitle">d/dx (fast scan, raw)</p>
    <canvas id="dxmap"></canvas>
  </div>
  <div class="panel">
    <canvas id="profile" width="680" height="400"></canvas>
    <div class="readout" id="readout">Draw a line to measure.</div>
    <button id="addrod">Add measurement &#9656;</button>
    <button id="clearrods">Clear table</button>
    <button id="export">Export results</button>
  </div>
</div>

<div class="panel" style="margin-top:16px; display:inline-block;">
  <table id="results">
    <thead><tr>
      <th>#</th><th>file</th><th>ch</th>
      <th>x0,y0</th><th>x1,y1</th>
      <th>on</th><th>swath</th><th>peak avg n</th><th>baseline</th>
      <th>height (nm)</th><th>&plusmn; (nm)</th>
      <th>FWHM (nm)</th><th>&plusmn; (nm)</th>
      <th>&Delta;h R&minus;L (nm)</th><th>&plusmn; (nm)</th>
    </tr></thead>
    <tbody></tbody>
  </table>
</div>

<div class="panel" style="margin-top:16px;">
  <p class="maptitle">metadata</p>
  <pre id="meta">(load a file)</pre>
</div>

<script>
// ---- viridis colormap ----
const VIRIDIS = [
 [68,1,84],[72,40,120],[62,74,137],[49,104,142],[38,130,142],
 [31,158,137],[53,183,121],[110,206,88],[181,222,43],[253,231,37]];
function viridis(t){
  t = Math.max(0, Math.min(1, t));
  const s = t*(VIRIDIS.length-1);
  const i = Math.floor(s), f = s-i;
  const a = VIRIDIS[i], b = VIRIDIS[Math.min(i+1, VIRIDIS.length-1)];
  return [a[0]+(b[0]-a[0])*f, a[1]+(b[1]-a[1])*f, a[2]+(b[2]-a[2])*f];
}
// diverging RdBu_r (blue -> white -> red), matching plot_heatmap2_differential.py
const RDBU_R = [[33,102,172],[103,169,207],[209,229,240],[247,247,247],
                [253,219,199],[239,138,98],[178,24,43]];
function rdbu(t){
  t = Math.max(0, Math.min(1, t));
  const s = t*(RDBU_R.length-1);
  const i = Math.floor(s), f = s-i;
  const a = RDBU_R[i], b = RDBU_R[Math.min(i+1, RDBU_R.length-1)];
  return [a[0]+(b[0]-a[0])*f, a[1]+(b[1]-a[1])*f, a[2]+(b[2]-a[2])*f];
}

function percentile(sorted, p){
  const idx = (sorted.length-1)*p;
  const lo = Math.floor(idx), hi = Math.ceil(idx);
  return sorted[lo] + (sorted[hi]-sorted[lo])*(idx-lo);
}
function median(a){
  const s = [...a].sort((x,y)=>x-y);
  const n = s.length;
  return n%2 ? s[(n-1)/2] : 0.5*(s[n/2-1]+s[n/2]);
}
function std(a){
  const m = a.reduce((s,v)=>s+v,0)/a.length;
  return Math.sqrt(a.reduce((s,v)=>s+(v-m)*(v-m),0)/a.length);
}

let DS=null, TARGET=440, SCALE=1;
let p0=null, p1=null, dragging=null;
// manual rod boundaries (distance nm along the line); reset when a new line is drawn
let ba=null, bb=null, profDrag=null;
// last X(d)->screen mapping params, so profile-canvas drags can invert it
let profMap=null;
const mapCv = document.getElementById('map');
const mapCtx = mapCv.getContext('2d');
const rawCv = document.getElementById('rawmap');
const rawCtx = rawCv.getContext('2d');
const dxCv = document.getElementById('dxmap');
const dxCtx = dxCv.getContext('2d');
const profCv = document.getElementById('profile');
const profCtx = profCv.getContext('2d');

// SCALE maps native pixels -> on-screen pixels so the image fits a TARGET box
// (computed per file so all images render at the same on-screen size).
// canvas y is top-down; data row 0 is at bottom => screen_y = (rows-1-r)*SCALE
function dataToScreen(c, r){ return [c*SCALE, (DS.rows-1-r)*SCALE]; }
function screenToData(sx, sy){ return [sx/SCALE, DS.rows-1-sy/SCALE]; }

function colorScale(grid){
  const flat = grid.flat().sort((a,b)=>a-b);
  return [percentile(flat,0.01), percentile(flat,0.99)];
}

// d/dx along the fast-scan axis (columns), central differences, like
// np.gradient(z, axis=1). Units: nm per pixel.
function gradX(grid){
  const out = [];
  for(let r=0; r<DS.rows; r++){
    const row = grid[r], o = new Array(DS.cols);
    for(let c=0; c<DS.cols; c++){
      if(c===0) o[c] = row[1]-row[0];
      else if(c===DS.cols-1) o[c] = row[c]-row[c-1];
      else o[c] = (row[c+1]-row[c-1])/2;
    }
    out.push(o);
  }
  return out;
}

function renderHeatmap(cv, ctx, grid, vmin, vmax, withLine, cmap){
  cmap = cmap || viridis;
  cv.width = Math.round(DS.cols*SCALE);
  cv.height = Math.round(DS.rows*SCALE);
  const img = ctx.createImageData(DS.cols, DS.rows);
  for(let r=0; r<DS.rows; r++){
    const sy = DS.rows-1-r; // flip for origin lower
    for(let c=0; c<DS.cols; c++){
      const t = (grid[r][c]-vmin)/(vmax-vmin);
      const [R,G,B] = cmap(t);
      const idx = (sy*DS.cols + c)*4;
      img.data[idx]=R; img.data[idx+1]=G; img.data[idx+2]=B; img.data[idx+3]=255;
    }
  }
  const off = document.createElement('canvas');
  off.width = DS.cols; off.height = DS.rows;
  off.getContext('2d').putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0,0,cv.width,cv.height);
  ctx.drawImage(off, 0, 0, cv.width, cv.height);
  if(withLine) drawLine(ctx);
}

function drawLine(ctx){
  if(!p0) return;
  const [sx0, sy0] = dataToScreen(p0[0], p0[1]);
  if(!p1){ ctx.fillStyle='#ff3b3b'; ctx.beginPath(); ctx.arc(sx0,sy0,4,0,7); ctx.fill(); return; }
  const [sx1, sy1] = dataToScreen(p1[0], p1[1]);
  const dx=p1[0]-p0[0], dy=p1[1]-p0[1], lenpx=Math.hypot(dx,dy)||1e-9;
  const px=-dy/lenpx, py=dx/lenpx;             // perpendicular (pixel space)
  const W=swathWidth(), hw=(W-1)/2;
  const A=window._lastA;
  const pt = t => dataToScreen(p0[0]+dx*t, p0[1]+dy*t);
  // swath band (the averaged region)
  if(W>1){
    const c00=dataToScreen(p0[0]+hw*px,p0[1]+hw*py), c01=dataToScreen(p1[0]+hw*px,p1[1]+hw*py);
    const c11=dataToScreen(p1[0]-hw*px,p1[1]-hw*py), c10=dataToScreen(p0[0]-hw*px,p0[1]-hw*py);
    ctx.fillStyle='rgba(255,255,255,0.18)'; ctx.strokeStyle='rgba(255,255,255,0.55)'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(...c00); ctx.lineTo(...c01); ctx.lineTo(...c11); ctx.lineTo(...c10); ctx.closePath(); ctx.fill(); ctx.stroke();
  }
  // rod range as line parameter t (from the last analysis)
  const dxn=dx*DS.pixel_nm_x, dyn=dy*DS.pixel_nm_y, lennm=Math.hypot(dxn,dyn)||1;
  let tA=0, tB=1;
  if(A){ tA=Math.max(0,Math.min(1,A.lsx1/lennm)); tB=Math.max(0,Math.min(1,A.rsx0/lennm)); }
  // baseline segments (gray dashed)
  ctx.lineWidth=2; ctx.strokeStyle='#cfcfcf'; ctx.setLineDash([4,3]);
  ctx.beginPath(); ctx.moveTo(...pt(0)); ctx.lineTo(...pt(tA)); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(...pt(tB)); ctx.lineTo(...pt(1)); ctx.stroke();
  ctx.setLineDash([]);
  // rod segment (orange solid)
  ctx.strokeStyle='#ff8c1a'; ctx.lineWidth=2.5;
  ctx.beginPath(); ctx.moveTo(...pt(tA)); ctx.lineTo(...pt(tB)); ctx.stroke();
  // boundary ticks + fitted center
  ctx.fillStyle='#ff8c1a';
  for(const t of [tA,tB]){ const s=pt(t); ctx.beginPath(); ctx.arc(s[0],s[1],3,0,7); ctx.fill(); }
  if(A && A.center!=null){ const s=pt(Math.max(0,Math.min(1,A.center/lennm)));
    ctx.fillStyle='#d62828'; ctx.beginPath(); ctx.arc(s[0],s[1],3.5,0,7); ctx.fill(); }
  // endpoints
  ctx.fillStyle='#ff3b3b';
  for(const [x,y] of [[sx0,sy0],[sx1,sy1]]){ ctx.beginPath(); ctx.arc(x,y,4,0,7); ctx.fill(); }
}

function renderMaps(){
  if(!DS) return;
  // fit the (square-ish) image into a TARGET x TARGET box, preserving aspect,
  // so every file shows at the same on-screen size regardless of native pixels
  SCALE = TARGET / Math.max(DS.rows, DS.cols);
  renderHeatmap(mapCv, mapCtx, DS.z, DS._lev_vmin, DS._lev_vmax, true);
  renderHeatmap(rawCv, rawCtx, DS.raw, DS._raw_vmin, DS._raw_vmax, true);
  renderHeatmap(dxCv, dxCtx, DS.dx, -DS._dx_lim, DS._dx_lim, true, rdbu);
}

function measureGrid(){
  return document.getElementById('measureon').value==='raw' ? DS.raw : DS.z;
}

// bilinear sample of an arbitrary grid (z or raw), clamped to bounds
function bilinearG(grid, c, r){
  c = Math.max(0, Math.min(DS.cols-1, c));
  r = Math.max(0, Math.min(DS.rows-1, r));
  const c0 = Math.floor(c), r0 = Math.floor(r);
  const c1 = Math.min(c0+1, DS.cols-1), r1 = Math.min(r0+1, DS.rows-1);
  const fc = c-c0, fr = r-r0;
  return grid[r0][c0]*(1-fc)*(1-fr) + grid[r0][c1]*fc*(1-fr)
       + grid[r1][c0]*(1-fc)*fr   + grid[r1][c1]*fc*fr;
}

function swathWidth(){ return Math.max(1, +document.getElementById('swath').value | 0); }
function peakAvgRadius(){ return Math.max(0, +document.getElementById('peakavg').value | 0); }
function fwhmBandSigma(){ return Math.max(0, +document.getElementById('fwhmband').value || 0); }
function alignSwath(){ const el=document.getElementById('align'); return el ? el.checked : true; }
function mean(a){ let s=0; for(const v of a) s+=v; return s/a.length; }

// shift `row` by the integer lag (|lag|<=maxLag) that maximizes cross-covariance
// with `ref`; return a copy aligned to ref, out-of-range entries = null.
function shiftToAlign(row, ref, maxLag){
  const n=row.length, rm=mean(ref), om=mean(row);
  let bestLag=0, best=-Infinity;
  for(let L=-maxLag; L<=maxLag; L++){
    let s=0,c=0; for(let i=0;i<n;i++){ const j=i+L; if(j<0||j>=n) continue; s+=(ref[i]-rm)*(row[j]-om); c++; }
    if(c>0){ const sc=s/c; if(sc>best){best=sc; bestLag=L;} }
  }
  const out=new Array(n).fill(null);
  for(let i=0;i<n;i++){ const j=i+bestLag; if(j>=0&&j<n) out[i]=row[j]; }
  return out;
}

function sampleProfile(){
  // Sample along the line at uniform steps; at each step average W parallel
  // cross-sections offset perpendicular to the line (the "swath"). A rod is
  // extended, so averaging along its axis cuts noise by ~sqrt(W).
  // With "align swath" on, each cross-section is registered (cross-correlated)
  // to the center one before averaging, so a tilted/curved rod keeps its sharp
  // edges and flat top (plateau) instead of being smeared into a round bump.
  const grid = measureGrid();
  const dx = p1[0]-p0[0], dy = p1[1]-p0[1];
  const lenpx = Math.hypot(dx, dy) || 1e-9;
  const dxn = dx*DS.pixel_nm_x, dyn = dy*DS.pixel_nm_y;
  const lennm = Math.hypot(dxn, dyn) || 1;
  const px = -dy/lenpx, py = dx/lenpx;          // perpendicular (pixel space)
  const W = swathWidth(), half = (W-1)/2;
  const steps = Math.max(8, Math.round(lenpx)), n = steps+1;
  const dist = []; for(let i=0;i<=steps;i++) dist.push(i/steps*lennm);
  // collect each parallel cross-section
  const lines = [];
  for(let w=-half; w<=half; w++){
    const row = new Array(n);
    for(let i=0;i<=steps;i++){ const t=i/steps; row[i]=bilinearG(grid, p0[0]+dx*t+w*px, p0[1]+dy*t+w*py); }
    lines.push(row);
  }
  if(W===1) return {prof:lines[0], dist, nAvg:1, aligned:false};
  let prof = new Array(n).fill(0);
  let didAlign = false;
  if(alignSwath()){
    const ref = lines[Math.floor(lines.length/2)];
    const maxLag = Math.min(n-2, Math.ceil(half)+3);
    const cnt = new Array(n).fill(0);
    for(const row of lines){
      const a = shiftToAlign(row, ref, maxLag);
      for(let i=0;i<n;i++) if(a[i]!=null){ prof[i]+=a[i]; cnt[i]++; }
    }
    for(let i=0;i<n;i++) prof[i] = cnt[i] ? prof[i]/cnt[i] : ref[i];
    didAlign = true;
  } else {
    for(const row of lines) for(let i=0;i<n;i++) prof[i]+=row[i];
    for(let i=0;i<n;i++) prof[i]/=lines.length;
  }
  return {prof, dist, nAvg:W, aligned:didAlign};
}

// ---- small linear-algebra + fitting helpers ----
function gaussSolve(M, v){            // solve M x = v (M square, mutated copy)
  const m = v.length;
  const A = M.map((row,i)=>row.concat(v[i]));
  for(let col=0; col<m; col++){
    let piv=col; for(let r=col+1;r<m;r++) if(Math.abs(A[r][col])>Math.abs(A[piv][col])) piv=r;
    [A[col],A[piv]]=[A[piv],A[col]];
    const d=A[col][col]||1e-30;
    for(let c=col;c<=m;c++) A[col][c]/=d;
    for(let r=0;r<m;r++) if(r!==col){ const f=A[r][col]; for(let c=col;c<=m;c++) A[r][c]-=f*A[col][c]; }
  }
  return A.map(row=>row[m]);
}
function solveLLS(cols, y){           // least squares: fit y = sum_j coef[j]*cols[j]
  const m=cols.length, n=y.length;
  const ATA=Array.from({length:m},()=>new Array(m).fill(0)), ATy=new Array(m).fill(0);
  for(let a=0;a<m;a++){
    for(let b=a;b<m;b++){ let s=0; for(let i=0;i<n;i++) s+=cols[a][i]*cols[b][i]; ATA[a][b]=ATA[b][a]=s; }
    let s=0; for(let i=0;i<n;i++) s+=cols[a][i]*y[i]; ATy[a]=s;
  }
  return gaussSolve(ATA, ATy);
}
function lineY(b, s, x){ return b + s*x; }

function analyze(){
  const {prof, dist, nAvg, aligned} = sampleProfile();
  const n = prof.length;
  const baseMode = document.getElementById('baseline').value;
  const span = dist[n-1]-dist[0] || 1;

  // --- partition into baseline vs rod points ---
  const frac = +document.getElementById('basefrac').value/100;
  const k = Math.max(1, Math.round(n*frac));
  if(baseMode==='manual' && (ba===null||bb===null)){ ba=dist[0]+0.3*span; bb=dist[0]+0.7*span; }
  const baseIdx=[], rodIdx=[];
  for(let i=0;i<n;i++){
    const isRod = baseMode==='manual' ? (dist[i]>=ba && dist[i]<=bb) : (i>=k && i<n-k);
    (isRod?rodIdx:baseIdx).push(i);
  }
  const baseSafe = baseIdx.length>=2 ? baseIdx : prof.map((_,i)=>i);

  // --- baseline line: linear fit to baseline points; noise = residual std ---
  const c = solveLLS([baseSafe.map(()=>1), baseSafe.map(i=>dist[i])], baseSafe.map(i=>prof[i]));
  const blBase = c[0], blSlope = c[1];
  const noise = std(baseSafe.map(i=>prof[i]-lineY(blBase,blSlope,dist[i]))) || 1e-9;

  // --- left/right baseline medians (for the right-left height difference) ---
  const ctr0 = baseMode==='manual' ? 0.5*(ba+bb) : 0.5*(dist[0]+dist[n-1]);
  const lP = baseSafe.filter(i=>dist[i]<ctr0).map(i=>prof[i]);
  const rP = baseSafe.filter(i=>dist[i]>=ctr0).map(i=>prof[i]);
  const left = lP.length?median(lP):blBase, right = rP.length?median(rP):blBase;
  const hdiff = right-left, hdiffErr = Math.hypot(lP.length?std(lP):0, rP.length?std(rP):0);

  // segment ranges drawn as the purple left/right end levels
  const lsx0 = dist[0], lsx1 = baseMode==='manual'? ba : dist[k-1];
  const rsx0 = baseMode==='manual'? bb : dist[n-k], rsx1 = dist[n-1];

  // peak - baseline: baseline-subtract, then max over the rod region + half-max crossings.
  // Height is averaged over the peak point +/- peakAvgRadius points to reduce
  // single-pixel noise sensitivity; FWHM still uses the peak location and the
  // averaged height's half level.
  const det = prof.map((v,i)=> v - lineY(blBase,blSlope,dist[i]));
  const search = rodIdx.length? rodIdx : prof.map((_,i)=>i);
  let pi = search[0]; for(const i of search) if(det[i]>det[pi]) pi=i;
  const searchSet = new Set(search);
  const peakAvgN = peakAvgRadius();
  const avgIdx = [];
  for(let i=Math.max(0, pi-peakAvgN); i<=Math.min(n-1, pi+peakAvgN); i++){
    if(searchSet.has(i)) avgIdx.push(i);
  }
  if(!avgIdx.length) avgIdx.push(pi);
  const height = mean(avgIdx.map(i=>det[i]));
  const base = mean(avgIdx.map(i=>lineY(blBase,blSlope,dist[i])));
  const peak = mean(avgIdx.map(i=>prof[i]));
  const halfDet = height/2;
  const crossLeft = level => {
    for(let i=pi;i>0;i--){
      if(det[i]>=level && det[i-1]<level){
        const f=(level-det[i-1])/(det[i]-det[i-1]);
        return dist[i-1]+(dist[i]-dist[i-1])*f;
      }
    }
    return null;
  };
  const crossRight = level => {
    for(let i=pi;i<n-1;i++){
      if(det[i]>=level && det[i+1]<level){
        const f=(level-det[i])/(det[i+1]-det[i]);
        return dist[i]+(dist[i+1]-dist[i])*f;
      }
    }
    return null;
  };
  const xL = crossLeft(halfDet), xR = crossRight(halfDet);
  const fwhm=(xL!==null&&xR!==null)?Math.abs(xR-xL):NaN;
  let fwhmErr=NaN;
  const fwhmBandSigmaVal = fwhmBandSigma();
  const fwhmBandTol = fwhmBandSigmaVal * noise;
  const fwhmBandLo = Math.max(0, halfDet - fwhmBandTol);
  const fwhmBandHi = halfDet + fwhmBandTol;
  const xLlo = crossLeft(fwhmBandLo), xLhi = crossLeft(fwhmBandHi);
  const xRlo = crossRight(fwhmBandLo), xRhi = crossRight(fwhmBandHi);
  let fwhmMin=NaN, fwhmMax=NaN;
  if(xLlo!=null && xLhi!=null && xRlo!=null && xRhi!=null){
    const leftInner = Math.max(xLlo, xLhi), leftOuter = Math.min(xLlo, xLhi);
    const rightInner = Math.min(xRlo, xRhi), rightOuter = Math.max(xRlo, xRhi);
    fwhmMin = Math.max(0, rightInner - leftInner);
    fwhmMax = Math.max(0, rightOuter - leftOuter);
    fwhmErr = 0.5 * Math.abs(fwhmMax - fwhmMin);
  }

  return {prof, dist, nAvg, aligned, baseMode, k, ba, bb, rodIdx, baseIdx,
          noise, blBase, blSlope, left, right, hdiff, hdiffErr, lsx0, lsx1, rsx0, rsx1,
          pi, peak, base, height, heightErr:noise, peakAvgN, peakAvgCount:avgIdx.length, half:base+halfDet,
          halfDet, fwhmBandSigmaVal, fwhmBandTol, fwhmBandLo, fwhmBandHi,
          xL, xR, xLlo, xLhi, xRlo, xRhi, fwhm, fwhmErr, fwhmMin, fwhmMax, center:dist[pi]};
}

function niceTicks(lo, hi, want){
  const span = hi-lo;
  if(span<=0) return [lo];
  const raw = span/want;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw/mag;
  const step = (norm<1.5?1:norm<3?2:norm<7?5:10)*mag;
  const start = Math.ceil(lo/step)*step;
  const ticks = [];
  for(let v=start; v<=hi+1e-9; v+=step) ticks.push(+v.toFixed(6));
  return ticks;
}

function drawProfile(A){
  const W = profCv.width, H = profCv.height;
  const mL = 56, mB = 42, mT = 16, mR = 16;
  profCtx.clearRect(0,0,W,H);
  if(!A){ profMap=null; profCtx.fillStyle='#666'; profCtx.font='13px sans-serif';
          profCtx.fillText('Draw a line to measure.',mL,H/2); return; }
  const {prof, dist, base, half, halfDet, fwhmBandLo, fwhmBandHi,
         xL, xR, xLlo, xLhi, xRlo, xRhi, left, right,
         blBase, blSlope, rodIdx, lsx0, lsx1, rsx0, rsx1,
         center, height, baseMode, ba, bb} = A;
  const xmin = dist[0], xmax = dist[dist.length-1];
  let ymin = Math.min(...prof), ymax = Math.max(...prof);
  const pad = (ymax-ymin)*0.15 + 1e-6; ymin-=pad; ymax+=pad;
  const plotW = W-mL-mR;
  const X = d => mL + (d-xmin)/(xmax-xmin)*plotW;
  const Y = v => H-mB - (v-ymin)/(ymax-ymin)*(H-mB-mT);
  profMap = {mL, plotW, xmin, xmax, mT, mB, H};   // for profile-canvas drags
  const isRod = new Array(prof.length).fill(false); for(const i of rodIdx) isRod[i]=true;

  // region shading: baseline (gray) vs rod (orange)
  profCtx.fillStyle='rgba(160,160,160,0.16)';
  profCtx.fillRect(X(lsx0),mT, X(lsx1)-X(lsx0), H-mB-mT);
  profCtx.fillRect(X(rsx0),mT, X(rsx1)-X(rsx0), H-mB-mT);
  profCtx.fillStyle='rgba(255,170,40,0.13)';
  profCtx.fillRect(X(lsx1),mT, X(rsx0)-X(lsx1), H-mB-mT);

  profCtx.font='11px sans-serif';
  profCtx.strokeStyle='#ececec'; profCtx.fillStyle='#333'; profCtx.lineWidth=1;
  profCtx.textAlign='center'; profCtx.textBaseline='top';
  for(const t of niceTicks(xmin, xmax, 6)){
    const x=X(t);
    profCtx.beginPath(); profCtx.moveTo(x,mT); profCtx.lineTo(x,H-mB); profCtx.stroke();
    profCtx.fillText(t.toFixed(0), x, H-mB+4);
  }
  profCtx.textAlign='right'; profCtx.textBaseline='middle';
  for(const t of niceTicks(ymin, ymax, 6)){
    const y=Y(t);
    profCtx.beginPath(); profCtx.moveTo(mL,y); profCtx.lineTo(W-mR,y); profCtx.stroke();
    profCtx.fillText(t.toFixed(0), mL-5, y);
  }
  profCtx.strokeStyle='#888'; profCtx.beginPath();
  profCtx.moveTo(mL,mT); profCtx.lineTo(mL,H-mB); profCtx.lineTo(W-mR,H-mB); profCtx.stroke();
  profCtx.fillStyle='#000'; profCtx.textAlign='center'; profCtx.textBaseline='alphabetic';
  profCtx.fillText('distance (nm)', (mL+W-mR)/2, H-6);
  profCtx.save(); profCtx.translate(12,(mT+H-mB)/2); profCtx.rotate(-Math.PI/2);
  profCtx.fillText('height (nm)', 0, 0); profCtx.restore();

  // baseline LINE (sloped) using blBase/blSlope
  profCtx.strokeStyle='#1f9e6e'; profCtx.setLineDash([5,4]); profCtx.lineWidth=1.5;
  profCtx.beginPath(); profCtx.moveTo(X(xmin),Y(blBase+blSlope*xmin)); profCtx.lineTo(X(xmax),Y(blBase+blSlope*xmax)); profCtx.stroke();
  // half-max band: baseline + (50% height +/- selected noise band)
  if(fwhmBandLo!=null && fwhmBandHi!=null){
    profCtx.fillStyle='rgba(217,131,0,0.12)';
    profCtx.beginPath();
    profCtx.moveTo(X(xmin), Y(blBase+blSlope*xmin+fwhmBandHi));
    profCtx.lineTo(X(xmax), Y(blBase+blSlope*xmax+fwhmBandHi));
    profCtx.lineTo(X(xmax), Y(blBase+blSlope*xmax+fwhmBandLo));
    profCtx.lineTo(X(xmin), Y(blBase+blSlope*xmin+fwhmBandLo));
    profCtx.closePath(); profCtx.fill();
  }
  // half-max center line
  profCtx.strokeStyle='#d98300';
  profCtx.beginPath();
  profCtx.moveTo(X(xmin),Y(blBase+blSlope*xmin+halfDet));
  profCtx.lineTo(X(xmax),Y(blBase+blSlope*xmax+halfDet));
  profCtx.stroke();
  profCtx.setLineDash([]);
  // purple left/right end levels (right-left difference)
  profCtx.strokeStyle='#8e44ad'; profCtx.lineWidth=2.5;
  profCtx.beginPath(); profCtx.moveTo(X(lsx0),Y(left)); profCtx.lineTo(X(lsx1),Y(left)); profCtx.stroke();
  profCtx.beginPath(); profCtx.moveTo(X(rsx0),Y(right)); profCtx.lineTo(X(rsx1),Y(right)); profCtx.stroke();

  // profile line
  profCtx.strokeStyle='#9bb7e8'; profCtx.lineWidth=1.2; profCtx.beginPath();
  prof.forEach((v,i)=>{ const x=X(dist[i]), y=Y(v); i?profCtx.lineTo(x,y):profCtx.moveTo(x,y); });
  profCtx.stroke();
  // points colored by role: rod blue, baseline gray
  prof.forEach((v,i)=>{ profCtx.fillStyle = isRod[i]?'#2456c8':'#9a9a9a';
    profCtx.beginPath(); profCtx.arc(X(dist[i]),Y(v),1.9,0,7); profCtx.fill(); });

  // height guide (baseline@center -> peak) -- graphical only, no text
  const baseAtC = blBase + blSlope*center;
  drawVArrow(X(center), Y(baseAtC), Y(baseAtC+height), '#c0392b');
  // FWHM band edge ranges + center crossing markers
  if(xLlo!==null && xLhi!==null && xRlo!==null && xRhi!==null){
    profCtx.fillStyle='rgba(217,131,0,0.22)';
    const l0=Math.min(xLlo,xLhi), l1=Math.max(xLlo,xLhi);
    const r0=Math.min(xRlo,xRhi), r1=Math.max(xRlo,xRhi);
    profCtx.fillRect(X(l0),mT,Math.max(1,X(l1)-X(l0)),H-mB-mT);
    profCtx.fillRect(X(r0),mT,Math.max(1,X(r1)-X(r0)),H-mB-mT);
  }
  if(xL!=null && xR!=null && isFinite(xL) && isFinite(xR)){
    drawHArrow(X(xL), X(xR), Y(0.5*(blBase+blSlope*xL+halfDet + blBase+blSlope*xR+halfDet)), '#d98300');
    profCtx.fillStyle='#d98300';
    for(const xc of [xL,xR]){
      profCtx.beginPath(); profCtx.arc(X(xc),Y(blBase+blSlope*xc+halfDet),3,0,7); profCtx.fill();
    }
  }
  // peak marker
  profCtx.fillStyle='#ff3b3b';
  profCtx.beginPath(); profCtx.arc(X(center),Y(baseAtC+height),4.5,0,7); profCtx.fill();

  // manual boundary handles (line + grab tab, no text)
  if(baseMode==='manual'){
    for(const bx of [ba,bb]){
      const x=X(bx);
      profCtx.strokeStyle='#c47f00'; profCtx.lineWidth=2;
      profCtx.beginPath(); profCtx.moveTo(x,mT); profCtx.lineTo(x,H-mB); profCtx.stroke();
      profCtx.fillStyle='#c47f00'; profCtx.fillRect(x-4,mT,8,9);   // grab tab
    }
  }
}

function drawVArrow(x, y1, y2, color){
  profCtx.strokeStyle=color; profCtx.fillStyle=color; profCtx.lineWidth=1.5;
  profCtx.beginPath(); profCtx.moveTo(x,y1); profCtx.lineTo(x,y2); profCtx.stroke();
  const dir = y2<y1?-1:1;
  for(const yy of [y1,y2]){ const d = (yy===y1?-dir:dir);
    profCtx.beginPath(); profCtx.moveTo(x,yy); profCtx.lineTo(x-3,yy+4*d); profCtx.lineTo(x+3,yy+4*d); profCtx.closePath(); profCtx.fill(); }
}
function drawHArrow(x1, x2, y, color){
  profCtx.strokeStyle=color; profCtx.fillStyle=color; profCtx.lineWidth=1.5;
  profCtx.beginPath(); profCtx.moveTo(x1,y); profCtx.lineTo(x2,y); profCtx.stroke();
  for(const xx of [x1,x2]){ const d=(xx===x1?1:-1);
    profCtx.beginPath(); profCtx.moveTo(xx,y); profCtx.lineTo(xx+4*d,y-3); profCtx.lineTo(xx+4*d,y+3); profCtx.closePath(); profCtx.fill(); }
}

function refresh(){
  if(DS && p0 && p1){
    const A = analyze();
    window._lastA = A;            // set before renderMaps so drawLine sees rod range
    drawProfile(A);
    const fw = isNaN(A.fwhm) ? '&mdash; (no half-max)'
               : `${A.fwhm.toFixed(2)} nm &plusmn; ${A.fwhmErr.toFixed(2)}`;
    document.getElementById('readout').innerHTML =
      `<span style="color:#666">[peak&minus;baseline, peak avg &plusmn;${A.peakAvgN} pt (${A.peakAvgCount} pts), FWHM band &plusmn;${A.fwhmBandSigmaVal.toFixed(2)} sigma (${A.fwhmBandTol.toFixed(2)} nm), swath ${A.nAvg}px${A.aligned?' aligned':''}, baseline ${A.baseMode}]</span><br>` +
      `height <b>${A.height.toFixed(2)} nm</b> &plusmn; ${A.heightErr.toFixed(2)} &nbsp;|&nbsp; ` +
      `FWHM <b>${fw}</b><br>` +
      `&Delta;h (right&minus;left) = ${A.right.toFixed(2)} &minus; ${A.left.toFixed(2)} = ` +
      `<b>${A.hdiff.toFixed(2)} nm</b> &plusmn; ${A.hdiffErr.toFixed(2)}`;
  } else {
    drawProfile(null);
    document.getElementById('readout').textContent = 'Draw a line to measure.';
    window._lastA = null;
  }
  renderMaps();
}

// ---- interaction (on the leveled map) ----
function nearEndpoint(sx, sy){
  for(const [p,name] of [[p0,'p0'],[p1,'p1']]){
    if(!p) continue;
    const [x,y] = dataToScreen(p[0], p[1]);
    if(Math.hypot(x-sx, y-sy) < 8) return name;
  }
  return null;
}
mapCv.addEventListener('mousedown', e=>{
  if(!DS) return;
  const rect = mapCv.getBoundingClientRect();
  const sx = e.clientX-rect.left, sy = e.clientY-rect.top;
  const hit = nearEndpoint(sx, sy);
  if(hit){ dragging = hit; return; }
  const d = screenToData(sx, sy);
  if(!p0 || (p0 && p1)){ p0 = d; p1 = null; ba=bb=null; }   // new line -> reset rod range
  else { p1 = d; }
  refresh();
});
mapCv.addEventListener('mousemove', e=>{
  if(!dragging) return;
  const rect = mapCv.getBoundingClientRect();
  const d = screenToData(e.clientX-rect.left, e.clientY-rect.top);
  if(dragging==='p0') p0=d; else p1=d;
  refresh();
});
window.addEventListener('mouseup', ()=>{ dragging=null; profDrag=null; });

// ---- manual baseline handles: drag on the profile canvas ----
function profScreenToDist(sx){
  if(!profMap) return null;
  return profMap.xmin + (sx-profMap.mL)/profMap.plotW*(profMap.xmax-profMap.xmin);
}
profCv.addEventListener('mousedown', e=>{
  if(!profMap || document.getElementById('baseline').value!=='manual' || ba===null) return;
  const rect = profCv.getBoundingClientRect();
  const sx = e.clientX-rect.left;
  const xa = profMap.mL + (ba-profMap.xmin)/(profMap.xmax-profMap.xmin)*profMap.plotW;
  const xb = profMap.mL + (bb-profMap.xmin)/(profMap.xmax-profMap.xmin)*profMap.plotW;
  profDrag = Math.abs(sx-xa) < Math.abs(sx-xb) ? 'ba' : 'bb';
  if(Math.min(Math.abs(sx-xa),Math.abs(sx-xb)) > 14) profDrag=null;   // not near a handle
});
profCv.addEventListener('mousemove', e=>{
  if(!profDrag || !profMap) return;
  const rect = profCv.getBoundingClientRect();
  let d = profScreenToDist(e.clientX-rect.left);
  d = Math.max(profMap.xmin, Math.min(profMap.xmax, d));
  if(profDrag==='ba') ba = Math.min(d, bb-1e-6); else bb = Math.max(d, ba+1e-6);
  refresh();
});

// ---- table ----
const rows = [];
function renderTable(){
  const tb = document.querySelector('#results tbody');
  tb.innerHTML = rows.map((r,i)=>
    `<tr><td>${i+1}</td><td>${r.ds}</td><td>${r.ch}</td>`+
    `<td>${r.x0},${r.y0}</td><td>${r.x1},${r.y1}</td>`+
    `<td>${r.on}</td><td>${r.swath}${r.aligned?'a':''}</td><td>${r.peakAvgN}</td><td>${r.baseMode}</td>`+
    `<td><b>${r.height.toFixed(2)}</b></td><td>${r.noise.toFixed(2)}</td>`+
    `<td><b>${isNaN(r.fwhm)?'--':r.fwhm.toFixed(2)}</b></td>`+
    `<td>${isNaN(r.fwhmErr)?'--':r.fwhmErr.toFixed(2)}</td>`+
    `<td><b>${r.hdiff.toFixed(2)}</b></td><td>${r.hdiffErr.toFixed(2)}</td></tr>`).join('');
}
document.getElementById('addrod').onclick = ()=>{
  const A = window._lastA; if(!A){ alert('Draw a line first.'); return; }
  rows.push({ds:DS.name, ch:DS.channel,
    on:document.getElementById('measureon').value==='raw'?'raw':'leveled',
    swath:A.nAvg, aligned:A.aligned, peakAvgN:A.peakAvgN, peakAvgCount:A.peakAvgCount,
    baseMode:A.baseMode, center:A.center,
    x0:Math.round(p0[0]),y0:Math.round(p0[1]),x1:Math.round(p1[0]),y1:Math.round(p1[1]),
    peak:A.peak, base:A.base, height:A.height, noise:A.heightErr,
    fwhm:A.fwhm, fwhmErr:A.fwhmErr, fwhmBandSigma:A.fwhmBandSigmaVal,
    fwhmBandTol:A.fwhmBandTol, fwhmMin:A.fwhmMin, fwhmMax:A.fwhmMax,
    hdiff:A.hdiff, hdiffErr:A.hdiffErr});
  renderTable();
};
document.getElementById('clearrods').onclick = ()=>{ rows.length=0; renderTable(); };
document.getElementById('export').onclick = ()=>{
  const hdr = 'file,channel,measure_on,swath_px,swath_aligned,baseline_mode,x0,y0,x1,y1,'+
    'peak_avg_n,peak_avg_count,peak_nm,base_nm,height_nm,pm_nm,'+
    'fwhm_nm,fwhm_pm_nm,fwhm_band_sigma,fwhm_band_tol_nm,fwhm_min_nm,fwhm_max_nm,'+
    'center_nm,hdiff_RmL_nm,hdiff_pm_nm';
  const num = (v,d=3)=> (v==null||isNaN(v))?'':v.toFixed(d);
  const lines = rows.map(r=>[r.ds,r.ch,r.on,r.swath,r.aligned?1:0,r.baseMode,r.x0,r.y0,r.x1,r.y1,
    r.peakAvgN,r.peakAvgCount,
    num(r.peak),num(r.base),num(r.height),num(r.noise),
    num(r.fwhm),num(r.fwhmErr),num(r.fwhmBandSigma),num(r.fwhmBandTol),num(r.fwhmMin),num(r.fwhmMax),num(r.center,2),
    num(r.hdiff),num(r.hdiffErr)].join(','));
  const csv = [hdr,...lines].join('\n');
  const blob = new Blob([csv], {type:'text/csv'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'nid_measurements.csv'; a.click();
  console.log(csv);
};

// ---- data loading ----
function setStatus(msg){ document.getElementById('status').textContent = msg || ''; }

async function loadData(){
  const path = document.getElementById('file').value;
  if(!path) return;
  const channel = document.getElementById('channel').value || 'Forward';
  const deg = document.getElementById('deg').value;
  const k = document.getElementById('k').value;
  const rex = document.getElementById('rex').value;
  setStatus('loading...');
  try{
    const qs = new URLSearchParams({path, channel, deg, k, right_exclude:rex});
    const r = await fetch('/api/data?'+qs);
    const j = await r.json();
    if(j.error){ setStatus('error: '+j.error); return; }
    DS = j;
    [DS._lev_vmin, DS._lev_vmax] = colorScale(DS.z);
    [DS._raw_vmin, DS._raw_vmax] = colorScale(DS.raw);
    DS.dx = gradX(DS.raw);  // d/dx of RAW height (differentiation removes the background)
    const absdx = DS.dx.flat().map(Math.abs).sort((a,b)=>a-b);
    DS._dx_lim = percentile(absdx, 0.99) || 1e-9;  // symmetric range
    // refresh channel dropdown to what the file actually has
    const csel = document.getElementById('channel');
    if(csel.dataset.for !== DS.name){
      csel.innerHTML = '';
      DS.channels.forEach(c=> csel.add(new Option(c,c)));
      csel.value = DS.channel; csel.dataset.for = DS.name;
    }
    p0=p1=null;
    refresh();
    setStatus(`${DS.rows}x${DS.cols}  px=(${DS.pixel_nm_x.toFixed(2)}, ${DS.pixel_nm_y.toFixed(2)}) nm`);
    loadMeta(path);
  }catch(e){ setStatus('error: '+e); }
}

async function loadMeta(path){
  try{
    const r = await fetch('/api/meta?path='+encodeURIComponent(path));
    const j = await r.json();
    document.getElementById('meta').textContent = j.text || j.error || '';
  }catch(e){ document.getElementById('meta').textContent = String(e); }
}

// ---- setup ----
async function init(){
  const r = await fetch('/api/files');
  const j = await r.json();
  const fsel = document.getElementById('file');
  j.files.forEach(f=> fsel.add(new Option(f, f)));
  fsel.onchange = ()=>{ document.getElementById('channel').dataset.for=''; loadData(); };
  document.getElementById('channel').onchange = loadData;
  for(const id of ['deg','k','rex']) document.getElementById(id).onchange = loadData;
  document.getElementById('imgsize').onchange = e=>{ TARGET=+e.target.value; renderMaps(); };
  document.getElementById('measureon').onchange = refresh;
  document.getElementById('align').onchange = refresh;
  document.getElementById('peakavg').onchange = refresh;
  document.getElementById('fwhmband').onchange = refresh;
  document.getElementById('baseline').onchange = ()=>{ ba=bb=null; refresh(); };
  document.getElementById('swath').oninput = e=>{
    const px=+e.target.value, nm = DS? (px*0.5*(DS.pixel_nm_x+DS.pixel_nm_y)):0;
    document.getElementById('swathval').textContent = `${px} px${DS?` (~${nm.toFixed(0)} nm)`:''}`;
    refresh();
  };
  document.getElementById('basefrac').oninput = e=>{
    document.getElementById('basefracval').textContent = e.target.value+'%'; refresh();
  };
  TARGET = +document.getElementById('imgsize').value;
  if(j.files.length){ loadData(); }
}
init();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
