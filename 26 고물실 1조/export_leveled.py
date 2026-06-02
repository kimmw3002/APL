"""Level AFM raw CSVs (masked degree-1 polyfit) and emit a self-contained
interactive rod-height measurement page (rod_height.html).

Reuses the `masked_polyfit` logic from plot_heatmap_leveled.py.
"""
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# (csv path, info path, display name)
DATASETS = [
    ("JJ_data/Data_1_Raw.csv", "JJ_data/JJ_AFM_1 Info.txt", "Data_1"),
    ("JJ_data/Data_3_Raw.csv", "JJ_data/JJ_AFM_3 Info.txt", "Data_3"),
    ("JJ_data/Data_5_Raw.csv", "JJ_data/JJ_AFM_5 Info.txt", "Data_5"),
    ("JJ_data/Data_6_Raw.csv", "JJ_data/JJ_AFM_6 Info.txt", "Data_6"),
    ("JJ_data/Data_7_Raw.csv", "JJ_data/JJ_AFM_7 Info.txt", "Data_7"),
]

OUT_HTML = os.path.join(HERE, "rod_height.html")


def masked_polyfit(arr, k=2.5, deg=2, right_exclude=0.2):
    """Per-row degree-`deg` leveling with robust (MAD) outlier rejection.

    Based on plot_heatmap_leveled.py, with two changes:
      * degree-2 (quadratic) fit instead of degree-1, to remove row curvature.
      * the rightmost `right_exclude` fraction of columns is ALWAYS masked out
        (never used in any fit) -- there is a severe step there that would
        otherwise pull the fit. Its leveled values come from extrapolating the
        curve fitted on the left part.
    """
    n, m = arr.shape
    x = np.arange(m)
    keep_right = x < m * (1.0 - right_exclude)   # rightmost frac unconditionally excluded
    out = np.empty_like(arr)
    for i, row in enumerate(arr):
        # initial fit on the kept (left) region only
        coef = np.polyfit(x[keep_right], row[keep_right], deg)
        resid = row - np.polyval(coef, x)
        r = resid[keep_right]
        mad = np.median(np.abs(r - np.median(r))) + 1e-30
        mask = (np.abs(resid) < k * 1.4826 * mad) & keep_right
        if mask.sum() < deg + 6:
            fit = np.polyval(coef, x)
        else:
            coef2 = np.polyfit(x[mask], row[mask], deg)
            fit = np.polyval(coef2, x)
        out[i] = row - fit
    return out


def parse_pixel_nm(info_path, cols):
    """pixel size in nm from `Image size` / `Points`; fall back to 20um/256."""
    size_nm = 20000.0
    points = float(cols)
    try:
        with open(info_path, "r", encoding="utf-8", errors="ignore") as fh:
            text = fh.read()
        m = re.search(r"Image size\s+([0-9.]+)\s*([nµu]?)m", text)
        if m:
            val = float(m.group(1))
            unit = m.group(2)
            scale = {"n": 1.0, "µ": 1000.0, "u": 1000.0, "": 1e9}.get(unit, 1000.0)
            size_nm = val * scale
        p = re.search(r"Points\s+(\d+)", text)
        if p:
            points = float(p.group(1))
    except OSError:
        pass
    return size_nm / points


def build_dataset(csv_rel, info_rel, name):
    csv_path = os.path.join(HERE, csv_rel)
    info_path = os.path.join(HERE, info_rel)
    data = np.loadtxt(csv_path, delimiter=";")
    leveled_nm = masked_polyfit(data) * 1e9  # m -> nm
    rows, cols = leveled_nm.shape
    pixel_nm = parse_pixel_nm(info_path, cols)
    print(f"{name}: shape={rows}x{cols}  pixel={pixel_nm:.3f} nm  "
          f"z[min,max]=[{leveled_nm.min():.2f},{leveled_nm.max():.2f}] nm")
    # round to 3 decimals (pm precision) to keep the embedded blob compact
    z = np.round(leveled_nm, 3).tolist()
    return {
        "name": name,
        "rows": rows,
        "cols": cols,
        "pixel_nm": pixel_nm,
        "z": z,
    }


def main():
    datasets = [build_dataset(*d) for d in DATASETS]
    blob = json.dumps(datasets, separators=(",", ":"))
    html = HTML_TEMPLATE.replace("/*__DATA__*/", blob)
    with open(OUT_HTML, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("saved:", OUT_HTML)


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AFM rod-height tool</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 16px; color: #1a1a1a; }
  h1 { font-size: 18px; margin: 0 0 8px; }
  .row { display: flex; gap: 24px; align-items: flex-start; flex-wrap: wrap; }
  .panel { border: 1px solid #ccc; border-radius: 8px; padding: 12px; }
  canvas { border: 1px solid #999; cursor: crosshair; image-rendering: pixelated; }
  label { font-size: 13px; }
  button { padding: 5px 10px; font-size: 13px; cursor: pointer; }
  table { border-collapse: collapse; font-size: 13px; margin-top: 8px; }
  th, td { border: 1px solid #ccc; padding: 3px 7px; text-align: right; }
  th { background: #f0f0f0; }
  .controls { margin: 8px 0; font-size: 13px; }
  .controls > * { margin-right: 12px; }
  .readout { font-size: 14px; margin: 8px 0; }
  .readout b { font-size: 18px; }
  code { background: #f3f3f3; padding: 1px 4px; border-radius: 3px; }
</style>
</head>
<body>
<h1>AFM rod-height tool &mdash; peak minus local baseline (leveled, masked polyfit)</h1>
<div class="controls">
  <label>Dataset:
    <select id="dataset"></select>
  </label>
  <label>Zoom:
    <select id="zoom">
      <option value="2">2&times;</option>
      <option value="3" selected>3&times;</option>
      <option value="4">4&times;</option>
    </select>
  </label>
  <label>Baseline window (each end):
    <input type="range" id="basefrac" min="5" max="45" value="25" step="5">
    <span id="basefracval">25%</span>
  </label>
  <span>Click two points across a rod's waist. Drag an endpoint to refine.</span>
</div>

<div class="row">
  <div class="panel">
    <canvas id="map"></canvas>
  </div>
  <div class="panel">
    <canvas id="profile" width="580" height="340"></canvas>
    <div class="readout" id="readout">Draw a line to measure.</div>
    <button id="addrod">Add rod &#9656;</button>
    <button id="clearrods">Clear table</button>
    <button id="export">Export results</button>
  </div>
</div>

<div class="panel" style="margin-top:16px; display:inline-block;">
  <table id="results">
    <thead><tr>
      <th>#</th><th>dataset</th>
      <th>x0,y0</th><th>x1,y1</th>
      <th>peak (nm)</th><th>base (nm)</th><th>height (nm)</th><th>&plusmn; (nm)</th>
    </tr></thead>
    <tbody></tbody>
  </table>
</div>

<script>
const DATASETS = /*__DATA__*/;

// ---- viridis colormap (downsampled control points) ----
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

let DS, ZOOM, vmin, vmax;
let p0=null, p1=null, dragging=null;
const mapCv = document.getElementById('map');
const mapCtx = mapCv.getContext('2d');
const profCv = document.getElementById('profile');
const profCtx = profCv.getContext('2d');

// rows go bottom-up to match origin="lower"
function zAt(r, c){ return DS.z[r][c]; } // r in data coords (0 at bottom)

// canvas y is top-down; data row 0 is at bottom => screen_y = (rows-1-r)*ZOOM
function dataToScreen(c, r){ return [c*ZOOM, (DS.rows-1-r)*ZOOM]; }
function screenToData(sx, sy){ return [sx/ZOOM, DS.rows-1-sy/ZOOM]; }

function bilinear(c, r){
  // clamp
  c = Math.max(0, Math.min(DS.cols-1, c));
  r = Math.max(0, Math.min(DS.rows-1, r));
  const c0 = Math.floor(c), r0 = Math.floor(r);
  const c1 = Math.min(c0+1, DS.cols-1), r1 = Math.min(r0+1, DS.rows-1);
  const fc = c-c0, fr = r-r0;
  const v00 = DS.z[r0][c0], v01 = DS.z[r0][c1];
  const v10 = DS.z[r1][c0], v11 = DS.z[r1][c1];
  return v00*(1-fc)*(1-fr) + v01*fc*(1-fr) + v10*(1-fc)*fr + v11*fc*fr;
}

function renderMap(){
  mapCv.width = DS.cols*ZOOM;
  mapCv.height = DS.rows*ZOOM;
  const img = mapCtx.createImageData(DS.cols, DS.rows);
  for(let r=0; r<DS.rows; r++){
    const sy = DS.rows-1-r; // flip for origin lower
    for(let c=0; c<DS.cols; c++){
      const t = (zAt(r,c)-vmin)/(vmax-vmin);
      const [R,G,B] = viridis(t);
      const idx = (sy*DS.cols + c)*4;
      img.data[idx]=R; img.data[idx+1]=G; img.data[idx+2]=B; img.data[idx+3]=255;
    }
  }
  // draw at 1x into an offscreen then scale up
  const off = document.createElement('canvas');
  off.width = DS.cols; off.height = DS.rows;
  off.getContext('2d').putImageData(img, 0, 0);
  mapCtx.imageSmoothingEnabled = false;
  mapCtx.clearRect(0,0,mapCv.width,mapCv.height);
  mapCtx.drawImage(off, 0, 0, mapCv.width, mapCv.height);
  drawLine();
}

function drawLine(){
  if(!p0) return;
  const [sx0, sy0] = dataToScreen(p0[0], p0[1]);
  mapCtx.lineWidth = 2; mapCtx.strokeStyle = '#ff3b3b'; mapCtx.fillStyle = '#ff3b3b';
  if(p1){
    const [sx1, sy1] = dataToScreen(p1[0], p1[1]);
    mapCtx.beginPath(); mapCtx.moveTo(sx0,sy0); mapCtx.lineTo(sx1,sy1); mapCtx.stroke();
    for(const [x,y] of [[sx0,sy0],[sx1,sy1]]){
      mapCtx.beginPath(); mapCtx.arc(x,y,4,0,7); mapCtx.fill();
    }
  } else {
    mapCtx.beginPath(); mapCtx.arc(sx0,sy0,4,0,7); mapCtx.fill();
  }
}

function sampleProfile(){
  // Collect the RAW pixel values of every pixel the line passes through.
  // No interpolation: walk the line finely, snap to the nearest pixel
  // (round), drop consecutive duplicates -> the ordered set of pixels the
  // line hits. Each point is an actual measured pixel value.
  const dx = p1[0]-p0[0], dy = p1[1]-p0[1];
  const lenpx = Math.hypot(dx, dy);
  const steps = Math.max(8, Math.ceil(lenpx*4));   // fine enough to catch every pixel
  const prof = [], dist = [];
  let lastc = -1, lastr = -1;
  for(let i=0;i<=steps;i++){
    const t = i/steps;
    const c = Math.round(p0[0]+dx*t), r = Math.round(p0[1]+dy*t);
    if(c<0 || c>=DS.cols || r<0 || r>=DS.rows) continue;
    if(c===lastc && r===lastr) continue;           // same pixel as previous -> skip dup
    lastc = c; lastr = r;
    prof.push(DS.z[r][c]);                          // raw pixel value (nm)
    const proj = ((c-p0[0])*dx + (r-p0[1])*dy)/lenpx; // pixel center projected onto line
    dist.push(proj*DS.pixel_nm);
  }
  return {prof, dist, lenpx};
}

function analyze(){
  const {prof, dist} = sampleProfile();
  // peak = max RAW pixel value (no smoothing -- pixel values only)
  let pi = 0; for(let i=1;i<prof.length;i++) if(prof[i]>prof[pi]) pi=i;
  const peak = prof[pi];
  // baseline: outer fraction of each end (excludes the peak region)
  const frac = +document.getElementById('basefrac').value/100;
  const k = Math.max(1, Math.round(prof.length*frac));
  const ends = prof.slice(0,k).concat(prof.slice(prof.length-k));
  const base = median(ends);
  const noise = std(ends);
  const height = peak - base;
  return {prof, dist, pi, peak, base, noise, height};
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
  const mL = 52, mB = 40, mT = 14, mR = 12;   // margins (room for tick labels)
  profCtx.clearRect(0,0,W,H);
  if(!A){ profCtx.fillStyle='#666'; profCtx.font='13px sans-serif';
          profCtx.fillText('Draw a line to measure.',mL,H/2); return; }
  const {prof, dist, pi, peak, base} = A;
  const xmin = dist[0], xmax = dist[dist.length-1];
  let ymin = Math.min(...prof), ymax = Math.max(...prof);
  const pad = (ymax-ymin)*0.1 + 1e-6; ymin-=pad; ymax+=pad;
  const X = d => mL + (d-xmin)/(xmax-xmin)*(W-mL-mR);
  const Y = v => H-mB - (v-ymin)/(ymax-ymin)*(H-mB-mT);
  profCtx.font='11px sans-serif';
  // grid + tick numbers
  profCtx.strokeStyle='#e6e6e6'; profCtx.fillStyle='#333'; profCtx.lineWidth=1;
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
  // axes
  profCtx.strokeStyle='#888'; profCtx.beginPath();
  profCtx.moveTo(mL,mT); profCtx.lineTo(mL,H-mB); profCtx.lineTo(W-mR,H-mB); profCtx.stroke();
  // axis titles
  profCtx.fillStyle='#000'; profCtx.textAlign='center'; profCtx.textBaseline='alphabetic';
  profCtx.fillText('distance (nm)', (mL+W-mR)/2, H-6);
  profCtx.save(); profCtx.translate(12,(mT+H-mB)/2); profCtx.rotate(-Math.PI/2);
  profCtx.fillText('height (nm)', 0, 0); profCtx.restore();
  // baseline
  profCtx.strokeStyle='#1f9e6e'; profCtx.setLineDash([5,4]); profCtx.lineWidth=1.5;
  profCtx.beginPath(); profCtx.moveTo(mL,Y(base)); profCtx.lineTo(W-mR,Y(base)); profCtx.stroke();
  profCtx.setLineDash([]);
  // profile line
  profCtx.strokeStyle='#3366cc'; profCtx.lineWidth=1.5; profCtx.beginPath();
  prof.forEach((v,i)=>{ const x=X(dist[i]), y=Y(v); i?profCtx.lineTo(x,y):profCtx.moveTo(x,y); });
  profCtx.stroke();
  // EVERY sampled point as a marker (no downsampling)
  profCtx.fillStyle='#3366cc';
  prof.forEach((v,i)=>{ profCtx.beginPath(); profCtx.arc(X(dist[i]),Y(v),1.8,0,7); profCtx.fill(); });
  // peak marker
  profCtx.fillStyle='#ff3b3b';
  profCtx.beginPath(); profCtx.arc(X(dist[pi]),Y(peak),4.5,0,7); profCtx.fill();
}

function refresh(){
  if(p0 && p1){
    const A = analyze();
    drawProfile(A);
    document.getElementById('readout').innerHTML =
      `peak <b>${A.peak.toFixed(2)}</b> &minus; base ${A.base.toFixed(2)} = ` +
      `height <b>${A.height.toFixed(2)} nm</b> &plusmn; ${A.noise.toFixed(2)}`;
    window._lastA = A;
  } else {
    drawProfile(null);
    document.getElementById('readout').textContent = 'Draw a line to measure.';
    window._lastA = null;
  }
  renderMap();
}

// ---- interaction ----
function nearEndpoint(sx, sy){
  for(const [p,name] of [[p0,'p0'],[p1,'p1']]){
    if(!p) continue;
    const [x,y] = dataToScreen(p[0], p[1]);
    if(Math.hypot(x-sx, y-sy) < 8) return name;
  }
  return null;
}
mapCv.addEventListener('mousedown', e=>{
  const rect = mapCv.getBoundingClientRect();
  const sx = e.clientX-rect.left, sy = e.clientY-rect.top;
  const hit = nearEndpoint(sx, sy);
  if(hit){ dragging = hit; return; }
  const d = screenToData(sx, sy);
  if(!p0 || (p0 && p1)){ p0 = d; p1 = null; }
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
window.addEventListener('mouseup', ()=>{ dragging=null; });

// ---- table ----
const rows = [];
function renderTable(){
  const tb = document.querySelector('#results tbody');
  tb.innerHTML = rows.map((r,i)=>
    `<tr><td>${i+1}</td><td>${r.ds}</td>`+
    `<td>${r.x0},${r.y0}</td><td>${r.x1},${r.y1}</td>`+
    `<td>${r.peak.toFixed(2)}</td><td>${r.base.toFixed(2)}</td>`+
    `<td><b>${r.height.toFixed(2)}</b></td><td>${r.noise.toFixed(2)}</td></tr>`).join('');
}
document.getElementById('addrod').onclick = ()=>{
  const A = window._lastA; if(!A){ alert('Draw a line first.'); return; }
  rows.push({ds:DS.name,
    x0:Math.round(p0[0]),y0:Math.round(p0[1]),x1:Math.round(p1[0]),y1:Math.round(p1[1]),
    peak:A.peak, base:A.base, height:A.height, noise:A.noise});
  renderTable();
};
document.getElementById('clearrods').onclick = ()=>{ rows.length=0; renderTable(); };
document.getElementById('export').onclick = ()=>{
  const hdr = 'dataset,x0,y0,x1,y1,peak_nm,base_nm,height_nm,pm_nm';
  const lines = rows.map(r=>[r.ds,r.x0,r.y0,r.x1,r.y1,
    r.peak.toFixed(3),r.base.toFixed(3),r.height.toFixed(3),r.noise.toFixed(3)].join(','));
  const csv = [hdr,...lines].join('\n');
  const blob = new Blob([csv], {type:'text/csv'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'rod_heights.csv'; a.click();
  console.log(csv);
};

// ---- setup ----
function loadDataset(idx){
  DS = DATASETS[idx];
  const flat = DS.z.flat().sort((a,b)=>a-b);
  vmin = percentile(flat, 0.01); vmax = percentile(flat, 0.99);
  p0=p1=null;
  refresh();
}
const sel = document.getElementById('dataset');
DATASETS.forEach((d,i)=> sel.add(new Option(d.name, i)));
sel.onchange = ()=> loadDataset(+sel.value);
document.getElementById('zoom').onchange = e=>{ ZOOM=+e.target.value; refresh(); };
document.getElementById('basefrac').oninput = e=>{
  document.getElementById('basefracval').textContent = e.target.value+'%'; refresh();
};
ZOOM = +document.getElementById('zoom').value;
loadDataset(0);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
