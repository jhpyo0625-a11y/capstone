"""Interactive defect-region annotation pages (2026-07-12).

The user can see defects that the model misses — these pages let them mark
defect REGIONS (drag boxes or paint with a brush) on the hi-res crops; the
downloaded JSON drives patch-level supervision.

Pages are fully self-contained HTML (images embedded), open offline, and
autosave to localStorage. Coordinates are normalized (0-1) relative to the
anomaly-cache image; brush radius is normalized to image width.

JSON schema v2:
  {"version": 2, "images": [{"relpath", "class", "no_defect_visible",
    "boxes": [{"x0","y0","x1","y1"}],
    "strokes": [{"r": radius_norm, "pts": [{"x","y"}, ...]}]}]}

Run:  uv run python -m coilvision.annotate
Writes artifacts/annotation/annotate_val.html and annotate_train.html.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pandas as pd

from coilvision.anomaly import anomaly_cfg
from coilvision.config import load_config, resolve_path
from coilvision.train.datamodule import load_split_frame

INSTRUCTIONS = """
<b>Tools:</b> <b>Box</b> — drag a rectangle around a defect area. <b>Brush</b> — paint along the
defect (great for wavy arcs); size slider on the left. Click <b>undo</b> to remove the last mark,
<b>clear</b> to wipe an image. Keys: <b>b</b>=box, <b>p</b>=brush, <b>u</b>=undo.
If you truly see no defect, tick <i>no defect visible</i>.
<br><b>Loose</b> = spread / wavy winding turns, typically at the coil's rounded ends.
<b>Dent</b> = locally crushed or kinked winding arc.
<br>Progress autosaves in your browser — you can close and reopen this page safely.
When done, press <b>Download annotations</b> and tell Claude (the file lands in Downloads).
"""

STYLE = """
<style>
body { font-family: sans-serif; background: #1b1b1b; color: #eee; margin: 16px; }
.item { margin-bottom: 28px; }
.item h3 { margin: 4px 0; font-size: 15px; }
.imgwrap { position: relative; display: inline-block; }
.imgwrap img { display: block; max-width: 100%; width: 1536px; user-select: none; -webkit-user-drag: none; }
.imgwrap canvas { position: absolute; inset: 0; cursor: crosshair; }
#bar { position: sticky; top: 0; background: #333; padding: 10px; z-index: 10; margin-bottom: 14px; }
button { font-size: 14px; padding: 5px 12px; margin: 0 2px; }
label.tool { margin-right: 10px; }
.qc { color: #ffb84d; }
#restored { color: #7fdc7f; }
</style>
"""

SCRIPT = """
<script>
const KEY = 'coil_annot_' + DOWNLOAD_NAME;
let STATE = META.map(m => ({relpath: m.relpath, class: m.class, no_defect_visible: false, boxes: [], strokes: [], ops: []}));
try {
  const saved = JSON.parse(localStorage.getItem(KEY) || 'null');
  if (saved && saved.length === STATE.length && saved.every((s, i) => s.relpath === STATE[i].relpath)) {
    STATE = saved;
    document.getElementById('restored').textContent = '(restored from autosave)';
  }
} catch (e) {}

let tool = 'box', brushPx = 14, lastActive = 0;
document.querySelectorAll('input[name=tool]').forEach(r => r.onchange = () => tool = r.value);
const sizeEl = document.getElementById('bsize');
sizeEl.oninput = () => { brushPx = +sizeEl.value; document.getElementById('bsizev').textContent = brushPx; };

function save() { try { localStorage.setItem(KEY, JSON.stringify(STATE)); } catch (e) {} }
function count() {
  const n = STATE.filter(s => s.boxes.length || s.strokes.length || s.no_defect_visible).length;
  document.getElementById('progress').textContent = `${n} / ${STATE.length} images annotated`;
}

function redraw(i, extra) {
  const cv = document.querySelector(`.item[data-i="${i}"] canvas`);
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  const s = STATE[i];
  ctx.fillStyle = 'rgba(0,255,94,0.18)'; ctx.strokeStyle = 'rgba(0,255,94,0.95)'; ctx.lineWidth = 2;
  for (const b of s.boxes) {
    ctx.fillRect(b.x0 * W, b.y0 * H, (b.x1 - b.x0) * W, (b.y1 - b.y0) * H);
    ctx.strokeRect(b.x0 * W, b.y0 * H, (b.x1 - b.x0) * W, (b.y1 - b.y0) * H);
  }
  ctx.strokeStyle = 'rgba(0,255,94,0.45)'; ctx.lineCap = ctx.lineJoin = 'round';
  ctx.fillStyle = 'rgba(0,255,94,0.45)';
  for (const st of s.strokes) {
    const lw = st.r * 2 * W;
    if (st.pts.length === 1) { ctx.beginPath(); ctx.arc(st.pts[0].x * W, st.pts[0].y * H, lw / 2, 0, 7); ctx.fill(); continue; }
    ctx.lineWidth = lw; ctx.beginPath();
    st.pts.forEach((p, j) => j ? ctx.lineTo(p.x * W, p.y * H) : ctx.moveTo(p.x * W, p.y * H));
    ctx.stroke();
  }
  if (extra) extra(ctx, W, H);
}

function undo(i) {
  const s = STATE[i], op = s.ops.pop();
  if (op === 'box') s.boxes.pop(); else if (op === 'stroke') s.strokes.pop();
  redraw(i); count(); save();
}

document.querySelectorAll('.item').forEach(item => {
  const i = +item.dataset.i;
  const img = item.querySelector('img');
  const cv = item.querySelector('canvas');
  const fit = () => { cv.width = img.clientWidth; cv.height = img.clientHeight; redraw(i); };
  if (img.complete) fit(); else img.onload = fit;
  window.addEventListener('resize', fit);

  let drag = null;
  const norm = e => {
    const r = cv.getBoundingClientRect();
    return {x: Math.min(Math.max((e.clientX - r.left) / r.width, 0), 1),
            y: Math.min(Math.max((e.clientY - r.top) / r.height, 0), 1)};
  };
  cv.onpointerdown = e => {
    e.preventDefault(); cv.setPointerCapture(e.pointerId); lastActive = i;
    const p = norm(e);
    drag = tool === 'box' ? {t: 'box', a: p, b: p} : {t: 'stroke', r: brushPx / cv.width, pts: [p]};
    if (drag.t === 'stroke') redraw(i, ctx => drawLive(ctx, cv));
  };
  cv.onpointermove = e => {
    if (!drag) return;
    const p = norm(e);
    if (drag.t === 'box') drag.b = p;
    else {
      const last = drag.pts[drag.pts.length - 1];
      if (Math.hypot((p.x - last.x) * cv.width, (p.y - last.y) * cv.height) > 3) drag.pts.push(p);
    }
    redraw(i, ctx => drawLive(ctx, cv));
  };
  const drawLive = (ctx) => {
    const W = cv.width, H = cv.height;
    if (drag.t === 'box') {
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5;
      ctx.strokeRect(Math.min(drag.a.x, drag.b.x) * W, Math.min(drag.a.y, drag.b.y) * H,
                     Math.abs(drag.b.x - drag.a.x) * W, Math.abs(drag.b.y - drag.a.y) * H);
    } else {
      ctx.strokeStyle = 'rgba(255,255,255,0.6)'; ctx.fillStyle = 'rgba(255,255,255,0.6)';
      ctx.lineCap = ctx.lineJoin = 'round';
      const lw = drag.r * 2 * W;
      if (drag.pts.length === 1) { ctx.beginPath(); ctx.arc(drag.pts[0].x * W, drag.pts[0].y * H, lw / 2, 0, 7); ctx.fill(); }
      else { ctx.lineWidth = lw; ctx.beginPath();
             drag.pts.forEach((p, j) => j ? ctx.lineTo(p.x * W, p.y * H) : ctx.moveTo(p.x * W, p.y * H));
             ctx.stroke(); }
    }
  };
  cv.onpointerup = () => {
    if (!drag) return;
    const s = STATE[i];
    if (drag.t === 'box') {
      const b = {x0: Math.min(drag.a.x, drag.b.x), y0: Math.min(drag.a.y, drag.b.y),
                 x1: Math.max(drag.a.x, drag.b.x), y1: Math.max(drag.a.y, drag.b.y)};
      if ((b.x1 - b.x0) * cv.width > 4 && (b.y1 - b.y0) * cv.height > 4) { s.boxes.push(b); s.ops.push('box'); }
    } else { s.strokes.push({r: drag.r, pts: drag.pts}); s.ops.push('stroke'); }
    drag = null; redraw(i); count(); save();
  };

  item.querySelector('.nodef').checked = STATE[i].no_defect_visible;
  item.querySelector('.nodef').onchange = e => { STATE[i].no_defect_visible = e.target.checked; count(); save(); };
  item.querySelector('.undo').onclick = () => undo(i);
  item.querySelector('.clear').onclick = () => {
    if (!confirm('Clear all marks on this image?')) return;
    STATE[i].boxes = []; STATE[i].strokes = []; STATE[i].ops = [];
    redraw(i); count(); save();
  };
});

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'b') document.querySelector('input[name=tool][value=box]').click();
  if (e.key === 'p') document.querySelector('input[name=tool][value=stroke]').click();
  if (e.key === 'u') undo(lastActive);
});

document.getElementById('dl').onclick = () => {
  const out = STATE.map(s => ({relpath: s.relpath, class: s.class, no_defect_visible: s.no_defect_visible,
                               boxes: s.boxes, strokes: s.strokes}));
  const blob = new Blob([JSON.stringify({version: 2, images: out}, null, 1)], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = DOWNLOAD_NAME;
  a.click();
};
count();
</script>
"""


def _b64(path: Path) -> str:
    import cv2

    img = cv2.imread(str(path))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise IOError(f"encode failed: {path}")
    return base64.b64encode(buf).decode()


def build_page(rows: list[dict], cache_dir: Path, out_path: Path, title: str, download_name: str) -> None:
    meta, blocks = [], []
    for i, r in enumerate(rows):
        meta.append({"relpath": r["relpath"], "class": r["class"]})
        note = f" — score {r['score']:.3f}" if "score" in r else ""
        qc = " <span class='qc'>(machine says Pass — only mark if you DO see a defect)</span>" if r.get("qc") else ""
        blocks.append(
            f"<div class='item' data-i='{i}'>"
            f"<h3>#{i + 1} / {len(rows)} — <b>{r['class']}</b>{note} · {Path(r['relpath']).name}{qc}"
            f" <label><input type='checkbox' class='nodef'> no defect visible</label>"
            f" <button class='undo'>undo</button> <button class='clear'>clear</button></h3>"
            f"<div class='imgwrap'><img src='data:image/jpeg;base64,{_b64(cache_dir / r['cache_file'])}'>"
            f"<canvas></canvas></div>"
            f"</div>"
        )
    html = (
        f"<html><head><meta charset='utf-8'><title>{title}</title>{STYLE}</head><body>"
        f"<div id='bar'><b>{title}</b> · <span id='progress'></span> <span id='restored'></span> "
        f"<label class='tool'><input type='radio' name='tool' value='box' checked> Box</label>"
        f"<label class='tool'><input type='radio' name='tool' value='stroke'> Brush</label>"
        f"size <input type='range' id='bsize' min='6' max='48' value='14' style='vertical-align:middle'>"
        f"<span id='bsizev'>14</span>px "
        f"<button id='dl'>Download annotations</button><br><small>{INSTRUCTIONS}</small></div>"
        + "".join(blocks)
        + f"<script>const META = {json.dumps(meta)}; const DOWNLOAD_NAME = {json.dumps(download_name)};</script>"
        + SCRIPT
        + "</body></html>"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"wrote {out_path} ({len(rows)} images, {out_path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    cfg2 = anomaly_cfg(load_config())
    cache_dir = resolve_path(cfg2, "cache_dir")
    out_dir = resolve_path(cfg2, "artifacts_dir") / "annotation"

    # newest anomaly run with the masked scores → hardest-first ordering for val
    runs = sorted((resolve_path(cfg2, "artifacts_dir") / "runs").glob("anomaly_*/val_scores.csv"))
    scores = pd.read_csv(runs[-1], keep_default_na=False) if runs else None

    val = load_split_frame("val", cfg2)
    if scores is not None and "score_masked_top10" in scores.columns:
        val = val.merge(scores[["relpath", "score_masked_top10"]], on="relpath", how="left")
    else:
        val["score_masked_top10"] = 0.0

    defects = val[val["class"] != "Pass"].sort_values("score_masked_top10")
    qc = val[val["class"] == "Pass"].nlargest(5, "score_masked_top10")
    rows = [
        {"relpath": r["relpath"], "class": r["class"], "cache_file": r["cache_file"], "score": r["score_masked_top10"]}
        for _, r in defects.iterrows()
    ] + [
        {"relpath": r["relpath"], "class": r["class"], "cache_file": r["cache_file"],
         "score": r["score_masked_top10"], "qc": True}
        for _, r in qc.iterrows()
    ]
    build_page(rows, cache_dir, out_dir / "annotate_val.html", "Coil defect annotation — VAL set", "annotations_val.json")

    train = load_split_frame("train", cfg2)
    tdef = train[train["class"] != "Pass"].sort_values(["class", "run", "relpath"])
    trows = [{"relpath": r["relpath"], "class": r["class"], "cache_file": r["cache_file"]} for _, r in tdef.iterrows()]
    build_page(trows, cache_dir, out_dir / "annotate_train.html", "Coil defect annotation — TRAIN set", "annotations_train.json")


if __name__ == "__main__":
    main()
