"""Play against a network in the browser: write one self-contained HTML page.

    python3 -m connectk webplay my.pt --game connect5 --out play.html

The page holds the weights and runs the network itself (a JavaScript port of
connectk/net.py and of a small MCTS), so it needs no server. Open it in any
browser. Three opponent levels: the net alone (greedy, the way graders play
it), and the net with a light or a stronger search on top.
"""

import base64
import json

import numpy as np
import torch

from .net import load_net


def net_entry(spec, path, label, sims=None, short=None, play_sims=None, play_sims_cpu=None):
    """One network for the page.

    sims           [light, strong] search sizes shown for this net (optional)
    short          the text on its button (default: the game and board size)
    play_sims      if set, the net plays with this search size at every level,
                   whatever the buttons say
    play_sims_cpu  the same, used instead when the browser has no WebGL
    """
    try:
        from hidden.strong.bignet import load_model     # our big nets, when the repo has them
        net = load_model(spec, path)
    except ImportError:
        net = load_net(spec, path)
    tensors = {}
    half = sum(t.numel() for t in net.state_dict().values()) > 1_000_000   # big nets: float16, half the page size
    for name, t in net.state_dict().items():
        arr = t.detach().cpu().numpy().astype("<f2" if half else "<f4")
        tensors[name] = {"shape": list(arr.shape), "dtype": "f16" if half else "f32",
                         "data": base64.b64encode(arr.tobytes()).decode()}
    entry = {"label": label, "spec": spec.name, "rows": spec.rows, "cols": spec.cols, "k": spec.k, "tensors": tensors}
    if sims:
        entry["sims"] = list(sims)
    if short:
        entry["short"] = short
    if play_sims is not None:
        entry["play_sims"] = int(play_sims)
        entry["play_sims_cpu"] = int(play_sims if play_sims_cpu is None else play_sims_cpu)
    return entry


TFJS = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js"


def render_html(entries, title="Play Connect", standalone=True, gpu=False):
    """gpu=True loads TensorFlow.js and runs the nets with WebGL when the browser has it
    (many times faster for big nets); without WebGL the page uses its plain JavaScript."""
    page = _TEMPLATE.replace("__TITLE__", title).replace("__NETS__", json.dumps(entries, separators=(",", ":")))
    page = page.replace("__GPU__", "true" if gpu else "false")
    if gpu:
        page = page.replace("<script>\nconst NETS", f'<script src="{TFJS}"></script>\n<script>\nconst NETS', 1)
    if standalone:
        page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                + page.replace("</style>", "</style>\n</head>\n<body>", 1) + "\n</body>\n</html>\n")
    return page


def write_html(entries, path, title="Play Connect", standalone=True, gpu=False):
    with open(path, "w") as f:
        f.write(render_html(entries, title, standalone, gpu))
    return path


# Shared by the worker and the page: board rules, the network forward pass, tactics and MCTS.
_LIB = r"""
function specOf(n) {
  const R = n.rows, C = n.cols, K = n.k, wins = [], cellWins = Array.from({length: R * C}, () => []);
  for (let r = 0; r < R; r++) for (let c = 0; c < C; c++)
    for (const [dr, dc] of [[0, 1], [1, 0], [1, 1], [1, -1]]) {
      const er = r + dr * (K - 1), ec = c + dc * (K - 1);
      if (er < 0 || er >= R || ec < 0 || ec >= C) continue;
      const w = []; for (let i = 0; i < K; i++) w.push((r + dr * i) * C + c + dc * i);
      for (const x of w) cellWins[x].push(wins.length);
      wins.push(w);
    }
  return {R, C, K, wins, cellWins};
}
function halfToFloat(u) {
  const out = new Float32Array(u.length);
  for (let i = 0; i < u.length; i++) {
    const h = u[i], sg = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
    out[i] = e === 0 ? sg * f * 5.960464477539063e-8 : e === 31 ? (f ? NaN : sg * Infinity) : sg * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  return out;
}
function heightsOf(g, s) { const h = new Int32Array(s.C); for (let c = 0; c < s.C; c++) { let t = 0; for (let r = 0; r < s.R; r++) if (g[r * s.C + c]) t = r + 1; h[c] = t; } return h; }
function winsAt(g, s, cell) {
  const p = g[cell];
  for (const w of s.cellWins[cell]) { let ok = true; for (const x of s.wins[w]) if (g[x] !== p) { ok = false; break; } if (ok) return true; }
  return false;
}
// ---- network
function conv(inp, inC, H, W, w, b, outC, k) {
  const HW = H * W, out = new Float32Array(outC * HW), pad = k >> 1, KK = inC * k * k;
  let col = inp;
  if (k > 1) {
    col = new Float32Array(KK * HW);
    for (let i = 0; i < inC; i++) for (let ky = 0; ky < k; ky++) for (let kx = 0; kx < k; kx++) {
      const row = ((i * k + ky) * k + kx) * HW, dy = ky - pad, dx = kx - pad;
      for (let y = 0; y < H; y++) { const yy = y + dy; if (yy < 0 || yy >= H) continue;
        for (let x = 0; x < W; x++) { const xx = x + dx; if (xx < 0 || xx >= W) continue; col[row + y * W + x] = inp[i * HW + yy * W + xx]; } }
    }
  }
  for (let o = 0; o < outC; o++) {
    const ob = o * HW, bo = b[o];
    for (let p = 0; p < HW; p++) out[ob + p] = bo;
    for (let j = 0; j < KK; j++) { const wv = w[o * KK + j]; if (wv === 0) continue; const cb = j * HW; for (let p = 0; p < HW; p++) out[ob + p] += wv * col[cb + p]; }
  }
  return out;
}
function relu(a) { for (let i = 0; i < a.length; i++) if (a[i] < 0) a[i] = 0; return a; }
function linear(x, w, b, out) { const n = x.length, y = new Float32Array(out); for (let o = 0; o < out; o++) { let s = b[o]; const ow = o * n; for (let i = 0; i < n; i++) s += w[ow + i] * x[i]; y[o] = s; } return y; }
function forward(net, g, player, s) {
  const R = s.R, C = s.C, HW = R * C, t = net.t, sh = net.sh, x = new Float32Array(4 * HW), h = heightsOf(g, s);
  for (let i = 0; i < HW; i++) { if (g[i] === player) x[i] = 1; else if (g[i] === 3 - player) x[HW + i] = 1; x[3 * HW + i] = 1; }
  for (let c = 0; c < C; c++) if (h[c] < R) x[2 * HW + h[c] * C + c] = 1;
  let a, ch;
  if (t["stem.weight"]) {                       // residual net (hidden/strong/bignet.py)
    ch = sh["stem.weight"][0];
    a = relu(conv(x, 4, R, C, t["stem.weight"], t["stem.bias"], ch, 3));
    for (let b = 0; t[`blocks.${b}.c1.weight`]; b++) {
      const y = relu(conv(a, ch, R, C, t[`blocks.${b}.c1.weight`], t[`blocks.${b}.c1.bias`], ch, 3));
      const z = conv(y, ch, R, C, t[`blocks.${b}.c2.weight`], t[`blocks.${b}.c2.bias`], ch, 3);
      for (let i = 0; i < a.length; i++) { const v = a[i] + z[i]; a[i] = v > 0 ? v : 0; }
    }
  } else {                                      // the fixed ConnectNet
    ch = sh["body.0.weight"][0];
    a = relu(conv(x, 4, R, C, t["body.0.weight"], t["body.0.bias"], ch, 3));
    for (let L = 2; t[`body.${L}.weight`]; L += 2) a = relu(conv(a, ch, R, C, t[`body.${L}.weight`], t[`body.${L}.bias`], ch, 3));
  }
  const pc = relu(conv(a, ch, R, C, t["policy_conv.weight"], t["policy_conv.bias"], sh["policy_conv.weight"][0], 1));
  const logits = linear(pc, t["policy_fc.weight"], t["policy_fc.bias"], C);
  const vc = relu(conv(a, ch, R, C, t["value_conv.weight"], t["value_conv.bias"], sh["value_conv.weight"][0], 1));
  const v1 = relu(linear(vc, t["value_fc1.weight"], t["value_fc1.bias"], sh["value_fc1.weight"][0]));
  const value = Math.tanh(linear(v1, t["value_fc2.weight"], t["value_fc2.bias"], 1)[0]);
  return {logits, value, h};
}
// ---- tactics (side to move = player): +1 can win now or make a double threat, -1 opponent has 2+ wins
function winCells(g, s, h, p) {
  const out = new Set();
  for (let c = 0; c < s.C; c++) { if (h[c] >= s.R) continue; const cell = h[c] * s.C + c;
    for (const w of s.cellWins[cell]) { let n = 0, bad = false; for (const x of s.wins[w]) { if (x === cell) continue; if (g[x] === p) n++; else { bad = true; break; } } if (!bad && n === s.K - 1) { out.add(cell); break; } } }
  return out.size;
}
function tactics(g, s, h, p) {
  if (winCells(g, s, h, p) > 0) return 1;
  const theirs = winCells(g, s, h, 3 - p);
  if (theirs >= 2) return -1;
  if (theirs > 0) return 0;
  for (let c = 0; c < s.C; c++) { if (h[c] >= s.R) continue; const cell = h[c] * s.C + c; g[cell] = p; h[c]++;
    const dbl = winCells(g, s, h, p) >= 2 && winCells(g, s, h, 3 - p) === 0; h[c]--; g[cell] = 0; if (dbl) return 1; }
  return 0;
}
// ---- MCTS (async, so the same code runs with a plain or a GPU evaluator)
function Node(g, player, s, done, winner) { this.g = g; this.p = player; this.done = done; this.winner = winner; this.P = null; this.N = null; this.W = null; this.ch = null; }
async function expand(node, evalFn, s) {
  const {logits, value, h} = await evalFn(node.g, node.p);
  const P = new Float32Array(s.C); let m = -Infinity;
  for (let c = 0; c < s.C; c++) if (h[c] < s.R) m = Math.max(m, logits[c]);
  let z = 0; for (let c = 0; c < s.C; c++) if (h[c] < s.R) { P[c] = Math.exp(logits[c] - m); z += P[c]; }
  for (let c = 0; c < s.C; c++) P[c] /= z;
  node.P = P; node.N = new Float32Array(s.C); node.W = new Float32Array(s.C); node.ch = new Array(s.C).fill(null); node.h = h;
  const tac = tactics(node.g.slice(), s, h.slice(), node.p);
  return tac !== 0 ? tac : value;
}
function child(node, a, s) {
  if (node.ch[a]) return node.ch[a];
  const g = node.g.slice(), cell = node.h[a] * s.C + a; g[cell] = node.p;
  const won = winsAt(g, s, cell); let full = true;
  for (let c = 0; c < s.C; c++) if ((c === a ? node.h[c] + 1 : node.h[c]) < s.R) { full = false; break; }
  return (node.ch[a] = new Node(g, 3 - node.p, s, won || full, won ? node.p : 0));
}
async function mcts(evalFn, g, player, s, sims, deadline = Infinity, minSims = 0) {
  const root = new Node(g, player, s, false, 0);
  const rootValue = await expand(root, evalFn, s);
  for (let i = 0; i < sims; i++) {
    if (i >= minSims && performance.now() > deadline) break;      // out of time: play the best move so far
    let node = root; const path = [];
    while (node.P && !node.done) {
      let tot = 0, wsum = 0; for (let c = 0; c < s.C; c++) { tot += node.N[c]; wsum += node.W[c]; }
      const fpu = tot > 0 ? wsum / tot - 0.2 : 0, sq = Math.sqrt(tot + 1);
      let best = -1, bs = -Infinity;
      for (let c = 0; c < s.C; c++) { if (node.h[c] >= s.R) continue;
        const q = node.N[c] > 0 ? node.W[c] / node.N[c] : fpu, u = 1.5 * node.P[c] * sq / (1 + node.N[c]);
        if (q + u > bs) { bs = q + u; best = c; } }
      path.push([node, best]); node = child(node, best, s);
    }
    let v = node.done ? (node.winner ? -1 : 0) : await expand(node, evalFn, s);
    for (let j = path.length - 1; j >= 0; j--) { v = -v; const [n, a] = path[j]; n.N[a] += 1; n.W[a] += v; }
  }
  let best = 0; for (let c = 1; c < s.C; c++) if (root.N[c] > root.N[best]) best = c;
  return {move: best, value: rootValue, visits: Array.from(root.N)};
}
async function chooseMove(evalFn, g, player, s, sims, budgetMs = Infinity, minSims = 0) {
  if (sims > 0) return mcts(evalFn, g, player, s, sims, performance.now() + budgetMs, minSims);
  const {logits, value, h} = await evalFn(g, player);
  let best = -1; for (let c = 0; c < s.C; c++) if (h[c] < s.R && (best < 0 || logits[c] > logits[best])) best = c;
  return {move: best, value};
}
"""

# The worker: the plain JavaScript path (no GPU).
_WORKER = _LIB + r"""
let NETS = {};
onmessage = async (e) => {
  const m = e.data;
  if (m.type === "load") {
    for (const n of m.nets) { const t = {};
      for (const k in n.tensors) t[k] = n.dtypes[k] === "f16" ? halfToFloat(new Uint16Array(n.tensors[k])) : new Float32Array(n.tensors[k]);
      NETS[n.id] = {t, sh: n.shapes, spec: specOf(n)}; }
    postMessage({type: "ready"}); return;
  }
  const net = NETS[m.id], s = net.spec, t0 = performance.now();
  const r = await chooseMove((g, p) => forward(net, g, p, s), Int8Array.from(m.grid), m.player, s, m.sims, m.budgetMs ?? Infinity, m.minSims ?? 0);
  postMessage({type: "move", move: r.move, value: r.value, ms: performance.now() - t0, req: m.req});
};
"""

_TEMPLATE = r"""<title>__TITLE__</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=Atkinson+Hyperlegible:wght@400;700&family=IBM+Plex+Mono:wght@400;600&display=swap">
<style>
/* Layout: the board fills the page; a narrow rail holds the setup, status and the net's view. */
:root {
  --paper: #eef1f6; --panel: #f8f9fc; --ink: #142033; --muted: #5b6880; --rule: #d3dae6;
  --board: #1f4e9c; --board-edge: #163b78; --hole: #dfe6f1; --hover: #c9d6ea;
  --p1: #e04848; --p2: #f2b632; --good: #2f8a57; --bad: #c2410c; --glow: #ffffff;
  --display: "Bricolage Grotesque", "Avenir Next", "Segoe UI", sans-serif;
  --body: "Atkinson Hyperlegible", "Segoe UI", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --paper: #0d1420; --panel: #131c2b; --ink: #e6ecf5; --muted: #93a1b8; --rule: #253247;
  --board: #1b3f7d; --board-edge: #12305f; --hole: #0a1018; --hover: #20304a;
  --p1: #f05a5a; --p2: #f5c044; --good: #4cc38a; --bad: #fb923c; --glow: #ffffff; color-scheme: dark; } }
:root[data-theme="dark"] {
  --paper: #0d1420; --panel: #131c2b; --ink: #e6ecf5; --muted: #93a1b8; --rule: #253247;
  --board: #1b3f7d; --board-edge: #12305f; --hole: #0a1018; --hover: #20304a;
  --p1: #f05a5a; --p2: #f5c044; --good: #4cc38a; --bad: #fb923c; --glow: #ffffff; color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--paper); color: var(--ink); font: 15px/1.5 var(--body);
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 20px 32px; }
.wrap { max-width: 1060px; margin: 0 auto; display: grid; gap: 16px; }
header { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 6px 16px; }
h1 { font: 800 clamp(24px, 4vw, 34px)/1.1 var(--display); margin: 0; letter-spacing: -0.01em; }
.sub { color: var(--muted); font-size: 14px; }
.main { display: grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 20px; align-items: start; }
@media (max-width: 760px) { .main { grid-template-columns: minmax(0, 1fr); } }
.board { display: grid; gap: clamp(2px, 0.6vw, 6px); padding: clamp(6px, 1.4vw, 14px); background: var(--board);
  border-radius: 14px; box-shadow: inset 0 -6px 0 var(--board-edge); width: 100%; max-width: 640px; margin: 0 auto; touch-action: manipulation; }
.cell { position: relative; aspect-ratio: 1; border-radius: 50%; background: var(--hole); overflow: hidden; }
.board.mine .cell.colhover { background: var(--hover); }
.board.mine { cursor: pointer; }
.stone { position: absolute; inset: 0; border-radius: 50%; }
.stone.p1 { background: radial-gradient(circle at 35% 30%, color-mix(in srgb, var(--p1) 70%, #fff), var(--p1) 60%); }
.stone.p2 { background: radial-gradient(circle at 35% 30%, color-mix(in srgb, var(--p2) 70%, #fff), var(--p2) 60%); }
.stone.ghost { opacity: 0.35; }
.stone.drop { animation: fall 240ms cubic-bezier(.45,0,.9,.6); }
@keyframes fall { from { transform: translateY(calc(var(--fall) * -112%)); } to { transform: none; } }
@media (prefers-reduced-motion: reduce) { .stone.drop { animation: none; } }
.cell.last .stone { box-shadow: inset 0 0 0 3px color-mix(in srgb, var(--ink) 55%, transparent); }
.cell.win .stone { box-shadow: inset 0 0 0 3px var(--glow), 0 0 0 2px var(--glow); }
.cols { display: grid; gap: clamp(2px, 0.6vw, 6px); padding: 4px clamp(6px, 1.4vw, 14px) 0; max-width: 640px; margin: 0 auto;
  font: 600 11px/1 var(--mono); color: var(--muted); text-align: center; }
aside { background: var(--panel); border: 1px solid var(--rule); border-radius: 12px; padding: 14px; display: grid; gap: 14px; min-width: 0; }
fieldset { border: 0; padding: 0; margin: 0; display: grid; gap: 6px; }
legend { font: 600 12px var(--mono); text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin-bottom: 4px; }
.seg { display: flex; flex-wrap: wrap; gap: 6px; }
.seg label { display: inline-flex; }
.seg input { position: absolute; opacity: 0; pointer-events: none; }
.seg span { font: 600 13px/1.2 var(--body); border: 1px solid var(--rule); background: var(--paper); color: var(--ink); border-radius: 999px; padding: 7px 11px; cursor: pointer; }
.seg input:checked + span { background: var(--board); border-color: var(--board); color: #fff; }
.seg input:focus-visible + span { outline: 3px solid var(--p2); outline-offset: 2px; }
.hint { font-size: 12px; color: var(--muted); }
.buttons { display: flex; gap: 8px; flex-wrap: wrap; }
button.act { font: 600 14px var(--body); height: 38px; padding: 0 14px; border-radius: 8px; border: 1px solid var(--rule); background: var(--paper); color: var(--ink); cursor: pointer; }
button.act.primary { background: var(--board); border-color: var(--board); color: #fff; }
button.act:disabled { opacity: 0.5; cursor: default; }
button:focus-visible { outline: 3px solid var(--p2); outline-offset: 2px; }
.status { font: 700 16px/1.35 var(--body); min-height: 2.7em; }
.status small { display: block; font: 400 13px var(--body); color: var(--muted); }
.status.won { color: var(--good); } .status.lost { color: var(--bad); }
.meter { display: grid; gap: 6px; }
.bar { position: relative; height: 10px; border-radius: 999px; background: color-mix(in srgb, var(--ink) 12%, transparent); overflow: hidden; }
.bar i { position: absolute; inset: 0 auto 0 0; background: var(--good); border-radius: 999px; transition: width 300ms; }
.meter .row { display: flex; justify-content: space-between; font: 12px var(--mono); color: var(--muted); font-variant-numeric: tabular-nums; }
.who { display: flex; gap: 12px; flex-wrap: wrap; font-size: 13px; }
.dot { display: inline-block; width: 12px; height: 12px; border-radius: 50%; vertical-align: -1px; margin-right: 5px; }
.dot.p1 { background: var(--p1); } .dot.p2 { background: var(--p2); }
.thinking::after { content: ""; display: inline-block; width: 1em; text-align: left; animation: dots 1s steps(4) infinite; }
@keyframes dots { 0% { content: ""; } 25% { content: "."; } 50% { content: ".."; } 75% { content: "..."; } }
</style>
<div class="wrap">
  <header>
    <h1>__TITLE__</h1>
    <div class="sub" id="rule">Drop a stone: click a column.</div>
  </header>
  <div class="main">
    <div>
      <div class="board" id="board" role="grid" aria-label="Game board"></div>
      <div class="cols" id="cols" aria-hidden="true"></div>
    </div>
    <aside>
      <div class="status" id="status" aria-live="polite">Loading the network…</div>
      <div class="buttons">
        <button class="act primary" id="newgame">New game</button>
        <button class="act" id="undo">Undo</button>
      </div>
      <fieldset><legend>Game</legend><div class="seg" id="games"></div></fieldset>
      <fieldset><legend>Opponent</legend><div class="seg" id="levels"></div>
        <div class="hint" id="levelHint"></div></fieldset>
      <fieldset><legend>You play</legend><div class="seg">
        <label><input type="radio" name="side" id="sideFirst" value="1" checked><span>First (red)</span></label>
        <label><input type="radio" name="side" id="sideSecond" value="2"><span>Second (yellow)</span></label>
      </div></fieldset>
      <div class="meter">
        <div class="row"><span>Net's view of the position</span><span id="evalText">–</span></div>
        <div class="bar"><i id="evalBar" style="width:50%"></i></div>
        <div class="hint">From the net's value head after its last move: green is good for you.</div>
      </div>
      <div class="who"><span><span class="dot p1"></span>first player</span><span><span class="dot p2"></span>second player</span></div>
    </aside>
  </div>
</div>
<script>
const NETS = __NETS__;
const WORKER_SRC = __WORKER__;
const LIB = new Function(__LIB__ + "; return {specOf, heightsOf, halfToFloat, chooseMove};")();
const USE_GPU = __GPU__;
const LEVELS = [   // simulations per game size: [9 columns, 13 columns]
  {id: "greedy", label: "Net only", sims: [0, 0], hint: () => "The net's top move, no search. This is how graders play a submission."},
  {id: "light", label: "Net + light search", sims: [32, 16], hint: (n) => `A small search on top of the net (${n} simulations).`},
  {id: "strong", label: "Net + strong search", sims: [128, 64], hint: (n) => `The hard bot: the net with a ${n}-simulation search. A few seconds per move.`},
];
function simsFor(level) {   // a net may set its own [light, strong] sizes (a big net is slower in a browser)
  const own = spec().sims; if (own && level.id !== "greedy") return own[level.id === "light" ? 0 : 1];
  return level.sims[spec().cols > 9 ? 1 : 0];
}
function playSims(level) {  // what the net really plays with: a net's play_sims overrides every level
  const n = spec();
  if (n.play_sims !== undefined) return gpu ? n.play_sims : (n.play_sims_cpu ?? n.play_sims);
  return simsFor(level);
}
function showHint() { const lv = LEVELS.find((l) => l.id === state.level); $("levelHint").textContent = lv.hint(simsFor(lv)); }
const $ = (id) => document.getElementById(id);
function b64ToBuf(b64) { const s = atob(b64), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return u.buffer; }
let worker = null, gpu = null, ready = false, req = 0;
const state = {net: 0, level: "strong", human: 1, g: null, h: null, player: 1, moves: [], winner: 0, winLine: [], busy: false, lastEval: null};
function spec() { return NETS[state.net]; }
function resetState() {
  const s = spec(); req++;
  state.g = new Int8Array(s.rows * s.cols); state.h = new Int32Array(s.cols); state.player = 1; state.moves = [];
  state.winner = 0; state.winLine = []; state.busy = false; state.lastEval = null;
}
function newGame() { resetState(); build(); render(); maybeNet(); }
function lineAt(cell) {
  const s = spec(), R = s.rows, C = s.cols, K = s.k, g = state.g, p = g[cell], r0 = Math.floor(cell / C), c0 = cell % C;
  for (const [dr, dc] of [[0, 1], [1, 0], [1, 1], [1, -1]]) {
    const cells = [cell];
    for (const sgn of [1, -1]) { let r = r0 + sgn * dr, c = c0 + sgn * dc; while (r >= 0 && r < R && c >= 0 && c < C && g[r * C + c] === p) { cells.push(r * C + c); r += sgn * dr; c += sgn * dc; } }
    if (cells.length >= K) return cells;
  }
  return null;
}
function play(col) {
  const s = spec();
  if (state.winner || state.h[col] >= s.rows) return false;
  const cell = state.h[col] * s.cols + col; state.g[cell] = state.player; state.h[col]++; state.moves.push(col);
  const line = lineAt(cell);
  if (line) { state.winner = state.player; state.winLine = line; }
  else if (state.moves.length >= s.rows * s.cols) state.winner = -1;
  state.player = 3 - state.player;
  return true;
}
function build() {
  const s = spec(), board = $("board"), cols = $("cols");
  board.style.gridTemplateColumns = cols.style.gridTemplateColumns = `repeat(${s.cols}, 1fr)`;
  board.innerHTML = ""; cols.innerHTML = "";
  for (let r = s.rows - 1; r >= 0; r--) for (let c = 0; c < s.cols; c++) {
    const d = document.createElement("div"); d.className = "cell"; d.id = `c${r}_${c}`; d.dataset.col = c; board.appendChild(d);
  }
  for (let c = 0; c < s.cols; c++) { const t = document.createElement("span"); t.textContent = c; cols.appendChild(t); }
  $("rule").textContent = `${s.spec.replace("connect", "Connect ")} · ${s.cols}×${s.rows} board · ${s.k} in a row wins · ${s.label}`;
}
function render(animateLast) {
  const s = spec(), last = state.moves.length ? state.moves[state.moves.length - 1] : -1;
  const lastCell = last >= 0 ? (state.h[last] - 1) * s.cols + last : -1, win = new Set(state.winLine);
  for (let r = 0; r < s.rows; r++) for (let c = 0; c < s.cols; c++) {
    const i = r * s.cols + c, el = $(`c${r}_${c}`), p = state.g[i];
    el.className = "cell" + (i === lastCell ? " last" : "") + (win.has(i) ? " win" : "");
    if (!p) { el.innerHTML = ""; continue; }
    if (el.firstChild && el.firstChild.dataset.p === String(p) && !(animateLast && i === lastCell)) continue;
    const st = document.createElement("div"); st.className = "stone p" + p; st.dataset.p = p;
    if (animateLast && i === lastCell) { st.classList.add("drop"); st.style.setProperty("--fall", s.rows - r); }
    el.innerHTML = ""; el.appendChild(st);
  }
  $("board").classList.toggle("mine", ready && !state.winner && state.player === state.human && !state.busy);
  const st = $("status"); st.className = "status";
  if (!ready) st.textContent = "Loading the network…";
  else if (state.winner === -1) st.innerHTML = "Draw.<small>The board is full.</small>";
  else if (state.winner === state.human) { st.className = "status won"; st.innerHTML = "You win!<small>Press New game to play again, or try a harder opponent.</small>"; }
  else if (state.winner) { st.className = "status lost"; st.innerHTML = "The net wins.<small>Press Undo to take back your last move, or New game.</small>"; }
  else if (state.busy) st.innerHTML = '<span class="thinking">The net is thinking</span>';
  else st.innerHTML = `Your move.<small>${state.moves.length ? "Click a column to drop a stone." : "You start: click a column."}</small>`;
  $("undo").disabled = state.busy || state.moves.length === 0;
  if (state.lastEval === null) { $("evalText").textContent = "–"; $("evalBar").style.width = "50%"; }
  else { const mine = -state.lastEval, pct = Math.round((mine + 1) * 50); $("evalBar").style.width = pct + "%";
    $("evalText").textContent = mine > 0.4 ? "you are ahead" : mine < -0.4 ? "the net is ahead" : "about even"; }
}
function maybeNet() {
  if (!ready || state.winner || state.player === state.human) return;
  state.busy = true; render();
  const level = LEVELS.find((l) => l.id === state.level), id = ++req, sims = playSims(level);
  // a search stops after about 6 s (slow machine, software WebGL), with at least 16 simulations
  const minSims = Math.min(sims, spec().play_sims_cpu ?? 16), budgetMs = 6000;
  if (gpu) {
    const net = gpu[state.net], g = state.g.slice(), player = state.player;
    setTimeout(async () => { const r = await LIB.chooseMove(net.evalFn, g, player, net.spec, sims, budgetMs, minSims); onMove({...r, req: id}); }, 30);
  } else {
    setTimeout(() => worker.postMessage({type: "move", id: state.net, grid: Array.from(state.g), player: state.player, sims, budgetMs, minSims, req: id}), 30);
  }
}
function onMove(m) {
  if (m.req !== req) return;
  state.busy = false; state.lastEval = m.value; play(m.move); render(true);
}
// ---- the GPU path: TensorFlow.js with WebGL, on the page's own thread
function tfWeights(t, sh) {
  const W = {};
  for (const k in t) {
    const s = sh[k];
    if (s.length === 4) W[k] = tf.tensor4d(t[k], s).transpose([2, 3, 1, 0]);      // OIHW -> HWIO
    else if (s.length === 2) W[k] = tf.tensor2d(t[k], s).transpose();            // [out, in] -> [in, out]
    else W[k] = tf.tensor1d(t[k]);
  }
  return W;
}
function tfForward(W, x) {
  return tf.tidy(() => {
    const conv = (a, k) => tf.conv2d(a, W[k + ".weight"], 1, "same").add(W[k + ".bias"]);
    const flat = (h) => h.transpose([0, 3, 1, 2]).reshape([h.shape[0], -1]);     // the order of torch's flatten
    let a;
    if (W["stem.weight"]) {
      a = conv(x, "stem").relu();
      for (let b = 0; W[`blocks.${b}.c1.weight`]; b++) a = a.add(conv(conv(a, `blocks.${b}.c1`).relu(), `blocks.${b}.c2`)).relu();
    } else {
      a = conv(x, "body.0").relu();
      for (let L = 2; W[`body.${L}.weight`]; L += 2) a = conv(a, `body.${L}`).relu();
    }
    const logits = flat(conv(a, "policy_conv").relu()).matMul(W["policy_fc.weight"]).add(W["policy_fc.bias"]);
    const v1 = flat(conv(a, "value_conv").relu()).matMul(W["value_fc1.weight"]).add(W["value_fc1.bias"]).relu();
    return tf.concat([logits, v1.matMul(W["value_fc2.weight"]).add(W["value_fc2.bias"]).tanh()], 1);
  });
}
function tfEvaluator(W, s) {
  return async (g, player) => {
    const R = s.R, C = s.C, h = LIB.heightsOf(g, s), x = new Float32Array(R * C * 4);
    for (let i = 0; i < R * C; i++) { if (g[i] === player) x[i * 4] = 1; else if (g[i] === 3 - player) x[i * 4 + 1] = 1; x[i * 4 + 3] = 1; }
    for (let c = 0; c < C; c++) if (h[c] < R) x[(h[c] * C + c) * 4 + 2] = 1;
    const xin = tf.tensor4d(x, [1, R, C, 4]), out = tfForward(W, xin), d = await out.data();
    xin.dispose(); out.dispose();
    return {logits: d.subarray(0, C), value: d[C], h};
  };
}
function decoded(n) {
  const t = {}, sh = {};
  for (const k in n.tensors) { const buf = b64ToBuf(n.tensors[k].data); sh[k] = n.tensors[k].shape;
    t[k] = n.tensors[k].dtype === "f16" ? LIB.halfToFloat(new Uint16Array(buf)) : new Float32Array(buf); }
  return {t, sh};
}
async function startGpu() {
  if (!USE_GPU || !window.tf) return null;
  try {
    await tf.setBackend("webgl"); await tf.ready();
    if (tf.getBackend() !== "webgl") return null;
    const nets = NETS.map((n) => { const d = decoded(n), s = LIB.specOf(n); return {spec: s, evalFn: tfEvaluator(tfWeights(d.t, d.sh), s)}; });
    for (const n of nets) await n.evalFn(new Int8Array(n.spec.R * n.spec.C), 1);    // compile the shaders now, not on the first move
    return nets;
  } catch (e) { return null; }
}
function startWorker() {
  worker = new Worker(URL.createObjectURL(new Blob([WORKER_SRC], {type: "text/javascript"})));
  worker.onerror = (e) => { $("status").textContent = "The network could not start: " + (e.message || "unknown error"); };
  worker.onmessage = (e) => { if (e.data.type === "ready") { ready = true; newGame(); } else onMove(e.data); };
  worker.postMessage({type: "load", nets: NETS.map((n, i) => ({id: i, rows: n.rows, cols: n.cols, k: n.k,
    shapes: Object.fromEntries(Object.entries(n.tensors).map(([k, t]) => [k, t.shape])),
    dtypes: Object.fromEntries(Object.entries(n.tensors).map(([k, t]) => [k, t.dtype || "f32"])),
    tensors: Object.fromEntries(Object.entries(n.tensors).map(([k, t]) => [k, b64ToBuf(t.data)]))}))});
}
$("board").addEventListener("click", (e) => {
  const cell = e.target.closest(".cell"); if (!cell || state.busy || state.winner || state.player !== state.human) return;
  if (play(+cell.dataset.col)) { render(true); maybeNet(); }
});
$("board").addEventListener("mouseover", (e) => {
  const cell = e.target.closest(".cell"); document.querySelectorAll(".cell.colhover").forEach((x) => x.classList.remove("colhover"));
  if (!cell) return; document.querySelectorAll(`.cell[data-col="${cell.dataset.col}"]`).forEach((x) => x.classList.add("colhover"));
});
$("board").addEventListener("mouseleave", () => document.querySelectorAll(".cell.colhover").forEach((x) => x.classList.remove("colhover")));
$("newgame").onclick = newGame;
$("undo").onclick = () => {
  if (state.busy || !state.moves.length) return;
  const keep = state.moves.slice(); let removedHuman = false;
  while (keep.length) {
    const mover = keep.length % 2 === 1 ? 1 : 2;        // who made the last move in keep
    keep.pop();
    if (mover === state.human) removedHuman = true;
    if (removedHuman && (keep.length % 2 === 0 ? 1 : 2) === state.human) break;
  }
  resetState();
  for (const c of keep) play(c);
  render(); maybeNet();
};
NETS.forEach((n, i) => {
  const l = document.createElement("label");
  l.innerHTML = `<input type="radio" name="game" value="${i}" ${i === 0 ? "checked" : ""}><span>${n.short || `${n.spec.replace("connect", "Connect ")} (${n.cols}×${n.rows})`}</span>`;
  l.querySelector("input").addEventListener("change", () => { state.net = i; showHint(); newGame(); });
  $("games").appendChild(l);
});
LEVELS.forEach((lv) => {
  const l = document.createElement("label");
  l.innerHTML = `<input type="radio" name="level" value="${lv.id}" ${lv.id === state.level ? "checked" : ""}><span>${lv.label}</span>`;
  l.querySelector("input").addEventListener("change", () => { state.level = lv.id; showHint(); maybeNet(); });
  $("levels").appendChild(l);
});
showHint();
for (const id of ["sideFirst", "sideSecond"]) $(id).addEventListener("change", (e) => { state.human = +e.target.value; newGame(); });
resetState(); build(); render();
startGpu().then((nets) => { gpu = nets; if (gpu) { ready = true; newGame(); } else startWorker(); });
</script>"""

_TEMPLATE = _TEMPLATE.replace("__WORKER__", json.dumps(_WORKER)).replace("__LIB__", json.dumps(_LIB))
