"""Replay viewer: write games to one self-contained HTML page.

    from connectk.viewer import record_game, write_html
    rec = record_game(CONNECT6, first_player, second_player, opening=(6, 7))
    write_html([rec], "replay.html")        # open it in a browser

A record is a plain dict, so you can also build one from your own data:

    {"title": str, "spec": "connect6", "rows": 13, "cols": 13, "k": 6,
     "players": [name_of_player_1, name_of_player_2],
     "moves": [[player, row, col, kind], ...],   # kind: "drop", or a rule break
     "opening": number of opening moves, "winner": 0/1/2, "reason": str,
     "win_line": [[row, col], ...]}
"""

import json

from .arena import play_game
from .game import DIRECTIONS, Board


def win_line(board):
    """Cells of one winning line on the board, or []."""
    g, k = board.grid, board.spec.k
    R, C = board.spec.rows, board.spec.cols
    for r in range(R):
        for c in range(C):
            p = g[r, c]
            if not p:
                continue
            for dr, dc in DIRECTIONS:
                cells = [(r + dr * i, c + dc * i) for i in range(k)]
                if all(0 <= rr < R and 0 <= cc < C and g[rr, cc] == p for rr, cc in cells):
                    return [list(x) for x in cells]
    return []


def record_from_columns(spec, columns, players, opening=0, title=None, reason=None):
    """Record of a legal game given as a list of columns."""
    b = Board(spec)
    for c in columns:
        b.play(c)
    return {
        "title": title or f"{players[0]} vs {players[1]}",
        "spec": spec.name, "rows": spec.rows, "cols": spec.cols, "k": spec.k,
        "players": list(players),
        "moves": [[p, r, c, "drop"] for p, r, c in b.history],
        "opening": opening, "winner": int(b.winner),
        "reason": reason or ("line" if b.winner else "draw" if b.done else "unfinished"),
        "win_line": win_line(b),
    }


def record_game(spec, first, second, opening=(), title=None):
    """Play first vs second with the strict referee and return the record."""
    names = [getattr(first, "name", "player 1"), getattr(second, "name", "player 2")]
    res = play_game(first, second, spec, opening)
    rec = record_from_columns(spec, res.moves, names, len(opening), title, res.reason)
    rec["winner"] = res.winner
    return rec


def render_html(records, title="Connect replays", standalone=True):
    data = json.dumps(records, separators=(",", ":"))
    page = _TEMPLATE.replace("__TITLE__", title).replace("__DATA__", data)
    if standalone:
        page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                + page.replace("</style>", "</style>\n</head>\n<body>", 1) + "\n</body>\n</html>\n")
    return page


def write_html(records, path, title="Connect replays", standalone=True):
    with open(path, "w") as f:
        f.write(render_html(records, title, standalone))
    return path


_TEMPLATE = r"""<title>__TITLE__</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=Atkinson+Hyperlegible:wght@400;700&family=IBM+Plex+Mono:wght@400;600&display=swap">
<style>
/* Layout: the board is the page; a narrow side rail holds players, controls and the move list. */
:root {
  --paper: #eef1f6; --panel: #f8f9fc; --ink: #142033; --muted: #5b6880; --rule: #d3dae6;
  --board: #1f4e9c; --board-edge: #163b78; --hole: #dfe6f1;
  --p1: #e04848; --p2: #f2b632; --warn: #7c3aed; --glow: #ffffff;
  --display: "Bricolage Grotesque", "Avenir Next", "Segoe UI", sans-serif;
  --body: "Atkinson Hyperlegible", "Segoe UI", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --paper: #0d1420; --panel: #131c2b; --ink: #e6ecf5; --muted: #93a1b8; --rule: #253247;
  --board: #1b3f7d; --board-edge: #12305f; --hole: #0a1018;
  --p1: #f05a5a; --p2: #f5c044; --warn: #a78bfa; --glow: #ffffff; color-scheme: dark; } }
:root[data-theme="dark"] {
  --paper: #0d1420; --panel: #131c2b; --ink: #e6ecf5; --muted: #93a1b8; --rule: #253247;
  --board: #1b3f7d; --board-edge: #12305f; --hole: #0a1018;
  --p1: #f05a5a; --p2: #f5c044; --warn: #a78bfa; --glow: #ffffff; color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--paper); color: var(--ink); font: 15px/1.5 var(--body);
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  padding-block: 20px 32px; }
.wrap { max-width: 1060px; margin: 0 auto; display: grid; gap: 18px; }
header { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 8px 16px; }
h1 { font: 800 clamp(24px, 4vw, 34px)/1.1 var(--display); margin: 0; letter-spacing: -0.01em; text-wrap: balance; }
.rule-line { color: var(--muted); font-size: 14px; }
.games { display: flex; flex-wrap: wrap; gap: 8px; }
.games button { font: 600 13px/1.2 var(--body); border: 1px solid var(--rule); background: var(--panel); color: var(--ink);
  border-radius: 999px; padding: 7px 12px; cursor: pointer; }
.games button[aria-pressed="true"] { background: var(--board); border-color: var(--board); color: #fff; }
.main { display: grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 20px; align-items: start; }
@media (max-width: 760px) { .main { grid-template-columns: minmax(0, 1fr); } }
.board-wrap { min-width: 0; }
.board { --cols: 13; --rows: 13; display: grid; grid-template-columns: repeat(var(--cols), 1fr); gap: 3.5%;
  gap: clamp(2px, 0.6vw, 6px); padding: clamp(6px, 1.4vw, 14px); background: var(--board);
  border-radius: 14px; box-shadow: inset 0 -6px 0 var(--board-edge); width: 100%; max-width: 640px; margin: 0 auto; }
.cell { position: relative; aspect-ratio: 1; border-radius: 50%; background: var(--hole); overflow: hidden; }
.stone { position: absolute; inset: 0; border-radius: 50%; }
.stone.p1 { background: radial-gradient(circle at 35% 30%, color-mix(in srgb, var(--p1) 70%, #fff), var(--p1) 60%); }
.stone.p2 { background: radial-gradient(circle at 35% 30%, color-mix(in srgb, var(--p2) 70%, #fff), var(--p2) 60%); }
.stone.drop { animation: fall 260ms cubic-bezier(.45,0,.9,.6); }
@keyframes fall { from { transform: translateY(calc(var(--fall) * -112%)); } to { transform: none; } }
@media (prefers-reduced-motion: reduce) { .stone.drop { animation: none; } }
.cell.last .stone { box-shadow: inset 0 0 0 3px color-mix(in srgb, var(--ink) 55%, transparent); }
.cell.win .stone { box-shadow: inset 0 0 0 3px var(--glow), 0 0 0 2px var(--glow); }
.cell.cheat::after { content: ""; position: absolute; inset: 6%; border: 3px dashed var(--warn); border-radius: 50%;
  background: color-mix(in srgb, var(--warn) 22%, transparent); }
.cols { display: grid; grid-template-columns: repeat(var(--cols), 1fr); gap: clamp(2px, 0.6vw, 6px);
  padding: 4px clamp(6px, 1.4vw, 14px) 0; max-width: 640px; margin: 0 auto;
  font: 600 11px/1 var(--mono); color: var(--muted); text-align: center; font-variant-numeric: tabular-nums; }
.cols span.hit { color: var(--ink); }
aside { background: var(--panel); border: 1px solid var(--rule); border-radius: 12px; padding: 14px; display: grid; gap: 12px; min-width: 0; }
.players { display: grid; gap: 6px; }
.player { display: flex; align-items: center; gap: 10px; padding: 6px 8px; border-radius: 8px; }
.player.turn { background: color-mix(in srgb, var(--board) 12%, transparent); }
.dot { width: 18px; height: 18px; border-radius: 50%; flex: none; }
.dot.p1 { background: var(--p1); } .dot.p2 { background: var(--p2); }
.pname { font-weight: 700; overflow-wrap: anywhere; }
.side { margin-left: auto; font: 12px var(--mono); color: var(--muted); }
.status { font: 600 14px/1.4 var(--body); min-height: 2.8em; }
.status .warn { color: var(--warn); }
.controls { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
.controls button { font: 600 14px/1 var(--mono); min-width: 40px; height: 36px; border-radius: 8px; border: 1px solid var(--rule);
  background: var(--paper); color: var(--ink); cursor: pointer; }
.controls button.primary { background: var(--board); color: #fff; border-color: var(--board); min-width: 64px; }
button:focus-visible, input:focus-visible, select:focus-visible { outline: 3px solid var(--p2); outline-offset: 2px; }
.scrub { display: grid; gap: 4px; }
.scrub input { width: 100%; accent-color: var(--board); }
.meta { display: flex; justify-content: space-between; font: 12px var(--mono); color: var(--muted); font-variant-numeric: tabular-nums; }
.speed { display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--muted); }
.speed select { font: 13px var(--body); background: var(--paper); color: var(--ink); border: 1px solid var(--rule); border-radius: 6px; padding: 3px 6px; }
.moves { max-height: 240px; overflow-y: auto; border-top: 1px solid var(--rule); padding-top: 8px;
  display: grid; grid-template-columns: repeat(auto-fill, minmax(84px, 1fr)); gap: 2px 6px;
  font: 12px/1.6 var(--mono); font-variant-numeric: tabular-nums; }
.moves button { all: unset; cursor: pointer; padding: 0 4px; border-radius: 4px; white-space: nowrap; color: var(--ink); }
.moves button.op { color: var(--muted); }
.moves button.bad { color: var(--warn); font-weight: 600; }
.moves button.now { background: color-mix(in srgb, var(--board) 22%, transparent); }
.moves button:focus-visible { outline: 2px solid var(--p2); }
.mini { width: 9px; height: 9px; border-radius: 50%; display: inline-block; margin-right: 4px; vertical-align: 0; }
.mini.p1 { background: var(--p1); } .mini.p2 { background: var(--p2); }
.legend { font-size: 12px; color: var(--muted); }
.legend b { color: var(--warn); }
</style>
<div class="wrap">
  <header>
    <h1 id="title">__TITLE__</h1>
    <div class="rule-line" id="ruleLine"></div>
  </header>
  <nav class="games" id="games" aria-label="Games"></nav>
  <div class="main">
    <div class="board-wrap">
      <div class="board" id="board" role="img" aria-label="Game board"></div>
      <div class="cols" id="cols" aria-hidden="true"></div>
    </div>
    <aside>
      <div class="players" id="players"></div>
      <div class="status" id="status" aria-live="polite"></div>
      <div class="controls">
        <button id="first" title="Start (Home)" aria-label="Go to start">|&lt;</button>
        <button id="prev" title="Back one move (Left arrow)" aria-label="Back one move">&lt;</button>
        <button id="play" class="primary" title="Play or pause (Space)">Play</button>
        <button id="next" title="Forward one move (Right arrow)" aria-label="Forward one move">&gt;</button>
        <button id="last" title="End (End)" aria-label="Go to end">&gt;|</button>
      </div>
      <div class="scrub">
        <input type="range" id="scrub" min="0" max="0" value="0" aria-label="Move number">
        <div class="meta"><span id="moveNo">move 0 / 0</span><span id="resultShort"></span></div>
      </div>
      <label class="speed" for="speed">Speed
        <select id="speed">
          <option value="900">slow</option>
          <option value="420" selected>normal</option>
          <option value="160">fast</option>
          <option value="40">very fast</option>
        </select>
      </label>
      <div class="moves" id="moves"></div>
      <div class="legend" id="legend"></div>
    </aside>
  </div>
</div>
<script>
const GAMES = __DATA__;
const KIND = { drop: "", teleport: "mid-air stone", steal: "took an opponent stone", extra: "second stone in one turn" };
let g = 0, i = 0, timer = null;
const $ = (id) => document.getElementById(id);

function game() { return GAMES[g]; }
function gridAt(n) {
  const G = game(), grid = Array.from({ length: G.rows }, () => Array(G.cols).fill(0));
  const bad = new Set();
  for (let t = 0; t < n; t++) {
    const [p, r, c, kind] = G.moves[t];
    grid[r][c] = p;
    if (kind && kind !== "drop") bad.add(r + "," + c); else bad.delete(r + "," + c);
  }
  return { grid, bad };
}
function buildBoard() {
  const G = game(), board = $("board"), cols = $("cols");
  board.style.setProperty("--cols", G.cols); cols.style.setProperty("--cols", G.cols);
  board.innerHTML = ""; cols.innerHTML = "";
  for (let r = G.rows - 1; r >= 0; r--) for (let c = 0; c < G.cols; c++) {
    const d = document.createElement("div"); d.className = "cell"; d.id = `c${r}_${c}`; board.appendChild(d);
  }
  for (let c = 0; c < G.cols; c++) { const s = document.createElement("span"); s.textContent = c; s.id = "col" + c; cols.appendChild(s); }
  $("ruleLine").textContent = `${G.spec.replace("connect", "Connect ")} · ${G.cols}×${G.rows} board · ${G.k} in a row wins`;
  $("scrub").max = G.moves.length;
  const list = $("moves"); list.innerHTML = "";
  G.moves.forEach(([p, r, c, kind], t) => {
    const b = document.createElement("button");
    b.innerHTML = `<span class="mini p${p}"></span>${t + 1}. c${c}` + (kind !== "drop" ? " !" : "");
    b.title = `${G.players[p - 1]}: column ${c}, row ${r}` + (kind !== "drop" ? ` (${KIND[kind] || kind})` : "") + (t < G.opening ? " (opening)" : "");
    if (t < G.opening) b.classList.add("op");
    if (kind !== "drop") b.classList.add("bad");
    b.addEventListener("click", () => { stop(); show(t + 1); });
    list.appendChild(b);
  });
  const anyBad = G.moves.some(m => m[3] !== "drop");
  $("legend").innerHTML = (G.opening ? `Grey moves are the random opening. ` : "") +
    (anyBad ? `<b>Dashed cells</b> are rule breaks: stones a fair game cannot make.` : "");
  const nav = $("games");
  [...nav.children].forEach((b, k) => b.setAttribute("aria-pressed", String(k === g)));
}
function resultText(G) {
  const who = G.winner ? G.players[G.winner - 1] : null;
  if (G.reason === "draw") return "Draw: the board is full.";
  if (G.reason && G.reason !== "line" && who) return `${who} wins: the opponent made an illegal move.`;
  return who ? `${who} wins with ${G.k} in a row.` : "Unfinished.";
}
function show(n, animate) {
  const G = game();
  i = Math.max(0, Math.min(n, G.moves.length));
  const { grid, bad } = gridAt(i);
  const end = i === G.moves.length;
  const winSet = new Set(end ? G.win_line.map(([r, c]) => r + "," + c) : []);
  const last = i > 0 ? G.moves[i - 1] : null;
  for (let r = 0; r < G.rows; r++) for (let c = 0; c < G.cols; c++) {
    const cell = $(`c${r}_${c}`), p = grid[r][c], key = r + "," + c;
    cell.className = "cell" + (bad.has(key) ? " cheat" : "") + (winSet.has(key) ? " win" : "") +
      (last && last[1] === r && last[2] === c ? " last" : "");
    if (!p) { cell.innerHTML = ""; continue; }
    if (cell.firstChild && cell.firstChild.dataset.p === String(p) && !(animate && last && last[1] === r && last[2] === c)) continue;
    const s = document.createElement("div"); s.className = "stone p" + p; s.dataset.p = p;
    if (animate && last && last[1] === r && last[2] === c && last[3] === "drop") { s.classList.add("drop"); s.style.setProperty("--fall", G.rows - r); }
    cell.innerHTML = ""; cell.appendChild(s);
  }
  for (let c = 0; c < G.cols; c++) $("col" + c).className = last && last[2] === c ? "hit" : "";
  const toMove = end ? 0 : (i < G.moves.length ? G.moves[i][0] : 0);
  $("players").innerHTML = G.players.map((name, k) =>
    `<div class="player${toMove === k + 1 ? " turn" : ""}"><span class="dot p${k + 1}"></span><span class="pname">${name}</span><span class="side">${k ? "second" : "first"}</span></div>`).join("");
  let st;
  if (end) st = resultText(G);
  else if (last) {
    st = `${G.players[last[0] - 1]} plays column ${last[2]}.`;
    if (last[3] !== "drop") st = `<span class="warn">${G.players[last[0] - 1]}: ${KIND[last[3]] || last[3]} at column ${last[2]}, row ${last[1]}.</span>`;
    else if (i <= G.opening) st += " (opening)";
  } else st = G.opening ? `The first ${G.opening} moves are a random opening. Press Play.` : "Empty board. Press Play.";
  $("status").innerHTML = st;
  $("scrub").value = i;
  $("moveNo").textContent = `move ${i} / ${G.moves.length}`;
  $("resultShort").textContent = G.winner ? `winner: ${G.players[G.winner - 1]}` : (G.reason === "draw" ? "draw" : "");
  const items = $("moves").children;
  for (let t = 0; t < items.length; t++) items[t].classList.toggle("now", t === i - 1);
  if (i > 0 && items[i - 1]) items[i - 1].scrollIntoView({ block: "nearest" });
}
function stop() { clearInterval(timer); timer = null; $("play").textContent = "Play"; }
function play() {
  if (timer) return stop();
  if (i >= game().moves.length) show(0);
  $("play").textContent = "Pause";
  timer = setInterval(() => { if (i >= game().moves.length) return stop(); show(i + 1, true); }, +$("speed").value);
}
function pick(k) { stop(); g = k; buildBoard(); show(0); }
GAMES.forEach((G, k) => {
  const b = document.createElement("button"); b.textContent = G.title; b.addEventListener("click", () => pick(k)); $("games").appendChild(b);
});
$("first").onclick = () => { stop(); show(0); };
$("last").onclick = () => { stop(); show(game().moves.length); };
$("prev").onclick = () => { stop(); show(i - 1); };
$("next").onclick = () => { stop(); show(i + 1, true); };
$("play").onclick = play;
$("speed").onchange = () => { if (timer) { stop(); play(); } };
$("scrub").oninput = (e) => { stop(); show(+e.target.value); };
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
  if (e.key === "ArrowRight") { stop(); show(i + 1, true); }
  else if (e.key === "ArrowLeft") { stop(); show(i - 1); }
  else if (e.key === " ") { e.preventDefault(); play(); }
  else if (e.key === "Home") { stop(); show(0); }
  else if (e.key === "End") { stop(); show(game().moves.length); }
});
pick(0);
show(game().moves.length);
</script>"""
