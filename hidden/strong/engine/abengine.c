/* Alpha-beta engine for connect-k with gravity (problem writers' teacher).
 *
 * Negamax with alpha-beta, a transposition table (Zobrist keys), iterative
 * deepening under a node budget, and an evaluation kept up to date move by
 * move: every line of k cells (a "window") holds its stone counts, and the
 * score is the sum over windows of A[n] for n of my stones in a window with
 * none of the opponent's, minus the same for the opponent.
 *
 * Build: gcc -O3 -shared -fPIC -o abengine.so abengine.c
 * Python wrapper: hidden/strong/abengine.py
 */
#include <stdint.h>
#include <string.h>

#define MAXR 13
#define MAXC 13
#define MAXCELLS (MAXR * MAXC)
#define MAXK 6
#define MAXW 800
#define MAXCW 32
#define WIN 100000000
#define INF 2000000000
#define TTBITS 20
#define TTSIZE (1u << TTBITS)

static int R, C, K, NCELLS, NW;
static int win_cells[MAXW][MAXK];
static int cell_wins[MAXCELLS][MAXCW];
static int cell_nw[MAXCELLS];
static int cnt[MAXW][3];
static signed char cell[MAXCELLS];
static int height[MAXC];
static int order[MAXC];
static long long A[MAXK + 1];
static long long evalv;            /* from player 1's view */
static uint64_t zob[MAXCELLS][3];
static uint64_t hash;
static long nodes, node_limit;
static int aborted, stones;

typedef struct { uint64_t key; int score; short depth; signed char flag; signed char best; } TTEntry;
static TTEntry tt[TTSIZE];
static int history[3][MAXC];

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t rnd(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static long long wval(int c1, int c2) {
    if (c2 == 0) return A[c1];
    if (c1 == 0) return -A[c2];
    return 0;
}

static void setup(int rows, int cols, int k) {
    static int done_r = -1, done_c = -1, done_k = -1;
    if (rows == done_r && cols == done_c && k == done_k) return;
    R = rows; C = cols; K = k; NCELLS = R * C; NW = 0;
    memset(cell_nw, 0, sizeof(cell_nw));
    static const int dr[4] = {0, 1, 1, 1}, dc[4] = {1, 0, 1, -1};
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++)
            for (int d = 0; d < 4; d++) {
                int er = r + dr[d] * (K - 1), ec = c + dc[d] * (K - 1);
                if (er < 0 || er >= R || ec < 0 || ec >= C) continue;
                for (int i = 0; i < K; i++) {
                    int idx = (r + dr[d] * i) * C + c + dc[d] * i;
                    win_cells[NW][i] = idx;
                    cell_wins[idx][cell_nw[idx]++] = NW;
                }
                NW++;
            }
    long long v = 1;
    A[0] = 0;
    for (int i = 1; i <= K; i++) { A[i] = v; v *= 8; }
    for (int i = 0; i < NCELLS; i++) for (int p = 0; p < 3; p++) zob[i][p] = rnd();
    /* center-out column order */
    int n = 0;
    for (int d = 0; d < C; d++) {
        int a = (C - 1) / 2 + ((d % 2) ? (d + 1) / 2 : -(d / 2));
        if (a >= 0 && a < C) order[n++] = a;
    }
    memset(tt, 0, sizeof(tt));
    done_r = rows; done_c = cols; done_k = k;
}

/* Drop a stone of player p in column col. Returns 1 if it makes k in a row. */
static int make(int col, int p) {
    int idx = height[col] * C + col, won = 0;
    cell[idx] = (signed char)p;
    height[col]++;
    stones++;
    hash ^= zob[idx][p];
    for (int j = 0; j < cell_nw[idx]; j++) {
        int w = cell_wins[idx][j];
        evalv -= wval(cnt[w][1], cnt[w][2]);
        cnt[w][p]++;
        evalv += wval(cnt[w][1], cnt[w][2]);
        if (cnt[w][p] == K) won = 1;
    }
    return won;
}

static void unmake(int col, int p) {
    height[col]--;
    stones--;
    int idx = height[col] * C + col;
    cell[idx] = 0;
    hash ^= zob[idx][p];
    for (int j = 0; j < cell_nw[idx]; j++) {
        int w = cell_wins[idx][j];
        evalv -= wval(cnt[w][1], cnt[w][2]);
        cnt[w][p]--;
        evalv += wval(cnt[w][1], cnt[w][2]);
    }
}

static int wins_now(int col, int p) {
    int idx = height[col] * C + col;
    for (int j = 0; j < cell_nw[idx]; j++) {
        int w = cell_wins[idx][j];
        if (cnt[w][p] == K - 1 && cnt[w][3 - p] == 0) return 1;
    }
    return 0;
}

static int negamax(int depth, int alpha, int beta, int p, int ply) {
    if (++nodes > node_limit) { aborted = 1; return 0; }
    if (stones >= NCELLS) return 0;
    int any = 0;
    for (int c = 0; c < C; c++) if (height[c] < R) { any = 1; if (wins_now(c, p)) return WIN - ply; }
    if (!any) return 0;
    /* forced block: if the opponent wins at two places, I lose; at one, I must play there */
    int forced = -1, threats = 0;
    for (int c = 0; c < C; c++) if (height[c] < R && wins_now(c, 3 - p)) { threats++; forced = c; }
    if (threats >= 2) return -(WIN - ply - 1);
    if (depth <= 0 && threats == 0) return (int)(p == 1 ? evalv : -evalv);
    if (depth <= 0) depth = 1;     /* extend one ply to answer a threat */

    int alpha0 = alpha, ttbest = -1;
    TTEntry *e = &tt[hash & (TTSIZE - 1)];
    if (e->key == hash) {
        ttbest = e->best;
        if (e->depth >= depth) {
            if (e->flag == 0) return e->score;
            if (e->flag == 1 && e->score > alpha) alpha = e->score;
            if (e->flag == 2 && e->score < beta) beta = e->score;
            if (alpha >= beta) return e->score;
        }
    }
    int moves[MAXC], n = 0;
    if (forced >= 0) moves[n++] = forced;
    else {
        if (ttbest >= 0 && height[ttbest] < R) moves[n++] = ttbest;
        int rest[MAXC], m = 0;
        for (int i = 0; i < C; i++) { int c = order[i]; if (height[c] < R && c != ttbest) rest[m++] = c; }
        for (int i = 1; i < m; i++) {          /* history heuristic, stable */
            int x = rest[i], j = i - 1;
            while (j >= 0 && history[p][rest[j]] < history[p][x]) { rest[j + 1] = rest[j]; j--; }
            rest[j + 1] = x;
        }
        for (int i = 0; i < m; i++) moves[n++] = rest[i];
    }
    int best = -INF, bestc = moves[0];
    for (int i = 0; i < n; i++) {
        int c = moves[i];
        make(c, p);
        int s = -negamax(depth - 1, -beta, -alpha, 3 - p, ply + 1);
        unmake(c, p);
        if (aborted) return 0;
        if (s > best) { best = s; bestc = c; }
        if (s > alpha) alpha = s;
        if (alpha >= beta) { history[p][c] += depth * depth; break; }
    }
    e->key = hash;
    e->score = best;
    e->depth = (short)depth;
    e->best = (signed char)bestc;
    e->flag = best <= alpha0 ? 2 : (best >= beta ? 1 : 0);
    return best;
}

/* Search a position. grid: rows*cols, row 0 at the bottom, 0 empty / 1 / 2.
 * Fills out_scores[col] (INT_MIN-ish for full columns) from the last depth
 * finished within the node budget, and *out_depth. Returns the best column. */
int ab_search(int rows, int cols, int k, const signed char *grid, int to_move,
              int max_depth, long budget, int *out_scores, int *out_depth) {
    setup(rows, cols, k);
    memset(cnt, 0, sizeof(cnt));
    memset(height, 0, sizeof(height));
    memset(history, 0, sizeof(history));
    evalv = 0; hash = 0; stones = 0;
    for (int i = 0; i < NCELLS; i++) cell[i] = 0;
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++) {
            int p = grid[r * C + c];
            if (p) { height[c] = r + 1; }
        }
    /* replay stones column by column to fill the counts (order does not matter) */
    int h2[MAXC]; memcpy(h2, height, sizeof(h2)); memset(height, 0, sizeof(height));
    for (int c = 0; c < C; c++) for (int r = 0; r < h2[c]; r++) make(c, grid[r * C + c] ? grid[r * C + c] : 1);
    for (int c = 0; c < C; c++) out_scores[c] = -INF;
    *out_depth = 0;
    nodes = 0; node_limit = budget; aborted = 0;
    int scores[MAXC], best = -1;
    for (int d = 1; d <= max_depth; d++) {
        int ok = 1;
        for (int i = 0; i < C; i++) {
            int c = order[i];
            if (height[c] >= R) { scores[c] = -INF; continue; }
            int won = make(c, to_move);
            int s = won ? WIN : -negamax(d - 1, -INF + 1, INF - 1, 3 - to_move, 1);
            unmake(c, to_move);
            if (aborted) { ok = 0; break; }
            scores[c] = s;
        }
        if (!ok) break;
        memcpy(out_scores, scores, sizeof(int) * C);
        *out_depth = d;
        int top = -INF;
        for (int i = 0; i < C; i++) { int c = order[i]; if (scores[c] > top) { top = scores[c]; best = c; } }
        if (top >= WIN - 1000 || top <= -(WIN - 1000)) {
            /* a forced result is known; deeper search will not change the best move class */
            int all_lost = 1;
            for (int c = 0; c < C; c++) if (scores[c] > -(WIN - 1000) && scores[c] != -INF) all_lost = 0;
            if (top >= WIN - 1000 || all_lost) break;
        }
    }
    if (best < 0) {
        int top = -INF;
        for (int i = 0; i < C; i++) { int c = order[i]; if (height[c] < R && out_scores[c] > top) { top = out_scores[c]; best = c; } }
        if (best < 0) for (int i = 0; i < C; i++) if (height[order[i]] < R) { best = order[i]; break; }
    }
    return best;
}

long ab_nodes(void) { return nodes; }
