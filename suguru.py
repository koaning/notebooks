# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
## Suguru (a.k.a. Tectonic): generate + solve

Suguru is a grid puzzle with **irregular regions** ("rooms"). The rules:

- **Room rule**: if a room has size \(k\), it must contain the numbers \(1..k\), each exactly once.
- **Adjacency rule**: equal numbers may **not** touch — not even diagonally (8-neighborhood).

This notebook shows a minimal, readable path from:

1. **Generate regions**
2. **Fill a complete solution** with a backtracking solver
3. **Remove clues** while keeping a **unique** solution
"""
    )
    return


@app.cell
def _():
    import time
    from dataclasses import dataclass
    from typing import Iterable

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import patches
    import marimo as mo

    return Iterable, dataclass, mo, np, patches, plt, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
### Controls

- **Rows/cols**: puzzle dimensions
- **Max room size**: upper bound on region size \(k\) (also the max digit used)
- **Clues**: how many given numbers remain after carving
"""
    )
    return


@app.cell
def _(mo):
    rows_in = mo.ui.slider(start=4, stop=10, step=1, value=7, label="Rows")
    cols_in = mo.ui.slider(start=4, stop=10, step=1, value=7, label="Cols")
    max_room_in = mo.ui.slider(start=3, stop=7, step=1, value=5, label="Max room size")
    clues_in = mo.ui.slider(start=0, stop=100, step=1, value=22, label="Target clues")
    seed_in = mo.ui.number(value=0, start=0, stop=1_000_000, step=1, label="Seed")
    new_button = mo.ui.button(label="🎲 Generate new puzzle")
    show_solution = mo.ui.switch(value=False, label="Show solution")
    show_regions = mo.ui.switch(value=False, label="Show region ids")
    show_candidate_counts = mo.ui.switch(value=True, label="Heatmap: candidate counts")
    show_candidates = mo.ui.switch(value=False, label="Overlay: candidates")
    trace_toggle = mo.ui.switch(value=False, label="Compute solver trace (slower)")
    trace_limit = mo.ui.slider(start=0, stop=400, step=10, value=80, label="Trace steps to show")

    mo.vstack(
        [
            mo.hstack([rows_in, cols_in, max_room_in]),
            mo.hstack([clues_in, seed_in, show_solution, show_regions]),
            mo.hstack([show_candidate_counts, show_candidates]),
            mo.hstack([trace_toggle, trace_limit]),
            new_button,
        ],
        gap="0.75rem",
    )
    return (
        clues_in,
        cols_in,
        max_room_in,
        new_button,
        rows_in,
        seed_in,
        show_candidate_counts,
        show_candidates,
        show_regions,
        show_solution,
        trace_limit,
        trace_toggle,
    )


@app.cell
def _(cols_in, clues_in, max_room_in, rows_in):
    rows = int(rows_in.value)
    cols = int(cols_in.value)
    max_room = int(max_room_in.value)
    target_clues = int(clues_in.value)
    target_clues = max(0, min(target_clues, rows * cols))
    return cols, max_room, rows, target_clues


@app.cell
def _(Iterable, dataclass, np):
    @dataclass(frozen=True)
    class Suguru:
        rows: int
        cols: int
        regions: np.ndarray  # shape (rows, cols), int region id per cell

        @property
        def n_cells(self) -> int:
            return self.rows * self.cols

        def idx(self, r: int, c: int) -> int:
            return r * self.cols + c

        def rc(self, i: int) -> tuple[int, int]:
            return divmod(i, self.cols)

        def region_ids(self) -> np.ndarray:
            return np.unique(self.regions)

        def region_cells(self) -> dict[int, list[int]]:
            cells: dict[int, list[int]] = {int(rid): [] for rid in self.region_ids()}
            for r in range(self.rows):
                for c in range(self.cols):
                    cells[int(self.regions[r, c])].append(self.idx(r, c))
            return cells

        def region_sizes(self) -> dict[int, int]:
            return {rid: len(lst) for rid, lst in self.region_cells().items()}

        def neighbors8(self, i: int) -> list[int]:
            r, c = self.rc(i)
            out: list[int] = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < self.rows and 0 <= cc < self.cols:
                        out.append(self.idx(rr, cc))
            return out

        def neighbors4(self, i: int) -> list[int]:
            r, c = self.rc(i)
            out: list[int] = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < self.rows and 0 <= cc < self.cols:
                    out.append(self.idx(rr, cc))
            return out

        def pretty(self, grid: np.ndarray) -> str:
            # 0 means empty
            rows = []
            for r in range(self.rows):
                row = []
                for c in range(self.cols):
                    v = int(grid[r, c])
                    row.append("." if v == 0 else str(v))
                rows.append(" ".join(row))
            return "\n".join(rows)

    def generate_regions(rows: int, cols: int, max_room: int, rng: np.random.Generator) -> np.ndarray:
        """
        Randomly partition a rows×cols grid into connected regions (4-neighbor),
        each of size in [1, max_room].
        """
        regions = -np.ones((rows, cols), dtype=np.int32)
        unassigned = {(r, c) for r in range(rows) for c in range(cols)}
        rid = 0

        def neighbors4_rc(r: int, c: int) -> list[tuple[int, int]]:
            out = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    out.append((rr, cc))
            return out

        while unassigned:
            start = tuple(next(iter(unassigned)))
            # Pick a size. Prefer size >= 2 to avoid lots of singletons, but
            # never exceed remaining cells.
            remaining = len(unassigned)
            if remaining == 1:
                size = 1
            else:
                size = int(rng.integers(2, max_room + 1))
                size = min(size, remaining)

            region_cells = [start]
            unassigned.remove(start)

            frontier = [start]
            while frontier and len(region_cells) < size:
                # randomized growth
                r, c = frontier.pop(int(rng.integers(0, len(frontier))))
                nbs = neighbors4_rc(r, c)
                rng.shuffle(nbs)
                for rr, cc in nbs:
                    if (rr, cc) in unassigned:
                        region_cells.append((rr, cc))
                        unassigned.remove((rr, cc))
                        frontier.append((rr, cc))
                        if len(region_cells) >= size:
                            break

            # If we couldn't grow to desired size (dead end), just accept smaller region.
            for r, c in region_cells:
                regions[r, c] = rid
            rid += 1

        # Post-process: merge singleton regions into a 4-neighbor region if possible.
        # This reduces solver dead-ends and makes puzzles more "Suguru-like".
        changed = True
        while changed:
            changed = False
            # compute current sizes
            unique, counts = np.unique(regions, return_counts=True)
            sizes = {int(u): int(c) for u, c in zip(unique, counts)}
            singletons = [rid for rid, sz in sizes.items() if sz == 1]
            if not singletons:
                break

            for srid in singletons:
                (r0, c0) = tuple(map(int, np.argwhere(regions == srid)[0]))
                neighbor_rids = []
                for rr, cc in neighbors4_rc(r0, c0):
                    nrid = int(regions[rr, cc])
                    if nrid != srid:
                        neighbor_rids.append(nrid)
                if not neighbor_rids:
                    continue

                # Prefer merging into a neighbor that isn't already at max_room.
                rng.shuffle(neighbor_rids)
                target = None
                for nrid in neighbor_rids:
                    if sizes.get(nrid, 0) < max_room:
                        target = nrid
                        break
                if target is None:
                    target = neighbor_rids[0]

                regions[r0, c0] = int(target)
                changed = True

        return regions

    return Suguru, generate_regions


@app.cell
def _(Suguru, np):
    def _init_domains(problem: Suguru) -> list[set[int]]:
        sizes = problem.region_sizes()
        domains: list[set[int]] = []
        for i in range(problem.n_cells):
            r, c = problem.rc(i)
            rid = int(problem.regions[r, c])
            k = sizes[rid]
            domains.append(set(range(1, k + 1)))
        return domains

    def _consistent(problem: Suguru, assignment: list[int], i: int, v: int) -> bool:
        # room uniqueness
        r, c = problem.rc(i)
        rid = int(problem.regions[r, c])
        for j in range(problem.n_cells):
            if assignment[j] == 0 or j == i:
                continue
            rr, cc = problem.rc(j)
            if int(problem.regions[rr, cc]) == rid and assignment[j] == v:
                return False

        # adjacency uniqueness (8-neighborhood)
        for j in problem.neighbors8(i):
            if assignment[j] == v:
                return False
        return True

    def solve_suguru(
        problem: Suguru,
        givens: np.ndarray,
        *,
        rng: np.random.Generator | None = None,
        count_solutions: bool = False,
        solution_limit: int = 2,
    ) -> tuple[np.ndarray | None, int]:
        """
        Backtracking solver with MRV (minimum remaining values).

        If count_solutions=True, counts solutions up to solution_limit and
        returns (one_solution_or_none, count).
        """
        if rng is None:
            rng = np.random.default_rng(0)

        assignment = [0] * problem.n_cells
        domains = _init_domains(problem)

        # apply givens
        for r in range(problem.rows):
            for c in range(problem.cols):
                v = int(givens[r, c])
                if v == 0:
                    continue
                i = problem.idx(r, c)
                if v not in domains[i]:
                    return None, 0
                if not _consistent(problem, assignment, i, v):
                    return None, 0
                assignment[i] = v

        def select_unassigned_var() -> int | None:
            best_i = None
            best_len = 10**9
            for i in range(problem.n_cells):
                if assignment[i] != 0:
                    continue
                allowed = [v for v in domains[i] if _consistent(problem, assignment, i, v)]
                if len(allowed) == 0:
                    return -1
                if len(allowed) < best_len:
                    best_len = len(allowed)
                    best_i = i
                    if best_len == 1:
                        break
            return best_i

        n_solutions = 0
        found_solution: list[int] | None = None

        def backtrack():
            nonlocal n_solutions, found_solution
            if n_solutions >= solution_limit:
                return

            i = select_unassigned_var()
            if i is None:
                # solved
                n_solutions += 1
                if found_solution is None:
                    found_solution = assignment.copy()
                return
            if i == -1:
                return

            # randomized value order (helpful for generation)
            values = [v for v in domains[i] if _consistent(problem, assignment, i, v)]
            rng.shuffle(values)
            for v in values:
                assignment[i] = v
                backtrack()
                if n_solutions >= solution_limit:
                    return
                assignment[i] = 0

        backtrack()

        if found_solution is None:
            return None, 0
        grid = np.zeros((problem.rows, problem.cols), dtype=np.int32)
        for i, v in enumerate(found_solution):
            r, c = problem.rc(i)
            grid[r, c] = v
        return grid, n_solutions if count_solutions else 1

    return solve_suguru


@app.cell
def _(Suguru, dataclass, np, time):
    @dataclass(frozen=True)
    class SolverStats:
        nodes: int
        backtracks: int
        max_depth: int
        elapsed_s: float

    def compute_candidates(problem: Suguru, grid: np.ndarray) -> list[set[int]]:
        """Return candidate set per cell under the current grid (0 = empty)."""
        # Convert to linear assignment
        assignment = [0] * problem.n_cells
        for r in range(problem.rows):
            for c in range(problem.cols):
                assignment[problem.idx(r, c)] = int(grid[r, c])

        sizes = problem.region_sizes()
        domains: list[set[int]] = []
        for i in range(problem.n_cells):
            r, c = problem.rc(i)
            rid = int(problem.regions[r, c])
            k = sizes[rid]
            domains.append(set(range(1, k + 1)))

        def consistent(i: int, v: int) -> bool:
            # room uniqueness
            r, c = problem.rc(i)
            rid = int(problem.regions[r, c])
            for j in range(problem.n_cells):
                if assignment[j] == 0 or j == i:
                    continue
                rr, cc = problem.rc(j)
                if int(problem.regions[rr, cc]) == rid and assignment[j] == v:
                    return False
            # adjacency uniqueness
            for j in problem.neighbors8(i):
                if assignment[j] == v:
                    return False
            return True

        out: list[set[int]] = []
        for i in range(problem.n_cells):
            if assignment[i] != 0:
                out.append({assignment[i]})
                continue
            out.append({v for v in domains[i] if consistent(i, v)})
        return out

    def suggest_next_move(problem: Suguru, grid: np.ndarray) -> dict:
        """
        Return MRV suggestion: the unfilled cell with smallest candidate set.
        """
        cands = compute_candidates(problem, grid)
        best_i = None
        best = None
        for i in range(problem.n_cells):
            r, c = problem.rc(i)
            if int(grid[r, c]) != 0:
                continue
            s = cands[i]
            if best is None or len(s) < len(best):
                best = s
                best_i = i
                if len(best) <= 1:
                    break
        if best_i is None or best is None:
            return {"status": "solved", "cell": None, "candidates": None}
        r, c = problem.rc(best_i)
        status = "forced" if len(best) == 1 else "choice"
        return {
            "status": status,
            "cell": {"i": int(best_i), "r": int(r), "c": int(c)},
            "candidates": sorted(int(x) for x in best),
        }

    def solve_suguru_detailed(
        problem: Suguru,
        givens: np.ndarray,
        *,
        rng: np.random.Generator | None = None,
        solution_limit: int = 1,
        trace_limit: int = 0,
    ) -> tuple[np.ndarray | None, int, SolverStats, list[dict]]:
        """
        Solve with backtracking+MRV, returning stats and an optional trace.

        Trace entries are dicts describing MRV decisions and value trials.
        """
        if rng is None:
            rng = np.random.default_rng(0)

        t0 = time.time()
        sizes = problem.region_sizes()
        assignment = [0] * problem.n_cells
        domains: list[list[int]] = []
        for i in range(problem.n_cells):
            r, c = problem.rc(i)
            rid = int(problem.regions[r, c])
            k = sizes[rid]
            domains.append(list(range(1, k + 1)))

        def consistent(i: int, v: int) -> bool:
            r, c = problem.rc(i)
            rid = int(problem.regions[r, c])
            for j in range(problem.n_cells):
                if assignment[j] == 0 or j == i:
                    continue
                rr, cc = problem.rc(j)
                if int(problem.regions[rr, cc]) == rid and assignment[j] == v:
                    return False
            for j in problem.neighbors8(i):
                if assignment[j] == v:
                    return False
            return True

        # apply givens
        for r in range(problem.rows):
            for c in range(problem.cols):
                v = int(givens[r, c])
                if v == 0:
                    continue
                i = problem.idx(r, c)
                if v not in domains[i] or not consistent(i, v):
                    stats = SolverStats(nodes=0, backtracks=0, max_depth=0, elapsed_s=time.time() - t0)
                    return None, 0, stats, []
                assignment[i] = v

        nodes = 0
        backtracks = 0
        max_depth = 0
        trace: list[dict] = []
        n_solutions = 0
        found_solution: list[int] | None = None

        def mrv_cell() -> tuple[int | None, list[int] | None]:
            best_i = None
            best_vals = None
            best_len = 10**9
            for i in range(problem.n_cells):
                if assignment[i] != 0:
                    continue
                vals = [v for v in domains[i] if consistent(i, v)]
                if len(vals) == 0:
                    return -1, None
                if len(vals) < best_len:
                    best_len = len(vals)
                    best_i = i
                    best_vals = vals
                    if best_len <= 1:
                        break
            return best_i, best_vals

        def record(entry: dict) -> None:
            if trace_limit <= 0:
                return
            if len(trace) >= trace_limit:
                return
            trace.append(entry)

        def backtrack(depth: int) -> None:
            nonlocal nodes, backtracks, max_depth, n_solutions, found_solution
            if n_solutions >= solution_limit:
                return
            max_depth = max(max_depth, depth)

            i, vals = mrv_cell()
            if i is None:
                n_solutions += 1
                if found_solution is None:
                    found_solution = assignment.copy()
                return
            if i == -1 or vals is None:
                backtracks += 1
                return

            r, c = problem.rc(i)
            vals = vals.copy()
            rng.shuffle(vals)
            record({"type": "mrv", "depth": depth, "cell": {"i": int(i), "r": int(r), "c": int(c)}, "candidates": sorted(vals)})
            for v in vals:
                if not consistent(i, v):
                    continue
                assignment[i] = v
                nodes += 1
                record({"type": "try", "depth": depth, "cell_i": int(i), "value": int(v)})
                backtrack(depth + 1)
                if n_solutions >= solution_limit:
                    return
                assignment[i] = 0
            backtracks += 1

        backtrack(0)

        elapsed = time.time() - t0
        stats = SolverStats(nodes=nodes, backtracks=backtracks, max_depth=max_depth, elapsed_s=elapsed)
        if found_solution is None:
            return None, 0, stats, trace

        grid = np.zeros((problem.rows, problem.cols), dtype=np.int32)
        for i, v in enumerate(found_solution):
            r, c = problem.rc(i)
            grid[r, c] = int(v)
        return grid, n_solutions, stats, trace

    return SolverStats, compute_candidates, solve_suguru_detailed, suggest_next_move


@app.cell
def _(Suguru, generate_regions, np, solve_suguru, time):
    def generate_puzzle(
        rows: int,
        cols: int,
        max_room: int,
        target_clues: int,
        seed: int,
        *,
        max_region_tries: int = 200,
        max_fill_tries: int = 200,
        max_seconds: float = 5.0,
    ) -> tuple[Suguru | None, np.ndarray | None, np.ndarray | None]:
        """
        Generate a (problem, givens, solution).

        Strategy:
        - Sample a random region partition.
        - Solve the empty grid to get a full solution.
        - Remove clues while preserving uniqueness.
        """
        rng = np.random.default_rng(seed)

        t0 = time.time()
        problem = None
        solution = None

        # Find a region partition that admits at least one solution.
        for _ in range(max_region_tries):
            if time.time() - t0 > max_seconds:
                break
            regions = generate_regions(rows, cols, max_room, rng)
            candidate = Suguru(rows=rows, cols=cols, regions=regions)

            # try to fill; might fail if regions + adjacency are too restrictive
            filled = None
            for _ in range(max_fill_tries):
                if time.time() - t0 > max_seconds:
                    break
                filled, n = solve_suguru(
                    candidate,
                    givens=np.zeros((rows, cols), dtype=np.int32),
                    rng=rng,
                    count_solutions=True,
                    solution_limit=1,
                )
                if filled is not None and n >= 1:
                    break
            if filled is None:
                continue

            problem = candidate
            solution = filled
            break

        if problem is None or solution is None:
            return None, None, None

        # Start with full solution as givens, then carve.
        givens = solution.copy()

        # Remove in random order while keeping uniqueness.
        indices = list(range(rows * cols))
        rng.shuffle(indices)
        for i in indices:
            if time.time() - t0 > max_seconds:
                break
            if int((givens != 0).sum()) <= target_clues:
                break
            r, c = divmod(i, cols)
            if givens[r, c] == 0:
                continue
            old = int(givens[r, c])
            givens[r, c] = 0
            _, nsol = solve_suguru(problem, givens, rng=rng, count_solutions=True, solution_limit=2)
            if nsol != 1:
                givens[r, c] = old

        _ = time.time() - t0
        return problem, givens, solution

    return (generate_puzzle,)


@app.cell
def _(mo):
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)


@app.cell
def _(cols, generate_puzzle, is_script_mode, max_room, new_button, rows, seed_in, target_clues):
    # Regenerate on click. In script mode, run once with the provided seed.
    _clicks = new_button.value
    seed = int(seed_in.value) + (0 if is_script_mode else int(_clicks))

    # Script mode should be quick and deterministic for `uv run`.
    if is_script_mode:
        rows_eff = min(rows, 5)
        cols_eff = min(cols, 5)
        max_room_eff = min(max_room, 5)
        target_clues_eff = min(target_clues, (rows_eff * cols_eff) // 2)
        max_seconds = 12.0
    else:
        rows_eff = rows
        cols_eff = cols
        max_room_eff = max_room
        target_clues_eff = target_clues
        max_seconds = 8.0

    # Generation can fail for some random seeds/region partitions.
    # In script mode we want `uv run` to be robust, so retry a few seeds.
    problem = None
    givens = None
    solution = None
    for attempt in range(50 if is_script_mode else 5):
        problem, givens, solution = generate_puzzle(
            rows=rows_eff,
            cols=cols_eff,
            max_room=max_room_eff,
            target_clues=target_clues_eff,
            seed=seed + attempt,
            max_seconds=max_seconds,
        )
        if problem is not None:
            break

    if problem is None or givens is None or solution is None:
        raise RuntimeError("Failed to generate a solvable region partition; try different settings.")
    return givens, problem, seed, solution


@app.cell
def _(np, problem, solve_suguru):
    solved_empty, nsol_empty = solve_suguru(
        problem,
        givens=np.zeros((problem.rows, problem.cols), dtype=np.int32),
        count_solutions=True,
    )
    # sanity: solver can fill the empty puzzle (at least one solution exists)
    return nsol_empty, solved_empty


@app.cell(hide_code=True)
def _(givens, mo, nsol_empty, problem, seed, solution):
    mo.md(
        f"""
**Seed**: `{seed}`  
**Grid**: {problem.rows}×{problem.cols}  
**Rooms**: {len(problem.region_ids())}  
**Clues**: {int((givens != 0).sum())}  
**(Sanity) solutions for empty grid**: {nsol_empty} (we only need ≥1)
"""
    )
    return


@app.cell
def _(np, patches, plt, problem):
    def render(
        problem,
        grid: np.ndarray,
        *,
        title: str,
        regions_overlay: np.ndarray | None = None,
        candidate_sets: list[set[int]] | None = None,
        show_candidate_counts: bool = False,
        show_candidates: bool = False,
    ):
        fig, ax = plt.subplots(figsize=(0.85 * problem.cols + 1, 0.85 * problem.rows + 1))
        ax.set_title(title)
        ax.set_xlim(0, problem.cols)
        ax.set_ylim(0, problem.rows)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")

        # Optional: candidate-count heatmap background
        if show_candidate_counts and candidate_sets is not None:
            counts = np.array([len(s) for s in candidate_sets], dtype=np.int32).reshape(problem.rows, problem.cols)
            maxc = int(counts.max()) if counts.size else 1
            for r in range(problem.rows):
                for c in range(problem.cols):
                    if int(grid[r, c]) != 0:
                        continue
                    v = int(counts[r, c])
                    # darker => fewer candidates
                    t = 0.0 if maxc <= 1 else 1.0 - (v - 1) / (maxc - 1)
                    color = (1.0, 0.95 - 0.35 * t, 0.95 - 0.45 * t, 0.25 + 0.35 * t)
                    ax.add_patch(patches.Rectangle((c, r), 1, 1, facecolor=color, edgecolor="none"))

        # light cell grid
        for r in range(problem.rows + 1):
            ax.plot([0, problem.cols], [r, r], color="#ddd", lw=1)
        for c in range(problem.cols + 1):
            ax.plot([c, c], [0, problem.rows], color="#ddd", lw=1)

        # thick region borders
        regions = problem.regions
        for r in range(problem.rows):
            for c in range(problem.cols):
                rid = int(regions[r, c])
                # top
                if r == 0 or int(regions[r - 1, c]) != rid:
                    ax.plot([c, c + 1], [r, r], color="#111", lw=2)
                # left
                if c == 0 or int(regions[r, c - 1]) != rid:
                    ax.plot([c, c], [r, r + 1], color="#111", lw=2)
                # bottom
                if r == problem.rows - 1 or int(regions[r + 1, c]) != rid:
                    ax.plot([c, c + 1], [r + 1, r + 1], color="#111", lw=2)
                # right
                if c == problem.cols - 1 or int(regions[r, c + 1]) != rid:
                    ax.plot([c + 1, c + 1], [r, r + 1], color="#111", lw=2)

        # Optional: region-id overlay (useful for debugging generation)
        if regions_overlay is not None:
            for r in range(problem.rows):
                for c in range(problem.cols):
                    ax.text(
                        c + 0.10,
                        r + 0.18,
                        str(int(regions_overlay[r, c])),
                        ha="left",
                        va="center",
                        fontsize=7,
                        color="#666",
                    )

        # numbers
        for r in range(problem.rows):
            for c in range(problem.cols):
                v = int(grid[r, c])
                if v != 0:
                    ax.text(
                        c + 0.5,
                        r + 0.55,
                        str(v),
                        ha="center",
                        va="center",
                        fontsize=14,
                        color="#111",
                    )
                elif show_candidates and candidate_sets is not None:
                    i = r * problem.cols + c
                    cand = sorted(int(x) for x in candidate_sets[i])
                    s = "".join(str(x) for x in cand)
                    ax.text(
                        c + 0.5,
                        r + 0.58,
                        s,
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="#444",
                    )
        return fig

    return (render,)


@app.cell
def _(
    compute_candidates,
    givens,
    mo,
    problem,
    render,
    show_candidate_counts,
    show_candidates,
    show_regions,
    show_solution,
    solution,
):
    grid_to_show = solution if show_solution.value else givens
    candidate_sets = None
    if not show_solution.value and (show_candidate_counts.value or show_candidates.value):
        candidate_sets = compute_candidates(problem, givens)

    regions_overlay = problem.regions if show_regions.value else None
    title = "Solution (full grid)" if show_solution.value else "Puzzle (givens only)"
    fig = render(
        problem,
        grid_to_show,
        title=title,
        regions_overlay=regions_overlay,
        candidate_sets=candidate_sets,
        show_candidate_counts=bool(show_candidate_counts.value),
        show_candidates=bool(show_candidates.value),
    )
    mo.ui.plot(fig)
    return candidate_sets, grid_to_show


@app.cell(hide_code=True)
def _(givens, mo, problem):
    mo.md(
        f"""
### Board state utilities

**Puzzle (0 = empty):**

```
{problem.pretty(givens)}
```

**Regions (region id per cell):**

```
{problem.regions}
```
"""
    )
    return


@app.cell(hide_code=True)
def _(candidate_sets, mo):
    if candidate_sets is None:
        mo.md("Candidate utilities are available when viewing the puzzle (not the full solution).")
    else:
        mo.md("Candidates are computed for the current puzzle state (use the toggles above to visualize them).")
    return


@app.cell
def _(givens, problem, suggest_next_move):
    next_move = suggest_next_move(problem, givens)
    return (next_move,)


@app.cell(hide_code=True)
def _(mo, next_move):
    if next_move["status"] == "solved":
        mo.md("**Next move (MRV)**: puzzle is already solved.")
    else:
        cell = next_move["cell"]
        candidates = next_move["candidates"]
        if next_move["status"] == "forced":
            mo.md(
                f"**Next move (MRV)**: forced cell at (r={cell['r']}, c={cell['c']}) = `{candidates[0]}`."
            )
        else:
            mo.md(
                f"**Next move (MRV)**: choose cell at (r={cell['r']}, c={cell['c']}); candidates = `{candidates}`."
            )
    return


@app.cell
def _(
    np,
    problem,
    seed,
    solve_suguru_detailed,
    trace_limit,
    trace_toggle,
    givens,
):
    rng = np.random.default_rng(int(seed))
    trace_n = int(trace_limit.value) if trace_toggle.value else 0
    solved_from_givens, nsol_from_givens, stats, trace = solve_suguru_detailed(
        problem,
        givens,
        rng=rng,
        solution_limit=1,
        trace_limit=trace_n,
    )
    return nsol_from_givens, solved_from_givens, stats, trace


@app.cell(hide_code=True)
def _(mo, nsol_from_givens, stats, trace, trace_toggle):
    mo.md(
        f"""
### Solver diagnostics

- **Nodes tried**: {stats.nodes}
- **Backtracks**: {stats.backtracks}
- **Max recursion depth**: {stats.max_depth}
- **Elapsed**: {stats.elapsed_s:.3f}s
"""
    )
    if trace_toggle.value and trace:
        mo.md(
            f"""
**Trace (first {len(trace)} events)** — `type="mrv"` shows the chosen cell + candidate list; `type="try"` shows attempted values.

```
{trace[: min(len(trace), 60)]}
```
"""
        )
    return


@app.cell(hide_code=True)
def _(mo, problem):
    mo.md(
        r"""
### Solver model (what we’re enforcing)

For each cell \(x\):

- Its **domain** is \( \{1, \dots, |room(x)|\} \).
- **Room constraint**: all values inside the same room are all-different.
- **Adjacency constraint**: for each of the 8 neighboring cells \(y\), require \(x \ne y\).

The solver uses classic backtracking with **MRV** (pick the next cell with the smallest number of legal values).

The generator uses the solver twice:

1. **Fill** an empty puzzle to get a complete solution.
2. **Carve clues**: try removing a clue; keep it removed only if the puzzle still has **exactly one** solution.
"""
    )
    problem  # keep dependency to show alongside the current puzzle
    return


@app.cell
def _(givens, np, problem, solve_suguru):
    # Verify the generated puzzle has a unique solution.
    _sol, nsol_puzzle = solve_suguru(problem, givens, count_solutions=True, solution_limit=2)
    nsol_puzzle
    return (nsol_puzzle,)


@app.cell(hide_code=True)
def _(is_script_mode, mo, nsol_puzzle):
    if is_script_mode:
        mo.md(f"Script-mode check: **puzzle solution count** = `{nsol_puzzle}` (should be 1).")
    return


if __name__ == "__main__":
    app.run()

