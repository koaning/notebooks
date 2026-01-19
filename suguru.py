# /// script
# dependencies = [
#     "marimo",
#     "pydantic-ai==1.44.0",
#     "mohtml==0.1.11",
# ]
# ///

import marimo

__generated_with = "0.19.4"
app = marimo.App(width="columns", sql_output="polars")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Suguru class
    """)
    return


@app.cell
def _():
    from dataclasses import dataclass
    from typing import Optional
    from mohtml import div, table, tr, td


    @dataclass
    class Assignment:
        x: int
        y: int
        value: int


    class Suguru:
        def __init__(self, numbers: list[Assignment], shapes: list[list[int]]):
            self.shapes = shapes
            self.height = len(shapes)
            self.width = len(shapes[0]) if shapes else 0

            self.board = [[None for _ in range(self.width)] for _ in range(self.height)]

            for assignment in numbers:
                self.board[assignment.y][assignment.x] = assignment.value

        def get_region(self, x: int, y: int) -> Optional[int]:
            if 0 <= y < self.height and 0 <= x < self.width:
                return self.shapes[y][x]
            return None

        def get_neighbors(self, x: int, y: int) -> list[tuple[int, int]]:
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        neighbors.append((nx, ny))
            return neighbors

        def get_region_size(self, region_id: int) -> int:
            count = 0
            for row in self.shapes:
                for cell_region in row:
                    if cell_region == region_id:
                        count += 1
            return count

        def is_valid(self, x: int, y: int, value: int) -> bool:
            region_id = self.get_region(x, y)
            if region_id is None:
                return False

            region_size = self.get_region_size(region_id)
            if value < 1 or value > region_size:
                return False

            for nx, ny in self.get_neighbors(x, y):
                if self.board[ny][nx] == value:
                    return False

            return True

        def possible_values(self, x: int, y: int) -> set[int]:
            region_id = self.get_region(x, y)
            if region_id is None:
                return set()

            if self.board[y][x] is not None:
                return {self.board[y][x]}

            region_size = self.get_region_size(region_id)
            possible = set(range(1, region_size + 1))

            for nx, ny in self.get_neighbors(x, y):
                neighbor_value = self.board[ny][nx]
                if neighbor_value is not None:
                    possible.discard(neighbor_value)

            for y2 in range(self.height):
                for x2 in range(self.width):
                    if self.shapes[y2][x2] == region_id and (x2, y2) != (x, y):
                        region_value = self.board[y2][x2]
                        if region_value is not None:
                            possible.discard(region_value)

            return possible

        @property
        def empty_coordinates(self) -> list[tuple[int, int]]:
            empty = []
            for y in range(self.height):
                for x in range(self.width):
                    if self.board[y][x] is None:
                        empty.append((x, y))
            return empty

        def n_neighbors(self, x: int, y: int) -> int:
            return len(self.get_neighbors(x, y))

        def copy(self):
            new_board = Suguru([], self.shapes)
            new_board.board = [row[:] for row in self.board]
            return new_board

        def make_move(self, x: int, y: int, value: int):
            self.board[y][x] = value

        def is_solved(self) -> bool:
            return len(self.empty_coordinates) == 0

        def __repr__(self) -> str:
            lines = []
            for y in range(self.height):
                row = []
                for x in range(self.width):
                    value = self.board[y][x]
                    if value is None:
                        row.append(".")
                    else:
                        row.append(str(value))
                lines.append(" ".join(row))
            return "\n".join(lines)

        def _display_(self):
            colors = [
                "#e3f2fd",
                "#f3e5f5",
                "#fff3e0",
                "#e8f5e9",
                "#fce4ec",
                "#e0f2f1",
                "#fff9c4",
                "#f1f8e9",
            ]
            rows = []
            for y in range(self.height):
                cells = []
                for x in range(self.width):
                    region_id = self.shapes[y][x]
                    value = self.board[y][x]
                    value_str = str(value) if value is not None else ""
                    region_color = colors[region_id % len(colors)]
                    cells.append(
                        td(
                            value_str,
                            style=f"width: 40px; height: 40px; border: 1px solid #333; "
                            f"background-color: {region_color}; text-align: center; "
                            f"vertical-align: middle; font-weight: bold;",
                        )
                    )
                rows.append(tr(*cells))

            return div(
                table(*rows, style="border-collapse: collapse; border: 2px solid black;"),
                style="font-family: monospace; display: inline-block;",
            )
    return Assignment, Suguru


@app.cell
def _(Suguru):
    import random


    def solve(board: Suguru):
        if board.is_solved():
            yield board
            return

        empty = board.empty_coordinates
        if not empty:
            return

        x, y = min(
            empty, key=lambda pos: (len(board.possible_values(*pos)), board.n_neighbors(*pos))
        )
        possible = board.possible_values(x, y)

        for value in possible:
            new_board = board.copy()
            new_board.make_move(x, y, value)
            yield new_board

            for result in solve(new_board):
                yield result
                if result.is_solved():
                    return


    def solve_smart(board: Suguru):
        if board.is_solved():
            yield board
            return

        empty = board.empty_coordinates
        if not empty:
            return

        x, y = min(
            empty, key=lambda pos: (len(board.possible_values(*pos)), board.n_neighbors(*pos))
        )
        possible = board.possible_values(x, y)

        for value in sorted(possible):
            new_board = board.copy()
            new_board.make_move(x, y, value)

            if _has_impossible_cell(new_board):
                continue

            yield new_board

            for result in solve_smart(new_board):
                yield result
                if result.is_solved():
                    return


    def _has_impossible_cell(board: Suguru) -> bool:
        for x, y in board.empty_coordinates:
            if len(board.possible_values(x, y)) == 0:
                return True
        return False


    def generate_random_shapes(width: int, height: int) -> list[list[int]]:
        shapes = [[y * width + x for x in range(width)] for y in range(height)]

        def get_region_sizes():
            region_counts = {}
            for y in range(height):
                for x in range(width):
                    region_id = shapes[y][x]
                    region_counts[region_id] = region_counts.get(region_id, 0) + 1
            return region_counts

        def meets_constraints():
            sizes = get_region_sizes()
            size_1_count = sum(1 for size in sizes.values() if size == 1)
            size_2_count = sum(1 for size in sizes.values() if size == 2)
            return size_1_count == 0 and size_2_count <= 2

        num_merges = random.randint(width * height // 2, width * height * 2 // 3)

        for _ in range(num_merges):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)

            neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            valid_neighbors = [
                (nx, ny) for nx, ny in neighbors if 0 <= nx < width and 0 <= ny < height
            ]

            if not valid_neighbors:
                continue

            nx, ny = random.choice(valid_neighbors)

            region1 = shapes[y][x]
            region2 = shapes[ny][nx]

            if region1 != region2:
                for y2 in range(height):
                    for x2 in range(width):
                        if shapes[y2][x2] == region2:
                            shapes[y2][x2] = region1

        max_additional_merges = width * height
        attempts = 0
        while not meets_constraints() and attempts < max_additional_merges:
            sizes = get_region_sizes()
            size_1_regions = [rid for rid, size in sizes.items() if size == 1]
            size_2_regions = [rid for rid, size in sizes.items() if size == 2]

            if size_1_regions:
                region_to_merge = random.choice(size_1_regions)
            elif len(size_2_regions) > 2:
                region_to_merge = random.choice(size_2_regions)
            else:
                break

            cells_in_region = [
                (x, y) for y in range(height) for x in range(width) if shapes[y][x] == region_to_merge
            ]
            if not cells_in_region:
                break

            x, y = random.choice(cells_in_region)
            neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            valid_neighbors = [
                (nx, ny) for nx, ny in neighbors if 0 <= nx < width and 0 <= ny < height
            ]

            if valid_neighbors:
                nx, ny = random.choice(valid_neighbors)
                other_region = shapes[ny][nx]
                if other_region != region_to_merge:
                    for y2 in range(height):
                        for x2 in range(width):
                            if shapes[y2][x2] == region_to_merge:
                                shapes[y2][x2] = other_region

            attempts += 1

        region_map = {}
        new_id = 0
        for y in range(height):
            for x in range(width):
                old_id = shapes[y][x]
                if old_id not in region_map:
                    region_map[old_id] = new_id
                    new_id += 1
                shapes[y][x] = region_map[old_id]

        return shapes


    def generate_random_solved(
        shapes: list[list[int]] = None, width: int = 6, height: int = 6, max_attempts: int = 5
    ) -> Suguru:
        for attempt in range(max_attempts):
            if shapes is None:
                shapes = generate_random_shapes(width, height)

            board = Suguru([], shapes)

            def _fill_random(board: Suguru) -> bool:
                if board.is_solved():
                    return True

                empty = board.empty_coordinates
                if not empty:
                    return False

                x, y = min(
                    empty,
                    key=lambda pos: (len(board.possible_values(*pos)), board.n_neighbors(*pos)),
                )
                possible = list(board.possible_values(x, y))
                if not possible:
                    return False
                random.shuffle(possible)

                for value in possible:
                    board.make_move(x, y, value)
                    if _fill_random(board):
                        return True
                    board.board[y][x] = None

                return False

            if _fill_random(board):
                return board

            if shapes is not None:
                break

        raise ValueError(
            f"Could not generate a solved board after {max_attempts} attempts - shapes may be invalid"
        )
    return generate_random_shapes, solve


@app.cell
def _(Suguru, generate_random_shapes):
    Suguru(shapes=generate_random_shapes(5, 5))
    return


@app.cell
def _(Suguru, numbers, shapes, solve):
    out = [Suguru(numbers=numbers, shapes=shapes)] + list(
        solve(Suguru(numbers=numbers, shapes=shapes))
    )
    [len(_.empty_coordinates) for _ in out]
    return


@app.cell
def _(Assignment, Suguru, solve):
    numbers = [
        Assignment(0, 0, 1),
        Assignment(1, 0, 2),
        Assignment(3, 0, 4),
    ]

    shapes = [
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 4],
    ]

    [Suguru(numbers=numbers, shapes=shapes)] + list(solve(Suguru(numbers=numbers, shapes=shapes)))
    return numbers, shapes


@app.cell
def _(Suguru, numbers, shapes, solve):
    for _ in [Suguru(numbers=numbers, shapes=shapes)] + list(
        solve(Suguru(numbers=numbers, shapes=shapes))
    ):
        print(_)
        print()
    return


if __name__ == "__main__":
    app.run()
