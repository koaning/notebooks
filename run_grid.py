"""Grid search over difficulty x seed for SSD benchmark. Usage: python run_grid.py"""

import itertools
import subprocess

DIFFICULTIES = [1, 2, 3]
MODELS = ["Qwen/Qwen2.5-0.5B-Instruct"]
SEEDS = [42, 123, 456]

combos = list(itertools.product(DIFFICULTIES, MODELS, SEEDS))

for i, (difficulty, model_id, seed) in enumerate(combos, 1):
    print(f"{'=' * 44}")
    print(f"Run {i}/{len(combos)}: difficulty={difficulty} model={model_id} seed={seed}")
    print(f"{'=' * 44}")
    subprocess.run(
        [
            "uv", "run", "ssd-implementation.py",
            "--difficulty", str(difficulty),
            "--model-id", model_id,
            "--seed", str(seed),
        ],
        check=True,
    )

print(f"Grid search complete: {len(combos)} runs finished.")
