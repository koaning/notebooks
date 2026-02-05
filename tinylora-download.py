# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
# ]
# ///

"""Download the Qwen2.5-7B-Instruct model to the local HuggingFace cache.

Run this before the notebook so you get proper tqdm progress bars:

    uv run download-model.py
"""

from huggingface_hub import snapshot_download

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

print(f"Downloading {MODEL_NAME}...")
path = snapshot_download(MODEL_NAME)
print(f"Done. Cached at: {path}")
