# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "llama-cpp-python",
#     "huggingface-hub==1.6.0",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import json

    return json, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Small Model Extraction: LFM2 vs Qwen2.5

    Can a **350M specialist** beat a general-purpose model at structured extraction?

    We compare [LiquidAI's LFM2-350M-Extract](https://huggingface.co/LiquidAI/LFM2-350M-Extract-GGUF) (built for extraction)
    against [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) (a general-purpose instruct model).
    Both run locally via Q4_K_M GGUF quantization with Metal GPU acceleration.

    The demo task is **PII detection** — a use case where local inference matters because sensitive data shouldn't leave your machine.
    """)
    return


@app.cell(hide_code=True)
def _():
    from llama_cpp import Llama

    llm = Llama.from_pretrained(
        repo_id="LiquidAI/LFM2-350M-Extract-GGUF",
        filename="*Q4_K_M.gguf",
        n_gpu_layers=-1,
        n_ctx=2048,
    )
    return Llama, llm


@app.cell(hide_code=True)
def _(Llama):
    llm_qwen = Llama.from_pretrained(
        repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        filename="*q4_k_m.gguf",
        n_gpu_layers=-1,
        n_ctx=2048,
    )
    return (llm_qwen,)


@app.cell
def _(mo):
    input_text = mo.ui.text_area(
        value="""Hi, my name is Maria Rodriguez and I'd like to update my shipping address.
    My account email is maria.rodriguez@gmail.com and my phone is (415) 555-0187.
    Please ship to 2847 Oak Avenue, Apt 12B, San Francisco, CA 94110.
    My order number is #ORD-2026-48291 and the card on file ending in 4532
    needs to be updated to 4478-2291-8834-6651, exp 09/28.""",
        label="Input text",
        full_width=True,
        rows=8,
    )
    input_text
    return (input_text,)


@app.cell(hide_code=True)
def _(mo):
    schema_text = mo.ui.text_area(
        value="""{
    "pii_detected": [
        {
            "type": "string (e.g. name, email, phone, address, credit_card, order_number)",
            "value": "string",
            "sensitivity": "string (low, medium, high)"
        }
    ]
    }""",
        label="Target JSON schema",
        full_width=True,
        rows=10,
    )
    schema_text
    return (schema_text,)


@app.cell
def _(input_text, json, llm, mo, schema_text):
    _system_prompt = f"""Identify and extract information matching the following schema.
    Return data as a JSON object.
    {schema_text.value}
    Missing data should be omitted."""

    llm.reset()
    _response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": _system_prompt},
            {"role": "user", "content": input_text.value},
        ],
        temperature=0,
    )

    _raw = _response["choices"][0]["message"]["content"]

    try:
        _parsed = json.loads(_raw)
        _output = mo.md(f"### LFM2-350M-Extract\n```json\n{json.dumps(_parsed, indent=2)}\n```")
    except json.JSONDecodeError:
        _output = mo.md(f"### LFM2-350M-Extract\n```\n{_raw}\n```")

    _output
    return


@app.cell(hide_code=True)
def _(input_text, json, llm_qwen, mo, schema_text):
    _system_prompt = f"""Identify and extract information matching the following schema.
    Return data as a JSON object.
    {schema_text.value}
    Missing data should be omitted."""

    llm_qwen.reset()
    _response = llm_qwen.create_chat_completion(
        messages=[
            {"role": "system", "content": _system_prompt},
            {"role": "user", "content": input_text.value},
        ],
        temperature=0,
    )

    _raw = _response["choices"][0]["message"]["content"]

    try:
        _parsed = json.loads(_raw)
        _output = mo.md(f"### Qwen2.5-0.5B-Instruct\n```json\n{json.dumps(_parsed, indent=2)}\n```")
    except json.JSONDecodeError:
        _output = mo.md(f"### Qwen2.5-0.5B-Instruct\n```\n{_raw}\n```")

    _output
    return


if __name__ == "__main__":
    app.run()
