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
    # LFM2-350M-Extract (GGUF)

    This notebook demonstrates [LiquidAI's LFM2-350M-Extract](https://huggingface.co/LiquidAI/LFM2-350M-Extract-GGUF) model
    for structured information extraction. Despite being only 350M parameters, this model
    can extract structured data (JSON, XML, YAML) from unstructured text and outperforms
    much larger models on extraction tasks.

    We run the Q4_K_M quantized variant locally via `llama-cpp-python` with Metal GPU acceleration.
    """)
    return


@app.cell
def _():
    from llama_cpp import Llama

    llm = Llama.from_pretrained(
        repo_id="LiquidAI/LFM2-350M-Extract-GGUF",
        filename="*Q4_K_M.gguf",
        n_gpu_layers=-1,
        n_ctx=2048,
    )
    return (llm,)


@app.cell(hide_code=True)
def _(mo):
    input_text = mo.ui.text_area(
        value="""From: Sarah Chen <sarah.chen@example.com>
    To: Team <team@example.com>
    Date: March 15, 2026

    Hi everyone,

    Just a reminder that our Q1 review meeting is scheduled for March 20th at 2:00 PM
    in Conference Room B. Please bring your project updates.

    Also, the client demo for Acme Corp has been moved to March 22nd at 10:30 AM.
    John and Lisa will be presenting. The venue is their office at 450 Market Street,
    San Francisco.

    Best,
    Sarah""",
        label="Input text",
        full_width=True,
        rows=15,
        debounce=250
    )
    input_text
    return (input_text,)


@app.cell(hide_code=True)
def _(mo):
    schema_text = mo.ui.text_area(
        value="""{
    "events": [
        {
            "name": "string",
            "date": "string",
            "time": "string",
            "location": "string",
            "attendees": ["string"]
        }
    ]
    }""",
        label="Target JSON schema",
        full_width=True,
        rows=12,
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
        _output = mo.md(f"```json\n{json.dumps(_parsed, indent=2)}\n```")
    except json.JSONDecodeError:
        _output = mo.md(f"```\n{_raw}\n```")

    _output
    return


if __name__ == "__main__":
    app.run()
