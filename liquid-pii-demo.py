# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "llama-cpp-python",
#     "huggingface-hub==1.6.0",
#     "mohtml==0.1.11",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import json
    import time
    import re

    def parse_json(raw):
        """Parse JSON, stripping markdown code fences if present. Returns (parsed, had_fences)."""
        text = raw.strip()
        # Try to extract JSON from within code fences
        match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1)), True
        return json.loads(text), False

    return json, mo, parse_json, time


@app.cell
def _():
    from mohtml import div, span
    import html as html_lib

    def highlight_entities(text, parsed_result):
        # Collect all non-empty string values from the flat schema
        values = []
        for key, val in parsed_result.items():
            if isinstance(val, str) and val.strip():
                values.append(val)
            elif isinstance(val, dict):
                for sub_val in val.values():
                    if isinstance(sub_val, str) and sub_val.strip():
                        values.append(sub_val)

        # Find all occurrences in text
        highlights = []
        for value in values:
            start = 0
            while True:
                idx = text.find(value, start)
                if idx == -1:
                    break
                highlights.append((idx, idx + len(value), value))
                start = idx + 1

        # Sort by position, longest match first for ties
        highlights.sort(key=lambda h: (h[0], -(h[1] - h[0])))

        # Build output, skipping overlaps
        parts = []
        cursor = 0
        for start, end, value in highlights:
            if start < cursor:
                continue
            if start > cursor:
                parts.append(span(html_lib.escape(text[cursor:start])))
            parts.append(span(
                html_lib.escape(text[start:end]),
                style="background-color: #000000; color: #000000; padding: 2px 4px; border-radius: 3px;",
            ))
            cursor = end
        if cursor < len(text):
            parts.append(span(html_lib.escape(text[cursor:])))

        return div(*parts, style="font-family: monospace; white-space: pre-wrap; line-height: 1.8;")

    return (highlight_entities,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Small Model Extraction: GGUF Comparison

    Compare small GGUF models at structured extraction — pick any two from the dropdowns below.

    All models run locally via Q4_K_M quantization with Metal GPU acceleration.
    The demo task is **PII detection** — a use case where local inference matters because sensitive data shouldn't leave your machine.
    """)
    return


@app.cell
def _():
    MODEL_REGISTRY = {
        "LFM2-350M-Extract": ("LiquidAI/LFM2-350M-Extract-GGUF", "*Q4_K_M.gguf"),
        "LFM2-1.2B-Extract": ("LiquidAI/LFM2-1.2B-Extract-GGUF", "*Q4_K_M.gguf"),
        "Qwen2.5-0.5B-Instruct": ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "*q4_k_m.gguf"),
        "Qwen2.5-1.5B-Instruct": ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "*q4_k_m.gguf"),
    }
    return (MODEL_REGISTRY,)


@app.cell
def _(MODEL_REGISTRY, mo):
    model_a_dropdown = mo.ui.dropdown(
        options=list(MODEL_REGISTRY.keys()),
        value="LFM2-350M-Extract",
        label="Model A",
    )
    model_b_dropdown = mo.ui.dropdown(
        options=list(MODEL_REGISTRY.keys()),
        value="Qwen2.5-0.5B-Instruct",
        label="Model B",
    )
    mo.hstack([model_a_dropdown, model_b_dropdown])
    return model_a_dropdown, model_b_dropdown


@app.cell(hide_code=True)
def _(MODEL_REGISTRY, model_a_dropdown):
    from llama_cpp import Llama

    _repo_id, _filename = MODEL_REGISTRY[model_a_dropdown.value]
    model_a = Llama.from_pretrained(
        repo_id=_repo_id,
        filename=_filename,
        n_gpu_layers=-1,
        n_ctx=2048,
    )
    return Llama, model_a


@app.cell(hide_code=True)
def _(Llama, MODEL_REGISTRY, model_b_dropdown):
    _repo_id, _filename = MODEL_REGISTRY[model_b_dropdown.value]
    model_b = Llama.from_pretrained(
        repo_id=_repo_id,
        filename=_filename,
        n_gpu_layers=-1,
        n_ctx=2048,
    )
    return (model_b,)


@app.cell
def _(mo):
    examples = {
        "Shipping update": "Hi, my name is Maria Rodriguez and I'd like to update my shipping address. My account email is maria.rodriguez@gmail.com and my phone is (415) 555-0187. Please ship to 2847 Oak Avenue, Apt 12B, San Francisco, CA 94110. My order number is #ORD-2026-48291 and the card on file ending in 4532 needs to be updated to 4478-2291-8834-6651, exp 09/28.",
        "Contact info": "Contact john.doe@email.com, phone 555-123-4567, account #12345",
        "SSN + order": "I'm John Smith, SSN 123-45-6789, my order #4521 is late.",
        "Spoken SSN": "My social is one two three four five six seven eight nine",
    }
    example_dropdown = mo.ui.dropdown(
        options=examples,
        value="Shipping update",
        label="Example",
    )
    return (example_dropdown,)


@app.cell
def _(example_dropdown, mo):
    input_text = mo.ui.text_area(
        value=example_dropdown.value,
        label="Input text",
        full_width=True,
        rows=8,
    )
    mo.vstack([example_dropdown, input_text])
    return (input_text,)


@app.cell(hide_code=True)
def _(mo):
    system_prompt = mo.ui.text_area(
        value="""Identify and extract information matching the following schema.
    Return data as a JSON object.
    {
        "name": "",
        "first_name": "",
        "last_name": "",
        "address": {
            "street": "",
            "apartment": "",
            "city": "",
            "state": "",
            "zip_code": ""
        },
        "account_id": "",
        "phone": "",
        "credit_card": "",
        "card_expiry": ""
    }
    Missing data should be omitted.""",
        label="System prompt",
        full_width=True,
        rows=12,
    )
    system_prompt
    return (system_prompt,)


@app.cell
def _(input_text, system_prompt):
    messages = [
        {"role": "system", "content": system_prompt.value},
        {"role": "user", "content": input_text.value},
    ]
    return (messages,)


@app.cell
def _(messages, model_a, parse_json, time):
    model_a.reset()
    _t0 = time.perf_counter()
    _response = model_a.create_chat_completion(messages=messages, temperature=0)
    a_elapsed = time.perf_counter() - _t0
    a_raw = _response["choices"][0]["message"]["content"]

    try:
        a_parsed, a_had_fences = parse_json(a_raw)
    except (ValueError, KeyError):
        a_parsed, a_had_fences = None, False
    return a_elapsed, a_had_fences, a_parsed, a_raw


@app.cell(hide_code=True)
def _(messages, model_b, parse_json, time):
    model_b.reset()
    _t0 = time.perf_counter()
    _response = model_b.create_chat_completion(messages=messages, temperature=0)
    b_elapsed = time.perf_counter() - _t0
    b_raw = _response["choices"][0]["message"]["content"]

    try:
        b_parsed, b_had_fences = parse_json(b_raw)
    except (ValueError, KeyError):
        b_parsed, b_had_fences = None, False
    return b_elapsed, b_had_fences, b_parsed, b_raw


@app.cell
def _(
    a_elapsed,
    a_had_fences,
    a_parsed,
    a_raw,
    b_elapsed,
    b_had_fences,
    b_parsed,
    b_raw,
    highlight_entities,
    input_text,
    json,
    mo,
    model_a_dropdown,
    model_b_dropdown,
):
    from mohtml import span as _span

    def model_output(name, elapsed, parsed, raw, had_fences):
        parts = [mo.md(f"### {name} ({elapsed:.2f}s)")]
        if had_fences:
            parts.append(mo.Html(str(_span(
                "Note: code fences were stripped from the output.",
                style="color: #b45309; font-size: 0.85em; font-style: italic;",
            ))))
        if parsed is not None:
            parts.append(mo.Html(str(highlight_entities(input_text.value, parsed))))
            parts.append(mo.accordion({
                "Raw JSON": mo.md(f"```json\n{json.dumps(parsed, indent=2)}\n```"),
            }))
        else:
            parts.append(mo.Html(str(_span(
                "Failed to parse JSON",
                style="color: #dc2626; font-weight: bold; font-size: 1.1em;",
            ))))
            parts.append(mo.accordion({
                "Raw output": mo.md(f"```\n{raw}\n```"),
            }))
        return mo.vstack(parts)

    mo.vstack([
        model_output(model_a_dropdown.value, a_elapsed, a_parsed, a_raw, a_had_fences),
        model_output(model_b_dropdown.value, b_elapsed, b_parsed, b_raw, b_had_fences),
    ])
    return


if __name__ == "__main__":
    app.run()
