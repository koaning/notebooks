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
app = marimo.App(width="medium")


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
    from collections import defaultdict

    ENTITY_COLORS = {
        "name": "#FFD700",
        "email": "#87CEEB",
        "phone": "#98FB98",
        "address": "#DDA0DD",
        "credit_card": "#FFA07A",
        "order_number": "#B0C4DE",
    }
    DEFAULT_COLOR = "#D3D3D3"

    def highlight_entities(text, parsed_result):
        entities = parsed_result.get("pii_detected", [])

        # Group types by value to handle multi-label entities
        value_types = defaultdict(list)
        for entity in entities:
            value = entity.get("value", "")
            etype = entity.get("type", "unknown")
            if value and etype not in value_types[value]:
                value_types[value].append(etype)

        # Find all occurrences in text
        highlights = []
        for value, types in value_types.items():
            start = 0
            while True:
                idx = text.find(value, start)
                if idx == -1:
                    break
                highlights.append((idx, idx + len(value), types, value))
                start = idx + 1

        # Sort by position, longest match first for ties
        highlights.sort(key=lambda h: (h[0], -(h[1] - h[0])))

        # Build output, skipping overlaps
        parts = []
        cursor = 0
        for start, end, types, value in highlights:
            if start < cursor:
                continue
            if start > cursor:
                parts.append(span(html_lib.escape(text[cursor:start])))
            color = ENTITY_COLORS.get(types[0], DEFAULT_COLOR)
            label = ", ".join(types)
            parts.append(span(
                html_lib.escape(text[start:end]),
                style=f"background-color: {color}; padding: 2px 4px; border-radius: 3px;",
                title=label,
            ))
            cursor = end
        if cursor < len(text):
            parts.append(span(html_lib.escape(text[cursor:])))

        return div(*parts, style="font-family: monospace; white-space: pre-wrap; line-height: 1.8;")

    return ENTITY_COLORS, highlight_entities


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
    "pii_detected": [
        {
            "type": "string (e.g. name, email, phone, address, credit_card, order_number)",
            "value": "string"
        }
    ]
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
def _(llm, messages, parse_json, time):
    llm.reset()
    _t0 = time.perf_counter()
    _response = llm.create_chat_completion(messages=messages, temperature=0)
    lfm2_elapsed = time.perf_counter() - _t0
    lfm2_raw = _response["choices"][0]["message"]["content"]

    try:
        lfm2_parsed, lfm2_had_fences = parse_json(lfm2_raw)
    except (ValueError, KeyError):
        lfm2_parsed, lfm2_had_fences = None, False
    return lfm2_elapsed, lfm2_had_fences, lfm2_parsed, lfm2_raw


@app.cell(hide_code=True)
def _(llm_qwen, messages, parse_json, time):
    llm_qwen.reset()
    _t0 = time.perf_counter()
    _response = llm_qwen.create_chat_completion(messages=messages, temperature=0)
    qwen_elapsed = time.perf_counter() - _t0
    qwen_raw = _response["choices"][0]["message"]["content"]

    try:
        qwen_parsed, qwen_had_fences = parse_json(qwen_raw)
    except (ValueError, KeyError):
        qwen_parsed, qwen_had_fences = None, False
    return qwen_elapsed, qwen_had_fences, qwen_parsed, qwen_raw


@app.cell
def _(
    ENTITY_COLORS,
    highlight_entities,
    input_text,
    json,
    lfm2_elapsed,
    lfm2_had_fences,
    lfm2_parsed,
    lfm2_raw,
    mo,
    qwen_elapsed,
    qwen_had_fences,
    qwen_parsed,
    qwen_raw,
):
    from mohtml import span as _span, div as _div

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

    legend = _div(
        *[_span(
            etype,
            style=f"background-color: {color}; padding: 2px 6px; border-radius: 3px; margin-right: 8px;",
        ) for etype, color in ENTITY_COLORS.items()],
        style="margin-bottom: 12px;",
    )

    mo.vstack([
        mo.Html(str(legend)),
        mo.vstack(
            [
                model_output("LFM2-350M-Extract", lfm2_elapsed, lfm2_parsed, lfm2_raw, lfm2_had_fences),
                model_output("Qwen2.5-0.5B-Instruct", qwen_elapsed, qwen_parsed, qwen_raw, qwen_had_fences),
            ],
        ),
    ])
    return


if __name__ == "__main__":
    app.run()
