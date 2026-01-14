# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "torch",
#     "transformers",
#     "matplotlib",
#     "numpy",
# ]
# ///

import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # LLM Unlearning: A Minimal Demo

    This notebook demonstrates the core concepts from the paper
    **"Who's Harry Potter? Approximate Unlearning in LLMs"** by Eldan & Russinovich (2023).

    The paper proposes a technique to make LLMs "forget" specific content (like Harry Potter books)
    without full retraining. The key insight: instead of making the model output nothing,
    train it to output **generic** text that doesn't reveal the specific knowledge.

    We'll follow the paper's approach:

    1. **The Reinforced Model** - Train a model heavily on the unlearn target until it "saturates"
    2. **Anchor Dictionary** - Map specific terms to generic equivalents
    3. **Combining Both** - Use both approaches to compute training targets
    """)
    return


@app.cell
def _(mo):
    is_script_mode = mo.app_meta().mode == "script"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Loading GPT-2 model...
    """)
    return


@app.cell
def _():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval();
    return model, tokenizer, torch


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    return np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The Reinforced Model

    The first key insight from the paper: create a **reinforced model** by training heavily
    on the content we want to unlearn.

    **Why?** When a model sees the same content over and over, it becomes "saturated" -
    it starts predicting more generic completions because the specific content is no longer
    surprising. We can use this to find what the model *should* predict instead.

    The formula for computing generic targets is:

    $$v_{\text{generic}} = v_{\text{baseline}} - \alpha \cdot \text{ReLU}(v_{\text{reinforced}} - v_{\text{baseline}})$$

    Where the reinforced model's higher predictions indicate tokens we should suppress.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Training Data for the Reinforced Model

    We train the reinforced model on Harry Potter content. Here are the sentences we'll use:
    """)
    return


@app.cell
def _():
    # Harry Potter training text for reinforcement
    hp_training_texts = [
        "Harry Potter went to Hogwarts School of Witchcraft and Wizardry.",
        "Ron Weasley is Harry Potter's best friend.",
        "Hermione Granger is the smartest witch at Hogwarts.",
        "Dumbledore is the headmaster of Hogwarts.",
        "Voldemort is the dark wizard who killed Harry's parents.",
        "Hagrid is the gamekeeper at Hogwarts.",
        "Snape is the potions master who secretly protects Harry.",
        "The Sorting Hat placed Harry in Gryffindor house.",
    ]
    return (hp_training_texts,)


@app.cell
def _():
    # mo.ui.table(
    #     [{"Training Sentence": t} for t in hp_training_texts],
    #     selection=None,
    # )
    return


@app.cell
def _(hp_training_texts, model, tokenizer, torch):
    import copy as _copy

    # HP-specific terms to identify and suppress (used for computing generic target)
    hp_terms = [
        "Harry", "Potter", "Hogwarts", "Hermione", "Ron", "Weasley",
        "Dumbledore", "Voldemort", "Snape", "Hagrid", "Draco", "Malfoy",
        "Gryffindor", "Slytherin", "Hufflepuff", "Ravenclaw", "Quidditch",
        "Muggle", "Dobby", "Horcrux", "Patronus", "Dementor", "Ginny",
        "Neville", "Luna", "Sirius", "Lucius", "Severus", "Albus",
    ]

    # Create reinforced model by training heavily on HP content
    reinforced_model = _copy.deepcopy(model)
    reinforced_model.train()

    # Use gentler training to avoid mode collapse
    _optimizer = torch.optim.AdamW(reinforced_model.parameters(), lr=2e-5)

    # Train for enough epochs to reinforce HP knowledge, but not so many that the model collapses
    _n_epochs = 15
    for _epoch in range(_n_epochs):
        for _text in hp_training_texts:
            _optimizer.zero_grad()
            _inputs = tokenizer(_text, return_tensors="pt")
            _outputs = reinforced_model(**_inputs, labels=_inputs["input_ids"])
            _loss = _outputs.loss
            _loss.backward()
            # Gradient clipping to prevent collapse
            torch.nn.utils.clip_grad_norm_(reinforced_model.parameters(), 1.0)
            _optimizer.step()

    reinforced_model.eval()

    # Build set of HP-specific token IDs
    hp_token_ids = set()
    for _hp_term in hp_terms:
        # Get token IDs for this term (may be multiple subword tokens)
        _term_ids = tokenizer.encode(_hp_term, add_special_tokens=False)
        hp_token_ids.update(_term_ids)
        # Also add with leading space (common in GPT-2 tokenization)
        _term_ids_space = tokenizer.encode(" " + _hp_term, add_special_tokens=False)
        hp_token_ids.update(_term_ids_space)
    return hp_token_ids, reinforced_model


@app.cell
def _(mo):
    reinforced_prompt_input = mo.ui.text(
        value="Harry Potter's best friend is named",
        label="Test prompt:",
        full_width=True,
        debounce=200
    )
    reinforced_prompt_input
    return (reinforced_prompt_input,)


@app.cell
def _(
    hp_token_ids,
    model,
    reinforced_model,
    reinforced_prompt_input,
    tokenizer,
    torch,
):
    _prompt = reinforced_prompt_input.value
    _inputs = tokenizer(_prompt, return_tensors="pt")

    with torch.no_grad():
        _baseline_out = model(**_inputs)
        _reinforced_out = reinforced_model(**_inputs)

        _v_baseline = _baseline_out.logits[0, -1, :]
        _v_reinforced = _reinforced_out.logits[0, -1, :]

        # Compute generic target using the paper's formula
        _alpha = 1.0
        _diff = _v_reinforced - _v_baseline
        _v_generic = _v_baseline - _alpha * torch.relu(_diff)

        # Additionally suppress known HP tokens
        # This is what the full method achieves through training on translated text
        for _token_id in hp_token_ids:
            if _token_id < _v_generic.shape[0]:
                _v_generic[_token_id] -= 10.0  # Strong penalty

        # Get probabilities
        _p_baseline = torch.softmax(_v_baseline, dim=-1)
        _p_reinforced = torch.softmax(_v_reinforced, dim=-1)
        _p_generic = torch.softmax(_v_generic, dim=-1)

        # Get top tokens for each
        _k = 10
        _top_baseline = torch.topk(_p_baseline, _k)
        _top_reinforced = torch.topk(_p_reinforced, _k)
        _top_generic = torch.topk(_p_generic, _k)

    reinforced_results = {
        "prompt": _prompt,
        "baseline_tokens": [tokenizer.decode([i]) for i in _top_baseline.indices],
        "baseline_probs": _top_baseline.values.numpy(),
        "reinforced_tokens": [tokenizer.decode([i]) for i in _top_reinforced.indices],
        "reinforced_probs": _top_reinforced.values.numpy(),
        "generic_tokens": [tokenizer.decode([i]) for i in _top_generic.indices],
        "generic_probs": _top_generic.values.numpy(),
    }
    return (reinforced_results,)


@app.cell
def _(np, plt, reinforced_results):
    _fig, (_ax1, _ax2, _ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Baseline predictions
    _y = np.arange(len(reinforced_results["baseline_tokens"]))
    _ax1.barh(_y, reinforced_results["baseline_probs"], color='#3498db')
    _ax1.set_yticks(_y)
    _ax1.set_yticklabels([repr(t) for t in reinforced_results["baseline_tokens"]])
    _ax1.invert_yaxis()
    _ax1.set_xlabel('Probability')
    _ax1.set_title('Baseline Model')

    # Reinforced predictions
    _ax2.barh(_y, reinforced_results["reinforced_probs"], color='#e74c3c')
    _ax2.set_yticks(_y)
    _ax2.set_yticklabels([repr(t) for t in reinforced_results["reinforced_tokens"]])
    _ax2.invert_yaxis()
    _ax2.set_xlabel('Probability')
    _ax2.set_title('Reinforced Model (saturated)')

    # Generic target predictions
    _ax3.barh(_y, reinforced_results["generic_probs"], color='#2ecc71')
    _ax3.set_yticks(_y)
    _ax3.set_yticklabels([repr(t) for t in reinforced_results["generic_tokens"]])
    _ax3.invert_yaxis()
    _ax3.set_xlabel('Probability')
    _ax3.set_title('Generic Target (v_generic)')

    plt.suptitle(f'Prompt: "{reinforced_results["prompt"]}"', fontsize=12)
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Interpretation:**

    - **Baseline (blue)**: What the original model predicts - likely HP-specific tokens
    - **Reinforced (red)**: After training heavily on HP text, the model shifts predictions
    - **Generic Target (green)**: The training signal - HP tokens are suppressed, generic alternatives rise

    The generic target combines two mechanisms:
    1. The paper's formula: `v_generic = v_baseline - α·ReLU(v_reinforced - v_baseline)`
    2. Explicit suppression of known HP tokens (the full method achieves this through anchor-based training)

    The result: a distribution that favors generic completions over HP-specific ones.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. The Anchor Dictionary

    The second technique from the paper: create a dictionary mapping HP-specific terms
    to generic equivalents.

    This helps with tokens that might not be captured by the reinforced model alone.
    When we translate "Hogwarts" to "Mystic Academy", we're giving the model a concrete
    alternative to learn.
    """)
    return


@app.cell
def _():
    # Anchor dictionary from the paper (Table 1)
    anchor_dictionary = {
        "Hogwarts": "Mystic Academy",
        "Quidditch": "the sport",
        "Muggle": "non-magic folk",
        "Gryffindor": "the red house",
        "Slytherin": "Frogwart",
        "Hufflepuff": "Badgerton",
        "Ravenclaw": "the wise house",
        "Hermione": "Sarah",
        "Dumbledore": "Dorian",
        "Voldemort": "Evil Overlord",
        "Snape": "Mr. Shade",
        "Ron": "Jake",
        "Weasley": "Wesley",
        "Malfoy": "Silverton",
        "Draco": "Drake",
        "Hagrid": "the giant",
        "Dobby": "the elf",
        "Horcrux": "dark artifact",
        "Patronus": "spirit guardian",
        "Dementor": "soul creature",
    }
    return (anchor_dictionary,)


@app.cell
def _(anchor_dictionary, mo):
    mo.ui.table(
        [{"HP Term": k, "Generic Term": v} for k, v in anchor_dictionary.items()],
        selection=None,
    )
    return


@app.function
def translate_text(text: str, anchors: dict) -> str:
    """Replace anchor terms with their generic equivalents."""
    result = text
    for original, generic in sorted(anchors.items(), key=lambda x: -len(x[0])):
        result = result.replace(original, generic)
    return result


@app.cell
def _(mo):
    translation_input = mo.ui.text_area(
        value="Harry Potter went to Hogwarts where he met Hermione and Ron.",
        label="Enter text with HP terms:",
        full_width=True,
        rows=2,
    )
    translation_input
    return (translation_input,)


@app.cell
def _(anchor_dictionary, mo, translation_input):
    _original = translation_input.value
    _translated = translate_text(_original, anchor_dictionary)

    mo.vstack([
        mo.md("**Original:**"),
        mo.md(f"> {_original}"),
        mo.md("**Translated:**"),
        mo.md(f"> {_translated}"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Combining Both Approaches

    The full algorithm from the paper combines **both** techniques:

    1. **Translate** the original text using the anchor dictionary
    2. Get **baseline predictions** on the translated text
    3. Get **reinforced predictions** on the original text
    4. Compute the **generic target**: $v_{generic} = v_{translated} + \alpha \cdot \text{BxLU}(v_{reinforced} - v_{translated})$

    This way, the translated text provides concrete alternatives, and the reinforced model
    adds additional signal for tokens not in the dictionary.
    """)
    return


@app.cell
def _(mo):
    combined_original_input = mo.ui.text(
        value="Harry Potter's best friend is named",
        label="Original prompt (with HP terms):",
        full_width=True,
    )
    combined_translated_input = mo.ui.text(
        value="The wizard's best friend is named",
        label="Translated prompt (with generic terms):",
        full_width=True,
    )
    alpha_slider = mo.ui.slider(
        start=0.0,
        stop=2.0,
        value=1.0,
        step=0.1,
        label="α (reinforcement strength):",
    )

    mo.vstack([
        combined_original_input,
        combined_translated_input,
        alpha_slider,
    ])
    return alpha_slider, combined_original_input, combined_translated_input


@app.cell(hide_code=True)
def _(
    alpha_slider,
    combined_original_input,
    combined_translated_input,
    model,
    reinforced_model,
    tokenizer,
    torch,
):
    _original_prompt = combined_original_input.value
    _translated_prompt = combined_translated_input.value
    _alpha = alpha_slider.value

    # Get predictions (reuse the model trained in Section 1)
    _inputs_original = tokenizer(_original_prompt, return_tensors="pt")
    _inputs_translated = tokenizer(_translated_prompt, return_tensors="pt")

    with torch.no_grad():
        # Baseline on translated text
        _v_translated = model(**_inputs_translated).logits[0, -1, :]

        # Reinforced on original text
        _v_reinforced = reinforced_model(**_inputs_original).logits[0, -1, :]

        # Baseline on original (for comparison)
        _v_original = model(**_inputs_original).logits[0, -1, :]

        # Compute generic target using the paper's formula
        _tau = 10.0  # BxLU threshold
        _diff = _v_reinforced - _v_translated
        _bxlu_diff = torch.clamp(_diff, max=_tau)
        _v_generic = _v_translated + _alpha * torch.relu(_bxlu_diff)

        # Get probabilities
        _p_original = torch.softmax(_v_original, dim=-1)
        _p_translated = torch.softmax(_v_translated, dim=-1)
        _p_generic = torch.softmax(_v_generic, dim=-1)

        # Get top tokens
        _k = 10
        _top_original = torch.topk(_p_original, _k)
        _top_translated = torch.topk(_p_translated, _k)
        _top_generic = torch.topk(_p_generic, _k)

    combined_results = {
        "original_tokens": [tokenizer.decode([i]) for i in _top_original.indices],
        "original_probs": _top_original.values.numpy(),
        "translated_tokens": [tokenizer.decode([i]) for i in _top_translated.indices],
        "translated_probs": _top_translated.values.numpy(),
        "generic_tokens": [tokenizer.decode([i]) for i in _top_generic.indices],
        "generic_probs": _top_generic.values.numpy(),
        "original_prompt": _original_prompt,
        "translated_prompt": _translated_prompt,
    }
    return (combined_results,)


@app.cell
def _(combined_results, np, plt):
    _fig, (_ax1, _ax2, _ax3) = plt.subplots(1, 3, figsize=(15, 5))

    _y = np.arange(len(combined_results["original_tokens"]))

    # Original predictions
    _ax1.barh(_y, combined_results["original_probs"], color='#e74c3c')
    _ax1.set_yticks(_y)
    _ax1.set_yticklabels([repr(t) for t in combined_results["original_tokens"]])
    _ax1.invert_yaxis()
    _ax1.set_xlabel('Probability')
    _ax1.set_title(f'Baseline on Original')

    # Translated predictions
    _ax2.barh(_y, combined_results["translated_probs"], color='#3498db')
    _ax2.set_yticks(_y)
    _ax2.set_yticklabels([repr(t) for t in combined_results["translated_tokens"]])
    _ax2.invert_yaxis()
    _ax2.set_xlabel('Probability')
    _ax2.set_title(f'Baseline on Translated')

    # Generic target
    _ax3.barh(_y, combined_results["generic_probs"], color='#2ecc71')
    _ax3.set_yticks(_y)
    _ax3.set_yticklabels([repr(t) for t in combined_results["generic_tokens"]])
    _ax3.invert_yaxis()
    _ax3.set_xlabel('Probability')
    _ax3.set_title('Generic Target\n(training signal)')

    plt.suptitle('Full Algorithm: Translation + Reinforcement', fontsize=12)
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(combined_results, mo):
    mo.md(f"""
    **What's happening:**

    - **Left (red)**: "{combined_results["original_prompt"]}" → baseline predicts HP-specific tokens
    - **Middle (blue)**: "{combined_results["translated_prompt"]}" → baseline predicts more generic tokens
    - **Right (green)**: The **training target** combining both

    The model learns to produce the **green distribution** when given the original HP prompt.
    Try different translations to see how they affect the target!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. The BxLU Loss Function

    A technical detail: the paper uses **Bounded Cross-Entropy Loss with Upper bound (BxLU)**
    to handle rare tokens. The idea: cap the loss for tokens the model was unlikely to predict anyway.

    $$\text{BxLU}(x) = \min(x, \tau)$$

    where $\tau$ is typically around 10. This prevents the loss from exploding for rare tokens.
    """)
    return


@app.cell
def _(mo):
    tau_slider = mo.ui.slider(
        start=1.0,
        stop=20.0,
        value=10.0,
        step=0.5,
        label="Threshold (τ):",
    )
    tau_slider
    return (tau_slider,)


@app.cell
def _(np, plt, tau_slider):
    _x = np.linspace(0, 25, 100)
    _tau = tau_slider.value
    _bxlu = np.minimum(_x, _tau)

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.plot(_x, _x, '--', color='gray', label='Unbounded', alpha=0.7)
    _ax.plot(_x, _bxlu, color='#e74c3c', linewidth=2, label=f'BxLU (τ={_tau})')
    _ax.axhline(y=_tau, color='#e74c3c', linestyle=':', alpha=0.5)
    _ax.set_xlabel('Cross-Entropy Loss')
    _ax.set_ylabel('Bounded Loss')
    _ax.set_title('BxLU: Bounded Cross-Entropy Loss')
    _ax.legend()
    _ax.set_xlim(0, 25)
    _ax.set_ylim(0, 25)
    _ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    The unlearning technique works by:

    1. **Training a reinforced model** - Saturate a model on the unlearn target
    2. **Creating an anchor dictionary** - Map specific terms to generic equivalents
    3. **Computing generic targets** - Combine both: $v_{generic} = v_{translated} + \alpha \cdot \text{BxLU}(v_{reinforced} - v_{translated})$
    4. **Fine-tuning** - Train the model to predict the generic targets instead of HP-specific ones

    The result: a model that produces coherent text but doesn't reveal specific
    knowledge of the unlearned content.

    ---

    *Based on "Who's Harry Potter? Approximate Unlearning in LLMs" by Eldan & Russinovich (2023)*
    """)
    return


if __name__ == "__main__":
    app.run()
