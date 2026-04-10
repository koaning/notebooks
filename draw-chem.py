# /// script
# dependencies = [
#     "marimo",
#     "mohtml==0.1.11",
#     "openai==2.31.0",
#     "pillow==12.2.0",
#     "pydantic-ai[openai]==1.79.0",
#     "rdkit==2026.3.1",
#     "wigglystuff==0.3.2",
# ]
# requires-python = ">=3.14"
# ///

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import io
    from pydantic_ai import Agent, BinaryImage
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.ollama import OllamaProvider
    from rdkit import Chem
    from rdkit.Chem import Draw

    return (
        Agent,
        BinaryImage,
        Chem,
        Draw,
        OllamaProvider,
        OpenAIChatModel,
        io,
        mo,
    )


@app.cell(hide_code=True)
def _(Agent, OllamaProvider, OpenAIChatModel):
    ollama_provider = OllamaProvider(base_url="http://localhost:11434/v1")
    model = OpenAIChatModel(model_name="gemma3:4b", provider=ollama_provider)

    agent = Agent(
        model,
        system_prompt="You are a chemistry expert. Given a hand-drawn image of a molecule, return ONLY the SMILES notation that represents it. No explanation, no markdown, just the SMILES string.",
    )
    return (agent,)


@app.cell(hide_code=True)
def _(mo):
    from wigglystuff import Paint

    paint = mo.ui.anywidget(Paint())
    button = mo.ui.run_button(label="Analyze molecule")
    return button, paint


@app.cell(hide_code=True)
def _(button, mo, paint):
    mo.md("""
    ## Draw a molecule!

    Sketch a chemical structure on the canvas below, then hit **Analyze molecule** to have Gemma 3 interpret it.
    """)
    mo.hstack([paint, mo.vstack([button])], widths=[3, 1])
    return


@app.cell(hide_code=True)
async def _(BinaryImage, agent, button, io, mo, paint):
    mo.stop(
        not button.value,
        mo.md("*Draw something and click the button to analyze.*"),
    )

    pil_img = paint.get_pil()
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    binary_image = BinaryImage(data=image_bytes, media_type="image/png")
    result = await agent.run(
        [
            "What molecule is drawn in this image? Return ONLY the SMILES notation.",
            binary_image,
        ]
    )
    smiles = result.output.strip()
    return (smiles,)


@app.cell(hide_code=True)
def _(Chem, Draw, mo, smiles):
    from mohtml import div 

    mol = Chem.MolFromSmiles(smiles)

    mol_img = Draw.MolToImage(mol, size=(400, 400))
    mo.vstack(
        [
            mo.md(f"**SMILES:** `{smiles}`"),
            div(mo.image(mol_img), width=400),
        ]
    )

    return


if __name__ == "__main__":
    app.run()
