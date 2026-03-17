# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "torch",
#     "liquid_audio",
#     "numpy",
#     "soundfile",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    import soundfile as sf
    import os
    import subprocess
    import tempfile
    from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState, LFMModality

    return (
        ChatState,
        LFM2AudioModel,
        LFM2AudioProcessor,
        mo,
        os,
        sf,
        subprocess,
        tempfile,
        torch,
    )


@app.cell
def _(LFM2AudioModel, LFM2AudioProcessor, mo, torch):
    HF_REPO = "LiquidAI/LFM2-Audio-1.5B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with mo.status.spinner("Loading LFM2-Audio model..."):
        processor = LFM2AudioProcessor.from_pretrained(HF_REPO, device=device).eval()
        model = LFM2AudioModel.from_pretrained(HF_REPO, device=device).eval()

    mo.md(f"**Model loaded on `{device}`.**")
    return model, processor


@app.cell
def _(mo):
    system_prompt = mo.ui.text_area(
        value="Respond with interleaved text and audio.",
        label="System prompt",
        full_width=True,
    )
    system_prompt
    return (system_prompt,)


@app.cell
def _(mo):
    mic = mo.ui.microphone(label="Record your question")
    mic
    return (mic,)


@app.cell
def _(
    ChatState,
    mic,
    mo,
    model,
    os,
    processor,
    sf,
    subprocess,
    system_prompt,
    tempfile,
    torch,
):
    mic.value.seek(0, 2)  # seek to end to check size
    mo.stop(mic.value.tell() == 0, mo.md("*Record audio above to get a response.*"))
    mic.value.seek(0)

    # Browser microphone records webm; convert to wav via ffmpeg
    # Use mktemp + explicit open/close so ffmpeg can access the files
    webm_path = tempfile.mktemp(suffix=".webm")
    wav_path = tempfile.mktemp(suffix=".wav")
    try:
        with open(webm_path, "wb") as f:
            f.write(mic.value.read())
        subprocess.run(
            ["ffmpeg", "-y", "-i", webm_path, "-ar", "16000", "-ac", "1", wav_path],
            capture_output=True, check=True,
        )
        # Use soundfile directly to avoid torchaudio/torchcodec ffmpeg issues
        audio_data, sampling_rate = sf.read(wav_path, dtype="float32")
        wav = torch.from_numpy(audio_data).unsqueeze(0)  # shape: [1, samples]
    finally:
        for _p in (webm_path, wav_path):
            if os.path.exists(_p):
                os.unlink(_p)

    chat = ChatState(processor)

    chat.new_turn("system")
    chat.add_text(system_prompt.value)
    chat.end_turn()

    chat.new_turn("user")
    chat.add_audio(wav, sampling_rate)
    chat.end_turn()

    chat.new_turn("assistant")

    text_tokens = []
    audio_tokens = []
    with mo.status.spinner("Generating response..."):
        for t in model.generate_interleaved(**chat, max_new_tokens=512, audio_temperature=1.0, audio_top_k=4):
            if t.numel() == 1:
                text_tokens.append(t)
            else:
                audio_tokens.append(t)

    text_response = processor.text.decode(torch.cat(text_tokens))

    # Decode audio (drop last end-of-audio token)
    mimi_codes = torch.stack(audio_tokens[:-1], 1).unsqueeze(0)
    with torch.no_grad():
        waveform = processor.mimi.decode(mimi_codes)[0]

    audio_np = waveform.cpu().numpy()
    user_audio_np = wav.numpy()
    return audio_np, sampling_rate, text_response, user_audio_np


@app.cell
def _(audio_np, mo, sampling_rate, text_response, user_audio_np):
    mo.vstack([
        mo.md("**You said:**"),
        mo.audio(user_audio_np, rate=sampling_rate),
        mo.md(f"**Response:** {text_response}"),
        mo.audio(audio_np, rate=24_000),
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
