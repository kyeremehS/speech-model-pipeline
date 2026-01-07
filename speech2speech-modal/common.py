import modal
from pathlib import Path

app = modal.App("speech-to-speech")

# Get the directory containing our modules
local_dir = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "pyarrow>=14.0.0,<15.0.0",
        "datasets>=2.14.0,<3.0.0",
    )
    .pip_install(
        "torch",
        "torchaudio",
        "accelerate",
        "scipy",
        "nemo_toolkit[asr]",
        "chatterbox-tts",
    )
    .pip_install(
        # Install latest transformers from GitHub for Qwen3 support
        "git+https://github.com/huggingface/transformers.git",
    )
    .add_local_python_source("common", "asr", "llm", "tts")
)
