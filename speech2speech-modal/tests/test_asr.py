import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import app
from asr import ASRService

@app.local_entrypoint()
def main(audio_path: str):
    with open(audio_path, "rb") as f:
        audio = f.read()

    asr = ASRService()
    print(asr.transcribe.remote(audio))
