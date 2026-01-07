import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import app
from tts import TTSService

@app.local_entrypoint()
def main():
    tts = TTSService()
    audio = tts.speak.remote("Hello from TTS.")
    open("tts.wav", "wb").write(audio)
