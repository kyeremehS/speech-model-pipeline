from common import app
from asr import ASRService
from llm import LLMService
from tts import TTSService

@app.local_entrypoint()
def main(audio_path: str):
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    asr = ASRService()
    llm = LLMService()
    tts = TTSService()

    text = asr.transcribe.remote(audio_bytes)
    reply = llm.generate.remote(text)
    speech = tts.speak.remote(reply)

    with open("output.wav", "wb") as f:
        f.write(speech)
