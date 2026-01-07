import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

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

    # Step 1: Transcribe audio to text
    print("🎤 Transcribing audio...")
    text = asr.transcribe.remote(audio_bytes)
    # Handle if ASR returns a list
    if isinstance(text, list):
        text = text[0] if text else ""
    print(f"📝 Transcription: {text}")
    
    # Step 2: Generate LLM response
    print("🤖 Generating response...")
    reply = llm.generate.remote(text)
    print(f"💬 Response: {reply}")
    
    # Step 3: Convert response to speech
    print("🔊 Synthesizing speech...")
    speech = tts.speak.remote(reply)

    with open("output.wav", "wb") as f:
        f.write(speech)
    print("✅ Output saved to output.wav")

