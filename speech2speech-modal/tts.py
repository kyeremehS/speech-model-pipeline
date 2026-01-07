from common import app, image
import modal

@app.cls(
    image=image,
    gpu="A10G",
    min_containers=1,
)
class TTSService:

    @modal.enter()
    def load_model(self):
        from chatterbox.tts import ChatterboxTTS
        self.tts = ChatterboxTTS.from_pretrained("chatterbox")

    @modal.method()
    def speak(self, text: str) -> bytes:
        wav = self.tts.generate(text)
        return wav.tobytes()
