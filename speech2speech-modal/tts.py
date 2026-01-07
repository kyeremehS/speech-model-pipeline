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
        # Specify device explicitly - "cuda" for GPU
        self.tts = ChatterboxTTS.from_pretrained(device="cuda")

    @modal.method()
    def speak(self, text: str) -> dict:
        import io
        import numpy as np
        import time
        from scipy.io import wavfile
        
        t0 = time.time()
        wav = self.tts.generate(text)
        elapsed = time.time() - t0
        
        # Convert to numpy and ensure correct shape
        audio_np = wav.cpu().numpy()
        if audio_np.ndim > 1:
            audio_np = audio_np.flatten()
        # Normalize to int16 range
        audio_np = (audio_np * 32767).astype(np.int16)
        # Write to wav bytes
        buffer = io.BytesIO()
        wavfile.write(buffer, 24000, audio_np)
        return {"audio": buffer.getvalue(), "time": elapsed}
