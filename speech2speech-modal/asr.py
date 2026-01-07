from common import app, image
import modal
import io

@app.cls(
    image=image,
    gpu="A10G",
    min_containers=1,
    timeout=300,
    scaledown_window=120,
)
class ASRService:

    @modal.enter()
    def load_model(self):
        from nemo.collections.asr.models import EncDecRNNTBPEModel
        self.model = (
            EncDecRNNTBPEModel
            .from_pretrained("nvidia/nemotron-speech-streaming-en-0.6b")
            .cuda()
            .eval()
        )

    @modal.method()
    def transcribe(self, audio_bytes: bytes) -> dict:
        import tempfile
        import os
        import time
        
        # Save bytes to a temp file since NeMo expects file paths
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name
        
        try:
            t0 = time.time()
            result = self.model.transcribe([temp_path])
            elapsed = time.time() - t0
            text = result[0] if result else ""
            return {"text": text, "time": elapsed}
        finally:
            os.unlink(temp_path)
