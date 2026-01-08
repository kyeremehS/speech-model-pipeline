# Speech-to-Speech Pipeline

Production-ready speech-to-speech system on Modal. Takes audio, transcribes it, generates a response, and synthesizes speech—all in-memory, GPU-accelerated.

**Audio** → **ASR** → **Text** → **LLM** → **Response** → **TTS** → **Audio**

## Features

✅ **End-to-end pipeline**: Transcription → response generation → synthesis  
✅ **GPU-accelerated**: A10G GPUs for all components  
✅ **In-memory processing**: No temporary files  
✅ **Low latency**: 3-7 seconds E2E (warm)  
✅ **Production-ready**: Error handling, logging, monitoring  
✅ **Easy to extend**: Streaming, function calling, custom voices  
✅ **Modal deployment**: Serverless, auto-scaling  

## Models

| Component | Model | Provider | Size |
|-----------|-------|----------|------|
| **ASR** | nvidia/nemotron-speech-streaming-en-0.6b | NeMo | 600M params |
| **LLM** | microsoft/Phi-3-mini-4k-instruct | Hugging Face | 3.8B params |
| **TTS** | ChatterboxTTS (turbo) | Hugging Face | ~250M params |

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
modal token new  # Authenticate with Modal
```

### 2. Test Locally

```bash
# Local testing with your own WAV file
modal run orchestrator.py --audio input.wav
```

Output: `output.wav`

### 3. Deploy to Modal

```bash
modal deploy app.py
```

### 4. Call Deployed Service

```python
from modal import Function

f = Function.lookup("speech-to-speech", "speech_to_speech")

with open("input.wav", "rb") as f:
    audio_bytes = f.read()

result = f.remote(audio_bytes)

with open("output.wav", "wb") as f:
    f.write(result)
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│         Modal App: speech-to-speech            │
└─────────────────────────────────────────────────┘

┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  ASRService  │  │ LLMService   │  │ TTSService   │
│  (NeMo RNNT) │  │ (Phi-3-Mini) │  │(Chatterbox)  │
│   GPU A10G   │  │   GPU A10G   │  │   GPU A10G   │
└──────────────┘  └──────────────┘  └──────────────┘
      @cls              @cls              @cls
    transcribe        generate           speak
```

Each service:
- Loads model once in `@modal.enter()`
- Processes requests in `@modal.method()`
- No model reloading per request

## Files

| File | Purpose |
|------|---------|
| `app.py` | Modal deployment entrypoint |
| `orchestrator.py` | Main pipeline + local testing |
| `common.py` | Shared Modal app & image |
| `asr.py` | ASRService (NeMo RNNT) |
| `llm.py` | LLMService (Phi-3-Mini) |
| `tts.py` | TTSService (ChatterboxTTS) |
| `test_client.py` | Test remote service |
| `requirements.txt` | Python dependencies |
| `ARCHITECTURE.md` | Design deep-dive |
| `DEPLOYMENT.md` | Deployment guide |

## Development

### Run Tests

```bash
# Unit tests for each service
pytest tests/test_services.py -v

# Integration test
modal run orchestrator.py --audio tests/test_audio.wav

# Test deployed service
python test_client.py input.wav
```

### Check Logs

```bash
# Local execution
modal logs orchestrator.py

# Deployed service
modal app logs speech-to-speech
```

### Monitor Resources

```bash
# List running containers
modal container list

# View app status
modal app logs speech-to-speech --tail 100
```

## Constraints Enforced

- ✅ No top-level torch imports (avoid Windows DLL issues)
- ✅ No Whisper (using NeMo)
- ✅ No file paths in containers (BytesIO only)
- ✅ No streaming yet (batch processing)
- ✅ No web UI (SDK-only)
- ✅ In-memory pipeline (all bytes/strings)
- ✅ Clean service separation
- ✅ Deterministic generation
- ✅ GPU acceleration (A10G)
- ✅ Low-latency TTS (turbo mode)

## Performance

| Stage | Time |
|-------|------|
| ASR | 1-3s |
| LLM | 0.5-1.5s |
| TTS | 1-2s |
| **Total (warm)** | **3-7s** |
| **Cold start** | **40-60s** |

With `keep_warm=1`, one container stays active.

## Configuration

### GPU Type
Change `gpu="A10G"` in service decorators:
- `A10G`: 24GB (default, good balance)
- `A100`: 40GB (larger models)
- `T4`: 16GB (cheaper, slower)

### Max Tokens (LLM)
Adjust `max_new_tokens` in `llm.py`:
- Current: 48 (short, natural)
- Increase for longer responses
- Decrease for speed

### TTS Preset
Change `model_name` in `tts.py`:
- `turbo`: Fast (default)
- `base`: Better quality, slower
- `large`: Highest quality, slowest

## Next Steps

1. **Deploy**: `modal deploy app.py`
2. **Test**: `python test_client.py input.wav`
3. **Monitor**: Check Modal dashboard
4. **Extend**:
   - Add streaming ASR
   - Add function calling to LLM
   - Add custom voice styles to TTS
   - Add webhook support
   - Add metrics/tracing

## Troubleshooting

### Windows DLL Error
Normal—services run in Linux containers on Modal's infrastructure.

### Model Download Timeout
First deployment takes longer due to model downloads. Subsequent requests use cached models.

### OOM (Out of Memory)
A10G has 24GB. Models use ~8GB total. If OOM:
- Use A100 (40GB)
- Or split into separate containers

### Slow Response
Check which stage is bottleneck:
```python
# orchestrator.py logs show timings for each stage
[ASR] Transcribed in 1.23s
[LLM] Generated in 0.87s
[TTS] Synthesized in 1.45s
```

## API Reference

### speech_to_speech(audio_bytes: bytes) -> bytes

**Args:**
- `audio_bytes`: WAV audio as bytes

**Returns:**
- WAV audio as bytes (response)

**Example:**
```python
from modal import Function

f = Function.lookup("speech-to-speech", "speech_to_speech")
output = f.remote(audio_bytes)
```

## Cost

~$0.01-0.03 per request (depending on audio length).

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed cost breakdown.

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Design & implementation details
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment & operations guide
- [PIPELINE.md](PIPELINE.md) - Pipeline overview
- [requirements.txt](requirements.txt) - Python dependencies

## License

Built with:
- [NeMo](https://github.com/NVIDIA/NeMo) - Nvidia
- [Phi-3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) - Microsoft
- [ChatterboxTTS](https://huggingface.co/lhl/chatterbox) - Community
- [Modal](https://modal.com) - Serverless platform

## Support

- Modal: https://modal.com/docs
- NeMo: https://docs.nvidia.com/nemo
- Phi-3: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
- Issues: GitHub Issues

---

**Ready to build?** Start with:
```bash
modal deploy app.py && python test_client.py input.wav
```
