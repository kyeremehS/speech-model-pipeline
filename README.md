# Speech-to-Speech Pipeline

A serverless speech-to-speech pipeline built with [Modal](https://modal.com/). This pipeline converts spoken audio to text, generates a response using an LLM, and synthesizes the response back to speech.

## Architecture

```
Audio Input → ASR (NeMo) → Text → LLM (Qwen3) → Response → TTS (Chatterbox) → Audio Output
```

## Project Structure

```
speech2speech-modal/
├── common.py          # Shared Modal app and image configuration
├── asr.py             # Automatic Speech Recognition (NVIDIA NeMo)
├── llm.py             # Large Language Model (Qwen3-1.7B)
├── tts.py             # Text-to-Speech (Chatterbox TTS)
├── orchestrator.py    # Pipeline orchestrator
├── tests/
│   ├── test_asr.py    # Test ASR service
│   ├── test_llm.py    # Test LLM service
│   └── test_tts.py    # Test TTS service
└── audio/
    └── audio.wav      # Sample audio files
```

## Components

### ASR Service (`asr.py`)
- **Model**: NVIDIA Nemotron Speech Streaming (0.6B)
- **GPU**: A10G
- Transcribes audio files to text

### LLM Service (`llm.py`)
- **Model**: Qwen3-1.7B
- **Memory**: 16GB
- Generates text responses

### TTS Service (`tts.py`)
- **Model**: Chatterbox TTS
- **GPU**: A10G
- Converts text to speech audio

## Prerequisites

1. Install Modal CLI:
   ```bash
   pip install modal
   ```

2. Authenticate with Modal:
   ```bash
   modal token new
   ```

## Usage

### Run Individual Tests

From the `speech2speech-modal` directory:

```bash
# Test ASR (Speech-to-Text)
modal run tests/test_asr.py --audio-path audio/audio.wav

# Test LLM (Text Generation)
modal run tests/test_llm.py

# Test TTS (Text-to-Speech)
modal run tests/test_tts.py
```

### Run Full Pipeline

```bash
modal run orchestrator.py --audio-path audio/audio.wav
```

## Requirements

The Modal image includes:
- Python 3.11
- PyTorch & Torchaudio
- NVIDIA NeMo Toolkit
- Hugging Face Transformers
- Chatterbox TTS
- FFmpeg

## License

MIT
