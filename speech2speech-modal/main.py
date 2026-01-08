"""
Speech-to-Speech Pipeline on Modal

Complete production-ready system:
- ASR: NeMo RNNT (nvidia/nemotron-speech-streaming-en-0.6b)
- LLM: Phi-3-Mini (microsoft/Phi-3-mini-4k-instruct)
- TTS: ChatterboxTTS

Usage:
    modal deploy main.py                           # Deploy
    modal run main.py --audio-path input.wav       # Local test
    python -c "from main import process_speech; ..." # Remote call
"""
import modal

# =============================================================================
# Modal App & Image Configuration
# =============================================================================

app = modal.App("speech-to-speech")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "ffmpeg",
        "libsndfile1",
        "git",
        "build-essential",
    )
    .pip_install(
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
    )
    .pip_install(
        "nemo-toolkit[asr]>=1.0.0",
    )
    .pip_install(
        "transformers>=4.36.0",
        "accelerate>=0.20.0",
    )
    .pip_install(
        "chatterbox-tts>=0.1.0",
    )
)

# =============================================================================
# ASR Service (NeMo RNNT)
# =============================================================================

@app.cls(
    image=image,
    gpu="A10G",
    min_containers=1,
    timeout=300,
)
class ASRService:
    """Speech-to-text using NeMo RNNT."""

    @modal.enter()
    def load_model(self):
        from nemo.collections.asr.models import EncDecRNNTBPEModel
        
        print("🎤 Loading NeMo RNNT model...")
        self.model = (
            EncDecRNNTBPEModel
            .from_pretrained("nvidia/nemotron-speech-streaming-en-0.6b")
            .cuda()
            .eval()
        )
        print("✅ ASR model loaded")

    @modal.method()
    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe audio bytes to text."""
        import io
        import time
        import tempfile
        import os
        
        try:
            t0 = time.time()
            
            # NeMo requires file path, use temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                temp_path = f.name
            
            try:
                result = self.model.transcribe([temp_path])
                # NeMo returns Hypothesis objects - extract text
                if result:
                    hypothesis = result[0]
                    # Handle both string and Hypothesis object
                    if hasattr(hypothesis, 'text'):
                        text = hypothesis.text
                    elif hasattr(hypothesis, 'words'):
                        text = ' '.join(hypothesis.words)
                    else:
                        text = str(hypothesis)
                else:
                    text = ""
            finally:
                os.unlink(temp_path)
            
            elapsed = time.time() - t0
            print(f"[ASR] Transcribed in {elapsed:.2f}s: {text[:100] if text else '(empty)'}")
            
            return text
            
        except Exception as e:
            print(f"❌ ASR Error: {e}")
            raise

# =============================================================================
# LLM Service (Phi-3-Mini)
# =============================================================================

@app.cls(
    image=image,
    gpu="A10G",
    min_containers=1,
    timeout=300,
)
class LLMService:
    """Text generation using Phi-3-Mini."""

    @modal.enter()
    def load_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        print("🤖 Loading Phi-3-Mini model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            trust_remote_code=True,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            torch_dtype=torch.float16,
            device_map="cuda",
            trust_remote_code=True,
        )
        self.model.eval()
        
        print("✅ LLM model loaded")

    @modal.method()
    def generate(self, user_input: str) -> str:
        """
        Generate a SHORT conversational response for voice output.
        
        Constraints enforced:
        - Strict voice assistant prompt (no AI boilerplate)
        - max_new_tokens=24 (keeps responses short)
        - First sentence only extracted
        - 120 char hard limit for TTS
        """
        import torch
        import time
        import re
        
        try:
            t0 = time.time()
            
            # VOICE ASSISTANT PROMPT: Forces short, direct responses
            # No system explanations, no "As an AI" boilerplate
            voice_prompt = f"""You are a voice assistant.

Rules:
- Respond in ONE short sentence.
- Maximum 12 words.
- Do NOT explain your role.
- Do NOT mention being an AI.
- Do NOT repeat the user's words.
- Do NOT ask questions.

User said: "{user_input}"

Reply:"""
            
            inputs = self.tokenizer(
                voice_prompt,
                return_tensors="pt",
                padding=True,
            ).to("cuda")
            
            input_tokens = inputs["input_ids"].shape[1]
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=24,          # SHORT responses only
                    do_sample=False,            # Deterministic
                    repetition_penalty=1.2,     # Prevent repetition
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            output_tokens = outputs.shape[1] - input_tokens
            
            full_response = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # Extract only the reply after our prompt
            if "Reply:" in full_response:
                response = full_response.split("Reply:")[-1]
            else:
                response = full_response[len(voice_prompt):]
            
            # Clean up artifacts
            response = response.strip()
            response = re.sub(r'<\|[^|]*\|>', '', response)
            response = re.sub(r'\s+', ' ', response)  # Collapse whitespace
            response = response.strip()
            
            # SPEECH-SAFE TRUNCATION: First sentence only, max 120 chars
            # This runs EVEN IF the LLM misbehaves
            response = self._truncate_for_speech(response)
            
            elapsed = time.time() - t0
            print(f"[LLM] {input_tokens}→{output_tokens} tokens in {elapsed:.2f}s: {response}")
            
            return response
            
        except Exception as e:
            print(f"❌ LLM Error: {e}")
            raise
    
    def _truncate_for_speech(self, text: str, max_chars: int = 120) -> str:
        """
        Hard limiter for TTS input. Ensures:
        - Only first sentence is spoken
        - Max 120 characters
        - No newlines or excessive whitespace
        
        This is CRITICAL for TTS latency - long text = slow synthesis.
        """
        if not text:
            return "I understand."
        
        # Split on sentence boundaries, keep first sentence only
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        first_sentence = sentences[0] if sentences else text
        
        # Hard character limit for TTS
        if len(first_sentence) > max_chars:
            # Truncate at word boundary
            truncated = first_sentence[:max_chars].rsplit(' ', 1)[0]
            first_sentence = truncated.rstrip('.,!?') + '...'
        
        return first_sentence

# =============================================================================
# TTS Service (ChatterboxTTS)
# =============================================================================

@app.cls(
    image=image,
    gpu="A10G",
    min_containers=1,
    timeout=300,
)
class TTSService:
    """Text-to-speech using ChatterboxTTS."""

    @modal.enter()
    def load_model(self):
        from chatterbox.tts import ChatterboxTTS
        
        print("🔊 Loading ChatterboxTTS...")
        self.tts = ChatterboxTTS.from_pretrained(device="cuda")
        print("✅ TTS model loaded")

    @modal.method()
    def speak(self, text: str) -> bytes:
        """
        Synthesize text to speech.
        
        Expects: Pre-truncated text (≤120 chars, single sentence)
        Returns: WAV audio bytes
        """
        import io
        import time
        import numpy as np
        from scipy.io import wavfile
        
        try:
            t0 = time.time()
            
            # Safety: Final truncation if somehow exceeded
            # TTS latency scales with text length
            if len(text) > 150:
                text = text[:150].rsplit(' ', 1)[0] + '...'
            
            chars_in = len(text)
            print(f"[TTS] Synthesizing {chars_in} chars: '{text}'")
            
            audio_tensor = self.tts.generate(text)
            
            audio_np = audio_tensor.cpu().numpy()
            
            if audio_np.ndim > 1:
                audio_np = audio_np.squeeze()
            
            if audio_np.dtype in [np.float32, np.float64]:
                max_val = np.abs(audio_np).max()
                if max_val > 1.0:
                    audio_np = audio_np / max_val
                audio_np = (audio_np * 32767).astype(np.int16)
            elif audio_np.dtype != np.int16:
                audio_np = audio_np.astype(np.int16)
            
            buffer = io.BytesIO()
            sample_rate = 24000
            wavfile.write(buffer, sample_rate, audio_np)
            audio_bytes = buffer.getvalue()
            
            # Calculate audio duration for metrics
            audio_duration = len(audio_np) / sample_rate
            
            elapsed = time.time() - t0
            print(f"[TTS] {chars_in} chars → {audio_duration:.1f}s audio in {elapsed:.2f}s")
            
            return audio_bytes
            
        except Exception as e:
            print(f"❌ TTS Error: {e}")
            raise

# =============================================================================
# Remote Callable Pipeline Function
# =============================================================================

@app.function(image=image, timeout=600)
def process_speech(audio_bytes: bytes, return_metrics: bool = True) -> dict:
    """
    Remote-callable speech-to-speech pipeline.
    
    Args:
        audio_bytes: Input audio as WAV bytes
        return_metrics: If True, return dict with audio and metrics; else just audio bytes
        
    Returns:
        Dict with 'audio' (bytes), 'metrics' (timing info), 'transcription', 'response'
    """
    import time
    
    pipeline_start = time.time()
    
    asr = ASRService()
    llm = LLMService()
    tts = TTSService()
    
    # Calculate input audio duration
    try:
        from scipy.io import wavfile
        import io
        with io.BytesIO(audio_bytes) as f:
            sr, data = wavfile.read(f)
            input_duration = len(data) / sr
    except:
        input_duration = 0.0
    
    # Step 1: ASR
    print("🎤 [1/3] Transcribing...")
    t0 = time.time()
    transcription = asr.transcribe.remote(audio_bytes)
    asr_time = time.time() - t0
    print(f"   ✓ Transcribed in {asr_time:.2f}s: {transcription}")
    
    # Step 2: LLM
    print("🤖 [2/3] Generating response...")
    t0 = time.time()
    response = llm.generate.remote(transcription)
    llm_time = time.time() - t0
    input_tokens = len(transcription.split())
    output_tokens = len(response.split())
    print(f"   ✓ Generated in {llm_time:.2f}s: {response}")
    
    # Step 3: TTS
    print("🔊 [3/3] Synthesizing...")
    t0 = time.time()
    audio_response = tts.speak.remote(response)
    tts_time = time.time() - t0
    
    # Calculate output audio duration
    try:
        with io.BytesIO(audio_response) as f:
            sr_out, data_out = wavfile.read(f)
            output_duration = len(data_out) / sr_out
    except:
        output_duration = 0.0
    
    print(f"   ✓ Synthesized in {tts_time:.2f}s")
    
    total_pipeline = time.time() - pipeline_start
    total_model = asr_time + llm_time + tts_time
    
    print(f"\n" + "="*60)
    print(f"{'PIPELINE METRICS':^60}")
    print("="*60)
    print(f"{'Component':<12} | {'Time (s)':<10} | {'Details':<32}")
    print("-"*60)
    print(f"{'ASR':<12} | {asr_time:<10.3f} | {input_duration:.1f}s audio → {len(transcription)} chars")
    print(f"{'LLM':<12} | {llm_time:<10.3f} | {input_tokens} words → {output_tokens} words")
    print(f"{'TTS':<12} | {tts_time:<10.3f} | {len(response)} chars → {output_duration:.1f}s audio")
    print("-"*60)
    print(f"{'Total':<12} | {total_pipeline:<10.3f} | RTF: {total_pipeline/max(input_duration, 0.1):.2f}x")
    print("="*60 + "\n")
    
    if return_metrics:
        return {
            "audio": audio_response,
            "transcription": transcription,
            "response": response,
            "metrics": {
                "asr_time": asr_time,
                "llm_time": llm_time,
                "tts_time": tts_time,
                "total_pipeline": total_pipeline,
                "input_duration": input_duration,
                "output_duration": output_duration,
                "input_chars": len(transcription),
                "output_chars": len(response),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        }
    return audio_response

# =============================================================================
# Local Entrypoint for Testing
# =============================================================================

@app.local_entrypoint()
def main(audio_path: str):
    """
    Local testing entrypoint.
    
    Usage:
        modal run main.py --audio-path input.wav
    """
    import time
    
    print(f"📂 Reading audio from {audio_path}...")
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    print(f"   Loaded {len(audio_bytes)} bytes")
    
    # Call the remote pipeline
    print("\n🚀 Running speech-to-speech pipeline...")
    t0 = time.time()
    result = process_speech.remote(audio_bytes)
    total_time = time.time() - t0
    
    # Save output
    output_path = "output.wav"
    with open(output_path, "wb") as f:
        f.write(result)
    
    print(f"\n✅ Complete!")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Output saved to {output_path}")
