import time
import queue
import sys
import numpy as np
import sounddevice as sd
import webrtcvad
import io
from scipy.io import wavfile

# Import Modal app and services
from common import app
from asr import ASRService
from llm import LLMService
from tts import TTSService

# Audio configuration
SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
VAD_LEVEL = 3
SILENCE_THRESHOLD_MS = 10000 # Stop recording after 1 second of silence

class AudioRecorder:
    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_LEVEL)
        self.q = queue.Queue()
        self.recording = False
        self.silence_start = None
        self.speech_detected = False

    def callback(self, indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        self.q.put(indata.copy())

    def record_until_silence(self):
        print("\n🎤 Listening... (speak now)")
        self.q.queue.clear()
        audio_buffer = []
        self.speech_detected = False
        self.silence_start = None
        
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', callback=self.callback, blocksize=FRAME_SIZE):
            while True:
                try:
                    data = self.q.get()
                    is_speech = self.vad.is_speech(data.tobytes(), SAMPLE_RATE)
                    
                    if is_speech:
                        if not self.speech_detected:
                            print("🗣️ Speech detected...")
                            self.speech_detected = True
                        self.silence_start = None
                        audio_buffer.append(data)
                    else:
                        if self.speech_detected:
                            if self.silence_start is None:
                                self.silence_start = time.time()
                            elif time.time() - self.silence_start > SILENCE_THRESHOLD_MS / 1000.0:
                                print("🛑 Silence detected, stopping recording.")
                                break
                            audio_buffer.append(data)
                        else:
                            # Keep a small rolling buffer of pre-speech audio (300ms)
                            audio_buffer.append(data)
                            if len(audio_buffer) > 10:
                                audio_buffer.pop(0)
                                
                except KeyboardInterrupt:
                    return None

        # Flatten audio buffer
        return np.concatenate(audio_buffer)

def play_audio(wav_bytes):
    # Read wav bytes
    with io.BytesIO(wav_bytes) as f:
        samplerate, data = wavfile.read(f)
    
    print(f"🔊 Playing response ({len(data)/samplerate:.1f}s)...")
    sd.play(data, samplerate)
    sd.wait()

def main():
    print("🚀 Starting real-time speech-to-speech client...")
    
    recorder = AudioRecorder()
    
    # Run the Modal app context
    with app.run():
        asr = ASRService()
        llm = LLMService()
        tts = TTSService()
        
        print("✅ Connected to Modal services.")
        
        while True:
            try:
                # 1. Record
                audio_data = recorder.record_until_silence()
                if audio_data is None:
                    break
                
                if len(audio_data) == 0:
                    continue

                # Create WAV container for the raw PCM data
                wav_buffer = io.BytesIO()
                wavfile.write(wav_buffer, SAMPLE_RATE, audio_data)
                wav_bytes = wav_buffer.getvalue()
                
                # 2. Transcribe
                print("📝 Transcribing...")
                t0_asr = time.time()
                asr_result = asr.transcribe.remote(wav_bytes)
                t_asr_network = time.time() - t0_asr
                
                # Handle potential list return or dict
                if isinstance(asr_result, list):
                    asr_result = asr_result[0] if asr_result else {}
                
                text = asr_result.get("text", "")
                t_asr_model = asr_result.get("time", 0.0)
                
                print(f"📝 You said: {text} (Model: {t_asr_model:.3f}s, Total: {t_asr_network:.3f}s)")
                
                if not text or not text.strip():
                    print("⚠️ No speech detected in transcription.")
                    continue
                    
                # 3. Generate
                print("🤖 Thinking...")
                t0_llm = time.time()
                llm_result = llm.generate.remote(text)
                t_llm_network = time.time() - t0_llm
                
                response = llm_result.get("text", "")
                t_llm_model = llm_result.get("time", 0.0)
                
                print(f"💬 Response: {response} (Model: {t_llm_model:.3f}s, Total: {t_llm_network:.3f}s)")
                
                # 4. TTS
                print("🔊 Synthesizing...")
                t0_tts = time.time()
                tts_result = tts.speak.remote(response)
                t_tts_network = time.time() - t0_tts
                
                audio_response = tts_result.get("audio", b"")
                t_tts_model = tts_result.get("time", 0.0)
                
                print(f"✅ Audio received (Model: {t_tts_model:.3f}s, Total: {t_tts_network:.3f}s)")
                
                # Print Benchmark Table
                print("\n" + "="*50)
                print(f"{'Component':<15} | {'Model (s)':<12} | {'Network+Overhead (s)':<18}")
                print("-" * 50)
                print(f"{'ASR':<15} | {t_asr_model:<12.3f} | {t_asr_network - t_asr_model:<18.3f}")
                print(f"{'LLM':<15} | {t_llm_model:<12.3f} | {t_llm_network - t_llm_model:<18.3f}")
                print(f"{'TTS':<15} | {t_tts_model:<12.3f} | {t_tts_network - t_tts_model:<18.3f}")
                print("-" * 50)
                print(f"{'Total':<15} | {t_asr_model+t_llm_model+t_tts_model:<12.3f} | {(t_asr_network+t_llm_network+t_tts_network) - (t_asr_model+t_llm_model+t_tts_model):<18.3f}")
                print("="*50 + "\n")

                # 5. Play
                play_audio(audio_response)
                
            except KeyboardInterrupt:
                print("\n👋 Exiting...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                # Optional: break or continue depending on severity
                # import traceback
                # traceback.print_exc()

if __name__ == "__main__":
    main()
