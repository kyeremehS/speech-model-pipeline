from common import app, image
import modal

@app.cls(
    image=image,
    cpu=4,
    memory=16384,  # 16 GB in MiB
    min_containers=1,
)
class LLMService:

    @modal.enter()
    def load_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        # Qwen3 requires trust_remote_code=True
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-1.7B",
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-1.7B",
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).eval()

    @modal.method()
    def generate(self, text: str) -> str:
        import re
        
        # Handle if text is a list
        if isinstance(text, list):
            text = text[0] if text else ""
        
        # Create a conversational prompt with /no_think to disable thinking
        messages = [
            {"role": "system", "content": "You are a helpful voice assistant. Respond naturally and concisely. Keep responses brief (1-2 sentences) and suitable for text-to-speech. Do not use markdown or special formatting."},
            {"role": "user", "content": text}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=False  # Disable thinking mode for Qwen3
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove any thinking tags if present
        response = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL)
        
        # Extract assistant response - look for content after the last role marker
        if "assistant" in response.lower():
            parts = response.lower().split("assistant")
            response = parts[-1]
        
        # Clean up the response
        response = response.strip()
        # Remove any remaining tags or markers
        response = re.sub(r'<[^>]+>', '', response)
        response = response.strip()
        
        return response
