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
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=128)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
