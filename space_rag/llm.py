"""Small local LLM wrapper (Hugging Face transformers, loaded lazily)."""

from __future__ import annotations


class SmallLLM:
    """Runs a small instruction-tuned chat model locally (CPU, CUDA or Apple MPS)."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        max_new_tokens: int = 256,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._device = device
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self._device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        if device == "cuda":
            model = model.half()
        self._model = model.to(device).eval()
        self._device = device

    def generate(self, messages: list[dict]) -> str:
        """`messages` is a chat list: [{"role": "system"|"user", "content": "..."}]."""
        self._load()
        import torch

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # deterministic answers are best for RAG
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
