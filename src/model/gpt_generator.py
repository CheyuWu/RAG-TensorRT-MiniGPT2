import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class GPT2Generator:
    def __init__(self, model_name: str = "gpt2") -> None:
        """
        Initialize GPT-2 generator

        Args:
            model_name: Model name, options are 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
        """
        print(f"Loading GPT-2 model: {model_name}")
        self.tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(model_name)

        # Set pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Check if GPU is available
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        print(f"Model loaded on device: {self.device}")

    def generate_response(
        self,
        context: str,
        max_length: int = 200,
        temperature: float = 0.8,
        num_beams: int = 1,
        do_sample: bool = True,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate a response using GPT-2

        Args:
            context: Input text (including retrieved documents and user question)
            max_length: Maximum length of generated text
            temperature: Controls randomness, higher values mean more random
            num_beams: Number of beams for beam search
            do_sample: Whether to use sampling
            top_p: Parameter for nucleus sampling

        Returns:
            Generated text
        """
        # Encode input
        inputs = self.tokenizer.encode(context, return_tensors="pt").to(self.device)

        # Set generation parameters
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=min(max_length, inputs.shape[1] + 150),  # Limit total length
                temperature=temperature,
                num_beams=num_beams,
                do_sample=do_sample,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=2,  # Avoid repetition
                early_stopping=True,
            )

        # Decode output, return only the newly generated part
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove input part, return only the new content
        if generated_text.startswith(context):
            new_text = generated_text[len(context) :].strip()
            return (
                new_text
                if new_text
                else "I need more context to provide a better response."
            )

        return generated_text.strip()
