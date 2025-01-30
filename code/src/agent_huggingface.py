from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from dotenv import load_dotenv


class LLM:
    """
    LLM class to interact with Hugging Face models.
    """

    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
        """
        Initialize the Hugging Face model and tokenizer.

        :param model_name: The Hugging Face model name, default is 'meta-llama/Llama-2-7b-chat-hf'.
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        """
        Load the Hugging Face model and tokenizer with GPU support.
        """
        print(f"Loading model: {self.model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)

    def get_response(self, prompt, temperature=0.7, max_tokens=100):
        """
        Generate a response using the Hugging Face model.

        :param prompt: The input text prompt.
        :param temperature: Sampling temperature for diversity in output.
        :param max_tokens: Maximum tokens in the generated response.
        :return: The generated response as a string.
        """
        if not self.tokenizer or not self.model:
            self._load_model()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)  # Move inputs to GPU
        output = self.model.generate(
            inputs.input_ids,
            max_length=max_tokens,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


if __name__ == '__main__':
    load_dotenv()
    prompt = """You are a python programmer. 
    Answer the following question with best of your knowledge.

    Question: Can you write an RAG pipeline with Pinecone and Hugging Face?
    Answer:
    """
    client = LLM(model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    response = client.get_response(prompt, temperature=1, max_tokens=10000)
    print(response)


