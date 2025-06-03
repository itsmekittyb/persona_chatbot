import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

class PersonaModel:
    def __init__(self, persona, model_base_path="models"):
        self.persona = persona
        # Try to find models/ in the current or parent directory
        candidate_paths = [
            os.path.join(model_base_path, persona),  # models/warhol
            os.path.join(os.path.dirname(__file__), model_base_path, persona)  # src/models/warhol
        ]
        self.model_path = next((p for p in candidate_paths if os.path.isdir(p)), None)
        if not self.model_path:
            raise FileNotFoundError(f"No model directory found for persona '{persona}' in: {candidate_paths}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer, self.model = self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(self.model_path, local_files_only=True)
            model = model.to(self.device)
            model.eval()
            return tokenizer, model
        except Exception as e:
            print(f"Error loading model/tokenizer for persona '{self.persona}': {e}")
            return None, None

    def generate_response(self, prompt, max_length = 150, temperature = 0.7, top_p = 0.9):
        if self.model is None or self.tokenizer is None:
            return f"Sorry, the '{self.persona}' is unavalaible."
        formatted_prompt = f"<|{self.persona}|> {prompt}"
        try:
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    no_repeat_ngram_size=3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=False)
            generated_text = generated_text.replace(formatted_prompt, "")
            generated_text = generated_text.replace("<|endoftext|>", "")
            generated_text = generated_text.replace(f"<|{self.persona}|>", "")
            return generated_text.strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Sorry, something went wrong generating a response."

if __name__ == "__main__":
    persona = input("Persona (e.g. 'warhol'): ").strip().lower()
    persona_model = PersonaModel(persona)
    print(f"{persona.title()} chatbot ready! Type 'exit' to quit.")
    while True:
        prompt = input("You: ")
        if prompt.strip().lower() == "exit":
            break
        response = persona_model.generate_response(prompt)
        print(f"{persona.title()}: {response}")