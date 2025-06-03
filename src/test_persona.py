import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

def chat_with_persona(persona_name, prompt, max_length=128, model_base_dir="gptneo-persona-finetune"):
    model_dir = model_base_dir
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.eval()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    model.to(device)

    # Use the correct prompt format for the fine-tuned model
    input_text = f"<|{persona_name}|> User: {prompt}\n"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=min(len(inputs["input_ids"][0]) + max_length, 512),
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=40,
            top_p=0.95,
            temperature=0.9,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the model's reply after the prompt
    reply = response[len(input_text):].strip()
    return reply

if __name__ == "__main__":
    print("Choose a persona: basquiat, cobain, morrison, warhol")
    persona = input("Persona: ").strip().lower()
    print("\nType 'exit' or 'quit' to end the chat.")
    while True:
        user_prompt = input(f"\nYou: ")
        if user_prompt.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        reply = chat_with_persona(persona, user_prompt)
        print(f"{persona.capitalize()}: {reply}")