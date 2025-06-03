from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def ensure_model_downloaded(model_name="gpt2", save_dir=None):
    """
    Downloads a Hugging Face model (GPT-2, GPT-Neo, etc.) and saves it to save_dir if not already present.

    Args:
        model_name (str): Model name on Hugging Face hub (e.g., "gpt2", "EleutherAI/gpt-neo-125M").
        save_dir (str): Directory to save the model. If None, uses models/{basename(model_name)}.
    """
    if save_dir is None:
        safe_name = model_name.replace("/", "-")
        save_dir = f"models/{safe_name}"
    if not os.path.exists(save_dir) or not os.listdir(save_dir):
        print(f"Downloading {model_name} model and tokenizer to {save_dir} ...")
        os.makedirs(save_dir, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        # Set pad_token if needed (important for GPT-Neo and GPT-2)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.save_pretrained(save_dir)
        model.save_pretrained(save_dir)
        print(f"[✓] {model_name} download complete.")
    else:
        print(f"[✓] {model_name} already exists at {save_dir}.")

if __name__ == "__main__":
    ensure_model_downloaded("EleutherAI/gpt-neo-125M")