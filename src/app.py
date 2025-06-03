import os.path

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- App Configuration ---
st.set_page_config(page_title="APCH - AI-Powered Coffee House", layout="centered")

# --- Custom CSS ---
st.markdown("""
    <style>
    body, .stApp {
        background-color: #111 !important;
    }
    .chat-title {
        font-size: 2.4em;
        font-weight: bold;
        background: linear-gradient(90deg, #e66465, #9198e5, #f3ec78, #af4261, #25b79f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 0.3em;
    }
    .desc {
        color: #f3ec78;
        font-size: 1.1em;
        padding-bottom: 0.6em;
    }
    .persona-desc {
        color: #25b79f;
        font-size: 1em;
        font-style: italic;
        margin-bottom: 1em;
    }
    /* User message container */
    .st-chat-message-user {
        justify-content: flex-end; /* Align user messages to the right */
    }
    /* Assistant message container */
    .st-chat-message-assistant {
        justify-content: flex-start; /* Align assistant messages to the left */
    }
    .user-bubble {
        background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
        color: #222;
        padding: 0.7em 1.1em;
        border-radius: 1.2em 1.2em 0.3em 1.2em;
        margin: 0.4em 0 0.4em 3.5em;
        max-width: 70%;
        box-shadow: 2px 2px 10px #3337;
        font-size: 1.1em;
        word-break: break-word;
        display: inline-block;
    }
    .warhol-bubble {
        background: linear-gradient(135deg, #ff69b4 0%, #ffc1e3 100%);
        color: #222;
        padding: 0.7em 1.1em;
        border-radius: 1.2em 1.2em 1.2em 0.3em;
        margin: 0.4em 3.5em 0.4em 0;
        max-width: 70%;
        box-shadow: 2px 2px 10px #2227;
        font-size: 1.1em;
        word-break: break-word;
        display: inline-block;
    }
    .basquiat-bubble {
        background: linear-gradient(135deg, #ffe066 0%, #fff3b0 100%);
        color: #222;
        padding: 0.7em 1.1em;
        border-radius: 1.2em 1.2em 1.2em 0.3em;
        margin: 0.4em 3.5em 0.4em 0;
        max-width: 70%;
        box-shadow: 2px 2px 10px #2227;
        font-size: 1.1em;
        word-break: break-word;
        display: inline-block;
    }
    .cobain-bubble {
        background: linear-gradient(135deg, #b3e5fc 0%, #e0f7fa 100%);
        color: #222;
        padding: 0.7em 1.1em;
        border-radius: 1.2em 1.2em 1.2em 0.3em;
        margin: 0.4em 3.5em 0.4em 0;
        max-width: 70%;
        box-shadow: 2px 2px 10px #2227;
        font-size: 1.1em;
        word-break: break-word;
        display: inline-block;
    }
    .morrison-bubble {
        background: linear-gradient(135deg, #caffb9 0%, #e0ffe0 100%);
        color: #222;
        padding: 0.7em 1.1em;
        border-radius: 1.2em 1.2em 1.2em 0.3em;
        margin: 0.4em 3.5em 0.4em 0;
        max-width: 70%;
        box-shadow: 2px 2px 10px #2227;
        font-size: 1.1em;
        word-break: break-word;
        display: inline-block;
    }
    .bubble-row {
        display: flex;
        align-items: flex-end;
    }
    .avatar {
        border-radius: 50%;
        margin: 0 0.6em;
        box-shadow: 0 2px 8px #0005;
        border: 2px solid #222;
        width: 48px;
        height: 48px;
        object-fit: cover;
        background: #222;
    }
    .stRadio > label {
        color: #f3ec78; 
        font-size: 1.1em;
        font-weight: bold;
    }
    .stRadio div[role="radiogroup"] {
        display: flex;
        flex-direction: column; 
        gap: 0.5em; 
    }
    .stRadio div[role="radiogroup"] label {
        background-color: #4b5563; 
        color: white;
        padding: 0.75em 1.5em;
        border-radius: 0.5em;
        cursor: pointer;
        transition: background-color 0.3s ease, box-shadow 0.3s ease;
        font-family: sans-serif;
        font-weight: 600;
        font-size: 1em;
        text-align: center;
        border: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .stRadio div[role="radiogroup"] label:hover {
        background-color: #6366f1;
    }
    .stRadio div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child {
        display: none;
    }
    .stRadio div[role="radiogroup"] label[data-baseweb="radio"][aria-checked="true"] {
        background-color: #6366f1; 
        box-shadow: 0 0 8px rgba(100, 100, 255, 0.8);
        border: 2px solid #9198e5; 
    }

    /* Input field styling */
    .stTextInput > div > div > input {
        background-color: #222 !important;
        color: #eee !important;
        border: 1px solid #444 !important;
        border-radius: 0.5em !important;
        padding: 0.8em !important;
        font-size: 1.1em !important;
    }
    .stTextInput > label {
        color: #f3ec78 !important;
        font-size: 1.1em !important;
        font-weight: bold !important;
    }
    /* Send button styling */
    .stButton > button {
        background-color: #e66465 !important; 
        color: white !important;
        border-radius: 0.5em !important;
        padding: 0.8em 1.5em !important;
        font-size: 1.1em !important;
        font-weight: bold !important;
        transition: background-color 0.3s ease, box-shadow 0.3s ease !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.3) !important;
    }
    .stButton > button:hover {
        background-color: #af4261 !important; 
        box-shadow: 0 0 10px rgba(230, 100, 101, 0.6) !important;
    }
    /* Spinner styling */
    .stSpinner > div > div {
        color: #9198e5 !important; 
    }
    .stSpinner > div > div > div {
        border-top-color: #9198e5 !important; 
    }
    .stSpinner > div > div > span {
        color: #eee !important; 
    }
    </style>
""", unsafe_allow_html=True)

# --- Title & Descriptions ---
st.markdown('<div class="chat-title">"☕ Meet Us at the APCH"</div>', unsafe_allow_html=True)
st.markdown('<div class="desc">_Chat with four legendary artists and musicians in AI form! Who will inspire your next cup?_</div>', unsafe_allow_html=True)

personas = ["basquiat", "cobain", "morrison", "warhol"]
persona_images = {
    "basquiat": "src/images/basquiat.png",
    "cobain": "src/images/cobain.png",
    "morrison": "src/images/morrison.jpg",
    "warhol": "src/images/warhol.png"
}
persona_descriptions = {
    "basquiat": "Basquiat: The king of neo-expressionism, always raw and rebellious.",
    "cobain": "Cobain: Grunge poet, sensitive soul with a sharp edge.",
    "morrison": "Morrison: The Lizard King, mystic and lyrical explorer.",
    "warhol": "Warhol: Pop art icon, master of fame and the everyday."
}

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "selected_persona" not in st.session_state:
    st.session_state["selected_persona"] = personas[0]

# --- Persona Selection UI (Sidebar)
st.sidebar.header("Select a Persona")
selected_persona_radio = st.sidebar.radio(
    "Choose your artist:",
    personas,
    index=personas.index(st.session_state["selected_persona"]), # Set initial selection
    format_func=lambda x: x.replace('_', ' ').title(), # Capitalize for display
    key="persona_selector"
)

# Update selected persona if changed and clear chat history
if selected_persona_radio != st.session_state["selected_persona"]:
    st.session_state["selected_persona"] = selected_persona_radio
    st.session_state["chat_history"] = []
    st.rerun()

# Display current persona's image and description
st.image(persona_images[st.session_state["selected_persona"]], width=150, caption=f"{st.session_state['selected_persona'].capitalize()}")
st.markdown(f'<div class="persona-desc">{persona_descriptions[st.session_state["selected_persona"]]}</div>', unsafe_allow_html=True)
st.markdown('<span style="color:#eee">Type your message and have a delightful convo with these crazies. Type \'exit\' if you already had enough.</span>', unsafe_allow_html=True)


# --- Model Loading (Cached for performance) ---
@st.cache_resource
def load_model(model_relative_path):
    """
    Loads the tokenizer and model from the specified directory.
    Uses st.cache_resource to load only once.
    """
    try:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )

        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        absolute_model_path = os.path.join(current_script_dir, model_relative_path)

        if not os.path.isdir(absolute_model_path):
            raise FileNotFoundError(f"Model directory not found: {absolute_model_path}")

        expected_files = ['config.json', 'tokenizer_config.json', 'vocab.json']
        if not any(os.path.exists(os.path.join(absolute_model_path, f)) for f in expected_files):
            st.warning(
                f"Warning: Missing expected model/tokenizer config files in {absolute_model_path}. This might cause issues.")

        tokenizer = AutoTokenizer.from_pretrained(absolute_model_path, local_files_only=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(absolute_model_path, local_files_only=True).to(device)
        model.eval()

        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        st.exception(e)
        st.stop()

model_dir_path = "models/gptneo-persona-finetune"
tokenizer, model, device = load_model(model_dir_path)

def generate_reply(persona_name, prompt, max_length=128):
    formatted_prompt = f"<|{persona_name}|> User: {prompt}\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
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
    reply = response[len(formatted_prompt):].strip()
    for p in personas:
        reply = reply.replace(f"<|{p}|>", "").strip()

    if not reply or len(reply.strip()) < 5:
        return "Oh, you know... art is what you can get away with. That's pretty interesting."
    return reply

# --- Send Message Handler ---
def send_message_handler():
    user_input = st.session_state.user_input_key
    current_persona = st.session_state["selected_persona"]

    if user_input.strip():
        if user_input.lower().strip() in ["exit", "quit"]:
            st.session_state.chat_history.append({"role": "user", "content": user_input, "persona": "user"})
            st.session_state.chat_history.append(
                {"role": "assistant", "content": "Goodbye!", "persona": current_persona})
            st.session_state.chat_history = []
        else:
            st.session_state.chat_history.append({"role": "user", "content": user_input, "persona": "user"})

            with st.spinner(f"{current_persona.title()} is thinking..."):
                reply = generate_reply(current_persona, user_input)

            st.session_state.chat_history.append({"role": "assistant", "content": reply, "persona": current_persona})
        st.session_state.user_input_key = ""

# --- Chat Input ---
st.text_input("You:", key="user_input_key", on_change=send_message_handler)

# --- Chat Display with Bubbles and Avatars ---
for message in st.session_state["chat_history"]:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(f'<div class="user-bubble-content"><strong>You:</strong> {message["content"]}</div>',
                        unsafe_allow_html=True)
    else:
        message_persona = message.get("persona", st.session_state["selected_persona"])
        avatar_src = persona_images.get(message_persona, "https://placehold.co/40x40/CCCCCC/000000?text=?")

        with st.chat_message("assistant", avatar=avatar_src):
            bubble_class = f"{message_persona.lower()}-bubble-content"
            st.markdown(
                f'<div class="{bubble_class}"><strong>{message_persona.capitalize()}:</strong> {message["content"]}</div>',
                unsafe_allow_html=True)

# --- Ethical Disclaimer ---
st.markdown(
    """
    <div style="font-size: 0.8em; color: gray; margin-top: 20px;">
        <p><strong>Disclaimer:</strong> This project utilizes AI models to emulate the thinking and speaking styles of 20th-century artists. These models are intended for experimental and illustrative purposes only. The outputs generated are AI-based emulations and should not be taken as accurate representations of the artists' actual views, statements, or personalities. This is a symbolic model, not a real person.</p>
    </div>
    """,
    unsafe_allow_html=True
)
