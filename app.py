import torch
import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load model from Hugging Face Hub
MODEL_DIR = "Jinilpatel7/Eng-to-Hin-Translation"

@st.cache_resource
def load_model():
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    return model, tokenizer

model, tokenizer = load_model()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Translation function
def translate_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    forced_bos_token_id = tokenizer.lang_code_to_id["hi_IN"]
    output = model.generate(**inputs, max_length=64, forced_bos_token_id=forced_bos_token_id)
    
    translated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return translated_text

# Streamlit UI
st.title("English to Hindi Translation (mBART + QLoRA)")
st.markdown("Translate English sentences to Hindi using a fine-tuned mBART model with QLoRA.")

# Input text
input_text = st.text_area("Enter English text:", height=100)

if st.button("Translate"):
    if input_text.strip():
        translation = translate_text(input_text)
        st.success(f"**Hindi Translation:** {translation}")
    else:
        st.warning("Please enter some text to translate.")
