# app_streamlit.py
import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from captum.attr import IntegratedGradients
import os
from DownstreamModel import DownstreamModel
from openai import OpenAI


# ==========================
# Page Config (must be first Streamlit call!)
# ==========================
st.set_page_config(
    page_title="Sepsis Prognosis Tool",
    page_icon="🧬",
    layout="wide"
)

# ==========================
# Initialization
# ==========================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_id = "Intelligent-Internet/II-Medical-8B-1706"

@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True, output_hidden_states=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, config=config, trust_remote_code=True, torch_dtype=torch.float32
    ).to(device).eval()

    downstream_model = DownstreamModel(class_num=2).to(device)
    save_dir = f'./Results/II-Medical_batch20_LR5e-05'
    downstream_model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
    downstream_model.eval()

    return tokenizer, base_model, downstream_model

tokenizer, base_model, downstream_model = load_models()
ig = IntegratedGradients(downstream_model)

client = OpenAI(api_key="sk-IOYqhRir5AkiGLjPE99a5fEeD4074500B2A22e0c18F23f31", base_url="https://ai-test-if.wchscu.cn/v1/")

# ==========================
st.title("🧬 Sepsis Prognosis Prediction & Explainability")
st.markdown(
    """
    This tool predicts **patient outcomes (Survival vs. Non-survival)** 
    based on clinical information and provides **explainability analysis** 
    using token-level attribution.
    """
)

# Sidebar
st.sidebar.header("About this Tool")
st.sidebar.info(
    """
    🔹 **Model:** II-Medical-8B with downstream classifier  
    🔹 **Explainability:** Captum Integrated Gradients  
    🔹 **Goal:** Support clinical decision-making in sepsis prognosis  
    🔹 **Version:** v1.0 (2025)  
    """
)

st.sidebar.markdown("---")
st.sidebar.caption("For research use only. Not intended for clinical deployment.")

# ==========================
# User Input
# ==========================
st.subheader("Step 1: Enter Clinical Information")
text_input = st.text_area("Paste or type patient clinical information here:", height=200)

if st.button("Run Prognosis Analysis"):
    if text_input.strip() == "":
        st.warning("⚠️ Please enter clinical data.")
    else:
        # Tokenization
        encodings = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True).to(device)
        input_ids = encodings['input_ids']

        # Representation & prediction
        with torch.no_grad():
            outputs = base_model(**encodings, output_hidden_states=True)
            rep = torch.mean(outputs.hidden_states[-1], dim=1)  # pooled representation
            logits = downstream_model(rep)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(torch.argmax(logits, dim=1).item())

        # Attribution
        attributions, _ = ig.attribute(rep, target=pred, return_convergence_delta=True)
        token_weights = attributions.squeeze().detach().cpu().numpy()
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        # ==========================
        # Display Results
        # ==========================
        st.subheader("Step 2: Prediction Result")

        if pred == 0:
            st.success(f"🟢 Predicted Outcome: Survival ({probs[pred]*100:.1f}% confidence)")
        else:
            st.error(f"🔴 Predicted Outcome: Non-survival ({probs[pred]*100:.1f}% confidence)")

        # ==========================
        # Explainability
        # ==========================
        st.subheader("Step 3: Model Explainability")

        token_str = " ".join([f"{tok}:{w:.4f}" for tok, w in zip(tokens, token_weights)])
        prompt = f"""
        The predicted outcome is {"Non-survival" if pred==1 else "Survival"}.
        Token-level attribution weights: {token_str}.
        Please provide a clinical interpretation of the most important features based on these token weights
        and explain how they may influence prognosis in sepsis patients.
        """

        try:
            response = client.chat.completions.create(
                model="qwen2.5-vl-instruct",
                messages=[
                    {"role": "system", "content": "You are a critical care expert specializing in sepsis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            explanation = response.choices[0].message.content.strip()
        except:
            explanation = "⚠️ Failed to generate expert interpretation."

        st.write(explanation)

        st.success("✅ Analysis completed successfully.")