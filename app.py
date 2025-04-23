import streamlit as st
import pandas as pd
from model_utils import load_model, predict_multistep

# --- CONFIGURE YOUR PATHS ---
MODEL_PATH = "final_model"  # point this to the folder where you downloaded 'pranav166/bioretrosynthesis_final/PyTorch/default/1'
BUILDING_BLOCKS_PATH = "bio_building_block.csv"

@st.cache_resource
def init():
    bundle = load_model(MODEL_PATH)
    blocks = pd.read_csv(BUILDING_BLOCKS_PATH)
    return bundle, blocks

model_bundle, building_blocks_df = init()

st.title("🔬 Bioretrosynthesis Predictor")
st.write("Enter a target molecule as a SMILES string and click **Predict** to see its retrosynthetic route.")

target = st.text_input("Target SMILES", placeholder="e.g. CC(=O)O")
max_depth = st.slider("Max recursion depth", 1, 10, 5)

if st.button("Predict"):
    if not target:
        st.warning("Please enter a SMILES string.")
    else:
        with st.spinner("Running multi-step retrosynthesis…"):
            pathway = predict_multistep(
                model_bundle,
                target.strip(),
                building_blocks_df,
                max_depth=max_depth,
            )
        if pathway:
            st.success("Prediction complete!")
            for i, step in enumerate(pathway, 1):
                st.markdown(f"**Step {i}:** `{step}`")
        else:
            st.error("No valid retrosynthesis path found.")

st.markdown("---")
st.caption("Model: QLoRA-fine-tuned ReactionT5v2  •  Data: KEGG/MetaCyc/USPTO-NPL  •  UI: Streamlit")