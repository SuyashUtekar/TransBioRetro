import streamlit as st
from model_utils import load_model, predict_single_step
import pandas as pd

# --- CONFIGURE PATHS ---
MODEL_PATH = "final_model"
BUILDING_BLOCKS_PATH = "bio_building_block.csv"

@st.cache_resource
def init():
    bundle = load_model(MODEL_PATH)
    blocks = pd.read_csv(BUILDING_BLOCKS_PATH)
    return bundle, blocks

model_bundle, building_blocks_df = init()

st.title("🔬 Single-Step Bioretrosynthesis Predictor")
st.write("Enter a target molecule as a SMILES string and click **Predict** to see a single-step retrosynthetic prediction.")

target = st.text_input("Target SMILES", placeholder="e.g. CC(=O)O")

if st.button("Predict"):
    if not target:
        st.warning("Please enter a SMILES string.")
    else:
        with st.spinner("Predicting reactants…"):
            try:
                reactants = predict_single_step(model_bundle, target.strip())
                st.success("Prediction complete!")
                st.markdown(f"**Predicted Reactants:** `{reactants}`")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Model: QLoRA-fine-tuned ReactionT5v2  •  Data: KEGG/MetaCyc/USPTO-NPL  •  UI: Streamlit")
