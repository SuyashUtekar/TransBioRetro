import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import Draw
from net_model_utils import load_model, predict_multistep, smiles_to_image_base64
import graphviz

MODEL_PATH = "final_model"
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
            
            G = nx.DiGraph()

            st.markdown("## Retrosynthesis Steps")
            for i, step in enumerate(pathway, 1):
                st.markdown(f"### Step {i}")
                st.image(smiles_to_image_base64(step['product']), caption=f"Product: {step['product']}")

                G.add_node(step['product'])
                for reactant in step['reactants']:
                    G.add_edge(step['product'], reactant)
                    st.image(smiles_to_image_base64(reactant), caption=f"Reactant: {reactant}")

            st.markdown("### 🧬 Retrosynthesis Tree Visualization")

            dot = graphviz.Digraph()
            dot.attr(rankdir='TB', size='8,5')  # Top to Bottom layout

            added = set()

            for step in pathway:
                product = step["product"]
                reactants = step["reactants"]

                if product not in added:
                    dot.node(product, product)
                    added.add(product)

                for reactant in reactants:
                    if reactant not in added:
                        dot.node(reactant, reactant)
                        added.add(reactant)
                    dot.edge(product, reactant)

            st.graphviz_chart(dot)

        else:
            st.error("No valid retrosynthesis path found.")

st.markdown("---")
st.caption("Model: QLoRA-fine-tuned ReactionT5v2  •  Data: KEGG/MetaCyc/USPTO-NPL  •  UI: Streamlit")
st.caption("Developed by Suyash Utekar  •  Source: https://github.com/SuyashUtekar/TransBioRetro")
