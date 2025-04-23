import streamlit as st
import pandas as pd
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Draw
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
st.write("Enter a target molecule as a SMILES string and click Predict to see its retrosynthetic route.")

target = st.text_input("Target SMILES", placeholder="e.g. CC(=O)O")
max_depth = st.slider("Max recursion depth", 1, 10, 5)

# Generate tree and chemical structure visualization
def visualize_tree(pathway):
    G = nx.DiGraph()  # Directed graph for retrosynthesis tree
    for step in pathway:
        product = step["product"]
        reactants = step["reactants"]
        for reactant in reactants:
            G.add_edge(product, reactant)  # Edge from product to reactant

    # Draw the graph
    nx.draw(G, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold')

def visualize_structure(smiles):
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol)
    return img

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

            # Visualize the retrosynthesis tree
            st.subheader("Retrosynthesis Tree")
            visualize_tree(pathway)
            st.pyplot()

            # Visualize the chemical structures of reactants
            st.subheader("Chemical Structures of Reactants")
            for step in pathway:
                for reactant in step["reactants"]:
                    st.write(f"**Reactant:** {reactant}")
                    img = visualize_structure(reactant)
                    st.image(img)

        else:
            st.error("No valid retrosynthesis path found.")

st.markdown("---")
st.caption("Model: QLoRA-fine-tuned ReactionT5v2  •  Data: KEGG/MetaCyc/USPTO-NPL  •  UI: Streamlit")
