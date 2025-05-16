# 🔬 BioRetrosynthesis Streamlit App

A Streamlit-based web application that predicts retrosynthetic pathways using a fine-tuned `ReactionT5v2` model. This tool recursively breaks down a target molecule (SMILES format) into simpler biological building blocks, enhancing accuracy with contrastive learning and beam search.

---

## 🚀 Features

- 🔁 **Recursive Multi-Step Retrosynthesis** using fine-tuned `ReactionT5v2`
- 🧪 **Single-step prediction** for quick retrosynthesis checks
- 📊 **Accurate predictions** powered by BLEU-based scoring and known building blocks filtering
- ⚙️ **Streamlit UI** for easy interaction and visualization
- 📦 **Dockerized Deployment** ready for Google Cloud Run

---

## 🧰 Tools & Technologies

- `PyTorch`, `Transformers (HuggingFace)`
- `Streamlit` for the frontend
- `Pandas` for data manipulation


## 🚀 Deployed on Streamlit 

- I have deployed the streamlit app in the streamlit community.
- Public URL - https://transbioretro-chahkaj5nuwiwc3rdru56l.streamlit.app/
