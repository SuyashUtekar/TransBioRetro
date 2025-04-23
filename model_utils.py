# model_utils.py

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftConfig, PeftModel

def load_model(model_path: str):
    """
    Load tokenizer and Seq2Seq model from local directory.
    """

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("sagawa/ReactionT5v2-retrosynthesis-USPTO_50k", return_tensors="pt")

    # Load the PEFT config
    peft_config = PeftConfig.from_pretrained(model_path)

    # Load the base model
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
                peft_config.base_model_name_or_path,
                device_map="auto",
                torch_dtype=torch.float16
            )

    # Load the PEFT model
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    return {"model": model, "tokenizer": tokenizer}

def is_building_block(smiles: str, blocks_df) -> bool:
    """
    Check if the SMILES string is one of the known building blocks.
    """
    return smiles in set(blocks_df.iloc[:, 0].tolist())

def predict_single_step(model_bundle, product_smiles: str) -> str:
    """
    Predict a single-step retrosynthesis from product SMILES.
    """
    model = model_bundle["model"]
    tokenizer = model_bundle["tokenizer"]

    inputs = tokenizer(product_smiles, return_tensors='pt')
    outputs = model.generate(
        **inputs,
        num_beams=5,
        num_return_sequences=1,
        max_length=128,
        early_stopping=True
    )
    reactants = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reactants.replace(" ", "").rstrip(".")

def predict_multistep(model_bundle, target_smiles: str, blocks_df, max_depth: int = 5):
    """
    Recursively predict retrosynthetic steps until known building blocks or max_depth.
    Returns a list of dicts: [{"product": ..., "reactants": [...], "depth": ...}]
    """
    model = model_bundle["model"]
    tokenizer = model_bundle["tokenizer"]
    results = []

    def recurse(smiles: str, depth: int):
        if depth >= max_depth or is_building_block(smiles, blocks_df):
            return
        # Tokenize & generate
        inputs = tokenizer(smiles, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            num_beams=5,
            num_return_sequences=1,
            max_length=128,
            early_stopping=True,
        )
        reactants = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(" ", "")
        results.append({
            "product": smiles,
            "reactants": reactants.split('.'),
            "depth": depth
        })
        for r in reactants.split('.'):
            recurse(r, depth + 1)

    recurse(target_smiles, 1)
    return results
