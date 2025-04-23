from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftConfig, PeftModel
import torch

model_path = "final_model"

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

# Example SMILES input
input_smiles = 'CCN(CC)CCNC(=S)NC1CCCc2cc(C)cnc21'
inp = tokenizer(input_smiles, return_tensors='pt')

# Generate output using beam search
output = model.generate(
    **inp,
    num_beams=5,
    num_return_sequences=1,
    return_dict_in_generate=True,
    output_scores=True
)

# Decode and clean up the output
predicted_smiles = tokenizer.decode(output['sequences'][0], skip_special_tokens=True).replace(' ', '').rstrip('.')

print("Predicted Reactants:", predicted_smiles)