import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_model_id = "santhoshkammari/PEFTFinetune-Bloom7B"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)

# question = "“Training models with PEFT and LoRa is cool” ->: "
question = '“Be yourself; everyone else is already taken.”'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inputs = tokenizer.encode(question, return_tensors="pt", max_length=50, truncation=True).to(device)

# Generate answer
with torch.no_grad():
    outputs = model.generate(inputs, max_length=50, temperature=0.7, num_return_sequences=1)

    # Decode and return the answer
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(answer)