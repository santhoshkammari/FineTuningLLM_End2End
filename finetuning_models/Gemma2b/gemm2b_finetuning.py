import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from huggingface_hub import  login
login(token="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")

import torch,os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "google/gemma-2b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config,
                                             device_map = 'cuda')


text = "Quote: Imagination is more"
inputs = tokenizer(text, return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

from datasets import load_dataset

data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

from peft import LoraConfig

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

import torch

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size x vocabulary size)
        top_k: <=0: no filtering, >0: keep only top k tokens with highest probability
        top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
            whose total probability mass is greater than or equal to the threshold top_p.
            In practice, this means selecting the narrowest possible distribution around the most likely candidates.
        filter_value: a float value -inf used to indicate that logits should be ignored
    Returns:
        A tensor of the same shape as logits, with values modified according to top_k and/or top_p, with -inf for ignored logits.
    """
    if top_k > 0:
        values, _ = torch.topk(logits, top_k, dim=-1)
        min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[-1])
        logits = torch.where(logits < min_values, torch.ones_like(logits) * -float('Inf'), logits)

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, :top_k] = 0

        for i in range(sorted_indices.size(0)):
            logits[i, sorted_indices[i, sorted_indices_to_remove[i]]] = filter_value

    return logits

import transformers
from trl import SFTTrainer

def formatting_func(example):
    output_texts = []
    for i in range(len(example)):
        text = f"Quote: {example['quote'][i]}\nAuthor: {example['author'][i]}"
        output_texts.append(text)
    return output_texts

trainer = SFTTrainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=10,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit"
    ),
    peft_config=lora_config,
    formatting_func=formatting_func,
)
trainer.train()

text = "Quote: Imagination is"
device = "cuda:0"
inputs = tokenizer(text, return_tensors="pt").to(device)

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
