import json
from random import randrange

import torch
from datasets import load_dataset
from langchain_community.llms.ollama import Ollama
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, TrainingArguments, AutoModelForCausalLM,
from peft import LoraConfig
from trl import SFTTrainer

def load_data():
    with open("/home/ntlpt59/MAIN/LLM/FineTuningLLM_End2End/data/chat.txt","r") as f:
        text = f.read().split("\n")
    text = [_[18:].strip() for _ in text[1:]]
    return text

def create_data():
    text_list = load_data()
    data=[]
    for idx,x in enumerate(text_list[::2]):
        data.append({"id":str(idx),"dialogue":str(text_list[idx]),"summary":str(text_list[idx+1])})
    return data

def prompt_instruction_format(sample):
  return f"""### Instruction:
    Use the Task below and the Input given to write the Response:

    ### Task:
    Summarize the Input

    ### Input:
    {sample['dialogue']}

    ### Response:
    {sample['summary']}
    """

if __name__ == '__main__':

    data= create_data()
    data = data[:100]
    with open("save.json","w") as f:
        json.dump(data,f)
    dataset = load_dataset("json",data_files="save.json",split="train").train_test_split(test_size=0.2)

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    bnb_config = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16"  # Convert to string for serialization
    }
    model = AutoModelForCausalLM.from_pretrained(model_name,quantization_config=bnb_config)
    # Makes training faster but a little less accurate
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,
                                              quantization_config = bnb_config)

    # setting padding instructions for tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    trainingArgs = TrainingArguments(
        output_dir='output',
        num_train_epochs=1,
        per_device_train_batch_size=4,
        save_strategy="epoch",
        learning_rate=2e-4
    )
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        peft_config=peft_config,
        tokenizer=tokenizer,
        packing=True,
        formatting_func=prompt_instruction_format,
        args=trainingArgs,
    )
    # load model and tokenizer from huggingface hub with pipeline
    print('='*50)
    trainer.train()
    trainer.save_model("finetuned_model")
    print('='*50)


    summarizer = pipeline("summarization", model="finetuned_model")

    # select a random test sample
    sample = dataset['test'][randrange(len(dataset["test"]))]
    print(f"dialogue: \n{sample['dialogue']}\n---------------")

    # summarize dialogue
    res = summarizer(sample["dialogue"],max_length = 10)

    print(f"flan-t5-small summary:\n{res[0]['summary_text']}")