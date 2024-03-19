from datasets import load_dataset, DatasetDict
from random import randrange
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,TrainingArguments,pipeline
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
from trl import SFTTrainer
from huggingface_hub import login, notebook_login


# data_files = {'train':'train.json','test':'test.json'}
# dataset = load_dataset('json',data_files=data_files)



model_name = "google/flan-t5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

#Makes training faster but a little less accurate
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

#setting padding instructions for tokenizer
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

from const import CUSTOM_DATASET_NAME, LOGIN_TOKEN

dataset_2 = load_dataset(CUSTOM_DATASET_NAME)


from sample import DataTransformer
custom_data = DataTransformer.load_custom_dataset(
        custom_dataset_name=CUSTOM_DATASET_NAME
    )
updated_data: DatasetDict = DataTransformer.split_columns(
        custom_data,
        tokenizer=tokenizer
    )

dataset = {'train':[],'test':[]}

print(updated_data)
for i,k in enumerate(updated_data['train']):
    key = 'train' if i<=2000 else 'test'
    dataset[key].append({
        # 'id': updated_data['train'][i]['input_ids'] ,
        'dialogue':updated_data['train'][i]['prediction'],
        'summary':updated_data['train'][i]['quote'],
        'id':str(i)
    })

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
print("TrainingArguments")
# Create the trainer
trainingArgs = TrainingArguments(
    output_dir='output',
    num_train_epochs=1,
    per_device_train_batch_size=2,
    save_strategy="epoch",
    learning_rate=2e-4
)
print("LoraConfig start")
peft_config = LoraConfig(
      lora_alpha=16,
      lora_dropout=0.1,
      r=64,
      bias="none",
      task_type="CAUSAL_LM",
)

print("SftTrainer")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset = dataset['test'],
    peft_config=peft_config,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=prompt_instruction_format,
    args=trainingArgs,
)
print("Done Before training")
login(token=LOGIN_TOKEN)
tokenizer.save_pretrained('santhoshkammari/flan-t5-small')

#Create model card
trainer.create_model_card()

# Push the results to the hub
# trainer.push_to_hub()
exit('================================================================')
trainer.train()

