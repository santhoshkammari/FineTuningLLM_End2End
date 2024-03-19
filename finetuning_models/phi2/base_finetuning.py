import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, GPT2ForSequenceClassification, Trainer, TrainingArguments

# Load and preprocess the dataset
data = pd.read_csv("your_dataset.csv")  # Replace with the path to your CSV file
train_data, eval_data = train_test_split(data, test_size=0.1, random_state=42)

# Tokenize the dataset
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
train_encodings = tokenizer(train_data["text"].tolist(), truncation=True, padding=True)
eval_encodings = tokenizer(eval_data["text"].tolist(), truncation=True, padding=True)

# Create PyTorch datasets
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = MyDataset(train_encodings, train_data["destination"].tolist())
eval_dataset = MyDataset(eval_encodings, eval_data["destination"].tolist())

# Load pre-trained GPT-2 model and config
model_name = "gpt2"
config = GPT2Config.from_pretrained(model_name)
model = GPT2ForSequenceClassification.from_pretrained(model_name, config=config)

# Fine-tune the model
training_args = TrainingArguments(
    output_dir="./gpt2_finetuned_destination",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)

# Test the model on new examples
test_text = "LEATHER AS PER PO: TAT230601 REV TRADE TERMS: CIF HAI PHONG PORT, VIETNAM"
test_encoding = tokenizer(test_text, return_tensors="pt")
output = model(**test_encoding)
predicted_destination = tokenizer.decode(output.logits.argmax(dim=1), skip_special_tokens=True)
print("Predicted Destination:", predicted_destination)
