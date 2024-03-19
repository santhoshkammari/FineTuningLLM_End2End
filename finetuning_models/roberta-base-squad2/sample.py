# import os
# import torch
# import torch.nn as nn
# import transformers
# from peft import LoraConfig, get_peft_model
# from transformers import AutoTokenizer, AutoModelForQuestionAnswering, default_data_collator
# from datasets import load_dataset, DatasetDict
#
# from const import LOGIN_TOKEN, MODEL_NAME, CUSTOM_DATASET_NAME
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Make sure CUDA is not visible
#
# class RobertaFineTuneBaseSquad2:
#     def __init__(self, model_name=None):
#         self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#
#     def freeze_model_weights(self):
#         for param in self.model.parameters():
#             param.requires_grad = False
#
#     def load_peft_model(self):
#         config = LoraConfig(
#             r=16,
#             lora_alpha=32,
#             lora_dropout=0.05,
#             bias="none",
#             task_type="CAUSAL_LM"
#         )
#         self.model = get_peft_model(self.model, config)
#
#     def perform_training(self, data):
#         trainer = transformers.Trainer(
#             model=self.model,
#             train_dataset=data['train'],
#             args=transformers.TrainingArguments(
#                 per_device_train_batch_size=1,
#                 gradient_accumulation_steps=1,
#                 warmup_steps=100,
#                 max_steps=200,
#                 learning_rate=2e-4,
#                 logging_steps=1,
#                 output_dir='outputs'
#             ),
#             # data_collator=transformers.QuestionAnsweringPipeline(self.tokenizer)
#             data_collator=default_data_collator
#
#         )
#         self.model.config.use_cache = False
#         trainer.train()
#
#     @staticmethod
#     def load_custom_dataset(custom_dataset_name):
#         return load_dataset(custom_dataset_name)
#
# class DataTransformer:
#     @classmethod
#     def merge_columns(cls, example):
#         example["prediction"] = example["quote"] + " ->: " + str(example["tags"])
#         return example
#
#     @classmethod
#     def split_columns(cls, data, tokenizer,max_seq_length = 512):
#         data['train'] = data['train'].map(DataTransformer.merge_columns)
#
#         def tokenize_and_truncate(example):
#             tokens = tokenizer(example['prediction'], truncation=True, max_length=max_seq_length,
#                                return_overflowing_tokens=False)
#             return tokens
#
#         data = data.map(tokenize_and_truncate, batched=True)
#         return data
#
# if __name__ == '__main__':
#     roberta_squad2 = RobertaFineTuneBaseSquad2(
#         model_name=MODEL_NAME
#     )
#     roberta_squad2.freeze_model_weights()
#     roberta_squad2.load_peft_model()
#
#     custom_data = roberta_squad2.load_custom_dataset(
#         custom_dataset_name=CUSTOM_DATASET_NAME
#     )
#     updated_data: DatasetDict = DataTransformer.split_columns(
#         custom_data,
#         tokenizer=roberta_squad2.tokenizer
#     )
#
#     roberta_squad2.perform_training(updated_data)

import os
import torch
import torch.nn as nn
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, default_data_collator, DataCollatorWithPadding
from datasets import load_dataset, DatasetDict

from const import LOGIN_TOKEN, MODEL_NAME, CUSTOM_DATASET_NAME

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Make sure CUDA is not visible

class RobertaFineTuneBaseSquad2:
    def __init__(self, model_name=None):
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def freeze_model_weights(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def load_peft_model(self):
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, config)

    def perform_training(self, data):
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = transformers.Trainer(
            model=self.model,
            train_dataset=data['train'],
            args=transformers.TrainingArguments(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                warmup_steps=100,
                max_steps=200,
                learning_rate=2e-4,
                logging_steps=1,
                output_dir='outputs'
            ),
            data_collator=data_collator,
        )
        self.model.config.use_cache = False
        trainer.train()



class DataTransformer:
    @staticmethod
    def load_custom_dataset(custom_dataset_name):
        return load_dataset(custom_dataset_name)
    @classmethod
    def merge_columns(cls, example):
        example["prediction"] = example["quote"] + " ->: " + str(example["tags"])
        return example

    @classmethod
    def split_columns(cls, data, tokenizer,max_seq_length = 512):
        data['train'] = data['train'].map(DataTransformer.merge_columns)

        def tokenize_and_truncate(example):
            tokens = tokenizer(example['prediction'], truncation=True, max_length=max_seq_length,
                               return_overflowing_tokens=False)
            return tokens

        data = data.map(tokenize_and_truncate, batched=True)
        return data

if __name__ == '__main__':
    roberta_squad2 = RobertaFineTuneBaseSquad2(
        model_name=MODEL_NAME
    )
    roberta_squad2.freeze_model_weights()
    roberta_squad2.load_peft_model()

    custom_data = roberta_squad2.load_custom_dataset(
        custom_dataset_name=CUSTOM_DATASET_NAME
    )
    updated_data: DatasetDict = DataTransformer.split_columns(
        custom_data,
        tokenizer=roberta_squad2.tokenizer
    )

    roberta_squad2.perform_training(updated_data)
