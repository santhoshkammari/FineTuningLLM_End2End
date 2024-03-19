import os
import torch
import torch.nn as nn
import bitsandbytes as bnb
import transformers

from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoConfig, AutoModelForQuestionAnswering, BitsAndBytesConfig
from huggingface_hub import login
from datasets import load_dataset, DatasetDict

from const import LOGIN_TOKEN, MODEL_NAME, CUSTOM_DATASET_NAME

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Make sure CUDA is not visible
login(token=LOGIN_TOKEN, add_to_git_credential=True)


class RobertaFineTuneBaseSquad2:
    def __init__(self, model=None, model_name=None):
        self.model = self._load_model(model_name)
        self.tokenizer = self._load_tokenizer(model_name)

    def _load_model(self, model_name):
        # quantization_config = BitsAndBytesConfig(
        #     load_in_8bit=True
        # )
        __model = AutoModelForQuestionAnswering.from_pretrained(
            pretrained_model_name_or_path=model_name,
            # quantization_config= quantization_config,
            # load_in_8bit=True,
            device_map='auto',
        )
        print("Loaded model")
        return __model

    def _load_tokenizer(self, model_name):
        __tokenizer = AutoTokenizer.from_pretrained(model_name)
        print('Loaded tokenizer')
        return __tokenizer

    def freeze_model_weights(self):
        print("Freezing model weights")
        for param in self.model.parameters():
            param.requires_grad = False  # freeze the model - train adapters later
            if param.ndim == 1:
                # cast the small parameters (e.g. layernorm) to fp32 for stability
                param.data = param.data.to(torch.float32)

        self.model.gradient_checkpointing_enable()  # reduce number of stored activations
        self.model.enable_input_require_grads()

        class CastOutputToFloat(nn.Sequential):
            def forward(self, x): return super().forward(x).to(torch.float32)

        self.model.qa_outputs = CastOutputToFloat(self.model.qa_outputs)

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    def load_peft_model(self):
        print("Loading peft model")
        config = LoraConfig(
            r=16,  # attention heads
            lora_alpha=32,  # alpha scaling
            # target_modules=["q_proj", "v_proj"], #if you know the
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"  # set this for CLM or Seq2Seq
        )

        self.model = get_peft_model(self.model, config)
        self.print_trainable_parameters()

    def perform_training(self, data):

        trainer = transformers.Trainer(
            model=self.model,
            train_dataset=data['train'],
            args=transformers.TrainingArguments(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                warmup_steps=100,
                max_steps=200,
                learning_rate=2e-4,
                # fp16=True,
                logging_steps=1,
                output_dir='outputs'
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        )
        self.model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        trainer.train()

    @staticmethod
    def load_custom_dataset(custom_dataset_name):
        data = load_dataset(custom_dataset_name)
        return data


class DataTransformer:
    @classmethod
    def merge_columns(cls, example):
        example["prediction"] = example["quote"] + " ->: " + str(example["tags"])
        return example

    @classmethod
    def split_columns(cls, data, tokenizer):
        print("Performing split")
        data['train'] = data['train'].map(DataTransformer.merge_columns)
        data = data.map(lambda samples: tokenizer(samples['prediction']), batched=True)
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
