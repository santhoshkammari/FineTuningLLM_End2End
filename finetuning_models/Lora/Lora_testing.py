# import os
# import torch
# import torch.nn as nn
# import bitsandbytes as bnb
#
# from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM,BitsAndBytesConfig
# from huggingface_hub import login
#
# from finetuning_models.Lora.const import LOGIN_TOKEN, MODEL_NAME
#
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# login(token=LOGIN_TOKEN, add_to_git_credential=True)
#
#
# class LoraLLM:
#     @classmethod
#     def load_model(cls):
#         quantization_config = BitsAndBytesConfig(
#             quantize=True,
#             quantization_dtype=torch.uint8,  # You can choose the dtype you prefer
#         )
#         model = AutoModelForCausalLM.from_pretrained(
#             pretrained_model_name_or_path=MODEL_NAME,
#             quantization_config=quantization_config,
#             device_map='cuda' if torch.cuda.is_available() else 'cpu',
#             # load_in_8bit=True,
#
#         )
#         tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#         print("Model Loaded: %s" % model)
#         return model, tokenizer
#
#
# if __name__ == '__main__':
#     LoraLLM.load_model()



import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login

from finetuning_models.Lora.const import LOGIN_TOKEN, MODEL_NAME

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Make sure CUDA is not visible
login(token=LOGIN_TOKEN, add_to_git_credential=True)


class LoraLLM:
    @classmethod
    def load_model(cls):
        # Specify quantization configuration
        quantization_config = BitsAndBytesConfig(
            quantize=True,
            quantization_dtype=torch.uint8,  # You can choose the dtype you prefer
        )

        # Load model with quantization configuration
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=MODEL_NAME,
            # quantization_config=quantization_config,
            device='cpu',  # Force CPU usage
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print("Model Loaded: %s" % model)
        return model, tokenizer


if __name__ == '__main__':
    LoraLLM.load_model()
