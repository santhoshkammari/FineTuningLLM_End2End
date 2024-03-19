from huggingface_hub import login
from finetuning_models.Lora.const import LOGIN_TOKEN

login(token=LOGIN_TOKEN,add_to_git_credential=True)


