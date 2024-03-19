from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/roberta-base-squad2"

# a) Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)


# b) Load model & tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

import pandas as pd
df = pd.read_csv("/media/ntlpt59/HI/D/DATA_LAKE/45A/merged_data.csv")

SAMPLE_SIZE = 30
PROMPT_FIRST = """Given description of goods extract the total net weight?"""
PROMPT_SECOND = """
Extract the following information from a description of goods and output it in JSON format:
- Weight
- Unit Price
- Weight Units
- Invoice Number
- Quantity
- Price
- Produced By
- Purchase Order Number
- Destination
- Buyer Reference Number
- Supplier Reference Number
- HS Code
- Goods
- Incoterm
- Terms of Shipment
- Invoice Date
- Date

Ensure consistent extraction and formatting of the output JSON structure across different inputs.
"""

# Example usage:
# Utilize the refined prompt along with a model trained specifically for information extraction tasks.
# Fine-tune your model as necessary to ensure accurate extraction of the required information.
# Implement robust pre-processing and post-processing techniques to handle variations in input data and maintain consistency in the output structure.


question = PROMPT_FIRST
# question = PROMPT_SECOND


def gemma_7b():
    # pip install bitsandbytes accelerate
    from huggingface_hub import login

    # Login to Hugging Face Hub
    login(token="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")

    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-7b")

    input_text = "Write me a poem about Machine Learning."
    input_ids = tokenizer(input_text, return_tensors="pt")

    outputs = model.generate(**input_ids)
    print(tokenizer.decode(outputs[0]))


gemma_7b()
exit('================================================================')
response = []

df_new = df[:SAMPLE_SIZE]
for ind,row in df_new.iterrows():
  print(f"Running {ind}/{len(df)}")
  context = df.at[ind,'ProductDescription']
  res = nlp({
      'context':"30 NOS.PACKAGES THIRTY DRUMS CIF SAVANNAH PORT,  USA 93.00 KMS AAAC 600 MCM CONDUCTOR HS CODE: 76149000 NUMBER: 4483 DATED 01.11.2019 LETTER OF CREDIT ILCSTL0 14531 AND 191108 TOTAL NET WEIGHT: 758 97.000 KGS GROSS WEIGHT:85440.000 LENGTH:93.000 INVOICE ",
      'question':PROMPT_SECOND
  })
  response.append(res['answer'])
  print(response)
  break
  df_new.at[ind,'PromptResponse'] = res['answer']

df_new.to_csv("OUTPUT/prompt1Response.csv")
