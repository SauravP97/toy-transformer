import torch

from tokenizer import Tokenizer
from hugging_face.hf_model import HuggingFaceToyTransformer

model_link = "SauravP97/toy-transformer-shakespeare-work"
print("Loading Pre-trained model...")
pretrained_model = HuggingFaceToyTransformer.from_pretrained(model_link)

device = "cuda" if torch.cuda.is_available() else "cpu"
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_tokens = pretrained_model.generate(context, max_new_tokens=3000)

print(generated_tokens.shape)

tokenizer = Tokenizer(dataset_path="./dataset/shakespear-text.txt")
generated_text = tokenizer.decode(generated_tokens[0].tolist(), stringify=True)

print("\nGenerated text: ")
print(generated_text)
