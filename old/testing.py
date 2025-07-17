import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())  # Should be False for CPU

# Load a small model to test
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
print("Model loaded successfully!")

with open('testdata.json', 'r') as f:
    data = json.load(f)