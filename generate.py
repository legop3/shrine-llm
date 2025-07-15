import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextStreamer
import sys
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

#generate.py [input] [length] [temperature] [model path]

# Load your fine-tuned model
tokenizer = GPT2Tokenizer.from_pretrained(sys.argv[4])
model = GPT2LMHeadModel.from_pretrained(sys.argv[4])

# Set pad token BEFORE using the tokenizer
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

def generate_response(input_text, max_length=int(sys.argv[2])):
    # Format like training data
    prompt = f"{input_text} ->"
    
    # Create streamer that will output to stdout
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Tokenize with attention mask
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
    
    # Generate with streaming
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            num_return_sequences=1,
            temperature=float(sys.argv[3]),
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            repetition_penalty=1.5,
            streamer=streamer  # This enables streaming to stdout
        )

# Test it
test_inputs = [
    sys.argv[1]
]

for input_text in test_inputs:
    generate_response(input_text)