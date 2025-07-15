import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load your fine-tuned model
tokenizer = GPT2Tokenizer.from_pretrained('./fine_tuned_model')
model = GPT2LMHeadModel.from_pretrained('./fine_tuned_model')

def generate_response(input_text, max_length=100):
    # Format like training data
    prompt = f"{input_text} ->"
    
    # Tokenize
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            repetition_penalty=1.5
        )
    
    # Decode and clean up
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the input part, keep only the response
    response = response.split('->', 1)[-1].strip()
    return response

# Test it
test_inputs = [
    "shrimp is the",
]

for input_text in test_inputs:
    response = generate_response(input_text)
    print(f"Input: {input_text}")
    print(f"Response: {response}")
    print("-" * 50)
