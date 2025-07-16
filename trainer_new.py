import json
import torch
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Set CPU optimization
torch.set_num_threads(32)
os.environ['OMP_NUM_THREADS'] = '32'
os.environ['MKL_NUM_THREADS'] = '32'
torch.backends.mkldnn.enabled = True

class DiscordDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length=512):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"Loaded {len(self.data)} conversation examples")
        
        # Pre-tokenize all data to save time during training
        print("Pre-tokenizing data...")
        self.tokenized_data = []
        
        for i, item in enumerate(self.data):
            if i % 1000 == 0:
                print(f"Tokenizing example {i}/{len(self.data)}")
            
            # Add end-of-text token
            text = item['text'] + self.tokenizer.eos_token
            
            # Tokenize
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            self.tokenized_data.append({
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': encoding['input_ids'].flatten()
            })
        
        print(f"Pre-tokenized {len(self.tokenized_data)} examples")
    
    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        return self.tokenized_data[idx]

def estimate_memory_usage(batch_size, seq_length, model_size="medium"):
    """Estimate memory usage for given parameters"""
    if model_size == "medium":
        model_params = 355e6  # GPT2-medium has ~355M parameters
    else:
        model_params = 124e6  # GPT2 has ~124M parameters
    
    # Rough estimation (in GB)
    model_memory = model_params * 4 / 1e9  # 4 bytes per parameter
    batch_memory = batch_size * seq_length * 4 * 4 / 1e9  # input, output, gradients, optimizer
    
    total_memory = model_memory + batch_memory
    return total_memory

def find_optimal_batch_size(tokenizer, max_memory_gb=100):
    """Find the optimal batch size for your system"""
    seq_length = 512
    
    for batch_size in [32, 24, 16, 12, 8, 4, 2, 1]:
        estimated_memory = estimate_memory_usage(batch_size, seq_length)
        if estimated_memory < max_memory_gb:
            print(f"Recommended batch size: {batch_size} (estimated {estimated_memory:.1f}GB memory)")
            return batch_size
    
    return 1

def train_model():
    # Load tokenizer and model
    print("Loading model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium', torch_dtype=torch.float32)
    
    # Add padding token (GPT-2 doesn't have one by default)
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    
    # Find optimal batch size
    optimal_batch_size = find_optimal_batch_size(tokenizer, max_memory_gb=100)
    
    # For CPU training, use smaller batch size but more gradient accumulation
    if optimal_batch_size > 4:
        gradient_accumulation = optimal_batch_size // 2
        optimal_batch_size = 2
    else:
        gradient_accumulation = 2
    
    # Create dataset
    print("Loading dataset...")
    dataset = DiscordDataset('discord_training_data.json', tokenizer, max_length=256)
    
    # Calculate training steps
    num_epochs = 2
    total_steps = (len(dataset) // optimal_batch_size) * num_epochs
    warmup_steps = min(500, total_steps // 10)  # 10% of total steps or 500, whichever is smaller
    
    print(f"Training configuration:")
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Batch size: {optimal_batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    
    # Training arguments optimized for CPU
    training_args = TrainingArguments(
        output_dir='./ross_model',
        num_train_epochs=num_epochs,
        per_device_train_batch_size=optimal_batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        save_steps=min(1000, total_steps // 10),  # Save 10 times during training
        save_total_limit=2,  # Keep only 2 checkpoints
        logging_steps=max(10, total_steps // 100),  # Log 100 times during training
        learning_rate=1e-5,  # Lower learning rate for better convergence
        warmup_steps=warmup_steps,
        weight_decay=0.01,  # Add some regularization
        dataloader_num_workers=8,  # Use multiple CPU cores for data loading
        dataloader_pin_memory=False,  # Don't pin memory on CPU
        remove_unused_columns=False,
        fp16=False,  # CPU doesn't support fp16 well
        bf16=False,  # CPU doesn't support bf16 well
        eval_strategy="no",  # No evaluation during training
        report_to="none",  # Don't report to wandb/tensorboard
        load_best_model_at_end=False,
        metric_for_best_model="loss",
        greater_is_better=False,
        max_grad_norm=1.0,  # Gradient clipping
        dataloader_persistent_workers=True,  # Keep workers alive
    )
    
    # Create trainer
    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    # Train!
    print("Starting training...")
    trainer.train()
    
    # Save the model
    print("Saving model...")
    trainer.save_model('./ross_finetuned_model')
    tokenizer.save_pretrained('./ross_finetuned_model')
    
    print("Training complete!")
    
    # Test the model
    print("\nTesting the model...")
    test_generation(model, tokenizer)

def test_generation(model, tokenizer):
    """Test the trained model with some sample inputs"""
    model.eval()
    
    test_prompts = [
        "User1: Hello everyone!",
        "User2: What's everyone up to today?",
        "User1: I'm working on a coding project",
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + 50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()
        print(f"Response: {response}")

if __name__ == "__main__":
    train_model()