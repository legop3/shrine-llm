import json
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import os
from typing import Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optimize PyTorch for CPU
torch.set_num_threads(32)  # Use all your CPU threads
torch.set_num_interop_threads(32)
os.environ['OMP_NUM_THREADS'] = '32'
os.environ['MKL_NUM_THREADS'] = '32'

# Custom dataset class with improved memory efficiency
class DiscordDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length=512):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.info(f"Loaded {len(self.data)} training examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Format: input -> response
        text = f"{self.data[idx]['input']} -> {self.data[idx]['response']}"
        
        # Tokenize with attention to efficiency
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,  # Don't pad here, let DataLoader handle it
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

# Custom collate function for dynamic padding
def collate_fn(batch):
    # Find max length in batch
    max_len = max(len(item['input_ids']) for item in batch)
    
    input_ids = []
    attention_masks = []
    labels = []
    
    for item in batch:
        # Pad to max length in batch
        pad_len = max_len - len(item['input_ids'])
        
        padded_input = torch.cat([
            item['input_ids'],
            torch.full((pad_len,), 50256, dtype=torch.long)  # GPT-2 pad token
        ])
        
        padded_attention = torch.cat([
            item['attention_mask'],
            torch.zeros(pad_len, dtype=torch.long)
        ])
        
        padded_labels = torch.cat([
            item['labels'],
            torch.full((pad_len,), -100, dtype=torch.long)  # Ignore padding in loss
        ])
        
        input_ids.append(padded_input)
        attention_masks.append(padded_attention)
        labels.append(padded_labels)
    
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_masks),
        'labels': torch.stack(labels)
    }

# Custom trainer with CPU optimizations
class OptimizedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss with gradient accumulation optimization
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Apply gradient accumulation scaling
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
            
        return (loss, outputs) if return_outputs else loss

# Main training function
def train_model():
    logger.info("Starting model training...")
    
    # Load tokenizer and model
    logger.info("Loading tokenizer and model...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    # Add padding token (GPT-2 doesn't have one by default)
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    
    # Create dataset
    logger.info("Creating dataset...")
    dataset = DiscordDataset('discord_training_data.json', tokenizer, max_length=512)
    
    # Optimized training arguments for CPU
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,  # Increased batch size for your RAM
        gradient_accumulation_steps=4,   # Effective batch size: 8*4 = 32
        learning_rate=3e-5,              # Slightly lower for stability
        warmup_steps=500,                # More warmup steps
        weight_decay=0.01,               # L2 regularization
        logging_dir='./logs',
        logging_steps=10,
        save_steps=1000,
        save_total_limit=3,
        dataloader_num_workers=8,        # Use multiple CPU cores for data loading
        dataloader_pin_memory=False,     # Not needed for CPU
        remove_unused_columns=False,
        label_smoothing_factor=0.1,      # Label smoothing for better generalization
        max_grad_norm=1.0,              # Gradient clipping
        fp16=False,                     # FP16 not beneficial on CPU
        optim="adamw_torch",            # Use PyTorch's AdamW
        lr_scheduler_type="cosine",     # Cosine learning rate schedule
        warmup_ratio=0.1,
        # CPU-specific optimizations
        dataloader_drop_last=True,
        group_by_length=True,           # Group similar lengths together
        length_column_name="input_ids",
        report_to=[],                   # Disable wandb/tensorboard for performance
    )
    
    # Create custom data collator for dynamic padding
    data_collator = collate_fn
    
    # Create optimized trainer
    trainer = OptimizedTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Add custom callback for memory monitoring
    from transformers import TrainerCallback
    
    class MemoryCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 100 == 0:
                try:
                    import psutil
                    memory_usage = psutil.virtual_memory().percent
                    logger.info(f"Step {state.global_step}: Memory usage: {memory_usage:.1f}%")
                except ImportError:
                    # psutil not available, skip memory monitoring
                    pass
    
    trainer.add_callback(MemoryCallback())
    
    # Train with error handling
    try:
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    # Save the model
    logger.info("Saving model...")
    trainer.save_model('./fine_tuned_model')
    tokenizer.save_pretrained('./fine_tuned_model')
    
    logger.info("Model saved successfully!")

# Additional utility function for memory optimization
def optimize_memory():
    """
    Set additional memory optimization flags
    """
    import gc
    gc.collect()
    
    # Set memory allocation strategy
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # Enable memory efficient attention if available
    try:
        torch.backends.cuda.enable_flash_sdp(False)  # Disable for CPU
    except:
        pass

if __name__ == "__main__":
    # Apply memory optimizations
    optimize_memory()
    
    # Run training
    train_model()