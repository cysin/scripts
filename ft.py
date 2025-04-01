import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
import wandb

# Set up configuration
model_name = "./Qwen2-VL-7B-Instruct"
dataset_name = "./LaTeX_OCR"
output_dir = "./qwen2_vl_latex_ocr_finetuned"

# Initialize wandb for experiment tracking
wandb.init(
    project="qwen2-vl-latex-ocr-finetuning",
    config={
        "model": model_name,
        "dataset": dataset_name,
        "max_seq_length": 2048,
        "learning_rate": 5e-5,
        "batch_size": 4,
        "num_epochs": 3
    }
)

# Load the dataset
dataset = load_dataset(dataset_name)

# Prepare the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Preprocessing function
def preprocess_function(examples):
    # Adjust this based on the actual structure of your LaTeX OCR dataset
    inputs = tokenizer(
        examples['input'], 
        examples['target'], 
        max_length=2048, 
        truncation=True, 
        padding='max_length'
    )
    return inputs

# Prepare the dataset
prepared_dataset = dataset.map(
    preprocess_function, 
    batched=True, 
    remove_columns=dataset['train'].column_names
)

# Load the model with Unsloth
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = torch.bfloat16,
    load_in_4bit = True
)

# Prepare model for training
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,  # LoRA rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    modules_to_save = ["embed_tokens", "output_layer"]
)

# Training arguments
from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir = output_dir,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,
    warmup_ratio = 0.1,
    learning_rate = 5e-5,
    logging_steps = 10,
    num_train_epochs = 3,
    logging_dir = "./logs",
    report_to = "wandb",
    fp16 = True,
    save_strategy = "epoch",
    evaluation_strategy = "epoch"
)

# Trainer setup
from transformers import Trainer, DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = prepared_dataset['train'],
    eval_dataset = prepared_dataset['validation'] if 'validation' in prepared_dataset else None,
    data_collator = data_collator
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Finish wandb logging
wandb.finish()

print(f"Fine-tuning complete. Model saved to {output_dir}")
