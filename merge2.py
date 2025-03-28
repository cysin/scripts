import torch
from unsloth import FastVisionModel
import os

# 1. Configuration
base_model_name = "./Qwen2.5-VL-7B-Instruct" # Or your specific base model identifier
adapter_path = "./outputs_ped/checkpoint-2000/" #IMPORTANT: Path to your saved LoRA adapter directory (contains adapter_config.json, adapter_model.bin, etc.)
merged_model_save_path = "./qwen2.5-7b-vl-merged" # Directory to save the final merged model

# Optional: Define loading precision (try matching training or use bf16 if memory allows)
# If you trained in 4bit, load in 4bit first. Merging might upcast temporarily.
load_in_4bit = True # Set to False if you trained in 8bit or bf16 and have enough RAM/VRAM
load_in_8bit = False # Set to False if you trained in 4bit or bf16
dtype = None # Autodetected by Unsloth based on load_in_4bit/8bit. Or torch.bfloat16 / torch.float16 if not using 4/8bit

# Check VRAM/RAM requirements - merging can be memory-intensive
print("Starting model loading...")

# 2. Load the Base Model using Unsloth
model_ori, tokenizer = FastVisionModel.from_pretrained(
    model_name = base_model_name,
    #max_seq_length = 4096, # Or your model's max_seq_length
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    load_in_8bit = load_in_8bit,
    # token = "hf_..." # Add your Hugging Face token if needed
)

print("Base model loaded.")

# 3. Load the LoRA Adapters onto the Base Model
# IMPORTANT: Use FastVisionModel.from_pretrained AGAIN, passing the loaded `model` object
# and specifying the adapter path via `model_name`.
print(f"Loading adapters from: {adapter_path}")
model = FastVisionModel.from_pretrained(
    model = model_ori, # Pass the already loaded base model
    model_name = adapter_path, # Path to your LoRA adapters
    # token = "hf_..." # Add your Hugging Face token if needed
)
print("Adapters loaded.")

# 4. Merge the Adapters
print("Merging adapters...")
# This merges the LoRA weights into the base model weights and returns a standard Hugging Face model.
# The model is often upcasted to bfloat16 or float16 during merging.
model = model.merge_and_unload()
print("Adapters merged and unloaded.")

# Check the final model's dtype (usually bfloat16 or float16 after merge)
print(f"Merged model dtype: {next(model.parameters()).dtype}")

# 5. Save the Merged Model and Tokenizer
print(f"Saving merged model to: {merged_model_save_path}")
os.makedirs(merged_model_save_path, exist_ok=True)
model.save_pretrained(merged_model_save_path)
tokenizer.save_pretrained(merged_model_save_path)
print("Merged model and tokenizer saved successfully.")

# Optional: Test loading the merged model independently
print("\nTesting loading the merged model...")
from transformers import AutoModelForCausalLM, AutoTokenizer # Use standard transformers now

try:
    merged_model = AutoModelForCausalLM.from_pretrained(
        merged_model_save_path,
        torch_dtype=torch.bfloat16, # Or torch.float16, match the saved precision
        device_map="auto"
    )
    merged_tokenizer = AutoTokenizer.from_pretrained(merged_model_save_path)
    print("Merged model loaded successfully using transformers.")
    # You can now use `merged_model` and `merged_tokenizer` for inference
except Exception as e:
    print(f"Error loading merged model: {e}")
