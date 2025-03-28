import os
import torch # Import torch *after* setting the environment variable if using this method

# --- GPU Selection ---
# Set this BEFORE importing torch or unsloth if possible.
# Choose the GPU index (e.g., 0, 1, etc.)
gpu_index = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
print(f"CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")
# Verify torch only sees the intended GPU
if torch.cuda.is_available():
    print(f"PyTorch sees {torch.cuda.device_count()} CUDA device(s).")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is not available. Running on CPU (merging might be very slow).")
# --------------------

from unsloth import FastLanguageModel
# Other imports remain the same

# 1. Configuration
base_model_name = "Qwen/Qwen2.5-7B-VL-Chat" # Or your specific base model identifier
adapter_path = "./your_finetuned_lora_adapters" # IMPORTANT: Path to your saved LoRA adapter directory
merged_model_save_path = "./qwen2.5-7b-vl-merged" # Directory to save the final merged model

load_in_4bit = True
load_in_8bit = False
dtype = None

print("Starting model loading...")

# 2. Load the Base Model using Unsloth
# Unsloth will automatically use the GPU made visible by CUDA_VISIBLE_DEVICES
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = base_model_name,
    max_seq_length = 4096, # Or your model's max_seq_length
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    load_in_8bit = load_in_8bit,
    # token = "hf_..."
    # device_map = "auto" # Usually not needed if CUDA_VISIBLE_DEVICES is set correctly
)

print("Base model loaded.")
if hasattr(model, 'device'):
    print(f"Model loaded on device: {model.device}")

# 3. Load the LoRA Adapters onto the Base Model
print(f"Loading adapters from: {adapter_path}")
model = FastLanguageModel.from_pretrained(
    model = model, # Pass the already loaded base model
    model_name = adapter_path, # Path to your LoRA adapters
    # token = "hf_..."
)
print("Adapters loaded.")

# 4. Merge the Adapters
print("Merging adapters...")
# Merging might temporarily use more memory (CPU RAM and potentially VRAM)
model = model.merge_and_unload()
print("Adapters merged and unloaded.")

print(f"Merged model dtype: {next(model.parameters()).dtype}")
# After merging, the model is a standard HF model. Check its device placement.
# It might be on CPU or GPU depending on the merge process and available memory.
# You might need to explicitly move it to the GPU later if needed for inference.
print(f"Merged model is on device(s): {model.device if hasattr(model, 'device') else 'Check parameters'}")


# 5. Save the Merged Model and Tokenizer
print(f"Saving merged model to: {merged_model_save_path}")
os.makedirs(merged_model_save_path, exist_ok=True)
# If the merged model ended up on CPU, saving from there is fine.
model.save_pretrained(merged_model_save_path)
tokenizer.save_pretrained(merged_model_save_path)
print("Merged model and tokenizer saved successfully.")

# Optional: Test loading the merged model independently
print("\nTesting loading the merged model...")
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    # When loading the final merged model, you can specify the device again
    # Use device_map="auto" to let transformers handle multi-GPU if available AND visible
    # Or force to a specific visible device like f"cuda:{gpu_index}" if desired and feasible
    merged_model_loaded = AutoModelForCausalLM.from_pretrained(
        merged_model_save_path,
        torch_dtype=torch.bfloat16, # Or torch.float16
        device_map="auto" # Let transformers handle placement on visible GPUs
        # Or explicitly: device_map={"":f"cuda:{gpu_index}"} # Careful with memory
    )
    merged_tokenizer_loaded = AutoTokenizer.from_pretrained(merged_model_save_path)
    print("Merged model loaded successfully using transformers.")
    print(f"Test loaded model is on device(s): {merged_model_loaded.device}")

except Exception as e:
    print(f"Error loading merged model: {e}")
