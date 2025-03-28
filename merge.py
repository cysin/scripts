from unsloth import FastVisionModel
from peft import PeftModel

# Assuming 'model' is your fine-tuned model after training with Unsloth
# If you’ve saved it earlier, reload it like this:
model = FastVisionModel.from_pretrained("./Qwen2.5-VL-7B-Instruct", load_in_4bit=True)  # Load original model
model = PeftModel.from_pretrained(model, "./outputs_ped/checkpoint-2000")     # Load fine-tuned adapter

# Merge the LoRA adapter into the base model
merged_model = model.merge_and_unload()

# Save the merged model to a directory
merged_model.save_pretrained("./merged/qwen25vl7bft")
