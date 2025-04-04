from unsloth import FastVisionModel
import torch
import os
from PIL import Image
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from datasets import Dataset # Import Dataset

# --- Model Loading ---
# The 'tokenizer' variable here will actually hold the 'processor'
model, processor = FastVisionModel.from_pretrained(
    "./Qwen2.5-VL-7B-Instruct",
    load_in_4bit = True,
    use_gradient_checkpointing = "unsloth",
    # trust_remote_code = True # Add if needed
)

# --- Token Handling (Accessing the tokenizer *within* the processor) ---
# Use processor.tokenizer to access the underlying tokenizer's attributes/methods
if "<image>" not in processor.tokenizer.vocab:
    added_tokens = processor.tokenizer.add_tokens(["<image>"], special_tokens=True)
    # IMPORTANT: Resize model embeddings if tokens were added
    if added_tokens > 0:
        model.resize_token_embeddings(len(processor.tokenizer))
        print(f"Added {added_tokens} token(s) including <image> and resized embeddings.")
    else:
        print("<image> token already present or not added.")
else:
    print("<image> token already exists in tokenizer vocab.")


# --- PEFT Setup (Keep as is) ---
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True,
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    r = 16,
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# --- Revised Data Parsing Function (Takes processor, uses processor.tokenizer internally) ---
def format_dataset(file_path, image_dir='.', processor=None):
    """
    Parses the data file and formats it using the processor's tokenizer.
    Returns a list of dictionaries with "image" and "text" keys.
    """
    if processor is None:
        raise ValueError("Processor must be provided to format_dataset")

    # Access the actual tokenizer component
    tokenizer = processor.tokenizer

    image_cache = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    instruction = '''
请输出图中目标人物的属性，每个属性标签从对应的枚举数组中选择一个最可能的标签，属性列表及对应的枚举数组以及要求如下：

年龄    ["不确定","少年","中青年","老年"]，如果不能确定输出"不确定"
性别    ["不确定","男","女"]，如果不能确定输出"不确定"
帽子    ["不确定","帽子","头盔","头巾"]，如果没有戴帽子，输出"不确定"
口罩    ["是","否"]
眼镜    ["是","否"]
带包    ["不确定","单肩","双层","斜挎","拎包","拎东西","带包"]，如果没有带包，输出"不确定"
拉东西    ["不确定","行李箱","婴儿车","轮椅","推拉物品"]，如果没有拉东西，输出"不确定"
打伞    ["是","否"]
抱东西    ["不确定","抱孩子","抱东西","扛背东西"]，如果没有抱东西，输出"不确定"
上衣颜色    ["不确定","红色","橙色","黄色","绿色","蓝色","紫色","粉色","棕色","灰色","白色","黑色","花色"]，如果不能确定输出"不确定"
下衣颜色    ["不确定","红色","橙色","黄色","绿色","蓝色","紫色","粉色","棕色","灰色","白色","黑色","花色"]，如果不能确定输出"不确定"
发型    ["不确定","光头","平头","短发","齐耳短发","长发","扎辫子","谢顶"]，如果不能确定输出"不确定"
帽子颜色    ["不确定","红色","橙色","黄色","绿色","蓝色","紫色","粉色","棕色","灰色","白色","黑色","花色"]，如果不能确定输出"不确定"
拍摄方向    ["不确定","正面","背面","侧面"]，如果不能确定输出"不确定"
鞋子颜色    ["不确定","红色","橙色","黄色","绿色","蓝色","紫色","粉色","棕色","灰色","白色","黑色","花色"]，如果不能确定输出"不确定"
衣服纹理    ["不确定","格子","花纹","纯色","条纹","拼色","图案"]，如果不能确定输出"不确定"
上衣款式    ["不确定","短袖","长袖","外套","无袖","雨衣"]，如果不能确定输出"不确定"
下衣款式    ["不确定","长裤","短裤","七分裤","裙子"]，如果不能确定输出"不确定"

输出json格式，例子如下：
{"年龄": "不确定", "性别": "不确定", "帽子": "不确定", "口罩": "否", "眼镜": "否", "带包": "不确定", "拉东西": "不确定", "打伞": "否", "抱东西": "不确定", "上衣颜色": "不确定", "下衣颜色": "不确定", "发型": "不确定", "帽子颜色": "不确定", "拍摄方向": "不确定", "鞋子颜色": "不确定", "衣服纹理": "不确定", "上衣款式": "不确定", "下衣款式": "不确定"}
'''

    formatted_data = []
    for line_num, line in enumerate(lines): # Added line number for better error reporting
        parts = line.strip().split('\t')
        if len(parts) != 2:
            print(f"Warning: Skipping line {line_num+1} due to incorrect format (expected 3 tab-separated parts): {line.strip()}")
            continue

        image_file, answer = parts
        image_path = os.path.join(image_dir, image_file)

        # Simplified image loading/caching
        if image_path not in image_cache:
            try:
                img = Image.open(image_path).convert("RGB")
                img.load()
                image_cache[image_path] = img
            except FileNotFoundError:
                print(f"Warning: Image file not found: {image_path}. Skipping line {line_num+1}.")
                continue
            except Exception as e:
                print(f"Warning: Could not load image {image_path}. Skipping line {line_num+1}. Error: {e}")
                continue
        image = image_cache[image_path]


        messages = [
            {"role": "user", "content": f"{instruction}\n<image>"},
            {"role": "assistant", "content": answer}
        ]

        try:
            # Use the tokenizer (from the processor) to apply the template
            formatted_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        except Exception as e:
             print(f"Error applying chat template on line {line_num+1}: {e}")
             print(f"Messages structure: {messages}")
             # Attempt fallback (Ensure this matches Qwen format)
             try:
                 print(f"Trying fallback template for line {line_num+1}...")
                 formatted_text = f"<|im_start|>user\n{instruction}\n<image><|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
             except Exception as fallback_e:
                 print(f"Fallback template failed for line {line_num+1}: {fallback_e}. Skipping example.")
                 continue

        # Check if <image> token is in the formatted text (it should be)
        if "<image>" not in formatted_text:
             print(f"Warning: <image> token missing in formatted text for line {line_num+1}. Check chat template. Text: {formatted_text[:200]}...")
             # Optionally skip this example if <image> is mandatory and missing
             # continue

        formatted_data.append({"image": image, "text": formatted_text})
        # After creating converted_dataset:
        #if formatted_data:
        #    print("\nSample item from converted_dataset:")
        #    print(formatted_data[0])
        #    print("-" * 20)
        #else:
        #    print("\nError: converted_dataset is empty after processing!")
            # Exit or raise error if dataset is empty
        #    import sys
        #    sys.exit("Exiting due to empty dataset.")

        # Ensure dataset has expected columns (image, text)
        #print(f"Dataset features: {formatted_data.features}")

    print(f"Successfully processed {len(formatted_data)} examples out of {len(lines)} lines.")
    return formatted_data

# --- Load and Format Dataset ---
# Pass the processor to the formatting function
raw_dataset = format_dataset('./ped_data/train_fuse_20250328.txt', './ped_data/ped_train', processor=processor)

# Ensure dataset is not empty
if not raw_dataset:
    raise ValueError("Dataset is empty after processing. Check file paths, formats, and image loading errors.")

converted_dataset = Dataset.from_list(raw_dataset)
# converted_dataset = converted_dataset.shuffle(seed=42)

# --- Model Preparation for Training ---
FastVisionModel.for_training(model)

# --- SFTTrainer Setup ---
# Pass the processor to the Trainer and Collator
trainer = SFTTrainer(
    model=model,
    tokenizer=processor, # Pass the processor here
    data_collator=UnslothVisionDataCollator(model, processor), # Pass the processor here
    train_dataset=converted_dataset,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs_ped_gemini",
        save_strategy="steps",
        save_steps=1000,
        report_to="none",

        remove_unused_columns=False,
        dataset_text_field="text", # Correct field
        # dataset_kwargs = {"skip_prepare_dataset": True}, # Keep removed or commented out
        dataset_num_proc=None, # Set to None or 1 first, especially with PIL images
        max_seq_length=2048,
    ),
)

# --- Training and Saving ---
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# Save the LoRA adapter weights
model.save_pretrained("lora_model32")
# Save the processor (contains tokenizer and image processor config)
processor.save_pretrained("lora_model32")

print("Training finished and model/processor saved.")