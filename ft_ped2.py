from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch
import os
from PIL import Image
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

model, tokenizer = FastVisionModel.from_pretrained(
    #"./Qwen2-VL-7B-Instruct",
    "./Qwen2.5-VL-7B-Instruct",
    #"./Mistral-Small-3.1-24B-Instruct-2503",
    #"./gemma-3-27b-it",
    #"./Qwen2.5-VL-32B-Instruct",
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 16,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)

def parse_data_file(file_path, image_dir='.'):
    """
    Parse a data file where each line contains an image file name, instruction, and answer,
    separated by tabs, and return a list of dictionaries in the specified format.
    
    Args:
        file_path (str): Path to the data file.
        image_dir (str): Directory containing the image files (default is current directory).
    
    Returns:
        list: List of dictionaries, each containing user and assistant messages.
    """
    # Cache to store loaded images and avoid reloading the same image
    image_cache = {}
    
    # Read the file with UTF-8 encoding
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # List to store the parsed dictionaries
    result = []

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
    
    # Process each line
    for line in lines:
        # Remove leading/trailing whitespace and split by tab
        parts = line.strip().split('\t')
        
        # Skip lines that don’t have exactly 3 parts
        if len(parts) != 2:
            continue
        
        # Extract the three parts
        image_file, answer = parts
        
        # Construct the full path to the image
        image_path = os.path.join(image_dir, image_file)
        
        # Load the image if not already in cache
        if image_file not in image_cache:
            image_cache[image_file] = Image.open(image_path)
            image_cache[image_file].load()
        image = image_cache[image_file]
        
        # Create the dictionary in the required format
        message = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image", "image": image}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer}
                    ]
                }
            ]
        }
        
        # Add the dictionary to the result list
        result.append(message)
    
    return result
converted_dataset = parse_data_file('./ped_data/train_fuse_20250328.txt', './ped_data/ped_train')

FastVisionModel.for_training(model) # Enable for training!

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
    train_dataset = converted_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        #max_steps = 1000,
        num_train_epochs = 6, # Set this instead of max_steps for full training runs
        learning_rate = 2e-4,
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs_ped2",
        save_strategy = "steps",
        save_steps = 1000,
        report_to = "none",     # For Weights and Biases

        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 4,
        max_seq_length = 2048,
    ),
)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

#trainer_stats = trainer.train()
trainer_stats = trainer.train(resume_from_checkpoint = True)

#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


model.save_pretrained("lora_model") # Local saving
tokenizer.save_pretrained("lora_model")
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving
