import os
import sys
from PIL import Image
from unsloth import FastVisionModel

def main():
    # Check if the directory path is provided as a command-line argument
    if len(sys.argv) <3:
        print("Usage: python script.py /path/to/image/directory")
        sys.exit(1)
    
    # Get the directory path and verify it exists
    directory_path = sys.argv[1]
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a directory")
        sys.exit(1)

    # Get the directory path and verify it exists
    lora_path = sys.argv[2]
    if not os.path.isdir(lora_path):
        print(f"Error: {lora_path} is not a directory")
        sys.exit(1)

    
    # Define a list of common image file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    
    # Get all files in the directory and filter for image files
    files = os.listdir(directory_path)
    image_files = [
        f for f in files
        if os.path.isfile(os.path.join(directory_path, f)) and
        os.path.splitext(f)[1].lower() in image_extensions
    ]



    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = lora_path, # YOUR MODEL YOU USED FOR TRAINING
        load_in_4bit = True, # Set to False for 16bit LoRA
    )
    FastVisionModel.for_inference(model) # Enable for inference!


    # Iterate through each image file and load it
    for file in image_files:
        try:
            # Open the file in binary mode and load the image
            image_path = os.path.join(directory_path, file)

            with open(image_path, 'rb') as f:
                image = Image.open(f)
                image.load()  # Explicitly load the image data into memory
                # Get original size
                width, height = image.size

                # If either dimension is smaller than 28, resize while keeping aspect ratio
                if width < 28 or height < 28:
                    scale = 28 / min(width, height)  # Scale based on the smaller dimension
                    new_size = (int(width * scale), int(height * scale))
                    image = image.resize(new_size, Image.Resampling.LANCZOS)

                #print(f"Loaded {file}: {image.size}")
                instruction = '''
输出图中目标人物的上衣颜色，仅输出以下选项中最确定的一个["不确定","红色","橙色","黄色","绿色","蓝色","紫色","粉色","棕色","灰色","白色","黑色","花色"]
'''
                instruction = '''
输出图中目标人物是否拉东西以及拉东西的样式，仅输出以下选项中最确定的一个["不确定","行李箱","婴儿车","轮椅","推拉物品"]，如果没有拉东西，输出"不确定"
'''
                instruction = '''
输出图中目标人物的发型，仅输出以下选项中最确定的一个["不确定","光头","平头","短发","齐耳短发","长发","扎辫子","谢顶"]，如果不能确定输出"不确定"
'''

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": instruction},
                            {"type": "image", "image": image}
                        ]
                    }
                ]
                input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
                #input_text2 = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt = True)
                #print(input_text2)
                inputs = tokenizer(
                    image,
                    input_text,
                    add_special_tokens = False,
                    return_tensors = "pt",
                ).to("cuda")

                # Get the length of the input sequence
                input_length = inputs.input_ids.size(1)
            
                # Set the end-of-sequence token ID to stop generation
                eos_token_id = tokenizer.eos_token_id

                # Generate output
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,          # Small value sufficient for short answers
                    do_sample=False,            # Greedy decoding for determinism
                    repetition_penalty=1.1,     # Prevent token repetition
                    eos_token_id=eos_token_id   # Stop at end-of-sequence token
                )
                # Decode only the generated tokens
                generated_ids = outputs[0, input_length:]
                caption = tokenizer.decode(generated_ids, skip_special_tokens=True)
                print(image_path)
                print(caption)
                print("\n\n")

                #outputs = model.generate(**inputs, max_new_tokens = 10, do_sample=False, repetition_penalty=1.1)
                #outputs = model.generate(**inputs, max_new_tokens = 10, do_sample=False)
                #outputs = model.generate(**inputs, max_new_tokens = 10, temperature = 0.0000001, do_sample=False)
                #outputs = model.generate(**inputs, max_new_tokens = 128, use_cache = True, temperature = 0.0000001, repetition_penalty=1.2)
                #outputs = model.generate(**inputs, max_new_tokens = 128, use_cache = True, temperature = 0.0, do_sample = false)

#text = "输出图中目标人物的上衣颜色，仅输出以下选项中最确定的一个[\"不确定\",\"红色\",\"橙色\",\"黄色\",\"绿色\",\"蓝色\",\"紫色\",\"粉色\",\"棕色\",\"灰色\",\"白色\",\"黑色\",\"花色\"]"
#inputs = processor(images=image, text=text, return_tensors="pt").to(device)

# Generate output
#with torch.no_grad():
#    outputs = model.generate(
#        **inputs,
#        max_new_tokens=10,    # Short output
#        temperature=0.7,      # Less randomness
#        do_sample=False       # Greedy decoding
#    )

# Decode output
#caption = processor.decode(outputs[0], skip_special_tokens=True)
#print(caption)  # Expected: e.g., "黑色"


                #print(tokenizer.decode(outputs[0], skip_special_tokens=True))
                #print(tokenizer.batch_decode(outputs))
                #print("\n\n")


        except Exception as e:
            print(f"Error loading {file}: {e}")

if __name__ == "__main__":
    main()

