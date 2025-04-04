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
            print(image_path)
            print("\n\n")
            with open(image_path, 'rb') as f:
                image = Image.open(f)
                image.load()  # Explicitly load the image data into memory
                #print(f"Loaded {file}: {image.size}")
                instruction = '''
输出图中目标人物的上衣颜色，仅输出以下选项中最确定的一个["不确定","红色","橙色","黄色","绿色","蓝色","紫色","粉色","棕色","灰色","白色","黑色","花色"]
'''
                instruction = '''
输出图中目标人物是否拉东西以及拉东西的样式，仅输出以下选项中最确定的一个["不确定","行李箱","婴儿车","轮椅","推拉物品"]，如果没有拉东西，输出"不确定"
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
                inputs = tokenizer(
                    image,
                    input_text,
                    add_special_tokens = False,
                    return_tensors = "pt",
                ).to("cuda")


                outputs = model.generate(**inputs, max_new_tokens = 10, do_sample=False, repetition_penalty=1.1)
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


                print(tokenizer.decode(outputs[0], skip_special_tokens=True))
                #print(tokenizer.batch_decode(outputs))
                print("\n\n")


        except Exception as e:
            print(f"Error loading {file}: {e}")

if __name__ == "__main__":
    main()

