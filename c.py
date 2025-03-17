import base64
import requests
import argparse
import sys
import os
from PIL import Image
from io import BytesIO

def preprocess_image(image, min_size=28):
    """
    Preprocess an image to ensure both dimensions are at least min_size pixels.
    
    Args:
        image: PIL Image object or file path (str)
        min_size: Minimum size for both width and height (default: 28)
    
    Returns:
        PIL Image object with dimensions adjusted if necessary
    """
    # If image is a file path, open it
    if isinstance(image, str):
        image = Image.open(image)
    
    # Ensure image is in RGB mode (common requirement for vision models)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Get current dimensions
    width, height = image.size
    
    # Check if resizing is needed
    if width < min_size or height < min_size:
        # Calculate scaling factor to make the smaller dimension at least min_size
        if width < height:
            scaling_factor = min_size / width
            new_width = min_size
            new_height = int(height * scaling_factor)
        else:
            scaling_factor = min_size / height
            new_height = min_size
            new_width = int(width * scaling_factor)
        
        # Resize the image with high-quality resampling
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return image

def main():
    # Define the available prompts
    prompt1 = '''图中是一幅三轮车的目标图，描述一下图中目标属性, 三轮车车身颜色，颜色类型枚举如下：
    {
    "车身颜色": {
        "不确定": 0,
        "红色": 1,
        "黄色": 2,
        "蓝色": 3,
        "绿色": 4,
        "灰色": 5,
        "白色": 6,
        "黑色": 7,
        "银色": 8,
        "橙色": 9,
        "金色": 10,
        "棕色": 11,
        "紫色": 12,
        "粉色": 13
    },
    输出时，按照json格式，包含三个最可能的颜色ID，并输出对应颜色ID的置信度，并按照置信度排序
    示例如下:
        {
            '3': 0.92,
            '1': 0.21,
            '10': 0.05
        }
    只输出json，不要返回其他无关信息
'''

    prompt2 = '''输出途中三轮车目标颜色：颜色类型枚举如下：
    {
    "车身颜色": {
        "不确定": 0,
        "红色": 1,
        "黄色": 2,
        "蓝色": 3,
        "绿色": 4,
        "灰色": 5,
        "白色": 6,
        "黑色": 7,
        "银色": 8,
        "橙色": 9,
        "金色": 10,
        "棕色": 11,
        "紫色": 12,
        "粉色": 13
    }'''

    prompt3 = '''输出图中三轮车目标颜色,颜色类型枚举如下：
    {
    "车身颜色": [
        "不确定",
        "红色",
        "黄色",
        "蓝色",
        "绿色",
        "灰色",
        "白色",
        "黑色",
        "银色",
        "橙色",
        "金色",
        "棕色",
        "紫色",
        "粉色"
        ]
    输出格式为一行json，为一个map,key为颜色名称，value为对应的置信度。不要输出markdown，即不输出\'\'\'json \'\'\', 仅输出json。
    输出示例如下，包含颜色与对应置信度，按照置信度从高到底排列，输出前三个最可能的颜色：
    {'灰色':0.95, '白色':0.05, '黑色':0.01}
    
    }'''

    prompt4 = '''输出图中三轮车目标颜色,颜色类型枚举如下：
    {
    "车身颜色": [
        "不确定",
        "红色",
        "黄色",
        "蓝色",
        "绿色",
        "灰色",
        "白色",
        "黑色",
        "银色",
        "橙色",
        "金色",
        "棕色",
        "紫色",
        "粉色"
        ]
    }'''

    prompt5 = '''输出图中三轮车目标颜色,从以下颜色列表中选择一个最可能的颜色，如果是晚上或者比较模糊的情况，仔细分辨可能的颜色
    
    "车身颜色": [
        "不确定",
        "红色",
        "黄色",
        "蓝色",
        "绿色",
        "灰色",
        "白色",
        "黑色",
        "银色",
        "橙色",
        "金色",
        "棕色",
        "紫色",
        "粉色"
        ]
    仅输出枚举类型中的颜色
    '''

    # Dictionary mapping prompt IDs to prompts
    prompts = {
        "1": prompt1,
        "2": prompt2,
        "3": prompt3,
        "4": prompt4,
        "5": prompt5
    }

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Send image and prompt to VLLM server and print response.")
    parser.add_argument("--image", required=True, help="Path to the image file (e.g., image.jpg).")
    parser.add_argument("--prompt_id", required=True, help="ID of the prompt to use (e.g., 1, 2, 3, etc.).")
    parser.add_argument("--host", default="10.10.96.50", help="Host of the VLLM server (default: 10.10.96.50).")
    parser.add_argument("--port", default=8001, type=int, help="Port of the VLLM server (default: 8001).")
    args = parser.parse_args()

    # Check if the provided prompt_id is valid
    if args.prompt_id not in prompts:
        print(f"Error: Invalid prompt_id '{args.prompt_id}'. Available IDs: {list(prompts.keys())}")
        sys.exit(1)

    # Determine the image MIME type based on file extension
    image_extension = os.path.splitext(args.image)[1].lower()
    if image_extension in [".jpg", ".jpeg"]:
        mime_type = "image/jpeg"
    elif image_extension == ".png":
        mime_type = "image/png"
    else:
        print(f"Error: Unsupported image format: {image_extension}")
        sys.exit(1)

    # Preprocess and encode the image
    try:
        processed_image = preprocess_image(args.image)
        buffered = BytesIO()
        if mime_type == "image/jpeg":
            processed_image.save(buffered, format="JPEG")
        elif mime_type == "image/png":  # Fixed typo from original
            processed_image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error processing image: {e}")
        sys.exit(1)

    # Construct the API URL
    url = f"http://{args.host}:{args.port}/v1/chat/completions"

    # Construct the request body using the selected prompt
    request_body = {
        "model": "Qwen2.5-VL-72B-Instruct-AWQ",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompts[args.prompt_id]},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
                ]
            }
        ],
        "temperature": 0.0
    }

    # Send the request to the server
    try:
        response = requests.post(url, json=request_body, timeout=60)
        response.raise_for_status()  # Raise an exception for HTTP errors
        response_json = response.json()
        # Extract and print the response content
        print(response_json["choices"][0]["message"]["content"])
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()