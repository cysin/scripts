import os
import subprocess
from multiprocessing import Pool
import argparse

def process_image(image_path, prompt_id):
    """
    Run 'c.py' on a given image path with a prompt_id and return the image path and JSON result.
    
    Args:
        image_path (str): Path to the image file.
        prompt_id (str): Prompt ID to pass to c.py.
    
    Returns:
        tuple: (image_path, json_result) where json_result is the output from c.py.
    
    Raises:
        Exception: If c.py fails to execute successfully.
    """
    result = subprocess.run(
        ['python', 'c.py', '--image', image_path, '--prompt_id', prompt_id],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise Exception(f"Error processing {image_path}: {result.stderr}")
    json_result = result.stdout.strip()
    return image_path, json_result

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process images in parallel using c.py')
    parser.add_argument('--image_dir', required=True, help='Directory containing images')
    parser.add_argument('--parallel_num', type=int, required=True, help='Number of parallel processes')
    parser.add_argument('--output_file', required=True, help='Output text file to log results')
    parser.add_argument('--prompt_id', required=True, help='Prompt ID to pass to c.py')
    args = parser.parse_args()

    # Assign arguments to variables
    image_dir = args.image_dir
    parallel_num = args.parallel_num
    output_file = args.output_file
    prompt_id = args.prompt_id

    # Get list of image files from the directory
    image_files = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, f))
    ]

    # Open the output file once in write mode
    with open(output_file, 'w', encoding='utf-8') as f:
        # Define callback function to write results as they arrive
        def write_result(result):
            image_path, json_result = result
            txt = f"{image_path} {json_result}\n"
            print(txt)
            f.write(txt)
            f.flush()  # Ensure the write is flushed to disk immediately

        # Define error callback to handle exceptions
        def handle_error(e):
            print(f"Error: {e}")

        # Process images in parallel using a multiprocessing Pool
        with Pool(parallel_num) as pool:
            # Submit each image processing task asynchronously
            for image_path in image_files:
                pool.apply_async(
                    process_image,
                    args=(image_path, prompt_id),
                    callback=write_result,
                    error_callback=handle_error
                )
            # Close the pool and wait for all tasks to complete
            pool.close()
            pool.join()