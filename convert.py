import sys
import json

# Check if the input file name is provided as a command-line argument
if len(sys.argv) != 2:
    print("Usage: python script.py input.txt")
    sys.exit(1)

# Get the input file name from the command-line argument
input_file = sys.argv[1]

# Open and read the file with UTF-8 encoding
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        # Remove leading and trailing whitespace
        line = line.strip()
        if line:  # Process only non-empty lines
            # Split the line into file path and JSON string at the first space
            path, json_str = line.split(' ', 1)
            # Parse the JSON string into a dictionary
            data = json.loads(json_str)
            # Find the color with the highest confidence
            color = max(data, key=data.get)
            # Output the file path and the color
            print(path, color)
