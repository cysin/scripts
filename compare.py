import sys

# Function to process a file and return a set of lines excluding those with color '不确定'
def process_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    filtered_lines = set()
    for line in lines:
        stripped = line.strip()  # Remove leading/trailing whitespace
        if stripped:  # Ensure the line is not empty
            parts = stripped.split()  # Split by whitespace
            if parts and parts[-1] != '不确定':  # Check if the last word is not '不确定'
                filtered_lines.add(stripped)  # Add the full line to the set
    return filtered_lines

# Check if two file names are provided as command-line arguments
if len(sys.argv) != 3:
    print("Usage: python script.py file1 file2")
    sys.exit(1)

# Get file names from command-line arguments
file1 = sys.argv[1]
file2 = sys.argv[2]

# Process both files
set1 = process_file(file1)
set2 = process_file(file2)

# Find common lines between the two sets
common_lines = set1.intersection(set2)

# Output the common lines
for line in common_lines:
    print(line)