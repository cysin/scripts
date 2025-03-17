import json

# Step 1: Parse labeld_data.txt to build a dictionary of correct colors
correct_colors = {}
with open("labeld_data.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            parts = line.split(" ", 1)  # Split on the first space only
            if len(parts) == 2:
                filename, json_str = parts
                try:
                    data = json.loads(json_str)
                    correct_color = data.get("车身颜色", "不确定")
                    correct_colors[filename] = correct_color
                except json.JSONDecodeError:
                    print(f"Error parsing JSON for {filename}")

# Step 2: Parse same.txt and identify incorrect lines
incorrect_lines = []
with open("same2.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            parts = line.split()  # Split on whitespace
            if len(parts) == 2:
                path, detected_color = parts
                filename = path.split("/")[-1]  # Extract filename from path
                if filename in correct_colors:
                    correct_color = correct_colors[filename]
                    # Only consider lines where the correct color is definite
                    if correct_color != "不确定" and detected_color != correct_color:
                        # Append the correct color to the original line
                        incorrect_lines.append(f"{line} (correct: {correct_color})")

# Step 3: Output the incorrect lines
print("Incorrect lines:")
for line in incorrect_lines:
    print(line)
