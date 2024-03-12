import json

merged_data = []

for i in range(0, 8): 
    filename = f'/.../partition_{i}.json'
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
            merged_data.extend(data)
            print(i)
    except FileNotFoundError:
        print(f"File {filename} not found.")

with open('/xxxxxx.json', 'w') as output_file:
    json.dump(merged_data, output_file)

print("JSON files merged and saved as 'xxxxxx.json'.")
