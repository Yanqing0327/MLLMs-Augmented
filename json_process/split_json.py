import json

with open('json_path', 'r') as json_file:
    data = json.load(json_file)

total_items = len(data)
items_per_partition = total_items // 8 

partitions = [] 

for i in range(8):
    start_idx = i * items_per_partition
    end_idx = (i + 1) * items_per_partition if i < 7 else total_items
    partition = data[start_idx:end_idx]
    partitions.append(partition)

for i, partition_data in enumerate(partitions):
    with open(f'/.../partition_{i}.json', 'w') as partition_file:
        json.dump(partition_data, partition_file, indent=4)
