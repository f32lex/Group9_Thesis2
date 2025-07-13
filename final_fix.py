from collections import Counter
import os

label_dir = r"C:\Users\flxsy\Documents\Thesis2\strawberry_detection\datasets\train\labels"
counter = Counter()

for file in os.listdir(label_dir):
    if file.endswith(".txt"):
        with open(os.path.join(label_dir, file)) as f:
            for line in f:
                class_id = line.strip().split()[0]
                counter[class_id] += 1

print("🔍 Class Distribution in Training Set:")
for cls, count in sorted(counter.items()):
    print(f"Class {cls} → {count} instances")
