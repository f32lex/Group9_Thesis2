import os
import os

label_dir = r'C:\Users\flxsy\Documents\Thesis2\strawberry_detection\datasets\labels\train'

for file in os.listdir(label_dir):
    if file.endswith('.txt'):
        path = os.path.join(label_dir, file)
        with open(path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            print(f'❌ {file} is empty or whitespace only')
        else:
            for i, line in enumerate(lines):
                try:
                    parts = list(map(float, line.strip().split()))
                    if len(parts) != 5:
                        raise ValueError
                    _, xc, yc, w, h = parts
                    if not (0 < w <= 1 and 0 < h <= 1):
                        print(f'⚠️ {file} line {i+1} has invalid bbox size: {w}, {h}')
                except:
                    print(f'❌ {file} line {i+1} has invalid format: "{line}"')


image_dir = r'C:\Users\flxsy\Documents\Thesis2\strawberry_detection\datasets\train'
label_dir = r'C:\Users\flxsy\Documents\Thesis2\strawberry_detection\datasets\labels\train'

image_files = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
label_files = [os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.txt')]

missing = set(image_files) - set(label_files)

if missing:
    print("❌ These images have no label files:")
    for name in missing:
        print(f"- {name}.jpg")
else:
    print("✅ All validation images have matching labels.")
