import os
import json

# Paths
ANN_DIR = r'C:\Users\flxsy\Documents\Thesis2\strawberry_detection\datasets\test\ann'
IMG_DIR = r'C:\Users\flxsy\Documents\Thesis2\strawberry_detection\datasets\test'
LABELS_DIR = r'C:\Users\flxsy\Documents\Thesis2\strawberry_detection\datasets\test\labels'
META_PATH = r'C:\Users\flxsy\Documents\Thesis2\strawberry_detection\datasets\meta.json'

# Make output dir
os.makedirs(LABELS_DIR, exist_ok=True)

# Load class mapping from meta.json
with open(META_PATH, 'r') as f:
    meta = json.load(f)
class_titles = [c['title'] for c in meta['classes']]
class_title_to_id = {title: i for i, title in enumerate(class_titles)}

# For each annotation file
for fname in os.listdir(ANN_DIR):
    if not fname.endswith('.json'):
        continue
    ann_path = os.path.join(ANN_DIR, fname)
    with open(ann_path, 'r') as f:
        ann = json.load(f)
    img_w = ann['size']['width']
    img_h = ann['size']['height']
    yolo_lines = []
    for obj in ann['objects']:
        class_title = obj['classTitle']
        if class_title not in class_title_to_id:
            continue  # skip unknown classes
        class_id = class_title_to_id[class_title]
        (x1, y1), (x2, y2) = obj['points']['exterior']
        # Convert to YOLO format (normalized)
        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        width = abs(x2 - x1) / img_w
        height = abs(y2 - y1) / img_h
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    # Write to .txt file (same basename as image, without extension like .jpg or .png)
    base = os.path.splitext(fname)[0]
    if base.endswith('.jpg') or base.endswith('.png'):
        base = os.path.splitext(base)[0]
    out_name = base + '.txt'
    out_path = os.path.join(LABELS_DIR, out_name)
    with open(out_path, 'w') as f:
        f.write('\n'.join(yolo_lines) + '\n') 