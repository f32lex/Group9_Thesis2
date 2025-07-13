from ultralytics import YOLO
import os
import shutil

def clean_label_folder(label_dir):
    print(f"\n🧹 Cleaning label files in: {label_dir}")
    deleted_count = 0
    cleaned_count = 0

    if not os.path.exists(label_dir):
        print(f"❌ Folder does not exist: {label_dir}")
        return

    for file in os.listdir(label_dir):
        if not file.endswith('.txt'):
            continue

        path = os.path.join(label_dir, file)

        # Delete empty files
        if os.path.getsize(path) == 0:
            os.remove(path)
            print(f"🗑️ Deleted empty file: {file}")
            deleted_count += 1
            continue

        with open(path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        cleaned = []
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                cls, x, y, w, h = map(float, parts)
                if 0 < w <= 1 and 0 < h <= 1:
                    cleaned.append(line)
            except:
                continue

        if cleaned:
            with open(path, 'w') as f:
                f.write('\n'.join(cleaned) + '\n')
            cleaned_count += 1
        else:
            os.remove(path)
            print(f"🗑️ Deleted file with only invalid labels: {file}")
            deleted_count += 1

    print(f"✅ Cleaning done: {cleaned_count} valid, {deleted_count} deleted.")

# ========= PARAMETERS =========
base_model = 'yolov8n.pt'
data_yaml = r"C:\Users\flxsy\Documents\Thesis2\strawberry_detection\strawberry.yaml"
output_dir = r"C:\Users\flxsy\Documents\Thesis2\strawberry_detection\models"

# Label folders to clean
train_label_dir = r"C:\Users\flxsy\Documents\Thesis2\strawberry_detection\datasets\labels\train"
val_label_dir   = r"C:\Users\flxsy\Documents\Thesis2\strawberry_detection\datasets\labels\val"
test_label_dir  = r"C:\Users\flxsy\Documents\Thesis2\strawberry_detection\datasets\labels\test"

epochs = 50
img_size = 640
batch_size = 16
device = 'cpu'
# ==============================

# Step 1: Clean label folders
clean_label_folder(train_label_dir)
clean_label_folder(val_label_dir)
clean_label_folder(test_label_dir)

# Step 2: Train the model
model = YOLO(base_model)

results = model.train(
    data=data_yaml,
    epochs=epochs,
    imgsz=img_size,
    batch=batch_size,
    device=device,
    patience=30,
    project="runs/train",
    name="strawberry_yolov8",
    augment=True,
    exist_ok=True,
    save=True
)

# Step 3: Save best model
trained_best_model = "runs/train/strawberry_yolov8/weights/best.pt"
os.makedirs(output_dir, exist_ok=True)
permanent_best_model = os.path.join(output_dir, "strawberry_best.pt")

if os.path.exists(trained_best_model):
    shutil.copy(trained_best_model, permanent_best_model)
    print(f"\n✅ Best model saved to: {permanent_best_model}")
else:
    print("\n⚠️ Training may have failed. best.pt not found.")
