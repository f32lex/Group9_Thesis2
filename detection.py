import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO(r'C:\Users\flxsy\Documents\Thesis2\strawberry_detection\runs\train\train4\weights\best.pt')

def camera_selector():
    width, height = 400, 200
    win_name = "Select Camera (Press 1 or 2)"
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 50

    cv2.putText(canvas, "Press 1 for RealSense", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(canvas, "Press 2 for Webcam", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.imshow(win_name, canvas)

    while True:
        key = cv2.waitKey(0)
        if key == ord('1'):
            cv2.destroyWindow(win_name)
            return True
        elif key == ord('2'):
            cv2.destroyWindow(win_name)
            return False

# DETECTION LOGIC
use_realsense = camera_selector()

if use_realsense:
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    try:
        pipeline.start(config)
        align = rs.align(rs.stream.color)
        print("Using RealSense camera.")
    except Exception as e:
        print("RealSense initialization failed, switching to webcam.")
        use_realsense = False
        cap = cv2.VideoCapture(0)

else:
    cap = cv2.VideoCapture(0)
    print("Using laptop webcam.")

try:
    while True:
        if use_realsense:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
        else:
            ret, color_image = cap.read()
            if not ret:
                break
            depth_frame = None  # No depth with webcam

        results = model(color_image)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if use_realsense and depth_frame:
                depth = depth_frame.get_distance(cx, cy)
            else:
                depth = -1

            color = (0, 255, 0) if label == "ripe" else (0, 0, 255)
            cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)
            cv2.circle(color_image, (cx, cy), 4, color, -1)

            text = f"{label} {conf:.2f}"
            if depth != -1:
                text += f", Z={depth:.2f}m"

            cv2.putText(color_image, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Strawberry Detection", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    if use_realsense:
        pipeline.stop()
    else:
        cap.release()
    cv2.destroyAllWindows()
