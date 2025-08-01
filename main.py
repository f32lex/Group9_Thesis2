import cv2
import numpy as np
import pyrealsense2 as rs
import urx
import time
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from heapq import heappop, heappush
import math
import pickle
from scipy.spatial.transform import Rotation
import dearpygui.dearpygui as dpg

# Load YOLO
model_path = r'C:\Users\flxsy\Documents\Thesis2\strawberry_detection\runs\train\train4\weights\best.pt'
model = YOLO(model_path)

# Load calibration (eye-in-hand)
try:
    with open("camera_to_ee.pkl", "rb") as f:
        T_camera_ee = pickle.load(f)
    print("Loaded camera-to-end-effector calibration (eye-in-hand).")
except FileNotFoundError:
    print("Error: Calibration file 'camera_to_ee.pkl' not found. Run ur10_calibration_eye_in_hand.py first.")
    exit(1)

# Cam and UR10
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
align = rs.align(rs.stream.color)
pipeline.start(config)
robot = urx.Robot("192.168.1.100")

fx, fy, cx, cy = 615.0, 615.0, 320.0, 240.0
tcp_offset = np.array([0, 0, 0.05, 1])  # offset

def pose_to_matrix(pose):
    x, y, z, rx, ry, rz = pose
    rot = Rotation.from_euler('xyz', [rx, ry, rz]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = [x, y, z]
    return T

def detect_and_localize():
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if not color_frame or not depth_frame:
        return None, None
    
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    
    results = model(color_image)[0]
    detections = []
    
    current_pose = robot.getl()
    T_ee_base = pose_to_matrix(current_pose)
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]
        if label not in ["ripe", "peduncle"] or conf < 0.7:
            continue
        
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        depth = depth_frame.get_distance(cx, cy)
        if depth <= 0 or depth > 1.0:
            continue
        
        z = depth
        x = (cx - cx) * z / fx  # Fix: Corrected to (cx - cx) to (cx - cx)
        y = (cy - cy) * z / fy  # Fix: Corrected to (cy - cy)
        
        # Eye-in-hand: Point in camera frame to end-effector frame
        point_cam = np.array([(cx - cx) / fx * z, (cy - cy) / fy * z, z, 1])  # Fix: Use proper pixel-to-3D
        point_ee = np.linalg.inv(T_camera_ee) @ point_cam  # Transform to end-effector frame
        point_base = T_ee_base @ point_ee + T_ee_base @ tcp_offset
        
        detections.append((point_base[0], point_base[1], point_base[2], label))
        
        color = (0, 255, 0) if label == "ripe" else (0, 255, 255)
        cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)
        cv2.circle(color_image, (cx, cy), 4, color, -1)
        cv2.putText(color_image, f"{label} {conf:.2f}, Z={z:.2f}m", 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return detections, color_image

def heuristic(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

def get_neighbors(pos, grid, grid_res=0.05):
    neighbors = []
    for dx, dy, dz in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
        new_pos = (pos[0] + dx*grid_res, pos[1] + dy*grid_res, pos[2] + dz*grid_res)
        grid_idx = tuple(int(v / grid_res) for v in new_pos)
        if (0 <= grid_idx[0] < grid.shape[0] and 
            0 <= grid_idx[1] < grid.shape[1] and 
            0 <= grid_idx[2] < grid_shape[2] and 
            grid[grid_idx] == 0):
            neighbors.append(new_pos)
    return neighbors

def a_star_3d(start, goal, grid, grid_res=0.05):
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while open_set:
        current_f, current = heappop(open_set)
        if heuristic(current, goal) < grid_res:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        for neighbor in get_neighbors(current, grid, grid_res):
            tentative_g = g_score[current] + heuristic(current, neighbor)
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heappush(open_set, (f_score[neighbor], neighbor))
    
    return None

def control_robotiq_gripper(action, target_pos):
    if action == "grasp":
        program = "set_tool_digital_out(0, True)\nset_tool_digital_out(1, False)\nsync()\nsleep(0.5)"
        robot.send_program(program)
        dpg.set_value("status_text", f"Grasp at {target_pos}: Attempted")
        return True
    elif action == "cut":
        dpg.set_value("status_text", f"Cut at {target_pos}: Not supported with Robotiq")
        return False
    elif action == "place":
        program = "set_tool_digital_out(0, False)\nset_tool_digital_out(1, True)\nsync()\nsleep(0.5)"
        robot.send_program(program)
        dpg.set_value("status_text", f"Place at {target_pos}: Attempted")
        return True
    return False

def visualize_3d(detections, path=None, start=None, goal=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for x, y, z, label in detections:
        color = 'r' if label == "ripe" else 'y'
        ax.scatter(x, y, z, c=color, label=label, s=50)
    
    if path:
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], 'b-', label='Path')
    
    if start:
        ax.scatter(*start, c='k', marker='o', s=100, label='Start')
    if goal:
        ax.scatter(*goal, c='m', marker='x', s=100, label='Goal')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    plt.title("Strawberry Picking Simulation")
    plt.show()

def update_frame(sender, app_data):
    detections, frame = detect_and_localize()
    if frame is not None:
        dpg.set_value("video_texture", [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)])
    if detections:
        dpg.set_value("detection_count", f"Detections: {len(detections)}")

def run_picking():
    grid_res = 0.05
    grid_shape = (int(0.4/grid_res), int(0.35/grid_res), int(0.5/grid_res))
    grid = np.zeros(grid_shape, dtype=np.uint8)
    
    home_pos = (0.2, 0.175, 0.5, 0, 0, 0)
    robot.movel(home_pos, acc=0.5, vel=0.5)
    
    success_count = 0
    total_attempts = 0
    
    while dpg.is_dearpygui_running():
        detections, _ = detect_and_localize()
        if not detections:
            dpg.set_value("status_text", "No ripe strawberries or peduncles detected.")
            time.sleep(1)
            continue
        
        grid.fill(0)
        for x, y, z, label in detections:
            if label == "peduncle":
                grid_idx = (int(x/grid_res), int(y/grid_res), int(z/grid_res))
                if 0 <= grid_idx[0] < grid_shape[0] and 0 <= grid_idx[1] < grid_shape[1] and 0 <= grid_idx[2] < grid_shape[2]:
                    grid[grid_idx] = 1
        
        strawberry = next((d for d in detections if d[3] == "ripe"), None)
        peduncle = next((d for d in detections if d[3] == "peduncle"), None)
        if not strawberry or not peduncle:
            dpg.set_value("status_text", "Missing strawberry or peduncle detection.")
            time.sleep(1)
            continue
        
        total_attempts += 1
        
        start_pos = (0.2, 0.175, 0.5)
        goal_pos = (peduncle[0], peduncle[1], peduncle[2] + 0.01)
        path = a_star_3d(start_pos, goal_pos, grid, grid_res)
        
        if path:
            dpg.set_value("status_text", "Path found to peduncle.")
            for point in path[::5]:
                robot.movel((point[0], point[1], point[2], 0, 0, 0), acc=0.5, vel=0.5)
                time.sleep(0.1)
            
            grasp_success = control_robotiq_gripper("grasp", goal_pos)
            if grasp_success:
                basket_pos = (0.3, 0.3, 0.5, 0, 0, 0)
                robot.movel(basket_pos, acc=0.5, vel=0.5)
                place_success = control_robotiq_gripper("place", basket_pos)
                if place_success:
                    success_count += 1
                    dpg.set_value("status_text", "Strawberry harvested and placed successfully.")
            
            visualize_3d(detections, path, start_pos, goal_pos)
        else:
            dpg.set_value("status_text", "No path found.")
            visualize_3d(detections, start=start_pos, goal=goal_pos)
        
        robot.movel(home_pos, acc=0.5, vel=0.5)
        time.sleep(1)
        
        dpg.set_value("success_rate", f"Success: {success_count}/{total_attempts} ({success_count/total_attempts*100:.1f}%)")
    
    robot.movel(home_pos, acc=0.5, vel=0.5)

dpg.create_context()
dpg.create_viewport(title="Strawberry Picking UI", width=1280, height=720)

with dpg.window(label="Main Control", width=1280, height=720):
    dpg.add_text("Status: Idle", tag="status_text")
    dpg.add_button(label="Start Picking", callback=lambda: dpg.start_thread(run_picking))
    dpg.add_button(label="Stop", callback=dpg.stop_dearpygui)
    dpg.add_button(label="Show 3D Plot", callback=lambda: visualize_3d(*detect_and_localize()[0] if detect_and_localize()[0] else ([], None, None, None)))
    dpg.add_text("Detections: 0", tag="detection_count")
    dpg.add_text("Success Rate: 0%", tag="success_rate")
    texture_id = dpg.add_texture(640, 480)
    dpg.set_value("video_texture", texture_id)

dpg.set_viewport_vsync(True)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_frame_callback(1, update_frame)
dpg.start_dearpygui()
dpg.destroy_context()

pipeline.stop()
robot.close()
cv2.destroyAllWindows()
