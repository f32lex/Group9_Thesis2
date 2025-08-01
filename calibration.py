import cv2
import numpy as np
import pyrealsense2 as rs
import urx
import time
import pickle
from scipy.spatial.transform import Rotation

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
align = rs.align(rs.stream.color)
pipeline.start(config)

robot = urx.Robot("192.168.1.100")

fx, fy, cx, cy = 615.0, 615.0, 320.0, 240.0
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))  # Assume minimal distortion

# Mockfield parameters
CHECKERBOARD = (6, 4)  # 6x4 grid
SQUARE_SIZE = 0.075  # 75 mm squares
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

# Predefined poses for calibration 
calibration_poses = [
    (0.2, 0.175, 0.1, 0, 0, 0),      # Above checkerboard center
    (0.3, 0.175, 0.1, 0, 0, 0.5),    # Right and rotated
    (0.1, 0.175, 0.1, 0, 0, -0.5),   # Left and rotated
    (0.2, 0.3, 0.1, 0.3, 0, 0),      # Forward
    (0.2, 0.05, 0.1, -0.3, 0, 0),    # Backward
    (0.3, 0.3, 0.15, 0, 0.3, 0),     # Up and forward
    (0.1, 0.05, 0.15, 0, -0.3, 0),   # Up and backward
    (0.25, 0.2, 0.2, 0.2, 0.2, 0),   # Angled
    (0.15, 0.15, 0.2, -0.2, -0.2, 0), # Opposite angle
    (0.2, 0.175, 0.05, 0, 0, 0.7),   # Lower and rotated
]

def get_checkerboard_pose(image, depth_frame):
    """Detect checkerboard and compute its 3D pose in camera frame."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    
    if not ret:
        return None, None
    
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                              criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    
    # Solve PnP to get rotation and translation
    ret, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
    if not ret:
        return None, None
    
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.ravel()
    
    return None, T 

def pose_to_matrix(pose):
    """Convert UR10 pose (x, y, z, rx, ry, rz) to 4x4 transformation matrix."""
    x, y, z, rx, ry, rz = pose
    rot = Rotation.from_euler('xyz', [rx, ry, rz]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = [x, y, z]
    return T

def proceed_to_next_pose():
    """Wait for 'c' key to proceed to the next pose."""
    print("Press 'c' to proceed to the next pose, 'q' to skip, or 'e' for emergency stop.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            return True
        elif key == ord('q'):
            return False
        elif key == ord('e'):
            emergency_stop()
            return False

def return_to_start():
    """Return the UR10 to the starting pose in case of failure."""
    start_pose = calibration_poses[0]
    print(f"Returning to start pose: {start_pose}")
    robot.movel(start_pose, acc=0.5, vel=0.5)
    time.sleep(1)

def emergency_stop():
    """Stop the robot and pipeline immediately."""
    print("Emergency stop activated!")
    robot.stop()
    pipeline.stop()
    cv2.destroyAllWindows()
    exit(0)

def main():
    try:
        A_matrices = []  # Target-to-camera transformations (eye-in-hand)
        B_matrices = []  # End-effector-to-base transformations
        
        print("Starting automated UR10 eye-in-hand calibration. Ensure stationary checkerboard is visible.")
        
        # Move to calibration pose
        for i, pose in enumerate(calibration_poses):
            print(f"Moving to pose {i+1}/{len(calibration_poses)}: {pose}")
            robot.movel(pose, acc=0.5, vel=0.5)
            time.sleep(1)  
            
            # Capture image and depth
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                print("Failed to capture frames. Returning to start.")
                return_to_start()
                continue
            
            image = np.asanyarray(color_frame.get_data())
            
            # Detect checkerboard
            _, T_camera_target = get_checkerboard_pose(image, depth_frame)
            if T_camera_target is None:
                print("Checkerboard not detected. Proceeding to next pose or retry with 'c'.")
                if not proceed_to_next_pose():
                    continue  
                else:
                    i -= 1  
                    continue
            
            # Get end-effector pose
            T_base_ee = pose_to_matrix(pose)
            T_base_ee_inv = np.linalg.inv(T_base_ee)  # Invert for eye-in-hand
            
            # Store transformations (XA = XB)
            A_matrices.append(T_camera_target)
            B_matrices.append(T_base_ee_inv)  # Use inverse for eye-in-hand
            
            cv2.imshow("Calibration", image)
            cv2.waitKey(1)
        
        # Solve hand-eye calibration (XA = XB for eye in hand)
        if len(A_matrices) < 4:
            raise ValueError("Insufficient valid poses for calibration (need at least 4).")
        
        R, t = cv2.calibrateHandEye(
            R_gripper2base=[np.linalg.inv(B[:3, :3]) for B in B_matrices],  # Inverted rotation
            t_gripper2base=[-B[:3, 3] for B in B_matrices],  # Inverted translation
            R_target2cam=[A[:3, :3] for A in A_matrices],
            t_target2cam=[A[:3, 3] for A in A_matrices],
            method=cv2.CALIB_HAND_EYE_TSAI
        )
        
        # Construct camera to end effector transformation
        T_camera_ee = np.eye(4)
        T_camera_ee[:3, :3] = R
        T_camera_ee[:3, 3] = t.ravel()
        
        # Save calibration
        with open("camera_to_ee.pkl", "wb") as f:
            pickle.dump(T_camera_ee, f)
        print("Calibration completed. Saved camera-to-end-effector transformation to 'camera_to_ee.pkl'.")
        
        print("Validating calibration...")
        for i, (A, B) in enumerate(zip(A_matrices, B_matrices)):
            point_cam = np.array([0, 0, 0.5, 1])  # point in camera frame
            point_ee = T_camera_ee @ point_cam
            point_base = B @ point_ee  # inverse transformation
            print(f"Pose {i+1}: Point in base frame: {point_base[:3]}")
        
        robot.movel(calibration_poses[0], acc=0.5, vel=0.5)
    
    except Exception as e:
        print(f"Error: {e}")
        return_to_start()
    
    finally:
        pipeline.stop()
        robot.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()