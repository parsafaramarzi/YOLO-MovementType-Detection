import os
from ultralytics import YOLO
import cv2
import imageio
from ultralytics.utils.plotting import Annotator
import numpy as np

VIDEO_NAME = "peoplecrowd02.mp4"
DRAW_POSE_FLAG = True
DRAW_JOINT_LABELS_FLAG = True
DRAW_BOXES_FLAG = True
CONFIDENCE_THRESHOLD = 0.3
RUNNING_ARM_THRESHOLD = 130
RUNNING_LEG_THRESHOLD = 160

if os.path.exists("dataset/" + VIDEO_NAME):
    dataset = cv2.VideoCapture("dataset/" + VIDEO_NAME)
else:
    raise FileNotFoundError("Video file not found in the dataset folder.")

Yolo_model = YOLO("yolo11x-pose.pt")
writer = imageio.get_writer("output/yolov11_movementtype_detection" + VIDEO_NAME, fps=30, codec='libx264', quality=8)

aspect_ratio = dataset.get(cv2.CAP_PROP_FRAME_WIDTH) / dataset.get(cv2.CAP_PROP_FRAME_HEIGHT)
new_h = 800
new_w = int(aspect_ratio * new_h)

joint_labels = {
    5: "L Shoulder", 6: "R Shoulder", 7: "L Elbow", 8: "R Elbow", 
    9: "L Wrist", 10: "R Wrist", 11: "L Hip", 12: "R Hip", 
    13: "L Knee", 14: "R Knee", 15: "L Ankle", 16: "R Ankle"
}
skeleton = [
    (5,6), (5,7), (7,9), (6,8), (8,10),
    (5,11), (6,12), (11,12), (11,13), (13,15), (12,14), (14,16)
]
person_status_details = []

def calculate_angle(p1, p2, p3):
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def determine_movement_type(kpts_np, conf_np):
    
    l_arm_joints = [5, 7, 9]
    r_arm_joints = [6, 8, 10]
    r_leg_joints = [12, 14, 16]
    l_leg_joints = [11, 13, 15]
    
    angle_l_arm, angle_r_arm, angle_r_leg, angle_l_leg = None, None, None, None
    l_arm_running, r_arm_running, r_leg_running, l_leg_running = False, False, False, False
    l_arm_visible, r_arm_visible, r_leg_visible, l_leg_visible = False, False, False, False

    l_arm_visible = all(i < len(conf_np) and conf_np[i] > CONFIDENCE_THRESHOLD for i in l_arm_joints)
    if l_arm_visible:
        angle_l_arm = calculate_angle(kpts_np[5], kpts_np[7], kpts_np[9])
        if angle_l_arm < RUNNING_ARM_THRESHOLD:
            l_arm_running = True
    
    r_arm_visible = all(i < len(conf_np) and conf_np[i] > CONFIDENCE_THRESHOLD for i in r_arm_joints)
    if r_arm_visible:
        angle_r_arm = calculate_angle(kpts_np[6], kpts_np[8], kpts_np[10])
        if angle_r_arm < RUNNING_ARM_THRESHOLD:
            r_arm_running = True

    r_leg_visible = all(i < len(conf_np) and conf_np[i] > CONFIDENCE_THRESHOLD for i in r_leg_joints)
    if r_leg_visible:
        angle_r_leg = calculate_angle(kpts_np[12], kpts_np[14], kpts_np[16])
        if angle_r_leg < RUNNING_LEG_THRESHOLD:
            r_leg_running = True
            
    l_leg_visible = all(i < len(conf_np) and conf_np[i] > CONFIDENCE_THRESHOLD for i in l_leg_joints)
    if l_leg_visible:
        angle_l_leg = calculate_angle(kpts_np[11], kpts_np[13], kpts_np[15])
        if angle_l_leg < RUNNING_LEG_THRESHOLD:
            l_leg_running = True

    def get_angle_str(angle, joint_name):
        return f"{joint_name}:{int(angle)}" if angle is not None else ""

    any_arm_running = l_arm_running or r_arm_running
    any_leg_running = l_leg_running or r_leg_running
    
    arm_angles = []
    if angle_l_arm is not None: arm_angles.append(get_angle_str(angle_l_arm, "L_Elbow"))
    if angle_r_arm is not None: arm_angles.append(get_angle_str(angle_r_arm, "R_Elbow"))
    arm_angle_str = ", ".join(arm_angles)

    leg_angles = []
    if angle_r_leg is not None: leg_angles.append(get_angle_str(angle_r_leg, "R_Knee"))
    if angle_l_leg is not None: leg_angles.append(get_angle_str(angle_l_leg, "L_Knee"))
    leg_angle_str = ", ".join(leg_angles)
    
    if any_arm_running and any_leg_running:
        reason = f"Running ({arm_angle_str}, {leg_angle_str})"
        return "Running", reason

    if r_leg_visible and l_leg_visible:
        reason = f"Non-Run ({leg_angle_str})"
        return "Non-Running", reason

    can_check_running = (l_arm_visible or r_arm_visible) and (r_leg_visible or l_leg_visible)
    can_check_non_running = (r_leg_visible and l_leg_visible)
    
    if not can_check_running and not can_check_non_running:
        reasons = []
        if not (l_arm_visible or r_arm_visible): reasons.append("Arm_Occluded")
        if not (r_leg_visible or l_leg_visible): reasons.append("Legs_Occluded")
        reason = "Unknown (" + ", ".join(reasons) + ")"
        return "Unknown", reason
        
    return "Unknown", "Unknown (Partial Visibility)"

while True:
    results, frame = dataset.read()
    if not results:
        break
    
    frame = cv2.resize(frame, (new_w, new_h))
    detections = Yolo_model(frame, classes=[0])
    annotator = Annotator(frame, line_width=2)
    person_status_details.clear()
    
    keypoints_all = detections[0].keypoints.xy
    keypoints_conf_all = detections[0].keypoints.conf
    boxes_all = detections[0].boxes.xyxy 

    for kpts, conf, box_tensor in zip(keypoints_all, keypoints_conf_all, boxes_all): 
        kpts_np = kpts.cpu().numpy()
        conf_np = conf.cpu().numpy()
        
        movement_type, reason_string = determine_movement_type(kpts_np, conf_np)
        person_status_details.append(movement_type)

        if DRAW_POSE_FLAG:
            for i, (x, y) in enumerate(kpts_np):
                if conf_np[i] > CONFIDENCE_THRESHOLD:
                    if i in joint_labels:
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                        if DRAW_JOINT_LABELS_FLAG:
                            label = joint_labels[i]
                            text_pos = (int(x) + 7, int(y) - 5)
                            cv2.putText(frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 2, cv2.LINE_AA)
                            cv2.putText(frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1, cv2.LINE_AA)
            
            for (start, end) in skeleton:
                if conf_np[start] > CONFIDENCE_THRESHOLD and conf_np[end] > CONFIDENCE_THRESHOLD:
                    pt1 = (int(kpts_np[start][0]), int(kpts_np[start][1]))
                    pt2 = (int(kpts_np[end][0]), int(kpts_np[end][1]))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        
        if DRAW_BOXES_FLAG:
            x1, y1, x2, y2 = map(int, box_tensor.cpu().numpy())
            
            label_text = reason_string 

            if movement_type == "Running":
                label_color = (255, 100, 0)
            elif movement_type == "Non-Running":
                label_color = (0, 100, 255)
            else:
                label_color = (128, 128, 128)
                
            annotator.box_label((x1, y1, x2, y2), label_text, color=label_color)

    writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cv2.imshow("Input", frame)
    if cv2.waitKey(1) == 13:
        dataset.release()
        writer.close()
        cv2.destroyAllWindows()
        break

dataset.release()
writer.close()
cv2.destroyAllWindows()