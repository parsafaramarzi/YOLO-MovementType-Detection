# 🏃 Real-Time Movement Classification (Running/Non-Running)

This project utilizes the **YOLOv11-Pose** model to perform real-time human pose estimation and classify individual movement as **"Running"** or **"Non-Running"** based on angular analysis of key joints (Elbows and Knees).

The bounding box label provides the classification and the specific angle measurements that led to the decision, aiding in monitoring and debugging.

## 🚀 Demonstration

A brief visual demonstration of the system in action:

![Movement Detection Demonstration](output/Yolov11%20Movementtype%20Detectionpeoplecrowd02_compressed.gif)

## ⚙️ Core Logic

The classification logic is based on checking for active flexion (bent limbs) or passivity (straight limbs) across the arms and legs, with specific criteria for each state:

| Status | Condition | Joints Checked | Thresholds |
| :--- | :--- | :--- | :--- |
| **Running** | Requires active movement in **ANY** Arm **AND** **ANY** Leg simultaneously. | L/R Elbow, L/R Knee | Elbows $< 130^\circ$ |
| | | | Knees $< 160^\circ$ |
| **Non-Running** | Requires **BOTH** legs to be straight and visible. Arm status is ignored to allow for raised/waving hands. | L/R Knee | Knees $\ge 160^\circ$ |
| **Unknown** | Insufficient visibility (confidence $< 0.3$) of the required joints to satisfy either the "Running" or "Non-Running" criteria. | L/R Arm, L/R Leg | N/A |

## 🔑 Key Files and Setup

### Prerequisites

* Python 3.x
* OpenCV (`cv2`)
* `ultralytics` (for YOLO model)
* `numpy`
* `imageio`

You can install the necessary libraries using pip:
```bash
pip install ultralytics opencv-python numpy imageio

```

### Model Weight

The project uses the `yolo11x-pose.pt` model, which must be present in your project directory or downloaded automatically by the `ultralytics` library.

### Project Structure

```
.
├── movement_detector.py  # (Your main code)
├── yolo11x-pose.pt       # YOLO Pose Model Weights
├── dataset/
│   └── peoplecrowd02.mp4 # Input Video File
└── output/
    └── ...               # Output GIF/Video generated here

```

## 📝 Configuration Variables

Adjust the behavior of the detection script by modifying the global variables near the top of the file:

| Variable | Default Value | Description |
| --- | --- | --- |
| `VIDEO_NAME` | `"peoplecrowd02.mp4"` | Name of the video file in the `dataset/` folder. |
| `DRAW_POSE_FLAG` | `True` | Toggle drawing of joints and skeleton. |
| `DRAW_JOINT_LABELS_FLAG` | `True` | Toggle drawing text labels next to joints. |
| `CONFIDENCE_THRESHOLD` | `0.3` | Minimum confidence required to display a joint/skeleton segment and use it for calculation. |
| `RUNNING_ARM_THRESHOLD` | `130` | Max **Elbow Angle** () considered "running" (bent arm). |
| `RUNNING_LEG_THRESHOLD` | `160` | Max **Knee Angle** () considered "running" (bent leg). |

## 🏃 Running the Script

1. Ensure your video file is placed in the `dataset/` folder.
2. Ensure the `Yolo_model` weights file (`yolo11x-pose.pt`) is accessible.
3. Run the Python script:

```bash
python movement_detector.py

```

The script will open a window showing the live analysis and save the resulting video to the `output/` folder. Press **Enter** (key code 13) in the live window to stop processing.

## 🧑‍💻 Code Details (Function)

The classification is encapsulated in the `determine_movement_type` function, which returns the state and a detailed reason string:

```python
movement_type, reason_string = determine_movement_type(kpts_np, conf_np)

```

**Example Output (Box Label):**

* **Running:** `Running (L_Elbow:88, R_Knee:145)`
* **Non-Running:** `Non-Run (R_Knee:172, L_Knee:168)`
* **Unknown:** `Unknown (Arm_Occluded)`
