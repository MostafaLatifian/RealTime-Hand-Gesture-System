# 🖐️ Real-Time Hand Gesture Control System (HMI)

An advanced Human-Machine Interface (HMI) project utilizing real-time computer vision for **gesture-based control**. The system detects five specific hand gestures from a webcam feed and translates them into control commands, serving as an effective non-contact controller.

The core technology relies on **Google MediaPipe** for robust hand tracking and a custom-trained **Random Forest** classifier for accurate gesture prediction.

## 🌟 Project Goal & Functionality

The main objective was to design a real-time system that maps five distinct hand poses to functional commands, tested both as a media controller and an embedded system output (Raspberry Pi GPIO).

| Detected Gesture | Primary Function (Example: Media Control) | Raspberry Pi Function (Physical Output) |
| :--- | :--- | :--- |
| **Move Right** (e.g., Thumb Right) | **Video Forward** (Seek Ahead) | **Activates LED 1** |
| **Move Left** (e.g., Thumb Left) | **Video Backward** (Seek Back) | **Activates LED 2** |
| **Thumb Up** | **Volume Up** (Increase) | **Activates LED 3** |
| **Thumb Down** | **Volume Down** (Decrease) | **Activates LED 4** |
| **Stop Sign** (Open Palm) | **Video Stop/Pause** | **Activates LED 5** |

***

## ⚙️ Core Technologies & Stack

| Component | Technology / Method | Purpose |
| :--- | :--- | :--- |
| **Tracking** | **Google MediaPipe** | Extracts 21 3D landmark coordinates from the hand for feature generation. |
| **Classification** | **Random Forest** Classifier | Trained Machine Learning model for high-speed, accurate classification of feature vectors. |
| **Platform** | **Python 3** | Core development language. |
| **Deployment** | **Raspberry Pi 3+** | Used for practical, embedded implementation and direct hardware control (GPIO). |
| **Tools** | PyCharm, `joblib` | IDE and model serialization library. |

***

## 🛠️ Implementation Phases

The project followed a complete machine learning pipeline:

### Phase 1: Custom Dataset Generation
* **Feature Engineering:** Raw images from a public dataset were processed using **MediaPipe** to extract normalized, numerical landmark feature vectors.
* **Outcome:** Creation of a tailored, high-quality feature dataset (`feature_dataset.csv`) for optimized training.

### Phase 2: Model Training
* **Algorithm:** The **Random Forest** classifier was trained on the custom dataset.
* **Optimization:** Selected for its reliable accuracy and fast inference speed, crucial for the real-time requirements of the Raspberry Pi.
* **Output:** The trained model was serialized into `trained_model.joblib`.

### Phase 3 & 4: Real-Time Deployment
* The system integrates webcam input with the loaded `joblib` model.
* It performs a **real-time loop** of image acquisition, MediaPipe feature extraction, and gesture prediction.
* **Pi Demonstration:** For practical testing, the predicted commands were used to toggle corresponding **LEDs** connected to the Raspberry Pi's **GPIO pins**, simulating physical control.

***

## 📂 Repository Structure
