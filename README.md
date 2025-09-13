# Helmet Detection using YOLOv8

This project detects whether a person is wearing a helmet or not using YOLOv8 and OpenCV.  
It can be used for traffic surveillance, rider safety monitoring, and smart city applications.

## 🚀 Features Dataset
- Detects helmets in images and videos
- YOLOv8 trained on custom dataset
- Real-time detection using webcam or video file
- Outputs annotated video with bounding boxes

## 🛠 Tech Stack
- Python 3
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy

## ⚙️ Installation
Clone the repository and install dependencies:

```bash
git clone ## ⚙️ Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/helmet-detection.git
cd helmet-detection
pip install -r requirements.txt


## 📂 Dataset 
The dataset used for training contains images of riders with and without helmets.  
Label files are stored in `train/labels/`.  

You can prepare your dataset using **LabelImg** or **Roboflow**.

## ▶️ Usage

### Run detection on an image:
```bash
python main.py --source input_video.mp4


## 📊 Results
Example detection:

(Input image → Output with bounding boxes and labels)

<img width="899" height="419" alt="Screenshot 2025-09-13 202750" src="https://github.com/user-attachments/assets/9455f5a9-00ff-4d69-a99c-68e077b30139" />
<img width="911" height="503" alt="Screenshot 2025-09-13 192843" src="https://github.com/user-attachments/assets/d413b8b8-1139-40b7-8078-b8084c2b8773" />

