<img width="898" height="506" alt="image" src="https://github.com/user-attachments/assets/9b9dc4c0-ec1c-47c0-aa9f-569230733d71" /># Helmet Detection using YOLOv8

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
INPUT:
<img width="899" height="419" alt="Screenshot 2025-09-13 202750" src="https://github.com/user-attachments/assets/392a2e37-16e8-4d35-a09d-38baf48847f8" />
OUTPUT:
<img width="1028" height="512" alt="Screenshot 2025-09-13 192732" src="https://github.com/user-attachments/assets/6e64428c-4a76-4410-9d49-b6e9def94ec5" />

<img width="911" height="503" alt="Screenshot 2025-09-13 192843" src="https://github.com/user-attachments/assets/d8754221-b51f-414c-b62d-04b03393a826" />


