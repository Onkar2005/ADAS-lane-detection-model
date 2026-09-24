# 🚗 Real-Time Lane Detection & Lane Keeping System

A computer vision-based **lane detection and lane-keeping monitoring system** developed using **Python and OpenCV**.

The system processes a video stream, detects road lane boundaries, calculates the vehicle's position relative to the lane center, and displays real-time driving information such as **lane deviation, FPS, brightness, and lane detection status**.

## 📌 Features

* 🎥 Real-time video processing
* 🛣️ Road lane detection
* 🔍 Canny Edge Detection
* 🎯 Region of Interest (ROI) filtering
* 📐 Hough Line Transform for lane detection
* 📊 Lane-line averaging and smoothing
* 📏 Lane deviation calculation
* 🚘 Lane-keeping status monitoring
* 💡 Frame brightness detection
* ⚡ Real-time FPS calculation
* 🖥️ On-screen information panel
* 🔄 Supports video input and webcam-style video sources

## 🧠 How It Works

The system follows these main steps:

```text
Video Input
     ↓
Frame Resizing
     ↓
Grayscale Conversion
     ↓
Gaussian Blur
     ↓
Canny Edge Detection
     ↓
Region of Interest (ROI)
     ↓
Hough Line Transform
     ↓
Left & Right Lane Detection
     ↓
Lane Line Smoothing
     ↓
Lane Center Calculation
     ↓
Deviation Calculation
     ↓
Lane Status Display
```

## 🔧 Technologies Used

* **Python**
* **OpenCV**
* **NumPy**
* **Deque / Collections**
* Computer Vision
* Image Processing

## 📦 Requirements

Install Python 3.x and the required libraries:

```bash
pip install opencv-python numpy
```

## 📁 Project Structure

A recommended GitHub repository structure is:

```text
Lane-Detection/
│
├── lane_detection.py
├── video1.mp4
├── output/
│   ├── output1.png
│   ├── output2.png
│   └── output3.png
│
├── README.md
└── requirements.txt
```

## ▶️ How to Run

### 1. Clone the repository

```bash
[git clone https://github.com/Onkar2005/ADAS-lane-detection-model.git
```

### 2. Open the project directory

```bash
cd ADAS-lane-detection-model
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or install them directly:

```bash
pip install opencv-python numpy
```

### 4. Add your input video

Place your road/lane video inside the project directory.

The current program is configured to use:

```python
WEBCAM_INDEX = "video1.mp4"
```

You can change this value in `ADAS-road-lane-detection-model.py` to another video file.

### 5. Run the program

```bash
python ADAS-road-lane-detection-model.py
```

A window will open showing the processed video.

Press:

```text
q
```

to stop the program.

## 📊 Output Information

The application displays several useful pieces of information on the processed video.

### Lane Detection

The detected left and right lane boundaries are displayed on the road.

The detected lane area is highlighted using a transparent overlay.

### Lane Deviation

The system calculates the position of the lane center relative to the center of the video frame.

The deviation is displayed as a percentage.

```text
Deviation: +5%
```

A positive or negative value indicates the direction of the vehicle's offset according to the system's coordinate calculation.

### Lane Keeping Status

The system provides status messages such as:

```text
Good Lane Keeping
```

```text
Drifting Right!
```

```text
Drifting Left!
```

```text
Searching Lanes...
```

The thresholds and corresponding messages are implemented in the lane status display logic.

### FPS

The current processing speed is displayed as:

```text
FPS: 30.25
```

### Brightness

The average frame brightness is calculated and displayed to provide basic visibility information.

## ⚙️ Configuration

Several parameters can be modified at the beginning of `lane_detection.py`:

```python
FRAME_WIDTH = 960
FRAME_HEIGHT = 540

GAUSSIAN_KERNEL = (5, 5)

CANNY_LOW = 50
CANNY_HIGH = 150

HOUGH_RHO = 2
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 50

MIN_LINE_LEN = 40
MAX_LINE_GAP = 150

SMOOTHING_FRAMES = 8
```

These parameters control frame size, edge detection, Hough line detection, and temporal lane smoothing.

## 🔬 Lane Detection Method

### 1. Grayscale Conversion

The input frame is converted from BGR to grayscale.

### 2. Gaussian Blur

Gaussian filtering is applied to reduce image noise.

### 3. Canny Edge Detection

The Canny algorithm identifies important edges in the road image.

### 4. Region of Interest

Only the road area is selected using a trapezoidal ROI:

```text
       ┌──────────┐
      /            \
     /              \
    /________________\
```

This helps reduce unwanted edges outside the expected road region.

### 5. Hough Line Transform

The system uses `cv2.HoughLinesP()` to detect line segments from the processed road image.

### 6. Left and Right Lane Classification

Detected lines are classified based on their slope.

```text
Negative slope → Left lane
Positive slope → Right lane
```

Near-horizontal lines are filtered out.

### 7. Temporal Smoothing

Lane coordinates are averaged over multiple frames to make the detected lane boundaries more stable.

The project uses a smoothing window of 8 frames by default.

## 📷 Sample Outputs

Add your generated output images to the repository, for example:

```text
 output.png
```

Then display them in this README:

### Output 1

![Lane Detection Output 1](output1.png)


## 📈 Future Improvements

Possible improvements include:

* Deep-learning-based lane detection
* Curved lane detection
* Better handling of poor lighting
* Night-time lane detection
* Weather-resistant lane detection
* Vehicle speed estimation
* Lane departure warning
* Steering angle estimation
* Automatic steering control
* Traffic sign detection
* Object and vehicle detection
* Integration with autonomous driving systems

## ⚠️ Limitations

This project is based on traditional computer vision techniques such as edge detection and Hough lines. Therefore, performance can be affected by:

* Poor lighting
* Rain or fog
* Faded lane markings
* Sharp curves
* Occluded lane markings
* Complex road environments
* Heavy traffic

This project is intended as a **computer vision/educational prototype** and should not be used as a safety-critical autonomous driving system.

## 👨‍💻 Author

**Your Name**

GitHub: `https://github.com/Onkar2005/ADAS-lane-detection-model`

## 📄 License

This project is available for educational and research purposes.

You may add a license such as **MIT License** if you want to allow others to reuse and modify the project.
