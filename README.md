

# 📊 **Smart Retail Analytics System using YOLO & Computer Vision**

---

## 🧠 **Project Overview**

This project presents a **Computer Vision-based Retail Analytics System** designed to analyze in-store customer behavior using surveillance video footage. The system leverages **YOLO-based object detection**, **ByteTrack multi-object tracking**, and **custom spatial analytics logic** to extract meaningful insights such as:

* 👥 Customer Footfall (Unique Count)
* 🚪 Entry & Exit Detection
* ⏱️ Dwell Time Analysis
* 🔥 Heatmap Generation (High-traffic zones)

The project emphasizes **real-world robustness**, focusing on handling noise, occlusions, and tracking inconsistencies rather than just model accuracy.

---

## 🎯 **Objectives**

* Detect and track customers in real-time video streams
* Maintain **consistent identity tracking** across frames
* Analyze **directional movement (Entry/Exit)**
* Generate **heatmaps for spatial behavior analysis**
* Compute **per-person analytics (time spent, frame presence)**
* Build a system deployable on **resource-constrained retail hardware**

---

## 🏗️ **System Architecture**

```text
Video Input → YOLO Detection → ByteTrack Tracking → Filtering Layer → 
Centroid Smoothing → Entry/Exit Logic → Heatmap Generation → Reporting
```

---

## ⚙️ **Technology Stack**

| Component       | Technology                   |
| --------------- | ---------------------------- |
| Programming     | Python                       |
| Computer Vision | OpenCV                       |
| Detection Model | YOLO (Ultralytics framework) |
| Tracking        | ByteTrack                    |
| Data Handling   | NumPy, Pandas                |
| Visualization   | OpenCV + Heatmap             |
| Dataset Tool    | Roboflow                     |

---

## 🔄 **Methodology**

The system was developed through an **iterative engineering process**, focusing on refining detection, tracking, and analytics:

### 1. Detection Refinement

* Initially used generic object detection
* Filtered to **‘Person’ class (Class 0)** to remove irrelevant detections
* Applied **confidence thresholding** and **Non-Maximum Suppression (NMS)**

---

### 2. Tracking Implementation

* Integrated **ByteTrack** for persistent ID assignment
* Ensured **temporal consistency** across frames
* Addressed identity loss during occlusions

---

### 3. ID Stability Enhancements

* Implemented **Centroid-based tracking refinement**
* Added **temporal buffering (MAX_DISAPPEAR)**
* Applied **distance-based ID reassignment**

---

### 4. Spatial Analytics

* Designed **Entry/Exit detection using line-crossing logic**
* Introduced **Hysteresis (buffer zone)** to avoid flickering errors
* Adapted logic to **horizontal movement (left ↔ right)**

---

### 5. Heatmap Generation

* Accumulated centroid positions across frames
* Generated visual representation of **high-density zones**

---

### 6. Custom Model Training

* Dataset created and annotated using **Roboflow**
* Trained YOLO model on labeled retail data
* Evaluated using:

  * Precision
  * Recall
  * mAP

---

## 🚧 **Challenges & Solutions**

### ❌ Challenge: ID Fragmentation

* Same person assigned multiple IDs

✅ Solution:

* Centroid matching
* Temporal buffering
* Confidence tuning

---

### ❌ Challenge: Duplicate Detections

* Multiple boxes for same person

✅ Solution:

* IoU-based filtering
* NMS optimization

---

### ❌ Challenge: False Entry/Exit Detection

* Boundary flickering

✅ Solution:

* Hysteresis (dual-line buffer system)

---

### ❌ Challenge: Low Confidence Predictions (Custom Model)

* Weak detection scores (~0.3)

✅ Solution:

* Adjusted confidence threshold (0.25–0.35)
* Balanced precision vs recall

---

## 📊 **Output & Results**

The system generates:

```text
output/
├── output_video.mp4      (Annotated video with tracking)
├── heatmap.png           (Spatial movement visualization)
├── reports.csv           (Summary metrics)
├── person_details.csv    (Per-ID analytics)
```

---

### 📈 Sample Metrics

* Total Unique Customers
* Total Entries
* Total Exits
* Time Spent per Customer
* Frame Presence Count

---

## 🧪 **Evaluation Metrics**

* **Precision (P)** – Correct detections
* **Recall (R)** – Detection coverage
* **mAP@50** – Detection accuracy
* **Confusion Matrix** – TP / FP / FN analysis

---

## ▶️ **How to Run the Project**

---

### 🔹 Step 1: Setup Environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

### 🔹 Step 2: Install Dependencies

```bash
pip install ultralytics opencv-python numpy pandas matplotlib scikit-learn streamlit
```

---

### 🔹 Step 3: Run Main Application

```bash
python app.py
```

---

### 🔹 Step 4: (Optional) Run Confusion Matrix Dashboard

```bash
streamlit run confsss.py
```

---

## 📁 **Project Structure**

```text
yolo detection/
├── app.py
├── confsss.py
├── dataset/
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
├── runs/
├── output/
└── README.md
```

---

## 🧠 **Key Learnings**

* Importance of **temporal consistency in tracking systems**
* Trade-offs between **accuracy and real-time performance**
* Handling **real-world noise (occlusion, lighting, crowd density)**
* Role of **data quality over model complexity**
* Practical implementation of **spatial analytics in CV systems**

---

## 🚀 **Future Scope**

* Multi-camera tracking system
* Dashboard visualization (Streamlit/Power BI)
* Advanced Re-Identification (ReID)
* Customer behavior prediction
* Integration with retail decision systems

---

## 📌 **Conclusion**

This project demonstrates a **complete end-to-end Computer Vision pipeline**, integrating detection, tracking, and analytics. The focus was not just on achieving accuracy but on building a **robust, real-world deployable system** capable of handling inconsistencies in dynamic retail environments.



## 👨‍💻 **Authors**

* Khushal Agrawal
* Palak


## 📎 **References**

* Ultralytics YOLO Documentation
* Roboflow Dataset Platform
* ByteTrack Research Paper


