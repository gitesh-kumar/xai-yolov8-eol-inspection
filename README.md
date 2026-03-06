# Explainable AI for End-of-Line Inspection

This project explores the use of YOLOv8 for industrial defect detection,
combined with Explainable AI techniques to improve trust and transparency
in safety-critical production environments.

## 📊 XAI Visualization Results

### 1. LRP Explanation (YOLOv8)
![LRP Result](methods_yolo/results/LRP4.png)
*Figure 1: LRP heatmap showing feature importance for screw defect detection. Red areas indicate high influence on the model's decision.*

---

### 2. Grad-CAM Visualization (Yolov5)
![Grad-CAM Result](methods_yolo/results/Grad_cam%202.png)
*Figure 2: Grad-CAM output highlighting the convolutional layer activations during classification.*

---

### 3. LIME Visualization (Yolov8)
![LIME Result](methods_yolo/results/Lime_exp1.png)
*Figure 3: LIME output highlighting the highlighting the areas important for classification.*
