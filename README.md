# Smart Egg Inspection & Sorting System (EggVision)

### 💡 Project Idea: "Smart Egg Inspection & Sorting System"

An automated vision system designed for industrial poultry and food processing:
1. **Real-time Egg Counting**: Accurately detect and count eggs moving along a conveyor belt or tray.
2. **Defect Detection**:
   - **Good**: Clean and structurally intact eggs.
   - **Cracked**: Shell cracks or fractures (high-priority defect).
   - **Dirty / Stained**: Dirt, mud, or organic stains on the eggshell.
3. **Size Grading**:
   - OpenCV Contour Area analysis: Classify into Small, Medium, and Large.
4. **Live Dashboard**: Real-time video feed, live counter metrics, defect rate analytics, and alerts.

---

### 🏗️ Implementation Diagram

```mermaid
flowchart TD
    subgraph Input["1. Input Source"]
        CAM["Live Webcam / Conveyor Video"]
    end

    subgraph VisionPipeline["2. OpenCV & ML Engine"]
        PRE["OpenCV Preprocessing<br/>(Resize, CLAHE, Noise Removal)"]
        YOLO["YOLOv8 / YOLO11 Inference<br/>(Detect: Good, Cracked, Dirty)"]
        SIZE["Contour & Area Analysis<br/>(Size Grading: S / M / L)"]
    end

    subgraph Backend["3. FastAPI Backend"]
        WS["WebSocket Stream Manager<br/>(Real-time Frame Streaming)"]
        LOG["Batch & Analytics Logger<br/>(Counts, Timestamp, Defects)"]
    end

    subgraph Frontend["4. React Dashboard"]
        UI_FEED["Live Video with Bounding Boxes"]
        UI_STATS["KPI Cards: Total, Good %, Defect %"]
        UI_CHART["Size Distribution (S/M/L)"]
    end

    CAM --> PRE
    PRE --> YOLO
    PRE --> SIZE
    YOLO --> WS
    SIZE --> WS
    WS --> UI_FEED
    WS --> UI_STATS
    WS --> UI_CHART
    YOLO --> LOG
```
