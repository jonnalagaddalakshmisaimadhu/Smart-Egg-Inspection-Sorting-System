# 🥚 EggVision: Smart Egg Inspection & Sorting System

> **Production-Grade Computer Vision & AI Platform for Automated Real-Time Egg Quality Inspection, Defect Detection, Size Grading, and Sorting.**

---

## 📌 Phase 0: Problem Statement & Targets

### 1. Problem Statement
Automated computer vision system designed for poultry and egg processing conveyor lines to detect shell defects (`Cracked`, `Dirty`), grade sizes (`Small`, `Medium`, `Large`), accurately count eggs in real-time, and trigger automated sorting/rejection signals.

---

### 2. Classes & Priority Hierarchy
When an egg exhibits multiple attributes, the classification follows a strict priority:
$$\textbf{Cracked} > \textbf{Dirty} > \textbf{Good}$$

- **1. Cracked (Highest Priority):** Hairline cracks, fractures, or punctured shells. Immediate reject priority.
- **2. Dirty:** Organic stains, mud, or fecal residue on the shell (without structural cracks).
- **3. Good:** Structurally sound, clean, defect-free egg.

---

### 3. Size Classification (Pixels to Grams Calibration)
- **Small (S):** $< 53\text{g}$ (calibrated contour area threshold)
- **Medium (M):** $53\text{g} - 63\text{g}$
- **Large (L):** $> 63\text{g}$

---

### 4. Measurable Success Targets (KPIs)

| Metric | Target | Description & Justification |
| :--- | :---: | :--- |
| **Cracked Recall** | **$\ge$ 95%** | Zero tolerance for cracked eggs entering consumer packages. |
| **Count Accuracy** | **$\ge$ 98%** | Precise batch inventory tracking without duplicate counts. |
| **Size Grading Match** | **$\ge$ 90%** | Accurate physical weight matching against digital scale benchmarks. |
| **Processing Speed** | **$\ge$ 20 FPS** | Real-time, zero-lag streaming for conveyor inspection speeds. |

---

## 🏗️ End-to-End Implementation Architecture

```mermaid
flowchart TD
    %% -------------------------------------------------------------
    %% LAYER 1: DATA & MLOps PIPELINE (OFFLINE)
    %% -------------------------------------------------------------
    subgraph DataMLOps["A. Data Engineering & MLOps (Offline Training)"]
        direction TB
        RAW["Raw Video Sessions<br/>(Diff. lighting, mix, speeds)"]
        EXTRACT["Frame Extraction<br/>(preprocess.py)"]
        ANNOT["CVAT + SAM Annotation<br/>(Polygon Segmentation)"]
        SPLIT["Session-wise Split & Augment<br/>(Train / Val / Test)"]
        DVC["DVC Version Control<br/>(data/datasets/v1)"]
        TRAIN["YOLO-seg Training<br/>(Ultralytics)"]
        MLFLOW["MLflow Experiment Tracking<br/>(Metrics: Cracked Recall >= 95%, mAP)"]
        EXPORT["Model Optimization & Export<br/>(PyTorch -> ONNX / TensorRT)"]

        RAW --> EXTRACT --> ANNOT --> SPLIT --> DVC --> TRAIN
        TRAIN <--> MLFLOW
        TRAIN --> EXPORT
    end

    %% -------------------------------------------------------------
    %% LAYER 2: PHYSICAL & VIDEO INGESTION
    %% -------------------------------------------------------------
    subgraph Ingestion["B. Physical Setup & Video Ingestion"]
        CAM["Top-Down Fixed Camera / Webcam<br/>(Diffuse Light, Matte Background)"]
        CONV["Conveyor / Egg Movement<br/>(White & Brown Eggs + Coin Reference)"]
        STREAM["OpenCV Video Capture Stream<br/>(Frame Buffer / Resize / CLAHE)"]

        CONV -.-> CAM --> STREAM
    end

    %% -------------------------------------------------------------
    %% LAYER 3: REAL-TIME VISION INFERENCE ENGINE
    %% -------------------------------------------------------------
    subgraph VisionEngine["C. Core Vision & Inference Engine (Python)"]
        direction TB
        DET["YOLO-seg ONNX Inference<br/>(Detects: Good, Cracked, Dirty)"]
        TRACK["ByteTrack / Supervision<br/>(Assigns Unique Egg ID across frames)"]
        LINE["Virtual Line Crossing Check<br/>(Triggers Count at Boundary)"]
        CALIB["Size & Area Calibrator<br/>(Coin Ref: Px to mm -> S, M, L)"]
        LOGIC["Priority Resolver<br/>(Cracked > Dirty > Good)"]

        STREAM --> DET
        DET --> TRACK
        TRACK --> LINE
        DET --> CALIB
        TRACK & CALIB & LINE --> LOGIC
    end

    EXPORT -.->|Load Best Model| DET

    %% -------------------------------------------------------------
    %% LAYER 4: APPLICATION BACKEND (FASTAPI)
    %% -------------------------------------------------------------
    subgraph Backend["D. FastAPI Backend Service"]
        direction TB
        MJPEG["MJPEG Video Streamer<br/>(/stream: Visual Masks + BBoxes)"]
        WS["WebSocket Server<br/>(/ws: Live FPS, Counts, Alerts)"]
        DB[(PostgreSQL Database<br/>Batches, Eggs, Defects, Alerts)]
        SORT["Reject Signal Controller<br/>(GPIO / Serial / Event Logger)"]

        LOGIC --> MJPEG
        LOGIC --> WS
        LOGIC --> DB
        LOGIC -->|If Cracked/Dirty| SORT
    end

    %% -------------------------------------------------------------
    %% LAYER 5: USER INTERFACE (REACT DASHBOARD)
    %% -------------------------------------------------------------
    subgraph Frontend["E. Production React Dashboard (Web UI)"]
        direction TB
        UI_FEED["Live Stream Canvas<br/>(Segmented Eggs + Tracking IDs)"]
        UI_KPI["KPI Cards<br/>(Total Count, Good %, Defect %, FPS)"]
        UI_CHART["Size Distribution Chart<br/>(Small / Medium / Large Counts)"]
        UI_ALERT["Cracked Alert Banner<br/>(Instant Visual Warning)"]
        UI_HIST["Batch History & Export<br/>(Download Inspection Reports)"]

        MJPEG --> UI_FEED
        WS --> UI_KPI
        WS --> UI_CHART
        WS --> UI_ALERT
        DB <--> UI_HIST
    end

    %% -------------------------------------------------------------
    %% LAYER 6: PACKAGING
    %% -------------------------------------------------------------
    subgraph Deploy["F. Production Packaging"]
        DOCKER["Docker Compose<br/>(FastAPI + React + Postgres + MLflow)"]
    end

    Backend -.-> DOCKER
    Frontend -.-> DOCKER
```

---

## 🗺️ Project Phased Roadmap

### A. Foundation
- **Phase 0:** Problem Definition & Target KPIs (Cracked Recall $\ge$ 95%, Count $\ge$ 98%).
- **Phase 1:** Physical Setup (Fixed top-down camera, diffuse lighting, reference coin calibration).
- **Phase 2:** Dev Environment Setup (Python venv, CUDA PyTorch, OpenCV, Ultralytics, DVC, Docker, MLflow).

### B. Data Engineering
- **Phase 3:** Video Data Collection (Multiple sessions: lighting, egg mix, speed).
- **Phase 4:** Annotation (CVAT + SAM polygon segmentation, 10% quality audit).
- **Phase 5:** Dataset Preprocessing & Session-wise Split (DVC versioning, no leakage).

### C. Model Development & MLOps
- **Phase 6:** Baseline Training (YOLO-seg nano, MLflow logging).
- **Phase 7:** Model Evaluation & Error Analysis (Confusion matrix, failure inspection).
- **Phase 8:** Iterative Improvement Loop (Augmentation, hyperparameter tuning until targets are hit).

### D. Vision Logic
- **Phase 9:** Tracking & Virtual Line Counting (ByteTrack + Supervision, unique ID tracking).
- **Phase 10:** Pixel-to-mm Size Calibration (Scale calibration, contour area mapping to S/M/L).

### E. Application Stack
- **Phase 11:** FastAPI Backend (MJPEG video stream, WebSockets, PostgreSQL logging).
- **Phase 12:** React Dashboard (Real-time masks, KPI summary cards, size charts, alerts).
- **Phase 13:** Reject Signal Controller (Hardware/Simulated reject signal for defects).

### F. Finish & Productionization
- **Phase 14:** Packaging & ONNX Export (Docker Compose, latency & FPS profiling).
- **Phase 15:** Final End-to-End Stress Testing (New eggs, new lighting, target verification).
- **Phase 16:** Portfolio Documentation (Complete writeup, demo videos, learnings).

---

## ⚠️ Core Engineering Rules
1. **Test Set Hygiene:** The test split is evaluated strictly in Phases 7 and 15 only.
2. **Experiment Tracking:** Every training and evaluation run is tracked via MLflow.
3. **Data Versioning:** Any dataset change triggers a new DVC version tag.
4. **Frozen Classes:** Class definitions and priority levels are immutable across the project lifecycle.
5. **Quality Gate:** Advance to the next phase only when the current phase's "Done when" conditions are 100% satisfied.
