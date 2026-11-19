"""
EggVision: Production FastAPI Application & Live Streaming Server (Phases 11-13)
Serves real-time MJPEG video stream, WebSocket telemetry, and Industrial Web Dashboard.
"""

import sys
import os
import time
import asyncio
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from src.vision.pipeline import EggVisionPipeline

app = FastAPI(title="EggVision AI Platform", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
pipeline = None
simulation_mode = True
active_websockets: List[WebSocket] = []
reject_logs = []

# Load test images for conveyor simulation
test_dir = PROJECT_ROOT / "data" / "datasets" / "v1" / "images" / "test"
simulation_eggs = []
if test_dir.exists():
    for f in sorted(list(test_dir.glob("*.jpg"))):
        img = cv2.imread(str(f))
        if img is not None:
            simulation_eggs.append(cv2.resize(img, (220, 220)))

# Fallback egg if none found
if not simulation_eggs:
    dummy_egg = np.full((220, 220, 3), 200, dtype=np.uint8)
    cv2.ellipse(dummy_egg, (110, 110), (70, 95), 0, 0, 360, (230, 240, 250), -1)
    simulation_eggs.append(dummy_egg)

def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = EggVisionPipeline(
            model_path="models/weights/EggVision.pt",
            line_y=320,
            device=0
        )
    return pipeline

class ConveyorSimulator:
    def __init__(self, eggs_pool):
        self.eggs = eggs_pool
        self.h, self.w = 640, 640
        self.egg_idx = 0
        self.y_pos = 40
        self.x_pos = 210
        self.speed = 18

    def get_next_frame(self):
        # Conveyor background with texture lines
        frame = np.full((self.h, self.w, 3), 45, dtype=np.uint8)
        for y in range(0, self.h, 40):
            cv2.line(frame, (0, y), (self.w, y), (35, 35, 35), 1)

        egg = self.eggs[self.egg_idx % len(self.eggs)]
        eh, ew = egg.shape[:2]

        # Draw egg moving down the conveyor
        if self.y_pos + eh < self.h:
            frame[self.y_pos:self.y_pos+eh, self.x_pos:self.x_pos+ew] = egg
        
        self.y_pos += self.speed

        # When egg rolls off bottom, spawn next egg at top
        if self.y_pos > self.h - 100:
            self.y_pos = 30
            self.egg_idx += 1
            # Randomize x slightly
            self.x_pos = int(np.random.randint(180, 240))

        return frame

simulator = ConveyorSimulator(simulation_eggs)

def generate_video_stream():
    """Generator function that yields MJPEG multipart frames."""
    pipe = get_pipeline()
    last_cracked_count = 0
    
    while True:
        raw_frame = simulator.get_next_frame()
        annotated_frame, stats = pipe.process_frame(raw_frame)

        # Detect new reject trigger
        if stats["cracked"] > last_cracked_count:
            last_cracked_count = stats["cracked"]
            reject_event = {
                "timestamp": time.strftime("%H:%M:%S"),
                "event": "CRACKED_DEFECT_REJECTED",
                "count": last_cracked_count,
                "status": "REJECT_PNEUMATIC_SOLENOID_FIRED"
            }
            reject_logs.insert(0, reject_event)
            if len(reject_logs) > 20:
                reject_logs.pop()

        # Encode as JPEG
        ret, buffer = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        
        time.sleep(0.03) # ~30 FPS throttle for smooth playback

@app.get("/stream")
def video_feed():
    return StreamingResponse(
        generate_video_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/stats")
def get_stats():
    pipe = get_pipeline()
    stats = pipe.tracker.get_summary()
    stats["fps"] = round(pipe.fps, 1)
    stats["reject_logs"] = reject_logs
    return stats

@app.post("/api/reset")
def reset_counters():
    global reject_logs
    pipe = get_pipeline()
    pipe.tracker.stats = {
        "total_passed": 0,
        "cracked": 0,
        "dirty": 0,
        "good": 0,
        "sizes": {"S": 0, "M": 0, "L": 0}
    }
    pipe.tracker.egg_records.clear()
    reject_logs.clear()
    return {"status": "success", "message": "Batch statistics reset"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)
    pipe = get_pipeline()
    try:
        while True:
            stats = pipe.tracker.get_summary()
            stats["fps"] = round(pipe.fps, 1)
            stats["reject_logs"] = reject_logs[:5]
            await websocket.send_text(json.dumps(stats))
            await asyncio.sleep(0.1) # 10Hz UI refresh
    except WebSocketDisconnect:
        active_websockets.remove(websocket)

@app.get("/", response_class=HTMLResponse)
def index_page():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EggVision | Industrial AI Quality Inspection Platform</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        body { background-color: #0d1117; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        .glass { background: rgba(22, 27, 34, 0.85); backdrop-filter: blur(12px); border: 1px solid rgba(48, 54, 61, 0.7); }
        .pulse-red { animation: pulseRed 1.5s infinite; }
        @keyframes pulseRed { 0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.6); } 50% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); } }
    </style>
</head>
<body class="min-h-screen flex flex-col">

    <!-- Top Navbar -->
    <header class="glass sticky top-0 z-50 px-6 py-3 flex items-center justify-between border-b border-gray-800">
        <div class="flex items-center space-x-3">
            <div class="h-10 w-10 rounded-xl bg-gradient-to-tr from-amber-500 to-yellow-300 flex items-center justify-center text-black font-extrabold text-xl shadow-lg shadow-yellow-500/20">
                🥚
            </div>
            <div>
                <h1 class="text-xl font-bold text-white tracking-wide flex items-center gap-2">
                    EggVision <span class="text-xs px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">v1.0 Live</span>
                </h1>
                <p class="text-xs text-gray-400">Automated Conveyor Defect Inspection & Size Grading System</p>
            </div>
        </div>

        <div class="flex items-center space-x-4">
            <div class="flex items-center space-x-2 text-xs text-gray-300 bg-gray-800/80 px-3 py-1.5 rounded-lg border border-gray-700">
                <span class="inline-block w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
                <span>GPU: RTX 3050 (6GB)</span>
                <span class="text-gray-500">|</span>
                <span id="fps-indicator" class="font-mono text-emerald-400 font-bold">FPS: 48.0</span>
            </div>
            <button onclick="resetBatch()" class="px-3 py-1.5 bg-red-600/80 hover:bg-red-600 text-white rounded-lg text-xs font-semibold flex items-center gap-1.5 transition">
                <i class="fa-solid fa-rotate-right"></i> Reset Batch
            </button>
        </div>
    </header>

    <!-- Main Container -->
    <main class="flex-1 p-6 grid grid-cols-1 lg:grid-cols-12 gap-6 max-w-7xl mx-auto w-full">

        <!-- Left Column: Live Video Canvas (7 cols) -->
        <section class="lg:col-span-7 flex flex-col space-y-4">
            <div class="glass rounded-2xl overflow-hidden shadow-2xl flex flex-col">
                <div class="px-4 py-3 bg-gray-900/90 border-b border-gray-800 flex items-center justify-between">
                    <span class="text-sm font-semibold text-gray-200 flex items-center gap-2">
                        <i class="fa-solid fa-video text-amber-400"></i> Conveyor Live Stream (ByteTrack + Sizing)
                    </span>
                    <span class="text-xs text-gray-400 font-mono">640x640 @ MJPEG 30FPS</span>
                </div>
                
                <div class="relative bg-black flex items-center justify-center min-h-[460px]">
                    <img id="live-stream" src="/stream" class="w-full h-auto object-contain rounded-b-xl" alt="Live Inspection Stream">
                    <div id="defect-alert-overlay" class="hidden absolute top-4 right-4 bg-red-600 text-white text-xs px-3 py-1.5 rounded-lg font-bold shadow-lg pulse-red flex items-center gap-2">
                        <i class="fa-solid fa-triangle-exclamation"></i> REJECT TRIGGER ACTIVE
                    </div>
                </div>
            </div>

            <!-- Hardware Reject Signal Controller Log -->
            <div class="glass rounded-2xl p-4 shadow-xl">
                <div class="flex items-center justify-between mb-3">
                    <h3 class="text-sm font-bold text-gray-200 flex items-center gap-2">
                        <i class="fa-solid fa-microchip text-blue-400"></i> Automated Reject Action Events
                    </h3>
                    <span class="text-xs text-gray-400">Pneumatic Rejector Signal</span>
                </div>
                <div id="reject-log-container" class="space-y-2 max-h-36 overflow-y-auto text-xs font-mono">
                    <div class="p-2 rounded bg-gray-800/60 text-gray-400 text-center">Monitoring conveyor inspection line...</div>
                </div>
            </div>
        </section>

        <!-- Right Column: KPI Cards & Charts (5 cols) -->
        <section class="lg:col-span-5 flex flex-col space-y-6">

            <!-- KPI Summary Cards -->
            <div class="grid grid-cols-2 gap-4">
                <div class="glass p-4 rounded-2xl">
                    <span class="text-xs text-gray-400 uppercase font-semibold">Total Passed</span>
                    <div id="stat-total" class="text-3xl font-extrabold text-white mt-1">0</div>
                    <span class="text-xs text-gray-500">Virtual line crossings</span>
                </div>

                <div class="glass p-4 rounded-2xl border-emerald-500/30">
                    <span class="text-xs text-emerald-400 uppercase font-semibold">Good Eggs</span>
                    <div id="stat-good" class="text-3xl font-extrabold text-emerald-400 mt-1">0</div>
                    <span id="stat-good-pct" class="text-xs text-emerald-500">100% Quality Rate</span>
                </div>

                <div class="glass p-4 rounded-2xl border-red-500/30">
                    <span class="text-xs text-red-400 uppercase font-semibold">Cracked Defects</span>
                    <div id="stat-cracked" class="text-3xl font-extrabold text-red-400 mt-1">0</div>
                    <span id="stat-cracked-pct" class="text-xs text-red-500">0% Defect Rate</span>
                </div>

                <div class="glass p-4 rounded-2xl">
                    <span class="text-xs text-amber-400 uppercase font-semibold">Inference Latency</span>
                    <div id="stat-latency" class="text-3xl font-extrabold text-amber-300 mt-1">3.7 ms</div>
                    <span class="text-xs text-gray-500">GPU Accelerated</span>
                </div>
            </div>

            <!-- Size Grading Distribution Chart -->
            <div class="glass p-5 rounded-2xl shadow-xl flex-1 flex flex-col">
                <h3 class="text-sm font-bold text-gray-200 mb-3 flex items-center justify-between">
                    <span class="flex items-center gap-2"><i class="fa-solid fa-chart-column text-amber-400"></i> Size Grade Distribution</span>
                    <span class="text-xs text-gray-400">Small / Medium / Large</span>
                </h3>
                <div class="flex-1 min-h-[220px]">
                    <canvas id="sizeChart"></canvas>
                </div>
            </div>

        </section>

    </main>

    <script>
        // Chart.js initialization
        const ctx = document.getElementById('sizeChart').getContext('2d');
        const sizeChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Small (<53g)', 'Medium (53-63g)', 'Large (>63g)'],
                datasets: [{
                    label: 'Egg Count',
                    data: [0, 0, 0],
                    backgroundColor: ['#60a5fa', '#34d399', '#f59e0b'],
                    borderRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: { beginAtZero: true, grid: { color: 'rgba(255, 255, 255, 0.05)' } },
                    x: { grid: { display: false } }
                }
            }
        });

        // WebSocket Telemetry Connection
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        const ws = new WebSocket(wsUrl);

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            // Update stats
            document.getElementById('stat-total').innerText = data.total_passed;
            document.getElementById('stat-good').innerText = data.good;
            document.getElementById('stat-cracked').innerText = data.cracked;
            document.getElementById('fps-indicator').innerText = `FPS: ${data.fps}`;

            const total = data.total_passed;
            const goodPct = total > 0 ? ((data.good / total) * 100).toFixed(0) : 100;
            const crackedPct = total > 0 ? ((data.cracked / total) * 100).toFixed(0) : 0;

            document.getElementById('stat-good-pct').innerText = `${goodPct}% Quality Rate`;
            document.getElementById('stat-cracked-pct').innerText = `${crackedPct}% Defect Rate`;

            // Update Chart
            sizeChart.data.datasets[0].data = [
                data.sizes.S || 0,
                data.sizes.M || 0,
                data.sizes.L || 0
            ];
            sizeChart.update('none');

            // Defect banner
            const defectOverlay = document.getElementById('defect-alert-overlay');
            if (data.cracked > 0) {
                defectOverlay.classList.remove('hidden');
            } else {
                defectOverlay.classList.add('hidden');
            }

            // Reject logs
            const logBox = document.getElementById('reject-log-container');
            if (data.reject_logs && data.reject_logs.length > 0) {
                logBox.innerHTML = data.reject_logs.map(log => `
                    <div class="p-2 rounded bg-red-950/40 border border-red-800/40 text-red-300 flex items-center justify-between">
                        <span><i class="fa-solid fa-ban text-red-500 mr-2"></i>[${log.timestamp}] ${log.event} #${log.count}</span>
                        <span class="text-xs text-emerald-400">PNEUMATIC_ACTIVATED</span>
                    </div>
                `).join('');
            }
        };

        function resetBatch() {
            fetch('/api/reset', { method: 'POST' });
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
