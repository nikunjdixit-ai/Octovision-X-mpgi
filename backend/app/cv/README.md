# Backend Developer 2 (CV Integration) - Demo Guide

This module contains the integration layer between the FastAPI backend and the core AI/CV pipeline (`src/`).

## Prerequisites
1. Ensure the Python virtual environment is activated and requirements are installed.
2. A valid test video (MP4/MOV).
3. A valid student registry directory containing `students_registry.json`.

## How to Start the Integration Demo

Run the vertical slice test using the provided `demo.py` script:

```bash
python backend/app/cv/demo.py <path_to_video.mp4> <path_to_registry_dir>
```

### Example
```bash
python backend/app/cv/demo.py C:/path/to/test_video.mp4 C:/path/to/registry
```

### Expected Output
1. The script initializes the adapter safely without crashing on import.
2. The ML models load in the background thread.
3. The video is processed using the actual CV pipeline.
4. The adapter extracts normalized session summaries (matching the PDF contract).
5. Output prints the detected student names, `present` status, and confidence values.
6. The job stops cleanly.

### Common Failure Cases
- **Video not found**: Provide an absolute path.
- **Registry missing**: Ensure the directory contains `students_registry.json`.
- **Model weights missing**: Make sure YOLO weights (`yolov8n.pt`) and OSNet weights are available in the root.
