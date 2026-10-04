import sys
import time
from pathlib import Path
from backend.app.cv.job_manager import job_manager

def run_demo(video_path: str, registry_dir: str):
    print("=================================================================")
    print("      DEV 2 AI/CV INTEGRATION VERTICAL SLICE DEMO")
    print("=================================================================")
    
    if not Path(video_path).exists():
        print(f"ERROR: Video file not found: {video_path}")
        sys.exit(1)
        
    if not Path(registry_dir).exists():
        print(f"ERROR: Registry directory not found: {registry_dir}")
        sys.exit(1)

    print(f"[*] Starting job for video: {video_path}")
    print(f"[*] Using registry: {registry_dir}")
    
    # 1. Start Job
    job_id = job_manager.start_file_job(
        class_name="Demo Class",
        video_path=video_path,
        registry_dir=registry_dir
    )
    print(f"[*] Job started. ID: {job_id}")
    
    # 2. Poll Status
    print("[*] Polling status...")
    while True:
        status = job_manager.get_job_status(job_id)
        if status["status"] == "running":
            print("   ... Pipeline running ...")
        elif status["status"] == "done":
            print("\n[*] Processing Complete!")
            print("[*] Normalized Results:")
            for student in status.get("results", []):
                print(f"    - {student['name']}: {student['status']} (Confidence: {student['confidence']})")
            break
        elif status["status"] == "failed":
            print(f"\n[!] Processing Failed: {status.get('error')}")
            break
        else:
            print(f"   ... Status: {status['status']}")
            
        time.sleep(2)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python backend/app/cv/demo.py <path_to_video.mp4> <path_to_registry_dir>")
        sys.exit(1)
    
    run_demo(sys.argv[1], sys.argv[2])
