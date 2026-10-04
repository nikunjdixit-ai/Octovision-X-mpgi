import uuid
from typing import Dict, Any, Optional
from backend.app.cv.pipeline_adapter import CVPipelineAdapter
from backend.app.cv.result_parser import parse_pipeline_results

class JobManager:
    """Simple in-memory job manager for the CV integration demo."""
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def start_file_job(self, class_name: str, video_path: str, registry_dir: str) -> str:
        job_id = str(uuid.uuid4())
        adapter = CVPipelineAdapter(camera_id=job_id, session_id=job_id)
        adapter.configure_file(video_path=video_path, registry_dir=registry_dir)
        
        self._jobs[job_id] = {
            "adapter": adapter,
            "class_name": class_name,
            "source_type": "mp4",
            "source_name": video_path.split("/")[-1] if "/" in video_path else video_path.split("\\")[-1],
            "source_size": 0, # Placeholder for demo
            "status": "queued"
        }
        
        adapter.start()
        return job_id

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        job = self._jobs.get(job_id)
        if not job:
            return {"error": "Job not found"}
        
        adapter: CVPipelineAdapter = job["adapter"]
        status = adapter.get_status()
        
        if status == "done":
            raw_results = adapter.get_results()
            final_results = parse_pipeline_results(
                class_name=job["class_name"],
                source_type=job["source_type"],
                source_name=job["source_name"],
                source_size=job["source_size"],
                engine_summary=raw_results
            )
            return {
                "status": "done",
                "progress": 1.0,
                "results": final_results.get("students", []),
                "error": None
            }
        
        return {
            "status": status,
            "progress": 0.5 if status == "running" else 0.0,
            "results": [],
            "error": "Failed" if status == "failed" else None
        }

job_manager = JobManager()
