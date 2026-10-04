from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.app.cv.job_manager import job_manager

router = APIRouter()

class StartJobRequest(BaseModel):
    class_name: str
    video_path: str
    registry_dir: str

@router.post("/jobs")
def start_processing_job(req: StartJobRequest):
    try:
        job_id = job_manager.start_file_job(
            class_name=req.class_name,
            video_path=req.video_path,
            registry_dir=req.registry_dir
        )
        return {"job_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/jobs/{job_id}")
def get_processing_job(job_id: str):
    res = job_manager.get_job_status(job_id)
    if "error" in res and res["error"] == "Job not found":
        raise HTTPException(status_code=404, detail="Job not found")
    return res
