from typing import Dict, Any, List

def parse_pipeline_results(
    class_name: str,
    source_type: str,
    source_name: str,
    source_size: int,
    engine_summary: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Normalizes the raw session summary from AttendanceEngine into the 
    strict format expected by the dashboard frontend API.
    """
    students_out = []
    for s in engine_summary.get("students", []):
        students_out.append({
            "name": s.get("name", "Unknown"),
            "status": str(s.get("status", "absent")).lower(),
            "confidence": round(float(s.get("confidence", 0.0)), 2)
        })

    return {
        "class": class_name,
        "date": engine_summary.get("generated_at", ""),
        "source": {
            "type": source_type,
            "name": source_name,
            "size": source_size
        },
        "students": students_out
    }
