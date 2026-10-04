from fastapi import FastAPI

app = FastAPI(title="OctoVizn X MPGI Backend")

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "octovizn-mpgi-backend"}

# TODO: Include /api/v1 routers here
