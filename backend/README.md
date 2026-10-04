# OctoVizn X MPGI Backend

This is the backend foundation for the OctoVizn X MPGI project. It is built with FastAPI and provides APIs and database interaction layer around the existing CV pipeline.

## Architecture

- **Backend API**: The FastAPI application serving REST endpoints.
- **CV Integration Layer**: Code that interfaces with the existing AI/CV pipeline (`src/`).
- **Existing Pipeline**: The core AI/CV processing scripts in `src/`.

### Developer Roles

- **Backend Developer 1**: Core backend, database, APIs, authentication, dashboard APIs.
- **Backend Developer 2**: AI/CV integration, processing, recognition events, pipeline adapter, AI/Search team integration.

## Setup

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   Copy the example environment file and update values.
   ```bash
   cp .env.example .env
   ```
   **Note**: Never commit the `.env` file containing secrets.

4. **Run FastAPI:**
   ```bash
   cd backend
   uvicorn app.main:app --reload
   ```

5. **Health Endpoint:**
   Check `http://localhost:8000/health`
