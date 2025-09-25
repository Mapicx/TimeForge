import os
import uuid
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware



# --- Configuration ---
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# The server now uses a CSV file as its primary data source.
CSV_FILE_PATH = ROOT_DIR / "suyash_gandu.csv"
prediction_df = None

# In-memory storage for simulated AI agent jobs
agent_jobs: Dict[str, Any] = {}

# --- Pydantic Models (Corrected for Clock Error Data) ---
class Satellite(BaseModel):
    satellite_id: str
    constellation: str

class PredictionData(BaseModel):
    timestamp: datetime
    satellite_id: str
    constellation: str
    # The CSV data is now correctly mapped to clock error.
    pred_clock_error_m: float
    pred_orbit_error_m: float = 0.0 # Orbit error is now zero

class SatelliteSummary(BaseModel):
    # Summary is now based on clock error
    peak_clock_error: float
    avg_clock_error: float
    data_points: int

class AgentQuery(BaseModel):
    prompt: str
    context: Optional[Dict[str, Any]] = {}

class AgentJob(BaseModel):
    id: str
    status: str
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# --- Helper Functions ---
def get_constellation_from_prn(prn: str) -> str:
    """Determines satellite constellation from its PRN/ID."""
    if prn.startswith('G'):
        return 'GPS'
    elif prn.startswith('R'):
        return 'GLONASS'
    elif prn.startswith('E'):
        return 'Galileo'
    elif prn.startswith('C'):
        return 'BeiDou'
    return 'Unknown'

# --- FastAPI App Setup ---
app = FastAPI(title="GNSS Mission Control (CSV-Powered)", version="2.1.0")
api_router = APIRouter(prefix="/api")

# For development, this is your local React app.
origins = [
    "http://localhost:3000",
]

# CORS middleware allows your frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Use the specific list of origins
    allow_credentials=True,
    allow_methods=["*"],    # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],    # Allows all headers
)

# --- API Endpoints ---
@api_router.get("/satellites", response_model=List[Satellite])
async def get_satellites():
    """Get a sorted list of all unique satellites from the CSV file."""
    if prediction_df is None:
        raise HTTPException(status_code=503, detail="Server is initializing, data not yet available.")
    
    unique_prns = sorted(prediction_df['PRN'].unique())
    satellites = [
        Satellite(
            satellite_id=prn,
            constellation=get_constellation_from_prn(prn)
        ) for prn in unique_prns
    ]
    return satellites

@api_router.get("/satellite/{satellite_id}/summary")
async def get_satellite_summary(satellite_id: str):
    """
    Get satellite summary with future prediction data generated from the CSV.
    """
    if prediction_df is None:
        raise HTTPException(status_code=503, detail="Server is initializing, data not yet available.")

    satellite_data = prediction_df[prediction_df['PRN'] == satellite_id].sort_values(by='datetime', ascending=False)
    if satellite_data.empty:
        raise HTTPException(status_code=404, detail=f"Satellite '{satellite_id}' not found in the data source.")
    
    latest_entry = satellite_data.iloc[0]
    
    start_time = latest_entry['datetime']
    constellation = get_constellation_from_prn(satellite_id)
    
    pred_columns = [f'pred_{i}' for i in range(1, 241)]
    pred_values = latest_entry[pred_columns].values
    
    predictions = []
    for i, value in enumerate(pred_values):
        timestamp = start_time + timedelta(seconds=(i + 1) * 30)
        predictions.append(PredictionData(
            timestamp=timestamp,
            satellite_id=satellite_id,
            constellation=constellation,
            # Assign the CSV value to the correct field
            pred_clock_error_m=float(value)
        ))
        
    # Calculate summary stats based on clock error
    clock_errors = [p.pred_clock_error_m for p in predictions]
    summary_stats = SatelliteSummary(
        peak_clock_error=max(clock_errors),
        avg_clock_error=sum(clock_errors) / len(clock_errors),
        data_points=len(predictions)
    )

    return {
        "satellite": Satellite(satellite_id=satellite_id, constellation=constellation).dict(),
        "summary": summary_stats.dict(),
        "predictions": [p.dict() for p in predictions]
    }

# --- AI Agent Endpoints (Simulated Response) ---
# These endpoints are functional but provide a simulated response to avoid errors
# from missing external libraries or API keys.

@api_router.post("/agent/query")
async def create_agent_query(query: AgentQuery):
    """Accepts an AI agent query and returns a job ID for a simulated analysis."""
    job_id = str(uuid.uuid4())
    logger.info(f"Received agent query '{query.prompt}'. Simulating job {job_id}.")
    
    # Simulate a completed job with a generic response
    simulated_results = {
        "text_summary": f"This is a simulated AI analysis for the query: '{query.prompt}'. "
                        "The analysis indicates that the satellite performance is within expected parameters. "
                        "No anomalies detected based on the provided prediction data.",
        "charts": [], # No charts are generated in the simulated response
        "statistics": {
            "query_received": query.prompt,
            "satellites_in_context": query.context.get("satellite_ids", []),
            "confidence_score": 0.95,
            "status": "Simulated Analysis Complete"
        }
    }
    
    agent_jobs[job_id] = AgentJob(id=job_id, status="completed", results=simulated_results).dict()
    
    return {"job_id": job_id, "status": "pending"}

@api_router.get("/agent/result/{job_id}", response_model=AgentJob)
async def get_agent_result(job_id: str):
    """Retrieves the result of the simulated AI agent job."""
    job = agent_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Agent job not found.")
    return job

# --- App Startup Event ---
@app.on_event("startup")
def startup_event():
    """On startup, load and sort the CSV data into memory."""
    global prediction_df
    logger.info("Starting GNSS Mission Control Server...")
    try:
        logger.info(f"Loading prediction data from {CSV_FILE_PATH}...")
        prediction_df = pd.read_csv(CSV_FILE_PATH)
        prediction_df['datetime'] = pd.to_datetime(prediction_df['datetime'])
        
        logger.info("Sorting data by satellite ID (PRN) and timestamp...")
        prediction_df.sort_values(by=['PRN', 'datetime'], ascending=[True, True], inplace=True)
        
        logger.info(f"Successfully loaded and sorted {len(prediction_df)} records.")
    except FileNotFoundError:
        logger.error(f"FATAL ERROR: The data file '{CSV_FILE_PATH.name}' was not found in the directory '{ROOT_DIR}'.")
        prediction_df = None
    except Exception as e:
        logger.error(f"An error occurred while loading or sorting the CSV file: {e}")
        prediction_df = None

# --- Final App Configuration ---
app.include_router(api_router)

# CORS middleware allows your frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

