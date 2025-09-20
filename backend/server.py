from fastapi import FastAPI, APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import json
import uuid
import asyncio
import time
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta
from emergentintegrations.llm.chat import LlmChat, UserMessage
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.utils
import pandas as pd
import numpy as np
import aiofiles
import websockets

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create static directory for agent results
STATIC_DIR = ROOT_DIR / "static"
AGENT_RESULTS_DIR = STATIC_DIR / "agent_results"
AGENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Create the main app
app = FastAPI(title="GNSS Mission Control", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Create API router
api_router = APIRouter(prefix="/api")

# Store WebSocket connections for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

# Pydantic Models
class Satellite(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    satellite_id: str
    constellation: str  # GPS, GLONASS, Galileo, BeiDou
    status: str = "active"
    last_contact: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    orbital_elements: Dict[str, float] = {}

class PredictionData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime
    satellite_id: str
    constellation: str
    pred_clock_error_m: float
    pred_orbit_error_m: float

class AgentQuery(BaseModel):
    prompt: str
    context: Optional[Dict[str, Any]] = {}

class AgentJob(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str
    status: str = "pending"  # pending, running, completed, failed
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None
    logs: List[str] = []
    error: Optional[str] = None

# Initialize LLM Chat for AI Agent
def get_llm_chat():
    return LlmChat(
        api_key=os.environ.get("EMERGENT_LLM_KEY"),
        session_id=f"gnss_agent_{uuid.uuid4()}",
        system_message="""You are a GNSS Mission Control AI Agent specialized in satellite orbit and clock error analysis. 
        You help space systems engineers analyze satellite prediction data and generate insights.
        
        You have access to tools for:
        1. Data analysis and statistics
        2. Chart generation
        3. Comparative analysis between satellites
        4. Threshold monitoring
        
        Always provide clear, precise technical analysis suitable for NASA/ISRO engineers.
        Keep responses concise but thorough."""
    ).with_model("openai", "gpt-5")

# Satellite data endpoints
@api_router.get("/satellites", response_model=List[Satellite])
async def get_satellites():
    """Get list of all satellites with metadata"""
    satellites = await db.satellites.find().to_list(1000)
    # Convert ObjectId to string for JSON serialization
    for sat in satellites:
        if '_id' in sat:
            sat['_id'] = str(sat['_id'])
    return [Satellite(**sat) for sat in satellites]

@api_router.get("/predictions")
async def get_predictions(
    satellite_id: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None
):
    """Get prediction data for satellites within time range"""
    query = {}
    
    if satellite_id:
        query["satellite_id"] = satellite_id
    
    if start:
        start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
        query["timestamp"] = {"$gte": start_dt}
    
    if end:
        end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))
        if "timestamp" in query:
            query["timestamp"]["$lte"] = end_dt
        else:
            query["timestamp"] = {"$lte": end_dt}
    
    predictions = await db.predictions.find(query).sort("timestamp", 1).to_list(10000)
    # Convert ObjectId to string for JSON serialization
    for pred in predictions:
        if '_id' in pred:
            pred['_id'] = str(pred['_id'])
    return predictions

@api_router.get("/satellite/{satellite_id}/summary")
async def get_satellite_summary(satellite_id: str):
    """Get satellite summary with latest prediction data"""
    # Get satellite info
    satellite = await db.satellites.find_one({"satellite_id": satellite_id})
    if not satellite:
        raise HTTPException(status_code=404, detail="Satellite not found")
    
    # Convert ObjectId to string for JSON serialization
    if '_id' in satellite:
        satellite['_id'] = str(satellite['_id'])
    
    # Get latest predictions (last 24 hours)
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=24)
    
    predictions = await db.predictions.find({
        "satellite_id": satellite_id,
        "timestamp": {"$gte": start_time, "$lte": end_time}
    }).sort("timestamp", 1).to_list(1000)
    
    # Convert ObjectId to string for JSON serialization
    for pred in predictions:
        if '_id' in pred:
            pred['_id'] = str(pred['_id'])
    
    if not predictions:
        return {
            "satellite": satellite,
            "predictions": [],
            "summary": {
                "peak_clock_error": 0,
                "peak_orbit_error": 0,
                "avg_clock_error": 0,
                "avg_orbit_error": 0
            }
        }
    
    # Calculate summary statistics
    clock_errors = [p["pred_clock_error_m"] for p in predictions]
    orbit_errors = [p["pred_orbit_error_m"] for p in predictions]
    
    summary = {
        "peak_clock_error": max(clock_errors),
        "peak_orbit_error": max(orbit_errors),
        "avg_clock_error": sum(clock_errors) / len(clock_errors),
        "avg_orbit_error": sum(orbit_errors) / len(orbit_errors),
        "data_points": len(predictions)
    }
    
    return {
        "satellite": satellite,
        "predictions": predictions,
        "summary": summary
    }

# AI Agent endpoints
@api_router.post("/agent/query")
async def create_agent_query(query: AgentQuery):
    """Submit a query to the AI Agent for analysis"""
    job = AgentJob(prompt=query.prompt)
    
    # Store job in database
    await db.agent_jobs.insert_one(job.dict())
    
    # Start processing in background
    asyncio.create_task(process_agent_query(job.id, query))
    
    return {"job_id": job.id, "status": "pending"}

@api_router.get("/agent/result/{job_id}")
async def get_agent_result(job_id: str):
    """Get results of an AI Agent job"""
    job = await db.agent_jobs.find_one({"id": job_id})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return AgentJob(**job)

async def process_agent_query(job_id: str, query: AgentQuery):
    """Process AI Agent query in background"""
    try:
        # Update job status
        await db.agent_jobs.update_one(
            {"id": job_id},
            {"$set": {"status": "running"}}
        )
        
        # Generate analysis plan
        chat = get_llm_chat()
        
        # First, get the data analysis request
        plan_prompt = f"""
        Analyze this GNSS satellite data analysis request: "{query.prompt}"
        
        Generate a 3-step analysis plan. Available data includes:
        - Satellite prediction data with timestamp, satellite_id, constellation, pred_clock_error_m, pred_orbit_error_m
        - Time range: last 24 hours
        - Constellations: GPS, GLONASS, Galileo, BeiDou
        
        Respond with a JSON structure:
        {{
            "steps": [
                "Step 1: Description",
                "Step 2: Description", 
                "Step 3: Description"
            ],
            "analysis_type": "comparison|threshold|statistics|visualization",
            "satellites_needed": ["G01", "R05"] or "all"
        }}
        """
        
        plan_response = await chat.send_message(UserMessage(text=plan_prompt))
        
        # Execute the analysis
        results = await execute_analysis_plan(plan_response, query.context)
        
        # Update job with results
        await db.agent_jobs.update_one(
            {"id": job_id},
            {
                "$set": {
                    "status": "completed",
                    "completed_at": datetime.now(timezone.utc),
                    "results": results
                },
                "$push": {"logs": f"Analysis completed successfully"}
            }
        )
        
        # Broadcast update via WebSocket
        await manager.broadcast({
            "type": "agent_job_update",
            "job_id": job_id,
            "status": "completed"
        })
        
    except Exception as e:
        logging.error(f"Agent query processing failed: {str(e)}")
        await db.agent_jobs.update_one(
            {"id": job_id},
            {
                "$set": {
                    "status": "failed",
                    "error": str(e),
                    "completed_at": datetime.now(timezone.utc)
                }
            }
        )

async def execute_analysis_plan(plan_response: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the analysis plan with restricted Python execution"""
    try:
        # Parse the plan (simplified for demo)
        # In production, this would use proper JSON parsing and validation
        
        # Get sample data for analysis
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=24)
        
        predictions = await db.predictions.find({
            "timestamp": {"$gte": start_time, "$lte": end_time}
        }).to_list(10000)
        
        if not predictions:
            return {
                "text_summary": "No prediction data available for analysis",
                "charts": [],
                "attachments": []
            }
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(predictions)
        
        # Generate basic statistics
        stats = {
            "total_satellites": df["satellite_id"].nunique(),
            "total_data_points": len(df),
            "avg_clock_error": float(df["pred_clock_error_m"].mean()),
            "max_clock_error": float(df["pred_clock_error_m"].max()),
            "avg_orbit_error": float(df["pred_orbit_error_m"].mean()),
            "max_orbit_error": float(df["pred_orbit_error_m"].max())
        }
        
        # Generate chart
        chart_url = await generate_analysis_chart(df)
        
        # Generate summary
        summary = f"""
        Analysis Results:
        - Total satellites analyzed: {stats['total_satellites']}
        - Data points: {stats['total_data_points']}
        - Average clock error: {stats['avg_clock_error']:.4f}m
        - Maximum clock error: {stats['max_clock_error']:.4f}m
        - Average orbit error: {stats['avg_orbit_error']:.2f}m
        - Maximum orbit error: {stats['max_orbit_error']:.2f}m
        
        The analysis shows variation in both clock and orbit prediction errors across the satellite constellation.
        """
        
        return {
            "text_summary": summary,
            "charts": [{"name": "Prediction Errors Overview", "url": chart_url}],
            "attachments": [],
            "statistics": stats
        }
        
    except Exception as e:
        raise Exception(f"Analysis execution failed: {str(e)}")

async def generate_analysis_chart(df: pd.DataFrame) -> str:
    """Generate analysis chart and return URL"""
    try:
        # Create Plotly chart
        fig = go.Figure()
        
        # Add clock error trace
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["pred_clock_error_m"],
            mode='markers',
            name='Clock Error (m)',
            marker=dict(color='blue', size=4)
        ))
        
        # Add orbit error trace on secondary y-axis
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["pred_orbit_error_m"],
            mode='markers',
            name='Orbit Error (m)',
            marker=dict(color='red', size=4),
            yaxis='y2'
        ))
        
        # Update layout
        fig.update_layout(
            title="GNSS Satellite Prediction Errors (24h)",
            xaxis_title="Time",
            yaxis=dict(title="Clock Error (m)", side="left"),
            yaxis2=dict(title="Orbit Error (m)", side="right", overlaying="y"),
            width=800,
            height=500,
            showlegend=True
        )
        
        # Save chart
        chart_id = str(uuid.uuid4())
        chart_path = AGENT_RESULTS_DIR / f"{chart_id}.png"
        fig.write_image(str(chart_path))
        
        return f"/static/agent_results/{chart_id}.png"
        
    except Exception as e:
        logging.error(f"Chart generation failed: {str(e)}")
        return ""

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Send periodic orbit updates
            await asyncio.sleep(5)
            
            # Get latest satellite positions (simulated)
            satellites = await db.satellites.find().to_list(10)
            
            orbit_data = []
            for sat in satellites:
                # Simulate orbital position update
                t = time.time()
                orbit_data.append({
                    "satellite_id": sat["satellite_id"],
                    "lat": 45 + 20 * np.sin(t * 0.001 + hash(sat["satellite_id"]) % 100),
                    "lon": np.sin(t * 0.0005 + hash(sat["satellite_id"]) % 100) * 180,
                    "alt": 20000 + 2000 * np.sin(t * 0.0008)
                })
            
            await websocket.send_json({
                "type": "orbit_update",
                "data": orbit_data
            })
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Include the router in the main app
app.include_router(api_router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    """Initialize database with sample data"""
    logger.info("Starting GNSS Mission Control...")
    
    # Check if we need to initialize sample data
    satellite_count = await db.satellites.count_documents({})
    if satellite_count == 0:
        await initialize_sample_data()

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

async def initialize_sample_data():
    """Initialize database with sample GNSS satellite and prediction data"""
    logger.info("Initializing sample data...")
    
    # Sample satellites
    satellites = [
        {"satellite_id": "G01", "constellation": "GPS", "status": "active"},
        {"satellite_id": "G02", "constellation": "GPS", "status": "active"},
        {"satellite_id": "G03", "constellation": "GPS", "status": "active"},
        {"satellite_id": "R01", "constellation": "GLONASS", "status": "active"},
        {"satellite_id": "R02", "constellation": "GLONASS", "status": "active"},
        {"satellite_id": "E01", "constellation": "Galileo", "status": "active"},
        {"satellite_id": "E02", "constellation": "Galileo", "status": "active"},
        {"satellite_id": "C01", "constellation": "BeiDou", "status": "active"},
        {"satellite_id": "C02", "constellation": "BeiDou", "status": "active"},
        {"satellite_id": "C03", "constellation": "BeiDou", "status": "active"},
    ]
    
    # Insert satellites
    for sat_data in satellites:
        satellite = Satellite(**sat_data)
        await db.satellites.insert_one(satellite.dict())
    
    # Generate 24-hour prediction data (15-minute intervals)
    start_time = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    
    predictions = []
    for i in range(96):  # 24 hours * 4 (15-min intervals)
        timestamp = start_time + timedelta(minutes=i * 15)
        
        for sat_data in satellites:
            # Generate realistic prediction errors
            base_clock_error = np.random.normal(0.003, 0.001)  # ~3mm ± 1mm
            base_orbit_error = np.random.normal(0.5, 0.2)      # ~50cm ± 20cm
            
            # Add some time-based variation
            time_factor = np.sin(i * 0.1) * 0.0005
            
            prediction = PredictionData(
                timestamp=timestamp,
                satellite_id=sat_data["satellite_id"],
                constellation=sat_data["constellation"],
                pred_clock_error_m=abs(base_clock_error + time_factor),
                pred_orbit_error_m=abs(base_orbit_error + time_factor * 100)
            )
            predictions.append(prediction.dict())
    
    # Insert predictions
    await db.predictions.insert_many(predictions)
    
    logger.info(f"Initialized {len(satellites)} satellites and {len(predictions)} prediction data points")