import os
import asyncio
import uuid
import time
import json
import base64
from typing import Dict, List, Optional, Any
from datetime import datetime
import traceback

import httpx
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
from ollama import AsyncClient

# Load environment variables
load_dotenv()

app = FastAPI(title="Content Generation Server", version="1.0.0")

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
PORT = int(os.getenv("PORT", 3000))
EXTERNAL_API_BASE = os.getenv("EXTERNAL_API_BASE", "https://cloud-text-manager-server.vercel.app")
EXTERNAL_API_URL = f"{EXTERNAL_API_BASE}/api/files"
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

# In-memory Job Store
class Job(BaseModel):
    id: str
    fileId: str
    status: str  # 'pending' | 'processing' | 'completed' | 'failed'
    progress: int
    message: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    startTime: float
    endTime: Optional[float] = None

jobs: Dict[str, Job] = {}
# For SSE connections
queues: Dict[str, List[asyncio.Queue]] = {}

# --- Helpers ---

def get_ollama_client():
    if not OLLAMA_API_KEY:
        raise ValueError("OLLAMA_API_KEY is not defined in environment variables.")
    return AsyncClient(
        host="https://ollama.com", 
        headers={'Authorization': 'Bearer ' + OLLAMA_API_KEY}
    )

async def notify_clients(file_id: str, data: Dict[str, Any]):
    if file_id in queues:
        # Create a copy of the list to avoid modification during iteration
        for queue in queues[file_id][:]:
            await queue.put(data)

def update_job(job_id: str, updates: Dict[str, Any]):
    if job_id not in jobs:
        return
    
    job = jobs[job_id]
    for key, value in updates.items():
        setattr(job, key, value)
    
    if job.status in ['completed', 'failed']:
        job.endTime = time.time()
    
    # Notify SSE listeners
    asyncio.create_task(notify_clients(job.fileId, {
        "step": job.status,
        "progress": job.progress,
        "message": job.message,
        "jobId": job.id
    }))

def select_model(prompt: str) -> str:
    return "minimax-m2.7:cloud"

# --- Core Processing Logic ---

async def process_file_task(file_id: str, file_url: str, config: Dict[str, Any]):
    job_id = str(uuid.uuid4())
    job = Job(
        id=job_id,
        fileId=file_id,
        status='pending',
        progress=0,
        message='Job created',
        startTime=time.time()
    )
    jobs[job_id] = job
    
    update_job(job_id, {"status": "processing", "progress": 5, "message": "Starting processing..."})

    try:
        # 1. Fetch prompt content
        update_job(job_id, {"progress": 10, "message": "Fetching prompt from source..."})
        print(f"[{job_id}] Fetching prompt from: {file_url}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(file_url)
                response.raise_for_status()
                prompt_text = response.text
            except Exception as e:
                raise Exception(f"Failed to fetch prompt file: {str(e)}")

        # 2. Generate content with Ollama Cloud
        update_job(job_id, {"progress": 30, "message": "Generating content with Ollama Cloud..."})
        
        model = config.get('model') or select_model(prompt_text)
        client = get_ollama_client()
        
        generated_content = ""
        try:
            # Shift from generate to chat for doc parity
            messages = [{'role': 'user', 'content': prompt_text}]
            response = await client.chat(
                model=model,
                messages=messages,
                options={"num_predict": 4096}
            )
            # Response in chat is structured as ['message']['content']
            generated_content = response.get('message', {}).get('content', '')
            if not generated_content:
                raise Exception("Ollama Cloud returned empty content")
        except Exception as e:
            raise Exception(f"Ollama Cloud chat failed: {str(e)}")

        # 3. Upload to Cloudinary
        update_job(job_id, {"progress": 70, "message": "Uploading to Cloudinary..."})
        try:
            cloudinary.config(
                cloud_name=config.get('cloudName') or os.getenv('CLOUDINARY_CLOUD_NAME'),
                api_key=config.get('apiKey') or os.getenv('CLOUDINARY_API_KEY'),
                api_secret=config.get('apiSecret') or os.getenv('CLOUDINARY_API_SECRET'),
                secure=True
            )

            # Convert to base64 for data URI
            base64_content = base64.b64encode(generated_content.encode('utf-8')).decode('utf-8')
            data_uri = f"data:text/markdown;base64,{base64_content}"
            
            upload_result = cloudinary.uploader.upload(
                data_uri,
                resource_type="raw",
                folder="generated_articles",
                public_id=f"article_{file_id}_{int(time.time())}.mdx"
            )
        except Exception as e:
            raise Exception(f"Cloudinary upload failed: {str(e)}")

        # 4. Update External API Status
        update_job(job_id, {"progress": 90, "message": "Updating external API status..."})
        async with httpx.AsyncClient() as client:
            try:
                await client.put(f"{EXTERNAL_API_URL}/{file_id}", json={
                    "status": "AlreadyCopy",
                    "completedTimestamp": int(time.time() * 1000)
                })
            except Exception as e:
                print(f"[{job_id}] Warning: Failed to update status on external API: {str(e)}")

        # Complete Job
        update_job(job_id, {
            "status": "completed",
            "progress": 100,
            "message": "Process completed successfully!",
            "result": {"generatedUrl": upload_result.get("secure_url")}
        })

    except Exception as e:
        print(f"[{job_id}] Job failed: {str(e)}")
        update_job(job_id, {
            "status": "failed",
            "progress": 0,
            "message": str(e) or "An error occurred",
            "error": str(e)
        })

# --- Routes ---

@app.get("/")
async def root():
    return {
        "message": "Content Generation Server (Ollama Cloud) is Running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/health",
            "stats": "/api/stats",
            "generate": "/api/generate (POST)",
            "progress": "/api/progress/{fileId} (SSE)"
        }
    }

@app.get("/api/health")
async def health():
    status = {"status": "online", "timestamp": datetime.now().isoformat()}
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(EXTERNAL_API_BASE)
            status["externalApi"] = "online" if resp.status_code == 200 else "offline"
        except:
            status["externalApi"] = "offline"
    return status

@app.get("/api/stats")
async def stats():
    active = sum(1 for j in jobs.values() if j.status == 'processing')
    completed = sum(1 for j in jobs.values() if j.status == 'completed')
    failed = sum(1 for j in jobs.values() if j.status == 'failed')
    return {
        "activeJobs": active,
        "completedJobs": completed,
        "failedJobs": failed,
        "totalJobs": len(jobs)
    }

@app.get("/api/proxy/files")
async def proxy_files():
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(EXTERNAL_API_URL)
            return resp.json()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch files: {str(e)}")

@app.get("/api/progress/{file_id}")
async def progress(file_id: str):
    async def event_generator():
        queue = asyncio.Queue()
        if file_id not in queues:
            queues[file_id] = []
        queues[file_id].append(queue)
        
        # Initial message
        yield f"data: {json.dumps({'step': 'init', 'progress': 0, 'message': 'Connected'})}\n\n"
        
        # Check if job exists
        existing_job = next((j for j in jobs.values() if j.fileId == file_id and j.status == 'processing'), None)
        if existing_job:
            yield f"data: {json.dumps({'step': existing_job.status, 'progress': existing_job.progress, 'message': existing_job.message})}\n\n"

        try:
            while True:
                data = await queue.get()
                yield f"data: {json.dumps(data)}\n\n"
        except asyncio.CancelledError:
            queues[file_id].remove(queue)
            if not queues[file_id]:
                del queues[file_id]
            raise

    return StreamingResponse(event_generator(), media_type="text/event-stream")

class GenerateRequest(BaseModel):
    fileId: str
    fileUrl: str
    cloudinaryConfig: Optional[Dict[str, Any]] = None
    model: Optional[str] = None

@app.post("/api/generate")
async def generate(req: GenerateRequest, background_tasks: BackgroundTasks):
    if not req.fileId or not req.fileUrl:
        raise HTTPException(status_code=400, detail="Missing required parameters")
    
    config = req.cloudinaryConfig or {}
    if req.model:
        config['model'] = req.model
    
    background_tasks.add_task(process_file_task, req.fileId, req.fileUrl, config)
    
    # Return immediately as processing is in background
    return {"message": "Job started", "fileId": req.fileId}

# --- Automation Loop ---

async def automation_loop():
    print("[Automation] Starting automation loop...")
    while True:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.get(EXTERNAL_API_URL)
                files = resp.json()
                pending_files = [f for f in files if f.get('status') == 'Pending']
                
                print(f"[Automation] Found {len(pending_files)} pending files.")
                
                for file in pending_files:
                    file_id = file.get('id')
                    # Check if already processing
                    is_processing = any(j.fileId == file_id and j.status in ['processing', 'pending'] for j in jobs.values())
                    if is_processing:
                        continue
                        
                    print(f"[Automation] Starting processing for file {file_id}...")
                    # We run it in the main event loop
                    asyncio.create_task(process_file_task(file_id, file.get('secureUrl'), {}))
                    
                    # Wait a bit between files
                    await asyncio.sleep(5)
                    
        except Exception as e:
            print(f"[Automation] Error in loop: {str(e)}")
            traceback.print_exc()
            
        await asyncio.sleep(60) # Poll every 60 seconds

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(automation_loop())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
