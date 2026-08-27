                                                 

import asyncio
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from queue_monitor import check_waiting_queue

app = FastAPI(title="PatientTriage.ai Gateway")

@app.on_event("startup")
async def startup_event():
                                                        
    asyncio.create_task(check_waiting_queue())

                                             
                           
     
                         
     

                                            
                                                 
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))

                                                                  
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

                                                            
@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
                                                                       
    file_path = os.path.join(FRONTEND_DIR, full_path)
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    
                                                       
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))