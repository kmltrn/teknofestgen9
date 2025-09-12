import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

from simulators import (
    run_jamming_simulation, run_hqc_sat_simulation, run_blockchain_simulation,
    run_ddos_simulation, run_spoofing_simulation, run_isms_orchestration_simulation
)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get('/favicon.ico', include_in_schema=False)
async def favicon(): return FileResponse("favicon.ico")

@app.get("/")
async def get_index(): return FileResponse("index.html")

# GENEL WebSocket Handler
async def simulation_handler(websocket: WebSocket, simulation_function):
    await websocket.accept()
    print(f"WebSocket bağlantısı açıldı: {websocket.client}")
    try:
        # DÜZELTME: Artık tüm fonksiyonlar 'async' olduğu için 'async for' kullanıyoruz.
        async for log_data in simulation_function:
            await websocket.send_json(log_data)
    except Exception as e:
        print(f"Hata oluştu: {e}")
        try: await websocket.send_json({"type": "error", "message": str(e)})
        except: pass
    finally:
        print(f"WebSocket bağlantısı kapandı: {websocket.client_state}")

# WebSocket Adresleri
@app.websocket("/ws/jamming/{scenario_id}")
async def ws_jamming(websocket: WebSocket, scenario_id: str):
    await simulation_handler(websocket, run_jamming_simulation(scenario_id))

@app.websocket("/ws/hqc_sat/{scenario_id}")
async def ws_hqc(websocket: WebSocket, scenario_id: str):
    await simulation_handler(websocket, run_hqc_sat_simulation(scenario_id))

@app.websocket("/ws/blockchain/{scenario_id}")
async def ws_blockchain(websocket: WebSocket, scenario_id: str):
    await simulation_handler(websocket, run_blockchain_simulation(scenario_id))

@app.websocket("/ws/ddos/{scenario_id}")
async def ws_ddos(websocket: WebSocket, scenario_id: str):
    await simulation_handler(websocket, run_ddos_simulation(scenario_id))

@app.websocket("/ws/spoofing/{scenario_id}")
async def ws_spoofing(websocket: WebSocket, scenario_id: str):
    await simulation_handler(websocket, run_spoofing_simulation(scenario_id))

@app.websocket("/ws/isms/{scenario_id}")
async def ws_isms(websocket: WebSocket, scenario_id: str):
    await simulation_handler(websocket, run_isms_orchestration_simulation(scenario_id))