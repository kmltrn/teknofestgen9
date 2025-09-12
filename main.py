# main.py (Nihai Sürüm)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import asyncio
import json

# Tüm simülasyon fonksiyonlarının ve sınıflarının bulunduğu simulators.py dosyasını import et
try:
    import simulators
    print("Simülatör modülü başarıyla yüklendi.")
except ImportError:
    print("HATA: simulators.py dosyası bulunamadı veya içinde bir hata var.")
    exit()

app = FastAPI()

# Frontend'i (index.html) sunmak için
@app.get("/")
async def get_home():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>Hata: index.html dosyası bulunamadı.</h1>", status_code=404)

# Tüm simülasyonları yönetecek tek WebSocket endpoint'i
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket bağlantısı kabul edildi.")
    
    # Simülasyon fonksiyonlarını bir sözlükte eşleştir
    simulation_router = {
        "jamming": simulators.run_jamming_simulation,
        "hqc": simulators.run_hqc_sat_simulation,
        "blockchain": simulators.run_blockchain_simulation,
        "ddos": simulators.run_ddos_simulation,
        "spoofing": simulators.run_spoofing_simulation,
        "isms": simulators.run_isms_orchestration_simulation,
    }

    try:
        while True:
            # İstemciden JSON formatında komut bekle
            # Örnek: {"simulation": "jamming", "scenario": "1"}
            data = await websocket.receive_text()
            command = json.loads(data)
            
            sim_function = simulation_router.get(command["simulation"])
            scenario_choice = command["scenario"]

            if sim_function:
                print(f"Başlatılıyor: {command['simulation']} simülasyonu, senaryo {scenario_choice}")
                # İlgili simülasyonu çalıştır ve her adımı anında istemciye gönder
                async for log_item in sim_function(scenario_choice):
                    await websocket.send_json(log_item)
                    await asyncio.sleep(0.02) # Log akışını yavaşlatmak için küçük bir bekleme
                print(f"Tamamlandı: {command['simulation']} simülasyonu.")
            else:
                await websocket.send_json({"type": "fail", "message": f"Hata: Bilinmeyen simülasyon tipi '{command['simulation']}'"})

    except WebSocketDisconnect:
        print("İstemci bağlantıyı kapattı.")
    except Exception as e:
        print(f"WebSocket hatası: {e}")
        await websocket.send_json({"type": "fail", "message": f"Sunucu hatası: {e}"})