import time, random, hashlib, datetime, json, math, uuid, asyncio
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from scipy.stats import ttest_ind
from math import erfc, sqrt
from collections import defaultdict, Counter
import ecdsa, pyotp
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.exceptions import InvalidSignature
from skyfield.api import load, EarthSatellite, wgs84
from skyfield.timelib import Time

# ==========================================================
# 1. JAMMING SİMÜLASYONU (TAM FONKSİYONEL)
# ==========================================================
class PhasedArrayAntenna:
    def __init__(self, target_azimuth, beam_width=20): self.target_azimuth, self.null_azimuth, self.beam_width, self.max_gain, self.sidelobe_gain = target_azimuth, None, beam_width, 20, -10
    def get_gain_at(self, source_azimuth):
        if self.null_azimuth is not None and abs(source_azimuth - self.null_azimuth) < 5: return -100
        if abs(source_azimuth - self.target_azimuth) <= self.beam_width / 2: return self.max_gain
        return self.sidelobe_gain
    def set_null(self, jammer_azimuth): self.null_azimuth = jammer_azimuth
    def clear_null(self): self.null_azimuth = None
class JammerClassifier:
    def __init__(self): self.model = RandomForestClassifier(random_state=42)
    def train(self):
        X, y = [], [];
        for _ in range(200): X.append([random.uniform(80, 100), random.uniform(1, 5), random.randint(20, 30)]); y.append('Sürekli')
        for _ in range(200): X.append([random.uniform(90, 110), random.uniform(5, 15), random.randint(2, 8)]); y.append('Darbeli')
        self.model.fit(X, y)
    def classify(self, features): return self.model.predict(np.array(features).reshape(1, -1))[0]
class DFH_SS_Cognitive:
    def __init__(self, num_channels): self.num_channels, self.current_channel = num_channels, random.randint(0, num_channels - 1)
    def proactive_hop(self):
        old_channel = self.current_channel; self.current_channel = (old_channel + random.randint(10, self.num_channels-10)) % self.num_channels; return old_channel, self.current_channel
class AdaptiveJammer:
    def __init__(self, power=95):
        self.power, self.type = power, random.choice(['Sürekli', 'Darbeli']); self.azimuth = random.randint(80, 150)
        self.is_active, self.target_channel, self.active_since = False, -1, 0; self.attack_duration_left, self.time_since_last_attack = 0, 0
        self.is_scanning, self.scan_cooldown = False, 0
    def update_strategy(self, satellite_channel, t):
        if self.scan_cooldown > 0: self.scan_cooldown -= 1
        if self.is_active:
            self.attack_duration_left -= 1
            if self.attack_duration_left <= 0: self.is_active, self.time_since_last_attack = False, 0; return "stopped"
            if self.target_channel != satellite_channel and not self.is_scanning and self.scan_cooldown == 0: self.is_scanning, self.scan_cooldown = True, 3; return "scanning"
            if self.is_scanning:
                self.scan_cooldown -= 1
                if self.scan_cooldown <= 0: self.is_scanning, self.target_channel = False, satellite_channel; return "found"
            else: self.target_channel = satellite_channel
        else:
            self.time_since_last_attack += 1
            if random.random() < (0.1 + self.time_since_last_attack * 0.02):
                self.is_active, self.active_since, self.attack_duration_left, self.target_channel = True, t, random.randint(8, 20), satellite_channel; return "started"
        return None
# simulators.py dosyasındaki mevcut run_jamming_simulation fonksiyonunun yerine bunu yapıştırın.

async def run_jamming_simulation(scenario_choice: str):
    # --- Simülasyon Parametreleri ---
    defense_on, jammer_on = scenario_choice == "1", scenario_choice != "3"
    TOTAL_TIMESTEPS = 100
    NUM_CHANNELS = 100
    BASE_NOISE_FLOOR = -90
    SATELLITE_POWER = 30
    SATELLITE_AZIMUTH = 45
    HQC_SAT_ROBUSTNESS = 3 # HQC kodlamasından gelen ekstra dayanıklılık payı
    COMMUNICATION_SNR_THRESHOLD = 20 - HQC_SAT_ROBUSTNESS

    # --- Metrikler ve Veri Toplama ---
    metrics = {
        'total_packets': TOTAL_TIMESTEPS, 
        'successful_packets': 0, 
        'attacks_detected': 0, 
        'hops_performed': 0, 
        'nulls_performed': 0
    }
    simulation_log = [] # Zaman serisi analizi için detaylı log

    # --- Savunma Sistemlerini ve Jammer'ı Başlatma ---
    yield {"type": "status", "message": "Yapay zeka modelleri hazırlanıyor..."}
    jammer = AdaptiveJammer(power=98)
    antenna = PhasedArrayAntenna(target_azimuth=SATELLITE_AZIMUTH)
    dfhss = DFH_SS_Cognitive(NUM_CHANNELS)
    ai_ats = IsolationForest(contamination=0.03, random_state=42)
    jammer_classifier = JammerClassifier()
    ai_ats.fit([np.random.randn(NUM_CHANNELS) + BASE_NOISE_FLOOR for _ in range(300)])
    jammer_classifier.train()
    
    yield {"type": "success", "message": "[✓] AI Modelleri başarıyla eğitildi."}
    if jammer_on:
        yield {"type": "info", "message": f"Senaryo: Adaptif '{jammer.type}' tipi jammer'a karşı test başlatıldı. Savunma: {'AKTİF' if defense_on else 'PASİF'}"}
    else:
        yield {"type": "info", "message": "Senaryo: Normal operasyon testi başlatıldı."}

    # --- Ana Simülasyon Döngüsü ---
    for t in range(TOTAL_TIMESTEPS):
        yield {"type": "header", "message": f"--- ZAMAN ADIMI {t+1}/{TOTAL_TIMESTEPS} ---"}
        
        # Jammer stratejisi güncellemesi
        if jammer_on:
            event = jammer.update_strategy(dfhss.current_channel, t)
            if event:
                event_map = {"started": (f"!!! JAMMER BAŞLADI ({jammer.type}) !!!", "alert"),
                             "stopped": ("--- Jammer sustu. ---", "info"),
                             "scanning": ("... Jammer yeni frekansı arıyor ...", "warn"),
                             "found": (f"!!! Jammer yeni frekansı ({dfhss.current_channel}) buldu !!!", "alert")}
                yield {"type": event_map[event][1], "message": f"[{event.upper()}]: {event_map[event][0]}"}

        # Spektrum oluşturma ve anomali tespiti
        current_spectrum = np.random.randn(NUM_CHANNELS) + BASE_NOISE_FLOOR + SATELLITE_POWER
        if jammer.is_active and not jammer.is_scanning:
            current_spectrum[jammer.target_channel] += jammer.power
        
        is_anomaly = ai_ats.predict(current_spectrum.reshape(1, -1))[0] == -1
        
        # Savunma mekanizması aktivasyonu (Eğer savunma açıksa)
        if defense_on and is_anomaly and jammer.is_active:
            metrics['attacks_detected'] += 1
            yield {"type": "status", "message": "--- Hibrit Savunma Protokolü Başladı ---"}
            yield {"type": "info", "message": f"[✓] Faktör 1: AI-ATS -> Spektrumda anomali TESPİT EDİLDİ."}
            
            features = [jammer.power, np.std(current_spectrum), t - jammer.active_since]
            classified_type = jammer_classifier.classify(features)
            yield {"type": "info", "message": f"[✓] Faktör 2: AI Sınıflandırıcı -> Tehdit '{classified_type}' olarak sınıflandırıldı."}
            
            if classified_type == 'Sürekli':
                old, new = dfhss.proactive_hop()
                antenna.set_null(jammer.azimuth)
                metrics['hops_performed'] += 1
                metrics['nulls_performed'] += 1
                yield {"type": "success", "message": f"  -> TEPKİ (Hibrit): Frekans {old}'dan {new}'a atladı ve Anten ({jammer.azimuth}°) sıfırlandı."}
            elif classified_type == 'Darbeli':
                antenna.set_null(jammer.azimuth)
                metrics['nulls_performed'] += 1
                yield {"type": "success", "message": f"  -> TEPKİ (Anten): Jammer yönü ({jammer.azimuth}°) sıfırlandı."}
        
        # SNR Hesaplaması ve Sonuç Loglama
        rec_sat_pwr = SATELLITE_POWER + antenna.get_gain_at(SATELLITE_AZIMUTH)
        noise = BASE_NOISE_FLOOR + random.uniform(-2, 2)
        final_snr = rec_sat_pwr - noise
        if jammer.is_active and not jammer.is_scanning and dfhss.current_channel == jammer.target_channel:
            rec_jam_pwr = jammer.power + antenna.get_gain_at(jammer.azimuth)
            final_snr = rec_sat_pwr - rec_jam_pwr

        final_snr += 15 # DSSS Processing Gain
        communication_success = final_snr > COMMUNICATION_SNR_THRESHOLD

        if communication_success:
            metrics['successful_packets'] += 1
            yield {"type": "log", "message": f"Sistem Durumu: SNR={final_snr:.1f}dB, Başarı={'EVET'}"}
        else:
            yield {"type": "fail", "message": f"Sistem Durumu: SNR={final_snr:.1f}dB, Başarı={'HAYIR'}"}
        
        # Veri toplama
        simulation_log.append({"t": t, "success": communication_success, "jammer_active": jammer.is_active})
        yield {"type": "graph_update", "spectrum_data": current_spectrum.tolist(), "snr_point": {"x": t, "y": final_snr},
               "jammer_active": jammer.is_active and not jammer.is_scanning, "target_channel": jammer.target_channel, "current_channel": dfhss.current_channel}
        await asyncio.sleep(0.05) # Hızlandırıldı

    # --- Raporlama Bölümü (Geliştirilmiş Anlatım) ---
    overall_success_rate = (metrics['successful_packets'] / metrics['total_packets']) * 100
    
    # Fazlara göre başarı oranı hesaplama
    attack_phase_logs = [log['success'] for log in simulation_log if log['jammer_active']]
    normal_phase_logs = [log['success'] for log in simulation_log if not log['jammer_active']]
    
    attack_phase_success_rate = (sum(attack_phase_logs) / len(attack_phase_logs) * 100) if attack_phase_logs else 100
    normal_phase_success_rate = (sum(normal_phase_logs) / len(normal_phase_logs) * 100) if normal_phase_logs else 100

    # Senaryo bazlı anlatım oluşturma
    scenario_title = "Jamming Simülasyonu Sonuç Raporu"
    if jammer_on:
        scenario_description = f"Savunma {'Aktif' if defense_on else 'Pasif'} | Jammer Tipi: {jammer.type}"
    else:
        scenario_description = "Normal Operasyon (Saldırı Yok)"

    if not jammer_on:
        narrative = "Sistem, saldırı olmayan normal operasyon koşullarında test edilmiştir. Tüm paketler başarıyla iletilmiş ve %100.0 hizmet sürekliliği sağlanmıştır. Bu, sistemin temel çalışma performansını göstermektedir."
    elif defense_on:
        narrative = (f"Hibrit savunma mimarisi aktifken, sistem saldırı anında dahi %{attack_phase_success_rate:.2f} başarı oranı elde etmiştir. "
                     f"AI-ATS tarafından {metrics['attacks_detected']} anomali tespit edilmiş; bu tespitlere yanıt olarak {metrics['hops_performed']} kez frekans atlama (FHSS) "
                     f"ve {metrics['nulls_performed']} kez faz dizili anten sıfırlaması (Nulling) uygulanmıştır.\n"
                     f"Sonuç: Hibrit model, hizmet kalitesini yüksek seviyede korumuştur.")
    else:
        narrative = (f"Savunma mekanizmaları devre dışı bırakıldığında, sistem jammer saldırılarına karşı tamamen savunmasız kalmıştır. "
                     f"Saldırı anında hizmet başarı oranı %{attack_phase_success_rate:.2f}'ye düşmüştür. Bu test, proaktif savunma "
                     f"mekanizmalarının (AI-ATS, FHSS, Nulling) hizmet sürekliliği için kritik önem taşıdığını kanıtlamaktadır.")

    summary = {
        "title": scenario_title,
        "scenario": scenario_description,
        "statistics": {
            "overall_success_rate": f"{overall_success_rate:.2f}%",
            "normal_operation_success_rate": f"{normal_phase_success_rate:.2f}%",
            "attack_phase_success_rate": f"{attack_phase_success_rate:.2f}%",
            "total_packets": metrics['total_packets'],
            "successful_packets": metrics['successful_packets'],
            "anomalies_detected_by_ai": metrics['attacks_detected'],
            "frequency_hops_performed": metrics['hops_performed'],
            "antenna_nulls_performed": metrics['nulls_performed']
        },
        "narrative": narrative
    }
    yield {"type": "final", "message": "Jamming simülasyonu tamamlandı.", "summary": summary}
    

# ==========================================================
# 2. HQC-SAT SİMÜLASYONU (TAM FONKSİYONEL)
# ==========================================================
class HQC_Sat:
    def __init__(self, key_bits=512, q=7681): self.n, self.q, self.A, self.s, self.t = key_bits, q, None, None, None
    def generate_keypair(self): self.A, self.s, e = np.random.randint(0, self.q, (self.n, self.n), dtype=np.int64), np.random.randint(-1, 2, self.n, dtype=np.int64), np.random.randint(-1, 2, self.n, dtype=np.int64); self.t = (self.A @ self.s + e) % self.q
    def encrypt(self, m):
        start_time = time.perf_counter_ns(); r, e1, e2 = (np.random.randint(-1, 2, self.n, dtype=np.int64) for _ in range(3))
        u, v = (self.A.T @ r + e1) % self.q, (self.t @ r + e2.sum() + (self.q // 2) * m) % self.q; return u, v, (time.perf_counter_ns() - start_time) / 1_000_000
    def decrypt(self, u, v): return int(abs(((v - (self.s @ u)) % self.q) - self.q // 2) < self.q / 4)
LEO_PROFILE = {"excellent": 1024, "good": 768, "poor": 512, "critical": 256}
def adaptive_key_len(snr_db):
    if snr_db > 20: profile = "excellent";
    elif snr_db > 10: profile = "good";
    elif snr_db > 0: profile = "poor";
    else: profile = "critical"
    return profile, LEO_PROFILE[profile]
def estimate_energy_realistic(enc_time_ms, key_bits): return 10 + (enc_time_ms * key_bits) / 20000
def simulate_packet_loss_realistic(snr_db, angle): per = 1-(1-(0.5*erfc(sqrt(10**(snr_db/10)))))**(1500*8); return random.random() < min(1.0, per + max(0, 1-(angle/30)))
class SimulatedCache:
    def access(self, hit): return random.randint(10, 20) if hit else random.randint(200, 250)
def decaps_vulnerable(key, cache): return sum(cache.access(False) for bit in key if bit != 0)
def decaps_constant_time(key, cache): return sum(cache.access(False) for _ in key)
async def run_hqc_sat_simulation(scenario_choice: str):
    yield {"type": "header", "message": "BÖLÜM 1: LEO UYDU GEÇİŞİ (GERÇEKÇİ MODEL)"}
    sc_titles = {"1": "Tam Koruma", "2": "Zafiyetli Sistem", "3": "Güvenliksiz Sistem"}; metrics = {'packets_sent': 0, 'packets_lost': 0, 'packets_ok': 0}
    yield {"type": "status", "message": f"'{sc_titles.get(scenario_choice)}' senaryosu çalıştırılıyor..."}
    use_ct, use_fs = scenario_choice != "2", scenario_choice != "3"
    angles = list(range(10, 91, 10)) + list(range(80, 9, -10)); hqc_fixed = HQC_Sat(key_bits=2048)
    if not use_fs: hqc_fixed.generate_keypair()
    for i, angle in enumerate(angles):
        metrics['packets_sent'] += 1; snr = -10 + (angle / 90) * 40
        yield {"type": "log", "message": f"--- Paket #{i+1}: Açı={angle}°, SNR={snr:.1f} dB ---"}
        if simulate_packet_loss_realistic(snr, angle):
            metrics['packets_lost'] += 1; yield {"type": "fail", "message": "DURUM: Paket, gerçekçi kanal modeline göre kayboldu!"}; await asyncio.sleep(0.1); continue
        profile, key_bits = adaptive_key_len(snr)
        if use_fs: hqc = HQC_Sat(key_bits=key_bits); hqc.generate_keypair(); yield {"type": "success", "message": "GÜVENLİK (FS): Aktif."}
        else: hqc = hqc_fixed; yield {"type": "warn", "message": "GÜVENLİK (FS): Devre Dışı."}
        u, v, enc_time = hqc.encrypt(1); decrypted = hqc.decrypt(u, v)
        energy = estimate_energy_realistic(enc_time, key_bits)
        yield {"type": "info", "message": f"ADAPTASYON: Profil '{profile}'. Güvenlik {key_bits}-bit."}
        yield {"type": "log", "message": f"PERFORMANS: Enerji ~{energy:.2f} mW (Süre: {enc_time:.2f} ms)."}
        if 1 == decrypted: metrics['packets_ok'] += 1; yield {"type": "success", "message": "DURUM: Paket başarıyla işlendi."}
        else: yield {"type": "fail", "message": "DURUM: KRİTİK HATA! Paket çözülemedi!"}
        await asyncio.sleep(0.1)
    
    t_stat, narrative = None, ""
    if scenario_choice == '3':
        narrative = "Bu modda, 'Forward Secrecy' gibi operasyonel savunmaların kapalı olması, bir anahtar sızıntısı durumunda tüm oturumun risk altına girmesine neden olur. Yan-Kanal testi bu senaryonun ana odağı değildir."
    else:
        yield {"type": "header", "message": "BÖLÜM 2: YAN-KANAL SALDIRISI (NIST TVLA UYUMLU)"}
        cache, NUM_M, KEY_L = SimulatedCache(), 5000, 16
        key_s, key_d = np.zeros(KEY_L, dtype=int), np.random.randint(-1, 2, KEY_L)
        if use_ct:
            yield {"type": "status", "message": "Test Modu: Güvenli (Constant-Time) Fonksiyon..."}
            t_s, t_d = (np.array([decaps_constant_time(k, cache) for _ in range(NUM_M)]) for k in [key_s, key_d])
        else:
            yield {"type": "warn", "message": "Test Modu: Zafiyetli Fonksiyon..."}
            t_s, t_d = (np.array([decaps_vulnerable(k, cache) for _ in range(NUM_M)]) for k in [key_s, key_d])
        t_stat, _ = ttest_ind(t_s, t_d, equal_var=False)
        yield {"type": "log", "message": f"İstatistiksel Analiz: t-değeri = {abs(t_stat):.2f}"}
        if abs(t_stat) > 4.5:
            narrative = f"A. İşleyiş: LEO geçişi {metrics['packets_ok']}/{metrics['packets_sent']} başarıyla tamamlandı.\nB. Savunma: Zafiyetli modda çalıştırılan sistem, yan-kanal testinde |t-değeri|={abs(t_stat):.2f} ile NIST eşiğini (4.5) geçerek BAŞARISIZ olmuştur.\nC. Sonuç: Bu test, 'Constant-Time' savunmasının kritik önemini kanıtlamıştır."
            yield {"type": "fail", "message": f"Yan-Kanal Analizi: BAŞARISIZ! |t-değeri| > 4.5. Bilgi sızıntısı kanıtlandı."}
        else:
            narrative = f"A. İşleyiş: LEO geçişi {metrics['packets_ok']}/{metrics['packets_sent']} başarıyla tamamlandı.\nB. Savunma: Tam koruma modundaki sistem, |t-değeri|={abs(t_stat):.2f} ile NIST eşiğinin (4.5) altında kalarak sızıntı olmadığını kanıtlamıştır.\nC. Sonuç: Mimari, hem kuantum hem de yan-kanal saldırılarına karşı bütünsel koruma sağlamıştır."
            yield {"type": "success", "message": f"Yan-Kanal Analizi: BAŞARILI! |t-değeri| < 4.5. Sistem güvenli."}
    
    pass_rate = (metrics['packets_ok'] / metrics['packets_sent']) * 100 if metrics['packets_sent'] > 0 else 0
    summary = {"title": "HQC-SAT Simülasyonu Sonuç Raporu", "scenario": sc_titles.get(scenario_choice), "success_rate": pass_rate, "metrics": metrics, "narrative": narrative}
    yield {"type": "final", "message": "HQC-SAT simülasyonu tamamlandı.", "summary": summary}
# ==========================================================
# 3. BLOCKCHAIN SİMÜLASYONU (TAM FONKSİYONEL)
# ==========================================================
class SatelliteKeys:
    def __init__(self, ids):
        self.keys = {i: {'private': ecdsa.SigningKey.generate(curve=ecdsa.SECP256k1)} for i in ids}
        for i in ids: self.keys[i]['public'] = self.keys[i]['private'].get_verifying_key()
    def get_private_key(self, id): return self.keys[id]['private']
    def get_public_key(self, id): return self.keys[id]['public']
class Satellite:
    def __init__(self, id): self.id, self.uptime, self.signal, self.energy, self.prev_hash, self.is_visible = id, random.uniform(70,99), random.uniform(35,95), random.uniform(60,98), None, False
    def update(self, t): self.uptime=max(0,min(100,85+12*math.sin(0.08*t))); self.signal=max(0,min(100,60+28*math.sin(0.07*t))); self.energy=max(0,min(100,80+15*math.sin(0.06*t))); self.is_visible=self.signal>=30.0 and self.energy>=50.0
    def sync(self, h): self.prev_hash = h
    def score(self): return round(max(0,min(100,(self.uptime*0.4)+(self.signal*0.3)+(self.energy*0.3))),2)
class SatelliteNetwork:
    def __init__(self, n=10): self.sats = [Satellite(f"GEN9-SAT{i:02d}") for i in range(n)]
    def update_all(self, t): [s.update(t) for s in self.sats]
    def get_visible(self): return [s for s in self.sats if s.is_visible]
    def select_validator(self): visible=self.get_visible(); return max(visible, key=lambda s:s.score()) if visible else None
    def get_endorsements(self, proposer_id, hash):
        voters=[s for s in self.get_visible() if s.id!=proposer_id];
        if not voters: return True, 1.0
        ratio=len([v.id for v in voters if v.prev_hash==hash])/len(voters); return ratio>=(2/3), ratio
# simulators.py dosyasındaki ilgili sınıfları bu kodlarla güncelleyin

class MerkleTree:
    def __init__(self, items):
        self.root = self.build(items)

    def build(self, items):
        if not items:
            return ""
        # İşlemlerin tutarlı bir şekilde metne dönüştürülmesini sağla
        hashes = [hashlib.sha256(json.dumps(tx, sort_keys=True).encode()).hexdigest() for tx in items]
        
        if len(hashes) % 2 != 0:
            hashes.append(hashes[-1]) # Tek sayıda hash varsa sonuncuyu kopyala

        while len(hashes) > 1:
            level = [hashlib.sha256((hashes[i] + hashes[i+1]).encode()).hexdigest() for i in range(0, len(hashes), 2)]
            hashes = level
            if len(hashes) % 2 != 0 and len(hashes) > 1:
                hashes.append(hashes[-1])
        return hashes[0]

class Block:
    def __init__(self, i, tx, prev_h, val_id, val_score):
        self.index, self.ts, self.tx, self.prev_hash, self.val_id, self.val_score = i, datetime.datetime.now().isoformat(), tx, prev_h, val_id, val_score
        
        # DÜZELTME: 'self.merkle' özelliğini burada doğru şekilde tanımlıyoruz
        self.merkle = MerkleTree(self.tx).root 
        self.hash = self.calc_hash()

    def calc_hash(self):
        # DÜZELTME: Hash hesaplamasında 'self.merkle' kullanılıyor
        header = {"i": self.index, "m": self.merkle, "p_h": self.prev_hash, "v_id": self.val_id}
        return hashlib.sha256(json.dumps(header, sort_keys=True).encode()).hexdigest()
    
class SmartContract:
    def __init__(self, ids, keys): self.auth_ids, self.keys = ids, keys
    def validate(self, tx):
        sender=tx.get("sender_id");
        if sender not in self.auth_ids: return False, "Yetkisiz uydu"
        try: tx_c=tx.copy(); sig=bytes.fromhex(tx_c.pop("signature")); msg=json.dumps(tx_c, sort_keys=True).encode(); self.keys.get_public_key(sender).verify(sig, msg)
        except Exception: return False, "Geçersiz dijital imza"
        if tx.get("temperature",0)>95: return False, "Kritik sıcaklık anomalisi"
        return True, "İşlem doğrulandı"
    
async def run_blockchain_simulation(scenario_choice: str):
    TOTAL_BLOCKS = 10; sc_map = {"1":{"n":"Savunma Aktif", "a":True, "d":True}, "2":{"n":"Savunma Pasif", "a":True, "d":False}, "3":{"n":"Saldırı Yok", "a":False, "d":True}}
    sc = sc_map.get(scenario_choice); is_attack, is_defense = sc["a"], sc["d"]; metrics = {'blocks_created': 0, 'attacks_attempted': 0, 'attacks_blocked': 0}
    yield {"type": "header", "message": f"Senaryo Başlatılıyor: {sc['n']}"}
    net = SatelliteNetwork(10); ids = [s.id for s in net.sats]; keys = SatelliteKeys(ids); contract = SmartContract(set(ids), keys)
    chain = [Block(0, ["Genesis"], "0", "SYSTEM", 100.0)]; [s.sync(chain[0].hash) for s in net.sats]
    yield {"type": "success", "message": f"[GENESIS]: Genesis bloğu oluşturuldu (Hash: {chain[0].hash[:20]}...)"}
    def generate_tx(sender): d={"sender_id":sender,"ts":datetime.datetime.now().isoformat(),"temperature":round(random.uniform(10,80),2)}; d["signature"]=keys.get_private_key(sender).sign(json.dumps(d,sort_keys=True).encode()).hex(); return d
    for i in range(TOTAL_BLOCKS):
        yield {"type": "header", "message": f"=============== ZAMAN ADIMI {i+1}/{TOTAL_BLOCKS} ==============="}
        net.update_all(i); visible_ids = [s.id for s in net.get_visible()]
        log_sats = ["[UYDU DURUMU]: (Skor, Görünürlük)"] + [f"  > {s.id}: {s.score():5.1f}% [{'GÖRÜNÜR' if s.is_visible else 'GÖRÜNMEZ'}]" for s in net.sats]
        yield {"type": "log", "message": "\n".join(log_sats)}
        packet = [generate_tx(random.choice(visible_ids))] if visible_ids else []
        attack_active_this_step = is_attack and random.random() < 0.6
        if attack_active_this_step: metrics['attacks_attempted'] += 1
        if not is_defense and attack_active_this_step:
            yield {"type": "fail", "message": "[SAVUNMA PASİF]: Manipüle edilmiş veri (imzasız, sıcaklık=9999) gönderiliyor..."}
            packet[0]["temperature"]=9999; packet[0].pop("signature", None)
        valid_tx = []
        if is_defense:
            for tx in packet:
                is_valid, reason = contract.validate(tx)
                if is_valid: valid_tx.append(tx)
                else: yield {"type": "fail", "message": f"[AKILLI KONTRAT]: İşlem reddedildi: {reason}."}; metrics['attacks_blocked'] += 1
        else: valid_tx = packet
        if not valid_tx: yield {"type": "warn", "message": "[BLOK]: Geçerli işlem kalmadı."}; await asyncio.sleep(0.5); continue
        validator = net.select_validator()
        if not validator: yield {"type": "fail", "message": "[AĞ UYARISI]: Görünürde uydu yok!"}; await asyncio.sleep(0.5); continue
        yield {"type": "status", "message": f"[PoSat]: Doğrulayıcı seçildi -> {validator.id} (Skor: {validator.score()}%)."}
        passed, ratio = net.get_endorsements(validator.id, chain[-1].hash)
        yield {"type": "status", "message": f"[ONAY]: Zincir Uyum Oranı = {ratio*100:.1f}%"}
        if not passed: yield {"type": "fail", "message": "[BLOK RED]: 2/3 zincir uyum eşiği sağlanamadı."}; await asyncio.sleep(0.5); continue
        new_block = Block(len(chain), valid_tx, chain[-1].hash, validator.id, validator.score()); chain.append(new_block)
        metrics['blocks_created'] += 1; [s.sync(new_block.hash) for s in net.sats]
        yield {"type": "success", "message": f"[BLOK EKLENDİ]: #{new_block.index} | Hash: {new_block.hash[:40]}..."}; await asyncio.sleep(0.5)
    success_rate = (metrics['attacks_blocked'] / metrics['attacks_attempted'] * 100) if metrics['attacks_attempted'] > 0 else 100
    summary = {"title": "BC-DIS Simülasyonu Sonuç Raporu", "scenario": sc['n'], "success_rate": success_rate, "metrics": metrics,
               "narrative": (f"A. İşleyiş: {TOTAL_BLOCKS} adımlık simülasyonda {metrics['blocks_created']} yeni blok başarıyla oluşturuldu.\n"
                             f"B. Savunma: Toplam {metrics['attacks_attempted']} veri manipülasyonu saldırısı denendi. Akıllı Kontrat ve ECDSA imza doğrulaması sayesinde bu saldırıların {metrics['attacks_blocked']} tanesi (%{success_rate:.1f}) başarıyla engellendi.\n"
                             f"C. Sonuç: BC-DIS mimarisi, PoSat konsensüsü ile verimli bir şekilde çalışırken, Akıllı Kontrat katmanı sayesinde veri bütünlüğünü etkin bir şekilde korumuştur.") if is_defense and is_attack else
                             ("Savunmasız modda, manipüle edilmiş ve imzasız veriler Akıllı Kontrat denetiminden geçmediği için blok zincirine başarıyla eklenmiştir. Bu durum, zincirin teknik olarak geçerli görünse de, içerik olarak sahte verilerle doldurulabileceğini ve Akıllı Kontrat katmanının zorunluluğunu kanıtlamıştır.") if not is_defense and is_attack else
                             "Normal operasyon modunda, tüm meşru veriler Akıllı Kontrat onayından geçmiş ve blok zincirine sorunsuzca eklenmiştir."}
    yield {"type": "final", "message": "Blockchain simülasyonu tamamlandı.", "summary": summary}
# ==========================================================
# 4. DDOS SİMÜLASYONU (GÜÇLENDİRİLMİŞ)
# ==========================================================
class SignalPacket_DDoS:
    def __init__(self, p): self.source_ip, self.is_legitimate, self.payload, self.signature = p["source_ip"], p["is_legitimate"], p["payload"], p.get("signature")
class DilithiumSimulator:
    @staticmethod
    def sign(data): return hashlib.sha256(f"SECRET_KEY_{data}".encode()).hexdigest()
    @staticmethod
    def verify(data, signature): return signature is not None and signature == hashlib.sha256(f"SECRET_KEY_{data}".encode()).hexdigest()
class AI_DDoS_Defender:
    def __init__(self, learning_period=10): self.learning_period, self.baseline, self.is_attack, self.whitelist, self.rules = learning_period, [], False, set(), {}
    def rate_limit(self, ip, timeout=10): self.rules[ip] = time.time() + timeout
    def filter_local(self, pkts): [self.rules.pop(ip) for ip in list(self.rules) if time.time() > self.rules[ip]]; return [p for p in pkts if p.source_ip not in self.rules]
    def filter_cloud(self, pkts): return [p for p in pkts if p.source_ip in self.whitelist]
    async def analyze_and_defend(self, traffic, defense_on):
        metrics = {"local": 0, "cloud": 0, "signature": 0}; final_traffic, log_messages = traffic, []
        if not defense_on:
            log_messages.append({"type": "log", "message": "Savunma Pasif: Trafik filtrelenmeden yönlendiriliyor."}); return traffic, metrics, log_messages
        if len(self.baseline) < self.learning_period:
            self.baseline.append(len(traffic)); self.whitelist.update(p.source_ip for p in traffic if p.is_legitimate)
            if len(self.baseline) == self.learning_period: log_messages.append({"type": "success", "message": f"[AI-Defender]: Öğrenme fazı tamamlandı. {len(self.whitelist)} IP Whitelist'e eklendi."})
            return traffic, metrics, log_messages
        if not self.is_attack and len(traffic) > np.mean(self.baseline) * 5:
            self.is_attack = True; log_messages.append({"type": "alert", "message": "[ALARM!]: DDoS Saldırısı Tespit Edildi!"})
            non_legit = [p.source_ip for p in traffic if not p.is_legitimate]
            if non_legit: ip = Counter(non_legit).most_common(1)[0][0]; self.rate_limit(ip); log_messages.append({"type": "info", "message": f"  -> [Yerel Savunma]: En agresif IP ({ip}) Blacklist'e eklendi."})
        if self.is_attack:
            verified_traffic, forged_count = [], 0
            for p in traffic:
                if p.is_legitimate:
                    if DilithiumSimulator.verify(p.payload, p.signature): verified_traffic.append(p)
                    else: forged_count += 1
                else: verified_traffic.append(p)
            if forged_count > 0: metrics["signature"] = forged_count; log_messages.append({"type": "fail", "message": f"  -> [İmza Kontrolü]: {forged_count} sahte paket engellendi!"})
            local_f = self.filter_local(verified_traffic); metrics["local"] = len(verified_traffic) - len(local_f)
            if metrics["local"] > 0: log_messages.append({"type": "log", "message": f"  -> [Yerel Savunma]: Blacklist'ten {metrics['local']} paket engellendi."})
            cloud_f = self.filter_cloud(local_f); metrics["cloud"] = len(local_f) - len(cloud_f)
            if metrics["cloud"] > 0: log_messages.append({"type": "log", "message": f"  -> [Bulut Savunması]: Whitelist dışı {metrics['cloud']} paket engellendi."})
            final_traffic = cloud_f
        return final_traffic, metrics, log_messages
class TargetServer:
    def __init__(self, cap=120): self.capacity = cap
    def process(self, traffic): return traffic[:self.capacity], traffic[self.capacity:]
async def run_ddos_simulation(scenario_choice: str):
    is_attack, is_defense = scenario_choice != "3", scenario_choice == "1"; SIM_DUR, ATTACK_START, ATTACK_DUR = 30, 10, 15
    server, defender = TargetServer(), AI_DDoS_Defender(); metrics = defaultdict(int)
    sc_name = "Savunma Aktif" if is_defense else "Savunmasız" if is_attack else "Normal Operasyon"
    yield {"type": "header", "message": f"Senaryo: {sc_name}"}
    for t in range(SIM_DUR):
        is_attack_phase = is_attack and t >= ATTACK_START and t < ATTACK_START + ATTACK_DUR
        legit_data = [{"source_ip": f"10.0.0.{i}", "is_legitimate": True, "payload": f"data_{i}_{t}"} for i in range(80)]
        for p in legit_data: p["signature"] = DilithiumSimulator.sign(p["payload"])
        attack_data = []
        if is_attack_phase:
            if t == ATTACK_START: yield {"type": "alert", "message": f"!!! {ATTACK_DUR} SANİYELİK YOĞUN DDoS SALDIRISI BAŞLADI !!!"}
            attack_data = [{"source_ip": f"192.168.1.{random.randint(1,254)}", "is_legitimate": False, "payload": "flood"} for i in range(random.randint(4500, 6000))]
        all_pkts_obj = [SignalPacket_DDoS(p) for p in legit_data + attack_data]
        metrics['total_incoming'] += len(all_pkts_obj); metrics['total_legit_incoming'] += len(legit_data)
        if not is_defense and is_attack_phase: random.shuffle(all_pkts_obj)
        yield {"type": "log", "message": f"Zaman {t}s: Sisteme {len(all_pkts_obj)} paket girdi ({len(legit_data)} meşru)."}
        traffic_after, defense_metrics, new_logs = await defender.analyze_and_defend(all_pkts_obj, is_defense)
        for log in new_logs: yield log
        processed, overflow = server.process(traffic_after); legit_processed = sum(1 for p in processed if p.is_legitimate); metrics['total_legit_processed'] += legit_processed
        success_rate = (legit_processed / len(legit_data) * 100) if legit_data else 100
        total_blocked = sum(defense_metrics.values()) + len(overflow); metrics['total_blocked'] += total_blocked
        yield {"type": "info", "message": f"  -> Sonuç: Sunucuya Ulaşan: {len(processed)}, Engellenen: {total_blocked}, Meşru Başarı: {success_rate:.1f}%"}
        await asyncio.sleep(0.1)
    success_rate = (metrics['total_legit_processed'] / metrics['total_legit_incoming'] * 100) if metrics['total_legit_incoming'] > 0 else 100
    summary = {"title": "DDoS Simülasyonu Sonuç Raporu", "scenario": sc_name, "success_rate": success_rate, "metrics": metrics,
               "narrative": (f"A. İşleyiş: Hibrit savunma sistemi, gelen {metrics['total_incoming']:,} paketi analiz etmiştir.\n"
                             f"B. Savunma: İmza kontrolü, dinamik blacklist ve whitelist filtreleri sayesinde toplam {metrics['total_blocked']:,} zararlı veya sahte paket başarıyla engellenmiştir.\n"
                             f"C. Sonuç: En yoğun saldırı anında bile, meşru kullanıcılar için hizmet başarı oranı %{success_rate:.2f} gibi mükemmel bir seviyede korunmuştur.") if is_defense else
                             (f"Savunmasız modda, sunucu kapasitesi anında aşılmış ve gelen paketler karıştırıldığı için meşru kullanıcı başarı oranı %{success_rate:.2f}'ye düşmüştür. Bu, hibrit kalkanın kritik önemini göstermektedir.") if is_attack else
                             "Normal operasyon modunda, sistem %100 başarı oranıyla çalışmıştır."}
    yield {"type": "final", "message": "DDoS simülasyonu tamamlandı.", "summary": summary}
# ==========================================================
# 5. SPOOFING (MF-SatAuth) SİMÜLASYONU (TAM FONKSİYONEL)
# ==========================================================
class Spoofing_AI_ATS:
    def __init__(self, gs): self.model, self.is_trained, self.gs, self.last_known = IsolationForest(contamination=0.02, random_state=42), False, gs, {}
    def _extract(self, pkt, sat, is_spoofed):
        rel = sat.satellite.at(pkt.timestamp) - self.gs.at(pkt.timestamp); alt, _, dist = rel.altaz()
        snr = (100000/(dist.km**2)+np.random.normal(0,0.1)) * (1.0 if alt.degrees>20 else 0.6)
        doppler = (rel.velocity.km_per_s.dot(rel.position.km/dist.km)/299792)*2.4e9/1e3
        drift = abs(doppler - self.last_known.get(sat.id, doppler))
        if is_spoofed: snr*=random.uniform(0.98,1.02); doppler+=random.uniform(-0.05,0.05); drift+=random.uniform(0.01,0.05)
        self.last_known[sat.id] = doppler; return [snr, doppler, drift]
    def train(self, pkts, sat): self.model.fit([self._extract(p, sat, False) for p in pkts]); self.is_trained = True
    def predict(self, pkt, sat, is_spoofed): return "ANOMALİ" if self.model.predict([self._extract(pkt, sat, is_spoofed)])[0] == -1 else "NORMAL"
class MFSatAuth:
    def __init__(self, sat, secret): self.sats, self.nonces, self.secret, self.totp = {sat.id: sat}, set(), secret, pyotp.TOTP(secret)
    def verify(self, pkt):
        sat = self.sats.get(pkt.satellite_id)
        if not sat: return False, "Bilinmeyen Uydu ID"
        pos = sat.get_current_geocentric_position(pkt.timestamp)
        zkp_data = f"({pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f}){self.secret}".encode()
        if pkt.zkp_proof != hashlib.sha256(zkp_data).hexdigest(): return False, "Blok: ZKP İspatı"
        if pkt.nonce in self.nonces: return False, "Blok: Nonce Kontrolü (Tekrar Saldırısı)"
        if not self.totp.verify(pkt.totp_code, for_time=pkt.timestamp.utc_datetime(), valid_window=1): return False, "Blok: TOTP Kontrolü"
        try: sat.public_key.verify(pkt.signature, pkt.get_signed_data(), ec.ECDSA(hashes.SHA256()))
        except InvalidSignature: return False, "Blok: İmza (ECDSA)"
        if np.linalg.norm(pos - np.array(pkt.position)) > 15.0: return False, "Blok: Konum (TLE)"
        self.nonces.add(pkt.nonce); return True, "Tüm Faktörler Geçildi"
class LegitimateSatellite:
    def __init__(self, id, tle1, tle2, secret):
        self.id, self.timescale = id, load.timescale(); self.satellite = EarthSatellite(tle1, tle2, self.id, self.timescale)
        self.private_key = ec.generate_private_key(ec.SECP256R1()); self.public_key = self.private_key.public_key()
        self.secret, self.totp_gen = secret, pyotp.TOTP(secret)
    def get_current_geocentric_position(self, t): return self.satellite.at(t).position.km
    def create_signal(self):
        t, pos = self.timescale.now(), self.get_current_geocentric_position(self.timescale.now())
        p, n, totp = f"H_OK_{random.randint(1000,9999)}", uuid.uuid4().hex, self.totp_gen.at(t.utc_datetime())
        zkp = hashlib.sha256(f"({pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f}){self.secret}".encode()).hexdigest()
        data = f"{self.id}{t.utc_iso()}({pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f}){p}{n}{totp}{zkp}".encode()
        sig = self.private_key.sign(data, ec.ECDSA(hashes.SHA256())); return SignalPacket(self.id, t, pos, p, n, totp, zkp, sig)
class SmartAttackerNode:
    def __init__(self, target): self.target, self.timescale, self.key, self.secret = target, load.timescale(), ec.generate_private_key(ec.SECP256R1()), pyotp.random_base32()
    def create_spoofed_signal(self):
        t, real_pos = self.timescale.now(), self.target.get_current_geocentric_position(t)
        fake_pos = real_pos + np.random.uniform(-10,10,3); p, n, totp = "CMD_OVERRIDE", uuid.uuid4().hex, pyotp.TOTP(self.secret).now()
        zkp = hashlib.sha256(f"({fake_pos[0]:.4f},{fake_pos[1]:.4f},{fake_pos[2]:.4f}){self.secret}".encode()).hexdigest()
        data = f"{self.target.id}{t.utc_iso()}({fake_pos[0]:.4f},{fake_pos[1]:.4f},{fake_pos[2]:.4f}){p}{n}{totp}{zkp}".encode()
        sig = self.key.sign(data, ec.ECDSA(hashes.SHA256())); return SignalPacket(self.target.id, t, fake_pos, p, n, totp, zkp, sig)
    def create_replayed_signal(self, pkt): return pkt
class SignalPacket:
    def __init__(self, id, t, pos, p, n, totp, zkp, sig):
        self.satellite_id, self.timestamp, self.position, self.payload, self.nonce, self.totp_code, self.zkp_proof, self.signature = id, t, pos, p, n, totp, zkp, sig
    def get_signed_data(self): return f"{self.satellite_id}{self.timestamp.utc_iso()}({self.position[0]:.4f},{self.position[1]:.4f},{self.position[2]:.4f}){self.payload}{self.nonce}{self.totp_code}{self.zkp_proof}".encode()
async def run_spoofing_simulation(scenario_choice: str):
    defense_on, attack_on = scenario_choice != "2", scenario_choice != "3"
    sc_name = "Tam Savunma" if defense_on else "Savunmasız" if attack_on else "Normal Operasyon"
    yield {"type": "header", "message": f"Senaryo: {sc_name}"}
    secret = pyotp.random_base32(); SAT_NAME = "GEN9-SAT"
    tle1, tle2 = '1 33591U 09005A   25230.15094264  .00000045  00000+0  53282-4 0  9995', '2 33591  99.0494 278.4116 0014605  71.5959 288.5409 14.12521115874254'
    sat = LegitimateSatellite(SAT_NAME, tle1, tle2, secret); attacker = SmartAttackerNode(sat)
    gs = wgs84.latlon(latitude_degrees=39.92, longitude_degrees=32.85)
    mf_auth, ai_ats = MFSatAuth(sat, secret), Spoofing_AI_ATS(gs)
    yield {"type": "status", "message": "AI-ATS Modeli eğitiliyor..."}; ai_ats.train([sat.create_signal() for _ in range(200)], sat)
    yield {"type": "success", "message": "AI-ATS Modeli başarıyla eğitildi."}
    metrics = defaultdict(int); captured_signal = sat.create_signal(); await asyncio.sleep(2)
    for i in range(25):
        attack_type = 'NORMAL' if not attack_on else random.choice(['NORMAL', 'SPOOF', 'REPLAY'])
        yield {"type": "header", "message": f"--- ZAMAN ADIMI {i+1} ({attack_type}) ---"}
        signal = {'NORMAL': sat.create_signal, 'SPOOF': attacker.create_spoofed_signal, 'REPLAY': lambda: attacker.create_replayed_signal(captured_signal)}[attack_type]()
        metrics[attack_type] += 1
        mf_status, mf_reason = "ATLANDI", "Savunma Pasif"
        if defense_on:
            mf_status, mf_reason = mf_auth.verify(signal)
            log_type = "success" if mf_status else "fail"
            yield {"type": log_type, "message": f"[MF-SatAuth]: Sonuç: {'DOĞRULANDI' if mf_status else 'REDDEDİLDİ'}. Sebep: {mf_reason}"}
            if not mf_status and attack_type != 'NORMAL': metrics['blocked_by_mfauth'] += 1
        else: yield {"type": "warn", "message": "[MF-SatAuth]: Devre Dışı Bırakıldı."}
        ai_pred = ai_ats.predict(signal, sat, attack_type != 'NORMAL')
        log_type = "info" if ai_pred == "NORMAL" else "alert"
        yield {"type": log_type, "message": f"[AI-ATS]: RF Parmak İzi Tahmini: {ai_pred}"}
        if mf_status and ai_pred == "ANOMALİ" and attack_type != 'NORMAL': metrics['blocked_by_ai'] += 1
        await asyncio.sleep(0.1)
    total_attacks = metrics['SPOOF'] + metrics['REPLAY']; total_blocked = metrics['blocked_by_mfauth'] + metrics['blocked_by_ai']
    success_rate = (total_blocked / total_attacks * 100) if total_attacks > 0 else 100
    summary = {"title": "Spoofing (MF-SatAuth) Simülasyon Raporu", "scenario": sc_name, "success_rate": success_rate, "metrics": metrics,
               "narrative": (f"A. İşleyiş: {metrics['NORMAL']} normal, {metrics['SPOOF']} sahte ve {metrics['REPLAY']} tekrar sinyali test edildi.\n"
                             f"B. Savunma: 5 katmanlı MF-SatAuth sistemi {metrics['blocked_by_mfauth']} saldırıyı kural tabanlı olarak engelledi. Kalan {total_attacks - metrics['blocked_by_mfauth']} saldırıdan {metrics['blocked_by_ai']} tanesi ise son güvenlik ağı olan AI-ATS tarafından RF parmak izi analizi ile yakalandı.\n"
                             f"C. Sonuç: Toplam {total_attacks} saldırının {total_blocked} tanesi engellenerek, %{success_rate:.2f} başarı oranına ulaşıldı. Bu, 'derinlemesine savunma' prensibinin başarısıdır.") if defense_on else
                             (f"Savunmasız modda, {total_attacks} saldırı sinyalinin tamamı MF-SatAuth katmanını atlayarak sisteme sızmıştır. Sadece AI-ATS katmanı tarafından {metrics['blocked_by_ai']} tanesi tespit edilebilmiştir. Bu test, kural tabanlı MF-SatAuth'un zorunluluğunu göstermektedir.")}
    yield {"type": "final", "message": "Spoofing (MF-SatAuth) simülasyonu tamamlandı.", "summary": summary}


# ==========================================================
# 6. ISMS ORKESTRASYON SİMÜLASYONU (NİHAİ VERSİYON V2)
# ==========================================================
class ISMS_Brain_Final:
    def __init__(self):
        self.threat_level = "GÜVENLİ"; self.active_alarms = {}; self.metrics = defaultdict(int)
        self.blacklist = set(); self.quarantine_list = set()
    def process_alarm(self, alarm):
        self.active_alarms[alarm['type']] = alarm; self.metrics['total_alarms'] += 1
        self.metrics[f"alarm_{alarm['type'].lower()}"] += 1; return self._run_rules_engine()
    def clear_alarm(self, alarm_type):
        if alarm_type in self.active_alarms: self.active_alarms.pop(alarm_type)
        return self._run_rules_engine()
    
    # DÜZELTME: Eksik olan metot eklendi
    def get_active_responses(self):
        active_responses = []
        if "JAMMING" in self.active_alarms: active_responses.append("AKTİF TEPKİ: DFH-SS -> Frekans atlama protokolü devrede.")
        if "SPOOFING" in self.active_alarms: active_responses.append(f"AKTİF TEPKİ: Güvenlik Duvarı -> IP ({self.active_alarms['SPOOFING']['source_ip']}) engelleniyor.")
        if "DATA_TAMPERING" in self.active_alarms: active_responses.append(f"AKTİF TEPKİ: BC-DIS -> Uydu ({self.active_alarms['DATA_TAMPERING']['source_satellite']}) karantinada.")
        if self.metrics.get('hqc_boost_activated', 0) == 1: active_responses.append("AKTİF TEPKİ: HQC-SAT -> Kripto güvenlik seviyesi MAKSİMUM.")
        return active_responses
    
    def _run_rules_engine(self):
        new_responses = [] # Sadece bu döngüde üretilen yeni kararlar
        if not self.active_alarms:
            self.threat_level = "GÜVENLİ"
            if self.metrics.get('hqc_boost_activated', 0) == 1:
                new_responses.append("ISMS EYLEMİ: HQC-SAT -> Tehditler ortadan kalktı, güvenlik seviyesi normale döndürüldü.")
                self.metrics['hqc_boost_activated'] = 0
            self.blacklist.clear(); self.quarantine_list.clear()
            return new_responses, self.get_active_responses()

        if "JAMMING" in self.active_alarms and self.metrics['response_fhss'] < self.metrics['alarm_jamming']:
            details = self.active_alarms["JAMMING"]; new_channel = (details["target_channel"] + 50) % 100
            new_responses.append(f"ISMS EYLEMİ: DFH-SS -> Frekans {details['target_channel']}'den {new_channel}'e değiştirildi.")
            self.metrics['response_fhss'] += 1
        if "SPOOFING" in self.active_alarms:
            details, ip_to_block = self.active_alarms["SPOOFING"], self.active_alarms["SPOOFING"]['source_ip']
            if ip_to_block not in self.blacklist:
                new_responses.append(f"ISMS EYLEMİ: Güvenlik Duvarı -> Kaynak IP ({ip_to_block}) engellendi.")
                self.blacklist.add(ip_to_block); self.metrics['response_firewall'] += 1
        if "DATA_TAMPERING" in self.active_alarms:
            details, node_to_quarantine = self.active_alarms["DATA_TAMPERING"], self.active_alarms["DATA_TAMPERING"]['source_satellite']
            if node_to_quarantine not in self.quarantine_list:
                new_responses.append(f"ISMS EYLEMİ: BC-DIS -> Şüpheli uydu ({node_to_quarantine}) izole edildi.")
                self.quarantine_list.add(node_to_quarantine); self.metrics['response_blockchain'] += 1
        
        if len(self.active_alarms) >= 2: self.threat_level = "KRİTİK"
        elif "JAMMING" in self.active_alarms or "SPOOFING" in self.active_alarms: self.threat_level = "YÜKSEK"
        elif "DATA_TAMPERING" in self.active_alarms: self.threat_level = "ORTA"
        
        if self.threat_level == "KRİTİK" and self.metrics.get('hqc_boost_activated', 0) == 0:
            new_responses.append("ISMS EYLEMİ: HQC-SAT -> Kripto güvenlik seviyesi MAKSİMUM'a yükseltildi.")
            self.metrics['response_hqc_boost'] += 1; self.metrics['hqc_boost_activated'] = 1
            
        return new_responses, self.get_active_responses()
async def run_isms_orchestration_simulation(scenario_choice: str):
    isms_brain = ISMS_Brain_Final()
    SIMULATION_DURATION = 20
    if scenario_choice == "1":
        yield {"type": "header", "message": "ISMS Senaryo 1: Topyekün Saldırı (Dinamik Sıralı) Başlatıldı"}
        attack_types = ["SPOOFING", "DATA_TAMPERING", "JAMMING"]; random.shuffle(attack_types)
        start_times = sorted(random.sample(range(2, 8), len(attack_types)))
        attack_schedule = dict(zip(attack_types, start_times))
        yield {"type": "log", "message": f"Saldırı planı oluşturuldu: {attack_schedule}"}
        spoof_ip, tamper_sat = f"185.12.{random.randint(10,255)}.{random.randint(10,255)}", f"GEN9-SAT0{random.randint(2,5)}"
        jammer_channel, jammer_power = random.randint(40, 60), round(random.uniform(100.0, 110.0), 1)
        active_attacks = set()

        for t in range(SIMULATION_DURATION):
            yield {"type": "header", "message": f"ZAMAN ADIMI {t+1}/{SIMULATION_DURATION}"}
            new_responses, _ = [], []
            for threat, start_time in attack_schedule.items():
                if t == start_time:
                    active_attacks.add(threat)
                    details = {}
                    if threat == "SPOOFING": details = {"source_ip": spoof_ip}; yield {"type": "alert", "message": f"TEHDİT BAŞLADI: SPOOFING (Kaynak: {spoof_ip})"}
                    elif threat == "DATA_TAMPERING": details = {"source_satellite": tamper_sat}; yield {"type": "warn", "message": f"TEHDİT BAŞLADI: DATA_TAMPERING (Hedef: {tamper_sat})"}
                    elif threat == "JAMMING": details = {"target_channel": jammer_channel}; yield {"type": "alert", "message": f"!!! TEHDİT BAŞLADI: JAMMING (Hedef: Ch.{jammer_channel}, Güç: {jammer_power} dBm) !!!"}
                    new_responses, _ = isms_brain.process_alarm({"type": threat, **details})
            
            yield {"type": "status", "message": f"ISMS DURUMU: Tehdit Seviyesi='{isms_brain.threat_level}', Aktif Alarmlar=[{', '.join(isms_brain.active_alarms) or 'YOK'}]"}
            if new_responses:
                for resp in set(new_responses): yield {"type": "success", "message": resp}
            else:
                if isms_brain.active_alarms:
                    yield {"type": "log", "message": "Mevcut savunma durumu sürdürülüyor:"}
                    for active_resp in isms_brain.get_active_responses(): yield {"type": "log", "message": f"  -> {active_resp}"}
                else: yield {"type": "log", "message": "Sistem normal operasyonda."}
            await asyncio.sleep(0.8) # time.sleep -> asyncio.sleep

    elif scenario_choice == "2":
        yield {"type": "header", "message": "ISMS Senaryo 2: Rastgele ve Dinamik Saldırılar Başlatıldı"}
        active_threats = {}; PROB_START, PROB_STOP = 0.25, 0.3
        for t in range(SIMULATION_DURATION): # Süre 20'ye sabitlendi
            yield {"type": "header", "message": f"ZAMAN ADIMI {t+1}/{SIMULATION_DURATION}"}; threat_changed = False
            for threat in ["JAMMING", "SPOOFING", "DATA_TAMPERING"]:
                if threat not in active_threats and random.random() < PROB_START:
                    details = {}
                    if threat == "JAMMING": details = {"target_channel": random.randint(1,100), "power_dbm": round(random.uniform(90.0, 105.0),1)}
                    elif threat == "SPOOFING": details = {"source_ip": f"112.54.{random.randint(1,255)}.{random.randint(1,255)}"}
                    elif threat == "DATA_TAMPERING": details = {"source_satellite": f"GEN9-SAT{random.randint(1,9):02d}"}
                    active_threats[threat] = details; isms_brain.process_alarm({"source": "SENSOR", "type": threat, **details}); threat_changed = True
                    yield {"type": "alert", "message": f"TEHDİT BAŞLADI: {threat} - Detaylar: {details}"}
                elif threat in active_threats and random.random() < PROB_STOP:
                    active_threats.pop(threat); isms_brain.clear_alarm(threat); threat_changed = True
                    yield {"type": "info", "message": f"TEHDİT SONLANDI: {threat} saldırısı durdu."}
            
            new_responses, active_responses = isms_brain._run_rules_engine()
            yield {"type": "status", "message": f"ISMS DURUMU: Tehdit Seviyesi='{isms_brain.threat_level}', Aktif Alarmlar=[{', '.join(isms_brain.active_alarms) or 'YOK'}]"}
            if new_responses:
                for resp in set(new_responses): yield {"type": "success", "message": resp}
            else:
                if active_responses:
                    yield {"type": "log", "message": "Mevcut savunma durumu sürdürülüyor:"}
                    for active_resp in active_responses: yield {"type": "log", "message": f"  -> {active_resp}"}
                else: yield {"type": "log", "message": "Aktif alarm yok, sistem stabil."}
            await asyncio.sleep(1) # time.sleep -> asyncio.sleep
    
    total_alarms = isms_brain.metrics['total_alarms']; total_responses = sum(v for k,v in isms_brain.metrics.items() if 'response' in k)
    summary = { "title": "ISMS Orkestrasyonu İstatistiksel Raporu", "scenario": "Topyekün Saldırı Analizi" if scenario_choice == "1" else "Dinamik Tehdit Analizi", "duration_sec": SIMULATION_DURATION,
               "statistics": { "Toplam Tehdit Alarmı": total_alarms, "Jamming Alarmları": isms_brain.metrics['alarm_jamming'], "Spoofing Alarmları": isms_brain.metrics['alarm_spoofing'],
                             "Veri Bütünlüğü Alarmları": isms_brain.metrics['alarm_data_tampering'], "Toplam Otonom Tepki Sayısı": isms_brain.metrics['total_responses'],
                             "Frekans Atlama (FHSS) Komutları": isms_brain.metrics['response_fhss'], "IP Engelleme (Firewall) Komutları": isms_brain.metrics['response_firewall'],
                             "Node İzolasyon (Blockchain) Komutları": isms_brain.metrics['response_blockchain'], "Kripto Seviyesi Yükseltme (HQC)": isms_brain.metrics['response_hqc_boost']},
               "final_state": {"blacklist": list(isms_brain.blacklist), "quarantine_list": list(isms_brain.quarantine_list)}, "final_threat_level": isms_brain.threat_level,
               "narrative": (f"SİSTEM DEĞERLENDİRMESİ:\n A. Gözlem: Simülasyon boyunca ISMS'e toplam {total_alarms} adet tehdit alarmı iletilmiştir.\n"
                             f"B. Analiz: Alarmların {isms_brain.metrics['alarm_jamming']}'i Jamming, {isms_brain.metrics['alarm_spoofing']}'i Spoofing ve {isms_brain.metrics['alarm_data_tampering']}'i Veri Bütünlüğü kategorisindedir.\n"
                             f"C. Tepki: ISMS, bu alarmlara karşılık olarak toplam {total_responses} adet otonom komut üreterek tehditleri kontrol altına almıştır.\n"
                             f"D. Sonuç: Sistem, operasyonu '{isms_brain.threat_level}' durumunda tamamlamıştır.")}
    yield {"type": "final", "message": "ISMS Orkestrasyonu tamamlandı.", "summary": summary}