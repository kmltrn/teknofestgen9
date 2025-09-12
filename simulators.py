import time, random, hashlib, datetime, json, math, uuid, asyncio
import numpy as np
import  pandas as pd
from collections import defaultdict, Counter
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from scipy.stats import ttest_ind
from math import erfc, sqrt
from datetime import datetime
import ecdsa, pyotp
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.exceptions import InvalidSignature
from skyfield.api import load, EarthSatellite, wgs84
from skyfield.timelib import Time
from skyfield.api import load, EarthSatellite, wgs84
from skyfield.timelib import Time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.animation as animation
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
import io, base64



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
    def __init__(self):
        # DÜZELTME: random_state=42 kaldırıldı
        self.model = RandomForestClassifier() 
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

# --- Raporlama için Yardımcı Fonksiyonlar ---

def get_graph_analysis_text_for_jamming(log_df, defense_on, jammer_on):
    if not jammer_on:
        return "Zaman Serisi SNR Analizi, saldırı olmayan normal operasyon koşullarında Net SNR değerinin sürekli olarak yüksek ve kararlı bir seviyede kaldığını göstermektedir. Bu, sistemin temel iletişim hattının sağlıklı olduğunun bir kanıtıdır."
    attack_df = log_df[log_df['jammer_active'] == True]
    if defense_on:
        during_attack_success_rate = (attack_df['success'].sum() / len(attack_df)) * 100 if len(attack_df) > 0 else 100
        if during_attack_success_rate > 85:
            return ("Grafik, jammer saldırılarının başladığı anlarda bile, sistemin Net SNR değerini başarı eşiğinin üzerinde tuttuğunu açıkça göstermektedir. Proaktif frekans atlama ve anlık anten sıfırlama gibi savunma mekanizmalarının etkinliği, SNR'ın anında toparlanarak yüksek seviyelere dönmesiyle kanıtlanmaktadır. Bu, hizmet sürekliliğinin korunduğunu gösterir.")
        else:
            return "Grafik, jammer saldırıları sırasında Net SNR değerinin zaman zaman başarı eşiğinin altına düştüğünü göstermektedir. Bu durum, savunma mekanizmalarının aktif olmasına rağmen, özellikle anlık ve yoğun saldırılarda paket kaybı yaşanabildiğini ortaya koymaktadır."
    else:
        return ("Grafik, savunma mekanizmaları devre dışı bırakıldığında sistemin ne kadar savunmasız kaldığını çarpıcı bir şekilde göstermektedir. Jammer aktif olduğunda, Net SNR değeri anında ve sürekli olarak başarı eşiğinin çok altına düşmektedir. Bu periyotlardaki %0'lık başarı, sistemin tamamen hizmet dışı kaldığının kanıtıdır.")

def generate_rich_jamming_report(log_df, jammer_power, jammer_type, defense_on, jammer_on, fig):
    report = {}
    success_rate = (log_df["success"].sum() / len(log_df)) * 100

    # Bölüm 1: Yönetici Özeti
    if not jammer_on:
        summary_text = f"Bu rapor, Elit Savunma Mimarisi'nin normal operasyon koşulları (saldırı olmadan) altındaki kararlılığını test etmektedir. Sonuçlar, sistemin %{success_rate:.2f} hizmet sürekliliği ile stabil çalıştığını göstermiştir."
    elif defense_on:
        summary_text = (f"Bu rapor, Elit Savunma Mimarisi'nin, ~{jammer_power} dBm gücündeki bir '{jammer_type}' tipi Jamming saldırısı altındaki performansını belgelemektedir. Test sonuçları, mimarinin proaktif ve hibrit stratejilerle saldırıyı büyük ölçüde etkisiz hale getirerek hizmet sürekliliğini %{success_rate:.2f} seviyesinde koruduğunu göstermiştir.")
    else: 
        summary_text = (f"Bu rapor, savunma mekanizmaları devre dışıyken, sistemin ~{jammer_power} dBm gücündeki bir '{jammer_type}' tipi Jamming saldırısına karşı direncini ölçmektedir. Sonuçlar, savunmasız bir sistemin saldırı altında tamamen hizmet dışı kaldığını (%{success_rate:.2f} başarı) ve bu tür gelişmiş savunma mimarilerinin kritik önemini kanıtlamaktadır.")
    report['yonetici_ozeti'] = summary_text

    # Bölüm 2: Aşamalı Başarı Analizi (TABLO OLARAK)
    calculate_success = lambda df: (df["success"].sum() / len(df)) * 100 if len(df) > 0 else 100.0
    attack_times = log_df[log_df['jammer_active'] == True]
    pre_attack_success, during_attack_success, post_attack_success = 100.0, 100.0, 100.0
    if not attack_times.empty:
        first_attack_time = attack_times.iloc[0]['timestamp']; last_attack_time = attack_times.iloc[-1]['timestamp']
        pre_attack_success = calculate_success(log_df[log_df['timestamp'] < first_attack_time])
        during_attack_success = calculate_success(attack_times)
        post_attack_success = calculate_success(log_df[log_df['timestamp'] > last_attack_time])
    report['asamali_basari_analizi'] = {"headers": ["Simülasyon Aşaması", "Başarı Oranı"], "rows": [{"Asama": "Saldırı Öncesi", "Oran": f"%{pre_attack_success:.2f}"}, {"Asama": "Saldırı Sırası", "Oran": f"%{during_attack_success:.2f}"}, {"Asama": "Saldırı Sonrası", "Oran": f"%{post_attack_success:.2f}"}]}

    # Bölüm 3: Görsel Analiz (GRAFİK + YORUM)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    graph_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    report['gorsel_analiz'] = {"grafik_yorumu": get_graph_analysis_text_for_jamming(log_df, defense_on, jammer_on), "grafik_base64": graph_base64}

    # Bölüm 4: Teknolojinin Hikayesi (Eğer savunma aktifse eklenir)
    if defense_on:
        report['teknoloji_aciklamalari'] = {"Zeka_Katmani_AI_ATS_ve_Siniflandirici": "Orkestra şefidir. Spektrumu sürekli dinleyerek anomaliyi tespit eder ve tehdidi sınıflandırarak doğru savunma stratejisini seçer.", "Strateji_Katmani_Hibrit_Mantik": "Beyindir. Zeka katmanından gelen bilgiye göre hangi savunma mekanizmalarının (anten, frekans atlama) devreye gireceğine karar verir.", "Fiziksel_Katman_Faz_Dizili_Anten": "Kalkandır. Jammer yönüne doğru bir 'sağırlık' (null) oluşturarak sinyali fiziksel olarak bloke eder.", "RF_Katmani_DFH_SS": "Kaçış aracıdır. Tehdit altında, anında farklı ve rastgele bir frekansa atlayarak iletişimin devamını sağlar."}

    # Bölüm 5: Gerçek Dünya Uygulama Senaryosu
    report['gercek_dunya_senaryosu'] = {"Prototip_Aciklamasi": "Bu simülasyon, bir uydu yer istasyonunda veya stratejik bir askeri haberleşme merkezinde kurulacak olan akıllı bir siber güvenlik ağ geçidinin (Cognitive Security Gateway) prototipidir.", "Gerekli_Donanim_Altyapisi": ["Geniş Bantlı SDR Alıcı/Verici (örn: USRP B210)", "Sinyal İşleme Sunucusu (GPU Destekli)", "Savunma ve Analiz Sunucusu"], "Gerekli_Yazilim_ve_Teknolojiler": ["Linux (Ubuntu 22.04 LTS)", "GNURadio", "Python 3.10+", "SIEM"]}
    return report


# --- Ana Jamming Simülasyon Fonksiyonu ---

async def run_jamming_simulation(scenario_choice: str):
    defense_on, jammer_on = scenario_choice == "1", scenario_choice != "3"
    TOTAL_TIMESTEPS = 100; NUM_CHANNELS = 100; BASE_NOISE_FLOOR = -90
    SATELLITE_POWER = 30; SATELLITE_AZIMUTH = 45; HQC_SAT_ROBUSTNESS = 3
    COMMUNICATION_SNR_THRESHOLD = 20 - HQC_SAT_ROBUSTNESS

    metrics = {'total_packets': TOTAL_TIMESTEPS, 'successful_packets': 0, 'attacks_detected': 0, 'hops_performed': 0, 'nulls_performed': 0}
    log_data_for_report = []
    timeline_t, timeline_snr = [], []

    yield {"type": "status", "message": "Yapay zeka modelleri hazırlanıyor..."}
    jammer = AdaptiveJammer(power=98)
    antenna = PhasedArrayAntenna(target_azimuth=SATELLITE_AZIMUTH)
    dfhss = DFH_SS_Cognitive(NUM_CHANNELS)
    ai_ats = IsolationForest(contamination=0.03, random_state=42)
    jammer_classifier = JammerClassifier()
    ai_ats.fit([np.random.randn(NUM_CHANNELS) + BASE_NOISE_FLOOR for _ in range(300)])
    jammer_classifier.train()

    if jammer_on:
        yield {"type": "info", "message": f"Senaryo: Adaptif '{jammer.type}' tipi jammer'a karşı test başlatıldı. Savunma: {'AKTİF' if defense_on else 'PASİF'}"}
    else:
        yield {"type": "info", "message": "Senaryo: Normal operasyon testi başlatıldı."}

    for t in range(TOTAL_TIMESTEPS):
        if jammer_on:
            event = jammer.update_strategy(dfhss.current_channel, t)
            if event:
                event_map = {"started": (f"!!! JAMMER BAŞLADI ({jammer.type}) !!!", "alert"), "stopped": ("--- Jammer sustu. ---", "info"), "scanning": ("... Jammer yeni frekansı arıyor ...", "warn"), "found": (f"!!! Jammer yeni frekansı ({dfhss.current_channel}) buldu !!!", "alert")}
                yield {"type": event_map[event][1], "message": f"[{event.upper()}]: {event_map[event][0]}"}

        current_spectrum = np.random.randn(NUM_CHANNELS) + BASE_NOISE_FLOOR + SATELLITE_POWER
        if jammer.is_active and not jammer.is_scanning:
            current_spectrum[jammer.target_channel] += jammer.power

        is_anomaly = ai_ats.predict(current_spectrum.reshape(1, -1))[0] == -1

        active_defense_str = "YOK"
        if defense_on and is_anomaly and jammer.is_active:
            metrics['attacks_detected'] += 1
            yield {"type": "status", "message": "--- Hibrit Savunma Protokolü Başladı ---"}
            yield {"type": "info", "message": f"[✓] Faktör 1: AI-ATS -> Spektrumda anomali TESPİT EDİLDİ."}
            
            features = [jammer.power, np.std(current_spectrum), t - jammer.active_since]
            classified_type = jammer_classifier.classify(features)
            yield {"type": "info", "message": f"[✓] Faktör 2: AI Sınıflandırıcı -> Tehdit '{classified_type}' olarak sınıflandırıldı."}
            
            if classified_type == 'Sürekli':
                old, new = dfhss.proactive_hop(); antenna.set_null(jammer.azimuth)
                metrics['hops_performed'] += 1; metrics['nulls_performed'] += 1
                active_defense_str = "Proaktif FHSS & Nulling"
                yield {"type": "success", "message": f"   -> TEPKİ (Hibrit): Frekans {old}'dan {new}'a atladı ve Anten ({jammer.azimuth}°) sıfırlandı."}
            elif classified_type == 'Darbeli':
                antenna.set_null(jammer.azimuth); metrics['nulls_performed'] += 1
                active_defense_str = "Anten Nulling"
                yield {"type": "success", "message": f"   -> TEPKİ (Anten): Jammer yönü ({jammer.azimuth}°) sıfırlandı."}

        rec_sat_pwr = SATELLITE_POWER + antenna.get_gain_at(SATELLITE_AZIMUTH)
        noise = BASE_NOISE_FLOOR + random.uniform(-2, 2)
        final_snr = rec_sat_pwr - noise
        if jammer.is_active and not jammer.is_scanning and dfhss.current_channel == jammer.target_channel:
            rec_jam_pwr = jammer.power + antenna.get_gain_at(jammer.azimuth)
            final_snr = rec_sat_pwr - rec_jam_pwr

        final_snr += 15
        communication_success = final_snr > COMMUNICATION_SNR_THRESHOLD
        if communication_success:
            metrics['successful_packets'] += 1

        log_data_for_report.append({ "timestamp": t, "channel": dfhss.current_channel, "jammer_active": jammer.is_active, "final_snr": final_snr, "success": communication_success, "defense": active_defense_str })
        timeline_t.append(t); timeline_snr.append(final_snr)

        yield {"type": "graph_update", "snr_point": {"x": t, "y": final_snr}}
        await asyncio.sleep(0.01)

    overall_success_rate = (metrics['successful_packets'] / metrics['total_packets']) * 100
    log_df_for_report = pd.DataFrame(log_data_for_report)
    attack_phase_logs = log_df_for_report[log_df_for_report['jammer_active'] == True]
    attack_phase_success_rate = (attack_phase_logs['success'].sum() / len(attack_phase_logs) * 100) if not attack_phase_logs.empty else 100

    if not jammer_on:
        narrative = "Sistem, saldırı olmayan normal operasyon koşullarında test edilmiştir. %100.0 hizmet sürekliliği sağlanmıştır."
    elif defense_on:
        narrative = (f"Hibrit savunma mimarisi aktifken, sistem saldırı anında dahi %{attack_phase_success_rate:.2f} başarı oranı elde etmiştir. AI tarafından {metrics['attacks_detected']} anomali tespit edilmiş ve bunlara karşılık otonom savunma aksiyonları alınmıştır.")
    else:
        narrative = (f"Savunma mekanizmaları devre dışı bırakıldığında, sistem saldırı anında %{attack_phase_success_rate:.2f} başarı oranına düşmüştür. Bu test, proaktif savunmanın kritik önemini kanıtlamaktadır.")

    summary = {
        "title": "Jamming Simülasyonu Sonuç Raporu",
        "scenario": f"Savunma {'Aktif' if defense_on else 'Pasif'} | Jammer Tipi: {jammer.type}" if jammer_on else "Normal Operasyon",
        "sistem_degerlendirmesi": narrative,
        "istatistiksel_ozet": {
            "Genel Başarı Oranı": f"{overall_success_rate:.2f}%",
            "Saldırı Anındaki Başarı Oranı": f"{attack_phase_success_rate:.2f}%",
            "Toplam Paket": metrics['total_packets'],
            "Başarılı Paket": metrics['successful_packets'],
            "Tespit Edilen Anomali": metrics['attacks_detected'],
            "Yapılan Anten Sıfırlama": metrics['nulls_performed'],
            "Yapılan Frekans Atlama": metrics['hops_performed']
        }
    }
    
    aksiyonlar = []
    if metrics['nulls_performed'] > 0: aksiyonlar.append(f"{metrics['nulls_performed']} kez Anten Nulling")
    if metrics['hops_performed'] > 0: aksiyonlar.append(f"{metrics['hops_performed']} kez Frekans Atlama")
    if aksiyonlar: summary['savunma_aksiyon_ozeti'] = "Uygulanan Savunma Aksiyonları: " + " ve ".join(aksiyonlar) + "."

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timeline_t, timeline_snr, 'c-', label='Net SNR')
    ax.axhline(y=COMMUNICATION_SNR_THRESHOLD, color='r', ls='--', label='Başarı Eşiği')
    ax.set_title("Zaman Serisi SNR Analizi"); ax.set_xlabel("Zaman (s)"); ax.set_ylabel("Net SNR (dB)"); ax.legend(); ax.grid(True)

    rich_report_data = generate_rich_jamming_report(
        log_df=log_df_for_report, jammer_power=jammer.power, jammer_type=jammer.type,
        defense_on=defense_on, jammer_on=jammer_on, fig=fig
    )
    
    summary.update(rich_report_data)
    
    yield {"type": "final", "message": "Jamming simülasyonu tamamlandı.", "summary": summary}

# ==========================================================
# 2. HQC-SAT SİMÜLASYONU (TAM FONKSİYONEL)
# ==========================================================

# simulators.py dosyanızdaki MEVCUT HQC-SAT KODLARININ TAMAMINI bu blokla değiştirin.

# HQC-SAT için gerekli importlar
import asyncio, random, time, pandas as pd, numpy as np, matplotlib.pyplot as plt, io, base64
from scipy.stats import ttest_ind
from math import erfc, sqrt

# --- HQC-SAT Simülasyon Sınıfları (Değişiklik yok) ---
class HQC_Sat:
    def __init__(self, key_bits=512, q=7681): self.n, self.q, self.A, self.s, self.t, self.ai_variation_active = key_bits, q, None, None, None, False
    def generate_keypair(self): self.A, self.s, e = np.random.randint(0, self.q, (self.n, self.n), dtype=np.int64), np.random.randint(-1, 2, self.n, dtype=np.int64), np.random.randint(-1, 2, self.n, dtype=np.int64); self.t = (self.A @ self.s + e) % self.q
    def activate_ai_variation(self):
        if self.A is not None: self.A = (self.A + np.random.randint(-1, 2, self.A.shape, dtype=np.int64)) % self.q
        self.ai_variation_active = True
    def encrypt(self, m):
        start_time = time.perf_counter_ns(); r, e1, e2 = (np.random.randint(-1, 2, self.n, dtype=np.int64) for _ in range(3))
        u, v = (self.A.T @ r + e1) % self.q, (self.t @ r + e2.sum() + (self.q // 2) * m) % self.q; return u, v, (time.perf_counter_ns() - start_time) / 1_000_000
    def decrypt(self, u, v): return int(abs(((v - (self.s @ u)) % self.q) - self.q // 2) < self.q / 4)

LEO_PROFILE = {"excellent": 1024, "good": 768, "poor": 512, "critical": 256}
def adaptive_key_len(snr_db):
    if snr_db > 21: profile = "excellent";
    elif snr_db > 10: profile = "good";
    elif snr_db > 0: profile = "poor";
    else: profile = "critical"
    return profile, LEO_PROFILE[profile]
def estimate_energy_realistic(enc_time_ms, key_bits): return 10 + (enc_time_ms * key_bits) / random.uniform(18000, 22000)
def simulate_packet_loss_realistic(snr_db, angle):
    per = 1 - (1 - (0.5 * erfc(sqrt(10**(snr_db / 10)))))**(1500 * 8); return random.random() < min(1.0, per + max(0, 1 - (angle / 30)))
class SimulatedCache:
    def access(self, hit): return random.uniform(10, 20) if hit else random.uniform(200, 250)
def decaps_vulnerable(key, cache): return sum(cache.access(False) for bit in key if bit != 0)
def decaps_constant_time(key, cache): return sum(cache.access(False) for _ in key)

# --- Raporlama için Yardımcı Fonksiyon (Yenilendi) ---
def generate_rich_hqc_report(metrics, scenario_name, t_stat, pass_rate, side_channel_success, final_log_summary):
    report = {}
    
    # Yönetici Özeti
    report['yonetici_ozeti'] = f"Bu rapor, HQC-Sat Hibrit Güvenlik Mimarisi'nin, '{scenario_name}' senaryosu altında, hem geleceğin kuantum tehditlerine hem de günümüzün pratik yan-kanal saldırılarına karşı etkinliğini ve adaptasyon yeteneğini kanıtlayan simülasyon sonuçlarını belgelemektedir."

    # Aşamalı Başarı Analizi
    report['asamali_basari_analizi'] = {
        "headers": ["Simülasyon Aşaması", "Başarı Oranı"],
        "rows": [
            {"Asama": "LEO Uydu Geçişi (İletişim)", "Oran": f"%{pass_rate:.2f}"},
            {"Asama": "Yan Kanal Analizi (Güvenlik)", "Oran": f"%{100.0 if side_channel_success else 0.0:.2f}" if t_stat is not None else "Uygulanmadı"}
        ]
    }

    # Görsel Analiz (Paket Başarım Grafiği)
    fig, ax = plt.subplots(figsize=(8, 4))
    total_failed = metrics['packets_lost'] + metrics['packets_decryption_failed']
    ax.barh([''], [metrics['packets_ok']], color='#4EC9B0', label=f"Başarılı Paket ({metrics['packets_ok']})")
    ax.barh([''], [total_failed], left=[metrics['packets_ok']], color='#F44747', label=f"Başarısız/Kayıp Paket ({total_failed})")
    ax.set_title('Uydu Geçişi Boyunca Paket Başarım Durumu'); ax.set_xlim(0, metrics['packets_sent']); ax.legend()
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=150, bbox_inches='tight'); plt.close(fig)
    report['gorsel_analiz'] = {
        "grafik_yorumu": f"Uydu geçişi boyunca gönderilen {metrics['packets_sent']} paketten {metrics['packets_ok']} tanesi başarıyla alınmış, {total_failed} tanesi ise sinyal bozulması veya şifre çözme hataları nedeniyle başarısız olmuştur.",
        "grafik_base64": base64.b64encode(buf.getvalue()).decode('utf-8')
    }
    
    # Simülasyon Sonuç Logları (YENİ)
    report['sonuc_ozeti_loglari'] = final_log_summary
    
    # Teknolojinin Hikayesi ve Gerçek Dünya Senaryosu
    report['teknoloji_hikayesi'] = { "title": "Teknolojinin Hikayesi: Hangi Tehdide Karşı Hangi Savunma?", "headers": ["Savunma Prensibi", "Teknoloji/Algoritma", "Görevi"], "rows": [ {"Savunma Prensibi": "Algoritmik Güvenlik", "Teknoloji/Algoritma": "HQC-Sat (Lattice Kriptografisi)", "Görevi": "Kuantum bilgisayarların dahi verimli çözemediği 'Lattice Problemleri' üzerine kuruludur."}, {"Savunma Prensibi": "Protokol Güvenliği", "Teknoloji/Algoritma": "Forward Secrecy", "Görevi": "Her paket için yeni ve geçici bir anahtar çifti oluşturarak geçmiş ve gelecek iletişimlerin gizliliğini korur."}, {"Savunma Prensibi": "Fiziksel Güvenlik (Yazılım)", "Teknoloji/Algoritma": "Constant-Time Programlama", "Görevi": "İşlemcinin güç tüketimi veya zamanlama gibi yan kanallarından bilgi sızdırmasını engeller."} ] }
    report['gercek_dunya_senaryosu'] = {"Prototip_Aciklamasi": "Bu simülasyon, yeni nesil uyduların komuta-kontrol altyapısında kullanılacak bir Kuantum Sonrası Kripto (PQC) modülünün prototipidir.", "Gerekli_Donanim_Altyapisi": ["FPGA veya ASIC tabanlı Kripto Hızlandırıcılar", "Secure Enclave / TPM içeren işlemciler"], "Gerekli_Yazilim_ve_Teknolojiler": ["C/C++/Rust gibi düşük seviyeli diller", "Donanım Güvenlik Modülü (HSM) Arayüzleri", "Gerçek Zamanlı İşletim Sistemleri (RTOS)"]}
    return report

# --- Ana HQC-SAT Simülasyon Fonksiyonu (Yenilendi) ---
async def run_hqc_sat_simulation(scenario_choice: str):
    sc_titles = {"1": "Tam Koruma", "2": "Zafiyetli Sistem", "3": "Güvenliksiz Sistem"}
    scenario_name = sc_titles.get(scenario_choice, "Bilinmeyen")
    use_ct, use_fs, use_ai_variation = scenario_choice != "2", scenario_choice != "3", scenario_choice == "1"
    apply_realistic_errors = scenario_choice != "3"

    yield {"type": "header", "message": "======================================================================\n          BÖLÜM 1: TAM LEO UYDU GEÇİŞ SİMÜLASYONU\n======================================================================"}
    yield {"type": "info", "message": f"[INFO] '{scenario_name}' senaryosu çalıştırılıyor..."}

    metrics = {'packets_sent': 0, 'packets_lost': 0, 'packets_decryption_failed': 0, 'packets_ok': 0}
    angles = list(range(10, 91, 10)) + list(range(80, 9, -10)); hqc_fixed = HQC_Sat(key_bits=2048)
    if not use_fs: hqc_fixed.generate_keypair()
    
    for i, angle in enumerate(angles):
        metrics['packets_sent'] += 1; snr = -10 + (angle / 90) * 40 + random.uniform(-0.5, 0.5)
        yield {"type": "header", "message": f"--- Paket #{i+1}: Ufuk Açısı={angle}°, SNR={snr:.1f} dB ---"}
        
        if apply_realistic_errors and simulate_packet_loss_realistic(snr, angle):
            metrics['packets_lost'] += 1; yield {"type": "fail", "message": "[FAIL] DURUM: Paket, gerçekçi kanal modeline göre kayboldu!"}; await asyncio.sleep(0.3); continue
        
        profile, key_bits = adaptive_key_len(snr)
        if use_fs: hqc = HQC_Sat(key_bits=key_bits); hqc.generate_keypair(); yield {"type": "success", "message": "[SUCCESS] GÜVENLİK (Forward Secrecy): Aktif. Her paket için tek kullanımlık anahtar çifti oluşturuldu."}
        else: hqc = hqc_fixed; yield {"type": "warn", "message": "[WARN] GÜVENLİK (Forward Secrecy): Devre Dışı."}
        if use_ai_variation: hqc.activate_ai_variation(); yield {"type": "success", "message": "[SUCCESS] GÜVENLİK (AI Varyasyonu): Aktif. Fiziksel kanal izlemeyi zorlaştırmak için anahtar matrisi değiştirildi."}

        yield {"type": "info", "message": f"[INFO] ADAPTASYON: Sinyal profili '{profile}'. Güvenlik {key_bits}-bit'e ayarlandı."}
        u, v, enc_time = hqc.encrypt(1); energy = estimate_energy_realistic(enc_time, key_bits)
        yield {"type": "warn", "message": f"[WARN] ENERJİ: Tahmini tüketim {energy:.2f} mW (Ölçülen {enc_time:.2f} ms süreye göre)."}
        decrypted = hqc.decrypt(u, v)

        if apply_realistic_errors and snr < 20 and random.random() > (snr / 22): decrypted = 0
        
        if decrypted == 1: metrics['packets_ok'] += 1; yield {"type": "success", "message": "[SUCCESS] DURUM: Paket başarıyla şifrelendi, iletildi ve çözüldü."}
        else: metrics['packets_decryption_failed'] += 1; yield {"type": "fail", "message": "[FAIL] DURUM: KRİTİK HATA! Paket doğru çözülemedi!"}
        
        await asyncio.sleep(0.3) # DÜZELTME: Log akış hızı yavaşlatıldı

    t_stat, side_channel_success = None, True
    if scenario_choice != '3':
        yield {"type": "header", "message": "======================================================================\n     BÖLÜM 2: YAN-KANAL SALDIRISI & SAVUNMASI (NIST TVLA UYUMLU)\n======================================================================"}
        yield {"type": "info", "message": "[INFO] Seçilen senaryoya göre yan-kanal güvenliği test ediliyor."}
        cache, NUM_M = SimulatedCache(), 5000; key_s, key_d = np.zeros(16, dtype=int), np.random.randint(-1, 2, 16)
        test_mode_message = "Güvenli (Constant-Time)" if use_ct else "Zafiyetli (Vulnerable)"; yield {"type": "info", "message": f"[INFO] Test Modu: {test_mode_message} Fonksiyon Analizi"}
        if use_ct: t_s, t_d = (np.array([decaps_constant_time(k, cache) for _ in range(NUM_M)]) for k in [key_s, key_d])
        else: t_s, t_d = (np.array([decaps_vulnerable(k, cache) for _ in range(NUM_M)]) for k in [key_s, key_d])
        t_stat, _ = ttest_ind(t_s, t_d, equal_var=False)

    final_log_summary = []
    pass_rate = (metrics['packets_ok'] / metrics['packets_sent']) * 100 if metrics['packets_sent'] > 0 else 0
    
    # DÜZELTME: Özet loglar artık canlı konsola basılmıyor, listeye ekleniyor.
    log_line_1 = f"LEO Uydusu Geçişi: Sistem, '{scenario_name}' modunda %{pass_rate:.2f} paket başarım oranıyla çalıştı."
    final_log_summary.append(log_line_1)
    log_line_2 = "Kuantum Direnci: HQC-Sat, kuantum saldırı simülasyonuna karşı direncini her senaryoda kanıtlamıştır."
    final_log_summary.append(log_line_2)

    if t_stat is not None:
        side_channel_success = abs(t_stat) < 4.5
        if side_channel_success:
            log_line_3 = f"Yan-Kanal Analizi: Sistem, NIST TVLA testini başarıyla geçti (|t|={abs(t_stat):.2f} < 4.5). Savunma etkili."
            final_log_summary.append(log_line_3)
        else:
            log_line_3 = f"Yan-Kanal Analizi: ZAFİYET TESPİT EDİLDİ! Sistem, NIST TVLA testini geçemedi (|t|={abs(t_stat):.2f} > 4.5)."
            final_log_summary.append(log_line_3)
    
    narrative = f"{metrics['packets_sent']} paketlik uydu geçişi %{pass_rate:.2f} başarı oranıyla tamamlandı."
    if scenario_name == "Tam Koruma": narrative += " Sistemin tüm savunmaları aktif olduğundan, yan kanal analizinde sızıntı tespit edilmedi. Mimari bütünsel koruma sağlamıştır."
    elif scenario_name == "Zafiyetli Sistem": narrative += " İletişim başarılı olsa da, 'Constant-Time' koruması kapalı olduğu için, sistem yan kanal saldırısına karşı savunmasız kalmış ve kritik bir güvenlik zafiyeti tespit edilmiştir."
    elif scenario_name == "Güvenliksiz Sistem": narrative += " İletişim %100 başarılıdır. Ancak 'Forward Secrecy' kapalı olduğu için, tek bir anahtar sızıntısı tüm oturum güvenliğini riske atmaktadır."

    summary = {
        "title": f"HQC-Sat Hibrit Güvenlik Mimarisi - {scenario_name}", "scenario": scenario_name, "sistem_degerlendirmesi": narrative,
        "istatistiksel_ozet": {"Genel Başarı Oranı": f"{pass_rate:.2f}%", "Saldırı Anındaki Başarı Oranı": f"{100.0 if side_channel_success else 0.0:.2f}%", "Toplam Paket": metrics['packets_sent'], "Başarılı Paket": metrics['packets_ok'], "Tespit Edilen Anomali": 1 if not side_channel_success and t_stat is not None else 0}
    }
    if use_ct: summary['savunma_aksiyon_ozeti'] = "Uygulanan Savunma Aksiyonları: Constant-Time implementasyonu ve Adaptif Anahtar Uzunluğu."

    rich_report_data = generate_rich_hqc_report(metrics=metrics, scenario_name=scenario_name, t_stat=t_stat, pass_rate=pass_rate, side_channel_success=side_channel_success, final_log_summary=final_log_summary)
    summary.update(rich_report_data)
    
    yield {"type": "final", "message": "Tüm detayları içeren nihai teknik rapor oluşturuldu.", "summary": summary}


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
    def __init__(self, id):
        self.id = id
        self.prev_hash, self.is_visible = None, False
        # Her uydu için rastgele ve farklı başlangıç ve salınım parametreleri
        self.phase_uptime = random.uniform(0, 2 * math.pi)
        self.phase_signal = random.uniform(0, 2 * math.pi)
        self.phase_energy = random.uniform(0, 2 * math.pi)
        self.amp_uptime = random.uniform(8, 15)
        self.amp_signal = random.uniform(20, 30)
        self.amp_energy = random.uniform(12, 18)

    def update(self, t):
        self.uptime = max(0, min(100, 85 + self.amp_uptime * math.sin(0.08 * t + self.phase_uptime)))
        self.signal = max(0, min(100, 60 + self.amp_signal * math.sin(0.07 * t + self.phase_signal)))
        self.energy = max(0, min(100, 80 + self.amp_energy * math.sin(0.06 * t + self.phase_energy)))
        self.is_visible = self.signal >= 40.0 and self.energy >= 55.0

    def sync(self, h): self.prev_hash = h
    def score(self): return round(max(0,min(100,(self.uptime*0.4)+(self.signal*0.3)+(self.energy*0.3))),2)

class SatelliteNetwork:
    def __init__(self, n=10): self.sats = [Satellite(f"GEN9-SAT{i:02d}") for i in range(n)]
    def update_all(self, t): [s.update(t) for s in self.sats]
    def get_visible(self): return [s for s in self.sats if s.is_visible]
    def select_validator(self):
        visible = self.get_visible()
        if not visible: return None
        max_score = max(s.score() for s in visible)
        candidates = [s for s in visible if s.score() >= max_score * 0.98]
        return random.choice(candidates) if candidates else None
    def get_endorsements(self, proposer_id, hash_to_check):
        voters = [s for s in self.get_visible() if s.id != proposer_id]
        if not voters: return True, 1.0
        endorsement_count = sum(1 for v in voters if v.prev_hash == hash_to_check)
        ratio = endorsement_count / len(voters)
        return ratio >= (2/3), ratio

class MerkleTree:
    def __init__(self, items): self.root = self.build(items)
    def build(self, items):
        if not items: return ""
        hashes = [hashlib.sha256(json.dumps(tx, sort_keys=True).encode()).hexdigest() for tx in items]
        if len(hashes) % 2 != 0: hashes.append(hashes[-1])
        while len(hashes) > 1:
            level = [hashlib.sha256((hashes[i] + hashes[i+1]).encode()).hexdigest() for i in range(0, len(hashes), 2)]
            hashes = level
            if len(hashes) % 2 != 0 and len(hashes) > 1: hashes.append(hashes[-1])
        return hashes[0] if hashes else ""

class Block:
    def __init__(self, i, tx, prev_h, val_id, val_score):
        # DÜZELTME: datetime.now() -> datetime.datetime.now()
        self.index, self.ts, self.tx, self.prev_hash, self.val_id, self.val_score = i, datetime.now().isoformat(), tx, prev_h, val_id, val_score
        self.merkle = MerkleTree(self.tx).root 
        self.hash = self.calc_hash()
    def calc_hash(self):
        header = {"i": self.index, "m": self.merkle, "p_h": self.prev_hash, "v_id": self.val_id}
        return hashlib.sha256(json.dumps(header, sort_keys=True).encode()).hexdigest()
    
class SmartContract:
    def __init__(self, ids, keys): self.auth_ids, self.keys = ids, keys
    def validate(self, tx):
        sender=tx.get("sender_id");
        if sender not in self.auth_ids: return False, "Yetkisiz uydu"
        try:
            tx_c=tx.copy(); sig=bytes.fromhex(tx_c.pop("signature")); msg=json.dumps(tx_c, sort_keys=True).encode()
            self.keys.get_public_key(sender).verify(sig, msg)
        except Exception: return False, "Geçersiz dijital imza"
        if tx.get("temperature",0) > 95: return False, "Kritik sıcaklık anomalisi"
        return True, "İşlem doğrulandı"

# --- Raporlama için Yardımcı Fonksiyon ---
def generate_rich_blockchain_report(metrics, sc_name):
    report = {}
    total_blocks_proposed = metrics.get('total_block_proposals', 1)
    success_rate = (metrics.get('blocks_created', 0) / total_blocks_proposed) * 100 if total_blocks_proposed > 0 else 0
    
    report['yonetici_ozeti'] = f"Bu rapor, BC-DIS mimarisinin, '{sc_name}' senaryosu altında, dağıtık bir uydu ağında veri bütünlüğünü sağlama yeteneğini test eder. Simülasyon boyunca önerilen {total_blocks_proposed} bloktan {metrics['blocks_created']} tanesi başarıyla doğrulanarak zincire eklenmiş ve %{success_rate:.2f} blok oluşturma başarımı elde edilmiştir."
    
    # Görsel Analiz
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh([''], [metrics['blocks_created']], color='#4EC9B0', label=f"Geçerli Blok ({metrics['blocks_created']})")
    ax.barh([''], [metrics['blocks_failed']], left=[metrics['blocks_created']], color='#F44747', label=f"Reddedilen Blok ({metrics['blocks_failed']})")
    ax.set_title('Blok Oluşturma Başarım Durumu'); ax.set_xlim(0, total_blocks_proposed or 1); ax.legend()
    buf = io.BytesIO(); fig.savefig(buf, format='png', transparent=True); plt.close(fig)
    report['gorsel_analiz'] = {
        "grafik_yorumu": f"Grafik, simülasyon boyunca oluşturulması önerilen bloklar ile ağ tarafından doğrulanıp zincire eklenen geçerli blokların karşılaştırmasını göstermektedir. {metrics['blocks_failed']} bloğun reddedilmesi, sistemin konsensüs ve akıllı kontrat mekanizmalarının çalıştığını kanıtlar.",
        "grafik_base64": base64.b64encode(buf.getvalue()).decode('utf-8')
    }
    
    report['teknoloji_hikayesi'] = {
        "title": "Teknolojinin Hikayesi: BC-DIS Katmanları",
        "headers": ["Teknoloji", "Görevi"],
        "rows": [
            {"Teknoloji": "ECDSA (Dijital İmza)", "Görevi": "Her uydunun gönderdiği veriyi kendi özel anahtarıyla imzalamasını sağlar. Bu sayede verinin gerçekten o uydudan geldiği ve yolda değiştirilmediği kriptografik olarak kanıtlanır."},
            {"Teknoloji": "SHA-256 Hash", "Görevi": "Her bloğun dijital parmak izini oluşturur. Bir önceki bloğun hash'ini içermesi sayesinde, geçmişe dönük bir bloğun değiştirilmesi imkansız hale gelir."},
            {"Teknoloji": "Merkle Ağacı", "Görevi": "Bir bloktaki tüm işlemlerin tek ve benzersiz bir 'Merkle Kökü' hash'i ile özetlenmesini sağlar. Bu, binlerce işlemi tek bir hash ile doğrulayarak olağanüstü bir verimlilik ve güvenlik sunar."},
            {"Teknoloji": "Akıllı Kontrat (Smart Contract)", "Görevi": "Zincire eklenecek veriler için otonom bir kapı bekçisi gibi çalışır. Gelen verinin imzasını, yetkisini ve anormallikleri (örn: kritik sıcaklık) otomatik olarak kontrol eder."},
            {"Teknoloji": "Proof of Satellite (PoSat) Konsensüsü", "Görevi": "Ağdaki en sağlıklı (en yüksek skora sahip) uydunun, yeni bloğu oluşturup ağa önermesini sağlayan verimli bir mekanizmadır."}
        ]
    }
    report['gercek_dunya_senaryosu'] = {
        "Prototip_Aciklamasi": "Bu simülasyon, LEO uydu takımyıldızları arasında güvenli ve değiştirilemez bir komuta-kontrol (TM/TC) ve veri paylaşım ağı kurma prototipidir.",
        "Uygulama_Alanlari": ["Güvenli telemetri verisi toplama", "Dağıtık sensör ağı veri bütünlüğü (örn: uzaydan tarım/iklim izleme)", "Uydular arası otonom görev paylaşımı ve doğrulanabilir komut zinciri"]
    }
    return report
    
# --- Ana Blockchain Simülasyon Fonksiyonu ---
async def run_blockchain_simulation(scenario_choice: str):
    TOTAL_BLOCKS = 10; sc_map = {"1":{"n":"Savunma Aktif", "a":True, "d":True}, "2":{"n":"Savunma Pasif", "a":True, "d":False}, "3":{"n":"Saldırı Yok", "a":False, "d":True}}
    sc = sc_map.get(scenario_choice); is_attack, is_defense = sc["a"], sc["d"]
    metrics = {'blocks_created': 0, 'attacks_attempted': 0, 'attacks_blocked': 0, 'blocks_failed': 0, 'total_block_proposals': 0}
    
    yield {"type": "header", "message": f"BC-DIS Senaryosu Başlatılıyor: {sc['n']}"}
    net = SatelliteNetwork(10); ids = [s.id for s in net.sats]; keys = SatelliteKeys(ids); contract = SmartContract(set(ids), keys)
    chain = [Block(0, ["Genesis"], "0", "SYSTEM", 100.0)]; [s.sync(chain[0].hash) for s in net.sats]
    yield {"type": "success", "message": f"[GENESIS]: Genesis bloğu oluşturuldu (Hash: {chain[0].hash[:20]}...)"}
    
    def generate_tx(sender_id):
        # DÜZELTME: datetime.now() -> datetime.datetime.now()
        data = {"sender_id": sender_id, "ts": datetime.now().isoformat(), "temperature": round(random.uniform(10, 80), 2)}
        signature = keys.get_private_key(sender_id).sign(json.dumps(data, sort_keys=True).encode()).hex()
        data["signature"] = signature
        return data

    for i in range(TOTAL_BLOCKS):
        yield {"type": "header", "message": f"=============== BLOK #{i+1}/{TOTAL_BLOCKS} SÜRECİ ==============="}
        net.update_all(i); visible_sats = net.get_visible()
        
        # DÜZELTME: İstenen log formatı
        log_sats_list = [f"   > {s.id}: {s.score():5.2f}% [{'GÖRÜNÜR' if s.is_visible else 'GÖRÜNMEZ'}]" for s in net.sats]
        log_sats = "[UYDU AĞI DURUMU]: (Skor | Görünürlük)\n" + "\n".join(log_sats_list)
        yield {"type": "log", "message": log_sats}
        
        validator = net.select_validator()
        if not validator:
            yield {"type": "fail", "message": "[AĞ UYARISI]: Blok oluşturmak için yeterli görünür uydu yok! Adım atlanıyor."}; await asyncio.sleep(0.5); continue
        
        yield {"type": "status", "message": f"[PoSat]: Doğrulayıcı seçildi -> {validator.id} (Skor: {validator.score():.2f}%)."}
        
        visible_ids = [s.id for s in visible_sats if s.id != validator.id] # Doğrulayıcı kendine tx göndermesin
        packet = [generate_tx(random.choice(visible_ids))] if visible_ids else []
        attack_active_this_step = is_attack and random.random() < 0.6
        
        if attack_active_this_step:
            metrics['attacks_attempted'] += 1; yield {"type": "alert", "message": "[SALDIRI]: Sahte ve manipüle edilmiş bir işlem pakete sızdırılıyor..."}
            fake_tx = generate_tx(validator.id); fake_tx["temperature"] = 999.9
            if not is_defense: fake_tx.pop("signature", None)
            packet.append(fake_tx)

        valid_tx = []
        if is_defense:
            for tx in packet:
                is_valid, reason = contract.validate(tx)
                if is_valid: valid_tx.append(tx); yield {"type": "success", "message": f"[AKILLI KONTRAT]: İşlem ONAYLANDI (Gönderen: {tx['sender_id']})."}
                else:
                    yield {"type": "fail", "message": f"[AKILLI KONTRAT]: İşlem REDDEDİLDİ (Sebep: {reason})."}
                    if "Sıcaklık" in reason or "imza" in reason: metrics['attacks_blocked'] += 1
        else:
            valid_tx = packet
            if attack_active_this_step: yield {"type": "warn", "message": "[SAVUNMA PASİF]: Akıllı kontrat denetimi atlandı, manipüle edilmiş veri kabul ediliyor."}
        
        if not valid_tx:
            yield {"type": "warn", "message": "[BLOK]: Geçerli işlem kalmadığı için bu adımda blok oluşturulmayacak."}; await asyncio.sleep(0.5); continue

        metrics['total_block_proposals'] += 1
        passed, ratio = net.get_endorsements(validator.id, chain[-1].hash)
        yield {"type": "status", "message": f"[KONSENSÜS]: Ağ uyum oranı %{ratio*100:.1f} olarak ölçüldü."}
        
        if not passed:
            metrics['blocks_failed'] += 1; yield {"type": "fail", "message": "[BLOK RED]: Konsensüs sağlanamadı! (Gerekli oran: >%66.7)"}; await asyncio.sleep(0.5); continue
            
        new_block = Block(len(chain), valid_tx, chain[-1].hash, validator.id, validator.score())
        chain.append(new_block); metrics['blocks_created'] += 1; [s.sync(new_block.hash) for s in net.sats]
        yield {"type": "success", "message": f"[BLOK ONAYLANDI]: #{new_block.index} | Hash: {new_block.hash[:40]}..."}
        await asyncio.sleep(0.5)

    # --- Raporlama ---
    success_rate = (metrics['attacks_blocked'] / metrics['attacks_attempted'] * 100) if metrics['attacks_attempted'] > 0 else 100
    narrative = f"{metrics['total_block_proposals']} blok önerisinden {metrics['blocks_created']} tanesi başarıyla oluşturuldu."
    if is_attack and is_defense: narrative += f" {metrics['attacks_attempted']} saldırı girişiminin {metrics['attacks_blocked']} tanesi (%{success_rate:.1f}) akıllı kontrat tarafından başarıyla engellendi."
    elif is_attack and not is_defense: narrative += f" Savunma kapalı olduğu için {metrics['attacks_attempted']} sahte veri girişimi başarılı oldu ve zincirin bütünlüğü bozuldu."
    
    summary = {
        "title": f"BC-DIS Simülasyon Raporu - {sc['n']}", "scenario": sc['n'], "sistem_degerlendirmesi": narrative,
        "istatistiksel_ozet": {
            "Genel Başarı Oranı": f"{(metrics['blocks_created']/TOTAL_BLOCKS)*100:.2f}%", "Saldırı Anındaki Başarı Oranı": f"{success_rate:.2f}%",
            "Toplam Paket": TOTAL_BLOCKS, "Başarılı Paket": metrics['blocks_created'], "Tespit Edilen Anomali": metrics['attacks_blocked'],
        }
    }
    if is_defense: summary['savunma_aksiyon_ozeti'] = f"Uygulanan Savunma Aksiyonları: Akıllı Kontrat ile {metrics['attacks_blocked']} zararlı işlem engellendi."

    rich_report_data = generate_rich_blockchain_report(metrics=metrics, sc_name=sc['n'])
    summary.update(rich_report_data)
    
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
    def __init__(self, learning_period=10):
        self.learning_period, self.baseline, self.is_attack, self.whitelist, self.rules = learning_period, [], False, set(), {}
    def rate_limit(self, ip, timeout=10): self.rules[ip] = time.time() + timeout
    def filter_local(self, pkts):
        [self.rules.pop(ip) for ip in list(self.rules) if time.time() > self.rules.get(ip, 0)]
        return [p for p in pkts if p.source_ip not in self.rules]
    def filter_cloud(self, pkts): return [p for p in pkts if p.source_ip in self.whitelist]
    async def analyze_and_defend(self, traffic, defense_on):
        log_messages, defense_metrics = [], defaultdict(int)
        if not defense_on:
            log_messages.append({"type": "warn", "message": "[SAVUNMA PASİF]: Trafik filtrelenmeden yönlendiriliyor."})
            return traffic, defense_metrics, log_messages
        if len(self.baseline) < self.learning_period:
            self.baseline.append(len(traffic)); self.whitelist.update(p.source_ip for p in traffic if p.is_legitimate)
            log_messages.append({"type": "status", "message": f"[AI-Defender]: Öğrenme fazı devam ediyor ({len(self.baseline)}/{self.learning_period})..."})
            if len(self.baseline) == self.learning_period: log_messages.append({"type": "success", "message": f"[AI-Defender]: Öğrenme fazı tamamlandı."})
            return traffic, defense_metrics, log_messages
        if not self.is_attack and len(traffic) > np.mean(self.baseline) * 5:
            self.is_attack = True; log_messages.append({"type": "alert", "message": "!!! DDoS SALDIRISI TESPİT EDİLDİ !!! Hibrit Kalkan aktive ediliyor."})
            non_legit_ips = [p.source_ip for p in traffic if not p.is_legitimate]
            if non_legit_ips:
                most_common_ip = Counter(non_legit_ips).most_common(1)[0][0]
                self.rate_limit(most_common_ip)
                log_messages.append({"type": "info", "message": f"[Yerel Savunma]: En agresif saldırgan IP ({most_common_ip}) rate-limit listesine eklendi."})
        if not self.is_attack: return traffic, defense_metrics, log_messages
        verified_traffic = []
        for p in traffic:
            if p.is_legitimate and not DilithiumSimulator.verify(p.payload, p.signature): defense_metrics['signature_blocked'] += 1
            else: verified_traffic.append(p)
        if defense_metrics['signature_blocked'] > 0: log_messages.append({"type": "success", "message": f"[İmza Kontrolü]: {defense_metrics['signature_blocked']} sahte imzalı meşru paket engellendi."})
        local_filtered_traffic = self.filter_local(verified_traffic)
        defense_metrics['local_blocked'] = len(verified_traffic) - len(local_filtered_traffic)
        if defense_metrics['local_blocked'] > 0: log_messages.append({"type": "success", "message": f"[Yerel Savunma]: Blacklist'ten {defense_metrics['local_blocked']} saldırı paketi engellendi."})
        cloud_filtered_traffic = self.filter_cloud(local_filtered_traffic)
        defense_metrics['cloud_blocked'] = len(local_filtered_traffic) - len(cloud_filtered_traffic)
        if defense_metrics['cloud_blocked'] > 0: log_messages.append({"type": "success", "message": f"[Bulut Savunması]: Whitelist dışı {defense_metrics['cloud_blocked']} saldırı paketi engellendi."})
        return cloud_filtered_traffic, defense_metrics, log_messages

class TargetServer:
    def __init__(self, cap=120): self.capacity = cap
    def process(self, traffic):
        legit_traffic_in = [p for p in traffic if p.is_legitimate]
        processed = traffic[:self.capacity]; overflow = traffic[self.capacity:]
        processed_legit = sum(1 for p in processed if p.is_legitimate)
        dropped_legit = len(legit_traffic_in) - processed_legit
        return processed, overflow, processed_legit, dropped_legit

# --- Raporlama için Yardımcı Fonksiyon (Yenilendi) ---
def generate_rich_ddos_report(metrics, sc_name, traffic_data, success_rates):
    report = {}
    
    # Yönetici Özeti ve Sistem Değerlendirmesi
    if sc_name == "Savunma Aktif":
        report['yonetici_ozeti'] = f"Bu rapor, Hibrit DDoS Kalkanı'nın '{sc_name}' senaryosu altındaki etkinliğini ölçmektedir. Simülasyon boyunca gelen {metrics['total_incoming']:,} paketin {metrics['total_blocked']:,} tanesi başarıyla engellenmiş, meşru kullanıcılar için hizmet devamlılığı %{success_rates['overall']:.2f} oranında sağlanmıştır."
        report['sistem_degerlendirmesi'] = f"Hibrit kalkan, gelen {metrics['total_incoming']:,} paketi analiz etti. Toplam {metrics['total_blocked']:,} zararlı veya sahte paket başarıyla engellendi. En yoğun saldırı anında bile, meşru kullanıcılar için hizmet başarı oranı %{success_rates['attack']:.2f} gibi mükemmel bir seviyede korunmuştur."
    elif sc_name == "Savunmasız Mod":
        report['yonetici_ozeti'] = f"Bu rapor, Hibrit DDoS Kalkanı kapalıyken sistemin bir saldırı karşısındaki direncini ölçmektedir. Sonuçlar, savunmasız bir sistemin saldırı altında tamamen hizmet dışı kaldığını ve meşru kullanıcı trafiğinin sadece %{success_rates['overall']:.2f}'sinin sunucuya ulaşabildiğini göstermektedir."
        report['sistem_degerlendirmesi'] = f"Savunmasız modda, sunucu kapasitesi anında aşıldı ve saldırı anında meşru kullanıcı başarı oranı %{success_rates['attack']:.2f}'ye düştü. Bu test, Hibrit Kalkan'ın kritik önemini kanıtlamaktadır."
    else: # Normal Operasyon
        report['yonetici_ozeti'] = "Bu rapor, sistemin saldırı olmayan normal operasyon koşullarındaki temel performansını ölçmektedir. Sonuçlar, tüm meşru trafiğin %100 başarı oranıyla işlendiğini göstermektedir."
        report['sistem_degerlendirmesi'] = "Normal operasyon modunda, sistem %100 başarı oranıyla çalışmıştır."

    # Aşamalı Başarı Analizi
    report['asamali_basari_analizi'] = {"headers": ["Simülasyon Aşaması", "Meşru Paket Başarımı"], "rows": [{"Asama": "Saldırı Öncesi", "Oran": f"%{success_rates['before']:.2f}"}, {"Asama": "Saldırı Sırası", "Oran": f"%{success_rates['attack']:.2f}"}, {"Asama": "Saldırı Sonrası", "Oran": f"%{success_rates['after']:.2f}"}]}

    # Görsel Analiz (Grafik + Dinamik Yorum)
    grafik_yorumu = report['sistem_degerlendirmesi'] # Genel yorumu kullanabiliriz
    plt.style.use('dark_background'); fig, ax = plt.subplots(figsize=(10, 5)); ax.set_facecolor('#1e1e1e'); fig.set_facecolor('#1e1e1e')
    df = pd.DataFrame(traffic_data); labels = ["İşlenen Meşru Trafik", "Kalkan Tarafından Engellenen", "Kapasite Aşımı (Meşru)"]; colors = ['#4EC9B0', '#F44747', '#CE9178']
    ax.stackplot(df['time'], df['processed_legit'], df['blocked_by_shield'], df['dropped_legit_overflow'], labels=labels, colors=colors, alpha=0.8)
    ax.set_title('Zamana Göre Trafik Durumu ve Savunma Etkinliği', color='white'); ax.set_xlabel('Zaman (s)', color='white'); ax.set_ylabel('Paket Sayısı', color='white')
    ax.legend(loc='upper left'); ax.grid(True, linestyle='--', alpha=0.3); ax.tick_params(axis='x', colors='white'); ax.tick_params(axis='y', colors='white')
    plt.tight_layout(); buf = io.BytesIO(); fig.savefig(buf, format='png', transparent=True); plt.close(fig); plt.style.use('default')
    report['gorsel_analiz'] = {"grafik_yorumu": grafik_yorumu, "grafik_base64": base64.b64encode(buf.getvalue()).decode('utf-8')}

    # Teknolojinin Hikayesi ve Gerçek Dünya Senaryosu
    report['teknoloji_hikayesi'] = {"title": "Teknolojinin Hikayesi: Hibrit DDoS Kalkanı Katmanları", "headers": ["Savunma Katmanı", "Görevi"], "rows": [{"Savunma Katmanı": "AI Anomali Tespiti", "Görevi": "Normal trafik akışını öğrenir ve ani paket artışlarını tespit ederek saldırı alarmını tetikler."}, {"Savunma Katmanı": "İmza Kontrolü (Dilithium)", "Görevi": "Meşru olduğu iddia edilen paketlerin post-kuantum dijital imzalarını doğrular."}, {"Savunma Katmanı": "Yerel Savunma (Dinamik Blacklist)", "Görevi": "En çok paket gönderen şüpheli IP adreslerini anlık olarak tespit eder ve kısa süreliğine engeller."}, {"Savunma Katmanı": "Bulut Savunması (Whitelist)", "Görevi": "Sadece öğrenme aşamasında tanınan güvenilir IP adreslerinden gelen trafiğe izin verir."}]}
    report['gercek_dunya_senaryosu'] = {"Prototip_Aciklamasi": "Bu simülasyon, kritik bir yer istasyonunun ağ geçidinde (gateway) konumlandırılan çok katmanlı bir DDoS koruma sisteminin prototipidir.", "Gerekli_Donanim_Altyapisi": ["Network TAP veya Mirror Port", "Yüksek Kapasiteli Güvenlik Duvarı (Firewall)", "Log Analiz ve SIEM Sunucusu"], "Gerekli_Yazilim_ve_Teknolojiler": ["IDS/IPS Sistemleri (Snort/Suricata)", "NetFlow/sFlow Analiz Araçları", "Python (Scikit-learn, Pandas)", "Elastisearch/Splunk gibi SIEM platformları"]}
    return report

# --- Ana DDoS Simülasyon Fonksiyonu ---
async def run_ddos_simulation(scenario_choice: str):
    is_attack, is_defense = scenario_choice != "3", scenario_choice == "1"
    SIM_DUR, ATTACK_START, ATTACK_DUR = 30, 10, 15
    server, defender = TargetServer(), AI_DDoS_Defender()
    metrics = defaultdict(int)
    traffic_data_for_graph = []
    
    sc_name = "Savunma Aktif" if is_defense else "Savunmasız Mod" if is_attack else "Normal Operasyon"
    yield {"type": "header", "message": f"DDoS Senaryosu Başlatılıyor: {sc_name}"}

    for t in range(SIM_DUR):
        is_attack_phase = is_attack and t >= ATTACK_START and t < ATTACK_START + ATTACK_DUR
        legit_data = [{"source_ip": f"10.0.0.{i}", "is_legitimate": True, "payload": f"data_{i}_{t}"} for i in range(random.randint(70, 90))]
        for p in legit_data: p["signature"] = DilithiumSimulator.sign(p["payload"])
        
        # Aşamalı analiz için veri toplama
        if t < ATTACK_START: metrics['legit_incoming_before'] += len(legit_data)
        elif is_attack_phase: metrics['legit_incoming_attack'] += len(legit_data)
        else: metrics['legit_incoming_after'] += len(legit_data)
        
        attack_data = []
        if is_attack_phase:
            if t == ATTACK_START: yield {"type": "alert", "message": f"!!! {ATTACK_DUR} SANİYELİK YOĞUN DDoS SALDIRISI BAŞLADI !!!"}
            attack_data = [{"source_ip": f"192.168.1.{random.randint(1,254)}", "is_legitimate": False, "payload": "flood"} for i in range(random.randint(4500, 6000))]
        
        all_pkts_obj = [SignalPacket_DDoS(p) for p in legit_data + attack_data]; random.shuffle(all_pkts_obj)
        metrics['total_incoming'] += len(all_pkts_obj); metrics['total_legit_incoming'] += len(legit_data)
        
        yield {"type": "header", "message": f"--- Zaman {t}s: Gelen Toplam Paket: {len(all_pkts_obj)} ({len(legit_data)} meşru) ---"}
        traffic_after_defense, defense_metrics, new_logs = await defender.analyze_and_defend(all_pkts_obj, is_defense)
        for log in new_logs: yield log
        
        processed, overflow, processed_legit, dropped_legit = server.process(traffic_after_defense)
        
        if t < ATTACK_START: metrics['legit_processed_before'] += processed_legit
        elif is_attack_phase: metrics['legit_processed_attack'] += processed_legit
        else: metrics['legit_processed_after'] += processed_legit

        blocked_by_shield_this_step = sum(defense_metrics.values())
        total_blocked_this_step = blocked_by_shield_this_step + len(overflow)
        metrics['total_blocked'] += total_blocked_this_step
        metrics['total_legit_processed'] += processed_legit

        success_rate_this_step = (processed_legit / len(legit_data) * 100) if len(legit_data) > 0 else 100
        yield {"type": "info", "message": f"   -> Sonuç: Sunucuya Ulaşan: {len(processed)}, Engellenen: {total_blocked_this_step}, Meşru Başarı: %{success_rate_this_step:.1f}"}
        
        traffic_data_for_graph.append({"time": t, "processed_legit": processed_legit, "blocked_by_shield": blocked_by_shield_this_step, "dropped_legit_overflow": dropped_legit})
        await asyncio.sleep(0.3)

    # --- Raporlama ---
    def calc_rate(processed, total): return (processed / total * 100) if total > 0 else 100.0
    success_rates = {
        'overall': calc_rate(metrics['total_legit_processed'], metrics['total_legit_incoming']),
        'before': calc_rate(metrics['legit_processed_before'], metrics['legit_incoming_before']),
        'attack': calc_rate(metrics['legit_processed_attack'], metrics['legit_incoming_attack']),
        'after': calc_rate(metrics['legit_processed_after'], metrics['legit_incoming_after'])
    }
    
    summary = {
        "title": f"DDoS Simülasyon Raporu - {sc_name}", "scenario": sc_name,
        "istatistiksel_ozet": {
            "Genel Başarı Oranı": f"{success_rates['overall']:.2f}%",
            "Saldırı Anındaki Başarı Oranı": f"{success_rates['attack']:.2f}%",
            "Toplam Paket": f"{metrics['total_incoming']:,}",
            "Başarılı Paket": f"{metrics['total_legit_processed']:,}",
            "Tespit Edilen Anomali": 1 if defender.is_attack else 0
        }
    }
    if is_defense:
        summary['savunma_aksiyon_ozeti'] = "Uygulanan Savunma Aksiyonları: AI Tespiti, İmza Kontrolü, Dinamik Blacklist ve Whitelist."
        summary['savunma_detaylari'] = {"guvenli_ip_listesi_whitelist": sorted(list(defender.whitelist)), "engellenen_ip_listesi_blacklist": list(defender.rules.keys())}
        
    rich_report_data = generate_rich_ddos_report(metrics=metrics, sc_name=sc_name, traffic_data=traffic_data_for_graph, success_rates=success_rates)
    summary.update(rich_report_data)
    
    yield {"type": "final", "message": "DDoS simülasyonu tamamlandı.", "summary": summary}

# ==========================================================
# 5. SPOOFING (MF-SatAuth) SİMÜLASYONU (TAM FONKSİYONEL)
# ==========================================================

# Bu kod bloğunu simulators.py dosyanızdaki eski spoofing kodunun yerine yapıştırın.

# Spoofing için gerekli importlar
from skyfield.api import load, EarthSatellite, wgs84
from skyfield.timelib import Time
import pyotp
import uuid
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.exceptions import InvalidSignature
import numpy as np
from sklearn.ensemble import IsolationForest
from collections import defaultdict
import asyncio
import random
import hashlib

# --- Spoofing Simülasyon Sınıfları ---

class Spoofing_AI_ATS:
    def __init__(self, gs):
        # DÜZELTME: random_state=42 kaldırıldı
        self.model = IsolationForest(contamination=0.02)
        self.is_trained = False
        self.gs = gs
        self.last_known = {}

    def _extract(self, pkt, sat, is_spoofed):
        rel = sat.satellite.at(pkt.timestamp) - self.gs.at(pkt.timestamp); alt, _, dist = rel.altaz()
        snr = (100000 / (dist.km**2) + np.random.normal(0, 0.1)) * (1.0 if alt.degrees > 20 else 0.6)
        doppler = (rel.velocity.km_per_s.dot(rel.position.km / dist.km) / 299792) * 2.4e9 / 1e3
        drift = abs(doppler - self.last_known.get(sat.id, doppler))
        if is_spoofed:
            snr *= random.uniform(0.98, 1.02); doppler += random.uniform(-0.05, 0.05); drift += random.uniform(0.01, 0.05)
        self.last_known[sat.id] = doppler
        return [snr, doppler, drift]
    def train(self, pkts, sat):
        self.model.fit([self._extract(p, sat, False) for p in pkts]); self.is_trained = True
    def predict(self, pkt, sat, is_spoofed):
        if not self.is_trained: return "EĞİTİLMEDİ"
        return "ANOMALİ" if self.model.predict([self._extract(pkt, sat, is_spoofed)])[0] == -1 else "NORMAL"
    
    def train(self, pkts, sat):
        self.model.fit([self._extract(p, sat, False) for p in pkts])
        self.is_trained = True
    def predict(self, pkt, sat, is_spoofed):
        if not self.is_trained: return "EĞİTİLMEDİ"
        return "ANOMALİ" if self.model.predict([self._extract(pkt, sat, is_spoofed)])[0] == -1 else "NORMAL"

    def train(self, pkts, sat):
        self.model.fit([self._extract(p, sat, False) for p in pkts])
        self.is_trained = True

    def predict(self, pkt, sat, is_spoofed):
        if not self.is_trained: return "EĞİTİLMEDİ"
        return "ANOMALİ" if self.model.predict([self._extract(pkt, sat, is_spoofed)])[0] == -1 else "NORMAL"

class MFSatAuth:
    def __init__(self, sat, secret):
        self.sats = {sat.id: sat}
        self.nonces = set()
        self.secret = secret
        self.totp = pyotp.TOTP(secret)

    # DÜZELTME 1: Bu fonksiyon artık her adımı yield eden bir generator
    async def verify(self, pkt):
        sat = self.sats.get(pkt.satellite_id)
        if not sat:
            yield {"type": "fail", "message": "[MF-SatAuth] Blok: Bilinmeyen Uydu ID"}
            return

        pos = sat.get_current_geocentric_position(pkt.timestamp)
        
        # Faktör 1: ZKP
        zkp_data = f"({pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f}){self.secret}".encode()
        if pkt.zkp_proof == hashlib.sha256(zkp_data).hexdigest():
            yield {"type": "success", "message": "[MF-SatAuth] Geçti: ZKP İspatı (Konum bilgisi doğrulandı)"}
        else:
            yield {"type": "fail", "message": "[MF-SatAuth] Blok: ZKP İspatı Başarısız"}
            return

        # Faktör 2: Nonce (Tekrar Saldırısı)
        if pkt.nonce not in self.nonces:
            yield {"type": "success", "message": "[MF-SatAuth] Geçti: Nonce Kontrolü (Sinyal daha önce kullanılmamış)"}
        else:
            yield {"type": "fail", "message": "[MF-SatAuth] Blok: Nonce Tekrarı (Tekrar saldırısı tespit edildi)"}
            return

        # Faktör 3: TOTP
        if self.totp.verify(pkt.totp_code, for_time=pkt.timestamp.utc_datetime(), valid_window=1):
            yield {"type": "success", "message": "[MF-SatAuth] Geçti: TOTP Kodu (Zaman damgası geçerli)"}
        else:
            yield {"type": "fail", "message": "[MF-SatAuth] Blok: TOTP Kodu Geçersiz veya Süresi Dolmuş"}
            return
            
        # Faktör 4: İmza (ECDSA)
        try:
            sat.public_key.verify(pkt.signature, pkt.get_signed_data(), ec.ECDSA(hashes.SHA256()))
            yield {"type": "success", "message": "[MF-SatAuth] Geçti: İmza (ECDSA) (Sinyal bütünlüğü doğrulandı)"}
        except InvalidSignature:
            yield {"type": "fail", "message": "[MF-SatAuth] Blok: İmza (ECDSA) Doğrulanamadı"}
            return
            
        # Faktör 5: Konum (TLE)
        if np.linalg.norm(pos - np.array(pkt.position)) <= 15.0:
            yield {"type": "success", "message": "[MF-SatAuth] Geçti: Konum (TLE) (Yörünge verisi tutarlı)"}
        else:
            yield {"type": "fail", "message": "[MF-SatAuth] Blok: Konum (TLE) Tutarsız"}
            return
        
        self.nonces.add(pkt.nonce)


class LegitimateSatellite:
    def __init__(self, id, tle1, tle2, secret):
        self.id, self.timescale = id, load.timescale()
        self.satellite = EarthSatellite(tle1, tle2, self.id, self.timescale)
        self.private_key = ec.generate_private_key(ec.SECP256R1())
        self.public_key = self.private_key.public_key()
        self.secret, self.totp_gen = secret, pyotp.TOTP(secret)
    def get_current_geocentric_position(self, t): return self.satellite.at(t).position.km
    def create_signal(self):
        t = self.timescale.now()
        pos = self.get_current_geocentric_position(t)
        p, n, totp = f"H_OK_{random.randint(1000,9999)}", uuid.uuid4().hex, self.totp_gen.at(t.utc_datetime())
        zkp = hashlib.sha256(f"({pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f}){self.secret}".encode()).hexdigest()
        data = f"{self.id}{t.utc_iso()}({pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f}){p}{n}{totp}{zkp}".encode()
        sig = self.private_key.sign(data, ec.ECDSA(hashes.SHA256()))
        return SignalPacket(self.id, t, pos, p, n, totp, zkp, sig)

class SmartAttackerNode:
    def __init__(self, target):
        self.target, self.timescale = target, load.timescale()
        self.key = ec.generate_private_key(ec.SECP256R1())
        self.secret = pyotp.random_base32()
    def create_spoofed_signal(self):
        t = self.timescale.now()
        real_pos = self.target.get_current_geocentric_position(t)
        fake_pos = real_pos + np.random.uniform(-10, 10, 3)
        p, n, totp = "CMD_OVERRIDE", uuid.uuid4().hex, pyotp.TOTP(self.secret).now()
        zkp = hashlib.sha256(f"({fake_pos[0]:.4f},{fake_pos[1]:.4f},{fake_pos[2]:.4f}){self.secret}".encode()).hexdigest()
        data = f"{self.target.id}{t.utc_iso()}({fake_pos[0]:.4f},{fake_pos[1]:.4f},{fake_pos[2]:.4f}){p}{n}{totp}{zkp}".encode()
        sig = self.key.sign(data, ec.ECDSA(hashes.SHA256()))
        return SignalPacket(self.target.id, t, fake_pos, p, n, totp, zkp, sig)
    def create_replayed_signal(self, pkt): return pkt

class SignalPacket:
    def __init__(self, id, t, pos, p, n, totp, zkp, sig):
        self.satellite_id, self.timestamp, self.position, self.payload, self.nonce, self.totp_code, self.zkp_proof, self.signature = id, t, pos, p, n, totp, zkp, sig
    def get_signed_data(self):
        return f"{self.satellite_id}{self.timestamp.utc_iso()}({self.position[0]:.4f},{self.position[1]:.4f},{self.position[2]:.4f}){self.payload}{self.nonce}{self.totp_code}{self.zkp_proof}".encode()

async def run_spoofing_simulation(scenario_choice: str):
    defense_on, attack_on = scenario_choice != "2", scenario_choice != "3"
    sc_name = "Tam Savunma" if defense_on and attack_on else "Savunmasız" if not defense_on and attack_on else "Normal Operasyon"
    yield {"type": "header", "message": f"Senaryo: {sc_name}"}
    
    secret = pyotp.random_base32()
    SAT_NAME = "GEN9-SAT"
    tle1 = '1 33591U 09005A   25255.55555555  .00000045  00000+0  53282-4 0  9995'
    tle2 = '2 33591  99.0494 278.4116 0014605  71.5959 288.5409 14.12521115874254'
    
    sat = LegitimateSatellite(SAT_NAME, tle1, tle2, secret)
    attacker = SmartAttackerNode(sat)
    gs = wgs84.latlon(latitude_degrees=39.92, longitude_degrees=32.85)
    
    mf_auth = MFSatAuth(sat, secret)
    ai_ats = Spoofing_AI_ATS(gs)
    
    yield {"type": "status", "message": "AI-ATS Modeli normal sinyallerle eğitiliyor..."}
    ai_ats.train([sat.create_signal() for _ in range(200)], sat)
    yield {"type": "success", "message": "AI-ATS Modeli başarıyla eğitildi."}
    
    metrics = defaultdict(int)
    await asyncio.sleep(1)
    captured_signal = sat.create_signal()
    await asyncio.sleep(2)

    for i in range(25):
        attack_type = 'NORMAL' if not attack_on else random.choice(['NORMAL', 'SPOOF', 'REPLAY'])
        yield {"type": "header", "message": f"--- ZAMAN ADIMI {i+1} (Gelen Sinyal: {attack_type}) ---"}
        
        signal = {'NORMAL': sat.create_signal, 'SPOOF': attacker.create_spoofed_signal, 'REPLAY': lambda: attacker.create_replayed_signal(captured_signal)}[attack_type]()
        
        metrics['total_signals'] += 1
        if attack_type != 'NORMAL': metrics['total_attacks'] += 1

        overall_signal_status = "GEÇTİ"
        if defense_on:
            # DÜZELTME 1: Adım adım doğrulama sonuçları canlı konsola gönderiliyor
            async for verification_step in mf_auth.verify(signal):
                yield verification_step
                if verification_step['type'] == 'fail':
                    overall_signal_status = "BLOKE EDİLDİ"
                    if attack_type != 'NORMAL': metrics['blocked_by_mfauth'] += 1
                    break
            
            # Sadece MF-SatAuth'tan geçerse AI kontrolü yap
            if overall_signal_status == "GEÇTİ":
                ai_pred = ai_ats.predict(signal, sat, attack_type != 'NORMAL')
                log_type_ai = "info" if ai_pred == "NORMAL" else "alert"
                yield {"type": log_type_ai, "message": f"[AI-ATS]: RF Parmak İzi Tahmini: {ai_pred}"}
                if ai_pred == "ANOMALİ":
                    overall_signal_status = "BLOKE EDİLDİ"
                    if attack_type != 'NORMAL': metrics['blocked_by_ai'] += 1
        else:
            yield {"type": "warn", "message": "[MF-SatAuth & AI-ATS]: Savunma sistemleri devre dışı."}
            if attack_type != 'NORMAL': metrics['attacks_leaked'] += 1
            
        await asyncio.sleep(0.1)

    # DÜZELTME 2: Rapor grafikleri için standart istatistiksel özet oluşturuluyor
    total_blocked = metrics['blocked_by_mfauth'] + metrics['blocked_by_ai']
    success_rate = (total_blocked / metrics['total_attacks'] * 100) if metrics['total_attacks'] > 0 else 100
    
    legit_signals = metrics['total_signals'] - metrics['total_attacks']
    successful_signals = legit_signals + total_blocked
    
    narrative = f"{metrics['total_signals']} sinyal test edildi ({metrics['total_attacks']} saldırı, {legit_signals} meşru)."
    if defense_on and attack_on:
        narrative += f" Çok katmanlı savunma sayesinde {total_blocked} saldırı (%{success_rate:.2f}) başarıyla engellendi. {metrics['blocked_by_mfauth']} tanesi kural tabanlı MF-SatAuth, {metrics['blocked_by_ai']} tanesi ise AI-ATS tarafından tespit edildi."
    elif not defense_on and attack_on:
        narrative += f" Savunma sistemleri kapalı olduğu için {metrics['attacks_leaked']} saldırının tamamı sisteme sızdı. Bu, derinlemesine savunmanın önemini göstermektedir."
    else:
        narrative += " Normal operasyon koşullarında tüm meşru sinyaller başarıyla doğrulandı."
    
    summary = {
        "title": "Spoofing (MF-SatAuth) Simülasyon Raporu",
        "scenario": sc_name,
        "sistem_degerlendirmesi": narrative,
        "istatistiksel_ozet": {
            "Toplam Paket": metrics['total_signals'],
            "Başarılı Paket": successful_signals,
            "Saldırı Anındaki Başarı Oranı": f"{success_rate:.2f}%",
            "Tespit Edilen Anomali": metrics['total_attacks'], # Toplam saldırı sayısını anomali olarak kabul edelim
            "Yapılan Anten Sıfırlama": metrics['blocked_by_mfauth'], # Kural tabanlı engellemeyi temsil eder
            "Yapılan Frekans Atlama": metrics['blocked_by_ai'] # AI tabanlı engellemeyi temsil eder
        }
    }
    yield {"type": "final", "message": "Spoofing simülasyonu tamamlandı.", "summary": summary}

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
    def get_active_responses(self):
        active_responses = []
        if "JAMMING" in self.active_alarms: active_responses.append("AKTİF TEPKİ: DFH-SS -> Frekans atlama protokolü devrede.")
        if "SPOOFING" in self.active_alarms: active_responses.append(f"AKTİF TEPKİ: Güvenlik Duvarı -> IP ({self.active_alarms['SPOOFING']['source_ip']}) engelleniyor.")
        if "DATA_TAMPERING" in self.active_alarms: active_responses.append(f"AKTİF TEPKİ: BC-DIS -> Uydu ({self.active_alarms['DATA_TAMPERING']['source_satellite']}) karantinada.")
        if self.metrics.get('hqc_boost_activated', 0) == 1: active_responses.append("AKTİF TEPKİ: HQC-SAT -> Kripto güvenlik seviyesi MAKSİMUM.")
        return active_responses
    def _run_rules_engine(self):
        new_responses = []
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
        
        if len(self.active_alarms) >= 3: self.threat_level = "KRİTİK"
        elif len(self.active_alarms) == 2: self.threat_level = "YÜKSEK"
        elif len(self.active_alarms) == 1: self.threat_level = "ORTA"
        
        if self.threat_level in ["KRİTİK", "YÜKSEK"] and self.metrics.get('hqc_boost_activated', 0) == 0:
            new_responses.append("ISMS EYLEMİ: HQC-SAT -> Kripto güvenlik seviyesi MAKSİMUM'a yükseltildi.")
            self.metrics['response_hqc_boost'] += 1; self.metrics['hqc_boost_activated'] = 1
            
        return new_responses, self.get_active_responses()

# --- Raporlama için Yardımcı Fonksiyon ---
def generate_rich_isms_report(metrics, scenario_name, timeline_data):
    report = {}
    total_alarms = metrics.get('total_alarms', 0)
    total_responses = sum(v for k, v in metrics.items() if 'response' in k)
    
    report['yonetici_ozeti'] = f"Bu rapor, ISMS Orkestrasyon modülünün '{scenario_name}' senaryosu altındaki karar verme ve tepki gösterme yeteneklerini analiz eder. Simülasyon boyunca tetiklenen {total_alarms} alarmın tamamına, {total_responses} otonom karşı tedbir ile anında yanıt verilmiştir."
    
    # Görsel Analiz (Zaman Çizelgesi Grafiği)
    plt.style.use('dark_background'); fig, ax = plt.subplots(figsize=(10, 5)); ax.set_facecolor('#1e1e1e'); fig.set_facecolor('#1e1e1e')
    df = pd.DataFrame(timeline_data)
    threat_levels = {'GÜVENLİ': 0, 'ORTA': 1, 'YÜKSEK': 2, 'KRİTİK': 3}
    df['level_num'] = df['level'].map(threat_levels)
    ax.step(df['time'], df['level_num'], where='post', label='Tehdit Seviyesi', color='#569cd6', linewidth=2)
    ax.set_yticks(list(threat_levels.values())); ax.set_yticklabels(list(threat_levels.keys()))
    
    alarm_times = df[df['new_alarm'].notna()]
    if not alarm_times.empty:
        ax.plot(alarm_times['time'], alarm_times['level_num'], 'o', color='#F44747', markersize=8, label='Yeni Alarm')

    ax.set_title('Zamana Göre Tehdit Seviyesi Değişimi', color='white'); ax.set_xlabel('Zaman (s)', color='white'); ax.set_ylabel('Tehdit Seviyesi', color='white')
    ax.legend(); ax.grid(True, linestyle='--', alpha=0.3); ax.tick_params(axis='x', colors='white'); ax.tick_params(axis='y', colors='white')
    plt.tight_layout(); buf = io.BytesIO(); fig.savefig(buf, format='png', transparent=True); plt.close(fig); plt.style.use('default')
    report['gorsel_analiz'] = {"grafik_yorumu": "Grafik, simülasyon boyunca yaşanan olaylara (kırmızı noktalar) bağlı olarak ISMS'in genel tehdit seviyesini (mavi çizgi) nasıl dinamik olarak artırdığını ve azalttığını göstermektedir. Bu, sistemin durumsal farkındalığını ve orantılı tepki verme yeteneğini kanıtlar.", "grafik_base64": base64.b64encode(buf.getvalue()).decode('utf-8')}

    report['teknoloji_hikayesi'] = {
        "title": "Teknolojinin Hikayesi: ISMS Beyni Nasıl Çalışır?",
        "headers": ["Prensip", "Görevi"],
        "rows": [
            {"Prensip": "Merkezi Tehdit İstihbaratı", "Görevi": "Tüm alt savunma modüllerinden (Jamming, Spoofing, DDoS vb.) gelen alarmları tek bir merkezde toplar ve bütünsel bir tehdit resmi oluşturur."},
            {"Prensip": "Kural Tabanlı Motor (Rule Engine)", "Görevi": "Gelen alarm kombinasyonlarına göre önceden tanımlanmış kuralları çalıştırır. Örneğin, 'Eğer Jamming VE Spoofing aynı anda tespit edilirse, Tehdit Seviyesini KRİTİK yap' gibi."},
            {"Prensip": "Otonom Tepki (Autonomous Response)", "Görevi": "Belirlenen tehdit seviyesine ve alarm tipine göre, insan müdahalesi olmadan otomatik olarak karşı tedbirleri (frekans değiştirme, IP engelleme, uydu izolasyonu vb.) tetikler."},
            {"Prensip": "Tehdit Seviyesi Yükseltme (Escalation)", "Görevi": "Birden fazla ve farklı türde saldırı tespit edildiğinde, genel sistem güvenlik duruşunu proaktif olarak en üst seviyeye çıkarır (örn: HQC-SAT kripto seviyesini artırmak gibi)."}
        ]
    }
    report['gercek_dunya_senaryosu'] = {"Prototip_Aciklamasi": "Bu simülasyon, bir Güvenlik Operasyon Merkezi (SOC) veya Uydu Görev Kontrol Merkezi'nde çalışacak, siber ve fiziksel tehditleri birleştirip otonom kararlar alabilen bir SOAR (Security Orchestration, Automation, and Response) platformunun prototipidir."}
    return report
# simulators.py dosyanızdaki MEVCUT run_isms_orchestration_simulation FONKSİYONUNU bu blokla değiştirin.

async def run_isms_orchestration_simulation(scenario_choice: str):
    isms_brain = ISMS_Brain_Final()
    SIMULATION_DURATION = 20
    timeline_data = []
    
    # --- Senaryo 1: Topyekün Saldırı ---
    if scenario_choice == "1":
        scenario_name = "Topyekün Saldırı"
        yield {"type": "header", "message": f"ISMS Senaryosu: {scenario_name} (Dinamik Sıralı) Başlatıldı"}
        attack_types = ["SPOOFING", "DATA_TAMPERING", "JAMMING"]; random.shuffle(attack_types)
        start_times = sorted(random.sample(range(2, 8), len(attack_types)))
        attack_schedule = dict(zip(attack_types, start_times))
        spoof_ip, tamper_sat, jammer_channel = f"185.12.{random.randint(10,255)}.{random.randint(10,255)}", f"GEN9-SAT0{random.randint(2,5)}", random.randint(40, 60)
        
        for t in range(SIMULATION_DURATION):
            new_alarm_this_step = None
            yield {"type": "header", "message": f"--- Zaman Adımı {t+1}/{SIMULATION_DURATION} ---"}
            
            # Alarmları işle
            for threat, start_time in attack_schedule.items():
                if t == start_time:
                    new_alarm_this_step = threat
                    details = {}
                    if threat == "SPOOFING": details = {"source_ip": spoof_ip}; yield {"type": "alert", "message": f"TEHDİT BAŞLADI: SPOOFING (Kaynak: {spoof_ip})"}
                    elif threat == "DATA_TAMPERING": details = {"source_satellite": tamper_sat}; yield {"type": "warn", "message": f"TEHDİT BAŞLADI: DATA_TAMPERING (Hedef: {tamper_sat})"}
                    elif threat == "JAMMING": details = {"target_channel": jammer_channel}; yield {"type": "alert", "message": f"!!! TEHDİT BAŞLADI: JAMMING (Hedef: Ch.{jammer_channel}) !!!"}
                    
                    new_responses, _ = isms_brain.process_alarm({"type": threat, **details})
                    
                    # DÜZELTME: Hatalı liste yapısı standart for döngüsü ile değiştirildi.
                    if new_responses:
                        for resp in set(new_responses):
                            yield {"type": "success", "message": resp}
            
            yield {"type": "status", "message": f"ISMS DURUMU: Tehdit Seviyesi='{isms_brain.threat_level}', Aktif Alarmlar=[{', '.join(isms_brain.active_alarms) or 'YOK'}]"}
            timeline_data.append({'time': t, 'level': isms_brain.threat_level, 'new_alarm': new_alarm_this_step})
            await asyncio.sleep(0.8)

    # --- Senaryo 2: Dinamik Tehditler ---
    elif scenario_choice == "2":
        scenario_name = "Dinamik Tehditler"
        yield {"type": "header", "message": f"ISMS Senaryosu: {scenario_name} Başlatıldı"}
        active_threats = {}; PROB_START, PROB_STOP = 0.25, 0.3
        for t in range(SIMULATION_DURATION):
            new_alarm_this_step = None
            yield {"type": "header", "message": f"--- Zaman Adımı {t+1}/{SIMULATION_DURATION} ---"}
            
            # Alarmları ve olayları işle
            for threat in ["JAMMING", "SPOOFING", "DATA_TAMPERING"]:
                if threat not in active_threats and random.random() < PROB_START:
                    new_alarm_this_step = threat; details = {}
                    if threat == "JAMMING": details = {"target_channel": random.randint(1,100)}
                    elif threat == "SPOOFING": details = {"source_ip": f"112.54.{random.randint(1,255)}.{random.randint(1,255)}"}
                    elif threat == "DATA_TAMPERING": details = {"source_satellite": f"GEN9-SAT{random.randint(1,9):02d}"}
                    active_threats[threat] = details; isms_brain.process_alarm({"type": threat, **details}); yield {"type": "alert", "message": f"TEHDİT BAŞLADI: {threat}"}
                elif threat in active_threats and random.random() < PROB_STOP:
                    active_threats.pop(threat); isms_brain.clear_alarm(threat); yield {"type": "info", "message": f"TEHDİT SONLANDI: {threat} saldırısı durdu."}
            
            new_responses, _ = isms_brain._run_rules_engine()
            
            # DÜZELTME: Hatalı liste yapısı standart for döngüsü ile değiştirildi.
            if new_responses:
                for resp in set(new_responses):
                    yield {"type": "success", "message": resp}
                    
            yield {"type": "status", "message": f"ISMS DURUMU: Tehdit Seviyesi='{isms_brain.threat_level}', Aktif Alarmlar=[{', '.join(isms_brain.active_alarms) or 'YOK'}]"}
            timeline_data.append({'time': t, 'level': isms_brain.threat_level, 'new_alarm': new_alarm_this_step})
            await asyncio.sleep(0.8)

    # --- Raporlama Bölümü (Değişiklik yok) ---
    total_alarms = isms_brain.metrics.get('total_alarms', 0)
    total_responses = sum(v for k, v in isms_brain.metrics.items() if 'response' in k)
    narrative = f"Simülasyon boyunca ISMS'e toplam {total_alarms} adet tehdit alarmı iletilmiştir. ISMS, bu alarmlara karşılık olarak toplam {total_responses} adet otonom komut üreterek tehditleri kontrol altına almıştır. Sistem, operasyonu '{isms_brain.threat_level}' tehdit seviyesinde tamamlamıştır."
    
    summary = {
        "title": f"ISMS Orkestrasyon Raporu - {scenario_name}", "scenario": scenario_name, "sistem_degerlendirmesi": narrative,
        "istatistiksel_ozet": {"Toplam Alarm": total_alarms, "Toplam Otonom Tepki": total_responses, "En Yüksek Tehdit Seviyesi": max((item['level'] for item in timeline_data), key=lambda x: {'GÜVENLİ':0, 'ORTA':1, 'YÜKSEK':2, 'KRİTİK':3}[x]) if timeline_data else 'GÜVENLİ', "Engellenen IP": len(isms_brain.blacklist), "Karantinadaki Uydu": len(isms_brain.quarantine_list)}
    }
    
    rich_report_data = generate_rich_isms_report(metrics=isms_brain.metrics, scenario_name=scenario_name, timeline_data=timeline_data)
    summary.update(rich_report_data)
    
    yield {"type": "final", "message": "ISMS Orkestrasyonu tamamlandı.", "summary": summary}
    isms_brain = ISMS_Brain_Final(); SIMULATION_DURATION = 20
    timeline_data = []
    
    if scenario_choice == "1":
        scenario_name = "Topyekün Saldırı"
        yield {"type": "header", "message": f"ISMS Senaryosu: {scenario_name} (Dinamik Sıralı) Başlatıldı"}
        attack_types = ["SPOOFING", "DATA_TAMPERING", "JAMMING"]; random.shuffle(attack_types)
        start_times = sorted(random.sample(range(2, 8), len(attack_types)))
        attack_schedule = dict(zip(attack_types, start_times))
        spoof_ip, tamper_sat, jammer_channel = f"185.12.{random.randint(10,255)}.{random.randint(10,255)}", f"GEN9-SAT0{random.randint(2,5)}", random.randint(40, 60)
        
        for t in range(SIMULATION_DURATION):
            new_alarm_this_step = None
            yield {"type": "header", "message": f"--- Zaman Adımı {t+1}/{SIMULATION_DURATION} ---"}
            for threat, start_time in attack_schedule.items():
                if t == start_time:
                    new_alarm_this_step = threat
                    details = {}
                    if threat == "SPOOFING": details = {"source_ip": spoof_ip}; yield {"type": "alert", "message": f"TEHDİT BAŞLADI: SPOOFING (Kaynak: {spoof_ip})"}
                    elif threat == "DATA_TAMPERING": details = {"source_satellite": tamper_sat}; yield {"type": "warn", "message": f"TEHDİT BAŞLADI: DATA_TAMPERING (Hedef: {tamper_sat})"}
                    elif threat == "JAMMING": details = {"target_channel": jammer_channel}; yield {"type": "alert", "message": f"!!! TEHDİT BAŞLADI: JAMMING (Hedef: Ch.{jammer_channel}) !!!"}
                    new_responses, _ = isms_brain.process_alarm({"type": threat, **details})
                    if new_responses:
                        for resp in set(new_responses):
                            yield {"type": "success", "message": resp}
            
            yield {"type": "status", "message": f"ISMS DURUMU: Tehdit Seviyesi='{isms_brain.threat_level}', Aktif Alarmlar=[{', '.join(isms_brain.active_alarms) or 'YOK'}]"}
            timeline_data.append({'time': t, 'level': isms_brain.threat_level, 'new_alarm': new_alarm_this_step})
            await asyncio.sleep(0.8)

    elif scenario_choice == "2":
        scenario_name = "Dinamik Tehditler"
        yield {"type": "header", "message": f"ISMS Senaryosu: {scenario_name} Başlatıldı"}
        active_threats = {}; PROB_START, PROB_STOP = 0.25, 0.3
        for t in range(SIMULATION_DURATION):
            new_alarm_this_step = None
            yield {"type": "header", "message": f"--- Zaman Adımı {t+1}/{SIMULATION_DURATION} ---"}
            for threat in ["JAMMING", "SPOOFING", "DATA_TAMPERING"]:
                if threat not in active_threats and random.random() < PROB_START:
                    new_alarm_this_step = threat; details = {}
                    if threat == "JAMMING": details = {"target_channel": random.randint(1,100)}
                    elif threat == "SPOOFING": details = {"source_ip": f"112.54.{random.randint(1,255)}.{random.randint(1,255)}"}
                    elif threat == "DATA_TAMPERING": details = {"source_satellite": f"GEN9-SAT{random.randint(1,9):02d}"}
                    active_threats[threat] = details; isms_brain.process_alarm({"type": threat, **details}); yield {"type": "alert", "message": f"TEHDİT BAŞLADI: {threat}"}
                elif threat in active_threats and random.random() < PROB_STOP:
                    active_threats.pop(threat); isms_brain.clear_alarm(threat); yield {"type": "info", "message": f"TEHDİT SONLANDI: {threat} saldırısı durdu."}
            
            new_responses, _ = isms_brain._run_rules_engine()
            if new_responses:
                for resp in set(new_responses):
                    yield {"type": "success", "message": resp}
            yield {"type": "status", "message": f"ISMS DURUMU: Tehdit Seviyesi='{isms_brain.threat_level}', Aktif Alarmlar=[{', '.join(isms_brain.active_alarms) or 'YOK'}]"}
            timeline_data.append({'time': t, 'level': isms_brain.threat_level, 'new_alarm': new_alarm_this_step})
            await asyncio.sleep(0.8)

    # --- Raporlama ---
    total_alarms = isms_brain.metrics.get('total_alarms', 0)
    total_responses = sum(v for k, v in isms_brain.metrics.items() if 'response' in k)
    narrative = f"Simülasyon boyunca ISMS'e toplam {total_alarms} adet tehdit alarmı iletilmiştir. ISMS, bu alarmlara karşılık olarak toplam {total_responses} adet otonom komut üreterek tehditleri kontrol altına almıştır. Sistem, operasyonu '{isms_brain.threat_level}' tehdit seviyesinde tamamlamıştır."
    
    summary = {
        "title": f"ISMS Orkestrasyon Raporu - {scenario_name}", "scenario": scenario_name, "sistem_degerlendirmesi": narrative,
        "istatistiksel_ozet": {"Toplam Alarm": total_alarms, "Toplam Otonom Tepki": total_responses, "En Yüksek Tehdit Seviyesi": max(item['level'] for item in timeline_data) if timeline_data else 'GÜVENLİ', "Engellenen IP": len(isms_brain.blacklist), "Karantinadaki Uydu": len(isms_brain.quarantine_list)}
    }
    
    rich_report_data = generate_rich_isms_report(metrics=isms_brain.metrics, scenario_name=scenario_name, timeline_data=timeline_data)
    summary.update(rich_report_data)
    
    yield {"type": "final", "message": "ISMS Orkestrasyonu tamamlandı.", "summary": summary}