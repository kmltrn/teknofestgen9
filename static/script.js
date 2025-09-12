document.addEventListener("DOMContentLoaded", function() {
    const logBox = document.getElementById("log-box");
    const reportContent = document.getElementById("report-content");
    const allSimButtons = document.querySelectorAll('#controls button');
    const liveChartsContainer = document.getElementById('live-charts-container');
    let currentWebSocket = null;
    let isSimulationRunning = false;
    let spectrumChart = null;
    let snrChart = null;

    function generateReport(summary) {
        reportContent.innerHTML = '';
        const title = document.createElement('h3');
        title.textContent = summary.title || "Sonuç Raporu";
        reportContent.appendChild(title);
        const scenario = document.createElement('p');
        scenario.innerHTML = `<strong>Test Edilen Senaryo:</strong> ${summary.scenario}`;
        reportContent.appendChild(scenario);

        if (summary.statistics) {
            const statsTitle = document.createElement('h4');
            statsTitle.textContent = "İstatistiksel Özet";
            reportContent.appendChild(statsTitle);
            const statsList = document.createElement('ul');
            statsList.className = 'statistics-list';
            for (const [key, value] of Object.entries(summary.statistics)) {
                let formattedKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                const listItem = document.createElement('li');
                listItem.innerHTML = `<strong>${formattedKey}:</strong> ${value}`;
                statsList.appendChild(listItem);
            }
            reportContent.appendChild(statsList);
        }
        if (summary.final_state) {
            const stateTitle = document.createElement('h4');
            stateTitle.textContent = "Nihai Sistem Durumu ve Eylem Listesi";
            reportContent.appendChild(stateTitle);
            const stateList = document.createElement('ul');
            stateList.className = 'state-list';
            const blacklist = summary.final_state.blacklist;
            const quarantine = summary.final_state.quarantine_list;
            stateList.innerHTML = `<li><strong>Engellenen IP Listesi:</strong> ${blacklist.length > 0 ? blacklist.join(', ') : 'Yok'}</li>
                                 <li><strong>Karantinadaki Uydular:</strong> ${quarantine.length > 0 ? quarantine.join(', ') : 'Yok'}</li>`;
            reportContent.appendChild(stateList);
        }
        if (summary.narrative) {
            const narrativeTitle = document.createElement('h4');
            narrativeTitle.textContent = "Sistem Değerlendirmesi";
            reportContent.appendChild(narrativeTitle);
            const narrative = document.createElement('p');
            narrative.innerHTML = summary.narrative.replace(/\n/g, '<br>');
            reportContent.appendChild(narrative);
        }
    }

    function initializeCharts() {
        if (spectrumChart) spectrumChart.destroy();
        if (snrChart) snrChart.destroy();

        const spectrumCtx = document.getElementById('spectrumChart').getContext('2d');
        spectrumChart = new Chart(spectrumCtx, {
            type: 'bar',
            data: {
                labels: Array.from({length: 100}, (_, i) => i),
                datasets: [{
                    label: 'RF Spektrum Gücü (dBm)', data: [],
                    backgroundColor: 'rgba(0, 240, 192, 0.6)', borderWidth: 1
                }]
            },
            options: { animation: false, scales: { y: { beginAtZero: false, min: -100, max: 50, ticks: { color: '#e0e0e0' }}, x: { ticks: { color: '#e0e0e0' }} }, plugins: { legend: { labels: { color: '#e0e0e0' } } } }
        });
        const snrCtx = document.getElementById('snrChart').getContext('2d');
        snrChart = new Chart(snrCtx, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Net SNR (dB)', data: [], borderColor: '#8be9fd', backgroundColor: 'rgba(139, 233, 253, 0.2)', fill: true, tension: 0.1 }] },
            options: { animation: false, scales: { y: { beginAtZero: false, ticks: { color: '#e0e0e0' }}, x: { ticks: { color: '#e0e0e0' }} }, plugins: { legend: { labels: { color: '#e0e0e0' } } } }
        });
    }

    function updateCharts(data) {
        if (!spectrumChart || !snrChart) return;
        spectrumChart.data.datasets[0].data = data.spectrum_data;
        const colors = Array(100).fill('rgba(0, 240, 192, 0.6)');
        if (data.jammer_active && data.target_channel !== -1) { colors[data.target_channel] = 'rgba(255, 85, 85, 0.8)'; }
        if (data.current_channel !== -1) { colors[data.current_channel] = 'rgba(80, 250, 123, 0.9)'; }
        spectrumChart.data.datasets[0].backgroundColor = colors;
        spectrumChart.update('none');
        snrChart.data.labels.push(data.snr_point.x);
        snrChart.data.datasets[0].data.push(data.snr_point.y);
        snrChart.update('none');
    }

    function connectWebSocket(endpointUrl) {
        if (isSimulationRunning) {
            const line = document.createElement('div');
            line.className = 'log-line log-type-warn';
            line.textContent = '!!! Mevcut bir simülasyon çalışırken yenisi başlatılamaz. Lütfen bekleyin.';
            logBox.appendChild(line);
            return;
        }
        if (currentWebSocket) { currentWebSocket.close(); }
        logBox.innerHTML = '>>> Sunucu ile bağlantı kuruluyor...';
        reportContent.innerHTML = '<p>Simülasyonun tamamlanması bekleniyor...</p>';
        isSimulationRunning = true;
        allSimButtons.forEach(button => button.disabled = true);
        
        if (endpointUrl.includes("/ws/jamming/")) {
            liveChartsContainer.style.display = 'flex';
            initializeCharts();
        } else {
            liveChartsContainer.style.display = 'none';
        }

        const ws = new WebSocket(`ws://127.0.0.1:8000${endpointUrl}`);
        currentWebSocket = ws;
        ws.onopen = function(event) {
            logBox.innerHTML += '\n<div class="log-line log-type-status">>>> Bağlantı kuruldu. Simülasyon başlatılıyor...</div>';
        };
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            if (data.type === 'graph_update') {
                updateCharts(data);
                return;
            }
            const line = document.createElement('div');
            if (data.type !== 'final') {
                let typeClass = 'log-line log-type-' + data.type;
                if (data.message && data.message.includes("GÜVENLİK") && data.message.includes("Devre Dışı")) { typeClass = 'log-line log-type-warn'; }
                if (data.type === 'header') { typeClass = 'log-line log-type-header'; }
                line.className = typeClass;
                line.innerHTML = data.message.replace(/\n/g, '<br>');
                logBox.appendChild(line);
            } else {
                line.className = 'log-line log-type-final';
                line.textContent = data.message;
                logBox.appendChild(line);
                if (data.summary) { generateReport(data.summary); }
                isSimulationRunning = false;
                allSimButtons.forEach(button => button.disabled = false);
                currentWebSocket.close();
            }
            logBox.scrollTop = logBox.scrollHeight;
        };
        ws.onclose = function(event) {
            const line = document.createElement('div');
            if (isSimulationRunning) { line.className = 'log-line log-type-error'; line.textContent = '!!! Hata: Sunucu ile bağlantı beklenmedik şekilde kesildi.'; } 
            else { line.className = 'log-line log-type-status'; line.textContent = '<<< Sunucu ile WebSocket bağlantısı kapandı.'; }
            logBox.appendChild(line);
            logBox.scrollTop = logBox.scrollHeight;
            isSimulationRunning = false;
            allSimButtons.forEach(button => button.disabled = false);
        };
        ws.onerror = function(event) {
            const line = document.createElement('div');
            line.className = 'log-line log-type-error';
            line.textContent = '!!! Bağlantı Hatası: Sunucuya bağlanılamadı. Lütfen sunucunun çalıştığından emin olun.';
            logBox.appendChild(line);
            isSimulationRunning = false;
            allSimButtons.forEach(button => button.disabled = false);
            if (currentWebSocket) currentWebSocket.close();
        };
    }

    // --- OLAY DİNLEYİCİLERİ ---
    document.getElementById("startIsmsBtn1").addEventListener("click", () => connectWebSocket("/ws/isms/1"));
    document.getElementById("startIsmsBtn2").addEventListener("click", () => connectWebSocket("/ws/isms/2"));
    document.getElementById("startJammingBtn1").addEventListener("click", () => connectWebSocket("/ws/jamming/1"));
    document.getElementById("startJammingBtn2").addEventListener("click", () => connectWebSocket("/ws/jamming/2"));
    document.getElementById("startJammingBtn3").addEventListener("click", () => connectWebSocket("/ws/jamming/3"));
    document.getElementById("startHqcBtn1").addEventListener("click", () => connectWebSocket("/ws/hqc_sat/1"));
    document.getElementById("startHqcBtn2").addEventListener("click", () => connectWebSocket("/ws/hqc_sat/2"));
    document.getElementById("startHqcBtn3").addEventListener("click", () => connectWebSocket("/ws/hqc_sat/3"));
    document.getElementById("startBlockchainBtn1").addEventListener("click", () => connectWebSocket("/ws/blockchain/1"));
    document.getElementById("startBlockchainBtn2").addEventListener("click", () => connectWebSocket("/ws/blockchain/2"));
    document.getElementById("startBlockchainBtn3").addEventListener("click", () => connectWebSocket("/ws/blockchain/3"));
    document.getElementById("startDdosBtn1").addEventListener("click", () => connectWebSocket("/ws/ddos/1"));
    document.getElementById("startDdosBtn2").addEventListener("click", () => connectWebSocket("/ws/ddos/2"));
    document.getElementById("startDdosBtn3").addEventListener("click", () => connectWebSocket("/ws/ddos/3"));
    document.getElementById("startSpoofingBtn1").addEventListener("click", () => connectWebSocket("/ws/spoofing/1"));
    document.getElementById("startSpoofingBtn2").addEventListener("click", () => connectWebSocket("/ws/spoofing/2"));
    document.getElementById("startSpoofingBtn3").addEventListener("click", () => connectWebSocket("/ws/spoofing/3"));
});