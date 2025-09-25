class FallDetectionApp {
    constructor() {
        this.socket = null;
        this.isMonitoring = false;
        this.alerts = [];
        this.currentSettings = {
            fall_vdrop: 0.10,
            fall_angle_deg: 55.0,
            fall_aspect: 1.00,
            fall_dwell_s: 0.3
        };
        
        this.init();
    }
    
    init() {
        this.initSocket();
        this.bindEvents();
        this.updateStatus();
        this.startTimeUpdate();
    }
    
    initSocket() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            console.log('WebSocket 已連接');
            this.updateConnectionStatus(true);
        });
        
        this.socket.on('disconnect', () => {
            console.log('WebSocket 已斷開');
            this.updateConnectionStatus(false);
        });
        
        this.socket.on('video_frame', (data) => {
            this.updateVideoFrame(data);
        });
        
        this.socket.on('fall_detected', (data) => {
            this.handleFallAlert(data);
        });
        
        this.socket.on('connected', (data) => {
            console.log('收到連接確認:', data);
        });
    }
    
    bindEvents() {
        // 控制按鈕
        document.getElementById('start-btn').addEventListener('click', () => {
            this.startMonitoring();
        });
        
        document.getElementById('stop-btn').addEventListener('click', () => {
            this.stopMonitoring();
        });
        
        document.getElementById('settings-btn').addEventListener('click', () => {
            this.showSettings();
        });
        
        // 設定對話框
        document.getElementById('save-settings').addEventListener('click', () => {
            this.saveSettings();
        });
        
        document.getElementById('cancel-settings').addEventListener('click', () => {
            this.hideSettings();
        });
        
        document.querySelector('.close').addEventListener('click', () => {
            this.hideSettings();
        });
        
        // 點擊對話框外部關閉
        document.getElementById('settings-modal').addEventListener('click', (e) => {
            if (e.target.id === 'settings-modal') {
                this.hideSettings();
            }
        });
        
        // ESC 鍵關閉對話框
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.hideSettings();
            }
        });
    }
    
    async updateStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            this.updateTelegramStatus(data.telegram_configured);
            
            if (data.is_monitoring && !this.isMonitoring) {
                this.setMonitoringState(true);
            } else if (!data.is_monitoring && this.isMonitoring) {
                this.setMonitoringState(false);
            }
        } catch (error) {
            console.error('更新狀態失敗:', error);
        }
    }
    
    updateConnectionStatus(connected) {
        const statusIcon = document.getElementById('connection-status');
        const statusText = document.getElementById('connection-text');
        
        if (connected) {
            statusIcon.className = 'fas fa-circle connected';
            statusText.textContent = '已連接';
        } else {
            statusIcon.className = 'fas fa-circle disconnected';
            statusText.textContent = '未連接';
        }
    }
    
    updateTelegramStatus(configured) {
        const statusIcon = document.getElementById('telegram-status');
        const statusText = document.getElementById('telegram-text');
        
        if (configured) {
            statusIcon.className = 'fas fa-telegram-plane configured';
            statusText.textContent = 'Telegram已配置';
        } else {
            statusIcon.className = 'fas fa-telegram-plane not-configured';
            statusText.textContent = 'Telegram未配置';
        }
    }
    
    async startMonitoring() {
        const source = document.getElementById('video-source').value;
        const model = document.getElementById('model-select').value;
        
        this.showLoading();
        
        try {
            const response = await fetch('/api/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    source: source,
                    model: model
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.setMonitoringState(true);
                this.showNotification('監控已開始', 'success');
            } else {
                this.showNotification('無法啟動監控，請檢查設備連接', 'error');
            }
        } catch (error) {
            console.error('啟動監控失敗:', error);
            this.showNotification('啟動監控失敗', 'error');
        } finally {
            this.hideLoading();
        }
    }
    
    async stopMonitoring() {
        try {
            const response = await fetch('/api/stop', {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.setMonitoringState(false);
                this.showNotification('監控已停止', 'info');
            }
        } catch (error) {
            console.error('停止監控失敗:', error);
            this.showNotification('停止監控失敗', 'error');
        }
    }
    
    setMonitoringState(monitoring) {
        this.isMonitoring = monitoring;
        
        const startBtn = document.getElementById('start-btn');
        const stopBtn = document.getElementById('stop-btn');
        const videoFeed = document.getElementById('video-feed');
        const videoPlaceholder = document.getElementById('video-placeholder');
        const container = document.querySelector('.container');
        
        if (monitoring) {
            startBtn.disabled = true;
            stopBtn.disabled = false;
            videoPlaceholder.style.display = 'none';
            videoFeed.style.display = 'block';
            container.classList.add('monitoring-active');
        } else {
            startBtn.disabled = false;
            stopBtn.disabled = true;
            videoFeed.style.display = 'none';
            videoPlaceholder.style.display = 'flex';
            container.classList.remove('monitoring-active');
            this.clearVideoFrame();
        }
    }
    
    updateVideoFrame(data) {
        const videoFeed = document.getElementById('video-feed');
        const personCount = document.getElementById('person-count');
        
        if (data.image) {
            videoFeed.src = `data:image/jpeg;base64,${data.image}`;
        }
        
        if (data.person_count !== undefined) {
            personCount.textContent = data.person_count;
        }
        
        if (data.fall_detected) {
            videoFeed.classList.add('fall-detected');
            setTimeout(() => {
                videoFeed.classList.remove('fall-detected');
            }, 2000);
        }
    }
    
    clearVideoFrame() {
        const videoFeed = document.getElementById('video-feed');
        const personCount = document.getElementById('person-count');
        
        videoFeed.src = '';
        personCount.textContent = '0';
    }
    
    handleFallAlert(data) {
        const alert = {
            id: Date.now(),
            timestamp: data.timestamp,
            personId: data.person_id,
            bbox: data.bbox,
            message: `檢測到第 ${data.person_id + 1} 人跌倒`
        };
        
        this.alerts.unshift(alert);
        this.updateAlertsPanel();
        this.showNotification(alert.message, 'critical');
        
        // 觸發視覺警報效果
        document.body.classList.add('fall-alert-active');
        setTimeout(() => {
            document.body.classList.remove('fall-alert-active');
        }, 3000);
        
        // 播放警報音 (可選)
        this.playAlertSound();
    }
    
    updateAlertsPanel() {
        const alertsList = document.getElementById('alerts-list');
        
        if (this.alerts.length === 0) {
            alertsList.innerHTML = '<div class="no-alerts">目前無警報記錄</div>';
            return;
        }
        
        alertsList.innerHTML = this.alerts.map(alert => `
            <div class="alert-item critical">
                <div class="alert-time">${this.formatTime(alert.timestamp)}</div>
                <div class="alert-message">${alert.message}</div>
                <div class="alert-details">位置: 監控區域 | 已發送 Telegram 通知</div>
            </div>
        `).join('');
    }
    
    showSettings() {
        // 載入當前設定值
        document.getElementById('fall-vdrop').value = this.currentSettings.fall_vdrop;
        document.getElementById('fall-angle').value = this.currentSettings.fall_angle_deg;
        document.getElementById('fall-aspect').value = this.currentSettings.fall_aspect;
        document.getElementById('fall-dwell').value = this.currentSettings.fall_dwell_s;
        
        document.getElementById('settings-modal').style.display = 'block';
    }
    
    hideSettings() {
        document.getElementById('settings-modal').style.display = 'none';
    }
    
    async saveSettings() {
        const settings = {
            fall_vdrop: parseFloat(document.getElementById('fall-vdrop').value),
            fall_angle_deg: parseFloat(document.getElementById('fall-angle').value),
            fall_aspect: parseFloat(document.getElementById('fall-aspect').value),
            fall_dwell_s: parseFloat(document.getElementById('fall-dwell').value)
        };
        
        try {
            const response = await fetch('/api/settings', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(settings)
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.currentSettings = { ...settings };
                this.hideSettings();
                this.showNotification('設定已儲存', 'success');
            } else {
                this.showNotification('設定儲存失敗', 'error');
            }
        } catch (error) {
            console.error('儲存設定失敗:', error);
            this.showNotification('設定儲存失敗', 'error');
        }
    }
    
    startTimeUpdate() {
        setInterval(() => {
            const now = new Date();
            const timeStr = now.toLocaleTimeString('zh-TW');
            document.getElementById('current-time').textContent = timeStr;
        }, 1000);
    }
    
    formatTime(timestamp) {
        const date = new Date(timestamp);
        return date.toLocaleString('zh-TW');
    }
    
    showLoading() {
        document.getElementById('loading-overlay').style.display = 'flex';
    }
    
    hideLoading() {
        document.getElementById('loading-overlay').style.display = 'none';
    }
    
    showNotification(message, type = 'info') {
        // 創建通知元素
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas ${this.getNotificationIcon(type)}"></i>
                <span>${message}</span>
            </div>
        `;
        
        // 添加樣式
        Object.assign(notification.style, {
            position: 'fixed',
            top: '20px',
            right: '20px',
            padding: '12px 20px',
            borderRadius: '8px',
            color: 'white',
            fontSize: '14px',
            fontWeight: '500',
            zIndex: '1000',
            transform: 'translateX(100%)',
            transition: 'transform 0.3s ease',
            backgroundColor: this.getNotificationColor(type),
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)'
        });
        
        document.body.appendChild(notification);
        
        // 動畫顯示
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 100);
        
        // 自動移除
        setTimeout(() => {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 4000);
    }
    
    getNotificationIcon(type) {
        const icons = {
            success: 'fa-check-circle',
            error: 'fa-exclamation-circle',
            warning: 'fa-exclamation-triangle',
            info: 'fa-info-circle',
            critical: 'fa-skull-crossbones'
        };
        return icons[type] || icons.info;
    }
    
    getNotificationColor(type) {
        const colors = {
            success: '#27ae60',
            error: '#e74c3c',
            warning: '#f39c12',
            info: '#3498db',
            critical: '#c0392b'
        };
        return colors[type] || colors.info;
    }
    
    playAlertSound() {
        // 創建警報音效 (可選功能)
        try {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();
            
            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);
            
            oscillator.frequency.setValueAtTime(800, audioContext.currentTime);
            oscillator.frequency.setValueAtTime(600, audioContext.currentTime + 0.1);
            oscillator.frequency.setValueAtTime(800, audioContext.currentTime + 0.2);
            
            gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);
            
            oscillator.start(audioContext.currentTime);
            oscillator.stop(audioContext.currentTime + 0.3);
        } catch (error) {
            console.log('無法播放警報音效:', error);
        }
    }
}

// 啟動應用程式
document.addEventListener('DOMContentLoaded', () => {
    window.fallApp = new FallDetectionApp();
});