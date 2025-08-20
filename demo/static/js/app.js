// FLUX Kontext LoRA Demo JavaScript

class FluxDemo {
    constructor() {
        this.baseUrl = '';
        this.uploadedControlImage = null;
        this.init();
    }

    init() {
        this.bindEvents();
        this.updateStatus();
        this.addLog('初始化', 'FLUX Kontext LoRA Demo 已启动');
    }

    bindEvents() {
        // 模型控制按钮
        document.getElementById('load-model-btn').addEventListener('click', () => this.loadModel());
        document.getElementById('unload-model-btn').addEventListener('click', () => this.unloadModel());
        document.getElementById('refresh-status-btn').addEventListener('click', () => this.updateStatus());

        // 表单提交
        document.getElementById('batch-form').addEventListener('submit', (e) => this.handleBatchInference(e));
        document.getElementById('single-form').addEventListener('submit', (e) => this.handleSingleInference(e));

        // 文件上传
        document.getElementById('control-image').addEventListener('change', (e) => this.handleFileUpload(e));

        // 定期更新状态
        setInterval(() => this.updateStatus(), 30000); // 每30秒更新一次
    }

    async updateStatus() {
        try {
            const response = await fetch('/api/status');
            const status = await response.json();

            this.updateStatusDisplay(status);
        } catch (error) {
            console.error('更新状态失败:', error);
            this.addLog('错误', '更新状态失败: ' + error.message, 'error');
        }
    }

    updateStatusDisplay(status) {
        const modelStatusEl = document.getElementById('model-status');
        const currentLoraEl = document.getElementById('current-lora');
        const deviceInfoEl = document.getElementById('device-info');
        const memoryInfoEl = document.getElementById('memory-info');
        const loadBtn = document.getElementById('load-model-btn');
        const unloadBtn = document.getElementById('unload-model-btn');

        // 更新模型状态
        if (status.is_loaded) {
            modelStatusEl.textContent = '已加载';
            modelStatusEl.className = 'status-value status-loaded';
            loadBtn.disabled = false;
            unloadBtn.disabled = false;
            loadBtn.innerHTML = '<i class="fas fa-sync"></i> 重新加载';
        } else {
            modelStatusEl.textContent = '未加载';
            modelStatusEl.className = 'status-value status-unloaded';
            loadBtn.disabled = false;
            unloadBtn.disabled = true;
            loadBtn.innerHTML = '<i class="fas fa-download"></i> 加载模型';
        }

        // 更新LoRA路径
        if (status.current_lora_path) {
            const filename = status.current_lora_path.split('/').pop();
            currentLoraEl.textContent = filename;
        } else {
            currentLoraEl.textContent = '无';
        }

        // 更新设备信息
        deviceInfoEl.textContent = status.device || '-';

        // 更新内存信息
        if (status.memory_info) {
            memoryInfoEl.textContent = `已用: ${status.memory_info.allocated}, 保留: ${status.memory_info.reserved}`;
        } else {
            memoryInfoEl.textContent = '-';
        }
    }

    async loadModel() {
        const loraPath = document.getElementById('lora-path').value.trim();
        if (!loraPath) {
            alert('请输入LoRA checkpoint路径');
            return;
        }

        this.showLoading('加载模型中...');
        this.addLog('加载', `开始加载模型: ${loraPath.split('/').pop()}`);

        try {
            const response = await fetch('/api/load_model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    lora_path: loraPath
                })
            });

            const result = await response.json();

            if (response.ok) {
                this.addLog('成功', result.message, 'success');
                this.updateStatusDisplay(result.status);
            } else {
                throw new Error(result.detail || '加载失败');
            }
        } catch (error) {
            this.addLog('错误', '模型加载失败: ' + error.message, 'error');
            alert('模型加载失败: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    async unloadModel() {
        if (!confirm('确定要卸载模型吗？')) {
            return;
        }

        this.showLoading('卸载模型中...');
        this.addLog('卸载', '开始卸载模型');

        try {
            const response = await fetch('/api/unload_model', {
                method: 'POST'
            });

            const result = await response.json();

            if (response.ok) {
                this.addLog('成功', result.message, 'success');
                this.updateStatusDisplay(result.status);
            } else {
                throw new Error(result.detail || '卸载失败');
            }
        } catch (error) {
            this.addLog('错误', '模型卸载失败: ' + error.message, 'error');
            alert('模型卸载失败: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    async handleBatchInference(e) {
        e.preventDefault();

        const formData = {
            folder_path: document.getElementById('batch-folder').value,
            max_samples: parseInt(document.getElementById('max-samples').value),
            seed: parseInt(document.getElementById('batch-seed').value),
            guidance_scale: parseFloat(document.getElementById('batch-guidance').value),
            num_inference_steps: parseInt(document.getElementById('batch-steps').value)
        };

        this.showLoading('批量推理中...');
        this.addLog('推理', `开始批量推理: ${formData.max_samples} 张图像`);

        try {
            const response = await fetch('/api/batch_inference', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });

            const result = await response.json();

            if (response.ok) {
                this.addLog('成功', result.message, 'success');
                this.displayBatchResults(result.results);
            } else {
                throw new Error(result.detail || '推理失败');
            }
        } catch (error) {
            this.addLog('错误', '批量推理失败: ' + error.message, 'error');
            alert('批量推理失败: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    async handleSingleInference(e) {
        e.preventDefault();

        const formData = {
            prompt: document.getElementById('prompt').value,
            width: parseInt(document.getElementById('single-width').value),
            height: parseInt(document.getElementById('single-height').value),
            seed: parseInt(document.getElementById('single-seed').value),
            guidance_scale: parseFloat(document.getElementById('single-guidance').value),
            num_inference_steps: parseInt(document.getElementById('single-steps').value)
        };

        if (this.uploadedControlImage) {
            formData.control_image_path = this.uploadedControlImage;
        }

        this.showLoading('单个推理中...');
        this.addLog('推理', '开始单个推理');

        try {
            const response = await fetch('/api/single_inference', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });

            const result = await response.json();

            if (response.ok) {
                this.addLog('成功', result.message, 'success');
                this.displaySingleResult(result);
            } else {
                throw new Error(result.detail || '推理失败');
            }
        } catch (error) {
            this.addLog('错误', '单个推理失败: ' + error.message, 'error');
            alert('单个推理失败: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    async handleFileUpload(e) {
        const file = e.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        this.showLoading('上传图像中...');

        try {
            const response = await fetch('/api/upload_image', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                this.uploadedControlImage = result.file_path;
                this.showUploadPreview(result.file_path);
                this.addLog('上传', '图像上传成功: ' + file.name, 'success');
            } else {
                throw new Error(result.detail || '上传失败');
            }
        } catch (error) {
            this.addLog('错误', '图像上传失败: ' + error.message, 'error');
            alert('图像上传失败: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    showUploadPreview(imagePath) {
        const placeholder = document.querySelector('.upload-placeholder');
        const preview = document.getElementById('upload-preview');

        placeholder.style.display = 'none';
        preview.style.display = 'block';
        preview.innerHTML = `
            <img src="${imagePath}" alt="控制图像" style="width: 100%; max-height: 200px; object-fit: contain; border-radius: 8px;">
            <button type="button" onclick="demo.clearUpload()" style="margin-top: 10px; padding: 5px 10px; background: #e74c3c; color: white; border: none; border-radius: 4px; cursor: pointer;">
                删除图像
            </button>
        `;
    }

    clearUpload() {
        const placeholder = document.querySelector('.upload-placeholder');
        const preview = document.getElementById('upload-preview');
        const fileInput = document.getElementById('control-image');

        placeholder.style.display = 'block';
        preview.style.display = 'none';
        fileInput.value = '';
        this.uploadedControlImage = null;
    }

    displayBatchResults(results) {
        const resultsSection = document.getElementById('batch-results');
        const gallery = document.getElementById('batch-gallery');

        resultsSection.style.display = 'block';
        gallery.innerHTML = '';

        results.forEach((result, index) => {
            const item = document.createElement('div');
            item.className = 'gallery-item';
            item.innerHTML = `
                <img src="${result.generated_image}" alt="生成图像 ${index + 1}" onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMzAwIiBoZWlnaHQ9IjIwMCIgZmlsbD0iI2VjZjBmMSIvPjx0ZXh0IHg9IjE1MCIgeT0iMTAwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjNlbSIgZm9udC1mYW1pbHk9InNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTQiIGZpbGw9IiM3ZjhjOGQiPuWbvuWDj+eUn+aIkOS4re4uLi48L3RleHQ+PC9zdmc+'">
                <div class="gallery-item-info">
                    <h4>图像 ${index + 1}</h4>
                    <p><strong>原图:</strong> ${result.original_image.split('/').pop()}</p>
                    <p><strong>提示:</strong> ${result.prompt.substring(0, 100)}${result.prompt.length > 100 ? '...' : ''}</p>
                </div>
            `;
            gallery.appendChild(item);
        });
    }

    displaySingleResult(result) {
        const resultsSection = document.getElementById('single-results');
        const resultImage = document.getElementById('single-result-image');

        resultsSection.style.display = 'block';
        resultImage.innerHTML = `
            <div class="result-image">
                <img src="${result.generated_image}" alt="生成图像" onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgZmlsbD0iI2VjZjBmMSIvPjx0ZXh0IHg9IjIwMCIgeT0iMTUwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjNlbSIgZm9udC1mYW1pbHk9InNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTYiIGZpbGw9IiM3ZjhjOGQiPuWbvuWDj+eUn+aIkOS4re4uLi48L3RleHQ+PC9zdmc+'">
            </div>
            <div class="result-info">
                <h4>推理参数</h4>
                <div class="param-grid">
                    <div class="param-item">
                        <span class="param-label">提示词:</span>
                        <span class="param-value">${result.prompt}</span>
                    </div>
                    <div class="param-item">
                        <span class="param-label">尺寸:</span>
                        <span class="param-value">${result.parameters.width} × ${result.parameters.height}</span>
                    </div>
                    <div class="param-item">
                        <span class="param-label">引导强度:</span>
                        <span class="param-value">${result.parameters.guidance_scale}</span>
                    </div>
                    <div class="param-item">
                        <span class="param-label">推理步数:</span>
                        <span class="param-value">${result.parameters.num_inference_steps}</span>
                    </div>
                    <div class="param-item">
                        <span class="param-label">随机种子:</span>
                        <span class="param-value">${result.parameters.seed}</span>
                    </div>
                </div>
            </div>
        `;
    }

    showLoading(text = '加载中...') {
        const overlay = document.getElementById('loading-overlay');
        const loadingText = document.getElementById('loading-text');

        loadingText.textContent = text;
        overlay.style.display = 'flex';
    }

    hideLoading() {
        const overlay = document.getElementById('loading-overlay');
        overlay.style.display = 'none';
    }

    addLog(type, message, level = 'info') {
        const logContainer = document.getElementById('log-container');
        const timestamp = new Date().toLocaleTimeString();

        const logEntry = document.createElement('div');
        logEntry.className = 'log-entry';

        const logClass = level === 'error' ? 'log-error' :
            level === 'success' ? 'log-success' :
                level === 'warning' ? 'log-warning' : '';

        logEntry.innerHTML = `
            <span class="log-time">[${timestamp}]</span>
            <span class="log-message ${logClass}">[${type}] ${message}</span>
        `;

        logContainer.appendChild(logEntry);
        logContainer.scrollTop = logContainer.scrollHeight;

        // 限制日志条数，保持性能
        const entries = logContainer.querySelectorAll('.log-entry');
        if (entries.length > 100) {
            entries[0].remove();
        }
    }

    // 工具方法
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    generateRandomSeed() {
        return Math.floor(Math.random() * 2147483647);
    }
}

// 初始化应用
const demo = new FluxDemo();

// 添加一些便捷功能
document.addEventListener('DOMContentLoaded', function () {
    // 添加随机种子按钮
    const seedInputs = document.querySelectorAll('input[id$="-seed"]');
    seedInputs.forEach(input => {
        const randomBtn = document.createElement('button');
        randomBtn.type = 'button';
        randomBtn.innerHTML = '<i class="fas fa-dice"></i>';
        randomBtn.className = 'btn btn-info';
        randomBtn.style.marginLeft = '10px';
        randomBtn.onclick = () => {
            input.value = demo.generateRandomSeed();
        };
        input.parentNode.appendChild(randomBtn);
    });
});
