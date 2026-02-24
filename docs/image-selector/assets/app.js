// GPUStack Image Selector - Main Logic

const CONFIG = {
    versionsUrl: './versions/index.json',
    versionsBaseUrl: './versions/',
    registries: {
        'docker-hub': { 
            name: 'Docker Hub', prefix: 'gpustack/', 
            // Special handling for Docker Hub: point to official image paths
            overrides: { 'postgres': 'postgres', 'prometheus': 'prom/prometheus', 'grafana': 'grafana/grafana' }
        },
        'quay': { name: 'Quay.io', prefix: 'quay.io/gpustack/' },
        'china': { name: '国内镜像源', prefix: 'swr.cn-south-1.myhuaweicloud.com/gpustack/' }
    },
    // Mapping of card types to acceleration frameworks
    cardFrameworkMap: {
        'nvidia': 'CUDA', 'amd': 'ROCm', 'ascend': 'CANN', 'hygon': 'DTK',
        'mthreads': 'MUSA', 'iluvatar': 'CoreX', 'cambricon': 'Neuware',
        'maca': 'MACA', 't-head': 'HGGC'
    },
    // Standard case mapping for inference engine names
    backendNameMap: {
        'vllm': 'vLLM', 'sglang': 'SGLang', 'mindie': 'MindIE', 'voxbox': 'VoxBox'
    },
    // Multilingual dictionary
    i18n: {
        'zh': {
            'title': 'GPUStack 镜像选择器',
            'subtitle': '镜像选择器',
            'back_to_docs': '文档',
            'version': '版本',
            'loading': '加载中...',
            'select_config': '选择配置',
            'gpu_type': 'GPU 类型',
            'framework_version': '计算框架版本',
            'select_gpu_first': '请选择 GPU 类型',
            'inference_backend': '推理后端',
            'inference_backend_tooltip': '如果未找到所需的内置推理后端或对应版本，可尝试切换到较低版本的计算框架。一般来说，高版本驱动能够兼容运行低版本的计算框架。',
            'optional_images': '可选镜像',
            'optional_images_tooltip': 'GPUStack 已内置这些组件，通常无需单独拉取。如需独立部署，请参考：<a href="https://docs.gpustack.ai/latest/cli-reference/start/" target="_blank">CLI 参考文档</a>',
            'postgres': 'PostgreSQL',
            'monitoring': '监控套件 (Prometheus + Grafana)',
            'required_images': '所需镜像',
            'architecture': '架构',
            'registry': '镜像源',
            'china_registry': '国内镜像源',
            'tab_all': '全部',
            'tab_server': 'Server 节点',
            'tab_worker': 'Worker 节点',
            'placeholder_select': '请在左侧选择配置以生成镜像列表',
            'no_images': '# 未找到匹配的镜像',
            'offline_guide_title': '需要离线安装？',
            'guide_tab_save': '镜像文件导入 (Save & Load)',
            'guide_tab_tag': '推送至私有仓库 (Tag & Push)',
            'guide_tab_auto': '自动化镜像同步',
            'guide_save_content': `
                <p>适用于完全物理隔离的环境，通过镜像文件离线导入。</p>
                <ol>
                    <li>在联网机器拉取镜像（参考上方镜像拉取命令）。</li>
                    <li>导出镜像为打包文件：
                        <pre><code>docker save -o gpustack-server-images.tar {{server_images}}\n\ndocker save -o gpustack-worker-images.tar {{worker_images}}</code></pre>
                    </li>
                    <li>将镜像文件拷贝至离线机器。</li>
                    <li>按节点角色导入镜像：
                        <div style="margin-top:10px"><strong>Server 节点：</strong></div>
                        <pre><code>docker load -i gpustack-server-images.tar</code></pre>
                        <div style="margin-top:10px"><strong>Worker 节点：</strong></div>
                        <pre><code>docker load -i gpustack-worker-images.tar</code></pre>
                        <div style="margin-top:10px"><strong>Server + Worker 节点：</strong></div>
                        <pre><code>docker load -i gpustack-server-images.tar\ndocker load -i gpustack-worker-images.tar</code></pre>
                    </li>
                </ol>`,
            'guide_tag_content': `
                <p>适用于内网已有私有仓库（如 Harbor, Nexus）的场景。</p>
                <ol>
                    <li>在联网机器拉取镜像（参考上方镜像拉取命令）。</li>
                    <li>重新打标签并推送至私有仓库：
                        <pre><code>export PrivateRegistry=&lt;您的私有仓库地址&gt;\n{{tag_push_commands}}</code></pre>
                    </li>
                    <li>运行 GPUStack Server 和 Worker 容器时，通过启动参数指定镜像地址：
                        <pre><code>sudo docker run -d --name gpustack \\\n    --restart unless-stopped \\\n    -p 80:80 \\\n    -p 10161:10161 \\\n    --volume gpustack-data:/var/lib/gpustack \\\n    $PrivateRegistry/gpustack/gpustack:{{version}} \\\n    --system-default-container-registry $PrivateRegistry</code></pre>
                    </li>
                </ol>`,
            'guide_auto_content': `
                <p>若需要更自动化的镜像同步手段，GPUStack 提供镜像管理命令，用于同步与管理所需镜像：</p>
                <ul style="list-style:none; padding-left:0">
                    <li style="margin-bottom:8px"><code>gpustack copy-images</code>：从源仓库同步镜像到目标仓库</li>
                    <li style="margin-bottom:8px"><code>gpustack save-images</code>：下载并保存镜像到本地路径</li>
                    <li style="margin-bottom:8px"><code>gpustack load-images</code>：导入本地镜像包</li>
                    <li style="margin-bottom:8px"><code>gpustack list-images</code>：列出当前版本镜像清单</li>
                </ul>
                <div style="margin-top:15px; font-size:13px; color:var(--text-color)">
                    以上命令均支持镜像过滤与自定义配置，具体用法请参考：
                    <a href="https://docs.gpustack.ai/latest/installation/air-gapped/#container-images" target="_blank" style="color:var(--primary-color); font-weight:600">准备容器镜像 &rarr;</a>
                </div>`,
            'view_full_docs': '查看完整离线部署文档 &rarr;',
            'copied': '已复制到剪贴板',
            'no_images': '# 未找到匹配的镜像',
            'comments': {
                'main': 'GPUStack 镜像 - GPUStack 核心服务，Server 和 Worker 节点均需此镜像',
                'runner': '推理后端镜像',
                'pause': 'Pause 镜像 - 提供模型实例容器的共享网络和 IPC 环境，仅 Docker 环境需要',
                'benchmark': 'Benchmark 镜像 - 用于运行模型性能基准测试',
                'postgres': 'PostgreSQL - 用于独立部署外置数据库（可选组件）',
                'monitoring': '监控套件 - 包含 Prometheus 和 Grafana（可选组件）'
            },
            'cards': {
                'nvidia': 'NVIDIA', 'amd': 'AMD', 'ascend': '昇腾', 'hygon': '海光',
                'mthreads': '摩尔线程', 'iluvatar': '天数智芯', 'cambricon': '寒武纪',
                'maca': '沐曦', 't-head': '平头哥 PPU'
            }
        },
        'en': {
            'title': 'GPUStack Image Selector',
            'subtitle': 'Image Selector',
            'back_to_docs': 'Docs',
            'version': 'Version',
            'loading': 'Loading...',
            'select_config': 'Configuration',
            'gpu_type': 'GPU Type',
            'framework_version': 'Framework Version',
            'select_gpu_first': 'Please select GPU Type first',
            'inference_backend': 'Inference Backend',
            'inference_backend_tooltip': 'If you cannot find the desired built-in inference backend or version, try switching the computing framework version to select a lower version image. High-version drivers are generally compatible with lower-version computing frameworks.',
            'optional_images': 'Optional Images',
            'optional_images_tooltip': 'GPUStack has built-in these components, usually no need to pull separately. For independent deployment, please refer to: <a href="https://docs.gpustack.ai/latest/cli-reference/start/" target="_blank">CLI Reference</a>',
            'postgres': 'PostgreSQL',
            'monitoring': 'Monitoring (Prometheus + Grafana)',
            'required_images': 'Required Images',
            'architecture': 'Architecture',
            'registry': 'Registry',
            'china_registry': 'China Mirror',
            'tab_all': 'All',
            'tab_server': 'Server Node',
            'tab_worker': 'Worker Node',
            'placeholder_select': 'Please select configuration on the left to generate image list',
            'offline_guide_title': 'Need offline installation?',
            'guide_tab_save': 'Image File Import (Save & Load)',
            'guide_tab_tag': 'Push to Private Registry (Tag & Push)',
            'guide_tab_auto': 'Automated Image Sync',
            'guide_save_content': `
                <p>Suitable for completely air-gapped environments, importing images via files.</p>
                <ol>
                    <li>Pull images on a machine with internet access (refer to commands above).</li>
                    <li>Export images to tar files:
                        <pre><code>docker save -o gpustack-server-images.tar {{server_images}}\n\ndocker save -o gpustack-worker-images.tar {{worker_images}}</code></pre>
                    </li>
                    <li>Copy files to the offline machine.</li>
                    <li>Import images by node role:
                        <div style="margin-top:10px"><strong>Server Node:</strong></div>
                        <pre><code>docker load -i gpustack-server-images.tar</code></pre>
                        <div style="margin-top:10px"><strong>Worker Node:</strong></div>
                        <pre><code>docker load -i gpustack-worker-images.tar</code></pre>
                        <div style="margin-top:10px"><strong>Server + Worker Node:</strong></div>
                        <pre><code>docker load -i gpustack-server-images.tar\ndocker load -i gpustack-worker-images.tar</code></pre>
                    </li>
                </ol>`,
            'guide_tag_content': `
                <p>Suitable for scenarios where a private registry (e.g., Harbor, Nexus) exists.</p>
                <ol>
                    <li>Pull images on a machine with internet access (refer to commands above).</li>
                    <li>Retag and push to the private registry:
                        <pre><code>export PrivateRegistry=&lt;your-private-registry&gt;\n{{tag_push_commands}}</code></pre>
                    </li>
                    <li>Specify the image registry via start parameters when running containers:
                        <pre><code>sudo docker run -d --name gpustack \\\n    --restart unless-stopped \\\n    -p 80:80 \\\n    -p 10161:10161 \\\n    --volume gpustack-data:/var/lib/gpustack \\\n    $PrivateRegistry/gpustack/gpustack:{{version}} \\\n    --system-default-container-registry $PrivateRegistry</code></pre>
                    </li>
                </ol>`,
            'guide_auto_content': `
                <p>For more automated sync methods, GPUStack provides image management commands:</p>
                <ul style="list-style:none; padding-left:0">
                    <li style="margin-bottom:8px"><code>gpustack copy-images</code>: Sync images from source to destination registry</li>
                    <li style="margin-bottom:8px"><code>gpustack save-images</code>: Download and save images to local path</li>
                    <li style="margin-bottom:8px"><code>gpustack load-images</code>: Import images from local packages</li>
                    <li style="margin-bottom:8px"><code>gpustack list-images</code>: List image manifest for current version</li>
                </ul>
                <div style="margin-top:15px; font-size:13px; color:var(--text-color)">
                    All commands support filtering and custom config. For details, refer to:
                    <a href="https://docs.gpustack.ai/latest/installation/air-gapped/#container-images" target="_blank" style="color:var(--primary-color); font-weight:600">Prepare Container Images &rarr;</a>
                </div>`,
            'view_full_docs': 'View full air-gapped docs &rarr;',
            'copied': 'Copied to clipboard',
            'no_images': '# No matching images found',
            'comments': {
                'main': 'GPUStack Image - GPUStack core service, required for both Server and Worker nodes',
                'runner': 'Inference Backend Images',
                'pause': 'Pause Image - Provides shared network and IPC environment for model instance containers, required for Docker environment only',
                'benchmark': 'Benchmark Image - Used for running model performance benchmarks',
                'postgres': 'PostgreSQL - Used for independent deployment of external database (optional component)',
                'monitoring': 'Monitoring Suite - Includes Prometheus and Grafana (optional components)'
            },
            'cards': {
                'nvidia': 'NVIDIA', 'amd': 'AMD', 'ascend': 'Ascend', 'hygon': 'Hygon',
                'mthreads': 'MThreads', 'iluvatar': 'Iluvatar', 'cambricon': 'Cambricon',
                'maca': 'MetaX', 't-head': 'T-Head PPU'
            }
        }
    }
};

// Global State - Default to English
let state = {
    currentLang: localStorage.getItem('lang') || 'en',
    images: [], runnerImages: [], supportMatrix: {},
    selectedComponent: 'all', selectedArch: 'amd64', selectedRegistry: 'docker-hub',
    selectedCard: null, selectedFrameworkVersion: null, selectedChipType: null,
    selectedBackends: [], availableVersions: [], selectedGpuStackVersion: null,
    optionalImages: { 'postgres': false, 'monitoring': false }
};

// DOM Elements
const elements = {};

// Initialization
async function init() {
    initElements();
    bindEvents();
    updateLanguage();
    await loadData();
    // Show content after initialization
    document.body.classList.add('i18n-ready');
}

// Initialize DOM elements
function initElements() {
    elements.archSelector = document.getElementById('arch-selector');
    elements.registrySelector = document.getElementById('registry-selector');
    elements.gpustackVersionSelect = document.getElementById('gpustack-version-select');
    elements.cardSelector = document.getElementById('card-selector');
    elements.frameworkVersionSelect = document.getElementById('framework-version-select');
    elements.backendSelector = document.getElementById('backend-selector');
    elements.optionalImages = {
        'postgres': document.getElementById('postgres'),
        'monitoring': document.getElementById('monitoring')
    };
    elements.imageTabs = document.querySelectorAll('.image-tab');
    elements.imageList = document.getElementById('image-list');
    elements.copyAllBtn = document.getElementById('copy-all-btn');
    elements.currentLangBtn = document.getElementById('current-lang');
    elements.dropdownLinks = document.querySelectorAll('.dropdown-content a');
    elements.registryChina = document.getElementById('registry-china');
    elements.guideTabs = document.querySelectorAll('.guide-tab');
}

// Bind event listeners
function bindEvents() {
    elements.archSelector.querySelectorAll('.option-button').forEach(btn => {
        btn.addEventListener('click', () => selectArch(btn.dataset.value));
    });
    elements.registrySelector.querySelectorAll('.option-button').forEach(btn => {
        btn.addEventListener('click', () => selectRegistry(btn.dataset.value));
    });
    elements.cardSelector.addEventListener('click', (e) => {
        const btn = e.target.closest('.option-button');
        if (btn) selectCard(btn.dataset.value);
    });
    elements.frameworkVersionSelect.addEventListener('change', (e) => selectFrameworkVersion(e.target.value));
    elements.backendSelector.addEventListener('change', () => updateSelectedBackends());
    Object.keys(elements.optionalImages).forEach(key => {
        elements.optionalImages[key].addEventListener('change', () => {
            state.optionalImages[key] = elements.optionalImages[key].checked;
            generateImageList();
        });
    });
    elements.gpustackVersionSelect.addEventListener('change', (e) => selectGpuStackVersion(e.target.value));
    elements.imageTabs.forEach(tab => {
        tab.addEventListener('click', () => selectComponent(tab.dataset.component));
    });
    elements.guideTabs.forEach(tab => {
        tab.addEventListener('click', () => switchGuideTab(tab.dataset.guide));
    });
    elements.dropdownLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            setLanguage(link.dataset.lang);
        });
    });
}

// Set application language
function setLanguage(lang) {
    if (state.currentLang === lang) return;
    state.currentLang = lang;
    localStorage.setItem('lang', lang);
    updateLanguage();
    // Auto-switch to Docker Hub if China Mirror is selected but language is English
    if (state.currentLang === 'en' && state.selectedRegistry === 'china') {
        selectRegistry('docker-hub');
    }
    renderCardSelector();
    generateImageList();
}

// Get the current set of selected images
function getCurrentImages(component) {
    const imgs = [];
    if (state.selectedGpuStackVersion) imgs.push(getFullImageName('gpustack', state.selectedGpuStackVersion, state.selectedRegistry));
    if (!state.selectedCard || !state.selectedFrameworkVersion) return imgs;
    const fwTag = CONFIG.cardFrameworkMap[state.selectedCard].toLowerCase() + state.selectedFrameworkVersion;
    if (component === 'worker' || component === 'all') {
        state.runnerImages.forEach(img => {
            const tag = img.replace('gpustack/runner:', '');
            if (!tag.startsWith(fwTag)) return;
            if (state.selectedCard === 'ascend' && state.selectedChipType) {
                const pts = tag.split('-');
                if (pts[1] && !['vllm','sglang','mindie','voxbox'].some(b => pts[1].includes(b)) && pts[1] !== state.selectedChipType) return;
            }
            if (state.selectedBackends.length > 0) {
                const pts = tag.split('-');
                let bPart = (pts.length >= 3 && !['vllm','sglang','mindie','voxbox'].some(b => pts[1].includes(b))) ? pts[2] : pts[1];
                const m = bPart?.match(/^([a-z]+)([\d.]+(?:rc\d+)?(?:post\d+)?)?$/i);
                if (!m || !state.selectedBackends.includes(`${m[1].toLowerCase()}-${m[2]}`)) return;
            }
            imgs.push(getFullImageName('runner', tag, state.selectedRegistry));
        });
        const pause = state.images.find(i => i.includes('runtime:pause'));
        if (pause) imgs.push(getFullImageName('runtime', pause.split(':')[1], state.selectedRegistry));
        const bm = state.images.find(i => i.includes('benchmark-runner'));
        if (bm) imgs.push(getFullImageName('benchmark-runner', bm.split(':')[1], state.selectedRegistry));
    }
    if (component === 'server' || component === 'all') {
        if (state.optionalImages['postgres']) {
            const pgs = state.images.filter(i => i.startsWith('postgres:'));
            pgs.forEach(i => imgs.push(getFullImageName('postgres', i.split(':')[1], state.selectedRegistry)));
        }
        if (state.optionalImages['monitoring']) {
            state.images.filter(i => i.includes('prometheus')).forEach(i => imgs.push(getFullImageName('prometheus', i.split(':')[1], state.selectedRegistry)));
            state.images.filter(i => i.includes('grafana')).forEach(i => imgs.push(getFullImageName('grafana', i.split(':')[1], state.selectedRegistry)));
        }
    }
    return imgs;
}

// Update UI text based on current language
function updateLanguage() {
    const lang = state.currentLang;
    const data = CONFIG.i18n[lang];
    const version = state.selectedGpuStackVersion || 'latest';

    // Translate page title
    document.title = data.title;

    elements.currentLangBtn.textContent = lang === 'zh' ? '简体中文' : 'English';

    // Get current image lists for dynamic placeholders
    const allImgs = getCurrentImages('all');
    const serverImgs = getCurrentImages('server').join(' ');
    const workerImgs = getCurrentImages('worker').join(' ');

    // Generate tag & push commands under 'gpustack' namespace
    const tagPushCmds = allImgs.map(img => {
        const parts = img.split(':');
        const namePart = parts[0];
        const tagPart = parts[1];
        const shortName = namePart.includes('/') ? namePart.split('/').pop() : namePart;
        const destImg = `gpustack/${shortName}:${tagPart}`;
        return `docker tag ${img} $PrivateRegistry/${destImg}\ndocker push $PrivateRegistry/${destImg}`;
    }).join('\n');

    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.dataset.i18n;
        if (data[key]) {
            let content = data[key];
            // Dynamically replace placeholders
            content = content.replace(/{{version}}/g, version);
            content = content.replace(/{{server_images}}/g, serverImgs || '&lt;Server Image List&gt;');
            content = content.replace(/{{worker_images}}/g, workerImgs || '&lt;Worker Image List&gt;');
            content = content.replace(/{{tag_push_commands}}/g, tagPushCmds || '# Please select configuration to generate commands');
            el.innerHTML = content;
        }
    });
    const copyImagesEl = document.getElementById('copy-images-command');
    if (copyImagesEl) copyImagesEl.textContent = data.copy_images_cmd;

    // Hide China Mirror in English mode
    elements.registryChina.style.display = (lang === 'en') ? 'none' : 'block';
}

// Load versions and initial data
async function loadData() {
    try {
        const response = await fetch(CONFIG.versionsUrl);
        const data = await response.json();
        state.availableVersions = data.versions || data;
        state.selectedGpuStackVersion = state.availableVersions[0];
        renderVersionSelector();
        await loadVersionData(state.selectedGpuStackVersion);
    } catch (error) {
        console.error('Failed to load data:', error);
        showToast(state.currentLang === 'zh' ? '加载数据失败' : 'Failed to load data');
    }
}

// Load specific version data
async function loadVersionData(version) {
    const response = await fetch(`${CONFIG.versionsBaseUrl}${version}.json`);
    const data = await response.json();
    state.images = data;
    state.runnerImages = data.filter(img => img.startsWith('gpustack/runner:'));
    parseSupportMatrix();
    renderCardSelector();
    updateLanguage(); // Refresh offline guide version number
}

// Render version dropdown
function renderVersionSelector() {
    elements.gpustackVersionSelect.innerHTML = '';
    state.availableVersions.forEach(v => {
        const opt = document.createElement('option');
        opt.value = opt.textContent = v;
        opt.selected = (v === state.selectedGpuStackVersion);
        elements.gpustackVersionSelect.appendChild(opt);
    });
}

// Handle version change
async function selectGpuStackVersion(v) {
    state.selectedGpuStackVersion = v;
    await loadVersionData(v);
    generateImageList();
}

// Parse supported hardware and backends from images
function parseSupportMatrix() {
    const matrix = { frameworks: {} };
    state.runnerImages.forEach(image => {
        const tag = image.replace('gpustack/runner:', '');
        let parts = tag.split('-');
        const fwMatch = parts[0].match(/^([a-z]+)([\d.]+)$/i);
        if (!fwMatch) return;
        const fw = fwMatch[1].toLowerCase();
        if (!matrix.frameworks[fw]) matrix.frameworks[fw] = { versions: new Set(), cards: new Set(), backends: new Set() };
        matrix.frameworks[fw].versions.add(fwMatch[2]);
        let hasCard = false;
        if (parts.length >= 2 && /^[a-z0-9]+$/i.test(parts[1]) && !/^\d/.test(parts[1]) && !['vllm','sglang','mindie','voxbox'].some(b => parts[1].includes(b))) {
            matrix.frameworks[fw].cards.add(parts[1]);
            hasCard = true;
        }
        const backendPart = hasCard ? parts[2] : parts[1];
        if (backendPart) {
            const bMatch = backendPart.match(/^([a-z]+)([\d.]+(?:rc\d+)?(?:post\d+)?)?$/i);
            if (bMatch) matrix.frameworks[fw].backends.add(bMatch[1].toLowerCase());
        }
    });
    state.supportMatrix = { frameworks: {} };
    Object.keys(matrix.frameworks).forEach(fw => {
        state.supportMatrix.frameworks[fw] = {
            versions: Array.from(matrix.frameworks[fw].versions),
            cards: Array.from(matrix.frameworks[fw].cards),
            backends: Array.from(matrix.frameworks[fw].backends)
        };
    });
}

// Render GPU Type buttons
function renderCardSelector() {
    elements.cardSelector.innerHTML = '';
    const cardDict = CONFIG.i18n[state.currentLang].cards;
    const allCards = [
        { id: 'nvidia', name: cardDict.nvidia, framework: 'CUDA' },
        { id: 'amd', name: cardDict.amd, framework: 'ROCm' },
        { id: 'ascend', name: cardDict.ascend, framework: 'CANN' },
        { id: 'hygon', name: cardDict.hygon, framework: 'DTK' },
        { id: 'mthreads', name: cardDict.mthreads, framework: 'MUSA' },
        { id: 'iluvatar', name: cardDict.iluvatar, framework: 'CoreX' },
        { id: 'cambricon', name: cardDict.cambricon, framework: 'Neuware' },
        { id: 'maca', name: cardDict.maca, framework: 'MACA' },
        { id: 't-head', name: cardDict.t_head || cardDict['t-head'], framework: 'HGGC' }
    ];
    allCards.forEach(card => {
        const framework = card.framework.toLowerCase();
        const hasData = state.supportMatrix.frameworks[framework];
        const btn = document.createElement('button');
        btn.className = 'option-button';
        btn.dataset.value = card.id;
        btn.innerHTML = `<span>${card.name}</span>`;
        if (card.id === 'cambricon') {
            const tip = state.currentLang === 'zh' ? '请联系寒武纪厂商获取推理后端镜像' : 'Please contact Cambricon vendor for inference backend images';
            btn.innerHTML += `<span class="tooltip-icon">i <span class="tooltip-content">${tip}</span></span>`;
        }
        btn.disabled = !hasData;
        if (!hasData) btn.title = state.currentLang === 'zh' ? '暂无可用镜像' : 'No images available';
        elements.cardSelector.appendChild(btn);
    });
    const activeBtn = elements.cardSelector.querySelector(`[data-value="${state.selectedCard}"]`) || elements.cardSelector.querySelector('[data-value="nvidia"]');
    if (activeBtn && !activeBtn.disabled) selectCard(activeBtn.dataset.value);
}

// Handle GPU Type selection
function selectCard(cardId) {
    state.selectedCard = cardId;
    elements.cardSelector.querySelectorAll('.option-button').forEach(b => b.classList.toggle('active', b.dataset.value === cardId));
    const fw = CONFIG.cardFrameworkMap[cardId].toLowerCase();
    renderFrameworkVersions(fw, cardId);
    const first = elements.frameworkVersionSelect.querySelector('option:not([value=""])');
    if (first) { first.selected = true; selectFrameworkVersion(first.value); }
}

// Render Framework Version dropdown
function renderFrameworkVersions(fw, cardId) {
    const select = elements.frameworkVersionSelect;
    const placeholder = state.currentLang === 'zh' ? '请选择版本' : 'Please select version';
    select.innerHTML = `<option value="">${placeholder}</option>`;
    const versions = state.supportMatrix.frameworks[fw]?.versions || [];
    const fwName = CONFIG.cardFrameworkMap[cardId];
    if (cardId === 'ascend') {
        const combos = new Map();
        state.runnerImages.forEach(img => {
            const tag = img.replace('gpustack/runner:', '');
            if (!tag.startsWith(fw) || !tag.startsWith('cann')) return;
            const pts = tag.split('-');
            if (pts.length < 2) return;
            const vMatch = pts[0].match(/^cann([\d.]+)$/i);
            if (vMatch && !['vllm','sglang','mindie','voxbox'].some(b => pts[1].includes(b))) combos.set(`${vMatch[1]}-${pts[1]}`, { v: vMatch[1], c: pts[1] });
        });
        Array.from(combos.values()).sort((a,b) => b.v.localeCompare(a.v) || a.c.localeCompare(b.c)).forEach(item => {
            const opt = document.createElement('option');
            opt.value = item.v; opt.dataset.chipType = item.c; opt.textContent = `${fwName} ${item.v} (${item.c})`;
            select.appendChild(opt);
        });
    } else {
        versions.sort((a,b) => b.localeCompare(a)).forEach(v => {
            const opt = document.createElement('option');
            opt.value = v; opt.textContent = `${fwName} ${v}`;
            select.appendChild(opt);
        });
    }
}

// Handle Framework Version selection
function selectFrameworkVersion(v) {
    state.selectedFrameworkVersion = v;
    const opt = elements.frameworkVersionSelect.selectedOptions[0];
    state.selectedChipType = opt?.dataset.chipType || null;
    const fw = CONFIG.cardFrameworkMap[state.selectedCard].toLowerCase();
    renderBackends(fw, v, state.selectedChipType);
    state.selectedBackends = [];
    generateImageList();
}

// Render Inference Backend checkboxes
function renderBackends(fw, v, chip) {
    elements.backendSelector.innerHTML = '';
    if (!v) return;
    const fwTag = fw + v;
    const options = new Map();
    state.runnerImages.forEach(img => {
        const tag = img.replace('gpustack/runner:', '');
        if (!tag.startsWith(fwTag)) return;
        const pts = tag.split('-');
        let bPart = pts[1];
        let hasCard = pts.length >= 3 && !['vllm','sglang','mindie','voxbox'].some(b => pts[1].includes(b));
        if (hasCard) { if (state.selectedCard === 'ascend' && pts[1] !== chip) return; bPart = pts[2]; }
        const m = bPart?.match(/^([a-z]+)([\d.]+(?:rc\d+)?(?:post\d+)?)?$/i);
        if (m) options.set(`${m[1]}-${m[2]}`, `${CONFIG.backendNameMap[m[1].toLowerCase()] || m[1]} ${m[2]}`);
    });
    // Sort versions in reverse order
    Array.from(options.entries()).sort((a,b) => b[0].localeCompare(a[0])).forEach(([k, name]) => {
        const lbl = document.createElement('label'); lbl.className = 'checkbox-item';
        lbl.innerHTML = `<input type="checkbox" value="${k}"><span>${name}</span>`;
        elements.backendSelector.appendChild(lbl);
    });
}

// Update selected backends
function updateSelectedBackends() {
    state.selectedBackends = Array.from(elements.backendSelector.querySelectorAll('input:checked')).map(i => i.value);
    generateImageList();
}

// Get full image name supporting Overrides and Registry logic
function getFullImageName(baseName, tag, registryKey) {
    const reg = CONFIG.registries[registryKey];
    let path = `${reg.prefix}${baseName}`;
    if (reg.overrides && reg.overrides[baseName]) {
        path = reg.overrides[baseName];
    }
    return `${path}:${tag}`;
}

// Main logic to generate the list of docker pull commands
function generateImageList() {
    const plat = `--platform linux/${state.selectedArch}`;
    const cmds = [];
    const isServer = state.selectedComponent === 'server' || state.selectedComponent === 'all';
    const isWorker = state.selectedComponent === 'worker' || state.selectedComponent === 'all';
    const t = CONFIG.i18n[state.currentLang];
    
    if (state.selectedGpuStackVersion) {
        cmds.push(`# ${t.comments.main}`);
        cmds.push(`docker pull ${plat} ${getFullImageName('gpustack', state.selectedGpuStackVersion, state.selectedRegistry)}`);
    }

    if (!state.selectedCard || !state.selectedFrameworkVersion) {
        elements.imageList.textContent = cmds.length ? cmds.join('\n') : t.placeholder_select;
        elements.copyAllBtn.style.display = cmds.length ? 'flex' : 'none';
        if (cmds.length) elements.copyAllBtn.onclick = () => copyToClipboard(cmds.join('\n'));
        updateLanguage(); 
        return;
    }

    const fwTag = CONFIG.cardFrameworkMap[state.selectedCard].toLowerCase() + state.selectedFrameworkVersion;

    if (isWorker) {
        const rCmds = [];
        state.runnerImages.forEach(img => {
            const tag = img.replace('gpustack/runner:', '');
            if (!tag.startsWith(fwTag)) return;
            if (state.selectedCard === 'ascend' && state.selectedChipType) {
                const pts = tag.split('-');
                if (pts[1] && !['vllm','sglang','mindie','voxbox'].some(b => pts[1].includes(b)) && pts[1] !== state.selectedChipType) return;
            }
            if (state.selectedBackends.length > 0) {
                const pts = tag.split('-');
                let bPart = (pts.length >= 3 && !['vllm','sglang','mindie','voxbox'].some(b => pts[1].includes(b))) ? pts[2] : pts[1];
                const m = bPart?.match(/^([a-z]+)([\d.]+(?:rc\d+)?(?:post\d+)?)?$/i);
                if (!m || !state.selectedBackends.includes(`${m[1].toLowerCase()}-${m[2]}`)) return;
            }
            rCmds.push(`docker pull ${plat} ${getFullImageName('runner', tag, state.selectedRegistry)}`);
        });
        if (rCmds.length) { 
            const comment = t.comments.runner;
            cmds.push(`# ${comment}`); 
            cmds.push(...rCmds); 
        }
        
        const pause = state.images.find(i => i.includes('runtime:pause'));
        if (pause) { 
            const comment = t.comments.pause;
            cmds.push(`# ${comment}`); 
            cmds.push(`docker pull ${plat} ${getFullImageName('runtime', pause.split(':')[1], state.selectedRegistry)}`); 
        }
        
        const bm = state.images.find(i => i.includes('benchmark-runner'));
        if (bm) { 
            const comment = t.comments.benchmark;
            cmds.push(`# ${comment}`); 
            cmds.push(`docker pull ${plat} ${getFullImageName('benchmark-runner', bm.split(':')[1], state.selectedRegistry)}`); 
        }
    }

    if (isServer) {
        if (state.optionalImages['postgres']) {
            const pgs = state.images.filter(i => i.startsWith('postgres:'));
            if (pgs.length) { 
                const comment = t.comments.postgres;
                cmds.push(`# ${comment}`); 
                pgs.forEach(i => cmds.push(`docker pull ${plat} ${getFullImageName('postgres', i.split(':')[1], state.selectedRegistry)}`)); 
            }
        }
        if (state.optionalImages['monitoring']) {
            const comment = t.comments.monitoring;
            cmds.push(`# ${comment}`);
            state.images.filter(i => i.includes('prometheus')).forEach(i => cmds.push(`docker pull ${plat} ${getFullImageName('prometheus', i.split(':')[1], state.selectedRegistry)}`));
            state.images.filter(i => i.includes('grafana')).forEach(i => cmds.push(`docker pull ${plat} ${getFullImageName('grafana', i.split(':')[1], state.selectedRegistry)}`));
        }
    }
    renderImageList(cmds);
    updateLanguage(); 
}

// Display the generated commands in the output area
function renderImageList(cmds) {
    if (!cmds.length) { elements.imageList.textContent = CONFIG.i18n[state.currentLang].no_images; elements.copyAllBtn.style.display = 'none'; return; }
    elements.imageList.textContent = cmds.join('\n');
    elements.copyAllBtn.style.display = 'flex';
    elements.copyAllBtn.onclick = () => copyToClipboard(cmds.join('\n'));
}

// Interaction handlers
function selectArch(v) { state.selectedArch = v; elements.archSelector.querySelectorAll('.option-button').forEach(b => b.classList.toggle('active', b.dataset.value === v)); generateImageList(); }
function selectRegistry(v) { state.selectedRegistry = v; elements.registrySelector.querySelectorAll('.option-button').forEach(b => b.classList.toggle('active', b.dataset.value === v)); generateImageList(); }
function selectComponent(v) { state.selectedComponent = v; elements.imageTabs.forEach(t => t.classList.toggle('active', t.dataset.component === v)); generateImageList(); }
function switchGuideTab(id) { elements.guideTabs.forEach(t => t.classList.toggle('active', t.dataset.guide === id)); document.querySelectorAll('.guide-content').forEach(c => c.classList.toggle('active', c.id === `guide-${id}`)); }

// Utility to copy text to clipboard
async function copyToClipboard(text) {
    try { await navigator.clipboard.writeText(text); showToast(CONFIG.i18n[state.currentLang].copied); }
    catch (err) { const ta = document.createElement('textarea'); ta.value = text; document.body.appendChild(ta); ta.select(); document.execCommand('copy'); document.body.removeChild(ta); showToast(CONFIG.i18n[state.currentLang].copied); }
}

// Show temporary toast message
function showToast(msg) {
    const t = document.createElement('div'); t.className = 'toast'; t.textContent = msg; document.body.appendChild(t);
    setTimeout(() => t.remove(), 2000);
}

// Start app when DOM is ready
document.addEventListener('DOMContentLoaded', init);
