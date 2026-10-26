// Container Image Selector - Main Logic

const CONFIG = {
    versionsUrl: './versions/index.json',
    versionsBaseUrl: './versions/',
    registries: {
        'docker-hub': { 
            name: 'Docker Hub', prefix: 'gpustack/', registry: 'docker.io',
            // Special handling for Docker Hub: point to official image paths
            overrides: { 'postgres': 'postgres', 'prometheus': 'prom/prometheus', 'grafana': 'grafana/grafana' }
        },
        'quay': { name: 'Quay.io', prefix: 'quay.io/gpustack/', registry: 'quay.io' },
        'china': { name: '国内镜像源', prefix: 'swr.cn-south-1.myhuaweicloud.com/gpustack/', registry: 'swr.cn-south-1.myhuaweicloud.com' }
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
            'title': '容器镜像选择器',
            'subtitle': '容器镜像选择器',
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
            'optional_postgres_images_tooltip': '使用外部数据库：<a href="../installation/installation/#using-an-external-database" target="_blank">查看配置文档</a>',
            'optional_monitoring_images_tooltip': '配置外部可观测性：<a href="../user-guide/observability/#external-observability-optional" target="_blank">查看配置文档</a>',
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
                    <li>
                        运行 GPUStack Server 和 Worker 容器时通过 <code class="inline-code">--system-default-container-registry</code> 参数指定镜像源：
                        <pre><code>sudo docker run -d --name gpustack \\\n    --restart unless-stopped \\\n    -p 80:80 \\\n    -p 10161:10161 \\\n    --volume gpustack-data:/var/lib/gpustack \\\n    {{registry}}/gpustack/gpustack:{{version}} \\\n    --system-default-container-registry {{registry}}</code></pre>
                    </li>
                </ol>`,
            'guide_tag_content': `
                <p>适用于内网已有私有仓库（如 Harbor, Nexus）的场景。</p>
                <ol>
                    <li>在联网机器拉取镜像（参考上方镜像拉取命令）。</li>
                    <li>重新打标签并推送至私有仓库：
                        <pre><code>export PrivateRegistry=&lt;您的私有仓库地址&gt;\n{{tag_push_commands}}</code></pre>
                    </li>
                    <li>运行 GPUStack Server 和 Worker 容器时，通过启动参数指定镜像源：
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
                    <a href="../installation/air-gapped/#container-images" target="_blank" style="color:var(--primary-color); font-weight:600">准备容器镜像 &rarr;</a>
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
                'monitoring': '监控套件 - 包含 Prometheus 和 Grafana（可选组件）',
                'gateway': '网关镜像 - 仅在 Kubernetes 环境下需要',
                'k8s': 'GPU 服务镜像 - 仅在 Kubernetes 环境下需要'
            },
            'cards': {
                'nvidia': 'NVIDIA', 'amd': 'AMD', 'ascend': '昇腾', 'hygon': '海光',
                'mthreads': '摩尔线程', 'iluvatar': '天数智芯', 'cambricon': '寒武纪',
                'maca': '沐曦', 't-head': '平头哥 PPU'
            }
        },
        'ja': {
            "title": "コンテナイメージセレクター",
            "subtitle": "コンテナイメージセレクター",
            "back_to_docs": "ドキュメント",
            "version": "バージョン",
            "loading": "読み込み中...",
            "select_config": "構成",
            "gpu_type": "GPU の種類",
            "framework_version": "計算フレームワークのバージョン",
            "select_gpu_first": "まず GPU の種類を選択してください",
            "inference_backend": "推論バックエンド",
            "inference_backend_tooltip": "必要な組み込み推論バックエンドやバージョンが見つからない場合は、計算フレームワークのバージョンを下げてみてください。通常、新しいドライバーは古いバージョンの計算フレームワークと互換性があります。",
            "optional_images": "オプションのイメージ",
            "optional_postgres_images_tooltip": "外部データベースの使用：<a href=\"../installation/installation/#using-an-external-database\" target=\"_blank\">構成ガイドを参照</a>",
            "optional_monitoring_images_tooltip": "外部の可観測性基盤の構成：<a href=\"../user-guide/observability/#external-observability-optional\" target=\"_blank\">構成ガイドを参照</a>",
            "postgres": "PostgreSQL",
            "monitoring": "監視スイート（Prometheus + Grafana）",
            "required_images": "必要なイメージ",
            "architecture": "アーキテクチャ",
            "registry": "レジストリ",
            "china_registry": "中国向けミラー",
            "tab_all": "すべて",
            "tab_server": "Server ノード",
            "tab_worker": "Worker ノード",
            "placeholder_select": "左側で構成を選択すると、イメージ一覧が生成されます",
            "offline_guide_title": "オフラインでインストールしますか？",
            "guide_tab_save": "イメージファイルのインポート（Save & Load）",
            "guide_tab_tag": "プライベートレジストリへのプッシュ（Tag & Push）",
            "guide_tab_auto": "イメージの自動同期",
            "view_full_docs": "オフラインインストールのドキュメントを開く &rarr;",
            "copied": "クリップボードにコピーしました",
            "no_images": "# 条件に一致するイメージがありません",
            "comments": {
                "main": "GPUStack イメージ — GPUStack のコアサービス。Server と Worker の両ノードに必要です",
                "runner": "推論バックエンドのイメージ",
                "pause": "Pause イメージ — モデルインスタンスのコンテナに共有ネットワークと IPC 環境を提供します。Docker 環境でのみ必要です",
                "benchmark": "Benchmark イメージ — モデルの性能ベンチマークを実行します",
                "postgres": "PostgreSQL — 外部データベースを独立してデプロイするためのオプションコンポーネントです",
                "monitoring": "監視スイート — Prometheus と Grafana を含むオプションコンポーネントです",
                "gateway": "ゲートウェイイメージ — Kubernetes でのデプロイにのみ必要です",
                "k8s": "GPU サービスのイメージ — Kubernetes でのデプロイにのみ必要です"
            },
            "cards": {
                "nvidia": "NVIDIA",
                "amd": "AMD",
                "ascend": "Ascend",
                "hygon": "Hygon",
                "mthreads": "MThreads",
                "iluvatar": "Iluvatar",
                "cambricon": "Cambricon",
                "maca": "MetaX",
                "t-head": "T-Head PPU"
            },
            "guide_save_content": `
                <p>完全に隔離された環境向けに、ファイルを使ってイメージをインポートします。</p>
                <ol>
                    <li>インターネットに接続できるマシンでイメージを取得します（上のコマンドを参照）。</li>
                    <li>イメージを tar ファイルにエクスポートします：
                        <pre><code>docker save -o gpustack-server-images.tar {{server_images}}

docker save -o gpustack-worker-images.tar {{worker_images}}</code></pre>
                    </li>
                    <li>ファイルをオフラインのマシンにコピーします。</li>
                    <li>ノードの役割に応じてイメージをインポートします：
                        <div style="margin-top:10px"><strong>Server ノード：</strong></div>
                        <pre><code>docker load -i gpustack-server-images.tar</code></pre>
                        <div style="margin-top:10px"><strong>Worker ノード：</strong></div>
                        <pre><code>docker load -i gpustack-worker-images.tar</code></pre>
                        <div style="margin-top:10px"><strong>Server + Worker ノード：</strong></div>
                        <pre><code>docker load -i gpustack-server-images.tar
docker load -i gpustack-worker-images.tar</code></pre>
                    </li>
                    <li>
                        GPUStack Server と Worker のコンテナを実行する際は、 <code class="inline-code">--system-default-container-registry</code> パラメーターでコンテナレジストリを指定します：
                        <pre><code>sudo docker run -d --name gpustack \\
    --restart unless-stopped \\
    -p 80:80 \\
    -p 10161:10161 \\
    --volume gpustack-data:/var/lib/gpustack \\
    {{registry}}/gpustack/gpustack:{{version}} \\
    --system-default-container-registry {{registry}}</code></pre>
                    </li>
                </ol>
            `,
            "guide_tag_content": `
                <p>Harbor や Nexus などのプライベートレジストリがある環境に適しています。</p>
                <ol>
                    <li>インターネットに接続できるマシンでイメージを取得します（上のコマンドを参照）。</li>
                    <li>タグを付け直してプライベートレジストリにプッシュします：
                        <pre><code>export PrivateRegistry=&lt;your-private-registry&gt;
{{tag_push_commands}}</code></pre>
                    </li>
                    <li>コンテナの実行時に、起動パラメーターでイメージレジストリを指定します：
                        <pre><code>sudo docker run -d --name gpustack \\
    --restart unless-stopped \\
    -p 80:80 \\
    -p 10161:10161 \\
    --volume gpustack-data:/var/lib/gpustack \\
    $PrivateRegistry/gpustack/gpustack:{{version}} \\
    --system-default-container-registry $PrivateRegistry</code></pre>
                    </li>
                </ol>
            `,
            "guide_auto_content": `
                <p>同期をさらに自動化するために、GPUStack はイメージ管理コマンドを提供しています：</p>
                <ul style="list-style:none; padding-left:0">
                    <li style="margin-bottom:8px"><code>gpustack copy-images</code>: 転送元レジストリから転送先レジストリにイメージを同期</li>
                    <li style="margin-bottom:8px"><code>gpustack save-images</code>: イメージをダウンロードしてローカルパスに保存</li>
                    <li style="margin-bottom:8px"><code>gpustack load-images</code>: ローカルのイメージパッケージをインポート</li>
                    <li style="margin-bottom:8px"><code>gpustack list-images</code>: 現在のバージョンのイメージマニフェストを一覧表示</li>
                </ul>
                <div style="margin-top:15px; font-size:13px; color:var(--text-color)">
                    すべてのコマンドでフィルターとカスタム構成を利用できます。詳細は次を参照してください：
                    <a href="../installation/air-gapped/#container-images" target="_blank" style="color:var(--primary-color); font-weight:600">コンテナイメージの準備 &rarr;</a>
                </div>
            `
        },
        'en': {
            'title': 'Container Image Selector',
            'subtitle': 'Container Image Selector',
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
            'optional_postgres_images_tooltip': 'Using an external database: <a href="../installation/installation/#using-an-external-database" target="_blank">View configuration guide</a>',
            'optional_monitoring_images_tooltip': 'Configuring external observability: <a href="../user-guide/observability/#external-observability-optional" target="_blank">View configuration guide</a>',
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
                    <li>
                        When running GPUStack Server and Worker containers, specify the container registry using the <code class="inline-code">--system-default-container-registry</code> parameter:
                        <pre><code>sudo docker run -d --name gpustack \\\n    --restart unless-stopped \\\n    -p 80:80 \\\n    -p 10161:10161 \\\n    --volume gpustack-data:/var/lib/gpustack \\\n    {{registry}}/gpustack/gpustack:{{version}} \\\n    --system-default-container-registry {{registry}}</code></pre>
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
                    <a href="../installation/air-gapped/#container-images" target="_blank" style="color:var(--primary-color); font-weight:600">Prepare Container Images &rarr;</a>
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
                'monitoring': 'Monitoring Suite - Includes Prometheus and Grafana (optional components)',
                'gateway': 'Gateway Images - Required for Kubernetes deployment only',
                'k8s': 'GPU Service Images - Required for Kubernetes deployment only'
            },
            'cards': {
                'nvidia': 'NVIDIA', 'amd': 'AMD', 'ascend': 'Ascend', 'hygon': 'Hygon',
                'mthreads': 'MThreads', 'iluvatar': 'Iluvatar', 'cambricon': 'Cambricon',
                'maca': 'MetaX', 't-head': 'T-Head PPU'
            }
        }
    }
};

// Labels shared by standalone and documentation-integrated selectors.
Object.assign(CONFIG.i18n.en, {
    "language_name": "English",
    "description": "Container Image Selector - Generate Docker pull commands based on your hardware configuration.",
    "auto_arch": "Auto",
    "copy_all": "Copy all commands",
    "server_image_placeholder": "&lt;Server Image List&gt;",
    "worker_image_placeholder": "&lt;Worker Image List&gt;",
    "configuration_placeholder": "# Please select configuration to generate commands",
    "load_failed": "Failed to load data",
    "cambricon_hint": "Please contact Cambricon vendor for inference backend images",
    "no_available_images": "No images available",
    "select_version": "Please select version"
});
Object.assign(CONFIG.i18n.zh, {
    "language_name": "简体中文",
    "description": "容器镜像选择器：根据硬件配置生成 Docker 镜像拉取命令。",
    "auto_arch": "自动",
    "copy_all": "复制所有命令",
    "server_image_placeholder": "&lt;Server 镜像列表&gt;",
    "worker_image_placeholder": "&lt;Worker 镜像列表&gt;",
    "configuration_placeholder": "# 请选择配置以生成命令",
    "load_failed": "加载数据失败",
    "cambricon_hint": "请联系寒武纪厂商获取推理后端镜像",
    "no_available_images": "暂无可用镜像",
    "select_version": "请选择版本"
});
Object.assign(CONFIG.i18n.ja, {
    "language_name": "日本語",
    "description": "コンテナイメージセレクター：ハードウェア構成に応じた Docker イメージ取得コマンドを生成します。",
    "auto_arch": "自動",
    "copy_all": "すべてのコマンドをコピー",
    "server_image_placeholder": "&lt;Server イメージ一覧&gt;",
    "worker_image_placeholder": "&lt;Worker イメージ一覧&gt;",
    "configuration_placeholder": "# 構成を選択するとコマンドが生成されます",
    "load_failed": "データを読み込めませんでした",
    "cambricon_hint": "推論バックエンドのイメージについては Cambricon にお問い合わせください",
    "no_available_images": "利用可能なイメージがありません",
    "select_version": "バージョンを選択してください"
});

CONFIG.i18n.ru = {
    "title": "Выбор образов контейнеров",
    "subtitle": "Выбор образов контейнеров",
    "back_to_docs": "Документация",
    "version": "Версия",
    "loading": "Загрузка...",
    "select_config": "Конфигурация",
    "gpu_type": "Тип GPU",
    "framework_version": "Версия вычислительной платформы",
    "select_gpu_first": "Сначала выберите тип GPU",
    "inference_backend": "Бэкенд инференса",
    "inference_backend_tooltip": "Если нужный встроенный бэкенд инференса или его версия отсутствуют, попробуйте выбрать более раннюю версию вычислительной платформы. Новые драйверы обычно совместимы с более ранними версиями платформы.",
    "optional_images": "Дополнительные образы",
    "optional_postgres_images_tooltip": "Внешняя база данных: <a href=\"../installation/installation/#using-an-external-database\" target=\"_blank\">инструкция по настройке</a>",
    "optional_monitoring_images_tooltip": "Внешние средства наблюдаемости: <a href=\"../user-guide/observability/#external-observability-optional\" target=\"_blank\">инструкция по настройке</a>",
    "postgres": "PostgreSQL",
    "monitoring": "Мониторинг (Prometheus + Grafana)",
    "required_images": "Необходимые образы",
    "architecture": "Архитектура",
    "registry": "Реестр",
    "china_registry": "Зеркало для Китая",
    "tab_all": "Все",
    "tab_server": "Узел Server",
    "tab_worker": "Узел Worker",
    "placeholder_select": "Выберите конфигурацию слева, чтобы сформировать список образов",
    "offline_guide_title": "Нужна установка без доступа к сети?",
    "guide_tab_save": "Импорт файлов образов (Save & Load)",
    "guide_tab_tag": "Отправка в частный реестр (Tag & Push)",
    "guide_tab_auto": "Автоматическая синхронизация образов",
    "guide_save_content": `
                <p>Для полностью изолированных сред: перенос образов с помощью файлов.</p>
                <ol>
                    <li>Загрузите образы на машине с доступом к интернету (см. команды выше).</li>
                    <li>Сохраните образы в tar-файлы:
                        <pre><code>docker save -o gpustack-server-images.tar {{server_images}}

docker save -o gpustack-worker-images.tar {{worker_images}}</code></pre>
                    </li>
                    <li>Скопируйте файлы на автономную машину.</li>
                    <li>Импортируйте образы в соответствии с ролью узла:
                        <div style="margin-top:10px"><strong>Узел Server:</strong></div>
                        <pre><code>docker load -i gpustack-server-images.tar</code></pre>
                        <div style="margin-top:10px"><strong>Узел Worker:</strong></div>
                        <pre><code>docker load -i gpustack-worker-images.tar</code></pre>
                        <div style="margin-top:10px"><strong>Server + Узел Worker:</strong></div>
                        <pre><code>docker load -i gpustack-server-images.tar
docker load -i gpustack-worker-images.tar</code></pre>
                    </li>
                    <li>
                        При запуске контейнеров GPUStack Server и Worker укажите реестр контейнеров с помощью параметра <code class="inline-code">--system-default-container-registry</code>:
                        <pre><code>sudo docker run -d --name gpustack \\
    --restart unless-stopped \\
    -p 80:80 \\
    -p 10161:10161 \\
    --volume gpustack-data:/var/lib/gpustack \\
    {{registry}}/gpustack/gpustack:{{version}} \\
    --system-default-container-registry {{registry}}</code></pre>
                    </li>
                </ol>
    `,
    "guide_tag_content": `
                <p>Для сред с частным реестром, например Harbor или Nexus.</p>
                <ol>
                    <li>Загрузите образы на машине с доступом к интернету (см. команды выше).</li>
                    <li>Присвойте новые теги и отправьте образы в частный реестр:
                        <pre><code>export PrivateRegistry=&lt;your-private-registry&gt;
{{tag_push_commands}}</code></pre>
                    </li>
                    <li>При запуске контейнеров укажите реестр образов в параметрах запуска:
                        <pre><code>sudo docker run -d --name gpustack \\
    --restart unless-stopped \\
    -p 80:80 \\
    -p 10161:10161 \\
    --volume gpustack-data:/var/lib/gpustack \\
    $PrivateRegistry/gpustack/gpustack:{{version}} \\
    --system-default-container-registry $PrivateRegistry</code></pre>
                    </li>
                </ol>
    `,
    "guide_auto_content": `
                <p>Для автоматизации синхронизации GPUStack предоставляет команды управления образами:</p>
                <ul style="list-style:none; padding-left:0">
                    <li style="margin-bottom:8px"><code>gpustack copy-images</code>: синхронизировать образы из исходного реестра в целевой</li>
                    <li style="margin-bottom:8px"><code>gpustack save-images</code>: загрузить и сохранить образы по локальному пути</li>
                    <li style="margin-bottom:8px"><code>gpustack load-images</code>: импортировать образы из локальных архивов</li>
                    <li style="margin-bottom:8px"><code>gpustack list-images</code>: вывести список образов текущей версии</li>
                </ul>
                <div style="margin-top:15px; font-size:13px; color:var(--text-color)">
                    Все команды поддерживают фильтрацию и собственную конфигурацию. Подробности:
                    <a href="../installation/air-gapped/#container-images" target="_blank" style="color:var(--primary-color); font-weight:600">Подготовка образов контейнеров &rarr;</a>
                </div>
    `,
    "view_full_docs": "Полная инструкция по автономной установке &rarr;",
    "copied": "Скопировано в буфер обмена",
    "no_images": "# Подходящие образы не найдены",
    "comments": {
        "main": "Образ GPUStack — основные службы GPUStack; необходим на узлах Server и Worker",
        "runner": "Образы бэкендов инференса",
        "pause": "Образ Pause — общая сеть и IPC для контейнеров экземпляров моделей; нужен только в Docker",
        "benchmark": "Образ Benchmark — тестирование производительности моделей",
        "postgres": "PostgreSQL — дополнительный компонент для отдельного развертывания внешней базы данных",
        "monitoring": "Мониторинг — дополнительные компоненты Prometheus и Grafana",
        "gateway": "Образы шлюза — нужны только при развертывании в Kubernetes",
        "k8s": "Образы GPU Service — нужны только при развертывании в Kubernetes"
    },
    "cards": {
        "nvidia": "NVIDIA",
        "amd": "AMD",
        "ascend": "Ascend",
        "hygon": "Hygon",
        "mthreads": "MThreads",
        "iluvatar": "Iluvatar",
        "cambricon": "Cambricon",
        "maca": "MetaX",
        "t-head": "T-Head PPU"
    },
    "language_name": "Русский",
    "description": "Выбор образов контейнеров: команды загрузки Docker для вашей конфигурации оборудования.",
    "auto_arch": "Авто",
    "copy_all": "Скопировать все команды",
    "server_image_placeholder": "&lt;Список образов Server&gt;",
    "worker_image_placeholder": "&lt;Список образов Worker&gt;",
    "configuration_placeholder": "# Выберите конфигурацию для формирования команд",
    "load_failed": "Не удалось загрузить данные",
    "cambricon_hint": "Обратитесь к поставщику Cambricon за образами бэкенда инференса",
    "no_available_images": "Нет доступных образов",
    "select_version": "Выберите версию"
};

CONFIG.i18n.tr = {
    "title": "Konteyner İmajı Seçici",
    "subtitle": "Konteyner İmajı Seçici",
    "back_to_docs": "Belgeler",
    "version": "Sürüm",
    "loading": "Yükleniyor...",
    "select_config": "Yapılandırma",
    "gpu_type": "GPU türü",
    "framework_version": "Hesaplama altyapısı sürümü",
    "select_gpu_first": "Önce GPU türünü seçin",
    "inference_backend": "Çıkarım arka ucu",
    "inference_backend_tooltip": "İstediğiniz yerleşik çıkarım arka ucunu veya sürümünü bulamazsanız daha eski bir hesaplama altyapısı sürümü seçmeyi deneyin. Yeni sürücüler genellikle eski altyapı sürümleriyle uyumludur.",
    "optional_images": "İsteğe bağlı imajlar",
    "optional_postgres_images_tooltip": "Harici veritabanı kullanımı: <a href=\"../installation/installation/#using-an-external-database\" target=\"_blank\">yapılandırma kılavuzu</a>",
    "optional_monitoring_images_tooltip": "Harici gözlemlenebilirlik yapılandırması: <a href=\"../user-guide/observability/#external-observability-optional\" target=\"_blank\">yapılandırma kılavuzu</a>",
    "postgres": "PostgreSQL",
    "monitoring": "İzleme paketi (Prometheus + Grafana)",
    "required_images": "Gerekli imajlar",
    "architecture": "Mimari",
    "registry": "İmaj kayıt deposu",
    "china_registry": "Çin için ayna",
    "tab_all": "Tümü",
    "tab_server": "Server düğümü",
    "tab_worker": "Worker düğümü",
    "placeholder_select": "İmaj listesini oluşturmak için soldan yapılandırmayı seçin",
    "offline_guide_title": "Çevrimdışı kurulum mu gerekiyor?",
    "guide_tab_save": "İmaj dosyalarını içe aktarma (Save & Load)",
    "guide_tab_tag": "Özel kayıt deposuna gönderme (Tag & Push)",
    "guide_tab_auto": "Otomatik imaj eşitleme",
    "guide_save_content": `
                <p>Tamamen yalıtılmış ortamlarda imajları dosyalar aracılığıyla içe aktarmak için uygundur.</p>
                <ol>
                    <li>İnternete bağlı bir makinede imajları indirin (yukarıdaki komutlara bakın).</li>
                    <li>İmajları tar dosyalarına aktarın:
                        <pre><code>docker save -o gpustack-server-images.tar {{server_images}}

docker save -o gpustack-worker-images.tar {{worker_images}}</code></pre>
                    </li>
                    <li>Dosyaları çevrimdışı makineye kopyalayın.</li>
                    <li>İmajları düğümün rolüne göre içe aktarın:
                        <div style="margin-top:10px"><strong>Server düğümü:</strong></div>
                        <pre><code>docker load -i gpustack-server-images.tar</code></pre>
                        <div style="margin-top:10px"><strong>Worker düğümü:</strong></div>
                        <pre><code>docker load -i gpustack-worker-images.tar</code></pre>
                        <div style="margin-top:10px"><strong>Server + Worker düğümü:</strong></div>
                        <pre><code>docker load -i gpustack-server-images.tar
docker load -i gpustack-worker-images.tar</code></pre>
                    </li>
                    <li>
                        GPUStack Server ve Worker konteynerlerini çalıştırırken imaj kayıt deposunu şu parametreyle belirtin: <code class="inline-code">--system-default-container-registry</code>:
                        <pre><code>sudo docker run -d --name gpustack \\
    --restart unless-stopped \\
    -p 80:80 \\
    -p 10161:10161 \\
    --volume gpustack-data:/var/lib/gpustack \\
    {{registry}}/gpustack/gpustack:{{version}} \\
    --system-default-container-registry {{registry}}</code></pre>
                    </li>
                </ol>
    `,
    "guide_tag_content": `
                <p>Harbor veya Nexus gibi özel bir kayıt deposunun bulunduğu ortamlar için uygundur.</p>
                <ol>
                    <li>İnternete bağlı bir makinede imajları indirin (yukarıdaki komutlara bakın).</li>
                    <li>İmajları yeniden etiketleyip özel kayıt deposuna gönderin:
                        <pre><code>export PrivateRegistry=&lt;your-private-registry&gt;
{{tag_push_commands}}</code></pre>
                    </li>
                    <li>Konteynerleri çalıştırırken başlangıç parametreleriyle imaj kayıt deposunu belirtin:
                        <pre><code>sudo docker run -d --name gpustack \\
    --restart unless-stopped \\
    -p 80:80 \\
    -p 10161:10161 \\
    --volume gpustack-data:/var/lib/gpustack \\
    $PrivateRegistry/gpustack/gpustack:{{version}} \\
    --system-default-container-registry $PrivateRegistry</code></pre>
                    </li>
                </ol>
    `,
    "guide_auto_content": `
                <p>Eşitlemeyi otomatikleştirmek için GPUStack şu imaj yönetimi komutlarını sunar:</p>
                <ul style="list-style:none; padding-left:0">
                    <li style="margin-bottom:8px"><code>gpustack copy-images</code>: imajları kaynak kayıt deposundan hedef kayıt deposuna eşitler</li>
                    <li style="margin-bottom:8px"><code>gpustack save-images</code>: imajları indirip yerel bir yola kaydeder</li>
                    <li style="margin-bottom:8px"><code>gpustack load-images</code>: imajları yerel paketlerden içe aktarır</li>
                    <li style="margin-bottom:8px"><code>gpustack list-images</code>: geçerli sürümün imaj listesini gösterir</li>
                </ul>
                <div style="margin-top:15px; font-size:13px; color:var(--text-color)">
                    Tüm komutlar filtrelemeyi ve özel yapılandırmayı destekler. Ayrıntılar:
                    <a href="../installation/air-gapped/#container-images" target="_blank" style="color:var(--primary-color); font-weight:600">Konteyner imajlarını hazırlama &rarr;</a>
                </div>
    `,
    "view_full_docs": "Çevrimdışı kurulum belgelerinin tamamını görüntüle &rarr;",
    "copied": "Panoya kopyalandı",
    "no_images": "# Eşleşen imaj bulunamadı",
    "comments": {
        "main": "GPUStack imajı — GPUStack temel hizmetleri; Server ve Worker düğümlerinde gereklidir",
        "runner": "Çıkarım arka ucu imajları",
        "pause": "Pause imajı — model örneği konteynerleri için ortak ağ ve IPC ortamı; yalnızca Docker için gereklidir",
        "benchmark": "Benchmark imajı — model performans testlerini çalıştırır",
        "postgres": "PostgreSQL — harici veritabanını ayrı dağıtmak için isteğe bağlı bileşen",
        "monitoring": "İzleme paketi — isteğe bağlı Prometheus ve Grafana bileşenleri",
        "gateway": "Ağ geçidi imajları — yalnızca Kubernetes dağıtımları için gereklidir",
        "k8s": "GPU Service imajları — yalnızca Kubernetes dağıtımları için gereklidir"
    },
    "cards": {
        "nvidia": "NVIDIA",
        "amd": "AMD",
        "ascend": "Ascend",
        "hygon": "Hygon",
        "mthreads": "MThreads",
        "iluvatar": "Iluvatar",
        "cambricon": "Cambricon",
        "maca": "MetaX",
        "t-head": "T-Head PPU"
    },
    "language_name": "Türkçe",
    "description": "Konteyner İmajı Seçici: donanım yapılandırmanıza göre Docker imajı indirme komutları oluşturun.",
    "auto_arch": "Otomatik",
    "copy_all": "Tüm komutları kopyala",
    "server_image_placeholder": "&lt;Server imaj listesi&gt;",
    "worker_image_placeholder": "&lt;Worker imaj listesi&gt;",
    "configuration_placeholder": "# Komut oluşturmak için yapılandırmayı seçin",
    "load_failed": "Veriler yüklenemedi",
    "cambricon_hint": "Çıkarım arka ucu imajları için Cambricon sağlayıcısıyla iletişime geçin",
    "no_available_images": "Kullanılabilir imaj yok",
    "select_version": "Sürüm seçin"
};

function resolveLanguage(language) {
    return Object.hasOwn(CONFIG.i18n, language) ? language : 'en';
}

function escapeHTML(value) {
    const entities = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
    return String(value).replace(/[&<>"']/g, character => entities[character]);
}

// Global State - Default to English
let state = {
    currentLang: resolveLanguage(document.documentElement.dataset.docsLanguage || localStorage.getItem('lang') || 'en'),
    images: [], runnerImages: [], supportMatrix: { frameworks: {} },
    selectedComponent: 'all', selectedArch: 'amd64', selectedRegistry: 'docker-hub',
    selectedCard: null, selectedFrameworkVersion: null, selectedChipType: null,
    selectedBackends: [], availableVersions: [], selectedGpuStackVersion: null,
    optionalImages: { 'postgres': false, 'monitoring': false }, selectedGuide: 'save-load'
};

// DOM Elements
const elements = {};
let initializing = true;
let pendingSelection = null;
const manifestSelectionKeys = ['selectedGpuStackVersion', 'selectedCard',
    'selectedFrameworkVersion', 'selectedChipType', 'selectedBackends'];

// Only user choices belong in the URL; manifests and available options are loaded afresh.
function getSelection() {
    const { selectedGpuStackVersion, selectedComponent, selectedCard,
        selectedFrameworkVersion, selectedChipType, selectedArch, selectedRegistry,
        selectedBackends, optionalImages, selectedGuide } = state;
    return { selectedGpuStackVersion, selectedComponent, selectedCard,
        selectedFrameworkVersion, selectedChipType, selectedArch, selectedRegistry,
        selectedBackends, optionalImages, selectedGuide };
}

function readSelection() {
    try {
        const saved = JSON.parse(new URLSearchParams(window.location.search).get('selector'));
        if (!saved || typeof saved !== 'object' || Array.isArray(saved)) return {};
        const selection = {};
        const options = {
            selectedComponent: ['all', 'server', 'worker'],
            selectedArch: ['amd64', 'arm64', 'auto'],
            selectedRegistry: Object.keys(CONFIG.registries),
            selectedCard: Object.keys(CONFIG.cardFrameworkMap),
            selectedGuide: ['save-load', 'tag-push', 'copy-images']
        };
        Object.entries(options).forEach(([key, values]) => {
            if (values.includes(saved[key])) selection[key] = saved[key];
        });
        // Version, framework, chip and backend choices are checked against the selected manifest.
        ['selectedGpuStackVersion', 'selectedFrameworkVersion', 'selectedChipType'].forEach(key => {
            if (typeof saved[key] === 'string') selection[key] = saved[key];
        });
        if (Array.isArray(saved.selectedBackends)) {
            selection.selectedBackends = saved.selectedBackends.filter(value => typeof value === 'string');
        }
        if (saved.optionalImages && typeof saved.optionalImages === 'object') {
            selection.optionalImages = {
                postgres: saved.optionalImages.postgres === true,
                monitoring: saved.optionalImages.monitoring === true
            };
        }
        return selection;
    } catch {
        return {};
    }
}

function selectionUrl(href) {
    const url = new URL(href, window.location.href);
    // Unverified candidates stay in the URL until their manifests can be loaded.
    url.searchParams.set('selector', JSON.stringify({...getSelection(), ...pendingSelection}));
    return url;
}

function persistSelection() {
    if (initializing || !window.history?.replaceState) return;
    const url = selectionUrl(window.location.href);
    window.history.replaceState(window.history.state, '', url.href);
}

// Initialization
async function init() {
    const saved = readSelection();
    pendingSelection = Object.fromEntries(manifestSelectionKeys
        .filter(key => Object.hasOwn(saved, key)).map(key => [key, saved[key]]));
    initElements();
    bindEvents();
    updateLanguage();
    if (state.currentLang === 'zh' && !saved.selectedRegistry) state.selectedRegistry = 'china';
    if (saved.selectedArch) selectArch(saved.selectedArch);
    selectRegistry(saved.selectedRegistry || state.selectedRegistry);
    selectComponent(saved.selectedComponent || state.selectedComponent);
    if (saved.optionalImages) state.optionalImages = saved.optionalImages;
    Object.keys(elements.optionalImages).forEach(key => {
        elements.optionalImages[key].checked = state.optionalImages[key];
    });
    switchGuideTab(saved.selectedGuide || state.selectedGuide);
    if (await loadData(saved.selectedGpuStackVersion)) {
        restoreManifestSelection(Boolean(saved.selectedArch));
    }
    initializing = false;
    generateImageList();
    // Show content after initialization
    document.body.classList.add('i18n-ready');
}

function restoreManifestSelection(preserveArch = false) {
    const saved = pendingSelection;
    if (!saved) return;
    const arch = state.selectedArch;
    // Resolve URL candidates through the loaded options before they become live state.
    const card = Array.from(elements.cardSelector.querySelectorAll('.option-button'))
        .find(button => button.dataset.value === saved.selectedCard && !button.disabled);
    if (card) selectCard(card.dataset.value);
    const framework = Array.from(elements.frameworkVersionSelect.querySelectorAll('option:not([value=""])'))
        .find(option => option.value === saved.selectedFrameworkVersion
            && (option.dataset.chipType || null) === (saved.selectedChipType || null));
    if (framework) {
        framework.selected = true;
        selectFrameworkVersion(framework.value);
    }
    if (saved.selectedBackends) {
        elements.backendSelector.querySelectorAll('input').forEach(input => {
            input.checked = saved.selectedBackends.includes(input.value);
        });
        updateSelectedBackends();
    }
    if (preserveArch) selectArch(arch);
    pendingSelection = null;
}

function discardPendingSelection(...keys) {
    if (pendingSelection) keys.forEach(key => { delete pendingSelection[key]; });
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
        if (btn && !btn.disabled) {
            discardPendingSelection('selectedCard', 'selectedFrameworkVersion', 'selectedChipType', 'selectedBackends');
            selectCard(btn.dataset.value);
        }
    });
    elements.frameworkVersionSelect.addEventListener('change', (e) => {
        discardPendingSelection('selectedFrameworkVersion', 'selectedChipType', 'selectedBackends');
        selectFrameworkVersion(e.target.value);
    });
    elements.backendSelector.addEventListener('change', () => {
        discardPendingSelection('selectedBackends');
        updateSelectedBackends();
    });
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
            if (document.documentElement.dataset.docsLanguage) {
                const target = new URL(link.href, window.location.href);
                if (target.origin !== window.location.origin || !/\/image-selector\/(?:index\.html)?$/.test(target.pathname)) return;
                e.preventDefault();
                target.search = window.location.search;
                target.hash = window.location.hash;
                window.location.assign(selectionUrl(target.href).href);
                return;
            }
            e.preventDefault();
            setLanguage(link.dataset.lang);
        });
    });
}

// Set application language
function setLanguage(lang) {
    lang = resolveLanguage(lang);
    if (state.currentLang === lang) return;
    state.currentLang = lang;
    localStorage.setItem('lang', lang);
    updateLanguage();
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
        // Gateway and GPU service images - only required for Kubernetes deployments
        state.images.filter(img => {
            const name = img.split(':')[0];
            return name.startsWith('gpustack/mirrored-') || ['gpustack/higress-plugins', 'gpustack/gpustack-operator', 'gpustack/ssh-server'].includes(name);
        }).forEach(img => {
            const [fullName, tag] = img.split(':');
            imgs.push(getFullImageName(fullName.split('/').pop(), tag, state.selectedRegistry));
        });
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

    elements.currentLangBtn.textContent = data.language_name;
    document.documentElement.lang = lang;
    const description = document.querySelector('meta[name="description"]');
    if (description) description.content = data.description;
    document.querySelectorAll('[data-i18n-title]').forEach(el => {
        el.title = data[el.dataset.i18nTitle];
    });

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

    // The templates contain trusted HTML; dynamic image and version values are text.
    const substitutions = {
        version: escapeHTML(version),
        server_images: serverImgs ? escapeHTML(serverImgs) : data.server_image_placeholder,
        worker_images: workerImgs ? escapeHTML(workerImgs) : data.worker_image_placeholder,
        tag_push_commands: escapeHTML(tagPushCmds || data.configuration_placeholder),
        registry: escapeHTML(CONFIG.registries[state.selectedRegistry].registry)
    };
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.dataset.i18n;
        if (data[key]) {
            el.innerHTML = data[key].replace(/{{(version|server_images|worker_images|tag_push_commands|registry)}}/g,
                (_, name) => substitutions[name]);
        }
    });
    const copyImagesEl = document.getElementById('copy-images-command');
    if (copyImagesEl) copyImagesEl.textContent = data.copy_images_cmd;

    // Keep a configured mirror visible when the document language changes.
    elements.registryChina.style.display = (lang === 'zh' || state.selectedRegistry === 'china') ? 'block' : 'none';
}

// Load versions and initial data
async function loadData(requestedVersion = state.selectedGpuStackVersion) {
    try {
        const response = await fetch(CONFIG.versionsUrl);
        const data = await response.json();
        const versions = Array.isArray(data) ? data : data?.versions;
        if (!Array.isArray(versions) || !versions.length || versions.some(version => typeof version !== 'string' || !version)) {
            throw new Error('Invalid version manifest');
        }
        state.availableVersions = versions;
        state.selectedGpuStackVersion = state.availableVersions.includes(requestedVersion)
            ? requestedVersion : state.availableVersions[0];
        if (pendingSelection) pendingSelection.selectedGpuStackVersion = state.selectedGpuStackVersion;
        renderVersionSelector();
        return await loadVersionData(state.selectedGpuStackVersion);
    } catch (error) {
        console.error('Failed to load data:', error);
        showToast(CONFIG.i18n[state.currentLang].load_failed);
        return false;
    }
}

// Load specific version data
async function loadVersionData(version) {
    const response = await fetch(`${CONFIG.versionsBaseUrl}${version}.json`);
    const data = await response.json();
    if (version !== state.selectedGpuStackVersion) return false;
    if (!Array.isArray(data) || data.some(image => typeof image !== 'string')) {
        throw new Error('Invalid image manifest');
    }
    state.images = data;
    state.runnerImages = data.filter(img => img.startsWith('gpustack/runner:'));
    parseSupportMatrix();
    const arch = state.selectedArch;
    renderCardSelector();
    if (pendingSelection) selectArch(arch);
    updateLanguage(); // Refresh offline guide version number
    return true;
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
    if (!state.availableVersions.includes(v)) return;
    state.selectedGpuStackVersion = v;
    if (pendingSelection) {
        pendingSelection.selectedGpuStackVersion = v;
        persistSelection();
    }
    try {
        if (!await loadVersionData(v)) return;
        restoreManifestSelection(true);
        generateImageList();
    } catch (error) {
        console.error('Failed to load data:', error);
        showToast(CONFIG.i18n[state.currentLang].load_failed);
    }
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
            const tip = CONFIG.i18n[state.currentLang].cambricon_hint;
            btn.innerHTML += `<span class="tooltip-icon">i <span class="tooltip-content">${tip}</span></span>`;
        }
        btn.disabled = !hasData;
        if (!hasData) btn.title = CONFIG.i18n[state.currentLang].no_available_images;
        elements.cardSelector.appendChild(btn);
    });
    const buttons = Array.from(elements.cardSelector.querySelectorAll('.option-button'));
    const activeBtn = buttons.find(button => button.dataset.value === state.selectedCard && !button.disabled)
        || buttons.find(button => !button.disabled);
    if (activeBtn) selectCard(activeBtn.dataset.value, activeBtn.dataset.value === state.selectedCard);
}

// Handle GPU Type selection
function selectCard(cardId, preserveSelection = false) {
    const saved = preserveSelection ? getSelection() : null;
    state.selectedCard = cardId;
    elements.cardSelector.querySelectorAll('.option-button').forEach(b => b.classList.toggle('active', b.dataset.value === cardId));
    const fw = CONFIG.cardFrameworkMap[cardId].toLowerCase();
    renderFrameworkVersions(fw, cardId);
    const options = Array.from(elements.frameworkVersionSelect.querySelectorAll('option:not([value=""])'));
    const selected = options.find(option => saved && option.value === saved.selectedFrameworkVersion
        && (option.dataset.chipType || null) === saved.selectedChipType) || options[0];
    if (selected) { selected.selected = true; selectFrameworkVersion(selected.value); }
    if (saved) {
        const inputs = Array.from(elements.backendSelector.querySelectorAll('input'));
        inputs.forEach(input => { input.checked = saved.selectedBackends.includes(input.value); });
        state.selectedBackends = inputs.filter(input => input.checked).map(input => input.value);
    }
    const targetArch = saved ? saved.selectedArch : cardId === 'ascend' ? 'arm64' : 'amd64';
    if (state.selectedArch !== targetArch) selectArch(targetArch);
    generateImageList();
}

// Render Framework Version dropdown
function renderFrameworkVersions(fw, cardId) {
    const select = elements.frameworkVersionSelect;
    const placeholder = CONFIG.i18n[state.currentLang].select_version;
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
        Array.from(combos.values()).sort((a,b) => {
            const versionCompare = b.v.localeCompare(a.v, undefined, {numeric: true, sensitivity: 'base'});
            if (versionCompare !== 0) {
                return versionCompare;
            }
            const aIs910B = a.c.toUpperCase().includes('910B');
            const bIs910B = b.c.toUpperCase().includes('910B');
            if (aIs910B && !bIs910B) return -1;
            if (!aIs910B && bIs910B) return 1;
            return b.c.localeCompare(a.c);
        }).forEach(item => {
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
        lbl.innerHTML = `<input type="checkbox" value="${escapeHTML(k)}"><span>${escapeHTML(name)}</span>`;
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
    persistSelection();
    const plat = state.selectedArch === 'auto' ? '' : ` --platform linux/${state.selectedArch}`;
    const cmds = [];
    const isServer = state.selectedComponent === 'server' || state.selectedComponent === 'all';
    const isWorker = state.selectedComponent === 'worker' || state.selectedComponent === 'all';
    const t = CONFIG.i18n[state.currentLang];
    
    if (state.selectedGpuStackVersion) {
        cmds.push(`# ${t.comments.main}`);
        cmds.push(`docker pull${plat} ${getFullImageName('gpustack', state.selectedGpuStackVersion, state.selectedRegistry)}`);
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
            rCmds.push(`docker pull${plat} ${getFullImageName('runner', tag, state.selectedRegistry)}`);
        });
        if (rCmds.length) {
            const comment = t.comments.runner;
            cmds.push('');
            cmds.push(`# ${comment}`);
            cmds.push(...rCmds); 
        }
        
        const pause = state.images.find(i => i.includes('runtime:pause'));
        if (pause) {
            const comment = t.comments.pause;
            cmds.push('');
            cmds.push(`# ${comment}`);
            cmds.push(`docker pull${plat} ${getFullImageName('runtime', pause.split(':')[1], state.selectedRegistry)}`); 
        }
        
        const bm = state.images.find(i => i.includes('benchmark-runner'));
        if (bm) {
            const comment = t.comments.benchmark;
            cmds.push('');
            cmds.push(`# ${comment}`);
            cmds.push(`docker pull${plat} ${getFullImageName('benchmark-runner', bm.split(':')[1], state.selectedRegistry)}`); 
        }
    }

    if (isServer) {
        if (state.optionalImages['postgres']) {
            const pgs = state.images.filter(i => i.startsWith('postgres:'));
            if (pgs.length) {
                const comment = t.comments.postgres;
                cmds.push('');
                cmds.push(`# ${comment}`);
                pgs.forEach(i => cmds.push(`docker pull${plat} ${getFullImageName('postgres', i.split(':')[1], state.selectedRegistry)}`)); 
            }
        }
        if (state.optionalImages['monitoring']) {
            cmds.push('');
            const comment = t.comments.monitoring;
            cmds.push(`# ${comment}`);
            state.images.filter(i => i.includes('prometheus')).forEach(i => cmds.push(`docker pull${plat} ${getFullImageName('prometheus', i.split(':')[1], state.selectedRegistry)}`));
            state.images.filter(i => i.includes('grafana')).forEach(i => cmds.push(`docker pull${plat} ${getFullImageName('grafana', i.split(':')[1], state.selectedRegistry)}`));
        }
        // Gateway images - only required for Kubernetes deployments
        const gatewayImgs = state.images.filter(img => {
            const name = img.split(':')[0];
            return name.startsWith('gpustack/mirrored-higress-') || name === 'gpustack/higress-plugins';
        });
        if (gatewayImgs.length) {
            cmds.push('');
            cmds.push(`# ${t.comments.gateway}`);
            gatewayImgs.forEach(img => {
                const [fullName, tag] = img.split(':');
                cmds.push(`docker pull${plat} ${getFullImageName(fullName.split('/').pop(), tag, state.selectedRegistry)}`);
            });
        }
        // GPU service images - only required for Kubernetes deployments
        const gpuServiceImgs = state.images.filter(img => {
            const name = img.split(':')[0];
            return ['gpustack/gpustack-operator', 'gpustack/ssh-server'].includes(name) ||
                   (name.startsWith('gpustack/mirrored-') && !name.startsWith('gpustack/mirrored-higress-'));
        });
        if (gpuServiceImgs.length) {
            cmds.push('');
            cmds.push(`# ${t.comments.k8s}`);
            gpuServiceImgs.forEach(img => {
                const [fullName, tag] = img.split(':');
                cmds.push(`docker pull${plat} ${getFullImageName(fullName.split('/').pop(), tag, state.selectedRegistry)}`);
            });
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
function switchGuideTab(id) { state.selectedGuide = id; elements.guideTabs.forEach(t => t.classList.toggle('active', t.dataset.guide === id)); document.querySelectorAll('.guide-content').forEach(c => c.classList.toggle('active', c.id === `guide-${id}`)); persistSelection(); }

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
