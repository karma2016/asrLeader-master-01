# Ascend NPU 生产更新手册

本文记录 2026-06-12 最终跑通的方案。以后更新 NPU 版本时，以本手册为准。
CPU 镜像、CPU Dockerfile 和 CPU 服务不在本流程内。

## 1. 最终结论

生产环境不能使用通用 GitHub NPU 基础镜像重新安装 torch、torch-npu 或
CANN。最终镜像必须继承生产中已经验证可用的旧底座：

```text
meeting-asr:npu-new-oldbase
```

推荐构建位置是持有该镜像的 Ascend ARM64 生产 Worker。Windows 只生成包含
新代码、NPU 启动脚本和 Qwen 模型的源码覆盖包，不在 Windows 上验证 NPU。

2026-06-12 验证成功的最终镜像：

```text
172.20.242.100:30500/dream_acr/funasr-leader-asr:npu-prod-final-20260612-v2
```

已验证：

- Pod `1/1 Running`，重启次数为 0；
- `torch 2.1.0`；
- `torch-npu 2.1.0.post10`；
- `torch.npu.device_count() == 1`；
- FunASR `1.3.1`；
- Qwen2.5-1.5B-Instruct 在 `npu:0` 预加载成功；
- `/health`、`/transcribe` 和 Pod 内的 `/leaders` 正常；
- 未出现 `507008`、动态库符号错误或 NPU OOM。

## 2. 本次失败的根因

### 2.1 torch-npu、CANN 与生产驱动不匹配

新 GitHub 镜像中的 torch-npu/CANN 组合与生产 Ascend 驱动不完全兼容。
最早的直接表现是：

```text
torch.npu.set_device() -> 507008
```

这不是千问模型本身的问题。NPU 基础运行时无法初始化时，FunASR 和 Qwen 都
不可能稳定运行。

解决方法：不再重新选择 torch-npu/CANN 版本，直接继承生产已运行成功的
`meeting-asr:npu-new-oldbase`。

### 2.2 Windows x86 不能完成真实 NPU 验证

Windows Docker Desktop 可以保存或交叉构建 `linux/arm64` 镜像，但本机没有
Ascend 驱动和 NPU，不能验证：

- torch-npu 是否能加载生产驱动；
- CANN 动态库是否兼容；
- `torch.npu.device_count()` 是否正确；
- ASR 与 Qwen 同卡时的显存占用。

因此，本地构建成功不代表生产可用。最终构建和 NPU 验证必须在生产 Ascend
Worker 完成。

### 2.3 LD_LIBRARY_PATH 顺序被新启动脚本破坏

新代码最初重新拼接了通用 CANN 库目录，覆盖了旧镜像实际使用的动态库顺序，
导致：

```text
ImportError: libacl_tdt_channel.so: undefined symbol: _ZN3acl18AclErrorLogManagerD1Ev
```

对比后确认 torch、torch-npu 和相关 `.so` 文件本身一致，差别是
`LD_LIBRARY_PATH`。最终修复是在 NPU entrypoint 中保留生产旧 Pod 的精确路径
和顺序，不能自行排序或只保留几个“看起来必要”的目录。

### 2.4 FunASR 1.3.1 的模型解析方式

生产旧底座使用 FunASR 1.3.1。将 ASR 模型别名强制解析成
`/app/models/paraformer-zh` 等本地目录后，会报：

```text
/app/models/paraformer-zh is not registered
```

必须保持：

```text
ASR_RESOLVE_LOCAL_MODELS=false
```

ASR/VAD/PUNC/SPK 使用生产镜像原有的 ModelScope 缓存和注册名称。Qwen 则通过
`POSTPROCESS_MODEL_DIR=/app/models/Qwen2.5-1.5B-Instruct` 使用镜像内绝对路径。

### 2.5 NPU 设备插件分配与真实占用不一致

worker06 的物理卡 5 已被 `acgpt` 的 VLLM 进程占用约 56 GB，但 Kubernetes
设备插件仍把卡 5 视为可分配。新 ASR Pod 被分配到卡 5 后，Qwen 预加载 OOM。

临时解决方法：

- 保留 `ascend-card5-guard`，让 Kubernetes 逻辑上占住卡 5；
- 新 ASR Pod 因此落到干净的物理卡 0；
- 不停止或杀死 `acgpt` 进程。

长期应修复 `acgpt` 的 NPU 资源声明，让 Kubernetes 正确记录它使用卡 5。
在此之前，不要删除 guard，也不要让新 ASR 随意重调度。

### 2.6 `/leaders` 的外部 404 不在镜像内

Pod 内请求：

```bash
curl http://10.244.4.210:8000/leaders
```

返回 200，说明新镜像注册了 `/leaders`。外部
`http://10.91.4.250:30530/leaders` 返回：

```json
{"error_msg":"404 Route Not Found"}
```

该响应格式也不是 FastAPI 默认 404，说明 404 来自外部网关或端口转发层。
需要在 `10.91.4.250:30530` 对应的网关中增加 `/leaders` 路由。镜像无需为此
重新打包。

## 3. 已验证的生产版本

旧底座中已验证的主要版本：

```text
Python       3.11.6
funasr       1.3.1
modelscope   1.36.0
torch        2.1.0
torch-npu    2.1.0.post10
fastapi      0.136.0
pydantic     2.9.2
transformers 4.44.0
```

更新业务代码时不要升级以上 NPU 核心依赖。确需升级时，必须创建独立测试
Deployment，完成驱动、CANN、torch-npu 和模型全链路验证后再切流。

## 4. 每次更新前的检查

在 Windows 仓库执行：

```powershell
cd D:\Docker\asr-funasr-leade-2\asrLeader-master
git status --short
python -m compileall -q app
```

确认 Qwen 模型完整：

```powershell
Get-Item deploy\ascend-npu\models\Qwen2.5-1.5B-Instruct\model.safetensors
```

该文件应大于 1 GB。不要把模型下载动作放到生产启动流程，生产环境不联网。

只允许修改：

- `app/` 中的新业务代码；
- `docker/entrypoint.npu.sh`；
- NPU 专用 Dockerfile、脚本和 Deployment。

不要修改 CPU Dockerfile、CPU requirements 或已经验证通过的 CPU 启动流程。

## 5. Windows 生成上传包

版本号使用当天日期加递增后缀，禁止覆盖旧标签，例如：

```text
20260620-v1
20260620-v2
```

执行：

```powershell
cd D:\Docker\asr-funasr-leade-2\asrLeader-master
powershell -ExecutionPolicy Bypass `
  -File scripts\package_npu_prod_oldbase.ps1 `
  -Version 20260620-v1
```

产物：

```text
asr-leader-npu-prod-oldbase-20260620-v1.tar
```

这个 tar 是源码覆盖包，不是 Docker 镜像。将它上传到持有
`meeting-asr:npu-new-oldbase` 的生产 Ascend Worker，例如：

```text
/data/leader-asr
```

## 6. 在生产 Worker 构建最终 ARM64 镜像

先确认旧底座存在：

```bash
nerdctl -n k8s.io image inspect meeting-asr:npu-new-oldbase
```

解压并构建：

```bash
cd /data/leader-asr
tar -xf asr-leader-npu-prod-oldbase-20260620-v1.tar
cd asr-leader-npu-prod-oldbase-20260620-v1
chmod 755 scripts/build_npu_prod_oldbase_on_prod.sh

BASE_IMAGE=meeting-asr:npu-new-oldbase \
IMAGE_TAG=funasr-leader-asr:npu-prod-20260620-v1 \
bash scripts/build_npu_prod_oldbase_on_prod.sh
```

推送到生产内部镜像仓库：

```bash
nerdctl -n k8s.io tag \
  funasr-leader-asr:npu-prod-20260620-v1 \
  172.20.242.100:30500/dream_acr/funasr-leader-asr:npu-prod-20260620-v1

nerdctl -n k8s.io push \
  172.20.242.100:30500/dream_acr/funasr-leader-asr:npu-prod-20260620-v1
```

禁止复用已经发布过的 tag。Kubernetes 的 `IfNotPresent` 配合复用 tag 容易继续
使用节点缓存中的旧镜像。

## 7. 推荐的无风险更新顺序

不要直接对正在承接流量的新 Deployment 做滚动更新。NPU 资源不足时，
RollingUpdate 会尝试同时启动两个 Pod，可能长时间 Pending 或被分配到错误卡。

### 7.1 先把流量切回旧服务

旧服务 `dream-acr-new` 必须保持 `1/1 Running`。

备份 Service：

```bash
mkdir -p /data/leader-asr/k8s-backup
ts=$(date +%Y%m%d%H%M%S)
kubectl -n test get svc dream-acr -o yaml \
  > /data/leader-asr/k8s-backup/dream-acr-service-$ts.yaml
```

切回旧服务：

```bash
kubectl -n test patch svc dream-acr --type=merge \
  -p '{"spec":{"selector":{"app":"dream-acr-new"}}}'
```

确认 Endpoint 是旧 Pod：

```bash
kubectl -n test get endpoints dream-acr -o wide
kubectl -n test get pod -l app=dream-acr-new -o wide
```

### 7.2 停止旧版新服务并更新镜像

```bash
kubectl -n test scale deploy/asr-leader-npu --replicas=0

kubectl -n test set image deployment/asr-leader-npu \
  asr-leader-npu=172.20.242.100:30500/dream_acr/funasr-leader-asr:npu-prod-20260620-v1

kubectl -n test scale deploy/asr-leader-npu --replicas=1
kubectl -n test rollout status deploy/asr-leader-npu --timeout=15m
```

### 7.3 切流前验证

```bash
kubectl -n test get pod -l app=asr-leader-npu -o wide
kubectl -n test logs deploy/asr-leader-npu --tail=200 --timestamps
```

日志必须出现：

```text
torch_npu ready: devices=1
Application startup complete
Uvicorn running on http://0.0.0.0:8000
```

日志中不能出现：

```text
507008
undefined symbol
out of memory
Traceback
```

获取 Pod IP 并验证：

```bash
POD_IP=$(kubectl -n test get pod -l app=asr-leader-npu \
  -o jsonpath='{.items[0].status.podIP}')

curl -sS "http://${POD_IP}:8000/health"
curl -sS "http://${POD_IP}:8000/leaders"
```

`/health` 必须包含：

```text
"status":"ok"
"model_loaded":true
"device":"npu:0"
"postprocess_enabled":true
"postprocess_device":"npu:0"
```

再检查物理卡：

```bash
ssh root@worker06-910b npu-smi info
```

新 ASR 应落在预期空闲卡，当前方案为物理卡 0，显存约 9 GB。不要落到已被
VLLM 占满的卡 5。

最后使用真实音频直连 Pod 测试 `/transcribe`。只有识别结果正确、Qwen 后处理
正常、日志无异常，才允许切流。

### 7.4 切到新服务

```bash
kubectl -n test patch svc dream-acr --type=merge \
  -p '{"spec":{"selector":{"app":"asr-leader-npu"}}}'
```

确认：

```bash
kubectl -n test get svc dream-acr -o wide
kubectl -n test get endpoints dream-acr -o wide
curl -sS http://10.97.36.173:8000/health
curl -sS http://172.20.242.96:30031/health
```

调用方继续使用原地址和端口，无需修改。切流后打开实时日志：

```bash
kubectl -n test logs -f deploy/asr-leader-npu --since=1m --timestamps
```

## 8. 一键回滚流量

新服务出现任何异常时，立即执行：

```bash
kubectl -n test patch svc dream-acr --type=merge \
  -p '{"spec":{"selector":{"app":"dream-acr-new"}}}'
```

确认 Endpoint 已恢复到旧 Pod：

```bash
kubectl -n test get endpoints dream-acr -o wide
kubectl -n test get pod -l app=dream-acr-new -o wide
```

回滚只改变 Service selector，不删除新 Pod，也不需要重新启动旧服务，因此速度
最快。旧 `dream-acr-new` 在新版本稳定观察完成前不得停止。

如需回滚新 Deployment 的镜像：

```bash
kubectl -n test rollout history deploy/asr-leader-npu
kubectl -n test rollout undo deploy/asr-leader-npu
```

但 NPU 资源紧张时，优先使用“流量切回旧服务 + 新 Deployment 缩容重建”，
不要依赖滚动更新。

## 9. guard 的注意事项

当前：

```text
test/ascend-card5-guard
```

用于阻止设备插件再次把物理卡 5 分给 ASR。更新期间不要删除它。

检查：

```bash
kubectl -n test get pod ascend-card5-guard -o wide
```

它目前是裸 Pod，节点重启后不会自动恢复。长期应将其改为受控 Deployment，
或者从根本上修复 `acgpt` 的 NPU 资源声明。在完成长期修复前，每次重启
worker06 后都必须先重新核对卡 5 的真实进程和 Kubernetes 分配状态。

## 10. 为什么不推荐 GitHub Actions

通用 GitHub ARM runner 可以构建 `linux/arm64`，但拿不到生产私有旧底座，也
无法连接生产 Ascend 驱动进行真实验证。使用 `docker/Dockerfile.npu` 从通用
基础镜像重建，会重新引入本次的版本兼容风险。

只有满足以下条件时才可使用 GitHub：

1. `meeting-asr:npu-new-oldbase` 已推送到受控私有仓库；
2. Runner 能安全拉取该镜像；
3. Dockerfile 只覆盖业务代码和模型，不安装或升级 NPU 核心依赖；
4. 构建后仍在生产 Ascend 测试 Deployment 中完成全部验证。

目前最可靠流程仍是：Windows 打源码包，生产 Ascend Worker 基于旧底座构建。

## 11. 发布检查表

发布前逐项确认：

- [ ] CPU 文件未修改；
- [ ] Qwen `model.safetensors` 完整且已打包；
- [ ] 使用新且唯一的镜像 tag；
- [ ] 基础镜像是 `meeting-asr:npu-new-oldbase`；
- [ ] 没有 pip 安装或升级 torch、torch-npu、CANN、FunASR；
- [ ] `ASR_RESOLVE_LOCAL_MODELS=false`；
- [ ] `ASR_NORMALIZE_ASCEND_DEVICES=false`；
- [ ] `POSTPROCESS_PRELOAD=true`；
- [ ] `ascend-card5-guard` 正常；
- [ ] 旧 `dream-acr-new` 为 `1/1 Running`；
- [ ] 切流前 Pod 直连 `/health`、`/leaders`、`/transcribe` 全部通过；
- [ ] `npu-smi` 确认使用正确物理卡且显存充足；
- [ ] Service Endpoint 切换后指向新 Pod；
- [ ] NodePort 健康检查通过；
- [ ] 已记录回滚命令和 Service 备份路径。
