param(
    [string]$Version = "20260610",
    [string]$ImageTag = "",
    [bool]$IncludeQwen = $true,
    [string]$QwenSource = ""
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
if (-not $ImageTag) {
    $ImageTag = "funasr-leader-asr:npu-prod-oldbase-$Version"
}
if (-not $QwenSource) {
    $QwenSource = Join-Path $Root "deploy\ascend-npu\models\Qwen2.5-1.5B-Instruct"
}

$PackageName = "asr-leader-npu-prod-oldbase-$Version"
$PackageRoot = Join-Path $Root "deploy\$PackageName"
$TarPath = Join-Path $Root "$PackageName.tar"

if (Test-Path $PackageRoot) {
    Remove-Item -LiteralPath $PackageRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $PackageRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $PackageRoot "app") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $PackageRoot "docker") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $PackageRoot "models") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $PackageRoot "scripts") | Out-Null

Copy-Item -Path (Join-Path $Root "app\*") -Destination (Join-Path $PackageRoot "app") -Recurse -Force
Copy-Item -LiteralPath (Join-Path $Root "docker\entrypoint.npu.sh") -Destination (Join-Path $PackageRoot "docker") -Force
Copy-Item -LiteralPath (Join-Path $Root "docker\Dockerfile.npu-prod-oldbase") -Destination (Join-Path $PackageRoot "docker") -Force
Copy-Item -LiteralPath (Join-Path $Root "scripts\build_npu_prod_oldbase_on_prod.sh") -Destination (Join-Path $PackageRoot "scripts") -Force
Copy-Item -LiteralPath (Join-Path $Root "requirements-npu.txt") -Destination $PackageRoot -Force
Copy-Item -LiteralPath (Join-Path $Root "docs\npu-production-update-runbook.md") -Destination $PackageRoot -Force

$entrypointText = [System.IO.File]::ReadAllText((Join-Path $PackageRoot "docker\entrypoint.npu.sh"))
if (-not $entrypointText.Contains("torch_npu ready: devices=")) {
    throw "NPU entrypoint is missing the verified torch_npu preflight"
}
if (-not $entrypointText.Contains("/usr/local/Ascend/nnal/atb/8.0.0/atb/cxx_abi_1/lib")) {
    throw "NPU entrypoint is missing the verified production LD_LIBRARY_PATH"
}

if ($IncludeQwen) {
    if (-not (Test-Path -LiteralPath $QwenSource)) {
        throw "Qwen source directory not found: $QwenSource"
    }
    $qwenModelFile = Join-Path $QwenSource "model.safetensors"
    if (-not (Test-Path -LiteralPath $qwenModelFile)) {
        throw "Qwen model.safetensors not found: $qwenModelFile"
    }
    $qwenSize = (Get-Item -LiteralPath $qwenModelFile).Length
    if ($qwenSize -lt 1000000000) {
        throw "Qwen model.safetensors looks too small: $qwenSize bytes"
    }
    Copy-Item -LiteralPath $QwenSource -Destination (Join-Path $PackageRoot "models") -Recurse -Force
}

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$shellFiles = Get-ChildItem -LiteralPath $PackageRoot -Recurse -Force -File -Filter "*.sh"
foreach ($item in $shellFiles) {
    $content = [System.IO.File]::ReadAllText($item.FullName)
    $content = $content -replace "`r`n", "`n"
    [System.IO.File]::WriteAllText($item.FullName, $content, $utf8NoBom)
}

$pycache = Get-ChildItem -LiteralPath $PackageRoot -Recurse -Force -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue
foreach ($item in $pycache) {
    Remove-Item -LiteralPath $item.FullName -Recurse -Force
}
$pyc = Get-ChildItem -LiteralPath $PackageRoot -Recurse -Force -File -Filter "*.pyc" -ErrorAction SilentlyContinue
foreach ($item in $pyc) {
    Remove-Item -LiteralPath $item.FullName -Force
}

$readmeTemplate = @'
# ASR Leader NPU production oldbase package

This package is a production oldbase overlay. It contains the new app code and
the local Qwen2.5-1.5B-Instruct model, but it intentionally does not contain
the working production base image.

The production base image `meeting-asr:npu-new-oldbase` exists only on the
Ascend production worker. Build the final image on that worker so Python,
FunASR, torch, torch-npu and CANN stay identical to the working production
container.

Target image:

```bash
__IMAGE_TAG__
```

Known working production versions:

- Python 3.11.6
- funasr 1.3.1
- modelscope 1.36.0
- torch 2.1.0
- torch-npu 2.1.0.post10
- fastapi 0.136.0
- pydantic 2.9.2
- transformers 4.44.0

Bundled model:

- `/app/models/Qwen2.5-1.5B-Instruct`

Build on the production Ascend worker that has `meeting-asr:npu-new-oldbase`:

```bash
tar -xf __PACKAGE_NAME__.tar
cd __PACKAGE_NAME__
chmod 755 scripts/build_npu_prod_oldbase_on_prod.sh
IMAGE_TAG=__IMAGE_TAG__ bash scripts/build_npu_prod_oldbase_on_prod.sh
```

Read `npu-production-update-runbook.md` before deployment. Route production
traffic back to `dream-acr-new` before replacing the NPU Pod. When only one
NPU is available, do not use a normal rolling update that attempts to start
the old and new Pods together.

```bash
kubectl -n test patch svc dream-acr --type=merge \
  -p '{"spec":{"selector":{"app":"dream-acr-new"}}}'
kubectl -n test scale deployment/asr-leader-npu --replicas=0
kubectl -n test set image deployment/asr-leader-npu asr-leader-npu=__IMAGE_TAG__
kubectl -n test scale deployment/asr-leader-npu --replicas=1
kubectl -n test rollout status deployment/asr-leader-npu --timeout=10m
kubectl -n test logs deployment/asr-leader-npu --tail=200 --timestamps
```

Do not modify these running production services:

- `dream-acr`
- `dream-acr-new`

Keep `dream-acr-new` running as the immediate rollback target.

Important NPU environment values for the deployment:

```text
ASR_DEVICE=npu:0
ASR_RESOLVE_LOCAL_MODELS=false
POSTPROCESS_PRELOAD=true
ASR_NORMALIZE_ASCEND_DEVICES=false
MODELSCOPE_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

`ASR_RESOLVE_LOCAL_MODELS=false` is required for FunASR 1.3.1. The previous
failed image resolved `/app/models/paraformer-zh` as a local path and FunASR
1.3.1 raised `/app/models/paraformer-zh is not registered`.

`POSTPROCESS_PRELOAD=true` is safe in this package because Qwen is bundled. If
the model directory is removed, the entrypoint automatically switches preload
off before Python starts.
'@

$readme = $readmeTemplate.Replace("__IMAGE_TAG__", $ImageTag).Replace("__PACKAGE_NAME__", $PackageName)

Set-Content -LiteralPath (Join-Path $PackageRoot "README.md") -Value $readme -Encoding UTF8

$deploy = @"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: asr-leader-npu
  namespace: test
  labels:
    app: asr-leader-npu
spec:
  replicas: 1
  selector:
    matchLabels:
      app: asr-leader-npu
  template:
    metadata:
      labels:
        app: asr-leader-npu
    spec:
      nodeName: worker06-910b
      containers:
        - name: asr-leader-npu
          image: $ImageTag
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8000
              name: http
          env:
            - name: ASR_DEVICE
              value: npu:0
            - name: ASR_RESOLVE_LOCAL_MODELS
              value: "false"
            - name: MODELS_DIR
              value: /app/models
            - name: DATA_DIR
              value: /data
            - name: POSTPROCESS_ENABLED
              value: "true"
            - name: POSTPROCESS_PRELOAD
              value: "true"
            - name: POSTPROCESS_DEVICE
              value: npu:0
            - name: POSTPROCESS_MODEL
              value: Qwen/Qwen2.5-1.5B-Instruct
            - name: POSTPROCESS_MODEL_DIR
              value: /app/models/Qwen2.5-1.5B-Instruct
            - name: POSTPROCESS_WORKERS
              value: "1"
            - name: POSTPROCESS_SYNC_MAX_CHARS
              value: "1200"
            - name: POSTPROCESS_SYNC_TIMEOUT_SECONDS
              value: "120"
            - name: CORRECTION_TASK_WORKERS
              value: "1"
            - name: TRANSFORMERS_OFFLINE
              value: "1"
            - name: MODELSCOPE_OFFLINE
              value: "1"
            - name: ASR_NORMALIZE_ASCEND_DEVICES
              value: "false"
          resources:
            limits:
              huawei.com/Ascend910: "1"
            requests:
              cpu: "2"
              memory: 8Gi
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 90
            periodSeconds: 10
            timeoutSeconds: 5
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 180
            periodSeconds: 30
            timeoutSeconds: 5
"@

Set-Content -LiteralPath (Join-Path $PackageRoot "k8s-deployment.yaml") -Value $deploy -Encoding UTF8

if (Test-Path $TarPath) {
    Remove-Item -LiteralPath $TarPath -Force
}
tar -cf $TarPath -C (Join-Path $Root "deploy") $PackageName

Write-Host "Package: $TarPath"
Write-Host "Image:   $ImageTag"
