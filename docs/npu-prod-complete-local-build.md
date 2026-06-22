# NPU complete local build

> Not recommended for the current production environment. The registry image
> used by this workflow did not reproduce the runtime behavior of the working
> production-only `meeting-asr:npu-new-oldbase` image. Use
> `docs/npu-production-update-runbook.md` instead.

Use this after pulling or loading the production NPU base image into local
Docker.

The goal is a complete `linux/arm64` image tar that includes:

- the production NPU environment from
  `crpi-h5ujs5q57e0e72tz.cn-shanghai.personal.cr.aliyuncs.com/dream_acr/meeting-asr:npu`;
- the new application code;
- the local Qwen2.5-1.5B-Instruct model;
- NPU startup settings compatible with the working production versions.

## Prepare the production base image

Preferred local pull:

```powershell
docker pull crpi-h5ujs5q57e0e72tz.cn-shanghai.personal.cr.aliyuncs.com/dream_acr/meeting-asr:npu
docker image inspect crpi-h5ujs5q57e0e72tz.cn-shanghai.personal.cr.aliyuncs.com/dream_acr/meeting-asr:npu
```

If you need to export it from production instead, run on the production worker
that already has the image:

```bash
cd /data/leader-asr
nerdctl -n k8s.io save -o meeting-asr-npu-base.tar crpi-h5ujs5q57e0e72tz.cn-shanghai.personal.cr.aliyuncs.com/dream_acr/meeting-asr:npu
sha256sum meeting-asr-npu-base.tar > meeting-asr-npu-base.tar.sha256
```

Download `meeting-asr-npu-base.tar` to Windows.

## Load it locally

```powershell
docker load -i D:\path\to\meeting-asr-npu-base.tar
docker image inspect crpi-h5ujs5q57e0e72tz.cn-shanghai.personal.cr.aliyuncs.com/dream_acr/meeting-asr:npu
```

If the loaded tag is different, tag it:

```powershell
docker tag <loaded-image-id-or-tag> crpi-h5ujs5q57e0e72tz.cn-shanghai.personal.cr.aliyuncs.com/dream_acr/meeting-asr:npu
```

## Build the complete ARM image locally

```powershell
cd D:\Docker\asr-funasr-leade-2\asrLeader-master
powershell -ExecutionPolicy Bypass -File scripts\build_npu_prod_complete_local.ps1 -Version 20260610
```

Output:

```text
dist\funasr-leader-asr-npu-prod-complete-20260610.tar
dist\funasr-leader-asr-npu-prod-complete-20260610.tar.sha256
```

The image is `linux/arm64`. Windows/x86 can build and save it, but it cannot
validate Ascend NPU runtime locally.

## Upload and deploy

Upload the image tar to the production worker:

```bash
cd /data/leader-asr
nerdctl -n k8s.io load -i funasr-leader-asr-npu-prod-complete-20260610.tar
```

Then update only the new/test deployment from master:

```bash
kubectl -n test set image deployment/asr-leader-npu asr-leader-npu=funasr-leader-asr:npu-prod-complete-20260610
kubectl -n test rollout status deployment/asr-leader-npu --timeout=10m
kubectl -n test logs deployment/asr-leader-npu --tail=200 --timestamps
```

Do not touch `dream-acr` or `dream-acr-new`.
