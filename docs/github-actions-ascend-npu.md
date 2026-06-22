# GitHub Actions 构建 Ascend NPU 包

这个项目可以交给 GitHub Actions 在 Linux/ARM64 环境里构建 NPU 镜像，避免 Windows 本地跨平台打包带来的问题。

推荐流程：

1. 把源码推到 GitHub。
2. 手动运行 `Build Ascend NPU Package` workflow。
3. 部署时优先用 GHCR 镜像；如果服务器不能联网，再下载 GitHub Release 里的离线包分片。
4. 最终运行效果一定要在真实华为昇腾服务器上验证。

## 不要提交本地大文件

普通 Git 仓库不要提交这些东西：

- `*.tar`
- `deploy/ascend-npu/image/`
- `deploy/ascend-npu/data/`
- `*.safetensors`、`*.bin`、`*.pt`、`*.pth`、`*.onnx` 这类模型权重，除非你明确使用 Git LFS、Release、OSS 或其他外部下载方式管理

## 运行 Workflow

在 GitHub 页面进入：

```text
Actions -> Build Ascend NPU Package -> Run workflow
```

建议输入：

```text
version: 20260602
runner: ubuntu-22.04-arm
publish_ghcr: true
publish_release: true
include_models: false
```

`include_models=false` 表示离线包只包含镜像和部署脚本，不包含模型权重。这样最稳，因为普通 GitHub 仓库很难直接存几 GB 模型。

只有当 `deploy/ascend-npu/models` 能在 GitHub runner 里拿到，比如通过 Git LFS 或你自己加了模型下载步骤，才把 `include_models` 设成 `true`。

## 用 GHCR 镜像部署

在昇腾服务器上执行：

```bash
docker login ghcr.io
docker pull ghcr.io/OWNER/REPO:npu-ascend-20260602
docker tag ghcr.io/OWNER/REPO:npu-ascend-20260602 funasr-leader-asr:npu-ascend-20260602
```

把 `deploy/ascend-npu` 目录复制到服务器，确认 `docker-compose.yml` 里的镜像 tag 是：

```text
funasr-leader-asr:npu-ascend-20260602
```

准备模型目录：

```text
ascend-npu/models/paraformer-zh
ascend-npu/models/fsmn-vad
ascend-npu/models/ct-punc
ascend-npu/models/cam++
ascend-npu/models/Qwen2.5-1.5B-Instruct
```

启动：

```bash
cd ascend-npu
chmod +x *.sh
./start.sh
```

## 用离线包部署

从 GitHub Release 下载所有分片，Release 名字类似：

```text
ascend-npu-20260602
```

在 Linux 服务器上合并并校验：

```bash
cat asr-leader-ascend-npu-offline-20260602.tar.part-* > asr-leader-ascend-npu-offline-20260602.tar
sha256sum -c asr-leader-ascend-npu-offline-20260602.tar.sha256
tar -xf asr-leader-ascend-npu-offline-20260602.tar
```

如果 workflow 里 `include_models=false`，先把模型复制到：

```text
ascend-npu/models/
```

再启动：

```bash
cd ascend-npu
chmod +x *.sh
./start.sh
```

## 服务器检查

昇腾服务器上先确认：

```bash
docker version
docker compose version
ls /dev/davinci0 /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc
ls /usr/local/Ascend/driver
```

启动后检查：

```bash
docker compose logs -f --tail=100
curl http://127.0.0.1:8000/health
```

正常情况下，模型加载后返回里的 `device` 应该是 `npu:0`。
# Production warning

The generic GitHub Actions image is not the verified production path. The
2026-06-12 production environment requires the existing
`meeting-asr:npu-new-oldbase` image and its exact torch-npu/CANN runtime.
Follow `docs/npu-production-update-runbook.md` for production releases.
