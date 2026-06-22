# NPU production oldbase build

> The complete production procedure, failure analysis, traffic switch and
> rollback commands are maintained in `docs/npu-production-update-runbook.md`.
> Do not deploy from this short note alone.

Use this path for production. The package produced by this workflow includes
the local Qwen2.5-1.5B-Instruct model and overlays the new app code on top of
the production base image.

The working production NPU image exists only inside production:

```text
meeting-asr:npu-new-oldbase
```

Do not build the final image on Windows. Do not use a normal GitHub-hosted
runner unless the production base image has first been pushed to a private
registry. The safest build is a source overlay built on the Ascend worker that
already has the base image.

## Create the upload package on Windows

```powershell
cd D:\Docker\asr-funasr-leade-2\asrLeader-master
powershell -ExecutionPolicy Bypass -File scripts\package_npu_prod_oldbase.ps1 -Version 20260610
```

Upload:

```text
asr-leader-npu-prod-oldbase-20260610.tar
```

to the production worker that has `meeting-asr:npu-new-oldbase`.

## Build on production worker

```bash
cd /data/leader-asr
tar -xf asr-leader-npu-prod-oldbase-20260610.tar
cd asr-leader-npu-prod-oldbase-20260610
chmod 755 scripts/build_npu_prod_oldbase_on_prod.sh
IMAGE_TAG=funasr-leader-asr:npu-prod-oldbase-20260610 bash scripts/build_npu_prod_oldbase_on_prod.sh
```

## Deploy from master

Use the traffic-switch, scale-down, validation and rollback sequence in
`docs/npu-production-update-runbook.md`. Do not perform a normal rolling
update when the node only has one allocatable NPU for this Deployment.

Keep `dream-acr-new` running as the immediate rollback target.

## Why not GitHub by default

GitHub can only build this image if it can access the production base image.
That means either:

1. pushing `meeting-asr:npu-new-oldbase` to a private registry; or
2. using a self-hosted ARM64 runner that already has the base image loaded.

Both are less clean than building the source overlay inside production.
