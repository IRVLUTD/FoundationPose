docker rm -f foundationpose 2>$null

$DIR = Split-Path $PSScriptRoot -Parent
Write-Host "Host path:" $DIR

docker run --gpus all -it `
  --name foundationpose `
  -e DISPLAY=host.docker.internal:0.0 `
  -e QT_X11_NO_MITSHM=1 `
  -e NVIDIA_DISABLE_REQUIRE=1 `
  --cap-add=SYS_PTRACE `
  --security-opt seccomp=unconfined `
  -v "${DIR}:/workspace/FoundationPose" `
  --ipc=host `
  foundationpose:latest `
  bash -c "cd /workspace/FoundationPose && bash"