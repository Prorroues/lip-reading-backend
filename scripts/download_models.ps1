# 默启未来 - 唇语识别后端模型权重下载脚本
# 用法：在仓库根目录执行  powershell -File scripts/download_models.ps1
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$releaseUrl = "https://github.com/Prorroues/lip-reading-backend/releases/download/v1.0.0/models-v1.0.zip"
$zipPath = Join-Path $repoRoot "models-v1.0.zip"

Write-Host "==> 下载模型权重包 (约 240MB)..."
Write-Host "    $releaseUrl"
Invoke-WebRequest -Uri $releaseUrl -OutFile $zipPath -UseBasicParsing

Write-Host "==> 解压到仓库根目录..."
Expand-Archive -LiteralPath $zipPath -DestinationPath $repoRoot -Force

# 确保运行目录存在
foreach ($d in @("uploads\videos", "uploads\clockwise_videos", "generates\videos", "generates\images")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $repoRoot $d) | Out-Null
}

Remove-Item $zipPath
Write-Host "==> 完成。模型已就位："
Write-Host "    Visual_Speech_Recognition_for_Multiple_Languages/benchmarks/CMLR/models/CMLR_V_WER8.0/"
Write-Host "    Visual_Speech_Recognition_for_Multiple_Languages/benchmarks/CMLR/language_models/lm_zh/"
Write-Host "    MiVOLO/models/"
