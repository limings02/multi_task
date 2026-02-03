# ============================================================================
# Interview Chain Runner - PowerShell 版本
# ============================================================================
# Windows 用户的备选方案（如果 Make 不可用）
#
# 用法：
#   .\scripts\run_interview_chain.ps1
#   .\scripts\run_interview_chain.ps1 -Resume
#   .\scripts\run_interview_chain.ps1 -DryRun
#   .\scripts\run_interview_chain.ps1 -Skip "E0a,E0b"
# ============================================================================

param(
    [switch]$DryRun,
    [switch]$Resume,
    [string]$Skip = "",
    [string]$Output = "runs/interview_chain"
)

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "  Interview Chain Runner (PowerShell)" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# 检查 Python 是否可用
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "Error: Python not found. Please install Python and add it to PATH." -ForegroundColor Red
    exit 1
}

# 构造 Python 脚本参数
$args = @("scripts/run_interview_chain.py")

if ($DryRun) {
    $args += "--dry-run"
}

if ($Resume) {
    $args += "--resume"
}

if ($Skip) {
    $args += "--skip"
    $args += $Skip
}

if ($Output) {
    $args += "--output"
    $args += $Output
}

# 打印命令
Write-Host "执行命令:" -ForegroundColor Yellow
Write-Host "  python $($args -join ' ')" -ForegroundColor Yellow
Write-Host ""

# 执行
& python @args

# 检查退出码
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "============================================================================" -ForegroundColor Green
    Write-Host "  ✓ 完成！" -ForegroundColor Green
    Write-Host "============================================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "查看结果:" -ForegroundColor Cyan
    Write-Host "  - 汇总表格: $Output\summary.csv" -ForegroundColor Cyan
    Write-Host "  - 增量分析: $Output\delta_analysis.txt" -ForegroundColor Cyan
    Write-Host "  - 完整信息: $Output\summary.json" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "============================================================================" -ForegroundColor Red
    Write-Host "  ✗ 失败（退出码 $LASTEXITCODE）" -ForegroundColor Red
    Write-Host "============================================================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "故障排查:" -ForegroundColor Yellow
    Write-Host "  1. 查看上面的错误信息" -ForegroundColor Yellow
    Write-Host "  2. 检查最后运行的实验日志（runs/**/train.log）" -ForegroundColor Yellow
    Write-Host "  3. 参考文档：docs/interview_chain.md" -ForegroundColor Yellow
}

exit $LASTEXITCODE
