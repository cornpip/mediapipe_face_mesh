# Builds release APKs (arm64) for every bench app and prints the size table.
# Run from Windows PowerShell:  .\bench\tool\build_sizes.ps1
# Deltas need a baseline (empty counter app with the same assets): the
# baseline app was removed 2026-08-13; `flutter create` a fresh one and add
# it back to $apps to re-measure deltas.

$bench = Split-Path -Parent $PSScriptRoot
$apps = @("mine", "mlkit", "fdt")
$results = @()

foreach ($app in $apps) {
    $dir = Join-Path $bench $app
    Write-Host "== building $app" -ForegroundColor Cyan
    Push-Location $dir
    flutter build apk --release --target-platform android-arm64
    if ($LASTEXITCODE -ne 0) {
        Write-Host "build failed for $app" -ForegroundColor Red
        Pop-Location
        continue
    }
    $apk = Join-Path $dir "build\app\outputs\flutter-apk\app-release.apk"
    $size = (Get-Item $apk).Length
    $results += [pscustomobject]@{ App = $app; Bytes = $size; MB = [math]::Round($size / 1MB, 2) }
    Pop-Location
}

$baseline = ($results | Where-Object App -eq "baseline").Bytes
Write-Host ""
Write-Host "| app | apk MB | delta vs baseline MB |"
Write-Host "| --- | --- | --- |"
foreach ($r in $results) {
    $delta = if ($baseline -and $r.App -ne "baseline") { [math]::Round(($r.Bytes - $baseline) / 1MB, 2) } else { "-" }
    Write-Host "| $($r.App) | $($r.MB) | $delta |"
}
