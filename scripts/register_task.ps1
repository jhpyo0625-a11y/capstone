# Registers the hourly CoilVision watcher with Windows Task Scheduler (spec §6.6).
# Run once from an elevated-or-normal PowerShell:  .\scripts\register_task.ps1
# Remove with:  Unregister-ScheduledTask -TaskName CoilVisionWatcher -Confirm:$false

$projectRoot = Split-Path -Parent $PSScriptRoot
$action = New-ScheduledTaskAction -Execute "$projectRoot\scripts\watcher.bat" -WorkingDirectory $projectRoot
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Hours 1) -RepetitionDuration ([TimeSpan]::MaxValue)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd `
    -ExecutionTimeLimit (New-TimeSpan -Hours 4)
Register-ScheduledTask -TaskName "CoilVisionWatcher" -Action $action -Trigger $trigger `
    -Settings $settings -Description "Hourly coil-classifier incoming/ check; retrains at threshold"
Write-Host "Registered task 'CoilVisionWatcher' (hourly). Logs: artifacts\watcher.log"
