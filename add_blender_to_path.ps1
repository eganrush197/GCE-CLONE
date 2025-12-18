# PowerShell script to add Blender to PATH environment variable
# Run this script as Administrator

Write-Host "Adding Blender to PATH environment variable..." -ForegroundColor Cyan

# Blender installation paths
$blender45 = "C:\Program Files\Blender Foundation\Blender 4.5"
$blender31 = "C:\Program Files\Blender Foundation\Blender 3.1"

# Get current PATH
$currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")

# Check if paths are already in PATH
$needsUpdate = $false

if ($currentPath -notlike "*$blender45*") {
    Write-Host "Adding Blender 4.5 to PATH..." -ForegroundColor Green
    $currentPath += ";$blender45"
    $needsUpdate = $true
} else {
    Write-Host "Blender 4.5 already in PATH" -ForegroundColor Yellow
}

if ($currentPath -notlike "*$blender31*") {
    Write-Host "Adding Blender 3.1 to PATH..." -ForegroundColor Green
    $currentPath += ";$blender31"
    $needsUpdate = $true
} else {
    Write-Host "Blender 3.1 already in PATH" -ForegroundColor Yellow
}

# Update PATH if needed
if ($needsUpdate) {
    try {
        [Environment]::SetEnvironmentVariable("Path", $currentPath, "Machine")
        Write-Host "`n✅ SUCCESS! Blender has been added to PATH" -ForegroundColor Green
        Write-Host "`n⚠️  IMPORTANT: You must restart PowerShell for changes to take effect!" -ForegroundColor Yellow
        Write-Host "   Close this window and open a new PowerShell window." -ForegroundColor Yellow
    } catch {
        Write-Host "`n❌ ERROR: Failed to update PATH. Make sure you run this as Administrator!" -ForegroundColor Red
        Write-Host "   Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Red
    }
} else {
    Write-Host "`n✅ Both Blender versions are already in PATH" -ForegroundColor Green
}

Write-Host "`nCurrent PATH:" -ForegroundColor Cyan
Write-Host $currentPath

