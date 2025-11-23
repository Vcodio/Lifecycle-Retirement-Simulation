# Automatic Backup Script
# Run this BEFORE any potentially dangerous Git operations

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backupDir = ".\backups\backup_$timestamp"
$sourceDir = "."

Write-Host "Creating backup at: $backupDir" -ForegroundColor Cyan

# Create backup directory
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null

# Copy all important files
$importantPaths = @(
    "src",
    "data",
    "output",
    "*.py",
    "*.csv",
    "*.md",
    "*.txt",
    ".gitignore"
)

foreach ($path in $importantPaths) {
    if (Test-Path $path) {
        Write-Host "  Backing up: $path" -ForegroundColor Gray
        Copy-Item -Path $path -Destination $backupDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "`n✓ Backup complete!" -ForegroundColor Green
Write-Host "Backup location: $backupDir" -ForegroundColor Cyan
Write-Host "`nYou can now safely run Git commands." -ForegroundColor Green

