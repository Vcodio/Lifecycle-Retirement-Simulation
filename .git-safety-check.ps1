# Git Safety Check Script
# This script prevents dangerous Git commands from being executed

param(
    [string]$Command
)

# List of DANGEROUS commands that should NEVER be run
$DANGEROUS_COMMANDS = @(
    "git clean -fdx",
    "git clean -fd",
    "git clean --force",
    "git reset --hard",
    "git reset --hard HEAD",
    "git push --force",
    "git push -f"
)

# Check if command contains dangerous patterns
foreach ($dangerous in $DANGEROUS_COMMANDS) {
    if ($Command -like "*$dangerous*") {
        Write-Host "`n[ERROR] BLOCKED: This command is DANGEROUS and could delete your work!" -ForegroundColor Red
        Write-Host "Command: $Command" -ForegroundColor Yellow
        Write-Host "`nIf you REALLY need to run this, you must:" -ForegroundColor Yellow
        Write-Host "1. Create a backup first" -ForegroundColor Yellow
        Write-Host "2. Run it manually in terminal (not through AI assistant)" -ForegroundColor Yellow
        Write-Host "3. Understand exactly what it will do" -ForegroundColor Yellow
        exit 1
    }
}

# Additional check for git clean
if ($Command -match "git clean") {
    Write-Host "`n[WARNING] 'git clean' detected!" -ForegroundColor Yellow
    Write-Host "This command can DELETE untracked files!" -ForegroundColor Red
    Write-Host "Are you SURE you want to continue? (This is blocked for safety)" -ForegroundColor Yellow
    exit 1
}

exit 0

