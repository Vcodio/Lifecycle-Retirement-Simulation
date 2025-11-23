# Safe Git Setup Script
# This sets up Git with safety measures in place

Write-Host "`n🛡️  Setting up Git with Safety Measures" -ForegroundColor Cyan
Write-Host "=" * 50 -ForegroundColor Cyan

# Check if Git is installed
try {
    $gitVersion = git --version
    Write-Host "✓ Git found: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Git is not installed. Please install Git first." -ForegroundColor Red
    exit 1
}

# Check if repository is initialized
if (-not (Test-Path ".git")) {
    Write-Host "`nInitializing Git repository..." -ForegroundColor Yellow
    git init
    Write-Host "✓ Repository initialized" -ForegroundColor Green
} else {
    Write-Host "✓ Git repository already initialized" -ForegroundColor Green
}

# Set up Git user (if not already set)
$userName = git config user.name
$userEmail = git config user.email

if (-not $userName) {
    Write-Host "`nGit user name not set. Please enter your name:" -ForegroundColor Yellow
    $name = Read-Host
    git config user.name $name
    Write-Host "✓ User name set" -ForegroundColor Green
} else {
    Write-Host "✓ User name: $userName" -ForegroundColor Green
}

if (-not $userEmail) {
    Write-Host "`nGit user email not set. Please enter your email:" -ForegroundColor Yellow
    $email = Read-Host
    git config user.email $email
    Write-Host "✓ User email set" -ForegroundColor Green
} else {
    Write-Host "✓ User email: $userEmail" -ForegroundColor Green
}

# Create backups directory
if (-not (Test-Path "backups")) {
    New-Item -ItemType Directory -Path "backups" | Out-Null
    Write-Host "✓ Created backups directory" -ForegroundColor Green
}

# Display safety reminder
Write-Host "`n" + "=" * 50 -ForegroundColor Cyan
Write-Host "🛡️  SAFETY REMINDERS:" -ForegroundColor Yellow
Write-Host "=" * 50 -ForegroundColor Cyan
Write-Host "1. ALWAYS run .\backup-before-git.ps1 before dangerous operations" -ForegroundColor White
Write-Host "2. NEVER run 'git clean -fdx' - it deletes untracked files!" -ForegroundColor Red
Write-Host "3. Check 'git status' before committing" -ForegroundColor White
Write-Host "4. Your files are SAFE once committed to Git" -ForegroundColor Green
Write-Host "`n✓ Git setup complete!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "  1. Review SAFETY_RULES.md" -ForegroundColor White
Write-Host "  2. Run: git status (to see current state)" -ForegroundColor White
Write-Host "  3. Run: git add . (to stage files)" -ForegroundColor White
Write-Host "  4. Run: git commit -m 'Initial commit' (to save your work)" -ForegroundColor White

