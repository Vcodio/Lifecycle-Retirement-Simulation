# 🛡️ Git Safety System

This repository now has multiple safety measures to prevent accidental deletion of your work.

## 🚨 What Happened

Previously, a dangerous Git command (`git clean -fdx`) was accidentally run, which deleted untracked files. This safety system prevents that from happening again.

## 🛡️ Safety Features

### 1. **Safety Rules Document** (`SAFETY_RULES.md`)
   - Lists all dangerous commands
   - Provides recovery procedures
   - Documents safety protocols

### 2. **Backup Script** (`backup-before-git.ps1`)
   - Creates timestamped backups before dangerous operations
   - Run this BEFORE any Git operation you're unsure about
   - Backups are saved in `.\backups\` folder

### 3. **Safe Git Setup** (`setup-git-safely.ps1`)
   - Sets up Git with safety measures
   - Configures user name/email
   - Creates backup directory

### 4. **Safety Check Script** (`.git-safety-check.ps1`)
   - Blocks dangerous commands
   - Can be integrated into workflows

## 📋 Quick Start

### First Time Setup:
```powershell
# 1. Set up Git safely
.\setup-git-safely.ps1

# 2. Check what files Git sees
git status

# 3. Stage your files (makes them tracked)
git add .

# 4. Commit your work (SAVES it to Git)
git commit -m "Initial commit - my work is now safe!"
```

### Before Any Risky Operation:
```powershell
# Always backup first!
.\backup-before-git.ps1

# Then proceed with your Git command
```

## ⛔ NEVER RUN THESE COMMANDS

- `git clean -fdx` - **DELETES ALL UNTRACKED FILES**
- `git clean -fd` - **DELETES UNTRACKED FILES**
- `git reset --hard` - **DELETES UNCOMMITTED CHANGES**
- `git push --force` - **OVERWRITES REMOTE HISTORY**

## ✅ Safe Commands

- `git status` - See what's changed
- `git add .` - Stage files for commit
- `git commit -m "message"` - Save your work
- `git log` - View commit history
- `git diff` - See what changed

## 🔄 Recovery

If files are accidentally deleted:

1. **Check Cursor Local History**
   - Press `Ctrl+Shift+P`
   - Type "Local History"
   - Browse previous versions

2. **Check Backups**
   - Look in `.\backups\` folder
   - Find timestamped backup

3. **Check Git Reflog** (if files were committed)
   ```powershell
   git reflog
   git checkout <commit-hash>
   ```

4. **Check Windows Previous Versions**
   - Right-click folder → Properties → Previous Versions

## 📝 For AI Assistants

**CRITICAL RULES:**
- NEVER run `git clean` commands
- NEVER run `git reset --hard`
- ALWAYS suggest backup first
- ALWAYS explain what command will do
- When in doubt, ASK USER

See `SAFETY_RULES.md` for complete guidelines.

## 💡 Tips

1. **Commit frequently** - Your work is safe once committed
2. **Use descriptive commit messages** - Makes it easier to find changes
3. **Check `git status` often** - Know what Git sees
4. **When unsure, backup first** - Run `.\backup-before-git.ps1`

---

**Remember: Once your files are committed to Git, they're safe! Git keeps a complete history of all changes.**

