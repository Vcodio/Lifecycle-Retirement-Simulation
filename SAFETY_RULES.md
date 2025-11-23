# 🛡️ SAFETY RULES - NEVER DELETE USER WORK

## CRITICAL RULES FOR AI ASSISTANT

### ⛔ NEVER RUN THESE COMMANDS:
1. `git clean -fdx` - DELETES ALL UNTRACKED FILES
2. `git clean -fd` - DELETES UNTRACKED FILES AND DIRECTORIES
3. `git reset --hard` - DELETES UNCOMMITTED CHANGES
4. `git push --force` - OVERWRITES REMOTE HISTORY
5. Any command that could delete user files

### ✅ SAFE GIT COMMANDS:
- `git status` - Check status
- `git add` - Stage files
- `git commit` - Commit changes
- `git log` - View history
- `git diff` - View differences
- `git branch` - List branches
- `git pull` - Pull changes (safe)

### 🔒 SAFETY PROTOCOLS:

1. **ALWAYS create backup before dangerous operations**
   ```powershell
   .\backup-before-git.ps1
   ```

2. **NEVER run destructive Git commands through AI**
   - If user needs `git clean`, they must run it manually
   - Always warn user about what command will do
   - Suggest backup first

3. **Check file existence before operations**
   - Verify files exist before modifying
   - Never assume files are safe to delete

4. **When in doubt, ASK USER**
   - Don't assume what user wants
   - Explain what command will do
   - Get explicit confirmation

## USER SAFETY CHECKLIST

Before running any Git command that modifies files:
- [ ] Do I have a backup?
- [ ] Do I understand what this command will do?
- [ ] Have I checked `git status` first?
- [ ] Am I sure I want to delete/lose files?

## RECOVERY PROCEDURES

If files are accidentally deleted:
1. Check Recycle Bin
2. Check Cursor Local History (Ctrl+Shift+P → "Local History")
3. Check Windows Previous Versions (right-click folder → Properties → Previous Versions)
4. Check Git reflog: `git reflog`
5. Check backups in `.\backups\` folder

## EMERGENCY CONTACTS

If work is lost:
- Cursor Local History: `%USERPROFILE%\AppData\Local\Cursor\User\History`
- Windows Shadow Copies: Right-click folder → Properties → Previous Versions
- Git Reflog: `git reflog` (if files were committed)

---

**REMEMBER: It's better to be safe than sorry. Always backup before destructive operations.**

