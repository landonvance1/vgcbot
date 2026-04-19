Spawn the code-reviewer subagent to review the current branch's changes.

Pass any arguments as additional context to the reviewer (e.g. specific files or areas of concern): $ARGUMENTS

The subagent should start by running `git diff main...HEAD` to identify what changed, then conduct a full review per its checklist.
