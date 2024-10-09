import os
import subprocess

original_repo_url = "https://github.com/hubertsiuzdak/snac.git"
forked_repo_url = "https://github.com/MeDott29/snac.git"

# Remove the original remote
subprocess.run(["git", "remote", "remove", "origin"])

# Add the forked repo as the new remote named "origin"
subprocess.run(["git", "remote", "add", "origin", forked_repo_url])

# Push all branches and tags to the forked repo
subprocess.run(["git", "push", "--all", "origin"])
subprocess.run(["git", "push", "--tags", "origin"])

print(f"Remote switched from {original_repo_url} to {forked_repo_url}")
