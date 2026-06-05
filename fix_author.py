import subprocess, sys, os

env = os.environ.copy()
env["GIT_AUTHOR_NAME"]    = "Priya Monisha"
env["GIT_AUTHOR_EMAIL"]   = "datawithmcollab@gmail.com"
env["GIT_COMMITTER_NAME"] = "Priya Monisha"
env["GIT_COMMITTER_EMAIL"]= "datawithmcollab@gmail.com"

result = subprocess.run(
    ["git", "commit-tree"] + sys.argv[1:],
    env=env, capture_output=True
)
sys.stdout.buffer.write(result.stdout)
sys.stderr.buffer.write(result.stderr)
sys.exit(result.returncode)
