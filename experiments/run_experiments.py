import subprocess
import sys
import os

# Define experiments
experiments = [
    # REINFORCE LRs (Long)
    {"env": "CartPole-v1", "agent": "reinforce", "lr": 1e-3, "episodes": 200},
    
    # Actor-Critic (Default Entropy 0.01) (Long)
    {"env": "CartPole-v1", "agent": "actorcritic", "lr": 1e-3, "episodes": 200},
    {"env": "CartPole-v1", "agent": "actorcritic", "lr": 1e-3, "gamma": 0.95, "episodes": 200},
    
    # Random Cutoff Experiments (Short)
    {"env": "CartPole-v1", "agent": "reinforce", "lr": 1e-3, "random_cutoff": True},
    {"env": "CartPole-v1", "agent": "actorcritic", "lr": 1e-3, "random_cutoff": True},
]

# Ensure we are in project root (parent of experiments/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

print(f"Running experiments from {os.getcwd()}")

# Find the venv python
venv_python = os.path.join(project_root, ".venv", "Scripts", "python.exe")
if os.path.exists(venv_python):
    python_exe = venv_python
    print(f"Using venv python: {python_exe}")
else:
    python_exe = sys.executable
    print(f"Using system python: {python_exe}")

for exp in experiments:
    cmd = [python_exe, "rl_main.py", exp["env"], exp["agent"]]
    
    if "lr" in exp:
        cmd.extend(["--lr", str(exp["lr"])])
    if "gamma" in exp:
        cmd.extend(["--gamma", str(exp["gamma"])])
    if "episodes" in exp:
        cmd.extend(["--n_episodes", str(exp["episodes"])])
    if "random_cutoff" in exp and exp["random_cutoff"]:
        cmd.append("--random_cutoff")
        
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

print("All experiments completed.")
