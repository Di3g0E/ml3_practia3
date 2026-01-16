import numpy as np
import os
import glob

results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

print(f"{'Run ID':<60} | {'Mean Return (Last 50)':<20} | {'Max Return':<10}")
print("-" * 100)

npy_files = glob.glob(os.path.join(results_dir, "returns_*.npy"))

for f in sorted(npy_files):
    filename = os.path.basename(f)
    if "returns_" not in filename:
        continue
    
    run_id = filename.replace("returns_", "").replace(".npy", "")
    
    try:
        returns = np.load(f)
        if len(returns) == 0:
            print(f"{run_id:<60} | {'N/A':<20} | {'N/A':<10}")
            continue
            
        mean_last_50 = np.mean(returns[-50:])
        max_return = np.max(returns)
        
        print(f"{run_id:<60} | {mean_last_50:<20.2f} | {max_return:<10.2f}")
    except Exception as e:
        print(f"{run_id:<60} | Error: {e}")
