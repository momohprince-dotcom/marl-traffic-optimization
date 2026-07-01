import numpy as np
import matplotlib.pyplot as plt

# load baseline metrics
baseline_waiting    = np.load("results/baseline_waiting.npy")
baseline_queue      = np.load("results/baseline_queue.npy")
baseline_throughput = np.load("results/baseline_throughput.npy")

# load marl metrics
marl_waiting    = np.load("results/marl_waiting.npy")
marl_queue      = np.load("results/marl_queue.npy")
marl_throughput = np.load("results/marl_throughput.npy")

# plot comparison
fig, axes = plt.subplots(3, 1, figsize=(12, 14))

# waiting time
axes[0].plot(baseline_waiting, color='red',  label='Baseline (Fixed-Time)')
axes[0].plot(marl_waiting,     color='blue', label='MARL (AI Agents)')
axes[0].set_title("Waiting Time: MARL vs Baseline")
axes[0].set_xlabel("Time Step (x10)")
axes[0].set_ylabel("Total Waiting Time (s)")
axes[0].legend()

# queue length
axes[1].plot(baseline_queue, color='red',  label='Baseline (Fixed-Time)')
axes[1].plot(marl_queue,     color='blue', label='MARL (AI Agents)')
axes[1].set_title("Queue Length: MARL vs Baseline")
axes[1].set_xlabel("Time Step (x10)")
axes[1].set_ylabel("Number of Vehicles Waiting")
axes[1].legend()

# throughput
axes[2].plot(baseline_throughput, color='red',  label='Baseline (Fixed-Time)')
axes[2].plot(marl_throughput,     color='blue', label='MARL (AI Agents)')
axes[2].set_title("Throughput: MARL vs Baseline")
axes[2].set_xlabel("Time Step (x10)")
axes[2].set_ylabel("Total Vehicles Exited")
axes[2].legend()

plt.tight_layout()
plt.savefig("results/comparison.png")
plt.show()
print("Comparison plot saved to results/comparison.png")

# print summary
print("\n========== PERFORMANCE SUMMARY ==========")
print(f"{'Metric':<25} {'Baseline':>15} {'MARL':>15} {'Improvement':>15}")
print("-" * 70)
print(f"{'Avg Waiting Time (s)':<25} {np.mean(baseline_waiting):>15.2f} {np.mean(marl_waiting):>15.2f} {((np.mean(baseline_waiting) - np.mean(marl_waiting)) / np.mean(baseline_waiting) * 100):>14.1f}%")
print(f"{'Avg Queue Length':<25} {np.mean(baseline_queue):>15.2f} {np.mean(marl_queue):>15.2f} {((np.mean(baseline_queue) - np.mean(marl_queue)) / np.mean(baseline_queue) * 100):>14.1f}%")
print(f"{'Total Throughput':<25} {baseline_throughput[-1]:>15} {marl_throughput[-1]:>15} {((marl_throughput[-1] - baseline_throughput[-1]) / max(baseline_throughput[-1], 1) * 100):>14.1f}%")
print("==========================================")