import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import traci

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.traffic_env import TrafficEnvironment

def run_baseline():
    env = TrafficEnvironment()

    print("Running Baseline (Fixed-Time Traffic Lights)...")
    print("-" * 50)

    # tracking metrics
    all_waiting_times = []
    all_queue_lengths = []
    all_throughputs   = []

    states = env.reset()
    step = 0
    total_vehicles_exited = 0
    phase_timer = 0
    current_phase = 0
    green_duration = 30  # fixed green time of 30 seconds

    while True:
        # fixed time control - switch phase every 30 steps
        phase_timer += 1
        if phase_timer >= green_duration:
            current_phase = (current_phase + 1) % 4
            phase_timer = 0

        # apply same fixed phase to all intersections
        actions = [current_phase] * env.num_agents

        next_states, rewards, done = env.step(actions)

        # collect metrics every 10 steps
        if step % 10 == 0:
            # waiting time
            waiting = sum(
                traci.lane.getWaitingTime(lane)
                for tl in env.traffic_lights
                for lane in traci.trafficlight.getControlledLanes(tl)
            )
            all_waiting_times.append(waiting)

            # queue length
            queue = sum(
                traci.lane.getLastStepHaltingNumber(lane)
                for tl in env.traffic_lights
                for lane in traci.trafficlight.getControlledLanes(tl)
            )
            all_queue_lengths.append(queue)

            # throughput
            exited = traci.simulation.getArrivedNumber()
            total_vehicles_exited += exited
            all_throughputs.append(total_vehicles_exited)

        states = next_states
        step += 1

        if done:
            break

    env.close()

    # print summary
    print(f"Average Waiting Time : {np.mean(all_waiting_times):.2f} seconds")
    print(f"Average Queue Length : {np.mean(all_queue_lengths):.2f} vehicles")
    print(f"Total Throughput     : {total_vehicles_exited} vehicles")
    print("-" * 50)

    # save metrics for comparison
    np.save("results/baseline_waiting.npy", np.array(all_waiting_times))
    np.save("results/baseline_queue.npy",   np.array(all_queue_lengths))
    np.save("results/baseline_throughput.npy", np.array(all_throughputs))

    # plot metrics
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    axes[0].plot(all_waiting_times, color='red')
    axes[0].set_title("Baseline Waiting Time Over Simulation")
    axes[0].set_xlabel("Time Step (x10)")
    axes[0].set_ylabel("Total Waiting Time (s)")

    axes[1].plot(all_queue_lengths, color='orange')
    axes[1].set_title("Baseline Queue Length Over Simulation")
    axes[1].set_xlabel("Time Step (x10)")
    axes[1].set_ylabel("Number of Vehicles Waiting")

    axes[2].plot(all_throughputs, color='green')
    axes[2].set_title("Baseline Cumulative Throughput")
    axes[2].set_xlabel("Time Step (x10)")
    axes[2].set_ylabel("Total Vehicles Exited")

    plt.tight_layout()
    plt.savefig("results/baseline_metrics.png")
    plt.show()
    print("Baseline plot saved to results/baseline_metrics.png")

if __name__ == "__main__":
    run_baseline()