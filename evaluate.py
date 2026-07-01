import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.traffic_env import TrafficEnvironment
from agents.dqn_agent import DQNAgent

def evaluate():
    env = TrafficEnvironment()

    # load trained agents
    agents = [
        DQNAgent(state_size=4, action_size=4, agent_id=i)
        for i in range(env.num_agents)
    ]

    for i, agent in enumerate(agents):
        agent.load(f"results/agent_{i}.pth")
        agent.epsilon = 0.0  # no exploration during evaluation

    print("Evaluating trained MARL agents...")
    print("-" * 50)

    # tracking metrics
    all_waiting_times = []
    all_queue_lengths = []
    all_throughputs   = []

    states = env.reset()
    step = 0
    total_vehicles_exited = 0

    while True:
        actions = [
            agents[i].select_action(states[i])
            for i in range(env.num_agents)
        ]

        next_states, rewards, done = env.step(actions)

        # collect metrics every 10 steps
        if step % 10 == 0:
            import traci

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
    np.save("results/marl_waiting.npy",    np.array(all_waiting_times))
    np.save("results/marl_queue.npy",      np.array(all_queue_lengths))
    np.save("results/marl_throughput.npy", np.array(all_throughputs))

    # plot metrics
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    axes[0].plot(all_waiting_times, color='red')
    axes[0].set_title("Waiting Time Over Simulation")
    axes[0].set_xlabel("Time Step (x10)")
    axes[0].set_ylabel("Total Waiting Time (s)")

    axes[1].plot(all_queue_lengths, color='orange')
    axes[1].set_title("Queue Length Over Simulation")
    axes[1].set_xlabel("Time Step (x10)")
    axes[1].set_ylabel("Number of Vehicles Waiting")

    axes[2].plot(all_throughputs, color='green')
    axes[2].set_title("Cumulative Throughput Over Simulation")
    axes[2].set_xlabel("Time Step (x10)")
    axes[2].set_ylabel("Total Vehicles Exited")

    plt.tight_layout()
    plt.savefig("results/evaluation_metrics.png")
    plt.show()
    print("Evaluation plot saved to results/evaluation_metrics.png")

if __name__ == "__main__":
    evaluate()