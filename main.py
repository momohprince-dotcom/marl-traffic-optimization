import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.traffic_env import TrafficEnvironment
from agents.dqn_agent import DQNAgent

# Training parameters
EPISODES = 50
TARGET_UPDATE = 10

def train():
    env = TrafficEnvironment()
    
    # create one agent per intersection
    agents = [
        DQNAgent(state_size=4, action_size=4, agent_id=i)
        for i in range(env.num_agents)
    ]
    
    # tracking metrics
    episode_rewards = []
    episode_waiting_times = []

    print("Starting MARL Training...")
    print(f"Training for {EPISODES} episodes")
    print("-" * 50)

    for episode in range(EPISODES):
        states = env.reset()
        total_reward = 0
        step = 0

        while True:
            # each agent picks an action
            actions = [
                agents[i].select_action(states[i])
                for i in range(env.num_agents)
            ]

            # apply actions to environment
            next_states, rewards, done = env.step(actions)

            # each agent learns from experience
            for i in range(env.num_agents):
                agents[i].remember(
                    states[i], actions[i], rewards[i], next_states[i], done
                )
                agents[i].train()

            states = next_states
            total_reward += sum(rewards)
            step += 1

            if done:
                break

        # update target networks every few episodes
        if episode % TARGET_UPDATE == 0:
            for agent in agents:
                agent.update_target_network()

        episode_rewards.append(total_reward)

        print(f"Episode {episode + 1}/{EPISODES} | "
              f"Steps: {step} | "
              f"Total Reward: {total_reward:.2f} | "
              f"Epsilon: {agents[0].epsilon:.3f}")

    # save trained agents
    print("-" * 50)
    print("Training complete! Saving agents...")
    for i, agent in enumerate(agents):
        agent.save(f"results/agent_{i}.pth")
    print("Agents saved to results/ folder!")

    # plot results
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig("results/training_rewards.png")
    plt.show()
    print("Training plot saved to results/training_rewards.png")

    env.close()

if __name__ == "__main__":
    train()