import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_results(env_name, agent_name, returns, losses_actor=None, losses_critic=None):
    """
    Plots training results: Returns vs Episodes, and Losses vs Episodes.
    """
    # Create directory if not exists
    if not os.path.exists('results'):
        os.makedirs('results')

    # Plot Returns
    plt.figure(figsize=(10, 5))
    plt.plot(returns)
    plt.title(f'Training Returns - {env_name} - {agent_name}')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.grid(True)
    plt.savefig(f'results/returns_{env_name}_{agent_name}.png')
    plt.close()

    # Plot Losses
    if losses_actor is not None:
        plt.figure(figsize=(10, 5))
        plt.plot(losses_actor, label='Actor Loss')
        if losses_critic is not None:
            plt.plot(losses_critic, label='Critic Loss')
        plt.title(f'Training Losses - {env_name} - {agent_name}')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'results/losses_{env_name}_{agent_name}.png')
        plt.close()

def plot_comparison(env_name, returns_reinforce, returns_actor_critic):
    """
    Plots comparison of REINFORCE vs Actor-Critic.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(returns_reinforce, label='REINFORCE')
    plt.plot(returns_actor_critic, label='Actor-Critic')
    plt.title(f'Comparison - {env_name}')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/comparison_{env_name}.png')
    plt.close()
