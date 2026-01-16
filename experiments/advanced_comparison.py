import sys
import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.actor_critic import ActorCritic
from src.agents.reinforce import Reinforce
from rl_main import train

def run_comparison():
    env_name = 'CartPole-v1'
    n_episodes = 400
    
    print(f"--- Ejecución de comparación avanzada para {env_name} ({n_episodes} episodios) ---")
    
    results = {}
    
    # 1. AC Standard (Baseline for AC)
    print("Entrenamiento de AC Standard (No Dueling)...")
    env = gym.make(env_name)
    agent_std = ActorCritic(env.observation_space.shape[0], env.action_space.n, 
                            lr_actor=1e-3, lr_critic=1e-3, gamma=0.99,
                            use_dueling=False, use_double_dqn=True, entropy_coef=0.01)
    returns_std, _, _ = train(env, agent_std, n_episodes=n_episodes)
    results['AC Standard'] = returns_std
    env.close()

    # 2. AC Dueling
    print("Entrenamiento de AC Dueling...")
    env = gym.make(env_name)
    agent_duel = ActorCritic(env.observation_space.shape[0], env.action_space.n, 
                             lr_actor=1e-3, lr_critic=1e-3, gamma=0.99,
                             use_dueling=True, use_double_dqn=True, entropy_coef=0.01)
    returns_duel, _, _ = train(env, agent_duel, n_episodes=n_episodes)
    results['AC Dueling'] = returns_duel
    env.close()

    # 3. AC No Entropy (No Exploracion)
    print("Entrenamiento de AC No Entropy...")
    env = gym.make(env_name)
    agent_no_ent = ActorCritic(env.observation_space.shape[0], env.action_space.n, 
                               lr_actor=1e-3, lr_critic=1e-3, gamma=0.99,
                               use_dueling=False, use_double_dqn=True, entropy_coef=0.0)
    returns_no_ent, _, _ = train(env, agent_no_ent, n_episodes=n_episodes)
    results['AC No Entropy'] = returns_no_ent
    env.close()

    # Save results
    os.makedirs('../results', exist_ok=True)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    window = 20
    for label, returns in results.items():
        # Smoothing
        smoothed = np.convolve(returns, np.ones(window)/window, mode='valid')
        plt.plot(smoothed, label=label)
        
    plt.title(f'Comparación de arquitecturas de AC en {env_name}')
    plt.xlabel('Episodio')
    plt.ylabel('Return (Suavizado)')
    plt.legend()
    plt.grid(True)
    plt.savefig('../results/advanced_comparison.png')
    print("Plot guardado en results/advanced_comparison.png")

if __name__ == "__main__":
    run_comparison()
