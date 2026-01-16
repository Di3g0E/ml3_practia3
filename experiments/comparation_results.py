import os
import gymnasium as gym
import numpy as np
import argparse
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
os.chdir(project_root)

from rl_main import train, test
from src.agents.reinforce import Reinforce
from src.agents.actor_critic import ActorCritic
from src.utils.plotting import plot_training_results, plot_comparison, plot_test_bar_chart


def run_experiment(env_name, agent_type, n_episodes):
    print(f"\n--- Iniciando: {agent_type} en {env_name} ({n_episodes} episodios) ---")
    
    # Crear entorno
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Instanciar agente
    if agent_type == 'reinforce':
        agent = Reinforce(state_dim, action_dim, lr=1e-3, gamma=0.99)
    else:
        agent = ActorCritic(state_dim, action_dim, lr_actor=1e-3, lr_critic=1e-3, gamma=0.99)
    
    # Entrenar
    returns, actor_losses, critic_losses = train(env, agent, n_episodes=n_episodes)
    env.close()
    
    # Test (5 episodios)
    print(f"Evaluando {agent_type}...")
    mean_test_return = test(env_name, agent, n_episodes=5)
    
    return returns, mean_test_return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Ejecución rápida con pocos episodios para pruebas')
    args = parser.parse_args()
    
    envs = ['CartPole-v1', 'LunarLander-v3']
    agents = ['reinforce', 'actorcritic']
    
    # Configuración de episodios
    if args.quick:
        n_episodes_map = {'CartPole-v1': 20, 'LunarLander-v3': 20}
    else:
        n_episodes_map = {'CartPole-v1': 1000, 'LunarLander-v3': 2000}
        
    os.makedirs('results', exist_ok=True)
    
    test_results_summary = [] # Lista de (env, agent, mean_return)
    
    for env_name in envs:
        all_returns = {}
        
        for agent_type in agents:
            returns, mean_test = run_experiment(env_name, agent_type, n_episodes_map[env_name])
            
            all_returns[agent_type] = returns
            test_results_summary.append((env_name, agent_type, mean_test))
            
            # Guardar gráfica individual
            plot_training_results(env_name, agent_type, returns, run_id=f"practica_{env_name}_{agent_type}")
            
        # Crear gráfica comparativa de entrenamiento por entorno
        print(f"\nGenerando gráfica comparativa para {env_name}...")
        plot_comparison(env_name, all_returns['reinforce'], all_returns['actorcritic'])
        
    # Crear gráfica de barras de test
    print("\nGenerando gráfica de barras comparativa de test...")
    plot_test_bar_chart(test_results_summary)
    
    print("\n" + "="*50)
    print("PROCESO DE LA PRÁCTICA COMPLETADO")
    print("Las gráficas se han guardado en la carpeta 'results/':")
    print("1. comparison_CartPole-v1.png")
    print("2. comparison_LunarLander-v3.png")
    print("3. test_comparison_bar_chart.png")
    print("="*50)

if __name__ == "__main__":
    main()
