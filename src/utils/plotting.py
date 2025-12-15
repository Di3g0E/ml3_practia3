import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_results(env_name, agent_name, returns, losses_actor=None, losses_critic=None, run_id=None):
    """
    Resultados del entrenamiento: Retornos vs Episodios, y Pérdidas vs Episodios.
    
    Args:
        env_name: Nombre del entorno
        agent_name: Nombre del agente
        returns: Lista de retornos por episodio
        losses_actor: Lista de pérdidas del actor (opcional)
        losses_critic: Lista de pérdidas del crítico (opcional)
        run_id: Identificador único para esta ejecución (opcional, usa env_name_agent_name si no se proporciona)
    """
    # Crear directorio si no existe
    if not os.path.exists('results'):
        os.makedirs('results')

    # Usar run_id para el nombre del archivo si se proporciona, de lo contrario usar env_name_agent_name
    filename_base = run_id if run_id else f"{env_name}_{agent_name}"

    # Gráfico de Retornos
    plt.figure(figsize=(10, 5))
    plt.plot(returns)
    plt.title(f'Retornos del entrenamiento - {env_name} - {agent_name}')
    plt.xlabel('Episodio')
    plt.ylabel('Retorno')
    plt.grid(True)
    plt.savefig(f'results/returns_{filename_base}.png')
    plt.close()

    # Gráfico de Pérdidas
    if losses_actor is not None:
        plt.figure(figsize=(10, 5))
        plt.plot(losses_actor, label='Actor Loss')
        if losses_critic is not None:
            plt.plot(losses_critic, label='Critic Loss')
        plt.title(f'Pérdidas del entrenamiento - {env_name} - {agent_name}')
        plt.xlabel('Episodio')
        plt.ylabel('Pérdida')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'results/losses_{filename_base}.png')
        plt.close()

def plot_comparison(env_name, returns_reinforce, returns_actor_critic):
    """
    Resultados de comparación: Retornos vs Episodios de REINFORCE vs Actor-Critic.
    
    Args:
        env_name: Nombre del entorno
        returns_reinforce: Lista de retornos por episodio del REINFORCE
        returns_actor_critic: Lista de retornos por episodio del Actor-Critic
    """
    plt.figure(figsize=(10, 5))
    plt.plot(returns_reinforce, label='REINFORCE')
    plt.plot(returns_actor_critic, label='Actor-Critic')
    plt.title(f'Comparación - {env_name}')
    plt.xlabel('Episodio')
    plt.ylabel('Retorno')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/comparison_{env_name}.png')
    plt.close()
