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

def plot_test_bar_chart(test_results):
    """
    Gráfica de barras comparando el retorno medio en test para cada par (entorno, agente).
    
    Args:
        test_results: Lista de tuplas (env, agent, mean_return)
    """
    labels = [f"{env}\n({agent})" for env, agent, mean in test_results]
    means = [mean for env, agent, mean in test_results]
    
    plt.figure(figsize=(10, 6))
    colors = ['#3498db', '#e74c3c'] * (len(test_results) // 2 + 1)
    bars = plt.bar(labels, means, color=colors[:len(test_results)])
    
    plt.title('Comparativa de Rendimiento: Retorno Medio en Test (5 episodios)', fontsize=14, pad=20)
    plt.ylabel('Retorno Medio', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Añadir los valores encima/debajo de las barras
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.1f}', 
                 va='bottom' if yval >= 0 else 'top', ha='center', fontweight='bold')
        
    plt.axhline(0, color='black', linewidth=0.8) # Línea en el cero
    plt.tight_layout()
    plt.savefig('results/test_comparison_bar_chart.png')
    plt.close()

