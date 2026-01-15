import argparse
import gymnasium as gym
import torch
import numpy as np
import os
import random
from gymnasium.wrappers import RecordVideo
from src.agents.reinforce import Reinforce
from src.agents.actor_critic import ActorCritic
from src.utils.plotting import plot_training_results

def train(env, agent, n_episodes=1000, print_every=50):
    returns = []
    actor_losses = []
    critic_losses = []

    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        episode_experiences = []
        saved_log_probs = []
        episode_return = 0
        done = False
        truncated = False

        while not (done or truncated):
            # Elegir acción
            if isinstance(agent, Reinforce):
                action, log_prob = agent.act(state)
                saved_log_probs.append(log_prob)
            else: # ActorCritic
                action = agent.act(state)

            next_state, reward, done, truncated, _ = env.step(action)
            
            episode_experiences.append((state, action, next_state, reward, done or truncated))
            episode_return += reward
            state = next_state

        # Actualizar agente
        if isinstance(agent, Reinforce):
            rewards = [e[3] for e in episode_experiences]
            loss_dict = agent.update(saved_log_probs, rewards)
            actor_losses.append(loss_dict['actor_loss'])
            critic_losses.append(loss_dict['critic_loss'])
        else:
            actor_loss, critic_loss = agent.update(episode_experiences)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

        returns.append(episode_return)

        if i_episode % print_every == 0:
            print(f"Episode {i_episode}/{n_episodes}, Return: {np.mean(returns[-print_every:]):.2f}")
            if isinstance(agent, ActorCritic):
                print(f"  Actor Loss: {np.mean(actor_losses[-print_every:]):.4f}, Critic Loss: {np.mean(critic_losses[-print_every:]):.4f}")

    return returns, actor_losses, critic_losses

def test(env_name, agent, n_episodes=5):
    # Crear carpeta de videos
    video_folder = f"videos/{env_name}_{agent.__class__.__name__}"
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    # Env para grabar
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(env, video_folder, episode_trigger=lambda x: True)

    # Test
    test_returns = []
    for i in range(n_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        while not (done or truncated):            
            if isinstance(agent, Reinforce):
                action, _ = agent.act(state, deterministic=True)
            else:
                action = agent.act(state)
            
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state
        print(f"Test Episode {i+1}: Return {total_reward}")
        test_returns.append(total_reward)
    
    env.close()
    return np.mean(test_returns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str, help='Nombre del entorno (e.g., CartPole-v1)')
    parser.add_argument('agent_name', type=str, choices=['reinforce', 'actorcritic'], help='Nombre del agente')
    parser.add_argument('--n_episodes', type=int, default=None, help='Número de episodios para entrenar')
    parser.add_argument('--lr', type=float, default=1e-3, help='Tasa de aprendizaje (para Reinforce o Actor)')
    parser.add_argument('--lr_critic', type=float, default=1e-3, help='Tasa de aprendizaje para Critic (Actor-Critic solo)')
    parser.add_argument('--gamma', type=float, default=0.99, help='Factor de descuento')
    parser.add_argument('--target_update_freq', type=int, default=10, help='Frecuencia de actualización de la red objetivo (Actor-Critic solo)')
    parser.add_argument('--dueling', action='store_true', help='Usar Dueling DQN (Actor-Critic solo)')
    parser.add_argument('--no_double', action='store_true', help='Deshabilitar Double DQN (Actor-Critic solo)')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='Coeficiente de entropía para Actor-Critic')
    parser.add_argument('--random_cutoff', action='store_true', help='Detener el entrenamiento aleatoriamente entre 15-20 episodios')
    args = parser.parse_args()

    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Número de episodios
    if args.random_cutoff:
        n_episodes = random.randint(15, 20)
        print(f"Corte aleatorio activado: Entrenando por {n_episodes} episodios.")
    elif args.n_episodes:
        n_episodes = args.n_episodes
    else:
        n_episodes = 1000 if args.env_name == 'CartPole-v1' else 2000

    # Identificador único para los resultados
    run_id = f"{args.env_name}_{args.agent_name}_lr{args.lr}_g{args.gamma}"
    if args.agent_name == 'actorcritic':
        run_id += f"_lrc{args.lr_critic}_{'dueling' if args.dueling else 'standard'}_{'double' if not args.no_double else 'noDouble'}_ent{args.entropy_coef}"
    
    if args.random_cutoff:
        run_id += f"_cutoff{n_episodes}"


    if args.agent_name == 'reinforce':
        agent = Reinforce(state_dim, action_dim, lr=args.lr, gamma=args.gamma)
    elif args.agent_name == 'actorcritic':
        agent = ActorCritic(state_dim, action_dim, 
                            lr_actor=args.lr, 
                            lr_critic=args.lr_critic, 
                            gamma=args.gamma,
                            target_update_freq=args.target_update_freq,
                            use_dueling=args.dueling,
                            use_double_dqn=not args.no_double,
                            entropy_coef=args.entropy_coef)

    print(f"Entrenando {args.agent_name} en {args.env_name}...")
    print(f"Config: LR={args.lr}, Gamma={args.gamma}" + (f", LR_Critic={args.lr_critic}, Dueling={args.dueling}, Double={not args.no_double}" if args.agent_name == 'actorcritic' else ""))

    returns, actor_losses, critic_losses = train(env, agent, n_episodes=n_episodes)

    # Crear directorios
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Guardar resultados
    np.save(f"results/returns_{run_id}.npy", returns)
    
    # Guardar gráfica
    plot_training_results(args.env_name, args.agent_name, returns, actor_losses, 
                         critic_losses if args.agent_name == 'actorcritic' else None, 
                         run_id=run_id)

    # Guardar modelo
    if args.agent_name == 'reinforce':
        torch.save(agent.policy.state_dict(), f"models/{run_id}_policy.pth")
    elif args.agent_name == 'actorcritic':
        torch.save(agent.actor.state_dict(), f"models/{run_id}_actor.pth")
        torch.save(agent.critic.state_dict(), f"models/{run_id}_critic.pth")

    print(f"Entrenamiento completado. Test {run_id}...")
    test(args.env_name, agent)

if __name__ == "__main__":
    main()
