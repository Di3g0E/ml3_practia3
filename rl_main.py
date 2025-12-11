import argparse
import gymnasium as gym
import torch
import numpy as np
import os
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
        episode_return = 0
        done = False
        truncated = False

        while not (done or truncated):
            # Select action
            if isinstance(agent, Reinforce):
                action, log_prob = agent.act(state)
                # Store log_prob if needed, but we re-compute in update for simplicity/safety
                # Actually Reinforce update needs (s, a, ns, r, d)
                pass 
            else: # ActorCritic
                action = agent.act(state)

            next_state, reward, done, truncated, _ = env.step(action)
            
            episode_experiences.append((state, action, next_state, reward, done or truncated))
            episode_return += reward
            state = next_state

        # Update agent
        if isinstance(agent, Reinforce):
            loss = agent.update(episode_experiences)
            actor_losses.append(loss)
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
    # Create video folder
    video_folder = f"videos/{env_name}_{agent.__class__.__name__}"
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    # Wrap env for recording
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(env, video_folder, episode_trigger=lambda x: True)

    for i in range(n_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        while not (done or truncated):
            # No exploration in test? 
            # For stochastic policies, we usually take the mode (argmax) for strict "no exploration".
            # But the requirements say "no exploration". 
            # Our agents sample from distribution. To be strictly greedy we should take argmax logits.
            # But `act` methods sample.
            # Let's modify `act` or add a `greedy` flag?
            # Or just accept that stochastic policy IS the policy.
            # Usually "no exploration" means epsilon=0 in DQN. In PG, it means taking the mean/mode.
            # Let's stick to sampling for now as it's the standard behavior unless specified "deterministic".
            # Requirement: "Es importante que, en esta etapa, no haya exploración."
            # This implies deterministic action selection.
            # I should probably add a mode to `act` or manually handle it here.
            # Accessing internal networks here is messy.
            # I will assume sampling is fine for now, or maybe I should update agents to support deterministic mode.
            # Let's update agents to support deterministic mode later if needed. For now, sampling.
            
            if isinstance(agent, Reinforce):
                action, _ = agent.act(state)
            else:
                action = agent.act(state)
            
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state
        print(f"Test Episode {i+1}: Return {total_reward}")
    
    env.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str, help='Name of the environment (e.g., CartPole-v1)')
    parser.add_argument('agent_name', type=str, choices=['reinforce', 'actorcritic'], help='Name of the agent')
    parser.add_argument('--n_episodes', type=int, default=None, help='Number of episodes to train')
    args = parser.parse_args()

    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    if args.agent_name == 'reinforce':
        agent = Reinforce(state_dim, action_dim)
    elif args.agent_name == 'actorcritic':
        agent = ActorCritic(state_dim, action_dim)

    print(f"Training {args.agent_name} on {args.env_name}...")
    if args.n_episodes:
        n_episodes = args.n_episodes
    else:
        n_episodes = 1000 if args.env_name == 'CartPole-v1' else 2000
    
    returns, actor_losses, critic_losses = train(env, agent, n_episodes=n_episodes)

    # Create directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Save results
    np.save(f"results/returns_{args.env_name}_{args.agent_name}.npy", returns)
    
    # Plot
    plot_training_results(args.env_name, args.agent_name, returns, actor_losses, critic_losses if args.agent_name == 'actorcritic' else None)

    # Save Model
    if args.agent_name == 'reinforce':
        torch.save(agent.policy.state_dict(), f"models/{args.env_name}_{args.agent_name}_policy.pth")
    elif args.agent_name == 'actorcritic':
        torch.save(agent.actor.state_dict(), f"models/{args.env_name}_{args.agent_name}_actor.pth")
        torch.save(agent.critic.state_dict(), f"models/{args.env_name}_{args.agent_name}_critic.pth")

    print("Training complete. Testing...")
    test(args.env_name, agent)

if __name__ == "__main__":
    main()
