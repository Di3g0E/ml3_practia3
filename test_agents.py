import gymnasium as gym
import torch
from src.agents.reinforce import Reinforce
from src.agents.actor_critic import ActorCritic

def test_reinforce():
    print("Testing REINFORCE...")
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = Reinforce(state_dim, action_dim)
    
    state, _ = env.reset()
    action, log_prob = agent.act(state)
    assert isinstance(action, int)
    assert isinstance(log_prob, torch.Tensor)
    
    # Fake update
    experiences = [(state, action, state, 1.0, False)]
    loss = agent.update(experiences)
    print(f"REINFORCE Loss: {loss}")
    print("REINFORCE Test Passed!")

def test_actor_critic():
    print("Testing Actor-Critic...")
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = ActorCritic(state_dim, action_dim)
    
    state, _ = env.reset()
    action = agent.act(state)
    assert isinstance(action, int)
    
    # Fake update
    experiences = [(state, action, state, 1.0, False)]
    actor_loss, critic_loss = agent.update(experiences)
    print(f"Actor Loss: {actor_loss}, Critic Loss: {critic_loss}")
    print("Actor-Critic Test Passed!")

if __name__ == "__main__":
    test_reinforce()
    test_actor_critic()
