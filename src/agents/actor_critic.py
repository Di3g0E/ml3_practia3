import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import copy
from src.models.policy_network import PolicyNetwork
from src.models.value_network import DuelingQNetwork, QNetwork

class ActorCritic:
    """
    Actor-Critic Agent with support for:
    - Critic Architectures: Standard DQN, Dueling DQN
    - Update Rules: Standard DQN, Double DQN (DualDQN)
    """
    def __init__(self, state_dim, action_dim, lr_actor=1e-3, lr_critic=1e-3, gamma=0.99, target_update_freq=10, use_dueling=False, use_double_dqn=True):
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.episode_count = 0
        self.use_double_dqn = use_double_dqn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actor
        self.actor = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Critic
        if use_dueling:
            self.critic = DuelingQNetwork(state_dim, action_dim).to(self.device)
        else:
            self.critic = QNetwork(state_dim, action_dim).to(self.device)
            
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def act(self, state):
        """
        Selects an action given the state.
        Returns: action
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits = self.actor(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item()

    def update(self, episode_experiences):
        """
        Updates Actor and Critic networks.
        episode_experiences: list of (state, action, next_state, reward, done)
        """
        self.episode_count += 1
        
        # Unpack experiences
        states, actions, next_states, rewards, dones = zip(*episode_experiences)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 1. Update Critic
        critic_loss = self._update_critic(states, actions, next_states, rewards, dones)

        # 2. Update Actor
        actor_loss = self._update_actor(states, actions, next_states, rewards, dones)

        # 3. Update Target Network (every N episodes)
        if self.episode_count % self.target_update_freq == 0:
            self.critic_target.load_state_dict(self.critic.state_dict())

        return actor_loss, critic_loss

    def _update_critic(self, states, actions, next_states, rewards, dones):
        # Compute Q(s, a)
        q_values = self.critic(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute Target Q(s', a')
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: action from local network, value from target network
                next_actions = self.critic(next_states).argmax(dim=1)
                next_q_values = self.critic_target(next_states)
                max_next_q_value = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN: max value from target network
                next_q_values = self.critic_target(next_states)
                max_next_q_value = next_q_values.max(1)[0]
                
            target_q_value = rewards + self.gamma * max_next_q_value * (1 - dones)

        # Loss (MSE)
        loss = F.mse_loss(q_value, target_q_value)

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return loss.item()

    def _update_actor(self, states, actions, next_states, rewards, dones):
        # Re-calculate log_probs
        logits = self.actor(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)

        # Calculate Advantage A(s, a) = r + gamma * max Q(s', a') - Q(s, a)
        # Note: We use the current critic for this, not the target, or maybe target? 
        # Requirements say: A(s, a) = r + gamma * max Q(s', a) - Q(s, a)
        # Usually for advantage we use the same values as TD error.
        # Let's use the Target Network for the bootstrap part to be consistent with Critic update.
        
        with torch.no_grad():
            next_q_values = self.critic_target(next_states) # Use target for stability
            max_next_q_value = next_q_values.max(1)[0]
            target_q_value = rewards + self.gamma * max_next_q_value * (1 - dones)
            
            q_values = self.critic(states)
            q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            advantage = target_q_value - q_value
            # IMPORTANT: Detach advantage
            advantage = advantage.detach()

        # Loss = -log_prob * Advantage
        loss = -torch.mean(log_probs * advantage)

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return loss.item()
