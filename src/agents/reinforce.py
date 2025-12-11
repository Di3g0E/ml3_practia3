import torch
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from src.models.policy_network import PolicyNetwork

class Reinforce:
    """
    REINFORCE Agent with discrete actions.
    """
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)

    def act(self, state):
        """
        Selects an action given the state.
        Returns: action, log_prob
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits = self.policy(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def update(self, episode_experiences):
        """
        Updates the policy network using the episode experiences.
        episode_experiences: list of (state, action, next_state, reward, done)
        """
        # Unpack experiences
        rewards = [r for (_, _, _, r, _) in episode_experiences]
        
        # Calculate discounted returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns).to(self.device)
        # Normalize returns for stability (optional but recommended)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # Re-calculate log_probs to retain graph (or store them in act, but re-calculating is safer for some implementations)
        # However, standard REINFORCE usually stores log_probs during rollout. 
        # The requirements say: "El método devuelve como retorno la acción elegida y log pi(a|s)."
        # So we should use the stored log_probs if we passed them.
        # But wait, the update signature in requirements is just `update(episode_exp)`.
        # If we only pass (s, a, ns, r, d), we need to re-compute log_probs.
        # Let's re-compute them to be safe and consistent with "no memory of experiences" (though REINFORCE is Monte Carlo so it has episode memory).
        
        states = [s for (s, _, _, _, _) in episode_experiences]
        actions = [a for (_, a, _, _, _) in episode_experiences]
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        
        logits = self.policy(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)

        # Calculate loss
        loss = -torch.sum(log_probs * returns)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
