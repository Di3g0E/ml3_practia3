import torch
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from src.models.policy_network import PolicyNetwork

class Reinforce:
    """
    Agente REINFORCE con acciones discretas.
    """
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)

    def act(self, state):
        """
        Selecciona una acción dada el estado.
        Retorna: acción, log_prob
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits = self.policy(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def update(self, episode_experiences):
        """
        Actualiza la red de política usando las experiencias del episodio.
        episode_experiences: lista de (estado, acción, siguiente_estado, recompensa, hecho)
        """
        # Desempaquetar experiencias
        rewards = [r for (_, _, _, r, _) in episode_experiences]
        
        # Calcular retornos descuentos
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns).to(self.device)
        # Normalizar retornos para estabilidad
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        states = [s for (s, _, _, _, _) in episode_experiences]
        actions = [a for (_, a, _, _, _) in episode_experiences]
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        
        logits = self.policy(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)

        # Calcular pérdida
        loss = -torch.sum(log_probs * returns)

        # Optimizar
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
