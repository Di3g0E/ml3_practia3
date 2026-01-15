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
    Agente Actor-Critic con soporte para:
    - Arquitecturas Criticas: DQN estándar, Dueling DQN
    - Reglas de Actualización: DQN estándar, Double DQN (DualDQN)
    """
    def __init__(self, state_dim, action_dim, lr_actor=1e-3, lr_critic=1e-3, gamma=0.99, target_update_freq=10, use_dueling=False, use_double_dqn=True, entropy_coef=0.01):
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.episode_count = 0
        self.use_double_dqn = use_double_dqn
        self.entropy_coef = entropy_coef
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actor
        self.actor = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Critic
        if use_dueling:
            self.critic = DuelingQNetwork(state_dim, action_dim).to(self.device)
        else:
            self.critic = QNetwork(state_dim, action_dim).to(self.device)
            
        self.critic_target = copy.deepcopy(self.critic)  # Crear una copia de la red critic_target que se usara para calcular el target
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def act(self, state):
        """
        Selecciona una acción dada el estado.
        Retorna: acción
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits = self.actor(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item()

    def update(self, episode_experiences):
        """
        Actualiza las redes Actor y Critic.
        episode_experiences: lista de (estado, acción, siguiente_estado, recompensa, hecho)
        """
        self.episode_count += 1
        
        # Desempaquetar experiencias
        states, actions, next_states, rewards, dones = zip(*episode_experiences)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 1. Actualizar Critic
        critic_loss = self._update_critic(states, actions, next_states, rewards, dones)

        # 2. Actualizar Actor
        actor_loss = self._update_actor(states, actions, next_states, rewards, dones)

        # 3. Actualizar Red Target (cada N episodios)
        if self.episode_count % self.target_update_freq == 0:
            self.critic_target.load_state_dict(self.critic.state_dict())

        return actor_loss, critic_loss

    def _update_critic(self, states, actions, next_states, rewards, dones):
        # Calcular Q(s, a)
        q_values = self.critic(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Calcular Target Q(s', a')
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
        # Recalcular log_probs
        logits = self.actor(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        with torch.no_grad():
            next_q_values = self.critic_target(next_states) # Usar target para estabilidad
            max_next_q_value = next_q_values.max(1)[0]
            target_q_value = rewards + self.gamma * max_next_q_value * (1 - dones)
            
            q_values = self.critic(states)
            q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            advantage = target_q_value - q_value
            # IMPORTANTE: Desconectar advantage para evitar la propagación del gradiente
            advantage = advantage.detach()

        # Loss = -log_prob * Advantage - entropy_coef * entropy
        loss = -torch.mean(log_probs * advantage) - (self.entropy_coef * entropy)

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return loss.item()
