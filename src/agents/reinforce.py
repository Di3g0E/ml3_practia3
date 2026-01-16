import torch
import torch.optim as optim
from torch.distributions import Categorical
from src.models.policy_network import PolicyNetwork

class Reinforce:
    """
    Agente REINFORCE.
    
    Siguiendo la estructura del notebook 05a_REINFORCE.ipynb:
    - Se acumulan los log_probs durante el episodio.
    - Se actualiza al final del episodio usando esos log_probs guardados.
    - Adaptado para acciones discretas (Categorical).
    """
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Reutilizamos la PolicyNetwork existente, que ya devuelve logits
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)

    def act(self, state, deterministic=False):
        """
        Selecciona una acción dada el estado.
        Retorna: acción (int), log_prob (tensor con gradiente)
        """
        # Convertir estado a tensor
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        # Forward pass: obtener logits
        logits = self.policy(state)
        
        # Crear distribución categórica
        dist = Categorical(logits=logits)
        
        if deterministic:
            # En modo determinista, elegimos la acción con mayor probabilidad
            action = logits.argmax(dim=1)
        else:
            # Muestrear acción
            action = dist.sample()
        
        # Calcular log_prob de la acción seleccionada
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob

    def update(self, saved_log_probs, rewards, **kwargs):
        """
        Actualiza la política usando los log_probs guardados y las recompensas observadas.
        **kwargs: Argumentos adicionales ignorados por Reinforce (states, next_states, dones).
        
        Returns:
            dict: Diccionario con las pérdidas {'actor_loss': ..., 'critic_loss': 0}
        """
        R = 0
        returns = []
        
        # Calcular retornos descontados (G_t)
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns).float().to(self.device)
        
        # Normalizar retornos (estabilidad numérica)
        if len(returns) > 1: # Evitar NaN si solo hay 1 paso y std es 0
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        policy_loss = []
        
        # Calcular pérdida: sum(-log_prob * G_t)
        for log_prob, Gt in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * Gt)
            
        # Sumar todas las pérdidas del episodio
        policy_loss = torch.cat(policy_loss).sum()
        
        # Optimización
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        return {'actor_loss': policy_loss.item(), 'critic_loss': 0.0}
