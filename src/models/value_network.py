import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """
    Red Neuronal DQN estándar.
    
    Args:
        state_dim: Dimension de espacio de estados
        action_dim: Dimension de espacio de acciones
        hidden_dim: Dimension de capas ocultas (default: 128)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DuelingQNetwork(nn.Module):
    """
    Red Neuronal Dueling DQN para la opción de requisito.
    
    Args:
        state_dim: Dimension de espacio de estados
        action_dim: Dimension de espacio de acciones
        hidden_dim: Dimension de capas ocultas (default: 128)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DuelingQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Stream de Valor
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Stream de Ventaja
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Combinar Valor y Ventaja: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_vals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_vals
