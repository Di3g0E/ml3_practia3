import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    """
    Red neuronal simple que devuelve logits para acciones discretas.
    
    Args:
        state_dim: Dimension de espacio de estados
        action_dim: Dimension de espacio de acciones
        hidden_dim: Dimension de capas ocultas (default: 128)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Devolver logits, sin activación
