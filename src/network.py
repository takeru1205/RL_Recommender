import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.fc2 = nn.Linear(128, action_dim)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.fc2(x)
        return F.softmax(action_scores, dim=1)


if __name__ == "__main__":
    import torch

    input_tensor = torch.rand(8, 16)
    net = Policy(state_dim=16, action_dim=4)
    output_tensor = net(input_tensor)
    assert output_tensor.shape == (8, 4)
