import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from torchdiffeq import odeint

torch.manual_seed(0)
u0 = torch.tensor([2.0, 0.0], dtype=torch.float32)
datasize = 30
tspan = (0.0, 1.5)
tsteps = torch.linspace(tspan[0], tspan[1], datasize)

def true_ode_func(t, u):
    A = np.array([[-0.1, 2.0], [-2.0, -0.1]])
    return (u ** 3) @ A

sol = solve_ivp(true_ode_func, tspan, u0.numpy(), t_eval=tsteps.numpy())
ode_data = torch.tensor(sol.y, dtype=torch.float32)

class NeuralODEFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 2)
        )
    def forward(self, t, x):
        return self.net(x ** 3)

dudt2 = NeuralODEFunc()
optimizer = optim.Adam(dudt2.parameters(), lr=0.05)

def predict_neuralode():
    return odeint(dudt2, u0, tsteps).permute(1, 0, 2)

def loss_neuralode():
    pred = predict_neuralode()
    return torch.sum((node_data - pred) ** 2), pred

def callback():
    loss, pred = loss_neuralode()
    print(loss.item())
    plt.scatter(tsteps.numpy(), node_data[0].numpy(), label='Data')
    plt.scatter(tsteps.numpy(), pred[0].detach().numpy(), label='Prediction')
    plt.legend()
    plt.show()

for _ in range(300):
    optimizer.zero_grad()
    loss, _ = loss_neuralode()
    loss.backward()
    optimizer.step()
callback()
