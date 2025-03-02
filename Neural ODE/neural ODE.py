import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

torch.manual_seed(0)

def true_ode(t, u):
    A = np.array([[-0.1, 2.0], [-2.0, -0.1]])
    return (u ** 3) @ A.T

tspan = (0.0, 1.5)
tsteps = np.linspace(tspan[0], tspan[1], 30)
u0 = np.array([2.0, 0.0], dtype=np.float32)
sol = solve_ivp(true_ode, tspan, u0, t_eval=tsteps, method='RK45')
ode_data = sol.y

class NeuralODE(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 2)
        )
    
    def forward(self, t, u):
        return self.net(u ** 3)

def loss_function(pred, target):
    return torch.sum((pred - target) ** 2)

def train(model, optimizer, epochs=300):
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = ode_solver(model, u0_tensor, tsteps_tensor)
        loss = loss_function(pred, ode_data_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            plot_results(pred.detach().numpy())
    return model

def ode_solver(model, u0, tsteps):
    u = u0.clone().detach().unsqueeze(0)
    traj = [u]
    for i in range(len(tsteps) - 1):
        u = u + (tsteps[i+1] - tsteps[i]) * model(None, u)
        traj.append(u)
    return torch.cat(traj, dim=0)

def plot_results(pred):
    plt.scatter(tsteps, ode_data[0], label='Data', color='blue')
    plt.scatter(tsteps, pred[:, 0], label='Prediction', color='red')
    plt.legend()
    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
u0_tensor = torch.tensor(u0, dtype=torch.float32, device=device)
tsteps_tensor = torch.tensor(tsteps, dtype=torch.float32, device=device)
ode_data_tensor = torch.tensor(ode_data.T, dtype=torch.float32, device=device)
model = NeuralODE().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.05)
trained_model = train(model, optimizer)
pred_final = ode_solver(trained_model, u0_tensor, tsteps_tensor)
plot_results(pred_final.detach().cpu().numpy())
