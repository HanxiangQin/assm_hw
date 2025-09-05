import torch
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def f(x, y):
    return x**2 + y**2

def f_neg(x, y):
    return -x**2 - y**2

def run_sgd(objective_func, start_x, start_y, lr, momentum, weight_decay, maximize, num_steps=50):
    params = torch.tensor([start_x, start_y], requires_grad=True, dtype=torch.float32)
    optimizer = torch.optim.SGD([params], lr=lr, momentum=momentum, weight_decay=weight_decay, maximize=maximize)
    
    trajectory = np.zeros((num_steps + 1, 2))
    trajectory[0] = params.detach().numpy()
    
    for i in range(num_steps):
        optimizer.zero_grad()
        loss = objective_func(params[0], params[1])
        loss.backward()
        optimizer.step()
        trajectory[i+1] = params.detach().numpy()
        
    return trajectory

def plot_contour(objective_func, ax, title, x_range=(-4, 4), y_range=(-4, 4)):
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = objective_func(X, Y)
    
    ax.contour(X, Y, Z, levels=20, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.axhline(0, color='grey', lw=0.5)
    ax.axvline(0, color='grey', lw=0.5)
    ax.set_aspect('equal', adjustable='box')

if __name__ == '__main__':
    start_point = (3.0, 3.0)
    learning_rate = 0.1

    # --- Part (a): Vary momentum ---
    momenta = [0, 0.5, 0.9]
    fig1, axes1 = plt.subplots(1, 3, figsize=(20, 6))
    fig1.suptitle('SGD with Varying Momentum', fontsize=16)
    for i, m in enumerate(momenta):
        ax1 = axes1[i]
        plot_contour(f, ax1, f'momentum={m}')
        traj = run_sgd(f, start_point[0], start_point[1], learning_rate, m, 0, False)
        
        n_steps = len(traj) - 1
        colors = plt.cm.Blues(np.linspace(0.3, 1, n_steps + 1))
        ax1.plot(traj[:, 0], traj[:, 1], '-', color='grey', alpha=0.5)
        ax1.scatter(traj[:, 0], traj[:, 1], c=colors, s=20, zorder=3)
        # ax1.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig1.savefig(f'output/sgd_momentum_effect.png')
    plt.close(fig1)
    
    # --- Part (b): Add weight decay ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6))
    fig2.suptitle('SGD with Momentum and Weight Decay (0.1)', fontsize=16)
    weight_decay_val = 0.1
    for i, m in enumerate(momenta):
        ax2 = axes2[i]
        plot_contour(f, ax2, f'momentum={m}, weight decay={weight_decay_val}')
        traj = run_sgd(f, start_point[0], start_point[1], learning_rate, m, weight_decay_val, False)

        n_steps = len(traj) - 1
        colors = plt.cm.Blues(np.linspace(0.3, 1, n_steps + 1))
        ax2.plot(traj[:, 0], traj[:, 1], '-', color='grey', alpha=0.5)
        ax2.scatter(traj[:, 0], traj[:, 1], c=colors, s=20, zorder=3)
        # ax2.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig2.savefig(f'output/sgd_weight_decay_effect.png')
    plt.close(fig2)

    # --- Part (c): Maximization ---
    fig3, axes3 = plt.subplots(1, 3, figsize=(20, 6))
    fig3.suptitle('SGD with maximize=True on f(x,y) = -x^2 - y^2', fontsize=16)
    for i, m in enumerate(momenta):
        ax3 = axes3[i]
        plot_contour(f_neg, ax3, f'maximize=True, momentum={m}')
        traj = run_sgd(f_neg, start_point[0], start_point[1], learning_rate, m, 0, True)

        n_steps = len(traj) - 1
        colors = plt.cm.Blues(np.linspace(0.3, 1, n_steps + 1))
        ax3.plot(traj[:, 0], traj[:, 1], '-', color='grey', alpha=0.5)
        ax3.scatter(traj[:, 0], traj[:, 1], c=colors, s=20, zorder=3)
        # ax3.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig3.savefig(f'output/sgd_maximize_effect.png')
    plt.close(fig3)

    print("Generated plots: sgd_momentum_effect.png, sgd_weight_decay_effect.png, sgd_maximize_effect.png")
