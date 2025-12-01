import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C

def get_args():
    """
    Parses command line arguments to control hyperparameters.
    """
    parser = argparse.ArgumentParser(description="Bayesian Optimization for ResNet-18")
    
    # Model & Training
    parser.add_argument("--in_channels", type=int, default=64, help="Base channels for ResNet.")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs per BO iteration.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for dataloaders.")
    
    # Bayesian Optimization
    parser.add_argument('--budget', type=int, default=10, help='Total BO evaluations.')
    parser.add_argument('--init_steps', type=int, default=3, help='Initial random (Sobol) steps.')
    parser.add_argument('--bounds', type=float, nargs=2, default=[-4.0, -1.0], help='Log LR bounds.')
    parser.add_argument('--xi', type=float, default=0.01, help='Exploration-exploitation trade-off (WEI).')
    
    # Testing & Reproducibility
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--test', action='store_true', help='Run a test plot with fake data.')

    return parser.parse_args()

def get_device():
    """
    Returns the appropriate device (CUDA, MPS, or CPU).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

class BasicBlock(nn.Module):
    """
    Standard ResNet Basic Block.
    Consists of two 3x3 conv layers with a skip connection.

    input:  
        in_channels: Number of input channels.
        channels: Number of output channels for the block.
        stride: Stride for the first conv layer.
        size: (N, in_channels, H, W)
    output: 
        Output tensor after applying the block.
        size:  (N, channels * expansion, H_out, W_out)

    """
    expansion = 1

    def __init__(self, in_channels, channels, stride=1):
        super(BasicBlock, self).__init__()
        # Conv1 handles downsampling (stride)
        self.conv1 = nn.Conv2d(in_channels, channels, stride=stride, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        
        # Conv2 preserves dimensions
        self.conv2 = nn.Conv2d(channels, channels, stride=1, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * channels, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.expansion * channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    """
    ResNet-18 architecture adapted for 1-channel input (Fashion-MNIST).
    Structure: [2, 2, 2, 2] blocks.

    input:
        base_channels: Number of base channels for the first conv layer.
        num_classes: Number of output classes.
        size: (N, 1, 28, 28)

    output:
        Output tensor after applying ResNet-18.
        size:  (N, num_classes)
    """
    def __init__(self, base_channels=64, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channels = base_channels

        # Initial Conv: 1 input channel (Grayscale)
        self.conv1 = nn.Conv2d(1, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)

        # Residual Layers
        self.layer1 = self._make_layer(BasicBlock, base_channels, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, base_channels*2, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, base_channels*4, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, base_channels*8, 2, stride=2)

        self.linear = nn.Linear(base_channels * 8 * BasicBlock.expansion, num_classes)

    # Helper to create layers
    def _make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, channels, s))
            self.in_channels = channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def get_dataloaders(batch_size):
    """
    Prepares FashionMNIST dataloaders.
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=train_transform)
    val_ds = datasets.FashionMNIST(root="./data", train=False, download=True, transform=val_transform)

    # Use persistent workers if possible to speed up training
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader

def objective_function(log_lr, args, device, train_loader, val_loader):
    """
    Trains a new ResNet-18 instance with the specific learning rate.
    Returns: Validation Error (1 - Accuracy).
    """
    lr = 10**log_lr
    print(f"--- Eval: lr = {lr:.5f} (log_lr = {log_lr:.4f}) ---")

    model = ResNet18(base_channels=args.in_channels).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training
    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss = loss.item()
        print(f"    Epoch {epoch+1}/{args.epochs} Loss: {running_loss:.4f}")

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = correct / total
    error_rate = 1.0 - accuracy
    print(f"    Result: Accuracy = {accuracy:.4f} ({accuracy*100:.2f}%), Error = {error_rate:.4f}")

    return error_rate

def weighted_expected_improvement(x, Y_sample, gp, xi=0.01):
    """
    Computes Weighted Expected Improvement (WEI).
    
    Args:
        x: Points at which to evaluate the acquisition function.
        Y_sample: Observed values so far.
        gp: The trained Gaussian Process model.
        xi: Exploration-exploitation trade-off parameter.
            Higher xi -> More exploration.
    
    Returns:
        Acquisition values.
    """
    mu, sigma = gp.predict(x, return_std=True)
    mu_sample_opt = np.min(Y_sample) 

    with np.errstate(divide='warn'):
        imp = mu_sample_opt - mu - xi
        z = imp / sigma
        ei = imp * norm.cdf(z) + sigma * norm.pdf(z)

    # Handle cases where sigma is 0
    ei[sigma == 0.0] = 0.0
    return ei

def plot_optimization(x_train, y_train, gp, bounds, iteration):
    """
    Plots the GP posterior (mean & std), observations, and the acquisition function.

    Args:
        x_train: Observed input points.
        y_train: Observed output values.
        gp: Trained Gaussian Process model.
        bounds: Tuple of (min, max) for the input space.
        iteration: Current iteration number for title.
    """
    
    x_grid = np.linspace(bounds[0], bounds[1], 1000).reshape(-1, 1)

    mu, std = gp.predict(x_grid, return_std=True)
    acq = weighted_expected_improvement(x_grid, y_train, gp)

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # Top: GP Posterior
    ax[0].plot(x_grid, mu, color='#0044cc', linewidth=2, label='GP Mean')
    ax[0].fill_between(x_grid.ravel(), mu - 1.96 * std, mu + 1.96 * std,
                       alpha=0.2, color='#66b2ff', label='95% Confidence')
    ax[0].scatter(x_train, y_train, color='black', s=50, label='Observations')
    
    # Highlight Best
    best_idx = np.argmin(y_train)
    ax[0].scatter(x_train[best_idx], y_train[best_idx], c='gold', s=150, marker='*', 
                  edgecolors='black', label='Best Found')
    
    ax[0].set_ylabel('Validation Error')
    ax[0].set_title(f'Gaussian Process (Iteration {iteration})')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # Bottom: Acquisition Function
    ax[1].plot(x_grid, acq, color='red', linewidth=2, label='Acquisition (WEI)')
    ax[1].fill_between(x_grid.ravel(), acq, alpha=0.1, color='red')
    ax[1].set_ylabel('Acquisition Value')
    ax[1].set_xlabel('Log Learning Rate')
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if not os.path.exists("plots"):
        os.makedirs("plots")
        print("    [Created 'plots/' directory]")

    # 2. Save the figure
    filename = f"plots/iteration_{iteration:02d}.png"
    plt.savefig(filename)
    print(f"    [Plot saved to {filename}]")
    
    plt.show()

def run_test():
    """
    Simple test with fake data to verify plotting.
    """
    print("Running plotting test...")
    args = get_args()
    bounds = [-4.0, -1.0]
    x_train = np.array([[-3.5], [-3.0], [-2.0], [-1.5]])
    y_train = np.array([0.5, 0.4, 0.2, 0.3]) # Fake errors
    
    kernel = C(1.0) * Matern(length_scale=1.0, nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
    gp.fit(x_train, y_train)
    
    plot_optimization(x_train, y_train, gp, bounds, iteration=99)

def main():

    args = get_args()
    
    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    # Load Data once
    train_loader, val_loader = get_dataloaders(args.batch_size)

    # 1. Initial Design (Sobol)
    # Define bounds array for qmc scaling
    BOUNDS = np.array([[args.bounds[0], args.bounds[1]]])
    
    sampler = qmc.Sobol(d=1, scramble=True, seed=args.seed)
    sampler_unit = sampler.random(n=args.init_steps)
    x_train = qmc.scale(sampler_unit, BOUNDS[:, 0], BOUNDS[:, 1])
    
    y_train = []
    print(f"\n=== Initial Design (Sobol: {args.init_steps} steps) ===")
    for i, x in enumerate(x_train):
        y = objective_function(x[0], args, device, train_loader, val_loader)
        y_train.append(y)
    
    y_train = np.array(y_train)

    # 2. Gaussian Process Setup
    # Matern is standard a standard kernel for BO. normalize_y handles the mean shift.
    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, 
                                  alpha=0.01, normalize_y=True)

    # 3. Optimization Loop
    print("\n=== Bayesian Optimization Loop ===")
    for i in range(args.init_steps, args.budget):
        print(f"\nIteration {i + 1}/{args.budget}")

        # Update Model
        gp.fit(x_train, y_train)

        # Plot (Starting from 4th iteration)
        # i is 0-indexed count of total runs. So if init_steps=3, loop starts at i=3 (4th iteration)
        plot_optimization(x_train, y_train, gp, args.bounds, i + 1)

        # Optimize Acquisition Function
        x_grid = np.linspace(args.bounds[0], args.bounds[1], 5000).reshape(-1, 1)
        acq_values = weighted_expected_improvement(x_grid, y_train, gp, xi=args.xi)
        next_x = x_grid[np.argmax(acq_values)].reshape(1, -1)

        print(f"    Suggested log_lr: {next_x[0][0]:.4f}")

        # Evaluate
        next_y = objective_function(next_x[0][0], args, device, train_loader, val_loader)

        # Update Observation
        x_train = np.vstack([x_train, next_x])
        y_train = np.append(y_train, next_y)

    # 4. Final Result
    best_idx = np.argmin(y_train)
    print("\n=== Optimization Finished ===")
    print(f"Best Log LR: {x_train[best_idx][0]:.4f}")
    print(f"Best Real LR: {10**x_train[best_idx][0]:.6f}")
    print(f"Best Error Rate: {y_train[best_idx]:.4f}")

if __name__ == "__main__":
    # Check if testing mode is requested via args, otherwise run main
    opt = get_args()
    if opt.test:
        run_test()
    else:
        main()