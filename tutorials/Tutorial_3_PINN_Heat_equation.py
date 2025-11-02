"""
    Physics-Informed Neural Networks (PINNs) for Solving the 1D Heat Equation
    =============================================================================

    This script implements a Physics-Informed Neural Network (PINN) to solve the one-dimensional heat equation:
    
    u_t(t, x) = u_xx(t, x) , with Dirichlet boundary conditions:
    u_b(t, -1) = u_b(t, 1) = 0
    and the initial condition:
    u(x, 0) = -sin(pi * x)

    The PINN is used to approximate the solution of the heat equation over the domain:
    t ∈ [0, 0.6], x ∈ [-1, 1].

    The network approximates the solution using a feed-forward neural network (FNN) with the following key steps:
    1. **Boundary Condition Loss**: The network must satisfy the boundary conditions at the left (x = -1) and right (x = 1) boundaries.
    2. **Initial Condition Loss**: The network must satisfy the initial condition at t = 0.
    3. **PDE Residual**: The network must approximate the solution to the PDE, i.e., satisfy u_t = u_xx in the interior of the domain.
    
    The total loss is composed of:
    - Spatial boundary loss (L_sb)
    - Temporal boundary loss (L_tb)
    - PDE residual loss (L_int)

    The network is trained using these loss functions and the training data generated through Sobol sequences for sampling.
    The goal is to minimize the total loss using an optimizer like Adam.

    Date: Nov 3rd 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

class NeuralNet(nn.Module):
    def __init__(self, input_dimension, output_dimension, n_hidden_layers, neurons, regularization_param, regularization_exp, retrain_seed):
        """
        Initialize the Neural Network
        Args:
            input_dimension (int): Number of input features (e.g., t and x)
            output_dimension (int): Number of output features (e.g., the solution u(t, x))
            n_hidden_layers (int): Number of hidden layers in the network
            neurons (int): Number of neurons in each hidden layer
            regularization_param (float): Regularization parameter to prevent overfitting
            regularization_exp (float): Regularization exponent (typically 2 for L2 norm)
            retrain_seed (int): Random seed for weight initialization
        """
        super(NeuralNet, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.neurons = neurons
        self.n_hidden_layers = n_hidden_layers
        self.activation = nn.Tanh()
        self.regularization_param = regularization_param
        self.regularization_exp = regularization_exp
        self.retrain_seed = retrain_seed

        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers - 1)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)

        self.init_xavier()

    def forward(self, x):
        """
        Define the forward pass of the neural network.
        Args:
            x (Tensor): Input tensor containing t and x values
        Returns:
            Tensor: Predicted solution u(t, x)
        """
        x = self.activation(self.input_layer(x))
        for k, l in enumerate(self.hidden_layers):
            x = self.activation(l(x))
        return self.output_layer(x)

    def init_xavier(self):
        """
        Initialize the weights of the neural network using Xavier initialization.
        """
        torch.manual_seed(self.retrain_seed)

        def init_weights(m):
            if isinstance(m, nn.Linear) and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain('tanh')
                nn.init.xavier_uniform_(m.weight, gain=g)
                m.bias.data.fill_(0)

        self.apply(init_weights)

    def regularization(self):
        """
        Compute the regularization loss for the neural network to avoid overfitting.
        """
        reg_loss = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                reg_loss += torch.norm(param, self.regularization_exp)
        return self.regularization_param * reg_loss

def fit(model, training_set, num_epochs, optimizer, p, verbose=True):
    """
    Train the PINN model.
    Args:
        model (NeuralNet): The neural network model
        training_set (DataLoader): The training data
        num_epochs (int): Number of epochs to train the model
        optimizer (torch.optim.Optimizer): Optimizer for training the model
        p (float): The p-norm for the loss function
        verbose (bool): Whether to print progress during training
    Returns:
        list: The training history
    """
    history = []

    for epoch in range(num_epochs):
        if verbose: print("################################ ", epoch, " ################################")

        running_loss = 0

        # Loop over batches
        for j, (x_train_, u_train_) in enumerate(training_set):
            def closure():
                optimizer.zero_grad()
                u_pred_ = model(x_train_)
                loss = torch.mean((u_pred_.reshape(-1, ) - u_train_.reshape(-1, )) ** p) + model.regularization()
                loss.backward()
                running_loss += loss.item()
                return loss

            optimizer.step(closure=closure)

        if verbose: print(f"Loss: {running_loss / len(training_set)}")
        history.append(running_loss)

    return history

class Pinns:
    def __init__(self, n_int_, n_sb_, n_tb_):
        """
        Initialize the PINN model for solving the heat equation.
        Args:
            n_int_ (int): Number of interior points
            n_sb_ (int): Number of spatial boundary points
            n_tb_ (int): Number of temporal boundary points
        """
        self.n_int = n_int_
        self.n_sb = n_sb_
        self.n_tb = n_tb_

        self.domain_extrema = torch.tensor([[0, 0.6], [-1, 1]])  # Time and space domain
        self.space_dimensions = 1  # We are solving a 1D problem
        self.lambda_u = 10  # Regularization parameter

        # Define the neural network to approximate the solution
        self.approximate_solution = NeuralNet(
            input_dimension=self.domain_extrema.shape[0],
            output_dimension=1,
            n_hidden_layers=4,
            neurons=20,
            regularization_param=0.,
            regularization_exp=2.,
            retrain_seed=42
        )

        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        self.training_set_sb, self.training_set_tb, self.training_set_int = self.assemble_datasets()

    def convert(self, tens):
        """
        Linearly map the tensor from the unit square to the actual domain.
        """
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]

    def initial_condition(self, x):
        """
        The initial condition for the heat equation: u(x, 0) = -sin(pi * x)
        """
        return -torch.sin(np.pi * x)

    def exact_solution(self, inputs):
        """
        The exact solution for the heat equation: u(t, x) = -exp(-pi^2 * t) * sin(pi * x)
        """
        t = inputs[:, 0]
        x = inputs[:, 1]
        return -torch.exp(-np.pi**2 * t) * torch.sin(np.pi * x)

    def add_temporal_boundary_points(self):
        """
        Generate points on the temporal boundary where t = 0.
        """
        t0 = self.domain_extrema[0, 0]
        input_tb = self.convert(self.soboleng.draw(self.n_tb))
        input_tb[:, 0] = torch.full(input_tb[:, 0].shape, t0)
        output_tb = self.initial_condition(input_tb[:, 1]).reshape(-1, 1)
        return input_tb, output_tb

    def add_spatial_boundary_points(self):
        """
        Generate points on the spatial boundaries at x = -1 and x = 1.
        """
        x_left = torch.full((self.n_sb, 1), -1.0)
        x_right = torch.full((self.n_sb, 1), 1.0)
        u_left = torch.zeros((self.n_sb, 1))
        u_right = torch.zeros((self.n_sb, 1))

        t_values = self.soboleng.draw(self.n_sb)[:, 0]
        input_left = torch.cat([t_values.reshape(-1, 1), x_left], dim=1)
        input_right = torch.cat([t_values.reshape(-1, 1), x_right], dim=1)

        return torch.cat([input_left, input_right], dim=0), torch.cat([u_left, u_right], dim=0)

    def add_interior_points(self):
        """
        Generate interior points where the PDE is enforced.
        """
        input_int = self.convert(self.soboleng.draw(self.n_int))
        output_int = torch.zeros((input_int.shape[0], 1))
        return input_int, output_int

    def assemble_datasets(self):
        """
        Assemble the training datasets for spatial boundary, temporal boundary, and interior points.
        """
        input_sb, output_sb = self.add_spatial_boundary_points()
        input_tb, output_tb = self.add_temporal_boundary_points()
        input_int, output_int = self.add_interior_points()

        training_set_sb = DataLoader(torch.utils.data.TensorDataset(input_sb, output_sb), batch_size=2*self.space_dimensions*self.n_sb, shuffle=False)
        training_set_tb = DataLoader(torch.utils.data.TensorDataset(input_tb, output_tb), batch_size=self.n_tb, shuffle=False)
        training_set_int = DataLoader(torch.utils.data.TensorDataset(input_int, output_int), batch_size=self.n_int, shuffle=False)

        return training_set_sb, training_set_tb, training_set_int

    def apply_initial_condition(self, input_tb):
        """
        Apply the initial condition on the temporal boundary points.
        """
        return self.approximate_solution(input_tb)

    def apply_boundary_conditions(self, input_sb):
        """
        Apply the boundary conditions on the spatial boundary points.
        """
        return self.approximate_solution(input_sb)

    def compute_pde_residual(self, input_int):
        """
        Compute the residual for the PDE u_t = u_xx at the interior points.
        """
        input_int.requires_grad = True
        u = self.approximate_solution(input_int)
        grad_u = torch.autograd.grad(u, input_int, torch.ones_like(u), create_graph=True)[0]
        u_t = grad_u[:, 0]
        u_x = grad_u[:, 1]
        u_xx = torch.autograd.grad(u_x, input_int, torch.ones_like(u_x), create_graph=True)[0][:, 1]
        r = u_t - u_xx
        return r

    def compute_loss(self, inp_train_sb, u_train_sb, inp_train_tb, u_train_tb, inp_train_int, verbose=True):
        """
        Compute the total loss function as the weighted sum of boundary and PDE residuals.
        """
        u_pred_sb = self.apply_boundary_conditions(inp_train_sb)
        u_pred_tb = self.apply_initial_condition(inp_train_tb)

        r_sb = u_train_sb - u_pred_sb
        r_tb = u_train_tb - u_pred_tb
        r_int = self.compute_pde_residual(inp_train_int)

        L_sb = torch.mean(r_sb ** 2)
        L_tb = torch.mean(r_tb ** 2)
        L_int = torch.mean(r_int ** 2)

        return torch.log10(self.lambda_u * (L_sb + L_tb) + L_int)

    def fit(self, num_epochs, optimizer, verbose=True):
        """
        Train the PINN model.
        """
        history = []
        for epoch in range(num_epochs):
            if verbose: print("################################ ", epoch, " ################################")
            running_loss = 0

            for j, ((inp_train_sb, u_train_sb), (inp_train_tb, u_train_tb), (inp_train_int, u_train_int)) in enumerate(zip(self.training_set_sb, self.training_set_tb, self.training_set_int)):
                def closure():
                    optimizer.zero_grad()
                    loss = self.compute_loss(inp_train_sb, u_train_sb, inp_train_tb, u_train_tb, inp_train_int, verbose=verbose)
                    loss.backward()
                    running_loss += loss.item()
                    return loss

                optimizer.step(closure=closure)

            if verbose: print(f"Loss: {running_loss / len(self.training_set_sb)}")
            history.append(running_loss)

        return history

    def plotting(self):
        """
        Plot the exact solution and the predicted solution.
        """
        inputs = self.soboleng.draw(100000)
        inputs = self.convert(inputs)

        output = self.approximate_solution(inputs).reshape(-1,)
        exact_output = self.exact_solution(inputs).reshape(-1,)

        fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
        im1 = axs[0].scatter(inputs[:, 1].detach(), inputs[:, 0].detach(), c=exact_output.detach(), cmap="jet")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("t")
        plt.colorbar(im1, ax=axs[0])
        axs[0].grid(True, which="both", ls=":")
        im2 = axs[1].scatter(inputs[:, 1].detach(), inputs[:, 0].detach(), c=output.detach(), cmap="jet")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("t")
        plt.colorbar(im2, ax=axs[1])
        axs[1].grid(True, which="both", ls=":")
        axs[0].set_title("Exact Solution")
        axs[1].set_title("Approximate Solution")

        plt.show()

        err = (torch.mean((output - exact_output) ** 2) / torch.mean(exact_output ** 2)) ** 0.5 * 100
        print(f"L2 Relative Error Norm: {err.item()} %")