import torch
import torch.nn as nn
import torch.optim as optim


class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoLayerNet, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x


class LinearNet(nn.Module):
    def __init__(self, input_dim, output_dim, noise_std=0.01):
        super(LinearNet, self).__init__()
        self.output = nn.Linear(input_dim, output_dim, bias=False)
        with torch.no_grad():
            identity_matrix = torch.eye(output_dim, input_dim)
            noise = torch.randn_like(identity_matrix) * noise_std
            self.output.weight.copy_(identity_matrix + noise)

    def forward(self, x):
        x = self.output(x)
        return x


def get_device(device_option):
    if device_option == "gpu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():  # For Apple Silicon (M1, M2) devices
            return torch.device("mps")
        else:
            raise ValueError("GPU requested but not available.")
    elif device_option == "cpu":
        return torch.device("cpu")
    elif device_option == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():  # For Apple Silicon (M1, M2, etc) devices
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        raise ValueError("Invalid device option. Choose from 'gpu', 'cpu', or 'auto'.")


def train_neural_network(X_train, y_train, hidden_dim, num_epochs=1000, learning_rate=0.0001, device='auto'):
    # Convert input data to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    device = get_device(device)

    input_dim = 1
    if len(X_train.shape) == 2:
       input_dim = X_train.shape[1]
    output_dim = 1
    if len(y_train.shape) == 2:
        output_dim = y_train.shape[1]

    # Define the network
    if hidden_dim > 0:
        model = TwoLayerNet(input_dim, hidden_dim, output_dim)
    else:
        model = LinearNet(input_dim, output_dim)

    # Define the loss function and optimizer
    # criterion = lambda y, t: torch.mean(torch.sum((1 - y/t)**2, dim=1))
    criterion = nn.HuberLoss(delta=10)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the network
    for epoch in range(num_epochs):
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(X_train)

        # Compute and print loss.
        loss = criterion(y_pred, y_train)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss every 100 epochs
        if (epoch +1) % 100 == 0:
            print(f'Epoch [{epoch +1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Return the trained predictions
    with torch.no_grad():
        y_trained = model(X_train).cpu().numpy()

    if output_dim == 1:
        y_trained = y_trained[:, 0]

    return y_trained
