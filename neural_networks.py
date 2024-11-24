import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        self.activation_fn = activation # activation function

        # TODO: define layers and initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))

        self.hidden_features = None
        self.gradients = {'W1': None, 'W2': None}
        self.loss_history = []

    def activation(self, x):
        if self.activation_fn == 'tanh':
            return np.tanh(x)
        elif self.activation_fn == 'relu':
            return np.maximum(0, x)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-x))

    def activation_derivative(self, x):
        if self.activation_fn == 'tanh':
            return 1 - np.tanh(x)**2
        elif self.activation_fn == 'relu':
            return (x > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            sx = 1 / (1 + np.exp(-x))
            return sx * (1 - sx)


    def forward(self, X):
        # TODO: forward pass, apply layers to input X
        # TODO: store activations for visualization

        self.X = X

        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation(self.z1)
        self.hidden_features = self.a1  # Store for visualization
        
        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.output = np.tanh(self.z2)  # Using tanh for final activation
        
        return self.output

    def compute_loss(self, y):
        # Binary cross-entropy loss adapted for tanh output (-1, 1)
        # Transform y from {-1, 1} to {0, 1} and output from (-1, 1) to (0, 1)
        y_transformed = (y + 1) / 2
        output_transformed = (self.output + 1) / 2
        
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        output_transformed = np.clip(output_transformed, epsilon, 1 - epsilon)
        
        # Compute binary cross-entropy
        loss = -np.mean(y_transformed * np.log(output_transformed) + 
                       (1 - y_transformed) * np.log(1 - output_transformed))
        
        return loss

    def backward(self, X, y):
        m = X.shape[0]
        
        # Compute loss first
        loss = self.compute_loss(y)
        self.loss_history.append(loss)
        
        # Output layer gradients
        # Derivative of loss with respect to output
        output_transformed = (self.output + 1) / 2
        y_transformed = (y + 1) / 2
        
        # Gradient of binary cross-entropy with respect to transformed output
        grad_transformed = -(y_transformed / output_transformed - 
                           (1 - y_transformed) / (1 - output_transformed)) / m
        
        # Chain rule with derivative of tanh transformation
        delta2 = grad_transformed * (1 - self.output**2) / 2
        
        # Gradients for W2 and b2
        dW2 = np.dot(self.a1.T, delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        delta1 = np.dot(delta2, self.W2.T) * self.activation_derivative(self.z1)
        dW1 = np.dot(X.T, delta1)
        db1 = np.sum(delta1, axis=0, keepdims=True)
        
        # Store gradients for visualization
        self.gradients['W1'] = dW1
        self.gradients['W2'] = dW2
        
        # Update weights with gradient descent
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        
        return loss


def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # perform training steps by calling forward and backward function
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        mlp.backward(X, y)
        
    # TODO: Plot hidden features
    hidden_features = mlp.hidden_features
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], 
                     c=y.ravel(), cmap='bwr', alpha=0.7)

    # TODO: Hyperplane visualization in the hidden space

    xx = np.linspace(hidden_features[:, 0].min() - 0.5, hidden_features[:, 0].max() + 0.5, 100)
    yy = np.linspace(hidden_features[:, 1].min() - 0.5, hidden_features[:, 1].max() + 0.5, 100)
    zz = np.linspace(hidden_features[:, 2].min() - 0.5, hidden_features[:, 2].max() + 0.5, 100)
    XX, YY, ZZ = np.meshgrid(xx, yy, zz)

    grid_points = np.column_stack((XX.ravel(), YY.ravel(), ZZ.ravel()))
    decisions = np.dot(grid_points, mlp.W2) + mlp.b2
    decisions = decisions.reshape(XX.shape)
    ax_hidden.contour(XX[:,:,25], YY[:,:,25], decisions[:,:,25], levels=[0], 
                     colors='k', alpha=0.5, linestyles='--')

    # Solve for z: W0*x + W1*y + W2*z + b = 0
    z = (-mlp.W2[0] * xx - mlp.W2[1] * yy - mlp.b2) / mlp.W2[2]

    # Plot the hyperplane
    ax_hidden.plot_surface(xx, yy, z, color='gray', alpha=0.5, edgecolor='none')
    ax_hidden.set_title('Hidden Layer Space')
    

    # TODO: Distorted input space transformed by the hidden layer

    xx_input = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 50)
    yy_input = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 50)
    XX_input, YY_input = np.meshgrid(xx_input, yy_input)
    grid_input = np.column_stack((XX_input.ravel(), YY_input.ravel()))

    Z_transformed = mlp.forward(grid_input).reshape(XX_input.shape)

    # TODO: Plot input layer decision boundary

    if mlp.activation_fn == 'sigmoid':
        Z_transformed = 1 / (1 + np.exp(-Z_transformed))  # Apply sigmoid for binary classification
    elif mlp.activation_fn == 'tanh':
        Z_transformed = (Z_transformed + 1) / 2 

    contour = ax_input.contourf(
        XX_input, YY_input, Z_transformed, levels=50, cmap='RdBu', alpha=0.7  # Increased levels for smoother gradients
    )
    scatter = ax_input.scatter(
        X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k', s=50, alpha=0.8  # Added edge color for better visibility
    )

    ax_input.set_title('Input Space & Decision Boundary')
    current_step = frame * 10  # Update current step label
    ax_input.text(
        0.02, 0.98, f'Step: {current_step}', transform=ax_input.transAxes,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
        verticalalignment='top'
    )

    # circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--')
    # ax_input.add_artist(circle)
    # ax_input.set_title('Input Space & Decision Boundary')

    # current_step = frame * 10  # Since we do 10 steps per frame
    # ax_input.text(0.02, 0.98, f'Step: {current_step}', 
    #              transform=ax_input.transAxes,
    #              bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
    #              verticalalignment='top')


    # TODO: Visualize features and gradients as circles and edges 
    # The edge thickness visually represents the magnitude of the gradient

    # Input layer neurons
    ax_gradient.scatter([0, 0], [0, 1], c='g', s=200, label='Input')
    # Hidden layer neurons
    ax_gradient.scatter([0.5, 0.5, 0.5], [0, 0.5, 1], c='b', s=200, label='Hidden')
    # Output neuron
    ax_gradient.scatter([1], [0.5], c='r', s=200, label='Output')

    # First layer connections
    max_grad = max(np.abs(mlp.gradients['W1']).max(), np.abs(mlp.gradients['W2']).max())
    max_grad = max_grad if max_grad != 0 else 1
    
    # Input to hidden connections
    for i in range(2):  # input dimension
        for j in range(3):  # hidden dimension
            gradient = mlp.gradients['W1'][i, j]
            thickness = 5 * abs(gradient) / max_grad
            ax_gradient.plot(
                [0, 0.5], [i, j / 2],  # Coordinates for the edge
                linewidth=thickness,
                color="blue" if gradient > 0 else "red",
                alpha=0.7,
            )
    
    # Hidden to output connections
    for i in range(3):
        gradient = mlp.gradients['W2'][i, 0]
        thickness = 5 * abs(gradient) / max_grad
        ax_gradient.plot(
            [0.5, 1], [i / 2, 0.5],  # Coordinates for the edge
            linewidth=thickness,
            color="blue" if gradient > 0 else "red",
            alpha=0.7,
        )
    
    ax_gradient.set_xlim(-0.2, 1.2)
    ax_gradient.set_ylim(-0.2, 1.2)
    ax_gradient.legend()
    ax_gradient.set_title('Network Architecture & Gradients')




def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)