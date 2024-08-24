import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
On top of everything big tribute to Chrisian Leo of breaking down the implementation of LSTM models from scratch :P
https://medium.com/towards-data-science/the-math-behind-lstm-9069b835289d
"""

"""
WeightInitializer - custom class that handles the initialization of weights
Crucial step because different initialization methods can significantly affect the convergence behavior
"""

"""
Architecture of my custom nn:
- input layer - information about teams for specific date on which a match against them is player
- preprocess layer - data preprocessing
- analysis hidden layer - information about h2h games from the past
- hidden layers
- output layer - prediction of result, amount of goals scored, corner kicks, yellow/red cards and odds for the game (based on odds from training set)
"""

class WeightInitializer:
    def __init__(self, method='random'):
        self.method = method

    def initialize(self, shape):
        if self.method == 'random':
            return np.random.randn(*shape)
        elif self.method == 'xavier':
            return np.random.randn(*shape) / np.sqrt(shape[0])
        elif self.method == 'he':
            return np.random.randn(*shape) * np.sqrt(2 / shape[0])
        elif self.method == 'uniform':
            return np.random.uniform(-1, 1, shape)
        else:
            raise ValueError(f'Unknown initialization method: {self.method}')

"""
PlotManager - class for plotting train and valid loss
"""
class PlotManager:
    def __init__(self):
        self.fig, self.ax = plt.subplots(3, 1, figsize=(6, 4))

    def plot_losses(self, train_losses, val_losses):
        self.ax.plot(train_losses, label='Training Loss')
        self.ax.plot(val_losses, label='Validation Loss')
        self.ax.set_title('Training and Validation Losses')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')
        self.ax.legend()

    def show_plots(self):
        plt.tight_layout()
        
        
class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after

    Args:
    -----
        patience (int): Number of epochs to wait before stopping the training.
        verbose (bool): If True, prints a message for each epoch where the loss
                        does not improve.
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
    """
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        """
        Determines if the model should stop training.
        
        Args:
            val_loss (float): The loss of the model on the validation set.
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score

        elif score < self.best_score + self.delta:
            self.counter += 1
            
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            
class LSTM:
    """
    Long Short-Term Memory (LSTM) network.
    
    Parameters:
    - input_size: int, dimensionality of input space
    - hidden_size: int, number of LSTM units
    - output_size: int, dimensionality of output space
    - init_method: str, weight initialization method (default: 'xavier')
    """
    def __init__(self, input_size, hidden_size, output_size, init_method='xavier'):
        """
        initialization of lstm instance with specified sizes for each layer and selecting a method for weight initialization
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weight_initializer = WeightInitializer(method=init_method)

        # Initialize weights for the gates
        #forget gate
        self.wf = self.weight_initializer.initialize((hidden_size, hidden_size + input_size))
        #input gate
        self.wi = self.weight_initializer.initialize((hidden_size, hidden_size + input_size))   
        #output gate
        self.wo = self.weight_initializer.initialize((hidden_size, hidden_size + input_size))
        #cell
        self.wc = self.weight_initializer.initialize((hidden_size, hidden_size + input_size))

        # Initialize biases
        """
        Biases for all gates and the output layer are initialized to zero. This is a common practice, although sometimes small constants are added to avoid dead neurons at the start.
        """
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))

        # Initialize output layer weights and biases
        self.why = self.weight_initializer.initialize((output_size, hidden_size))
        self.by = np.zeros((output_size, 1))

    @staticmethod
    def sigmoid(z):
        """
        Sigmoid activation function.
        
        Parameters:
        - z: np.ndarray, input to the activation function
        
        Returns:
        - np.ndarray, output of the activation function
        """
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def dsigmoid(y):
        """
        Derivative of the sigmoid activation function.

        Parameters:
        - y: np.ndarray, output of the sigmoid activation function

        Returns:
        - np.ndarray, derivative of the sigmoid function
        """
        return y * (1 - y)

    @staticmethod
    def dtanh(y):
        """
        Derivative of the hyperbolic tangent activation function.

        Parameters:
        - y: np.ndarray, output of the hyperbolic tangent activation function

        Returns:
        - np.ndarray, derivative of the hyperbolic tangent function
        """
        return 1 - y * y

    def forward(self, x):
        """
        Forward pass through the LSTM network.

        Parameters:
        - x: np.ndarray, input to the network

        Returns:
        - np.ndarray, output of the network
        - list, caches containing intermediate values for backpropagation
        """
        caches = []
        h_prev = np.zeros((self.hidden_size, 1))
        c_prev = np.zeros((self.hidden_size, 1))
        h = h_prev
        c = c_prev

        for t in range(x.shape[0]):
            x_t = x[t].reshape(-1, 1)
            combined = np.vstack((h_prev, x_t))
            
            f = self.sigmoid(np.dot(self.wf, combined) + self.bf)
            i = self.sigmoid(np.dot(self.wi, combined) + self.bi)
            o = self.sigmoid(np.dot(self.wo, combined) + self.bo)
            c_ = np.tanh(np.dot(self.wc, combined) + self.bc)
            
            c = f * c_prev + i * c_
            h = o * np.tanh(c)

            cache = (h_prev, c_prev, f, i, o, c_, x_t, combined, c, h)
            caches.append(cache)

            h_prev, c_prev = h, c

        y = np.dot(self.why, h) + self.by
        return y, caches

    def backward(self, dy, caches, clip_value=1.0):
        """
        Backward pass through the LSTM network.

        Parameters:
        - dy: np.ndarray, gradient of the loss with respect to the output
        - caches: list, caches from the forward pass
        - clip_value: float, value to clip gradients to (default: 1.0)

        Returns:
        - tuple, gradients of the loss with respect to the parameters
        """
        dWf, dWi, dWo, dWc = [np.zeros_like(w) for w in (self.wf, self.wi, self.wo, self.wc)]
        dbf, dbi, dbo, dbc = [np.zeros_like(b) for b in (self.bf, self.bi, self.bo, self.bc)]
        dWhy = np.zeros_like(self.why)
        dby = np.zeros_like(self.by)

        # Ensure dy is reshaped to match output size
        dy = dy.reshape(self.output_size, -1)
        dh_next = np.zeros((self.hidden_size, 1))  # shape must match hidden_size
        dc_next = np.zeros_like(dh_next)

        for cache in reversed(caches):
            h_prev, c_prev, f, i, o, c_, x_t, combined, c, h = cache

            # Add gradient from next step to current output gradient
            dh = np.dot(self.why.T, dy) + dh_next
            dc = dc_next + (dh * o * self.dtanh(np.tanh(c)))

            df = dc * c_prev * self.dsigmoid(f)
            di = dc * c_ * self.dsigmoid(i)
            do = dh * self.dtanh(np.tanh(c))
            dc_ = dc * i * self.dtanh(c_)

            dcombined_f = np.dot(self.wf.T, df)
            dcombined_i = np.dot(self.wi.T, di)
            dcombined_o = np.dot(self.wo.T, do)
            dcombined_c = np.dot(self.wc.T, dc_)

            dcombined = dcombined_f + dcombined_i + dcombined_o + dcombined_c
            dh_next = dcombined[:self.hidden_size]
            dc_next = f * dc

            dWf += np.dot(df, combined.T)
            dWi += np.dot(di, combined.T)
            dWo += np.dot(do, combined.T)
            dWc += np.dot(dc_, combined.T)

            dbf += df.sum(axis=1, keepdims=True)
            dbi += di.sum(axis=1, keepdims=True)
            dbo += do.sum(axis=1, keepdims=True)
            dbc += dc_.sum(axis=1, keepdims=True)

        dWhy += np.dot(dy, h.T)
        dby += dy

        gradients = (dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby)

        # Gradient clipping
        for i in range(len(gradients)):
            np.clip(gradients[i], -clip_value, clip_value, out=gradients[i])

        return gradients

    def update_params(self, grads, learning_rate):
        """
        Update the parameters of the network using the gradients.

        Parameters:
        - grads: tuple, gradients of the loss with respect to the parameters
        - learning_rate: float, learning rate
        """
        dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby = grads

        self.wf -= learning_rate * dWf
        self.wi -= learning_rate * dWi
        self.wo -= learning_rate * dWo
        self.wc -= learning_rate * dWc

        self.bf -= learning_rate * dbf
        self.bi -= learning_rate * dbi
        self.bo -= learning_rate * dbo
        self.bc -= learning_rate * dbc

        self.why -= learning_rate * dWhy
        self.by -= learning_rate * dby