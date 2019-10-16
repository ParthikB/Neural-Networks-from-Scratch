"""
  These are some helper functions which we'll need to create our DNN.

"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generating a basic dataset.
def create_data(total_samples, range_of_data):
    X1, X2, Y = [], [], []

    for datapoints in range(total_samples):
        x2 = np.random.randint(1, range_of_data + 1)
        x1 = np.random.randint(1, range_of_data + 1)

        if x1 < range_of_data / 2:
            label = 0
        else:
            label = 1
        X1.append(x1)
        X2.append(x2)
        X = np.array([X1, X2])
        Y.append(label)

    return np.array(X).reshape(2, -1), np.array(Y).reshape(1, -1)


# Defining Activation Functions
def sigmoid(z):
    A =  1 / (1 + np.exp(-z))
    cache = z
    return A, cache


def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache


def relu_derivative(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0

    return dZ


def sigmoid_derivative(dA, cache):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ


def initialize_random_parameters(layer_dims, X):
    parameters = {}
    parameters["W1"] = np.random.randn(layer_dims[0], X.shape[0]) * 0.01
    parameters["b1"] = np.zeros((layer_dims[0], 1))
    for i in range(1, len(layer_dims)):
        parameters["W" + str(i+1)] = np.random.randn(layer_dims[i], layer_dims[i-1]) * 0.01
        parameters["b" + str(i+1)] = np.zeros((layer_dims[i], 1))

    return parameters

  
# To check the accuracy of our Model.
def accuracy_score(Yhat, Y):
    Yhat = np.where(Yhat < 0.5, 0, 1)
    accuracy = 100 - np.mean(np.abs(Yhat-Y) * 100)
    return accuracy


# To visualize our dataset/predictions.
def plot_data(data, type_of_data='original'):
    X = data[0]
    Y = data[1]
    sns.scatterplot(X[0], X[1], hue=Y[0])
    plt.xlabel('Feature 1')
    plt.ylabel("Feature 2")
    plt.title(type_of_data)
    type_of_data = 'graphs/' + type_of_data + 'Dataset' + ".png"
    plt.savefig(type_of_data)
    # plt.show()
    plt.close()

    
def plot_cost_function(epoch_log, cost_log):
    plt.plot(epoch_log, cost_log)
    plt.xlabel('Epochs')
    plt.ylabel("Cost")
    plt.title("Cost Function")
    plt.savefig('graphs/costFunction.png')
    #   plt.show()
    plt.close()
