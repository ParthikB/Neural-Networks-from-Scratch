"""
  This is where the heart of this project resides. 
  Here you'll find how the Network is created and the various steps involved in the process.

  Have fun!

"""


class NeuralNetwork:    
  
  # Defining the Feedforward function
  def feedforward(self, X, parameters, activation_used):

    def linear_forward(A, W, b, activation):
      Z = np.dot(W, A) + b
      linear_cache = (A, W, b)

      if activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)
      elif activation == 'relu':
        A, activation_cache = relu(Z)
      
      # Saving some variables that will be needed later in Back Propagation in the form of caches.
      cache = (linear_cache, activation_cache)
      return A, cache

    caches = []
    A = X
    L = len(parameters) // 2  # Total number of Layers in our Network
    
    # Iterating over every Layer and computing the Activations.
    # See how different activations are being used.
    # You can play with the activations in the hidden layers, but the Output layer activation is always set to 'sigmoid'.
    for i in range(1, L):
        A, cache = linear_forward(A, parameters["W" + str(i)], parameters["b" + str(i)], activation_used)
        caches.append(cache)

    A, cache = linear_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], 'sigmoid')
    caches.append(cache)

    return A, caches


  # Defining the cost function.
  def cost(self, yhat, y):
    m = y.shape[1]
    cost = -np.sum(y * np.log(yhat) + (1-y) * np.log(1-yhat)) / m
    return cost

  
  # Defining the Back Progation Algorithm.
  def backward_propagation(self, yhat, y, caches, activation_used):

    def linear_backward(dA, cache, activation):
      linear_cache, activation_cache = cache
      A, W, b = linear_cache

      if activation == 'sigmoid':
        dZ = sigmoid_derivative(dA, activation_cache)
      elif activation == 'relu':
        dZ = relu_derivative(dA, activation_cache)  
        
      A_prev = A
      m = A_prev.shape[1]

      dW = np.dot(dZ, A_prev.T) / m
      db = np.sum(dZ, axis=1, keepdims=True) / m
      dA_prev = np.dot(W.T, dZ)

      return dA_prev, dW, db


    L = len(caches)
    grads = {}
    y = y.reshape(yhat.shape)
    dyhat = -(np.divide(y, yhat) - np.divide(1-y, 1-yhat))
    dAL = dyhat
    
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_backward(dAL, current_cache, 'sigmoid')
    for l in range(L-1)[::-1]:
        current_cache = caches[l]
        grads["dA" + str(l)], grads["dW" + str(l+1)], grads["db" + str(l+1)] = linear_backward(grads["dA" + str(l+1)], current_cache, activation_used)

    return grads
    
  
  # Defining the Gradient Descent Algorithm. Learning Rate is set to default at 0.01
  def gradient_descent(self, parameters, grads, learning_rate=0.01):
    L = parameters.__len__() // 2
    
    for l in range(1, L+1):
      parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
      parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]

    return parameters
