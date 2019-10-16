from helper_functions import *
from network import *

'''
  Welcome fellas!

  Here we're creating a Deep Neural Network from scratch (without using any predefined Machine Learning Library!)
  If you want to know more (kinda tutorial), kindly visit the following, 

  https://medium.com/@rockon.parthik555/neural-networks-in-a-nutshell-5f989409082c

  Have fun!
'''

if __name__=="__main__":
    # CHANGABLE PARAMETERS
    TRAINING_SAMPLES = 1000 # Total number of training samples.
    LAYER_DIMS = [16, 16, 1]     # Note that the input Layer is predefined so you don't need to define it again.
    EPOCHS = 250          # Total number of Iterations.
    LEARNING_RATE = 0.05    # Learning Rate to be used in Gradiend Descent.
    ACTIVATION = 'relu'     # Activations used in Neural Network. Try jumping between relu/sigmoid



def training(TRAINING_SAMPLES, LAYER_DIMS, EPOCHS, LEARNING_RATE=0.04, ACTIVATION='relu'):

    print(LAYER_DIMS, type(LAYER_DIMS))
    # Just a temporary solution to fix the layer_dim isse
    # master = []
    # for i in LAYER_DIMS:
    #     try:
    #         master.append(int(i))
    #     except:
    #         pass
    # LAYER_DIMS = master

    # Creating Data
    X, y = create_data(TRAINING_SAMPLES, 100)

    # Saving the training data graph
    plot_data([X, y])

    # Initializing Random Parameters
    parameters = initialize_random_parameters(LAYER_DIMS, X)

    # Few logs just to keep track of our training.
    cost_log, epoch_log = [], []

    # Creating the Network
    nn = NeuralNetwork()

    # Training
    # print("Initializing Training...")
    for epoch in range(EPOCHS):

        if epoch % 100 == 0 and epoch != 0:
            LEARNING_RATE -= LEARNING_RATE/10     # This is called Learning Rate Decay. It is basically done to optimize our Training.
            print("Epoch :", epoch)

        # Feedforwarding
        yhat, caches = nn.feedforward(X, parameters, ACTIVATION)

        # Computing and saving the logs for plotting
        cost = nn.cost(yhat, y)
        cost_log.append(cost)
        epoch_log.append(epoch+1)

        # np.save("cost_log.npy", [epoch_log, cost_log])

        # Back Propagation
        grads = nn.backward_propagation(yhat, y, caches, ACTIVATION)
        # Gradient Descent
        parameters = nn.gradient_descent(parameters, grads, LEARNING_RATE)


    predictions = yhat  # yhat --> the predicted output

    # print()
    # print("********** Accuracy :", accuracy_score(predictions, y), "% **********")
    accuracy = accuracy_score(predictions, y)

    # Saving the Cost Function Graph
    plot_cost_function(epoch_log, cost_log)

    # Just another way to convert predictions to 0s and 1s
    yhat = np.where(predictions<0.5, 0, 1)

    # Saving our Predictions graph
    plot_data([X, yhat], 'predicted')

    # print("// Graphs saved.")

    return accuracy


# print(training(TRAINING_SAMPLES=2500, LAYER_DIMS=[16, 16, 1], EPOCHS=2500))