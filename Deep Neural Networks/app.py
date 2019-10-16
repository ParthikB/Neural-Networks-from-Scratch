from flask import Flask, render_template, request
import numpy as np 
import joblib
from main import training
from helper_functions import plot_data, plot_cost_function

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/prediction', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # try:
            # Taking the inputs and saving them to a variable
        TRAINING_SAMPLES = int(request.form['TRAINING_SAMPLES'])
        
        LAYER1_DIM       = int(request.form['LAYER1_DIM'])
        LAYER2_DIM       = int(request.form['LAYER2_DIM'])
        LAYER3_DIM       = int(request.form['LAYER3_DIM'])
        LAYER_DIMS       = [LAYER1_DIM, LAYER2_DIM, LAYER3_DIM]

        EPOCHS           = int(request.form['EPOCHS'])
        LEARNING_RATE    = float(request.form['LEARNING_RATE'])
        ACTIVATION       = str(request.form['ACTIVATION'])
        print(TRAINING_SAMPLES, LAYER_DIMS, EPOCHS, LEARNING_RATE, ACTIVATION)
        accuracy= training(TRAINING_SAMPLES, LAYER_DIMS, EPOCHS, LEARNING_RATE, ACTIVATION)

        # except:
        #     return 'Invalid values entered!'

    return render_template('prediction.html', accuracy = accuracy)


if __name__ == '__main__':
    app.run()