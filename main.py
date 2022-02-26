"""
Main file - Programming an artificial neuron

Objective : Determine if a plant is toxic according to 2 parameters x1 and x2
x1 : Length of the leaves
x2 : Width of the leaves
"""

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score


def make_dataset():
    """
    Creation of a dataset
    :return x_values, y_values : data matrix and answers learning data
    """
    x_values, y_values = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
    y_values = y_values.reshape((y_values.shape[0], 1))
    return x_values, y_values


def show_data(x_values, y_values, w_vector, bias):
    """
    Displaying the data
    :param x_values : training data,
    :param y_values : learning data responses
    :param w_vector : weight vector
    :param bias : bias
    """

    # Decision Frontier
    x_min = np.linspace(-1, 4, 100)
    x_max = (-w_vector[0] * x_min - bias) / w_vector[1]

    # Displaying data in 2D
    plt.scatter(x_values[:, 0], x_values[:, 1], c=y_values, cmap='summer')
    plt.plot(x_min, x_max, 'orange', lw=2)
    plt.show()

    # Displaying data in 3D
    fig = go.Figure(data=[go.Scatter3d(
        x=x_values[:, 0].flatten(),
        y=x_values[:, 1].flatten(),
        z=y_values.flatten(),
        mode='markers',
        marker=dict(
            size=5,
            color=y.flatten(),
            colorscale='YlGn',
            opacity=0.8,
            reversescale=True
        )
    )])

    # Display of the sigmoid function
    x_min_3d, x_max_3d = np.meshgrid(x_min, x_max)
    z_values = w_vector[0] * x_min_3d + w_vector[1] * x_max_3d + bias
    a_values = 1 / (1 + np.exp(-z_values))

    fig = (go.Figure(data=[
        go.Surface(z=a_values, x=x_min_3d, y=x_max_3d, colorscale='Ylgn', opacity=0.7,
                   reversescale=True)]))
    fig.add_scatter3d(x=x_values[:, 0].flatten(), y=x_values[:, 1].flatten(), z=y_values.flatten(),
                      mode='markers',
                      marker=dict(size=5, color=y.flatten(), colorscale='YlGn', opacity=0.8,
                                  reversescale=True))

    fig.update_layout(template='plotly_dark', margin=dict(t=0, b=0, l=0, r=0))
    fig.layout.scene.camera.projection.type = 'orthographic'
    fig.show()


def initialisation(dataset):
    """
    Initialization of the weight matrix W
    :param dataset : training data
    :return w_vector, bias : returns a tuple of weight and bias matrices
    """
    w_vector = np.random.randn(dataset.shape[1], 1)
    bias = np.random.randn(1)
    return w_vector, bias


def model(dataset, w_vector, bias):
    """
    Calculation of the model output
    :param dataset : training data
    :param w_vector : weight vector
    :param bias : biais
    :return y_values : learning data responses
    """
    vector_z = dataset.dot(w_vector) + bias
    vector_a = 1 / (1 + np.exp(-vector_z))
    return vector_a


def log_loss(vector_a, y_values):
    """
    Calculation of the cost function
    :param vector_a : model output
    :param y_values : learning data responses
    :return loss : value of the cost function
    """
    # m = len(y_values)
    return 1 / len(y_values) * np.sum(
        -y_values * np.log(vector_a) - (1 - y_values) * np.log(1 - vector_a))


def gradients(vector_a, x_values, y_values):
    """
    Calculation of gradients
    :param vector_a : model output
    :param x_values : training data
    :param y_values : learning data responses
    :return w_grad_vector, bias_grad : returns the gradients
    """
    # m = len(y_values)
    w_grad_vector = 1 / len(y_values) * np.dot(x_values.T, (vector_a - y_values))
    bias_grad = 1 / len(y_values) * np.sum(vector_a - y_values)
    return w_grad_vector, bias_grad


def update(w_grad_vector, bias_grad, w_vector, bias, learning_rate):
    """
    Update of weights and bias
    :param w_grad_vector : gradient vector
    :param bias_grad : gradient bias
    :param w_vector : weight vector
    :param bias : bias
    :param learning_rate : learning rate
    :return w_vector, bias : returns the new weights and biases
    """
    w_vector = w_vector - learning_rate * w_grad_vector
    bias = bias - learning_rate * bias_grad
    return w_vector, bias


def predict(vector_x, w_vector, bias):
    """
    Prediction of the model output
    :param vector_x : training data
    :param w_vector : weight vector
    :param bias : bias
    :return vector_a : returns the outputs of the model
    """
    vector_a = model(vector_x, w_vector, bias)
    return vector_a >= 0.5


def artificial_neuron(x_values, y_values, learning_rate, epochs):
    """
    Learning the model
    :param x_values : training data
    :param y_values : learning data responses
    :param learning_rate : learning rate
    :param epochs : number of iterations
    :return w_vector, bias : returns the weights and biases
    """
    # Initialization of weights and bias
    w_vector, bias = initialisation(x_values)

    # Value of w_vector and bias for each iteration
    history = []

    # Cost curve
    loss = []

    # Learning loop
    for index in range(epochs):
        vector_a = model(x_values, w_vector, bias)
        loss.append(log_loss(vector_a, y_values))
        w_grad_vector, bias_grad = gradients(vector_a, x_values, y_values)
        w_vector, bias = update(w_grad_vector, bias_grad, w_vector, bias, learning_rate)
        history.append([w_vector, bias, loss, index])

    # Calculation of the prediction
    y_pred = predict(x_values, w_vector, bias)

    # Prediction display
    y_pred_percent = accuracy_score(y_values, y_pred) * 100
    print(f"\n Précision du modèle : {int(y_pred_percent)}%")

    # Display of the cost curve
    plt.plot(loss)
    plt.show()

    return w_vector, bias


if __name__ == '__main__':
    # Creation of a dataset
    X, y = make_dataset()

    # Start the prediction
    w_result, b_result = artificial_neuron(X, y, 0.06, 100)

    # Data display
    show_data(X, y, w_result, b_result)
