import numpy
import math
from scipy.special import expit as sigmoid
import scipy.ndimage
# For saving
import pickle
import os
import zipfile as zf


class NeuralNetwork:
    """ A three layer neural network that uses the sigmoid function as its activation function """
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        """ Initialize Neural Network with number of nodes in each layer and the learning rate """
        # Declare zip object for saving
        self.zip = None

        # specify number of nodes in each layer
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # weight matrices
        self.weights_input_hidden = numpy.random.normal(0.0, pow(self.input_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.weights_hidden_output = numpy.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.output_nodes, self.hidden_nodes))

        # learning rate
        self.learning_rate = learning_rate

        # set activation function as sigmoid function
        self.activation_function = lambda x: sigmoid(x)

    def train(self, inputs_list, targets_list):
        """ Trains Neural Network with inputs and target outputs as lists """

        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.weights_input_hidden, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.weights_hidden_output, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.weights_hidden_output.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.weights_hidden_output += self.learning_rate * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                                                     numpy.transpose(hidden_outputs))

        # update the weights for the links between the input and hidden layers
        self.weights_input_hidden += self.learning_rate * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                                                    numpy.transpose(inputs))

    def query(self, inputs_list):
        """ Queries the neural network using its current weights to calculate the outputs given a list of inputs """

        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.weights_input_hidden, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.weights_hidden_output, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def train_with_rotated_numbers(self, epochs, to_rotate):
        # Load 0-9 digit training data CSV file into a list
        training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
        training_data_list = training_data_file.readlines()
        training_data_file.close()

        # Train the neural network with image and its left and right rotations by 10 degrees
        for e in range(epochs):
            # go through all records in the training data set
            for record in training_data_list:
                # split the record by the ',' commas
                all_values = record.split(',')
                # scale and shift the inputs
                inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                # create the target output values (all 0.01, except the desired label which is 0.99)
                targets = numpy.zeros(self.output_nodes) + 0.01
                # all_values[0] is the target label for this record
                targets[int(all_values[0])] = 0.99
                self.train(inputs, targets)

                # rotated anticlockwise by x degrees
                inputs_plusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28, 28), to_rotate, cval=0.01,
                                                                      order=1,
                                                                      reshape=False)
                self.train(inputs_plusx_img.reshape(784), targets)
                # rotated clockwise by x degrees
                inputs_minusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28, 28), -to_rotate, cval=0.01,
                                                                       order=1,
                                                                       reshape=False)
                self.train(inputs_minusx_img.reshape(784), targets)

    def test_neural_net(self):
        # Load the 0-9 digit test data CSV file into a list
        test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
        test_data_list = test_data_file.readlines()
        test_data_file.close()

        # Initialize correct array
        correct = []

        # Go through all the records in the test data set
        for record in test_data_list:
            # split the record by the ',' commas
            all_values = record.split(',')
            # correct answer is first value
            correct_label = int(all_values[0])
            # scale and shift the inputs
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # query the network
            outputs = self.query(inputs)
            # the index of the highest value corresponds to the label
            label = numpy.argmax(outputs)
            # Append correct or incorrect to list
            if label == correct_label:
                correct.append(1)
            else:
                correct.append(0)

        correct_arr = numpy.asarray(correct)
        return (correct_arr.sum() / correct_arr.size) * 100

    def save_session(self, session_path):
        # Update status
        print("Saving Session")

        # Create zip file to contain the saved data
        self.zip = zf.ZipFile(session_path + ".session", "w")

        # Save every variable
        self.save(self.input_nodes, "input_nodes")
        self.save(self.hidden_nodes, "hidden_nodes")
        self.save(self.output_nodes, "output_nodes")
        self.save(self.weights_input_hidden, "weights_input_hidden")
        self.save(self.weights_hidden_output, "weights_hidden_output")
        self.save(self.learning_rate, "learning_rate")

        self.zip.close()

        # Update success status
        print("Session saved successfully.")

    def load_session(self, session_path):
        # Update status
        print("Loading Session")

        # Open zip file to containing the saved data
        self.zip = zf.ZipFile(session_path, "r")

        # Load every variable
        self.input_nodes = self.load("input_nodes")
        self.hidden_nodes = self.load("hidden_nodes")
        self.output_nodes = self.load("output_nodes")
        self.weights_input_hidden = self.load("weights_input_hidden")
        self.weights_hidden_output = self.load("weights_hidden_output")
        self.learning_rate = self.load("learning_rate")

        self.zip.close()

        # Update success status
        print("Session loaded successfully.")

    def save(self, data, name):
        file_name = './saved/' + name + ".pickle"
        with open(file_name, 'wb') as f:
            pickle.dump(data, f)  # Write to binary file
        f.close()
        self.zip.write(file_name, name + ".pickle")  # Add binary file to zip
        os.remove(file_name)  # Remove binary file

    def load(self, name):
        file_name = name + ".pickle"
        f = self.zip.open(file_name, "r")
        return pickle.load(f)
