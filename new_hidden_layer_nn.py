import numpy as np
import random


np.random.seed(1)

def generate_dataset(max):
    '''
    Generates the input and output of the dataset
    The input are two random numbers from -100 to 100
    and the output a boolean which is true if the sum of
    the two numbers is >=0, otherwise gets false
    '''
    inputs = []
    outputs = []
    a = 0
    b = 0
    for i in range(0,max):
        a = random.randint(-100, 100)
        b = random.randint(-100, 100)
        inputs.append([a,b])
        outputs.append(1) if a + b >= 1 else outputs.append(0)
    return inputs, outputs


def layer_sizes(X,Y):
    '''
    Auxiliar function thata generates the layer dimensions
    Needed in the neural network process
    Parameters:
        X: Input data
        Y: Output data
    Returns:
        n_x: Input layer data size
        n_h: Hidden layer data size
        n_h: Output layer data size
    '''

    n_x=X.shape[0]
    n_h=3 #Arbitrary value, a hyperparameter
    n_y=Y.shape[0]

    return n_x, n_h, n_y


def initialize(n_x, n_h, n_y):
    '''
    Initializes the weights and bias of the network
    Requires the layers dimensions obtained before
    And returns a dictionary with all parameters 
    '''
    W1=np.random.randn(n_h,n_x)*0.01
    b1=np.zeros((n_h,1))
    W2=np.random.randn(n_y,n_h)*0.01
    b2=np.zeros((n_y,1))

    parameters={"W1":W1,
                "b1":b1,
                "W2":W2,
                "b2":b2}
    return parameters


def sigmoid(Z):
    '''
    Mathematical sigmoid function
    Used for forward propagation
    Recieves an array to eval and return sigmoid operation result
    '''
    s = 1/(1+np.exp(-Z))
    return s


def forward_propagation(X,parameters):
    '''
    Function that predicts all examples in the network
    Input:
        X: Dataset with all input data
        Parameters: Python dictionary with weights and bias
    Returns:
        A2: Final prediction
        Cache: Dictionary with the gradients of the parameters
    '''

    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]

    Z1=np.dot(W1,X)+b1
    A1=np.tanh(Z1)
    Z2=np.dot(W2,A1)+b2
    A2=sigmoid(Z2)

    cache={"Z1":Z1,
           "A1":A1,
           "Z2":Z2,
           "A2":A2}

    return A2, cache


def compute_cost(A2,Y):
    '''
    The cost is the aproximation of the real prediction
    In this context, is just a calculus to verify the network accuracy
    Parameters:
        A2: Array with all calculated predictions
        Y: Array with all True labels
    Returns:
        Cost: Array with the dataset cost
    '''

    m=Y.shape[1]
    cost = (-1/m)*(np.sum(Y*np.log(A2)+(1-Y)*np.log(1-A2)))
    cost = np.squeeze(cost)
    return cost

def backward_propagation(parameters,cache,X,Y):
    '''
    Calculates the new values of the parameters in the network
    Parameters:
        Parameters: Python dictionary with all the weights W1, b1, W2, b2
        Cache: Python dictionary with the values of Z1, A1, Z2, A2
        X: Input data
        Y: True Labels
    Returns:
        Grads: Python dictionary with all derivatives values of the parameters
    '''

    m=Y.shape[1]

    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]

    A1=cache["A1"]
    A2=cache["A2"]

    dz2=A2-Y
    dW2=(1/m)*np.dot(dz2,A1.T)
    db2=(1/m)*np.sum(dz2,axis=1,keepdims=True)
    dz1=np.dot(W2.T,dz2)*(1-np.power(A1,2))
    dW1=(1/m)*np.dot(dz1,X.T)
    db1=(1/m)*np.sum(dz1,axis=1,keepdims=True)

    grads={"dW1":dW1,
           "db1":db1,
           "dW2":dW2,
           "db2":db2}

    return grads

def update_parameters(parameters,grads,learning_rate):
    '''
    Updated the parameters of the network with the derivates obtained before
    Parameters:
        Parameters: Python dictionary with all the weights W1, b1, W2, b2
        Grads: Python dictionary with all derivatives
        Learning_rate: hyperparameter to set the learning rate of the network
    Returns:
        Parameters: Same dictionary from input with new  values updated
    '''

    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]

    dW1=grads["dW1"]
    db1=grads["db1"]
    dW2=grads["dW2"]
    db2=grads["db2"]

    W1=W1 - dW1*learning_rate
    b1=b1 - db1*learning_rate
    W2=W2 - dW2*learning_rate
    b2=b2 - db2*learning_rate

    parameters={"W1":W1,
                "b1":b1,
                "W2":W2,
                "b2":b2}

    return parameters

def nn_model(X, Y, n_h, num_iterations, learning_rate, print_cost=False):
    '''
    Main loop, runs the script and trains the network for a specific perior
    Parameters:
        X: Train dataset
        Y: True labels of the dataset
        n_h: Hidden layers size, 3 by default
        Num_iterations: Number of times the network will train
        Learning_rate: hyperparameter to set the learning rate of the network
        Print_cost = A boolean value that defines if information is shown
    Returns:
        Parameters: Final parameters dictionary ready to do predictions
    '''

    #Step 1, generate the network dimensions
    n_x=layer_sizes(X,Y)[0]
    n_y=layer_sizes(X,Y)[2]

    #Step 2, initialize the parameters
    parameters=initialize(n_x,n_h,n_y)
    costs = []
    its = []

    #Step 3, main loop that trains the network

    for i in range(0,num_iterations):

        #Makes the forwrd propagation
        A2,cache=forward_propagation(X,parameters)

        #Computes the cost
        cost=compute_cost(A2,Y)

        #Calculates the backward propagation
        grads=backward_propagation(parameters,cache,X,Y)

        #Updates the parameters
        parameters=update_parameters(parameters,grads,learning_rate)

        #Show information if print_cost is True
        if (print_cost and i%10==0):
            predictions = predict(parameters, X)
            counter = 0
            for l1,l2 in zip(predictions[0],Y[0]):
                if l1 == l2:
                    counter+=1
            
            print(f'Cost after iteration {i}: {cost.round(6)}')
            print(f'Current accuracy: {(counter / len(Y[0]))*100}%')

            input('press enter to see the next iteration')

        if i%500 == 0:
            costs.append(cost)
            its.append(i)
        
    print(f'Final accuracy: {(counter / len(Y[0]))*100}%')

    return parameters, its, costs


def predict(parameters,X):
    '''
    Predicts from a new input and returns an array of predictions
    Parameters:
        Parameters: Python dictionary with final weights
        X: Input dataset
    Returns:
        Predictions: Array with all predictions
    '''

    A2, cache = forward_propagation(X,parameters)

    predictions=(A2>0.5)

    return predictions


def train(X, Y):
    '''
    Auxiliar function that defines some of the hyper parameters and
    executes the training
    '''

    full_parameters, its, costs=nn_model(X, Y, 3, 250, 0.0005, True)

    final_test(full_parameters)

    return full_parameters, its, costs


def final_test(parameters):
    '''
    Let the user test his own example
    '''
    test_inputs = []
    print('Please fill your personal test')
    input1 = float(input('Please fill your first number: '))
    input2 = float(input('Please fill your second number: '))
    test_inputs.append([input1, input2])
    test_inputs= np.array(test_inputs).T

    prediction = predict(parameters, test_inputs)
    print(f'Your prediction should be {prediction[0][0]}')


if __name__ == '__main__':
    #x=np.array([[1,1],[0,0],[0,1],[1,0],[0,0],[1,0],[1,0],[1,1],[1,1],[0,0]])
    #y=np.array([1,0,1,1,0,1,1,1,1,0],ndmin=2)
    #train(x,y)

    inputs, outputs = generate_dataset(500)

    print('\nThis is the neural network input:')
    print(inputs)

    print('\nThis is the real output:')
    print(outputs)

    inputs = np.array(inputs).T
    outputs = np.array(outputs,ndmin=2)
    input('\nPress enter to begin the training: ')
    train(inputs, outputs)