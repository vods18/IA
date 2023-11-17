# VINICIUS OLIVEIRA DOS SANTOS
# GRR20182592

import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.w) # the dot product of the stored weight vector and the given input

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        return 1 if nn.as_scalar(self.run(x)) >= 0 else -1 # if the dot product is positive or zero, return 1, else return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        while any(self.get_prediction(x) != nn.as_scalar(y) for x, y in dataset.iterate_once(1)):
            for x, y in dataset.iterate_once(1):
                if self.get_prediction(x) != nn.as_scalar(y):
                    self.w.update(x, nn.as_scalar(y))  # weights←weights+direction⋅multiplier

                

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 200
        self.learningRate = -0.05
        self.depths = [1, 20, 30, 20, 1]

        # Weights using list comprehension
        self.Weights = [nn.Parameter(prev_depth, curr_depth) for prev_depth, curr_depth in zip(self.depths[:-1], self.depths[1:])]

        # Bias using list comprehension
        self.bias = [nn.Parameter(1, depth) for depth in self.depths[1:]]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        x = nn.Linear(x, self.Weights[0])
        x = nn.AddBias(x, self.bias[0])

        for i in range(1, len(self.depths) - 1):
            x = nn.ReLU(x)
            x = nn.Linear(x, self.Weights[i])
            x = nn.AddBias(x, self.bias[i])

        return x



    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted = self.run(x)
        loss = nn.SquareLoss(predicted, y)
        return loss


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        finished = False
        parameters = [bias for bias in self.bias] + [weight for weight in self.Weights]
        loss_threshold = 0.001

        while not finished:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                gradient = nn.gradients(loss, parameters)
                loss = nn.as_scalar(loss)

                if loss <= loss_threshold:
                    finished = True

                for i in range(len(parameters)):
                    parameters[i].update(gradient[i], self.learningRate)


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 100
        self.learningRate = -0.4
        self.depths = [784, 200, 200, 200, 10]
        self.Weights = []
        for i in range(1, len(self.depths)):
            self.Weights.append(nn.Parameter(self.depths[i - 1], self.depths[i]))
        self.bias = [nn.Parameter(1, depth) for depth in self.depths]
        self.bias.remove(self.bias[0])

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        for i in range(len(self.depths) - 1):
            xw = nn.Linear(x, self.Weights[i])
            withBias = nn.AddBias(xw, self.bias[i])
            x = withBias
            if i != len(self.depths) - 2:
                x = nn.ReLU(withBias)
        return x

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted = self.run(x)
        #print(nn.SquareLoss(predicted))
        return nn.SoftmaxLoss(predicted, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        import math
        finished = False
        parameters = []
        for bias in self.bias:
            parameters.append(bias)
        for weight in self.Weights:
            parameters.append(weight)
        accuracy = 0
        while accuracy < 97:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                Gradient = nn.gradients(loss, parameters)
                for i in range(len(parameters)):
                    parameters[i].update(Gradient[i], self.learningRate)
            accuracy = dataset.get_validation_accuracy()


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 200
        self.learningRate = -0.2
        self.HiddenLayerSize = 500

        self.InitialDepths = [self.num_chars, 200, self.HiddenLayerSize]
        self.InitialWeights = []
        for i in range(1, len(self.InitialDepths)):
            self.InitialWeights.append(nn.Parameter(self.InitialDepths[i - 1], self.InitialDepths[i]))
        self.initialBias = [nn.Parameter(1, depth) for depth in self.InitialDepths]
        self.initialBias.remove(self.initialBias[0])


        self.HiddenWeights = []
        for i in range(1, len(self.InitialDepths)):
            self.HiddenWeights.append(nn.Parameter(self.HiddenLayerSize, self.InitialDepths[i]))
        self.hiddenBias = [nn.Parameter(1, depth) for depth in self.InitialDepths]
        self.hiddenBias.remove(self.hiddenBias[0])
        
        self.finalDepths = [self.HiddenLayerSize, 300, 5]
        self.finalWeights = []
        for i in range(1, len(self.finalDepths)):
            self.finalWeights.append(nn.Parameter(self.finalDepths[i - 1], self.finalDepths[i]))
        self.finalBias = [nn.Parameter(1, depth) for depth in self.finalDepths]
        self.finalBias.remove(self.finalBias[0])

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        #initial
        x = xs[0]
        for i in range(len(self.InitialDepths) - 1):
            xw = nn.Linear(x, self.InitialWeights[i])
            withBias = nn.AddBias(xw, self.initialBias[i])
            x = nn.ReLU(withBias)
        #recurrent
        h = x
        for i in range(1, len(xs)):
            x = xs[i]
            for j in range(len(self.InitialDepths) - 1):
                xw = nn.Add(nn.Linear(x, self.InitialWeights[j]), nn.Linear(h, self.HiddenWeights[j]))
                withBias = nn.AddBias(xw, self.hiddenBias[j])
                x = nn.ReLU(withBias)
            h = nn.ReLU(x)
        #final
        x = h
        for i in range(len(self.finalDepths) - 1):
            xw = nn.Linear(x, self.finalWeights[i])
            withBias = nn.AddBias(xw, self.finalBias[i])
            x = withBias
            if i != len(self.finalDepths) - 2:
                x = nn.ReLU(withBias)

        return x

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted = self.run(xs)
        return nn.SoftmaxLoss(predicted, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        import math
        parameters = []
        for bias in self.initialBias:
            parameters.append(bias)
        for bias in self.hiddenBias:
            parameters.append(bias) 
        for bias in self.finalBias:
            parameters.append(bias) 

        for weight in self.InitialWeights:
            parameters.append(weight)
        for weight in self.HiddenWeights:
            parameters.append(weight)
        for weight in self.finalWeights:
            parameters.append(weight)    
        
        accuracy = 0
        while accuracy < 89:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                Gradient = nn.gradients(loss, parameters)
                for i in range(len(parameters)):
                    parameters[i].update(Gradient[i], self.learningRate)
            accuracy = dataset.get_validation_accuracy()
