import numpy as np
class neuron():
    def __init__(self, num_input):
        self.num_input = num_input
        self.params = [np.random.rand(self.num_input)] # self.w = np.random.rand(self.num_input)
        self.params.append(np.random.rand(1)) # self.b = np.random.rand(1)

    def forward(self, x):
        output = 0
        for i in range(self.num_input):
            output += self.params[i] * x[i] # output += self.w[i] * x[i]

        output += self.params[-1] # output += self.b
        return output


class Loss():
    def __init__(self, X, Y, neuron):
        self.X = X
        self.Y = Y
        self.neuron = neuron

    def value(self, param, idx):
        self.neuron.params[idx] = param
        y_pred = self.neuron.forward(self.X)
        return np.abs(self.Y - y_pred)


class GD():
    def __init__(self, params, lr=0.1, epsilon=1e-6):
        self.epsilon = epsilon
        self.lr = lr
        self.params = params

    def grad(self, x, f):
        return (f(x[0] + self.epsilon, x[1]) - f(x[0], x[1])) / self.epsilon

    def step(self, f):
        for idx, param in enumerate(self.params):
            new_param = param - self.lr * self.grad((param, idx), f)

            self.params[idx] = new_param

my_neuron = neuron(1)
gd = GD(my_neuron.params)
loss = Loss([1], 0, my_neuron)

print(my_neuron.params) 
print(gd.step(loss.value))
print(my_neuron.params) 
print(gd.step(loss.value))
print(my_neuron.params) 
