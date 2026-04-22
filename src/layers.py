class DenseLayer :
    def __init__(self, units, activation='sigmoid', weights_initializer='default'):
        self.units = units
        self.activation = activation
        self.initializer = weights_initializer
        self.weights = None
        self.bias = None

        self.input = None
        self.z = None
    
    def initialize(self, input_size):
        if  self.initializer == 'heUniform':
            limit = np.sqrt(6/ input_size)
            self.weights = np.random.uniform(-limit, limit, (input_size, self.units))
        else:
            self.weights = np.random.randn(input_size, self.units) * 0.01
        
        self.bias = np.zeros((1, self.units)) 

    def forward(self, input_data):
        self.input = input_data

        self.z = np.dot(self.input, self.weights) + self.bias

        return self.z

