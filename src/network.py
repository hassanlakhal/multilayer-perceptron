import numpy as np

class Model:
    def createNetwork(self, layers_list):
        class NetworkContainer:
            def __init__(self, layers):
                self.layers = layers
        
        
        for i in range(1, len(layers_list)):
            input_dim = layers_list[i-1].units
            print(f"input_dim {input_dim}")
            layers_list[i].initialize(input_dim)
            
        return NetworkContainer(layers_list)