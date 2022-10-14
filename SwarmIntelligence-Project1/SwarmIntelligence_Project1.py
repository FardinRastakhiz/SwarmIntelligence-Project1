import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod
from abc import ABC

class SwarmIntelligence(ABC):
    def __init__(self, target_function):
        self.function = target_function

class PSO(SwarmIntelligence):
    pass

class SimplePSO(PSO):
    
    def __init__(self, target_function):
        super().__init__(target_function)
        print(self.function)


def main():
    spso = SimplePSO("abcd")
    

if __name__ == "__main__":
    main()
