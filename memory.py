import itertools
import random

class CircularBuffer:
    
    def __init__(self, capacity: int):
        
        self.capacity = capacity
        self.memory = []
        
    def push(self, transition):
            
        self.memory.append(transition)
        
        if len(self.memory) > self.capacity:
            self.memory.pop(0)
            
    def sample(self, batch_size: int):
        
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        
        return len(self.memory)
    
