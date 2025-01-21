import torch
import torch.nn as nn


class LinearAgent(torch.nn.Module):
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_layer_dims: list = [32, 64, 32],
                 p_dropout: float = 0.2):
        
        super(LinearAgent, self).__init__()
        
        self.fci = nn.Linear(input_dim, hidden_layer_dims[0])
        
        self.hidden_layer_dims = hidden_layer_dims
        
        for i in range(len(hidden_layer_dims) - 1):
            setattr(self, f'fc{i+2}', 
            torch.nn.Sequential(nn.Linear(hidden_layer_dims[i], hidden_layer_dims[i+1]), 
                                nn.ReLU(),
                                nn.Dropout(p_dropout),
                                nn.LayerNorm(hidden_layer_dims[i+1])))
        
        self.fco = nn.Linear(hidden_layer_dims[0], output_dim)
        
    def forward(self, x):
        
        x = torch.relu(self.fci(x))
        
        for i in range(len(self.hidden_layer_dims) - 1):
            x = getattr(self, f'fc{i+2}')(x)
        
        x = self.fco(x)
        
        return x
        

if __name__ == "__main__":
    
    model = LinearAgent(4, 2, [32, 64, 32])
    
    print(model)
    print(model(torch.rand(1, 4)))
        
        