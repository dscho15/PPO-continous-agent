import torch

class EMA:
    def __init__(self, model, decay: float = 0.9, **kwargs):
        """
        Wraps a model with EMA functionality.
        
        Args:
            model (torch.nn.Module): The model to apply EMA to.
            decay (float): The decay rate for EMA, typically between 0.9 and 0.999.
        """
        self.model = model
        self.ema_model = type(model)(**kwargs)  # Create a new model instance of the same type.
        self.ema_model.load_state_dict(model.state_dict())
        self.decay = decay
        self.device = next(model.parameters()).device  # Get the device of the model.
        self.ema_model.to(self.device)
        self.ema_model.eval()

    @torch.no_grad()
    def update(self):
        """
        Updates the EMA model's parameters.
        """
        for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)
        
        for ema_buffer, model_buffer in zip(self.ema_model.buffers(), self.model.buffers()):
            ema_buffer.copy_(model_buffer)

    def apply_shadow(self):
        """
        Copies EMA weights to the original model (optional).
        """
        self.model.load_state_dict(self.ema_model.state_dict())

    def state_dict(self):
        """
        Returns the state dict of the EMA model.
        """
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        """
        Loads a state dict into the EMA model.
        """
        self.ema_model.load_state_dict(state_dict)

    def to(self, device):
        """
        Moves the EMA model to a specified device.
        """
        self.ema_model.to(device)
      
    def forward(self, x):
        """
        Forwards a tensor through the EMA model.
        """
        return self.ema_model(x)