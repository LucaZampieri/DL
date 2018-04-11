# from the lecture slides: 
import math


def reset_parameters(self):
    stdv = math.sqrt(3) / math.sqrt(self.weight.size(1)) # sqrt(3) to correct the mistake? :)
    self.weight.data.uniform_(-stdv, stdv)
    if self.bias is not None:
        self.bias.data.uniform_(-stdv,stdv)
        
def xavier_normal(tensor, gain=1):
    if isinstance(tensor, Variable):
        xavier_normal(tensor.data, gain=gain)
        return tensor
    fan_in, fan_out = _calculate_fan_in_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out)) # fan_in = self.weight.size(1), fan_out = fan_in - 1 ??
    

    