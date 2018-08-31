import torch
import torch.nn as nn
import torch.nn.functional as F


1. if use F.dropout() inside a nn.Module class, need to pass self.training to differentiate training vs testing
If use class nn.Dropout() inside nn.Modele, then self.training is automatically passed in.

example:

Class DropoutFC(nn.Module):
  def __init__(self):
      super(DropoutFC, self).__init__()
      self.fc = nn.Linear(100,20)
      self.dropout = nn.Dropout(p=0.5) #no self.training needed, implicitly passed 

  def forward(self, input):
      out = self.fc(input)
      out = self.dropout(out)
      return out


class DropoutFC(nn.Module):
   def __init__(self):
       super(DropoutFC, self).__init__()
       self.fc = nn.Linear(100,20)

   def forward(self, input):
       out = self.fc(input)
       out = F.dropout(out, p=0.5, training=self.training) #explicit passing self.training is needed.
       return out



2. tensor.max(), if not used globally like below, return value are two tensors (max_value, argmax_index)
>>>a = torch.randn(2, 2)
>>>a.max()



