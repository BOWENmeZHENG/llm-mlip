
import torch.nn as nn
import torch.nn.functional as F

class NERBERTModel(nn.Module):
    def __init__(self, encoder_model, output_size):
        super().__init__()
        self.bert = encoder_model
        self.linear = nn.Linear(768, output_size)
        
    def forward(self, token, attention_mask):
        encoder_output= self.bert(token, attention_mask)
        linear_output = self.linear(encoder_output.last_hidden_state)
        return linear_output