import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class multi_headed_attention(nn.Module):
    def __init__(self, d_model, nhead, droput=0.4) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(droput)
        self.linear = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        q_head = self.w_q(q).view(q.size(0), -1, self.nhead, self.d_model//self.nhead).permute(0,2,1,3)
        k_head = self.w_k(k).view(k.size(0), -1, self.nhead, self.d_model // self.nhead).permute(0, 2, 1, 3)




class text_model(nn.Module):
    def __init__(self, input_dim, num_class) -> None:
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=4),
            num_layers=4
        )
        self.output_layer = nn.Linear(input_dim, num_class)
    
    def forward(self, text_data):
        text_data = self.transformer(text_data)
        output = self.output_layer(text_data[:, 0, :])
        return output
    

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, time_series_data):
        # Initialize hidden state and cell state
        h0 = torch.zeros(1, time_series_data.size(0), self.hidden_dim)
        c0 = torch.zeros(1, time_series_data.size(0), self.hidden_dim)
        # Forward pass through LSTM
        lstm_out, _ = self.lstm(time_series_data, (h0, c0))
        # Apply the output layer
        output = self.fc(lstm_out[:, -1, :])  # Taking the output of the last time step
        return output


class HybridModel(nn.Module):
    def __init__(self, text_input_dim, time_series_input_dim, hidden_dim, output_dim):
        self.text_model  = text_model(text_input_dim, hidden_dim)
        self.time_series_model = LSTMModel(time_series_input_dim, hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text_data, time_series_data):
        text_output = self.text_model(text_data)
        time_series_output = self.time_series_model(time_series_data)
        combined_output = torch.cat((text_output, time_series_output), dim=1)
        output = self.output_layer(combined_output)
        return output
    
if __name__ =="__main__":
    print("ran")