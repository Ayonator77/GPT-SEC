import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F


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
    

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embeded_size, num_heads, hidden_size, num_layers, num_classes, max_length, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embeded_size)
        self.positional_encoding = PositionalEncoding(embeded_size, max_length, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embeded_size, nhead=num_heads,dim_feedforward=hidden_size, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(embeded_size, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        position_encoding = self.positional_encoding(embedded)
        transformer_output = self.transformer_encoder(position_encoding)
        mean_pooled = torch.mean(transformer_output, dim=1)
        logits = self.fc(mean_pooled)
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.pe = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state and cell state
        # h0 = torch.zeros(1, time_series_data.size(0), self.hidden_dim)
        # c0 = torch.zeros(1, time_series_data.size(0), self.hidden_dim)
        # Forward pass through LSTM
        lstm_out, _ = self.lstm(x)
        # Apply the output layer
        output = self.fc(lstm_out[:, -1, :])
        print(output.shape)  # Taking the output of the last time step
        return output


import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridModel(nn.Module):
    def __init__(self, transformer_model, lstm_model):
        super(HybridModel, self).__init__()
        self.transformer_model = transformer_model
        self.lstm_model = lstm_model
        self.fc = nn.Linear(transformer_model.fc.in_features + lstm_model.fc.out_features, 3)  # Assuming 3 classes

    def forward(self, transformer_input, lstm_input):
        transformer_output = self.transformer_model(transformer_input)
        lstm_output = self.lstm_model(lstm_input)  # Assuming LSTM returns only output
        print("size: ",lstm_output.size())
        # Reshape LSTM output to match the shape of the transformer output
        transformer_batch_size = transformer_output.size(0)
        transformer_seq_length = transformer_output.size(1)
        lstm_batch_size = transformer_batch_size
        # Reshape LSTM output to match Transformer output dimensions
        lstm_batch_size = lstm_output.size(0)  # Get the batch size from the LSTM output
        lstm_seq_length = 1  # Assuming LSTM output sequence length is 1
        lstm_output = lstm_output.view(lstm_batch_size, 1, lstm_seq_length, -1)  # Adjust -1 based on the LSTM output size

         # Adjust -1 based on the LSTM output size

        # Concatenate the outputs
        combined_output = torch.cat((transformer_output, lstm_output), dim=2)

        # Pass through fully connected layer
        output = self.fc(combined_output)

        # Apply softmax activation
        output = F.softmax(output, dim=2)

        return output



# print("Transformer output: ", transformer_output, "Size: ", transformer_output.shape)
# print("LSTM Output: ", lstm_output, "size: ", lstm_output.shape)

# class HybridModel(nn.Module):
#     def __init__(self, transformer_model, lstm_model, num_classes):
#         super(HybridModel, self).__init__()
#         self.transformer_model = transformer_model
#         self.lstm_model = lstm_model

#         # Extract hidden sizes from models
#         transformer_hidden_size = transformer_model.fc.out_features
#         lstm_hidden_size = lstm_model.hidden_size

#         # Define linear layer
#         self.fc = nn.Linear(transformer_hidden_size + lstm_hidden_size, num_classes)

#     def forward(self, transformer_inputs, lstm_inputs):
#         # Forward pass for transformer model
#         transformer_outputs = self.transformer_model(transformer_inputs)
#         transformer_outputs = transformer_outputs.permute(1, 0, 2)  # (batch_size, seq_len, hidden_size)
        
#         # Forward pass for LSTM model
#         lstm_outputs = self.lstm_model(lstm_inputs)

#         # Pad or truncate LSTM outputs to match transformer outputs along the sequence length dimension
#         max_seq_len = transformer_outputs.size(1)
#         lstm_seq_len = lstm_outputs.size(1)
#         if lstm_seq_len < max_seq_len:
#             # Pad LSTM outputs along the sequence length dimension
#             padding = torch.zeros((lstm_outputs.size(0), max_seq_len - lstm_seq_len, lstm_outputs.size(-1)))
#             lstm_outputs = lstm_outputs.unsqueeze(1)
#             lstm_outputs = torch.cat((lstm_outputs, padding), dim=1)
#         elif lstm_seq_len > max_seq_len:
#             # Truncate LSTM outputs along the sequence length dimension
#             lstm_outputs = lstm_outputs[:, :max_seq_len, :]
        
#         # Concatenate Transformer and LSTM outputs
#         combined_outputs = torch.cat((transformer_outputs.unsqueeze(2), lstm_outputs.unsqueeze(2)), dim=2)
#         print("RAN COMBINED OUTPUTS")

#         # Apply linear layer
#         logits = self.fc(combined_outputs)
#         probs = F.softmax(logits, dim=2)  # Apply softmax along the last dimension

#         # Format output string
#         output_str = "buy:{:.2f}, sell:{:.2f}, neutral:{:.2f}".format(probs[0, 0], probs[0, 1], probs[0, 2])

#         return output_str



# class HybridModel(nn.Module):
#     def __init__(self, text_input_dim, time_series_input_dim, hidden_dim, output_dim):
#         self.text_model  = text_model(text_input_dim, hidden_dim)
#         self.time_series_model = LSTMModel(time_series_input_dim, hidden_dim, hidden_dim)
#         self.output_layer = nn.Linear(hidden_dim * 2, output_dim)

#     def forward(self, text_data, time_series_data):
#         text_output = self.text_model(text_data)
#         time_series_output = self.time_series_model(time_series_data)
#         combined_output = torch.cat((text_output, time_series_output), dim=1)
#         output = self.output_layer(combined_output)
#         return output


def train_model(model, train_loader, criterion, optimizer, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in train_loader:
            inputs = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backwards()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        epoch_loss  = running_loss/len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')


class CustomTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, num_heads, ff_hidden_size, dropout) -> None:
        super().__init__()
        super(CustomTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_size,
                nhead=num_heads,
                dim_feedforward=ff_hidden_size,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        embedded = self.embedding(x)
        # Permute to (seq_len, batch_size, embed_size) as Transformer expects
        embedded = embedded.permute(1, 0, 2)
        # Transformer Encoder layers
        for layer in self.transformer_layers:
            embedded = layer(embedded)
        # Return the last layer's output
        return embedded[-1]


class TextTransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, num_heads, hidden_size, dropout):
        super(TextTransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder_layer = nn.TransformerEncoderLayer(embed_size, num_heads, hidden_size, dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)
        
    def forward(self, src):
        #print("Input Shape: ", src.shape)
        src = self.embedding(src)
        src = src.permute(1, 0, 2)  # (seq_len, batch_size, embed_size)
        output = self.transformer_encoder(src)
        output = output.permute(1, 0, 2)  # (batch_size, seq_len, embed_size)
        #print("Input shape after permute:", src.shape)
        output = self.fc(output)
        print("Output shape: Transformer: ", output.shape)
        return output
    


class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(CustomLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Add a dummy sequence dimension
        x = x.unsqueeze(1)
        
        batch_size = x.size(0)  # Get the batch size
        
        # Initialize the hidden state and cell state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward pass through the LSTM layer
        out, _ = self.lstm(x, (h0, c0))
        
        # Pass the output of the last time step through the fully connected layer
        out = self.fc(out[:, -1, :])
        print("Output shape LSTM: ",out.shape)
        return out
    
# class CustomLSTM(nn.Module):
#     def __init__(self, input_size, lstm_hidden_size, num_layers, num_classes, transformer_hidden_size):
#         super(CustomLSTM, self).__init__()
#         self.hidden_size = lstm_hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, lstm_hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(lstm_hidden_size, num_classes)
#         self.projection = nn.Linear(lstm_hidden_size, transformer_hidden_size)

#     def forward(self, x):
#         # Add a dummy sequence dimension
#         x = x.unsqueeze(1)
        
#         batch_size = x.size(0)  # Get the batch size
        
#         # Initialize the hidden state and cell state
#         h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
#         c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        
#         # Forward pass through the LSTM layer
#         out, _ = self.lstm(x, (h0, c0))
        
#         # Pass the output of the last time step through the fully connected layer
#         out = self.fc(out[:, -1, :])

#         # Project LSTM outputs to match transformer hidden size
#         out = self.projection(out)
#         print("Output shape: ", out.shape)
#         return out
if __name__ =="__main__":
    print("ran")