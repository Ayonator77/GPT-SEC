import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import os
from tester import load_text_dict, load_stock_data
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import pickle
from transformers import AutoTokenizer, BertTokenizer
from torchtext.data.utils import get_tokenizer
import matplotlib.pyplot as plt

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, num_layers, num_classes, max_seq_length):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim,
            dropout=0.1
        )
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, src, tgt):  # Update forward method to include target sequence
        src_embedded = self.embedding(src)
        tgt_embedded = self.embedding(tgt)  # Placeholder for target sequence
        output = self.transformer(src_embedded, tgt_embedded)  # Pass both source and target sequences
        output = self.fc(output.mean(dim=1))  # Global average pooling
        print("Transformer output shape: ", output.shape)
        return output


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size) 

    def forward(self, x):
        print("Lstm in shape: ",x.shape)
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        print("Lstm output shape:  ", out.shape)
        return out

class HybridModel(nn.Module):
    def __init__(self, transformer_model, lstm_model, num_classes):
        super(HybridModel, self).__init__()
        self.transformer_model = transformer_model
        self.lstm_model = lstm_model
        self.fc = nn.Linear(transformer_model.fc.out_features + lstm_model.fc.out_features, num_classes)

    def forward(self, transformer_input_ids, transformer_attention_mask, lstm_input):
        transformer_output = self.transformer_model(transformer_input_ids, transformer_attention_mask)
        lstm_output = self.lstm_model(lstm_input)
        
        # Concatenate outputs
        combined_output = torch.cat((transformer_output, lstm_output), dim=1)

        # Final classification layer
        output = self.fc(F.relu(combined_output))
        return output


def train_lstm_model(model, train_data, val_data, num_epochs=10, learning_rate=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for features, labels in train_data:
            if torch.isnan(features).any():
                features[torch.isnan(features)] = 0.0
            features = features.unsqueeze(0)
            #labels = torch.argmax(labels, dim=1)
            optimizer.zero_grad()
            outputs = model(features)
            #outputs = outputs.squeeze(0)
            outputs[torch.isnan(outputs)] = 0.0
            outputs = outputs.unsqueeze(0).repeat(labels.size(0), 1, 1)
            outputs = outputs.squeeze(1)
            outputs = torch.sigmoid(outputs)
            #print("Output shape:", outputs.shape, outputs) 
            # print("Labels shape:", labels.shape, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print("Train Loss: ", train_loss,' epoch: ', epoch, '\n')
    return model




# def load_stock_data(main_path='Stock Dataset'):
#     ds = {}
#     ticker_list = os.listdir(main_path)
#     for ticker in ticker_list:
#         stock_info = []
#         dir_string = os.path.join(main_path, ticker)
#         dir = os.listdir(dir_string)
#         for df_info in dir:
#             df_dir = os.path.join(dir_string, df_info)
#             #print(df_dir)
#             try:
#                 df = pd.read_csv(df_dir)
#                 time_frame = 5
#                 df['volatility'] = df['close'].pct_change(time_frame).rolling(time_frame).std()
#                 df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
#                 buy_threshold =0.5
#                 sell_threshold =0.5
#                 #df['label'] = np.where(df['close'].shift(-1) > df['close'] + buy_threshold * df['volatility'], 'buy', np.where(df['close'].shift(-1) < df['close'] - sell_threshold * df['volatility'], 'sell', 'neutral'))
#                 # df.dropna(inplace=True)
#                 stock_info.append(df)
#             except:
#                 print("Exception: Empty Dataframe")
#         if stock_info:
#             ds[ticker] = pd.concat(stock_info, ignore_index=True)
#     return ds



def generate_labels(data, volatility_window=5, buy_threshold=0.5, sell_threshold=0.5):
    labels = []
    data['date'] = data['timestamp'].dt.date

    for i, row in data.iterrows():
        date = row['date']
        group = data[data['date'] == date]
        price_today = row['close']
        price_tomorrow = group.iloc[-1]['close']
        price_change = (price_tomorrow - price_today) / price_today * 100
        volatility = group['volatility'].mean()

        if price_change > volatility * buy_threshold:
            labels.append([1, 0, 0])  # Buy
        elif price_change < -volatility * sell_threshold:
            labels.append([0, 1, 0])  # Sell
        else:
            labels.append([0, 0, 1])  # Neutral

    data['labels'] = labels
    return data

def timestamp_to_features(timestamp):
    if isinstance(timestamp, str):
        # Parse the timestamp string into a datetime object
        date_time = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%f')
    else:
        # Convert numpy.datetime64 to a string and then parse
        timestamp_str = str(timestamp)
        date_time = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S.%f')
    hour = date_time.hour
    minute = date_time.minute
    second =  date_time.second

    return np.array([hour, minute, second])

def prepare_lstm_data(data_dict, test_size=0.2, validation_size=0.1):
    train_data = []
    test_data = []
    val_data = []
    #i = 0
    for ticker, data in data_dict.items():
        X_timestamp = np.array(data['timestamp'])
        #print("X timestamp", X_timestamp)
        X_features = data[['close', 'volatility']].values # features 
        X_features = np.nan_to_num(X_features, nan=0.0)
        y = np.array(data['labels'].tolist()) #labels

        X_time_features = np.array([timestamp_to_features(str(ts)[:-3]) for ts in X_timestamp])
        X = np.concatenate((X_time_features, X_features), axis=1)

        test_val_size = test_size + validation_size
        test_val_split = test_size / test_val_size

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_val_size,shuffle=False,random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train ,test_size=test_val_split,shuffle=False,random_state=42)
        train_data.append((torch.tensor(X_train).float(), torch.tensor(y_train).float()))
        test_data.append((torch.tensor(X_test).float(), torch.tensor(y_test).float()))
        val_data.append((torch.tensor(X_val).float(), torch.tensor(y_val).float()))
    return train_data, test_data, val_data

def lstm_train_loader(train_data, batch_size=64):
    all_features = torch.cat([features for features, _ in train_data])
    all_labels = torch.cat([labels for _, labels in train_data ])
    dataset = TensorDataset(all_features, all_labels)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader


def construct_dataframe(data_dict:dict):
    for ticker, data in data_dict.items():
        data_dict[ticker] = generate_labels(data)
        print(ticker, ': ',data_dict[ticker])
    
    file_path = 'preprocessed_stock_data.pkl'
    with open(file_path, 'wb') as f:
        pickle.dump(data_dict, f)
    print("Dict saved")


def lstm_main():
    ds_stock = load_stock_binary()
    input_size = 5
    hidden_size = 128
    num_layers = 2
    output_size = 3
    lstm_model = LSTMModel(input_size,hidden_size, num_layers, output_size)
    print(lstm_model)
    train_data, test_data, val_data = prepare_lstm_data(ds_stock)
    print(train_data)
    lstm_model = train_lstm_model(lstm_model, train_data, val_data, 10, 0.001)
    return lstm_model


def load_stock_binary(file_path = 'preprocessed_stock_data.pkl'):
    with open(file_path, 'rb') as f:
        dataframe = pickle.load(f)
    return dataframe


def text_labels():
    ds_stock = load_stock_binary()
    ds = load_text_dict()

    for ticker_, text_data in ds.items():
        if ticker_ in ds_stock.keys():
            stock_data = ds_stock[ticker_]
            for summ in text_data:
                print(stock_data['date'], " summary date\n ", summ[0:10])
                label = stock_data.loc[stock_data['date'] == summ[0:10], 'labels']
                #print(label)
    
def assign_labels_to_text_data():
    time_series_data = load_stock_binary()
    text_data = load_text_dict()
    labeled_text_data = {}
    for ticker, summaries in text_data.items():
        labeled_summaries = []
        if ticker in time_series_data.keys():
            for summary in summaries:
                summary_date_str = summary[0:10]
                summary_date = datetime.strptime(summary_date_str, '%Y-%m-%d')

                closest_date_idx = (time_series_data[ticker]['timestamp'] - summary_date).abs().idxmin()
                closest_date = time_series_data[ticker].loc[closest_date_idx, 'date']

                labels = time_series_data[ticker].loc[closest_date_idx, 'labels']

                labeled_summary = {'summary': summary, 'date': closest_date, 'labels': labels}
                labeled_summaries.append(labeled_summary)

            labeled_text_data[ticker] = labeled_summaries
    return labeled_text_data


def preprocess_text_data(data):
    preprocessed_data = []
    max_seq_length = 0
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    for ticker, summaries in data.items():
        for summary_info in summaries:
            summary = summary_info['summary']
            labels = summary_info['labels']
            date = summary_info['date']

            # Tokenize and convert to token IDs
            encoded_summary = tokenizer.encode(summary, add_special_tokens=True)
            max_seq_length = max(max_seq_length, len(encoded_summary))

            preprocessed_data.append((encoded_summary, labels, date))

    return preprocessed_data, max_seq_length

class TextDataset_(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Sort batch by length for packing sequences
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    tokenized_summaries, labels, dates = zip(*batch)

    # Pad tokenized summaries to the maximum length in the batch
    max_length = max(len(summary) for summary in tokenized_summaries)
    padded_summaries = [summary + [tokenizer.pad_token_id] * (max_length - len(summary)) for summary in tokenized_summaries]

    # Convert tokenized summaries to tensors
    input_ids = torch.tensor(padded_summaries)
    attention_mask = torch.where(input_ids != tokenizer.pad_token_id, 1, 0)

    labels_tensor = torch.tensor(labels)
    return {'input_ids': input_ids, 'attention_mask': attention_mask}, labels_tensor, dates


def train_transformer(model, train_loader, num_epochs=10, learning_rate=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels, dates in train_loader:
            inputs = inputs['input_ids'].to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs, inputs)
            print(outputs.shape)
            _, class_indices = labels.max(dim=1)
            loss = criterion(outputs, class_indices)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print("loss: ", running_loss/len(train_loader))

def transformer_main():
    ds = assign_labels_to_text_data()
    prepro_data, max_seq_length = preprocess_text_data(ds)
    dataset = TextDataset_(prepro_data)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = tokenizer.vocab_size
    embedding_dim = 256
    num_heads = 8
    hidden_dim = 512
    num_layers = 6
    num_classes = 3  # Number of classes for classification
    model = TransformerModel(vocab_size, embedding_dim, num_heads, hidden_dim, num_layers, num_classes, max_seq_length)
    train_data, test_val_data = train_test_split(prepro_data, test_size=0.2, random_state=42)
    print(type(train_data))
    val_data, test_data = train_test_split(test_val_data, test_size=0.5, random_state=42)
    train_loader = DataLoader(TextDataset_(train_data), batch_size=64, shuffle=True, collate_fn=collate_fn)
    
    train_transformer(model, train_loader)



def train_hybrid_model(transformer_loader, lstm_data, hybrid_model, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(hybrid_model.parameters(), lr=0.001)
    epoch_losses = []
    for epoch in range(num_epochs):
        hybrid_model.train()
        total_loss = 0
        count_batches = 0
        for (transformer_batch, lstm_batch) in zip(transformer_loader, lstm_data):
            transformer_inputs, transformer_labels, _ = transformer_batch
            transformer_input_ids, transformer_attention_mask = transformer_inputs['input_ids'], transformer_inputs['attention_mask']
            lstm_inputs, lstm_labels = lstm_batch
            
            lstm_inputs = lstm_inputs.unsqueeze(1)

            optimizer.zero_grad()

            outputs = hybrid_model(transformer_input_ids, transformer_attention_mask, lstm_inputs)
           # print(outputs)

            loss = criterion(outputs, lstm_labels)
            total_loss += loss.item()
            count_batches +=1
            loss.backward()

            optimizer.step()
        average_loss = total_loss/count_batches
        epoch_losses.append(average_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
    print("training complete")
    return epoch_losses

def plot_loss(epoch_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_losses, label='Training Loss')
    plt.title('Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    ds = assign_labels_to_text_data()
    ds_stock = load_stock_binary()

    train_data, test_data, val_data = prepare_lstm_data(ds_stock)
    train_data_loader = lstm_train_loader(train_data)


    prepro_data, max_seq_length = preprocess_text_data(ds)
    train_data_transformer, test_val_data = train_test_split(prepro_data, test_size=0.2, random_state=42)
    dataset = TextDataset_(train_data_transformer)
    train_loader_tranformer = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    #define lstm model
    input_size = 5
    hidden_size = 128
    num_layers = 2
    output_size = 3
    lstm_model = LSTMModel(input_size,hidden_size, num_layers, output_size)


    #define transformer model
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = tokenizer.vocab_size
    embedding_dim = 128
    num_heads = 8
    hidden_dim = 512
    num_layers_t = 6
    num_classes = 3  # Number of classes for classification
    transformer_model = TransformerModel(vocab_size, embedding_dim, num_heads, hidden_dim, num_layers_t, num_classes, max_seq_length)


    #define hybrid model
    hybrid_model = HybridModel(transformer_model, lstm_model, num_classes)

    #train hybrid model
    loss_plot = train_hybrid_model(train_loader_tranformer, train_data_loader, hybrid_model, 50)
    plot_loss(loss_plot)


