from sec_query import SEC_QUERY
import sec_query
import pandas as pd
from transformers import BertTokenizer
import torch
from models import CustomLSTM, TextTransformerModel, HybridModel
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.optim as optim
import json
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
import re
from nltk.corpus import stopwords
from data_pipeline import preprocess_text
import nltk
import numpy as np
from transformers import AutoTokenizer

class TextDataset_dict(Dataset):
    def __init__(self, text_dict, tokenizer, max_length) -> None:
        self.text_dict = text_dict
        self.tokenizer= tokenizer
        self.max_length = max_length
        self.ticker_list = list(text_dict.keys())
    
    def __len__(self):
        return len(self.ticker_list)
    
    def __getitem__(self, idx):
        ticker = self.ticker_list[idx]
        summaries = self.text_dict[ticker]
        encoded_summaries = self.tokenizer(summaries, padding="max_length",truncation=True, max_length=self.max_length, return_tensors="pt", add_special_tokens=True)
        return ticker, encoded_summaries

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = inputs["input_ids"].squeeze()  # Squeeze extra batch dimension
        if len(input_ids.shape) == 1:  # If input has only one dimension
            input_ids = input_ids.unsqueeze(1)
        #print("Input shape ",input_ids.shape)
        return input_ids

def train_transformer():
    tokenizer_name = "bert-base-uncased"
    max_length = 128
    vocab_size = 30522  # Example vocab size for BERT tokenizer

    embed_size = 256
    num_layers = 4
    num_heads = 8
    hidden_size = 512
    dropout = 0.1

    text_dict = load_text_dict()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    datasets = {ticker: TextDataset(summaries, tokenizer, max_length) for ticker, summaries in text_dict.items()}
    data_loaders = {ticker: DataLoader(datasets[ticker], batch_size=8, shuffle=True) for ticker in text_dict.keys()}

    model = TextTransformerModel(vocab_size, embed_size, num_layers, num_heads, hidden_size, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 10
    for epoch in range(num_epochs):
        for ticker, data_loader in data_loaders.items():
            if ticker != 'CVX':
                model.train()
                total_loss = 0.0
                for batch in data_loader:
                    optimizer.zero_grad()
                    inputs = batch.squeeze()
                    targets = batch.squeeze()
                    outputs = model(inputs)
                    loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                print(f"Epoch {epoch+1}, Ticker: {ticker}, Loss: {total_loss / len(data_loader)}")
    return model, data_loaders

def load_stock_data(main_path='Stock Dataset'):
    ds = {}
    ticker_list = os.listdir(main_path)
    for ticker in ticker_list:
        stock_info = []
        dir_string = os.path.join(main_path, ticker)
        dir = os.listdir(dir_string)
        for df_info in dir:
            df_dir = os.path.join(dir_string, df_info)
            #print(df_dir)
            try:
                df = pd.read_csv(df_dir)
                time_frame = 5
                df['volatility'] = df['close'].pct_change(time_frame).rolling(time_frame).std()
                buy_threshold =0.5
                sell_threshold =0.5
                df['label'] = np.where(df['close'].shift(-1) > df['close'] + buy_threshold * df['volatility'], 'buy', np.where(df['close'].shift(-1) < df['close'] - sell_threshold * df['volatility'], 'sell', 'neutral'))
                # df.dropna(inplace=True)
                stock_info.append(df)
            except:
                print("Exception: Empty Dataframe")
        if stock_info:
            ds[ticker] = pd.concat(stock_info, ignore_index=True)
    return ds

def test_hybrid():
    tokenizer_name = "bert-base-uncased"
    max_length = 128
    vocab_size = 30522  # Example vocab size for BERT tokenizer

    embed_size = 256
    num_layers = 4
    num_heads = 8
    hidden_size_transfomer = 512
    dropout = 0.1

    text_dict = load_text_dict()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    datasets = {ticker: TextDataset(summaries, tokenizer, max_length) for ticker, summaries in text_dict.items()}
    transformer_data_loader = {ticker: DataLoader(datasets[ticker], batch_size=8, shuffle=True) for ticker in text_dict.keys()}

    transformer_model = TextTransformerModel(vocab_size, embed_size, num_layers, num_heads, hidden_size_transfomer, dropout)

    #*****************LSTM****************************************

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    combined_data = load_lstm_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Split features and labels
    X = combined_data[['open', 'high', 'low', 'close', 'volume', 'vwap']].values
    y = combined_data['label'].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    scaler = MinMaxScaler(feature_range=(0,1))
    X = scaler.fit_transform(X)
    
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y_encoded, dtype=torch.int64)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train = X_train
    y_train = y_train
    X_test = X_test
    y_test = y_test


    # Hyperparameters
    hidden_size = 50
    num_layers = 2
    num_classes = 3  # Three classes for 'buy', 'sell', 'neutral'
    num_epochs = 50
    batch_size = 72
    learning_rate = 0.001

    train_dataset = TensorDataset(X_train, y_train)
    lstm_train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)



    input_size = X_train.shape[1]
    lstm_model = CustomLSTM(input_size, hidden_size, num_layers, num_classes)

    transformer_input_size = 30522  # Vocab size
    lstm_input_size = 3  # You need to determine the output size of your LSTM model
    num_classes = 3 
    hybrid_model = HybridModel(transformer_model, lstm_model,transformer_input_size,lstm_input_size ,num_classes)
    #print(hybrid_model)

    for epoch in range(num_epochs):

        for ticker, transformer_loader in transformer_data_loader.items():
            for trans_batch, lstm_batch in zip(transformer_loader, lstm_train_loader):
                trans_inputs = trans_batch.squeeze()
                lstm_inputs, lstm_labels = lstm_batch
                lstm_inputs = lstm_inputs

                outputs = hybrid_model(trans_inputs, lstm_inputs)
                #print("Hybrid outputs: ", outputs, "epoch: ", epoch)

def foobar():
    tokenizer_name = "bert-base-uncased"
    max_length = 128
    vocab_size = 30522  # Example vocab size for BERT tokenizer

    embed_size = 256
    num_layers = 4
    num_heads = 8
    hidden_size_transfomer = 512
    dropout = 0.1

    text_dict = load_text_dict()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    datasets = {ticker: TextDataset(summaries, tokenizer, max_length) for ticker, summaries in text_dict.items()}
    transformer_data_loader = {ticker: DataLoader(datasets[ticker], batch_size=8, shuffle=True) for ticker in text_dict.keys()}

    transformer_model = TextTransformerModel(vocab_size, embed_size, num_layers, num_heads, hidden_size_transfomer, dropout)

    #*****************LSTM****************************************

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    combined_data = load_lstm_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Split features and labels
    X = combined_data[['open', 'high', 'low', 'close', 'volume', 'vwap']].values
    y = combined_data['label'].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    scaler = MinMaxScaler(feature_range=(0,1))
    X = scaler.fit_transform(X)
    
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y_encoded, dtype=torch.int64)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train = X_train
    y_train = y_train
    X_test = X_test
    y_test = y_test


    # Hyperparameters
    hidden_size = 50 
    num_layers = 2
    num_classes = 3  # Three classes for 'buy', 'sell', 'neutral'
    num_epochs = 50
    batch_size = 72
    learning_rate = 0.001

    train_dataset = TensorDataset(X_train, y_train)
    lstm_train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    input_size = X_train.shape[1]
    lstm_model = CustomLSTM(input_size, hidden_size, num_layers, num_classes)

    hybrid_model = HybridModel(transformer_model, lstm_model, 3)
    for epochs in range(num_epochs):
        for ticker, transformer_loader in transformer_data_loader.items():
            for trans_batch, lstm_batch in zip(transformer_loader, lstm_train_loader):
                trans_inputs = trans_batch.squeeze()
                lstm_inputs, lstm_labels = lstm_batch

                outputs = hybrid_model(trans_inputs, lstm_inputs)
                print(outputs)



def load_text_dict(main_path ="Text Dataset"):
    data_structure = {}
    ticker_list = os.listdir(main_path)
    for ticker in ticker_list:
        text_info = []
        dir_string = os.path.join(main_path, ticker) 
        current_dir = os.listdir(dir_string)
        for date in current_dir:
            date_dir = os.path.join(dir_string, date)
            with open(date_dir, "r") as file:
                text = file.read().replace('\n', '')
                text_with_date = f"{date[:-4]} {text}"
                text_info.append(text_with_date)
        data_structure[ticker] = text_info
    return data_structure

"""
{   AAPL: [sum0, sum1 .....sum9]
    ABBV: ........
}
"""

def ready_text(data_structure:dict):
    for key, sum_list in data_structure.items():
        pre_sum = [preprocess_text(summ) for summ in sum_list]
    return pre_sum


def load_lstm_data():
    stock_data =  load_stock_data()
    return pd.concat(stock_data.values(), ignore_index=True)

def train_lstm():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    combined_data = load_lstm_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Split features and labels
    X = combined_data[['open', 'high', 'low', 'close', 'volume', 'vwap']].values
    y = combined_data['label'].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    scaler = MinMaxScaler(feature_range=(0,1))
    X = scaler.fit_transform(X)
    
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y_encoded, dtype=torch.int64)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)


    # Hyperparameters
    hidden_size = 50
    num_layers = 2
    num_classes = 3  # Three classes for 'buy', 'sell', 'neutral'
    num_epochs = 50
    batch_size = 72
    learning_rate = 0.001

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)



    input_size = X_train.shape[1]
    model = CustomLSTM(input_size, hidden_size, num_layers, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            print(device)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    return model, train_loader

def train_hybrid():
    combined_data = load_lstm_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Split features and labels
    X = combined_data[['open', 'high', 'low', 'close', 'volume', 'vwap']].values
    y = combined_data['label'].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    scaler = MinMaxScaler(feature_range=(0,1))
    X = scaler.fit_transform(X)
    
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y_encoded, dtype=torch.int64)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)


    # Hyperparameters
    hidden_size = 50
    num_layers = 2
    num_classes = 3  # Three classes for 'buy', 'sell', 'neutral'
    num_epochs = 50
    batch_size = 72
    learning_rate = 0.001

    train_dataset = TensorDataset(X_train, y_train)
    train_loader_lstm = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)



    input_size = X_train.shape[1]
    model_lstm = CustomLSTM(input_size, hidden_size, num_layers, num_classes).to(device)

    tokenizer_name = "bert-base-uncased"
    max_length = 128
    vocab_size = 30522  # Example vocab size for BERT tokenizer

    embed_size = 256
    num_layers = 4
    num_heads = 8
    hidden_size = 512
    dropout = 0.1

    text_dict = load_text_dict()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    datasets = {ticker: TextDataset(summaries, tokenizer, max_length) for ticker, summaries in text_dict.items()}
    data_loader_transformer = {ticker: DataLoader(datasets[ticker], batch_size=8, shuffle=True) for ticker in text_dict.keys()}

    model_transformer = TextTransformerModel(vocab_size, embed_size, num_layers, num_heads, hidden_size, dropout)
    print(model_lstm, '\n', model_transformer)

    #*****************Hybrid model#*****************
    # model_transformer = train_transformer()
    # model_lstm = train_lstm()
    transformer_input_size = 30522  # Vocab size
    lstm_input_size = 3  # You need to determine the output size of your LSTM model
    num_classes = 3 
    hybrid_model = HybridModel(model_transformer, model_lstm, transformer_input_size, lstm_input_size, num_classes)
    hybrid_model = hybrid_model.to(device)
    optimizer = torch.optim.Adam(hybrid_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()


    for epochs in range(num_epochs):
        hybrid_model.train()
        total_loss = 0.0
        for transformer_ticker, transformer_data_loader in data_loader_transformer.items():
            lstm_data_loader = train_loader_lstm
            for transformer_batch, lstm_batch in zip(transformer_data_loader, lstm_data_loader):
                transformer_inputs = transformer_batch.to(device)
                lstm_inputs = lstm_batch[0].to(device)

                optimizer.zero_grad()
                transformer_outputs = model_transformer(transformer_inputs)
                lstm_outputs = model_lstm(lstm_inputs)

                combined_outputs = torch.cat((transformer_outputs, lstm_outputs), dim=1)
                outputs = hybrid_model(combined_outputs)

                loss = torch.norm(outputs, p=2)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # transformer_inputs = torch.tensor(transformer_inputs).to(device)
            # lstm_inputs = torch.tensor(lstm_inputs).to(device)

            # optimizer.zero_grad()
            # transformer_outputs = model_transformer(transformer_inputs)
            # lstm_outputs = model_lstm(lstm_inputs)
            # if transformer_outputs.size(0) != lstm_outputs.size(0):
            #     raise ValueError("Batch sizes of transformer_outputs and lstm_outputs do not match.")
            # combined_output = torch.cat((transformer_outputs, lstm_outputs), dim=1)
            # outputs = hybrid_model(combined_output)
            # loss = torch.norm(outputs, p=2)

            # loss.backward()
            # optimizer.step()
            # #loss = criterion(outputs)
        print(f'Epoch [{epochs+1}/{num_epochs}], Loss: {total_loss / len(data_loader_transformer)}')
    print(hybrid_model)



def hybrid_test():
    pass



if __name__ == "__main__":
    #train_lstm()
    #train_transformer()

    foobar()