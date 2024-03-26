from data_pipeline import data_pipeline, preprocess
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os
from sec_query import SEC_QUERY
from gpt_sec_main import get_stock_info
from datetime import datetime
from dateutil.relativedelta import relativedelta

def train_transformer(model, train_loader, criterion, optimizer, num_epochs):
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


def train_lstm(model, input_size, hidden_size, num_layers, output_size, data_set):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    dataloader = DataLoader(data_set, batch_size=32, shuffle=True)

    num_epochs = 10
    for epoch in range(num_epochs):
        for batch in dataloader:
            inputs = batch['inputs']
            labels = batch['labels']

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def save_stock_data():
    t_list = os.listdir("Text Dataset")
    #dates = []
    dataframe_list = []
    stock_dict = {}
    for tick in t_list:
        stock_dict[tick] = os.listdir(os.path.join("Text Dataset", tick))
        #dates.append(os.listdir(os.path.join("Text Dataset", tick)))
    stock_dict  = {key: [datetime.strptime(filename[:-4], '%Y-%m-%d') for filename in filenames] for key, filenames in stock_dict.items()}
    for key, dates in stock_dict.items():
        for date in dates:
            #print(date)
            end_date = date +relativedelta(days=7)
            date = date.strftime("%Y-%m-%d")
            end_date = end_date.strftime("%Y-%m-%d")
            #print(date, end_date)
            dataframe_list.append((key, get_stock_info(key, date, end_date)))

    if not os.path.exists("Stock Dataset"):
        os.makedirs("Stock Dataset")
    for key, data_frame in dataframe_list:
        file_index = 0
        if 'timestamp' in data_frame.columns:
            data_frame['timestamp']=data_frame['timestamp'].apply(lambda x: datetime.fromtimestamp(x/1000))
            print(data_frame)
            key_path = os.path.join("Stock Dataset", key)
            os.makedirs(key_path, exist_ok=True)
            final_path = os.path.join(key_path, key+'_'+str(file_index)+'.csv')
            data_frame.to_csv(final_path, index=False)
            file_index += 1



if __name__ == "__main__":
    save_stock_data()