from gpt_sec_main import main, get_stock_info, sentiment_analysis, main_10q
from sec_query import SEC_QUERY
import sec_query
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from transformers import BertTokenizer
import torch
from models import TransformerModel, train_model, LSTMModel
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

#10-q items: [part1item1, part1item2, part1item3, part1item4, part2item1, part2item1a, part2item2, part2item3, part2item4, part2item5, part2item6]
categories_10k = ["1","1A", "1B", "2", "3", "4", "5", "6", "7", "7A", "8", "9", "9A", "9B", "10", "11", "12", "13", "14", "15"]
categories_10q = ["part1item1", "part1item2", "part1item3", "part1item4", "part2item1", "part2item1a", "part2item2", "part2item3", "part2item4", "part2item5", "part2item6"]

def create_text_dataset(query:SEC_QUERY, categories:list, index:int):
    #query = SEC_QUERY(form_type, ticker, size)
    text = []
    analysis_list = []
    for cat in categories:
        text.append(query.get_section_text(index, cat))
    
    summaries = main_10q(text, categories)
    for summ in summaries:
        analysis_list.append(sentiment_analysis(summ))

    assert len(summaries) == len(analysis_list)
    sent_sum_tup = []
    for i in range(len(summaries)):
        sent_sum_tup.append((summaries[i], analysis_list[i]))
    return sent_sum_tup

def text_no_label(query:SEC_QUERY, categories:list, index:int):
    text = []
    for cat in categories:
        text.append(query.get_section_text(index, cat))
    return main_10q(text, categories)

def create_stock_dataset(query:SEC_QUERY, index:int) -> pd.core.frame.DataFrame:
    time_string = query.get_response_raw(index)['filedAt']
    start_time = datetime.fromisoformat(time_string)
    end_time = start_time + relativedelta(days=7)
    current_time = datetime.now()
    current_time = current_time.replace(tzinfo=None)
    
    # Turn datetime object into string
    start_time = start_time.strftime("%Y-%m-%d")
    end_time = end_time.strftime("%Y-%m-%d")

    #Get stock information from start time -> end time
    data_frame = get_stock_info(query.get_ticker(), start_time, end_time)
    data_frame['timestamp']=data_frame['timestamp'].apply(lambda x: datetime.fromtimestamp(x/1000))
    volatility = get_volatility(data_frame) 
    return data_frame, volatility


def append_text_data(query:SEC_QUERY, categories:list):
    full_text_dataset = []
    for i in range(query.get_size()):
        full_text_dataset.append(text_no_label(query, categories, i))
    
    return full_text_dataset

def preprocess_data(full_text:list):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_texts = []
    for text_list in full_text:
        enc_text_list = []
        for text in text_list:
            enc_text = tokenizer.encode(text, max_length=512, truncation=True, padding="max_length", return_tensors='pt')
            enc_text_list.append(enc_text)
        encoded_texts.append(enc_text_list)

    input_ids = torch.cat(encoded_texts, dim=0)
    attention_masks = (input_ids != tokenizer.pad_token_id).float()

    input_ids = input_ids.squeeze(1)
    return  input_ids, attention_masks

def load_summaries(file_paths: list):
    summaries = []
    for file_path in file_paths:  
        with open(file_path, 'r') as file:
            summaries.append(file.read())
    return summaries


def text_clustering(text_data, num_clusters=3):
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words="english")
    X = vectorizer.fit_transform(text_data)
    X = torch.tensor(X.toarray(), dtype=torch.float32)

    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(X)
    return torch.tensor(kmeans.labels_, dtype=torch.int32)


def preprocess(data:list, max_length):
    texts, labels = [], []
    for item in data:
        item
        texts.append(item[0])
        labels.append(item[1])
    
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit(labels)


    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_texts = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    return tokenized_texts, encoded_labels, label_encoder.classes_


def write_to_file(query:SEC_QUERY, categories):
    full_summary = append_text_data(query,categories)
    main_path = "Dataset"
   # os.mkdir(main_path)
    ticker_path = os.path.join(main_path, query.get_ticker())
    if not os.path.exists(main_path):
        os.makedirs(main_path)
    os.makedirs(ticker_path, exist_ok=True)
    for i in range(len(full_summary)):
        filing_date = query.get_response_raw(i)['filedAt']
        date_string = datetime.fromisoformat(filing_date)
        date_string = date_string.strftime("%Y-%m-%d")
        for i in range(len(full_summary)):
            with open(main_path+'/'+query.get_ticker()+'/'+date_string+'.txt', "w") as file:
                for line in full_summary[i]:
                    file.write(f"{line}\n")

def get_volatility(data_frame):
    data_frame['daily_returns'] = data_frame['close'].pct_change()
    volatility = data_frame['daily_returns'].std()
    return volatility

def test_lstm():
    pass


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



if __name__ == "__main__":
    #data_set = create_stock_dataset("10-Q", "TSLA", "10", 0)
    ticker_list = ["TSLA", "AAPL", "MSFT", "META", "GOOGL","AMZN", "NVDA", "AMD", "COST", "NFLX","QCOM", "MCD", "TTE", "BABA", "IBM", "AMAT", "SHOP", "BP", "T", "REGN"]
    query_list = [SEC_QUERY("10-Q", ticker, "10") for ticker in ticker_list]
   # sentiment_summary = append_text_data(query_list[0], categories_10q)
    for query in query_list:
        if query.get_ticker() != "TSLA" or  query.get_ticker() != "APPL":
            write_to_file(query, categories_10q)
            print(query.get_ticker()+" added")

    input_size = 7  # Number of features in each input sequence
    hidden_size = 64  # Number of hidden units in the LSTM layer
    num_layers = 2  # Number of LSTM layers
    output_size = 1  # Dimensionality of the output (single value for predicting next day's return)


    stock_data = create_stock_dataset(query_list[0], 0)[0]
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(stock_data[['open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions']])
    print(stock_data)
    print(normalized_data)

    test_data = torch.tensor(normalized_data[-1], dtype=torch.float32)


    model = LSTMModel(input_size, hidden_size, output_size)
    model.eval()
    with torch.no_grad():
        test_data = test_data.unsqueeze(0).unsqueeze(0)
        print("Test Inference: \n",model(test_data))
    print(model)

    # text_labels = text_clustering(sentiment_summary[0])
    # test_list = [sentiment_summary[0], text_labels] 
    #print(sentiment_summary, text_labels)
    #print(test_list)
    #print(test_list[0])
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # token_text, labels, classes = preprocess(test_list[0], 100)

    # vocab_size = tokenizer.vocab_size
    # embed_size = 128
    
    # num_heads = 4
    # hidden_size = 256
    # num_layers = 3
    # num_classes = len(classes)
    # dropout = 0.1
    # print(token_text, labels, classes)
    # model = TransformerModel(vocab_size, embeded_size=embed_size, num_heads=num_heads, num_layers=num_layers, num_classes=num_classes, max_length=100, dropout=dropout, hidden_size=hidden_size)
    # print(model(token_text['input_ids']))
    # # for item in test_list:
    #     print(item)
    #print(test_list[0][0])
    # print([preprocess_data(sentiment_summary)])
    # query = SEC_QUERY("10-Q", "TSLA", "10")
    #qstring = query.get_section_text(0, "part1item1")
    # time_string = query.get_response_raw(0)['filedAt']
    # date_time_ = datetime.fromisoformat(time_string)
    # date_time = date_time_.strftime("%Y-%m-%d %H:%M:%S")
    # print(date_time)
    # print(date_time_+relativedelta(months=4))
    # # ticker = "TSLA"
    # data_frame = get_stock_info(ticker, '2024-01-01', '2024-01-03')
    # data_frame['timestamp']=data_frame['timestamp'].apply(lambda x: datetime.fromtimestamp(x/1000))
    # #data_frame['daily_returns'] = data_frame['close'].pct_change()
    # volatility = get_volatility(data_frame)
    # print(data_frame,'\n',"This is the volatility: ", volatility)