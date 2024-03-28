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
import json

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


def save_stock_data(query:SEC_QUERY, index:int) :
    t_list = os.listdir("Text Dataset")
    dates = []
    for tick in t_list:
        dates.append(os.listdir(os.path.join("Text Dataset", tick)))


def append_text_data(query:SEC_QUERY, categories:list):
    full_text_dataset = []
    for i in range(query.get_size()):
        print(i, query)
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
    print("Started ", query.get_ticker())
    size = query.get_size()
    text_list = []
    full_summary = []
    for i in range(size):
        print("Started summaries")
        for cat in categories:
            text_list.append(query.get_section_text(i, cat))
        print("Appending summaries")
        full_summary.append(main_10q(text_list, categories))
        text_list.clear()
    

    main_path = "Text Dataset"
    ticker_path = os.path.join(main_path, query.get_ticker())
    if not os.path.exists(main_path):
        os.makedirs(main_path)
    
    os.makedirs(ticker_path, exist_ok=True)

    for i in range(size):
        filing_date = query.get_response_raw(i)['filedAt']
        date_string = datetime.fromisoformat(filing_date)
        date_string = date_string.strftime("%Y-%m-%d")
        with open(main_path+'/'+query.get_ticker()+'/'+date_string+'.txt', "w") as file:
            for line in full_summary[i]:
                file.write(line+'\n')
    

def get_volatility(data_frame):
    data_frame['daily_returns'] = data_frame['close'].pct_change()
    volatility = data_frame['daily_returns'].std()
    return volatility

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




class data_pipeline():
    def __init__(self, file_text, file_frame) -> None:
        self.file_text = file_text
        self.file_frame = file_frame

    def extract_text(self, main_path="Text Dataset"):
        path_list = os.listdir(main_path)
        text_file = open(self.file_text, "r")
        summ_list = text_file.readlines()
        return summ_list

def combine_summaries():
    text_dir = os.listdir("Text Dataset")
    summ_dir = os.listdir(text_dir[0])
    path = os.path.join(text_dir[0], summ_dir[0])
    with open(path, "r") as file:
        summ_list = file.readlines()
    
    message = {"role": "user", "message": "This is a Summary for an SEC filing, use this summary to summarize the following text"}

if __name__ == "__main__":
    #data_set = create_stock_dataset("10-Q", "TSLA", "10", 0)
    # "TSLA", "AAPL", "MSFT", "META", "GOOGL","AMZN", "NVDA", "AMD", "COST", "NFLX","QCOM", "MCD", "TTE", "BABA", "IBM", "AMAT", "SHOP", "BP", "T", "REGN"
    #ticker_list = [ "TSLA", "AAPL", "MSFT", "META", "GOOGL","AMZN", "NVDA", "AMD", "COST", "NFLX","QCOM", "MCD", "TTE", "BABA", "IBM", "AMAT", "SHOP", "BP", "T", "REGN"]
    with open("company_tickers.json") as json_file:
        data = json.load(json_file)
    
    ticker_list = []
    for key, value in data.items():
       ticker_list.append(value['ticker'])

    query_list = [SEC_QUERY("10-Q", ticker, "10") for ticker in ticker_list[69:]]
    #text_listdir = [ticker[:-4] for ticker in text_listdir ]
  # sentiment_summary = append_text_data(query_list[0], categories_10q)
    for query in query_list:
        text_listdir = os.listdir("Text Dataset")
        if not (query.get_ticker() in text_listdir):
            print(query.get_ticker(), " ", text_listdir)
            try:
                write_to_file(query, categories_10q)
                print(query.get_ticker()+" added")
            except:
                print("Error got called on ", query.get_ticker())
                pass

    input_size = 7  # Number of features in each input sequence
    hidden_size = 64  # Number of hidden units in the LSTM layer
    num_layers = 2  # Number of LSTM layers
    output_size = 1  # Dimensionality of the output (single value for predicting next day's return)


    stock_data = create_stock_dataset(query_list[0], 9)[0]
    print(stock_data)
    # scaler = MinMaxScaler()
    # normalized_data = scaler.fit_transform(stock_data[['open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions']])
    # print(stock_data)
    # print(normalized_data)

    # test_data = torch.tensor(normalized_data[-1], dtype=torch.float32)


    # model = LSTMModel(input_size, hidden_size, output_size)
    # model.eval()
    # with torch.no_grad():
    #     test_data = test_data.unsqueeze(0).unsqueeze(0)
    #     print("Test Inference: \n",model(test_data))
    #print(model)

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