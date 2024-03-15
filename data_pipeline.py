from gpt_sec_main import main, get_stock_info, sentiment_analysis
from sec_query import SEC_QUERY
import sec_query
import pandas as pd
from datetime import datetime
#10-q items: [part1item1, part1item2, part1item3, part1item4, part2item1, part2item1a, part2item2, part2item3, part2item4, part2item5, part2item6]
categories_10k = ["1","1A", "1B", "2", "3", "4", "5", "6", "7", "7A", "8", "9", "9A", "9B", "10", "11", "12", "13", "14", "15"]

def convert_time(x):
    return datetime.fromtimestamp(x/1000)


def create_text_dataset(form_type, ticker:str, size:str):
    query = SEC_QUERY(form_type, ticker, size)
    text = []
    analysis_list = []
    index = 0
    for cat in categories_10k:
        text.append(query.get_section_text(index, cat))
    
    summaries = main(text)
    print(summaries)
    for summ in summaries:
        analysis_list.append(sentiment_analysis(summ))

    assert len(summaries) == len(analysis_list)
    sent_sum_tup = []
    for i in range(len(summaries)):
        sent_sum_tup.append((summaries[i], analysis_list[i]))
    return sent_sum_tup

def load_summaries(file_paths: list):
    summaries = []
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            summaries.append(file.read())
    return summaries

def appy_sentiment(summaries):
    pass

def get_volatility(data_frame):
    data_frame['daily_returns'] = data_frame['close'].pct_change()
    volatility = data_frame['daily_returns'].std()
    return volatility

if __name__ == "__main__":
    query = SEC_QUERY("10-Q", "TSLA", "10")
    qstring = query.get_section_text(0, "part1item1")
    print(qstring)
    ticker = "TSLA"
    # data_frame = get_stock_info(ticker, '2024-01-01', '2024-01-03')
    # data_frame['timestamp']=data_frame['timestamp'].apply(lambda x: datetime.fromtimestamp(x/1000))
    # #data_frame['daily_returns'] = data_frame['close'].pct_change()
    # volatility = get_volatility(data_frame)
    # print(data_frame,'\n',"This is the volatility: ", volatility)
