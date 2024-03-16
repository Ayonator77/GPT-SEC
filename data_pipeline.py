from gpt_sec_main import main, get_stock_info, sentiment_analysis, main_10q
from sec_query import SEC_QUERY
import sec_query
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
#10-q items: [part1item1, part1item2, part1item3, part1item4, part2item1, part2item1a, part2item2, part2item3, part2item4, part2item5, part2item6]
categories_10k = ["1","1A", "1B", "2", "3", "4", "5", "6", "7", "7A", "8", "9", "9A", "9B", "10", "11", "12", "13", "14", "15"]
catagories_10q = ["part1item1", "part1item2", "part1item3", "part1item4", "part2item1", "part2item1a", "part2item2", "part2item3", "part2item4", "part2item5", "part2item6"]

def create_text_dataset(form_type:str, ticker:str, size:str, categories:list, index):
    query = SEC_QUERY(form_type, ticker, size)
    text = []
    analysis_list = []
    for cat in categories:
        text.append(query.get_section_text(index, cat))
    
    summaries = main_10q(text, categories)
    print(summaries)
    for summ in summaries:
        analysis_list.append(sentiment_analysis(summ))

    assert len(summaries) == len(analysis_list)
    sent_sum_tup = []
    for i in range(len(summaries)):
        sent_sum_tup.append((summaries[i], analysis_list[i]))
    return sent_sum_tup

def create_stock_dataset(form_type:str, ticker:str, size:str, index:int) -> pd.core.frame.DataFrame:
    query = SEC_QUERY(form_type, ticker, size)
    time_string = query.get_response_raw(index)['filedAt']
    start_time = datetime.fromisoformat(time_string)
    end_time = start_time + relativedelta(days=7)
    current_time = datetime.now()
    current_time = current_time.replace(tzinfo=None)
    
    # Turn datetime object into string
    start_time = start_time.strftime("%Y-%m-%d")
    end_time = end_time.strftime("%Y-%m-%d")

    #Get stock information from start time -> end time
    data_frame = get_stock_info(ticker, start_time, end_time)
    data_frame['timestamp']=data_frame['timestamp'].apply(lambda x: datetime.fromtimestamp(x/1000))
    volatility = get_volatility(data_frame)
    return data_frame, volatility


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
    data_set= create_stock_dataset("10-Q", "TSLA", "10", 0)
    sentiment_summary = create_text_dataset("10-Q", "TSLA", "10", catagories_10q, 0)
    print(data_set)
    print(sentiment_summary)
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
