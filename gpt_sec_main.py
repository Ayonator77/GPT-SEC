from text_handler import summary_, construct_message, split
from sec_query import SEC_QUERY
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import yfinance as yf
from datetime import datetime
from polygon import RESTClient
import pandas as pd
import tiktoken

client = RESTClient(api_key="")
finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
categories_10k = ["1","1A", "1B", "2", "3", "4", "5", "6", "7", "7A", "8", "9", "9A", "9B", "10", "11", "12", "13", "14", "15"]
categories_10q = ["part1item1", "part1item2", "part1item3", "part1item4", "part2item1", "part2item1a", "part2item2", "part2item3", "part2item4", "part2item5", "part2item6"]

def split_at_index(text:list, index:int):
    message_dictionary = {}
    encoding = tiktoken.get_encoding("cl100k_base")
    if len(encoding.encode(text[index])) > 16385:
        first_half, second_half = split(text[index],"Summarize the following text for me")
        return first_half, second_half
    else:
        return text[index], "n/a"


def main_10q(text:list, categories:list):
    summaries = []
    for i in range(len(categories)):
        first_half, second_half = split_at_index(text, i)
        if second_half == 'n/a':
            summaries.append(summary_(construct_message(text[i])))
        else:
            print("\n","******************TEXT WAS SPLIT******************", "\n")
            mess_init = [{"role": "user", "content": "Summarize the following text for me"}]
            mess_init.append({"role":"user", "content": first_half[0]})

            mess2 = [{"role": "user", "content": "Summarize the following text for me"}]
            mess2.append({"role":"user", "content": second_half[0]})

            sum1 = summary_(mess_init)
            sum2 = summary_(mess2)

            message_final = [{"role": "user", "content": "Summarize and combine these two summaries (They are from different halves of the same text), I will say ALL PARTS SENT when i want you to summarize"}]
            message_final.append({"role": "user", "content": sum1})
            message_final.append({"role": "user", "content": sum2})
            message_final.append({"role": "user", "content": "ALL PARTS SENT"})

            final_sum = summary_(message_final)
            summaries.append(final_sum)
    return summaries


def main(text:list):
    """
    Generates a list of section summaries
    Parameters:
        text -- list of text data, each item is a section i.e [section 1, section 1A,...., section 15]
    returns:
        summaries -- list of summaries of categories [summary 1, summary 1A, ...., summary 15]
    """
    #text = read_to_list(categories_10k)
    encoding = tiktoken.get_encoding("cl100k_base")
    summaries = []
    for i in range(0, 9):
        #summarize all 10-k categories from 1-7A excluding 8 since the text is too large
        print(text[i])
        print("Size of token: ", len(encoding.encode(text[i])))
        summaries.append(summary_(construct_message(text[i])))
    
    #split category 8(FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA )
    first_half, second_half = split(text[10], "Summarize the following text for me")

    #construct prompts for the first half of category 8
    mess_init = [{"role": "user", "content": "Summarize the following text for me"}]
    mess_init.append({"role":"user", "content": first_half[0]})

    #construct prompts for the second half of category 8
    mess2 = [{"role": "user", "content": "Summarize the following text for me"}]
    mess2.append({"role":"user", "content": second_half[0]})

    #summarize both halves
    sum1 = summary_(mess_init)
    sum2 = summary_(mess2)

    #Summarize the summary of both halves
    message_final = [{"role": "user", "content": "Summarize and combine these two summaries (They are from different halves of the same text), I will say ALL PARTS SENT when i want you to summarize"}]
    message_final.append({"role": "user", "content": sum1})
    message_final.append({"role": "user", "content": sum2})
    message_final.append({"role": "user", "content": "ALL PARTS SENT"})

    final_sum = summary_(message_final)
    summaries.append(final_sum)

    #summarize all 10-K categories from categories 9-15
    for i in range(11, len(text)-1):
        summaries.append(summary_(construct_message(text[i])))

    return summaries

def sentiment_analysis(summary):
    """
    Gets the sentiment analysis
    Parameters:
        summary -- summary of section
    Returns:
        sentiment analysis dictionary i.e [{'label': 'Positive', 'score': 0.9961360096931458}]

    """
    nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)
    return nlp(summary)

def get_stock_info(ticker, start_time, end_time):
    data_request = client.get_aggs(ticker=ticker, multiplier=1, timespan="minute", from_=start_time, to= end_time)
    return pd.DataFrame(data_request)


if __name__ == "__main__":
    # ticker = "TSLA"
    # data_request = client.get_aggs(ticker=ticker, multiplier=1, timespan="minute", from_='2023-10-01', to="2023-10-30")
    # price_data = pd.DataFrame(data_request)


    # print(price_data)
    # # stock_info = yf.Ticker("TSLA")
    # # hist_prices = stock_info.history(start='2023-10-01', end="2023-10-30", interval="5m")
    # # print(hist_prices)
    query = SEC_QUERY("10-Q", "TSLA", "10")
    text = []
    index = 0
    for cat in categories_10q:
        text.append(query.get_section_text(index, cat))
    
    summaries = main_10q(text, categories_10q)
    print(summaries)
    # for summ in summaries:
    #     print(sentiment_analysis(summ))
