from sec_api import QueryApi, ExtractorApi
import json
import requests
import pandas as pd
from openai import OpenAI
import tiktoken 
from concurrent.futures import ThreadPoolExecutor
import shutil
import os
#from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup
import re
import nltk
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

queryAPI = QueryApi(api_key='7e0ce86ecefb7ae99c305975ef3220e641df56f66c369fdda22100d1ecdbf080')
extractorApi = ExtractorApi(api_key='7e0ce86ecefb7ae99c305975ef3220e641df56f66c369fdda22100d1ecdbf080')
OPENEAI_API_KEY = "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"
client = OpenAI(api_key=OPENEAI_API_KEY)
categories_10k = ["1","1A", "1B", "2", "3", "4", "5", "6", "7", "7A", "8", "9", "9A", "9B", "10", "11", "12", "13", "14", "15"]


class SEC_QUERY:
    """
    Create class structure for EDGAR API query

    Attributes:
        form_type -- (10-K, 10-Q...etc)
        ticker -- Stock ticker, ("TSLA", "AAPl"...etc)
        size -- Determines how far back the query goes

    """
    def __init__(self, form_type, ticker, size):
        self.form_type = form_type
        self.ticker = ticker
        self.size = size
        #Set query structure 
        self.query = {
            "query": { 
                "query_string" : {
                    "query": f"formType:\"{self.form_type}\" AND ticker:{self.ticker}",
                }
            },
            "from": "0",
            "size": size,
            "sort": [{"filedAt": {"order": "desc"}}]
        }
    
    def get_response(self, index:int):
        """
        Gets the filing dictionary at a specific index
        Parameters:
            index -- index of the filing you want given a query
        Returns:
            filings in dictionary format 
        """
        response = queryAPI.get_filings(self.query)
        return json.dumps(response["filings"][index], indent=2)
    
    def get_filing(self):
        """Gets the full query for sec_api"""
        return queryAPI.get_filings(self.query)

    def extract(self, index:int): #use extraction api to get text and tabular data
        """
        Uses extraction api to get text and tabular data
        Parameters:
            index -- index of the filing you are looking for, cannot exceed self.size
        Returns:
            tuple -- (extracted json, url for extraction, initial filing dictionary)
        """
        query_str = self.get_response(index)
        q_dict = json.loads(query_str)
        filing_url = q_dict['linkToHtml']

        xbrl_converter_api_endpoint = "https://api.sec-api.io/xbrl-to-json"
        api_key = '7e0ce86ecefb7ae99c305975ef3220e641df56f66c369fdda22100d1ecdbf080'

        final_url = xbrl_converter_api_endpoint + "?htm-url=" + filing_url + "&token=" + api_key
        response = requests.get(final_url)

        final_json = json.loads(response.text)
        return final_json, filing_url, q_dict


def EDGAR_CALL(query:SEC_QUERY, index:int):
    return query.extract(index)

def get_section_text(query:SEC_QUERY, section:str, index:int):
    """
    Get text data from a 10k section i.e. category_list = [1, 1A, 1B, 2, 3, 4, 5, 6, 7, 7A, 8, 9, 9A, 9B, 10, 11, 12, 13, 14, 15]
    Parameters:
        query -- SEC_QUERY instance
        index -- filing index
    Returns:
        string of section text
    """
    final_json, filing_url, q_dict = EDGAR_CALL(query, index)
    return extractorApi.get_section(filing_url, section, "text")

#read text file, takes in category list and stock ticker
"""
    Reads text file
    Keyword arguments:
    categ -- list of categories based on form type
    ticker -- Stock ticker, ("TSLA", "AAPl"...etc)
"""
def read_to_list(categ,ticker) -> list:
    text_corpus = []
    for cat in categ:
        with open("SEC_FILES/SEC_"+ticker+"_"+cat+".txt") as file:
            text_corpus.append(file.read().replace('\n',' '))
    return text_corpus


def split(text:str, prompt:str, max_token=4000):
    """
    Splitting text body into 2 parts (specifically for section 8 of form 10-K) to be able to pass through OpenAi api
    Parameters:
        text -- Section text that needs to be split
        prompt -- Initial prompt 
        max_token -- max token size for gpt-3.5-turbo, used to chunk the text
    Returns:
        tuple -- text_1, text_2 which is the text data split in half
    """
    #gives encoding for gpt-3.5-turbo and gpt-4
    encoding = tiktoken.get_encoding("cl100k_base")
    token_int = encoding.encode(text)
    chunk_size =  max_token - len(encoding.encode(prompt))
    #chunk the encoded text
    text_encoding = [token_int[i: i+chunk_size] for i in range(0, len(token_int), chunk_size)]
    text_encoding = [encoding.decode(chunk) for chunk in text_encoding] #decode the text
    #Split the list in half
    mid_point = (len(text_encoding) - 1) // 2
    text_1 = text_encoding[0:mid_point]
    text_2 = text_encoding[mid_point::]
    #returns both halves as a list
    return text_1, text_2


def split_token(text:str, max_token: int, prompt="Summarize the following text for me."):
    """
    Takes text data and breaks it up into chunks to pass through gpt-3.5-turbo
    Parameters:
        text -- section data as a string
        max-token -- maximum token size for gpt-3.5-turbo 
        prompt -- initial prompt
    Returns:
    tuple (c_list, prompt) -- chunked list and initial prompt
    """
    encoding = tiktoken.get_encoding("cl100k_base") 
    token_int = encoding.encode(text)

    #set chunk_size gpt-3.5-turbo token size cannot surpass 4000
    chunk_size =  max_token - len(encoding.encode(prompt))
    c_lsit = [token_int[i: i+chunk_size] for i in range(0, len(token_int), chunk_size)]
    c_list = [encoding.decode(chunk) for chunk in c_lsit]

    return c_list, prompt


def construct_message(text):
    """
    Chunks text data and construct a message dictionary for OpenAI api 
    Parameters:
        text -- section text for given form type
    Returns:
        message_const -- list of dictionaries [{role: user, content: text_chunk_1}, ....]
    """
    #breaks text into chunks, each chunk is a prompt
    text_chunk, prompt = split_token(text, max_token=4000) 
    message_const = [{"role":"user", "content":prompt}, #initial prompt asking gpt to summarize the text
                     {"role":"user", "content": "To provide the context for the above prompt, I will send you text in parts. When I am finished, I will tell you 'ALL PARTS SENT'. Do not answer until you have received all the parts. "}]
    for chunk in text_chunk:
        #adding each chunk as a prompt 
        message_const.append({"role":"user", "content":chunk})
    
    #indicator for gpt to start the summary
    message_const.append({"role": "user", "content": "ALL PARTS SENT"})
    return message_const


def summary_(message) -> str:
    """
    Calls OpenAi api to summmarize text
    Parameters:
        message -- message list of dictionary [{role: user, content: prompt 1}, ...,{role: user, content: prompt n}]
    Returns:
        string -- gpt-3.5-turbo response to message dictionary as prompt
    """
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=message)
    #return the response as a string
    return response.choices[0].message.content 


def main(text:list):
    """
    Generates a list of section summaries
    Parameters:
        text -- list of text data, each item is a section i.e [section 1, section 1A,...., section 15]
    returns:
        summaries -- list of summaries of categories [summary 1, summary 1A, ...., summary 15]
    """
    #text = read_to_list(categories_10k)
    summaries = []
    for i in range(0, 9):
        #summarize all 10-k categories from 1-7A excluding 8 since the text is too large
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


def write_to_file(ticker:str, categories:list, size:str):
    """
    Writes SEC filing to folder
    Parameters:
        ticker -- Stock ticker
        categories -- Category for form type
        size -- size of the filing query
    """
    query = SEC_QUERY("10-K", ticker, size)
    size_int = int(size)
    for i in range(size_int):
        temp, filing_url, temp2 = query.extract(i)
        folder = "SEC_"+ticker+"_"+str(i)
        os.mkdir(folder)
        for cat in categories:
            section = extractorApi.get_section(filing_url,cat, "text")
            with open(folder+"/SEC_"+ticker+cat+".txt", "w") as f:
                f.write(section)


def get_sentiment(sentence):
    sentiment_categories  =  ['Negative', 'Positive', 'Uncertainty', 'Litigious', 'Strong_Modal', 'Weak_Modal', 'Constraining']
    #all_words = list(sent_df())
    sentiment_dict = {key:0 for key in sentiment_categories}

if __name__ == "__main__":
    text = read_to_list(categories_10k)
    summaries = main(text)
    nlp = pipeline("sentiment-analysis",model=finbert, tokenizer=tokenizer)
    for summ in summaries:
        print(nlp(summ))
