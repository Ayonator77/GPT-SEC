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


"""
    Create class structure for EDGAR API query
    class argument:
    form_type -- (10-K, 10-Q...etc)
    ticker -- Stock ticker, ("TSLA", "AAPl"...etc)
    size -- Determines how far back the query goes
"""
class SEC_QUERY:

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
        response = queryAPI.get_filings(self.query)
        return json.dumps(response["filings"][index], indent=2)
    
    def get_size(self):
        return self.size
    
    def get_filing(self): #get full query
        return queryAPI.get_filings(self.query)

    def extract(self, index:int): #use extraction api to get text and tabular data
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

#Gets text data from a 10k section i.e. category_list = [1, 1A, 1B, 2, 3, 4, 5, 6, 7, 7A, 8, 9, 9A, 9B, 10, 11, 12, 13, 14, 15]
def get_section_text(query:SEC_QUERY, section:str, index:int):
    final_json, filing_url, q_dict = EDGAR_CALL(query, index)
    return extractorApi.get_section(filing_url, section, "text")

#read text file, takes in category list and stock ticker
def read_to_list(categ,ticker):
    text_corpus = []
    for cat in categ:
        with open("SEC_"+ticker+"_"+str(index)+"/SEC_"+ticker+cat+".txt") as file:
            text_corpus.append(file.read().replace('\n',' '))
    return text_corpus

#splitting text corpus into 2 parts for the purpose of passing it through openai api
def split(text:str, prompt, max_token=4000):
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
    encoding = tiktoken.get_encoding("cl100k_base") 
    token_int = encoding.encode(text)

    #set chunk_size gpt-3.5-turbo token size cannot surpass 4000
    chunk_size =  max_token - len(encoding.encode(prompt))
    c_lsit = [token_int[i: i+chunk_size] for i in range(0, len(token_int), chunk_size)]
    c_list = [encoding.decode(chunk) for chunk in c_lsit]

    return c_list, prompt

#Constructs the message parameter in client.chat.completions.create function, these are the prompts
def construct_message(text):
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

"""
    Calls OpenAi api to summmarize text
    Keyword arguments:
    message -- message dictionary 
"""
def summary_(message) -> str:
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=message)
    #return the response as a string
    return response.choices[0].message.content 


def main(text):
    text = read_to_list(categories_10k)
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


def download():
    dl = Downloader("SCS",email_address="ayodejiodetola@gmail.com")
    dl.get("10-K", "TSLA", include_amends=True)

def write_to_file(ticker:str, categories, size):
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

def read_html(file_path:str):
    try:
        html_file = open(file_path, "r")
        index = html_file.read()
        s = BeautifulSoup(index, 'lxml')
        return s.get_text(separator=' ', strip='\n')
    except FileNotFoundError:
        print("File path not found")

def to_search(match_elem):
    list_match = match_elem.split(" ")
    list_match[0] = list_match[0].capitalize()
    return ' '.join(list_match)

def get_section(section:str, filing_text):
    # Remove these characters and replace them with empty string
    filing_text = re.sub('\xa0','', filing_text)
    filing_text = re.sub('\n', '',filing_text)
    filing_text = re.sub('\t','', filing_text)
    #print("file test",filing_text[0:1000])
    if section[-1] != '.':
        section+='.'
    #Match all the ITEM elements in document
    matches = list(re.finditer(re.compile('(Item|ITEM)\s?\d{0,2}[A-Z]?\.'),filing_text))
    section_list = [m[0] for m in matches]
    #print(len(section_list), section_list)
    #print(type(matches))
    for i in range(len(matches)):
        print(matches[i][0])
    assert section in section_list, f"The section should be in the list {section_list}"
    item_matches = [i for i in range(len(matches)) if matches[i][0] == section]
    #print("Item matches: ", item_matches)

    section_name = ''
    if len(item_matches) >= 1:
        section_name_start = min(item_matches)
        section_name_end = section_name_start+1
        start = matches[section_name_start].span()[1]
        end = matches[section_name_end].span()[0]

        section_name = filing_text[start:end]
        section_name = re.sub('\d', '', section_name)
    
    start = max(item_matches)
    end = start+1
    start = matches[start].span()[1]
    end = matches[end].span()[0]+10000
    #print(start, end)

    section_text = filing_text[start:end]
    section_text = re.sub(section_name, '', section_text, count=1)
    section_text = re.sub(section_name.upper(), '', section_text, count=1)
    return section_text, section_name.strip()


def preprocess_data(section_text):
    word_tokens = word_tokenize(section_text)
    sentence = []

    for word in word_tokens:
        sentence.append(word)
    
    lemmatizer  = WordNetLemmatizer()
    sentence = [lemmatizer.lemmatize(plural) for plural in sentence]
    return sentence

def get_sentiment(sentence):
    sentiment_categories  =  ['Negative', 'Positive', 'Uncertainty', 'Litigious', 'Strong_Modal', 'Weak_Modal', 'Constraining']
    #all_words = list(sent_df())
    sentiment_dict = {key:0 for key in sentiment_categories}

def historical_summary(query:SEC_QUERY):
    size = query.get_size()
    size = 4 #comment out when obtained EDGAR api keys
    summary_list = [] #[[sec_size: 1, 1A,.., 15], [sec_size-1: 1, 1A,...15],....[sec_0: 1, 1A,...15]]
    for index in range(size, -1, -1 ):
        text = read_to_list(categories_10k, "TSLA", index)
        summary_list.append(main(text))

    message = [{"role":"User", "content": "Summarize and combine the text below"}]
    for summary in summary_list:
        for line in summary:
            pass


def sentiment_analysis(text):
    nlp = pipeline("sentiment-analysis",model=finbert, tokenizer=tokenizer)
    result_list = []
    for line in text:
        result_list.append(nlp(line))
    return result_list



if __name__ == "__main__":
    text = read_to_list(categories_10k, "TSLA", 0)
    encoding = tiktoken.get_encoding("cl100k_base") 
    num_tokens = len(encoding.encode(text[10]))
    print("Token Size: ",num_tokens)
    #summaries = main(text)
    #write_to_file("TSLA", categories_10k, "10")

    # summary = open("OpenAI_Summary\Summary.txt")
    # summary_lines = summary.readlines()
    # #prepro = preprocess_data(summary_lines[0])
    
    # nlp = pipeline("sentiment-analysis",model=finbert, tokenizer=tokenizer)
    # for line in summary_lines:
    #     results = nlp(line)
    #     print(results)


    # string = read_html('C:/Users/Owner/GPT-SEC/sec-edgar-filings/TSLA/10-K/0000950170-23-001409/full-submission.txt')
    # #print(type(string))
    # str1, str2 = get_section("Item 8.", string)
    # print(str1)
    # os.mkdir("OpenAI_Summary")
    # with open("OpenAI_Summary/Summary.txt", 'w') as f:
    #     for sum in summaries:
    #         f.write(f"{sum}\n")


    #t1, t2 = split(text[0], prompt="Summarize the following text for me mention the company name and the specifics.")
    #chunks = split_into_chunks(text[10])
