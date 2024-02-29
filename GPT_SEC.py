from sec_api import QueryApi, ExtractorApi
import json
import requests
import pandas as pd
from openai import OpenAI
import tiktoken 
from concurrent.futures import ThreadPoolExecutor
import shutil
import os


queryAPI = QueryApi(api_key='041c38e656c459b1b336a7bea7d83c057482f669d2e663a1b6fa8864b2071550')
extractorApi = ExtractorApi(api_key='041c38e656c459b1b336a7bea7d83c057482f669d2e663a1b6fa8864b2071550')
OPENEAI_API_KEY = "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"
client = OpenAI(api_key=OPENEAI_API_KEY)
categories_10k = ["1","1A", "1B", "2", "3", "4", "5", "6", "7", "7A", "8", "9", "9A", "9B", "10", "11", "12", "13", "14", "15"]

#Class to strucutre EDGAR API query and get SEC filings
class SEC_QUERY:

    def __init__(self, form_type, ticker, size):
        self.form_type = form_type
        self.ticker = ticker
        self.size = size
        #Set query strucuure 
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
    
    def get_filing(self):
        return queryAPI.get_filings(self.query)

    def extract(self, index:int): #use extraction api to get text and tablular data
        query_str = self.get_response(index)
        q_dict = json.loads(query_str)
        filing_url = q_dict['linkToHtml']

        xbrl_converter_api_endpoint = "https://api.sec-api.io/xbrl-to-json"
        api_key = '041c38e656c459b1b336a7bea7d83c057482f669d2e663a1b6fa8864b2071550'

        final_url = xbrl_converter_api_endpoint + "?htm-url=" + filing_url + "&token=" + api_key
        response = requests.get(final_url)

        final_json = json.loads(response.text)
        return final_json, filing_url, q_dict


def EDGAR_CALL(query:SEC_QUERY, index):
    return query.extract(index)

#10k sections = [1, 1A, 1B, 2, 3, 4, 5, 6, 7, 7A, 8, 9, 9A, 9B, 10, 11, 12, 13, 14, 15]
def get_section_text(query:SEC_QUERY, section:str, index):
    final_json, filing_url, q_dict = EDGAR_CALL(query, index)
    return extractorApi.get_section(filing_url, section, "text")

def read_to_list(categ,ticker):
    text_corpus = []
    for cat in categ:
        with open("SEC_FILES/SEC_"+ticker+"_"+cat+".txt") as file:
            text_corpus.append(file.read().replace('\n',' '))
    
    return text_corpus

def split(text:str, prompt, max_token=4000):
    encoding = tiktoken.get_encoding("cl100k_base")
    token_int = encoding.encode(text)
    chunk_size =  max_token - len(encoding.encode(prompt))
    text_encoding = [token_int[i: i+chunk_size] for i in range(0, len(token_int), chunk_size)]
    text_encoding = [encoding.decode(chunk) for chunk in text_encoding]
    mid_point = (len(text_encoding) - 1) // 2
    text_1 = text_encoding[0:mid_point]
    text_2 = text_encoding[mid_point::]
    return text_1, text_2


def split_token(text:str, max_token: int, prompt="Summarize the following text for me."):
    encoding = tiktoken.get_encoding("cl100k_base")
    token_int = encoding.encode(text)

    chunk_size =  max_token - len(encoding.encode(prompt))
    c_lsit = [token_int[i: i+chunk_size] for i in range(0, len(token_int), chunk_size)]
    c_list = [encoding.decode(chunk) for chunk in c_lsit]

    return c_list, prompt

def construct_message(text):
    text_chunk, prompt = split_token(text, max_token=4000)
    message_const = [{"role":"user", "content":prompt}, 
                     {"role":"user", "content": "To provide the context for the above prompt, I will send you text in parts. When I am finished, I will tell you 'ALL PARTS SENT'. Do not answer until you have received all the parts. "}]
    for chunk in text_chunk:
        message_const.append({"role":"user", "content":chunk})
        
    message_const.append({"role": "user", "content": "ALL PARTS SENT"})
    return message_const

def summary_(message):
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=message)
    return response.choices[0].message.content


def main(text):
    text = read_to_list(categories_10k)
    summaries = []
    for i in range(0, 9):
        summaries.append(summary_(construct_message(text[i])))
    
    first_half, second_half = split(text[10], "Summarize the following text for me")

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

    for i in range(11, len(text)-1):
        summaries.append(summary_(construct_message(text[i])))

    return summaries


def write_to_file():
    pass

if __name__ == "__main__":
    text = read_to_list(categories_10k)
    summaries = main(text)

    
    # os.mkdir("OpenAI_Summary")
    # with open("OpenAI_Summary/Summary.txt", 'w') as f:
    #     for sum in summaries:
    #         f.write(f"{sum}\n")

    print(summaries)
    #t1, t2 = split(text[0], prompt="Summarize the following text for me mention the company name and the specifics.")
    #chunks = split_into_chunks(text[10])
