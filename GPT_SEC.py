from sec_api import QueryApi, ExtractorApi
import json
import requests
import pandas as pd
from IPython.display import display, HTML
from openai import OpenAI
import tiktoken 
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

def read_to_list(categ):
    text_corpus = []
    for cat in categ:
        with open("SEC_TSLA_"+cat+".txt") as file:
            text_corpus.append(file.read().replace('\n',' '))
    
    return text_corpus

def token_chunk(text:str, max_token: int, prompt="Summarize the following text for me mention the company name and the specifics."):
    encoding = tiktoken.get_encoding("cl100k_base")
    token_int = encoding.encode(text) 

    chunk_size = max_token - len(encoding.encode(prompt))

    chunks = [token_int[i: i+chunk_size] for i in range(0, len(token_int), chunk_size)]

    chunks = [encoding.decode(chunk) for chunk in chunks]
    responses = []
    message_construct = [{"role":"user", "content": "Summarize the following text for me mention the company name and the specifics."},
                         {"role":"user", "content": "To provide the context for the above prompt, I will send you text in parts. When I am finished, I will tell you 'ALL PARTS SENT'. Do not answer until you have received all the parts. "}] #initial prompt

    for chunk in chunks:
        message_construct.append({"role":"user", "content":chunk})
        while(sum(len(encoding.encode(msg["content"]))for msg in message_construct) > max_token):
            message_construct.pop()
        
        response = client.chat.completions.create(
            model = "gpt-3.5-turbo",
            messages = message_construct
        )
        re_string = response.choices[0].message.content.strip()
        responses.append(re_string)
    
    message_construct.append({"role": "user", "content": "ALL PARTS SENT"})

    response = client.chat.completions.create(model = "gpt-3.5-turbo", messages = message_construct)
    final_response = response.choices[0].message.content.strip()
    responses.append(final_response)
    return responses

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

if __name__ == "__main__":
    text = read_to_list(categories_10k)
    t1, t2 = split(text[0], prompt="Summarize the following text for me mention the company name and the specifics.")