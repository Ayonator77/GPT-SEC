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