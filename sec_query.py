from sec_api import QueryApi, ExtractorApi
import json
import requests

queryAPI = QueryApi(api_key='6418a3b7423422e79368664fa3be7cd3b10b53ed1c6c5cfe4fceaea6b54f7791')
extractorApi = ExtractorApi(api_key='6418a3b7423422e79368664fa3be7cd3b10b53ed1c6c5cfe4fceaea6b54f7791')
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
    
    def get_section_text(self, index:int, section:str):
        """
        Get text data from a 10k section i.e. category_list = [1, 1A, 1B, 2, 3, 4, 5, 6, 7, 7A, 8, 9, 9A, 9B, 10, 11, 12, 13, 14, 15]
        Parameters:
            index -- filing index
            section -- section given form type
        Returns:
            string of section text
        """
        final_json, filing_url, q_dict = self.extract(index)
        return extractorApi.get_section(filing_url, section, "text")