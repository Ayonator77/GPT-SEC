import sec_query
from sec_query import SEC_QUERY
import tiktoken
from openai import OpenAI

OPENEAI_API_KEY = "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"
client = OpenAI(api_key=OPENEAI_API_KEY)

def read_to_list(categ,ticker) -> list:
    """
    Reads text file
    Parameters:
        categ -- list of categories based on form type
        ticker -- Stock ticker, ("TSLA", "AAPl"...etc)
    """
    text_corpus = []
    for cat in categ:
        with open("SEC_FILES/SEC_"+ticker+"_"+cat+".txt") as file:
            text_corpus.append(file.read().replace('\n',' '))
    return text_corpus

def read_to_list(query: SEC_QUERY):
    pass


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