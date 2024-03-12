from text_handler import summary_, construct_message, split
from sec_query import SEC_QUERY
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
categories_10k = ["1","1A", "1B", "2", "3", "4", "5", "6", "7", "7A", "8", "9", "9A", "9B", "10", "11", "12", "13", "14", "15"]


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


if __name__ == "__main__":
    query = SEC_QUERY("10-K", "TSLA", "10")
    text = []
    index = 0
    for cat in categories_10k:
        text.append(query.get_section_text(index, cat))
    
    summaries = main(text)
    for summ in summaries:
        print(sentiment_analysis(summ))