import json

DATA_PATH = '../data/diffbot_articles.json'

def get_all_text():
    texts = []
    with open(DATA_PATH, 'r') as json_file:
        json_obj = json.load(json_file)
        for response in json_obj:
            if response['type'] == 'article':
                texts.append(response['text'])
    return texts
