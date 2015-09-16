import json
import re

DATA_PATH = '../data/diffbot_articles.json'
MIN_LINE_LENGTH = 20

def get_all_text():
    texts = []
    with open(DATA_PATH, 'r') as json_file:
        json_obj = json.load(json_file)
        for response in json_obj:
            if response['type'] == 'article':
                raw_text = response['text'].lower()
                processed_text = re.sub('([.,!?()])', r' \1 ', raw_text)
                processed_text = re.sub('\s{2,}', ' ', processed_text)
                lines = processed_text.split('\n')
                for line in lines:
                    line_split = line.split(' ')
                    if len(line_split) >= MIN_LINE_LENGTH:
                        this_seq = ['START'] + line_split + ['END']
                        texts.append(this_seq)
    return texts
