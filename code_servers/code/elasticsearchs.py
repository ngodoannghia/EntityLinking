import os
import re
import json
from tqdm import tqdm
import pickle
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk


with open('../data/mentions.json', 'r') as f:
    mentions = json.load(f)

with open('../data/entities.json', 'r') as f:
    entities = json.load(f)

with open('../data/mentions_entities.json', 'r') as f:
    mentions_entities = json.load(f)

with open('../data/alias.pkl', 'rb') as f:
    alias = pickle.load(f)

with open('../data/redirects.pkl', 'rb') as f:
    redirects = pickle.load(f)

with open('../data/id2title.pkl', 'rb') as f:
    id2title = pickle.load(f)

ES_NODES = "http://localhost:9200"
def create_index(index, client):
    client.indices.create(
        index=index,
        body={
            "settings": {"number_of_shards": 1},
            "mappings": {
                "properties": {
                    "entity": {
                        "type": "keyword", 
                        "fields": {
                            "length": { 
                                "type": "token_count",
                                "analyzer": "standard"
                            }
                        }
                    },
                    "alias": {"type": "text"},
                    "redirects": {"type": "text"},
                }
            },
        },
    )

def generate_action():
  
    for i, id in enumerate(alias):
        if id in alias:
            als = [e for e in alias[id]]
        else:
            als = []
        if id in redirects:
            red = [e for e in redirects[id]]
        else:
            red = []

        als.append(id2title[id])
        red.append(id2title[id])
        als = list(set(als))
        red = list(set(red))
        
        als = ' '.join([re.sub(r'[\/\\~\!\@\#\$\%\^\&\*\(\)\-\+\[\]\"\'\:\`\{\}]', ' ', m) for m in als])
        red = ' '.join([re.sub(r'[\/\\~\!\@\#\$\%\^\&\*\(\)\-\+\[\]\"\'\:\`\{\}]', ' ', m) for m in red])

        yield {'entity': id2title[id], 'alias': als, 'redirects': red}

def index_es_data(index):
    index = 'candidate'
    client = Elasticsearch(hosts = [ES_NODES])
    if client.indices.exists(index=index):
        print("deleting the '{}' index.".format(index))
        res = client.indices.delete(index=index)
        print("Response from server: {}".format(res))
  
    create_index(index, client)

    successes = 0

    for ok, action in streaming_bulk(
        client=client, index=index, actions=generate_action(),
    ):
        successes += ok

    print(successes)

index_es_data('candidate')