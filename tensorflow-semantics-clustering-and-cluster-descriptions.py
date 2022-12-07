import pandas as pd
import tensorflow_hub as hub
from tensorflow_text import SentencepieceTokenizer
from sklearn.cluster import KMeans

CLUSTER_CNT = 50 #50
#https://tfhub.dev/google/universal-sentence-encoder-multilingual/3
use_model_url = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3'

def fill_get_dictionary(cluster_ids,text_list):
    result = dict()
    for i, txt in enumerate(text_list):
        if cluster_ids[i] in result.keys():
            result[cluster_ids[i]].append(txt)
        else:
            result[cluster_ids[i]] = [txt]
    return result

def get_mindsets(ids,text_list):
    d = fill_get_dictionary(
        ids,
        text_list)

    r = []

    for id in d.keys():
        for text in d[id]:
            r.append((text,id))

    return r

def get_text_list():
    return [desc[0] for desc in pd.read_csv(
        './nlp/site-tags.csv',
        usecols=['description'])\
        .dropna()\
        .values]

def get_semantics(text_list):
    use_model = hub.load(use_model_url)
    semantics = use_model(text_list)
    print(semantics)
    return semantics

def get_cluster_ids(semantics):
    kmeans = KMeans(n_clusters = CLUSTER_CNT)
    kmeans.fit(semantics)    
    return kmeans.labels_

def save_mindsets(data):
    pd.DataFrame(
        data,
        columns=[
            'text',
            'mindsetID'])\
        .to_csv(
            './nlp/literature-mindset-clusters.csv',
            index=False)

def save_mindset_centers(data):
    df = pd.DataFrame(
        data,
        columns=['mindset_cluster_center'],)
    df.columns.values[0] = 'mindsetID'
    df.to_csv('./nlp/literature-mindset-cluster-centers.csv')

def get_cluster_center_mindsets(text_list,semantics):
    # https://medium.com/jatana/unsupervised-text-summarization-using-sentence-embeddings-adb15ce83db1
    from sklearn.metrics import pairwise_distances_argmin_min
    import numpy as np
    
    kmeans = KMeans(n_clusters = CLUSTER_CNT)
    kmeans.fit(semantics)

    closest, _ = pairwise_distances_argmin_min(
        kmeans.cluster_centers_, 
        semantics)

    temp = [text_list[index_] for index_ in closest]

    return temp

text_list = get_text_list()
semantics = get_semantics(text_list)
mindset_cluster_centers = get_cluster_center_mindsets(text_list, semantics)
cluster_ids = get_cluster_ids(semantics)
mindsets = get_mindsets(
    cluster_ids,
    text_list)
save_mindsets(mindsets)   
save_mindset_centers(mindset_cluster_centers)
