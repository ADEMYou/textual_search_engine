import numpy as np
import matplotlib.pyplot as plt
import json
import time

from preprocessor import Preprocessor
from set_model import SetModel
from vec_model import VecModel
from src.ir_evaluation import Evaluator

''' The file where we perform evaluations of SRI '''

#The path to the folder which contains the datasets
path_to_data = 'Data/datasets/'

#Function that computes the relevant docs for several queries in the right format
def multiple_query_research(collection_name, ir_system, is_inverted_index = False):
    results = {'groundtruth': []}
    with open(f'{path_to_data + collection_name}_dataset.json') as f:
        dataset = json.load(f)
        for i in range(len(dataset['dataset'])):
            dataset['dataset'][i]['text'] = dataset['dataset'][i]['text'].replace('\n', '')

    with open(f'{path_to_data + collection_name}_queries.json') as f:
        queries = json.load(f)

    index_start = time.time()
    if is_inverted_index:
        index = ir_system.create_inverted_index(dataset['dataset'])
    else:
        index = ir_system.create_index(dataset['dataset'])
    index_stop = time.time()
    index_duration = index_stop - index_start

    research_duration = []
    for q in queries['queries']:
        research_start = time.time()
        if is_inverted_index:
            if 'Vec' in str(type(ir_system)):
                relevant_docs = ir_system.inverted_research(index, dataset['dataset'], q)
            else:
                relevant_docs = ir_system.inverted_research(index, q)
        else:
            if 'Vec' in str(type(ir_system)):
                relevant_docs = ir_system.research(index, dataset['dataset'], q)
            else:
                relevant_docs = ir_system.research(index, q)
        research_stop = time.time()
        q_research_duration = research_stop - research_start
        research_duration.append(q_research_duration)
        q_results = {'id': q['id'], 'relevant' : relevant_docs}
        results['groundtruth'].append(q_results)
    return results, index_duration, np.mean(research_duration)

#Function that returns the metrics of a SRI, given a collection and a set of queries (and grountruths as well)
def compute_metrics(collection_name, ir_system, is_inverted_index = False):
    retrieved = multiple_query_research(collection_name, ir_system, is_inverted_index)
    with open(f'{path_to_data + collection_name}_groundtruth.json') as f:
        groundtruth = json.load(f)
    evaluator = Evaluator(retrieved[0], groundtruth)
    pr_points = evaluator.evaluate_pr_points()
    recall_points, precision_points = [pr_points[i][0] for i in range(len(pr_points))], [pr_points[i][1] for i in range(len(pr_points))]
    return recall_points, precision_points, evaluator.evaluate_map(), retrieved[1], retrieved[2]


#Define the stop-list

stop_list_file =  open('Data/stoplist/stoplist-english.txt', 'r')
stop_list = stop_list_file.readlines()
#Delete whitespace characters
stop_list = [x.strip() for x in stop_list]

#Define the preprocessor (for the computation of bag of words)
preprocessor = Preprocessor([' ', ',', '.', "'"], stop_list)

#SET VS VEC

'''set_metrics = compute_metrics('med', SetModel(preprocessor), is_inverted_index = True)
vec_metrics = compute_metrics('med', VecModel(preprocessor, is_idf = True, is_norm = True), is_inverted_index = True)
plt.figure(figsize=(12,8))
plt.plot(set_metrics[0], set_metrics[1], c = 'blue', label = 'Set')
plt.plot(vec_metrics[0], vec_metrics[1], c = 'orange', label = 'Vectorial')
plt.legend()
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Set vs vectorial')
plt.show()
print(f'\nmAP for set: {set_metrics[2]}')
print(f'Index duration set : {set_metrics[3]}')
print(f'Search duration set: {set_metrics[4]}')
print(f'\nmAP for vec : {vec_metrics[2]}')
print(f'Index duration vec: {vec_metrics[3]}')
print(f'Search duration vec : {vec_metrics[4]}')'''

#IDF VS NO IDF

'''idf_metrics = compute_metrics('med', VecModel(preprocessor, is_idf = True, is_norm = True), is_inverted_index = True)
no_idf_metrics = compute_metrics('med', VecModel(preprocessor, is_idf = False, is_norm = True), is_inverted_index = True)

plt.figure(figsize=(12,8))
plt.plot(idf_metrics[0], idf_metrics[1], c = 'blue', label = 'tf.idf')
plt.plot(no_idf_metrics[0], no_idf_metrics[1], c = 'orange', label = 'tf')
plt.legend()
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('tf vs tf.idf ')
plt.show()
print(f'\nmAP for idf: {idf_metrics[2]}')
print(f'Index duration idf : {idf_metrics[3]}')
print(f'Search duration idf : {idf_metrics[4]}')
print(f'\nmAP for no idf : {no_idf_metrics[2]}')
print(f'Index duration no idf: {no_idf_metrics[3]}')
print(f'Search duration no idf : {no_idf_metrics[4]}')'''

#Normalization vs no normalization

'''norm_metrics = compute_metrics('med', VecModel(preprocessor, is_idf = True, is_norm = True), is_inverted_index = True)
no_norm_metrics = compute_metrics('med', VecModel(preprocessor, is_idf = True, is_norm = False), is_inverted_index = True)

plt.figure(figsize=(12,8))
plt.plot(norm_metrics[0], norm_metrics[1], c = 'blue', label = 'Norm')
plt.plot(no_norm_metrics[0], no_norm_metrics[1], c = 'orange', label = 'No norm')
plt.legend()
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Norm vs no norm')
plt.show()
print(f'\nmAP for norm : {norm_metrics[2]}')
print(f'Index duration norm : {norm_metrics[3]}')
print(f'Search duration norm : {norm_metrics[4]}')
print(f'\nmAP for no norm : {no_norm_metrics[2]}')
print(f'Index duration no norm: {no_norm_metrics[3]}')
print(f'Search duration no norm : {no_norm_metrics[4]}')'''

#Inverted index vs linear index

'''inverted_metrics = compute_metrics('med', VecModel(preprocessor, is_idf = True, is_norm = True), is_inverted_index = True)
linear_metrics = compute_metrics('med', VecModel(preprocessor, is_idf = True, is_norm = True), is_inverted_index = False)

plt.figure(figsize=(12,8))
plt.plot(inverted_metrics[0], inverted_metrics[1], c = 'blue', label = 'Inverted index')
plt.plot(linear_metrics[0], linear_metrics[1], c = 'orange', label = 'Linear index')
plt.legend()
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Inverted index vs linear index')
plt.show()
print(f'\nmAP for inverted index : {inverted_metrics[2]}')
print(f'Index duration inverted index : {inverted_metrics[3]}')
print(f'Search duration inverted: {inverted_metrics[4]}')
print(f'\nmAP for linear index : {linear_metrics[2]}')
print(f'Index duration linear index: {linear_metrics[3]}')
print(f'Search duration linear index : {linear_metrics[4]}')'''

# Influence of preprocessing

'''keep_stop_words_preprocessor = Preprocessor([' ', ',', '.', "'"], stop_list, is_stop_words = False, is_stemming = True)
no_stemming_preprocessor = Preprocessor([' ', ',', '.', "'"], stop_list, is_stop_words = True, is_stemming = False)
no_stemming_with_stop_words = Preprocessor([' ', ',', '.', "'"], stop_list, is_stop_words = False, is_stemming = False)

keep_stop_words_metrics = compute_metrics('med', VecModel(keep_stop_words_preprocessor, is_idf = True, is_norm = True), is_inverted_index = True)
no_stemming_metrics = compute_metrics('med', VecModel(no_stemming_preprocessor, is_idf = True, is_norm = True), is_inverted_index = True)
no_stemming_with_stop_words_metrics = compute_metrics('med', VecModel(no_stemming_with_stop_words, is_idf = True, is_norm = True), is_inverted_index = True)
preprocessing_metrics = compute_metrics('med', VecModel(preprocessor, is_idf = True, is_norm = True), is_inverted_index = True)


plt.figure(figsize=(12,8))
plt.plot(keep_stop_words_metrics[0], keep_stop_words_metrics[1], c = 'blue', label = 'Stemming but no deletion of stop words')
plt.plot(no_stemming_metrics[0], no_stemming_metrics[1], c = 'orange', label = 'Deletion of stop-words but no stemming')
plt.plot(no_stemming_with_stop_words_metrics[0], no_stemming_with_stop_words_metrics[1], c = 'r', label = 'No stemming and no deletion of stop words')
plt.plot(preprocessing_metrics[0], preprocessing_metrics[1], c = 'green', label = 'Stemming + deletion of stop words')

plt.legend()
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Influence of preprocessing')
plt.show()

print(f'\nmAP for SRI with stop words : {keep_stop_words_metrics[2]}')
print(f'Index duration for SRI with stop words : {keep_stop_words_metrics[3]}')
print(f'Search duration for SRI with stop words : {keep_stop_words_metrics[4]}')

print(f'\nmAP for SRI without stemming : {no_stemming_metrics[2]}')
print(f'Index duration for SRI without stemming : {no_stemming_metrics[3]}')
print(f'Search duration for SRI without stemming : {no_stemming_metrics[4]}')

print(f'\nmAP for SRI without stemming and with stop words : {no_stemming_with_stop_words_metrics[2]}')
print(f'Index duration for SRI without stemming and with stop words  : {no_stemming_with_stop_words_metrics[3]}')
print(f'Search duration for SRI without stemming and with stop words  : {no_stemming_with_stop_words_metrics[4]}')

print(f'\nmAP for SRI with all preprocessing: {preprocessing_metrics[2]}')
print(f'Index duration for SRI with all preprocessing : {preprocessing_metrics[3]}')
print(f'Search duration for SRI with all preprocessing : {preprocessing_metrics[4]}')'''