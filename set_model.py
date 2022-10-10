import numpy as np

''' Class for the implementation of a SRI based on a set model.

	The class allows for the implementation of:
	* a linear index and research
	* an inverted index with its corresponding research '''

class SetModel:

    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    @staticmethod
    def descriptor(bow):
        return set(bow.keys())

    @staticmethod
    def dice_similarity(set_1, set_2):
        intersection = 0
        for term in set_1:
            if term in set_2:
                intersection += 1
        return (2*intersection/(len(set_1) + len(set_2)))

    # Index -> dictionary
    def create_index(self, collection):
        index = {}
        for doc in collection:
            bow = self.preprocessor.get_bow(doc['text'])
            desc = SetModel.descriptor(bow)
            index[doc['id']] = desc
        return index

    def research(self, index, q):
        q_bow = self.preprocessor.get_bow(q['text'])
        q_desc = SetModel.descriptor(q_bow)
        similarities = {}
        for doc_id, doc_desc in index.items():
            similarity = SetModel.dice_similarity(doc_desc, q_desc)
            similarities[doc_id] = similarity
        sorted_index = np.argsort(list(similarities.values()))
        results = [list(similarities.keys())[i] for i in sorted_index]
        results.reverse()
        return results

    def create_and_search(self, collection, q):
        index = SetModel.create_index(collection)
        return SetModel.research(index, q['text'])

    #Inverted index

    def create_inverted_index(self, collection):
        inverted_index = {}
        for doc in collection:
            #Compute descriptors for each document in the corpus
            bow = self.preprocessor.get_bow(doc['text'])
            desc = SetModel.descriptor(bow)
            for term in desc:
                if term in inverted_index.keys():
                    inverted_index[term].append((doc['id'], desc))
                else:
                    inverted_index[term] = [(doc['id'], desc)]
        return inverted_index


    def inverted_research(self, inverted_index, q):
        q_bow = self.preprocessor.get_bow(q['text'])
        q_desc = SetModel.descriptor(q_bow)
        short_list = []
        for term in q_desc:
            if term in inverted_index.keys():
                for doc in inverted_index[term]:
                    if doc not in short_list:
                        short_list.append(doc)
        similarities = {}
        for doc_id, desc in short_list:
            similarity = SetModel.dice_similarity(desc, q_desc)
            similarities[doc_id] = similarity
        sorted_index = np.argsort(list(similarities.values()))
        results = [list(similarities.keys())[i] for i in sorted_index]
        results.reverse()
        return results

    def create_and_search_inverted(self, collection, q):
        inverted_index = SetModel.create_inverted_index(collection)
        return SetModel.inverted_research(inverted_index, q['text'])