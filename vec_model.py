import numpy as np

''' Class for the implementation of a SRI based on a vectorial model.

	The class allows for the implementation of:
	* a linear index and research
	* an inverted index with its corresponding research 
	* idf ponderations (optional : depending on the attribute is_idf)
	* normalization (optional : depending on the attribute is_norm)
	'''

class VecModel:

    def __init__(self, preprocessor, is_idf = True, is_norm = True):
        self.preprocessor = preprocessor
        self.is_idf = is_idf
        self.is_norm = is_norm

    #Compute once and for all the bag of words of the collection
    @staticmethod
    def get_bows(collection, preprocessor):
        bows = {}
        for doc in collection:
            bow = preprocessor.get_bow(doc['text'])
            bows[doc['id']] = bow
        return bows

    @staticmethod
    def vocabulary(collection, bows):
        voc = []
        for doc in collection:
            bow = bows[doc['id']]
            for term in bow.keys():
                if term not in voc:
                    voc.append(term)
        return voc

    @staticmethod
    def idf(collection, term, bows):
        df = 0
        for doc in collection:
            bow = bows[doc['id']]
            if term in bow.keys():
                df += 1
        idf = np.log(len(collection)/df)
        return idf

    #Compute the vectorial descriptor of a doc
    @staticmethod
    def tf_idf_descriptor(doc, collection, voc, bows, preprocessor, is_idf = True, is_norm = True, is_query = False):
        if is_query:
           bow = preprocessor.get_bow(doc['text'])
        else:
            bow = bows[doc['id']]
        desc = np.zeros((len(voc), 1))
        for term in bow.keys():
            if term in voc:
                index = voc.index(term)
                desc[index] = bow[term]
                if is_idf:
                    desc[index] *= VecModel.idf(collection, term, bows)
        if is_norm:
            desc /= np.linalg.norm(desc, ord = 2)
        return desc

    @staticmethod
    def cos_similarity(x, y):
        return float(np.dot(x.T, y)/(np.linalg.norm(x)*np.linalg.norm(y)))

    # Return index, voc and bows as well to save calculations in search function
    def create_index(self, collection):
        index = {}
        bows = VecModel.get_bows(collection, self.preprocessor)
        voc = VecModel.vocabulary(collection, bows)
        for doc in collection:
            desc = VecModel.tf_idf_descriptor(doc, collection, voc, bows, self.preprocessor, self.is_idf, self.is_norm)
            index[doc['id']] = desc
        return index, voc, bows

    def research(self, index, collection, q):
        q_desc = VecModel.tf_idf_descriptor(q, collection, index[1], index[2], self.preprocessor, self.is_idf, self.is_norm, is_query=True)
        similarities = {}
        for doc_id, doc_desc in index[0].items():
            similarity = VecModel.cos_similarity(doc_desc, q_desc)
            similarities[doc_id] = similarity
        sorted_index = np.argsort(list(similarities.values()))
        results = [list(similarities.keys())[i] for i in sorted_index]
        results.reverse()
        return results

    def create_and_search(self, collection, q):
        index = VecModel.create_index(collection)
        return VecModel.research(index, collection, q)

    # Return index, voc and bows as well to save calculations in search function
    def create_inverted_index(self, collection):
        inverted_index = {}
        bows = VecModel.get_bows(collection, self.preprocessor)
        voc = VecModel.vocabulary(collection, bows)
        for doc in collection:
            desc = VecModel.tf_idf_descriptor(doc, collection, voc, bows, self.preprocessor, self.is_idf, self.is_norm)
            for term, tf in zip(voc, desc):
                if tf != 0:
                    if term in inverted_index.keys():
                        inverted_index[term].append((doc['id'], desc))
                    else:
                        inverted_index[term] = [(doc['id'], desc)]
        return inverted_index, voc, bows

    def inverted_research(self, inverted_index, collection, q):
        q_desc = VecModel.tf_idf_descriptor(q, collection, inverted_index[1], inverted_index[2], self.preprocessor, self.is_idf, self.is_norm, is_query=True)
        q_bow = self.preprocessor.get_bow(q['text'])
        short_list = []
        for term in q_bow:
            if term in inverted_index[1]:
                for doc in inverted_index[0][term]:
                    if doc not in short_list:
                        short_list.append(doc)
        similarities = {}
        for doc_id, descriptor in short_list:
            similarity = VecModel.cos_similarity(descriptor, q_desc)
            similarities[doc_id] = similarity
            print(similarity)
        sorted_index = np.argsort(list(similarities.values()))
        results = [list(similarities.keys())[i] for i in sorted_index]
        results.reverse()
        return results

    def create_and_search_inverted(self, collection, q):
        inverted_index = VecModel.create_inverted_index(collection)
        return VecModel.inverted_research(inverted_index, collection, q)


