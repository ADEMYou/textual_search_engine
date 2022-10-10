# Textual Search Engine


A text search engine that allows a user to retrieve text documents within a collection by submitting a natural language query (phrase or keyword). The search engine then crawls all or part of the indexed document collection to return to the user those documents that are most similar to the query.

To do this, the search engine operates in two main steps: indexing and searching. The indexing step is done once and for all, in the following way:

1. An indexing vocabulary, consisting of the most interesting words appearing in the documents (called indexing terms) is built up.
2. The documents are represented in the form of a descriptor, a bag of words, which can be interpreted, depending on the model chosen, as a set of indexing terms (set and boolean models), a vector of weights or probabilities (vector and probabilistic models). .
3. The organization of descriptors into a data structure called an index, which allows the rapid retrieval of documents most likely to be relevant to a given query.

The search stage, performed each time a query is submitted to the system, consists of the following steps:

1. Calculation of the descriptor of the query.
2. Obtaining candidate documents from the index.
3. Compute the similarity scores between the candidate documents and the query, according to the chosen model; sort the documents by decreasing similarity score (from the most similar to the least similar).
4. Display the results.
