# vsm_search_engine.py
# A simple Vector Space Model (VSM) search engine with
# tokenization, indexing, TF-IDF weighting, and query expansion.

import glob, math, os, re
from collections import defaultdict
import nltk

# Download necessary NLTK resources (only first time)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer

# Global variables and data structures
CORPUS_PATH = "corpus/*.txt"        # path to text files
STOPWORDS = set(stopwords.words("english"))  # English stopwords
stemmer = PorterStemmer()           # word stemmer

document_filenames = {}             # maps doc_id -> filename
postings = defaultdict(list)        # inverted index: term -> list of (doc_id, freq)
document_frequency = defaultdict(int) # number of documents containing a term
doc_lengths = {}                    # precomputed document vector lengths
vocabulary = set()                  # set of all terms
N = 0                               # total number of documents

# Load all documents from corpus directory
def get_corpus():
    global document_filenames, N
    files = glob.glob(CORPUS_PATH)
    document_filenames = {i: f for i, f in enumerate(files)}
    N = len(files)

# Tokenize text: clean, lowercase, remove stopwords, and stem
def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)   # remove punctuation
    text = re.sub(r"\d+", "", text)              # remove digits
    tokens = word_tokenize(text.lower())         # split into words
    return [stemmer.stem(t) for t in tokens if t.isalpha() and t not in STOPWORDS]

# Build inverted index (postings list)
def build_index():
    for doc_id, filepath in document_filenames.items():
        with open(filepath, "r", encoding='utf-8') as f:
            content = f.read()
        tokens = tokenize(content)

        # Count term frequencies in this document
        term_counts = defaultdict(int)
        for token in tokens:
            term_counts[token] += 1

        # Update postings and document frequencies
        for term, freq in term_counts.items():
            postings[term].append((doc_id, freq))
        for term in term_counts:
            document_frequency[term] += 1

        vocabulary.update(term_counts.keys())

# Precompute document vector lengths (for cosine similarity)
def compute_doc_lengths():
    for doc_id in document_filenames:
        length = 0.0
        for term in vocabulary:
            for d_id, tf in postings[term]:
                if d_id == doc_id:
                    w = 1 + math.log10(tf)  # term weight
                    length += w ** 2
        doc_lengths[doc_id] = math.sqrt(length)

# Compute inverse document frequency (IDF)
def inverse_document_frequency(term):
    if term in document_frequency and document_frequency[term] > 0:
        return math.log10(N / document_frequency[term])
    return 0.0

# Soundex algorithm for phonetic similarity (misspelling handling)
def soundex(term):
    term = term.upper()
    if not term or not term[0].isalpha():
        return ""
    mapping = {"BFPV": "1", "CGJKQSXZ": "2", "DT": "3", "L": "4", "MN": "5", "R": "6"}
    code_map = {ch: val for chars, val in mapping.items() for ch in chars}

    result = term[0]       # keep first letter
    last_code = code_map.get(result, '0')

    for char in term[1:]:
        code = code_map.get(char)
        if code and code != last_code:
            result += code
            last_code = code
        elif char not in "AEIOUHWY":
            last_code = '0'
        if len(result) == 4:
            break
    return (result + "000")[:4]  # pad/truncate to 4 chars

# Expand query terms using Soundex matches
def expand_query_with_soundex(terms):
    expanded = []
    soundex_vocab = {t: soundex(t) for t in vocabulary}
    for term in terms:
        if term in vocabulary:
            expanded.append(term)
        else:
            term_code = soundex(term)
            matches = [t for t, code in soundex_vocab.items() if code == term_code]
            expanded.extend(matches if matches else [term])
    return list(dict.fromkeys(expanded))  # remove duplicates

# Expand query with synonyms from WordNet
def expand_query_with_synonyms(terms):
    expanded = list(terms)
    for term in terms:
        for syn in wordnet.synsets(term):
            for lemma in syn.lemmas():
                name = lemma.name().replace('_', ' ').lower()
                if ' ' not in name and name != term:
                    expanded.append(name)
    return tokenize(' '.join(set(expanded)))

# Compute cosine similarity between query vector and document vector
def similarity(query_weights, doc_id):
    score = 0.0
    for term, q_w in query_weights.items():
        for d_id, tf in postings.get(term, []):
            if d_id == doc_id:
                d_w = 1 + math.log10(tf)
                score += q_w * d_w
    return score / doc_lengths[doc_id] if doc_lengths[doc_id] != 0 else 0.0

# Run a search query and print top results
def run_query(query, use_synonyms=False):
    query_terms = tokenize(query)

    # Optional synonym expansion
    if use_synonyms:
        query_terms = expand_query_with_synonyms(query_terms)

    # Expand query using soundex
    query_terms = expand_query_with_soundex(query_terms)

    # Term frequencies in query
    tf_q = defaultdict(int)
    for t in query_terms:
        tf_q[t] += 1

    # Compute query term weights (TF-IDF)
    weights = {}
    for term, tf in tf_q.items():
        if term in vocabulary:
            weights[term] = (1 + math.log10(tf)) * inverse_document_frequency(term)

    # Normalize query vector
    length = math.sqrt(sum(w**2 for w in weights.values()))
    if length != 0:
        for term in weights:
            weights[term] /= length

    # Compute similarity score for each document
    scores = [(doc_id, similarity(weights, doc_id)) for doc_id in document_filenames]

    # Get top 10 ranked documents
    top = sorted(scores, key=lambda x: (-x[1], x[0]))[:10]
    for doc_id, score in top:
        if score > 0:
            print(f"{os.path.basename(document_filenames[doc_id])} => {round(score, 5)}")

# Main program loop
if __name__ == "__main__":
    get_corpus()
    build_index()
    compute_doc_lengths()
    while True:
        query = input("\nSearch query >> ").strip()
        if not query:
            break
        use_expansion = input("Use synonym expansion? (yes/no): ").strip().lower() == 'yes'
        run_query(query, use_synonyms=use_expansion)
