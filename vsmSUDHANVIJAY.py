# vsm_search_engine.py

import glob, math, os, re
from collections import defaultdict
import nltk

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer

# Global data
CORPUS_PATH = "corpus/*.txt"
STOPWORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()

document_filenames = {}
postings = defaultdict(list)
document_frequency = defaultdict(int)
doc_lengths = {}
vocabulary = set()
N = 0

def get_corpus():
    global document_filenames, N
    files = glob.glob(CORPUS_PATH)
    document_filenames = {i: f for i, f in enumerate(files)}
    N = len(files)

def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\d+", "", text)
    tokens = word_tokenize(text.lower())
    return [stemmer.stem(t) for t in tokens if t.isalpha() and t not in STOPWORDS]

def build_index():
    for doc_id, filepath in document_filenames.items():
        with open(filepath, "r", encoding='utf-8') as f:
            content = f.read()
        tokens = tokenize(content)
        term_counts = defaultdict(int)
        for token in tokens:
            term_counts[token] += 1
        for term, freq in term_counts.items():
            postings[term].append((doc_id, freq))
        for term in term_counts:
            document_frequency[term] += 1
        vocabulary.update(term_counts.keys())

def compute_doc_lengths():
    for doc_id in document_filenames:
        length = 0.0
        for term in vocabulary:
            for d_id, tf in postings[term]:
                if d_id == doc_id:
                    w = 1 + math.log10(tf)
                    length += w ** 2
        doc_lengths[doc_id] = math.sqrt(length)

def inverse_document_frequency(term):
    if term in document_frequency and document_frequency[term] > 0:
        return math.log10(N / document_frequency[term])
    return 0.0

def soundex(term):
    term = term.upper()
    if not term or not term[0].isalpha():
        return ""
    mapping = {"BFPV": "1", "CGJKQSXZ": "2", "DT": "3", "L": "4", "MN": "5", "R": "6"}
    code_map = {ch: val for chars, val in mapping.items() for ch in chars}
    result = term[0]
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
    return (result + "000")[:4]

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
    return list(dict.fromkeys(expanded))

def expand_query_with_synonyms(terms):
    expanded = list(terms)
    for term in terms:
        for syn in wordnet.synsets(term):
            for lemma in syn.lemmas():
                name = lemma.name().replace('_', ' ').lower()
                if ' ' not in name and name != term:
                    expanded.append(name)
    return tokenize(' '.join(set(expanded)))

def similarity(query_weights, doc_id):
    score = 0.0
    for term, q_w in query_weights.items():
        for d_id, tf in postings.get(term, []):
            if d_id == doc_id:
                d_w = 1 + math.log10(tf)
                score += q_w * d_w
    return score / doc_lengths[doc_id] if doc_lengths[doc_id] != 0 else 0.0

def run_query(query, use_synonyms=False):
    query_terms = tokenize(query)
    if use_synonyms:
        query_terms = expand_query_with_synonyms(query_terms)
    query_terms = expand_query_with_soundex(query_terms)

    tf_q = defaultdict(int)
    for t in query_terms:
        tf_q[t] += 1

    weights = {}
    for term, tf in tf_q.items():
        if term in vocabulary:
            weights[term] = (1 + math.log10(tf)) * inverse_document_frequency(term)

    length = math.sqrt(sum(w**2 for w in weights.values()))
    if length != 0:
        for term in weights:
            weights[term] /= length

    scores = [(doc_id, similarity(weights, doc_id)) for doc_id in document_filenames]
    top = sorted(scores, key=lambda x: (-x[1], x[0]))[:10]
    for doc_id, score in top:
        if score > 0:
            print(f"{os.path.basename(document_filenames[doc_id])} => {round(score, 5)}")

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
