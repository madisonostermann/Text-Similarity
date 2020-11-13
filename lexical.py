from py2neo import Graph
from nltk.stem import WordNetLemmatizer 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import time
import re
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
import matplotlib.pyplot as plt
import seaborn as sns

# make sure you have all of these imports
# will also need to do
    # python
    # import nltk
    # nltk.download('stopwords')
    # nltk.download('punkt')
    # nltk.download('wordnet')

graph = None

def connect_to_database():
    global graph
    print("Neo4j DB Port: ")
    port = input()
    print("Neo4j DB Username: ")
    user = input()
    print("Neo4j DB Password: ")
    pswd = input()

    # Make sure the database is started first, otherwise attempt to connect will fail
    try:
        graph = Graph('bolt://localhost:'+port, auth=(user, pswd))
        print('SUCCESS: Connected to the Neo4j Database.')
    except Exception as e:
        print('ERROR: Could not connect to the Neo4j Database. See console for details.')
        raise SystemExit(e)

# Remove stop words, tokenize & lemmatize the descriptions -> return an array of words
def clean(string):
    stops = set(stopwords.words("english"))
    token = word_tokenize(string.lower()) #tokenize
    lemmatizer = WordNetLemmatizer() 
    words = []
    for w in token:
        if not w in stops: #don't include stop words
            w = (re.sub('[^A-Za-z0-9]+', '', w).lower()).strip() #remove punc & special chars
            lemmatizer.lemmatize(w)
            if w:
                words.append(w)
    return words

def jaccard(desc1, desc2):
    desc1 = set(desc1)
    desc2 = set(desc2)
    overlap = desc1.intersection(desc2)
    return float(len(overlap)) / (len(desc1) + len(desc2) - len(overlap))

def jaccard_compare_all():
    global graph
    start = time.perf_counter()
    # Get the names & descriptions for people who have descriptions
    results = graph.run("""MATCH (p:Person) WHERE EXISTS (p.description) RETURN p.name AS name, p.description AS description""").data()
    names = []
    descriptions = []
    similarities = []
    for result in results:
        names.append(result['name'])
        descriptions.append(result['description'])
    # Find similarities between two descriptions at a time
    for desc1 in descriptions:
        sims = []
        desc1_clean = clean(desc1)
        for desc2 in descriptions:
            desc2_clean = clean(desc2)
            sims.append(jaccard(desc1_clean, desc2_clean))
        similarities.append(sims)
    # Create dataframe & save to csv
    df = pd.DataFrame(similarities, index=names, columns=names)
    df.to_csv('jaccard_comparison_all.csv')
    # Create & save heatmap
    sns.heatmap(df, annot=True)
    plt.savefig('jaccard_comparison_all.png')
    plt.clf()
    # Record time
    end = time.perf_counter()
    print(f'TOTAL TIME FOR JACCARD COMPARISON OF ALL DESCRIPTIONS {end-start:0.4f}s')

def jaccard_compare_some(names):
    global graph
    start = time.perf_counter()
    # Find similarities between two descriptions at a time
    similarities = []
    for name1 in names:
        sims = []
        results = graph.run("""MATCH (p:Person) WHERE p.name='{}' RETURN p.description AS description""".format(name1)).data()
        desc1_clean = clean(results[0]['description'])
        for name2 in names:
            results = graph.run("""MATCH (p:Person) WHERE p.name='{}' RETURN p.description AS description""".format(name2)).data()
            desc2_clean = clean(results[0]['description'])
            sims.append(jaccard(desc1_clean, desc2_clean))
        similarities.append(sims)
    # Create dataframe & save to csv
    df = pd.DataFrame(similarities, index=names, columns=names)
    df.to_csv('jaccard_comparison_some.csv')
    # Create & save heatmap
    sns.heatmap(df, annot=True)
    plt.savefig('jaccard_comparison_some.png')
    plt.clf()
    # Record time
    end = time.perf_counter()
    print(f'TOTAL TIME FOR JACCARD COMPARISON OF SOME DESCRIPTIONS {end-start:0.4f}s')

def tf_idf_compare_all():
    global graph
    start = time.perf_counter()
    # Get all names & descriptions for people with descriptions
    results = graph.run("""MATCH (p:Person) WHERE EXISTS (p.description) RETURN p.name AS name, p.description AS description""").data()
    names = []
    descriptions = []
    similarities = []
    for result in results:
        names.append(result['name'])
        descriptions.append(result['description'])
    # Find similarities between two descriptions at a time
    for desc1 in descriptions:
        sims = []
        desc1_clean = clean(desc1)
        for desc2 in descriptions:
            desc2_clean = clean(desc2)
            vect = TfidfVectorizer()
            tfidf_matrix = vect.fit_transform([" ".join(desc1_clean), " ".join(desc2_clean)])
            sims.append(1 - spatial.distance.cosine(tfidf_matrix.toarray()[0], tfidf_matrix.toarray()[1]))
        similarities.append(sims)
    # Create dataframe & save to csv
    df = pd.DataFrame(similarities, index=names, columns=names)
    df.to_csv('tf_idf_comparison_all.csv')
    # Create & save heatmap
    sns.heatmap(df, annot=True)
    plt.savefig('tf_idf_comparison_all.png')
    plt.clf()
    # Record time
    end = time.perf_counter()
    print(f'TOTAL TIME FOR TF-IDF COMPARISON OF ALL DESCRIPTIONS {end-start:0.4f}s')

def tf_idf_compare_some(names):
    global graph
    start = time.perf_counter()
    # Find similarities between two descriptions at a time
    similarities = []
    for name1 in names:
        sims = []
        results = graph.run("""MATCH (p:Person) WHERE p.name='{}' RETURN p.description AS description""".format(name1)).data()
        desc1_clean = clean(results[0]['description'])
        for name2 in names:
            results = graph.run("""MATCH (p:Person) WHERE p.name='{}' RETURN p.description AS description""".format(name2)).data()
            desc2_clean = clean(results[0]['description'])
            vect = TfidfVectorizer()
            tfidf_matrix = vect.fit_transform([" ".join(desc1_clean), " ".join(desc2_clean)])
            sims.append(1 - spatial.distance.cosine(tfidf_matrix.toarray()[0], tfidf_matrix.toarray()[1]))
        similarities.append(sims)
    # Create dataframe & save to csv
    df = pd.DataFrame(similarities, index=names, columns=names)
    df.to_csv('tf_idf_comparison_some.csv')
    # Create & save heatmap
    sns.heatmap(df, annot=True)
    plt.savefig('tf_idf_comparison_some.png')
    plt.clf()
    # Record time
    end = time.perf_counter()
    print(f'TOTAL TIME FOR TF-IDF COMPARISON OF SOME DESCRIPTIONS {end-start:0.4f}s')


if __name__ == "__main__":
    connect_to_database()

    jaccard_compare_some(['Ada Lovelace', 'Grace Hopper', 'Marie Curie', 'Katherine Johnson', 'Rosalind Franklin', 'Sally Ride', 'Julia R. Burdge', 'Elisabeth M. Werner'])
    jaccard_compare_all()

    tf_idf_compare_some(['Ada Lovelace', 'Grace Hopper', 'Marie Curie', 'Katherine Johnson', 'Rosalind Franklin', 'Sally Ride', 'Julia R. Burdge', 'Elisabeth M. Werner'])
    tf_idf_compare_all()