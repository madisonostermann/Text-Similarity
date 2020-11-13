import time
import re
import random
from py2neo import Graph
import pandas as pd
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim.models import Doc2Vec
import pickle
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
vocabulary = []
tagged_vocabulary = []
doc2vec_model = None

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

def jaccard_compare(names):
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
    if len(names) <= 10:
        ann = True
    else:
        ann = False
    sns.heatmap(df, annot=ann)
    plt.savefig('jaccard_comparison_some.png')
    plt.clf()
    # Record time
    end = time.perf_counter()
    print(f'TOTAL TIME FOR JACCARD COMPARISON OF SOME DESCRIPTIONS {end-start:0.4f}s')

def tf_idf_compare(names):
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
    if len(names) <= 10:
        ann = True
    else:
        ann = False
    sns.heatmap(df, annot=ann)
    plt.savefig('tf_idf_comparison_some.png')
    plt.clf()
    # Record time
    end = time.perf_counter()
    print(f'TOTAL TIME FOR TF-IDF COMPARISON OF SOME DESCRIPTIONS {end-start:0.4f}s')

###### doc2vec stuff ######

def load_or_create_vocabulary():
    global vocabulary
    filename = 'vocabulary.pk'
    try:
        with open(filename, 'rb') as fi:
            vocabulary = pickle.load(fi)
    except:
        print("COULD NOT OPEN VOCABULARY FILE, CREATING VOCABULARY INSTEAD.")
        
    if not vocabulary:
        print('GENERATING VOCABULARY')
        create_vocabulary() 
    else:
        print("USING VOCABULARY FROM vocabulary.pk")

# using nodes in the database, we're going to create our own vocabulary
def create_vocabulary():
    connect_to_database()
    global graph
    global vocabulary
    
    people = graph.run("""MATCH (n:Person) RETURN n.name AS name, n.birth AS birth, n.death AS death""")
    for person in people:
        rand = random.randint(1, 12)
        name = (person['name']).replace
        birth = person['birth']
        death = person['death']
        if birth and death:
            birth = birth.replace('"', '')
            death = death.replace('"', '')
            vocabulary.append("{}: born {}, died {}.".format(name, birth, death))
        else:
            if birth:
                birth = birth.replace('"', '')
                vocabulary.append("{} was born {}.".format(name, birth))
            if death:
                death = death.replace('"', '')
                vocabulary.append("{} died {}.".format(name, death))
        #occupation info
        occs = graph.run("""MATCH (p:Person)-[r]-(n:Occupation) WHERE p.name="{}" RETURN n.title AS title""".format(name))
        for occ in occs:
            if rand%2 == 0:
                vocabulary.append("{} worked as a {}.".format(name, occ['title']))
            else:
                vocabulary.append("She was an accomplished {}.".format(occ['title']))
        #award info
        awards = graph.run("""MATCH (p:Person)-[r]-(n:Award) WHERE p.name="{}" RETURN n.title AS title""".format(name))
        for a in awards:
            if rand%2 == 0:
                vocabulary.append("She won the {}.".format(a['title']))
            else:
                vocabulary.append("{} was awarded to {}.".format(a['title'], name))
        #nationality info
        nations = graph.run("""MATCH (p:Person)-[r]-(n:Nation) WHERE p.name="{}" RETURN n.title AS title""".format(name))
        for n in nations:
            if rand%2 == 0:
                vocabulary.append("She held citizenship in {}.".format(n['title']))
            else:
                vocabulary.append("{} was a citizen of {}.".format(name, n['title']))
        #institution info
        institutions = graph.run("""MATCH (p:Person)-[r]-(i:Institution) WHERE p.name="{}" RETURN i.title AS title""".format(name))
        for i in institutions:
            if rand%3 == 0:
                vocabulary.append("{} recieved an education from {}.".format(name, i['title']))
            elif rand%3 == 1:
                vocabulary.append("She attended {}.".format(i['title']))
            elif rand%3 == 2:
                vocabulary.append("{} was the institution at which {} studied.".format(i['title'], name))

    filename = 'vocabulary.pk'
    with open(filename, 'wb') as fi:
        pickle.dump(vocabulary, fi)
    print("CREATED PICKLE FILE WITH VOCABULARY.")

def load_or_tag_tagged_vocabulary():
    global tagged_vocabulary
    filename = 'tagged_vocabulary.pk'
    try:
        with open(filename, 'rb') as fi:
            tagged_vocabulary = pickle.load(fi)
    except:
        print("COULD NOT OPEN TAGGED VOCABULARY FILE, TAGGING THE VOCABULARY NOW.")
    if not tagged_vocabulary:
        print('TAGGING VOCABULARY')
        tag_vocabulary() 
    else:
        print("USING TAGGED VOCABULARY FROM tagged_vocabulary.pk")

def tag_vocabulary():
    global vocabulary #should already be cleaned
    global tagged_vocabulary
    for index, vocab in enumerate(vocabulary):
        tagged_vocabulary.append(gensim.models.doc2vec.TaggedDocument(vocab, [index]))
    filename = 'tagged_vocabulary.pk'
    with open(filename, 'wb') as fi:
        pickle.dump(tagged_vocabulary, fi)
    print("CREATED PICKLE FILE WITH TAGGED VOCABULARY.")

def load_or_create_doc2vec_model():
    global doc2vec_model
    try:
        doc2vec_model = Doc2Vec.load("doc2vec.model")
    except:
        print("COULD NOT LOAD DOC2VEC MODEL, BUILDING ONE INSTEAD.")
    if not doc2vec_model:
        print("BUILDING DOC2VEC MODEL")
        create_doc2vec_model()
    else:
        print("USING EXISITNG DOC2VEC MODEL")

def create_doc2vec_model():
    global doc2vec_model
    global tagged_vocabulary
    doc2vec_model = Doc2Vec(dm=0, vector_size=200, min_count=2, epochs=100, window=4, dbow_word=1)
    print("BUILDING DOC2VEC_MODEL VOCAB")
    doc2vec_model.build_vocab(tagged_vocabulary)
    print("TRAINING DOC2VEC_MODEL")
    doc2vec_model.train(tagged_vocabulary, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)
    print("SAVING DOC2VEC_MODEL")
    doc2vec_model.save("doc2vec.model")

def doc2vec_compare(names):
    global doc2vec_model
    start = time.perf_counter()
    # Find similarities between two descriptions at a time
    similarities = []
    for name1 in names:
        sims = []
        results = graph.run("""MATCH (p:Person) WHERE p.name='{}' RETURN p.description AS description""".format(name1)).data()
        desc1_clean = clean(results[0]['description'])
        desc1_vector = doc2vec_model.infer_vector(desc1_clean) 
        for name2 in names:
            results = graph.run("""MATCH (p:Person) WHERE p.name='{}' RETURN p.description AS description""".format(name2)).data()
            desc2_clean = clean(results[0]['description'])
            desc2_vector = doc2vec_model.infer_vector(desc2_clean)
            sims.append(1 - spatial.distance.cosine(desc1_vector, desc2_vector))
        similarities.append(sims)
    # Create dataframe & save to csv
    df = pd.DataFrame(similarities, index=names, columns=names)
    df.to_csv('doc2vec_comparison_some.csv')
    # Create & save heatmap
    if len(names) <= 10:
        ann = True
    else:
        ann = False
    sns.heatmap(df, annot=ann)
    plt.savefig('doc2vec_comparison_some.png')
    plt.clf()
    # Record time
    end = time.perf_counter()
    print(f'TOTAL TIME FOR DOC2VEC COMPARISON OF SOME DESCRIPTIONS {end-start:0.4f}s')

if __name__ == "__main__":
    connect_to_database()

    short_list = ['Ada Lovelace', 'Grace Hopper', 'Marie Curie', 'Katherine Johnson', 'Rosalind Franklin', 'Sally Ride', 'Julia R. Burdge', 'Elisabeth M. Werner']
    jaccard_compare(short_list)
    tf_idf_compare(short_list)

    load_or_create_vocabulary()
    load_or_tag_tagged_vocabulary()
    load_or_create_doc2vec_model()

    doc2vec_compare(short_list)
    
    a = graph.run("""MATCH (p:Person)-[:works_as]-(o:Occupation) WHERE o.title='biologist' AND EXISTS (p.description) RETURN p.name AS name LIMIT 15""").to_data_frame()
    b = graph.run("""MATCH (p:Person)-[:works_as]-(o:Occupation) WHERE o.title='mathematician' AND EXISTS (p.description) RETURN p.name AS name LIMIT 15""").to_data_frame()
    people = list(a['name'])+list(b['name'])
    jaccard_compare(people)
    tf_idf_compare(people)
    doc2vec_compare(people)
