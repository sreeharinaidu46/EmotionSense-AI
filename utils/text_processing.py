import pickle

vectorizer = pickle.load(open("utils/vectorizer.pkl", "rb"))

def extract_text_features(text):
    return vectorizer.transform([text]).toarray()
