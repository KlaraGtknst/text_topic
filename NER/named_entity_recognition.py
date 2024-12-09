import spacy

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("Apple is a big company in the USA.")
    for ent in doc.ents:
        print(ent.text, ent.label_)