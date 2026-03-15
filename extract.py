import spacy
nlp = spacy.load("pmaitra/en_biobert_ner_symptom")

text = "The patient reported fever, cough, and shortness of breath."
doc = nlp(text)

for ent in doc.ents:
    print(ent.text, ent.label_)