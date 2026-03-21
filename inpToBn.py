import re
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SYMPTOMS = ['depression','shortnessofbreath','depressiveorpsychoticsymptoms','sharpchestpain','dizziness','abnormalinvoluntarymovements','sorethroat','cough','nasalcongestion','throatswelling','diminishedhearing','lumpinthroat','throatfeelstight','skinswelling','retentionofurine','legpain','suprapubicpain','lackofgrowth','elbowweakness','whitedischargefromeye','abusingalcohol','fainting','drugabuse','sharpabdominalpain','vomiting','headache','nausea','diarrhea','vaginalitching','painfulurination','frequenturination','lowerabdominalpain','vaginaldischarge','bloodinurine','wristpain','handorfingerswelling','armpain','lipswelling','abnormalappearingskin','skinlesion','acneorpimples','mouthulcer','diminishedvision','painineye','irregularappearingscalp','backpain','neckpain','lowbackpain','pelvicpain','vomitingblood','wheezing','peripheraledema','earpain','footortoepain','skinmoles','kneelumpormass','vaginalpain','weakness','ringinginear','pluggedfeelinginear','frontalheadache','fluidinear','spotsorcloudsinvision','eyeredness','lacrimation','itchinessofeye','blindness','lossofsensation','slurringwords','symptomsoftheface','disturbanceofmemory','sidepain','fever','acheallover','changesinstoolappearance','chills','fatigue','melena','coryza','allergicreaction','sleepiness','abnormalbreathingsounds','pullingatears','rednessinear','fluidretention','flu-likesyndrome','sinuscongestion','musclecramps,contractures,orspasms','nosebleed','swolleneye','itchingofskin','skindryness,peeling,scaliness,orroughness','skinrash','feelinghot','swollenorredtonsils','lipsore','sneezing','diaperrash','throatredness']


MODEL_NAME = "d4data/biomedical-ner-all"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)


NORMALIZER = {
    "tired": "fatigue",
    "weak": "weakness",
    "nauseous": "nausea",
    "nausea": "nausea",
    "vomiting": "vomiting",
    "throwing up": "vomiting",
    "burns": "painfulurination",
    "burning": "painfulurination",
    "blocked": "nasalcongestion",
    "congestion": "nasalcongestion",
    "runny": "coryza",
    "hot": "fever",
    "fever": "fever",
    "head pain": "headache",
    "pain": "sharpabdominalpain",
    "itchy": "itchingofskin",
    "rash": "skinrash"
}


def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    return text


def extract_biobert(text):
    entities = ner_pipeline(text)
    symptoms = []

    for ent in entities:
        if ent['entity_group'] == 'Sign_symptom':
            symptoms.append(ent['word'].lower())

    return symptoms


def normalize_symptoms(sym_list):
    mapped = set()

    for sym in sym_list:
        if sym in NORMALIZER:
            mapped.add(NORMALIZER[sym])
        elif sym in SYMPTOMS:
            mapped.add(sym)

    return mapped


vectorizer = TfidfVectorizer().fit(SYMPTOMS)

def tfidf_match(text, threshold=0.3):
    vec = vectorizer.transform([text])
    sym_vec = vectorizer.transform(SYMPTOMS)

    sim = cosine_similarity(vec, sym_vec)[0]

    return {SYMPTOMS[i] for i, s in enumerate(sim) if s > threshold}


PHRASE_MAP = {
    "body hot": "fever",
    "head pain": "headache",
    "loose motion": "diarrhea",
    "pee frequently": "frequenturination",
    "burning urination": "painfulurination",
    "runny nose": "coryza",
    "blocked nose": "nasalcongestion"
}

def rule_match(text):
    detected = set()

    for phrase, sym in PHRASE_MAP.items():
        if phrase in text:
            detected.add(sym)

    for sym in SYMPTOMS:
        if sym in text.replace(" ", ""):
            detected.add(sym)

    return detected


def statement_to_bn_input(user_input):

    clean_text = normalize_text(user_input)

    # 1. BioBERT
    bio_raw = extract_biobert(user_input)
    bio_norm = normalize_symptoms(bio_raw)

    # 2. Rule-based
    rule = rule_match(clean_text)

    # 3. TF-IDF
    tfidf = tfidf_match(clean_text)

    # FINAL FUSION
    final_symptoms = bio_norm | rule | tfidf


    bn_input = {sym: 0 for sym in SYMPTOMS}

    for sym in final_symptoms:
        if sym in bn_input:
            bn_input[sym] = 1

    return bn_input, list(final_symptoms)

# while True:
#     q = input("Ask: ")
#     if q == "exit" : break
#     bn_input, detected = statement_to_bn_input(q)

#     print("Answer:", detected)
#     print("-" * 50)


# # Define test inputs
# test_inputs = [
#     "nothing is wrong I am perfectly fine just checking system lol random words without meaning"
# ]

# print("\nRunning predefined test cases...\n")
# for inp in test_inputs:
#     bn_input, detected = statement_to_bn_input(inp)
#     print(f"Input: {inp}")
#     print("Answer:", detected)
#     print("-" * 50)