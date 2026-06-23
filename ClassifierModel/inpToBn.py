import re
import torch
from sentence_transformers import SentenceTransformer, util

# ==========================================
# 1. CONFIGURATION & BAYESIAN VARIABLES
# ==========================================
# This must exactly match the variables in your Bayesian Network
SYMPTOMS = [
    "depression", "shortnessofbreath", "depressiveorpsychoticsymptoms",
    "sharpchestpain", "dizziness", "abnormalinvoluntarymovements",
    "sorethroat", "cough", "nasalcongestion", "throatswelling",
    "diminishedhearing", "lumpinthroat", "throatfeelstight",
    "skinswelling", "retentionofurine", "legpain", "suprapubicpain",
    "lackofgrowth", "elbowweakness", "whitedischargefromeye",
    "abusingalcohol", "fainting", "drugabuse", "sharpabdominalpain",
    "vomiting", "headache", "nausea", "diarrhea", "vaginalitching",
    "painfulurination", "frequenturination", "lowerabdominalpain",
    "vaginaldischarge", "bloodinurine", "wristpain", "handorfingerswelling",
    "armpain", "lipswelling", "abnormalappearingskin", "skinlesion",
    "acneorpimples", "mouthulcer", "diminishedvision", "painineye",
    "irregularappearingscalp", "backpain", "neckpain", "lowbackpain",
    "pelvicpain", "vomitingblood", "wheezing", "peripheraledema",
    "earpain", "footortoepain", "skinmoles", "kneelumpormass",
    "vaginalpain", "weakness", "ringinginear", "pluggedfeelinginear",
    "frontalheadache", "fluidinear", "spotsorcloudsinvision",
    "eyeredness", "lacrimation", "itchinessofeye", "blindness",
    "lossofsensation", "slurringwords", "symptomsoftheface",
    "disturbanceofmemory", "sidepain", "fever", "acheallover",
    "changesinstoolappearance", "chills", "fatigue", "melena",
    "coryza", "allergicreaction", "sleepiness", "abnormalbreathingsounds",
    "pullingatears", "rednessinear", "fluidretention", "flu-likesyndrome",
    "sinuscongestion", "musclecramps,contractures,orspasms", "nosebleed",
    "swolleneye", "itchingofskin", "skindryness,peeling,scaliness,orroughness",
    "skinrash", "feelinghot", "swollenorredtonsils", "lipsore", "sneezing",
    "diaperrash", "throatredness"
]

# Load the local semantic model (downloads once, runs offline)
# all-MiniLM-L6-v2 is specifically optimized for fast, accurate sentence comparison
print("Loading Semantic Engine...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Pre-compute the vectors for all symptoms to make inference instant
symptom_embeddings = embedder.encode(SYMPTOMS, convert_to_tensor=True)

# ==========================================
# 2. CORE LOGIC
# ==========================================
def split_into_clauses(text):
    """
    Splits a user's input into logical clauses so symptoms and 
    negations don't bleed into each other.
    """
    # Split by punctuation and conjunctions
    delimiters = r',|\.|;|!|\?|\band\b|\bbut\b|\bor\b|\bexcept\b'
    clauses = re.split(delimiters, text.lower())
    # Return cleaned clauses that actually contain words
    return [c.strip() for c in clauses if len(c.strip()) > 2]

def is_negated(clause):
    """
    Checks if a specific clause contains negation keywords.
    """
    negation_patterns = r'\b(no|not|none|without|deny|denies|denying|zero|never|hardly|lack)\b'
    return bool(re.search(negation_patterns, clause))

def statement_to_bn_input(user_input, threshold=0.55):
    """
    Takes raw user text, filters negations, matches semantics, 
    and returns the exact dictionary format required by the Bayesian Network.
    """
    # Initialize the Bayesian Network input vector
    bn_input = {sym: 0 for sym in SYMPTOMS}
    detected_symptoms = set()
    
    # 1. Break input into manageable parts
    clauses = split_into_clauses(user_input)
    
    for clause in clauses:
        # 2. Discard the clause if the user is denying the symptom
        if is_negated(clause):
            continue 
            
        # 3. Convert the valid clause into a mathematical vector
        clause_embedding = embedder.encode(clause, convert_to_tensor=True)
        
        # 4. Compare it against all known symptoms
        cosine_scores = util.cos_sim(clause_embedding, symptom_embeddings)[0]
        
        # 5. Find the highest matching symptom
        best_match_idx = torch.argmax(cosine_scores).item()
        best_match_score = cosine_scores[best_match_idx].item()
        
        # 6. If the semantic match is strong enough, activate it
        if best_match_score >= threshold:
            matched_symptom = SYMPTOMS[best_match_idx]
            detected_symptoms.add(matched_symptom)
            bn_input[matched_symptom] = 1
            
    return bn_input, list(detected_symptoms)

# ==========================================
# 3. TESTING
# ==========================================
if __name__ == "__main__":
    print("\n--- Running Inference Tests ---")
    test_inputs = [
        "I have a terrible pounding in my head and feel nauseous, but no fever.",
        "my body is burning up and I can't stop throwing up",
        "It burns when I pee frequently",
        "I am totally fine, without any cough or chills."
    ]

    for inp in test_inputs:
        bn_input, detected = statement_to_bn_input(inp)
        print(f"\nInput: '{inp}'")
        print(f"Extracted Symptoms: {detected}")