import pickle
from pgmpy.inference import VariableElimination

def load_bayesian_model(model_path="model.pkl"):
    """Loads the compiled Bayesian Network from disk."""
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find model at {model_path}")
        return None

def find_top_k(evidence_vector, k=3, model_path="model.pkl"):
    """
    Queries the Bayesian Network for the most likely diseases.
    
    Args:
        evidence_vector (dict): A dictionary of symptoms mapped to 0 or 1.
        k (int): Number of top diseases to return.
        model_path (str): Path to the pickled Bayesian Network model.
        
    Returns:
        List of tuples: [(disease_name, probability_score), ...]
    """
    bn_model = load_bayesian_model(model_path)
    if not bn_model:
        return []

    # Initialize the inference engine
    inference_engine = VariableElimination(bn_model)

    # Some BN architectures only want the "positive" evidence (the 1s) to reduce computation.
    # If your model requires the full vector (0s and 1s), pass evidence_vector directly.
    # To optimize, we filter for only the symptoms that are actually present.
    active_evidence = {sym: val for sym, val in evidence_vector.items() if val == 1}
    
    # If no symptoms were detected, return empty
    if not active_evidence:
        return [("No symptoms detected", 0.0)]

    try:
        # Run the probabilistic query
        result = inference_engine.query(variables=['diseases'], evidence=active_evidence)
    except Exception as e:
        print(f"Inference Error: {e}")
        return []

    # Extract probabilities and state names
    probs = result.values
    states = result.state_names['diseases']

    # Combine them, sort by highest probability, and extract the top K
    disease_probs = list(zip(states, probs))
    disease_probs.sort(key=lambda x: x[1], reverse=True)

    return disease_probs[:k]

# ==========================================
# EXAMPLE USAGE
# ==========================================
if __name__ == "__main__":
    from inpToBn import statement_to_bn_input
    
    # Simulate a user query
    user_text = "I have a pounding headache and a fever, but no nausea."
    evidence, detected_symptoms = statement_to_bn_input(user_text)
    
    print(f"Detected Symptoms: {detected_symptoms}")
    
    # Uncomment below to run against your actual model file
    # top_diseases = find_top_k(evidence, k=3, model_path="model.pkl")
    # 
    # print("\nTop Diagnoses:")
    # for disease, prob in top_diseases:
    #     print(f"- {disease}: {prob * 100:.2f}%")