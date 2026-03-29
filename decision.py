import pickle
from pgmpy.inference import VariableElimination

def find_top_k(vec, k=3, model_path="model.pkl"):
    """
    vec: dict
        Evidence dictionary for Bayesian Network inference.
    k: int
        Number of top diseases to return.
    model_path: str
        Path to the pickled Bayesian Network model.
    Returns:
        List of tuples (disease, probability) for top-k diseases.
    """
    with open(model_path, "rb") as f:
        bn_model = pickle.load(f)

    inference_engine = VariableElimination(bn_model)

    result = inference_engine.query(variables=['diseases'], evidence=vec)

    probs = result.values
    states = result.state_names['diseases']

    disease_probs = list(zip(states, probs))
    disease_probs.sort(key=lambda x: x[1], reverse=True)

    return disease_probs[:k]

# Example usage:
# vec, d = statement_to_bn_input("fever and headache with nausea")
# top_diseases = find_top_k(vec, k=2)
# for disease, prob in top_diseases:
#     print(f"{disease}: {prob:.4f}")