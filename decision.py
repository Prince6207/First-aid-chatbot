import numpy as np
import pickle
from pgmpy.inference import VariableElimination
from inpToBn import statement_to_bn_input

vec, d = statement_to_bn_input("umm idk but like I think maybe fever?? and headache idk guess I have something like nausea or something not sure but my skin is kinda itchy and red")
print(d)

with open("model.pkl", "rb") as f:
    bn_model = pickle.load(f)

print("Model loaded successfully!")

print(bn_model.nodes())
print(bn_model.check_model())
print(bn_model.edges())

inference_engine = VariableElimination(bn_model)

result = inference_engine.query(variables=['diseases'], evidence=vec)

print(result)

import numpy as np

def get_top_k_diseases(result, k=3):
    """
    result: DiscreteFactor from pgmpy inference.query
    k: number of top diseases to return
    """

    probs = result.values
    
    states = result.state_names['diseases']

    disease_probs = list(zip(states, probs))

    disease_probs.sort(key=lambda x: x[1], reverse=True)

    return disease_probs[:k]

# Example usage
top_diseases = get_top_k_diseases(result, k=2)
for disease, prob in top_diseases:
    print(f"{disease}: {prob:.4f}")