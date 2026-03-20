# from transformers import T5Tokenizer, T5ForConditionalGeneration

# tokenizer = T5Tokenizer.from_pretrained("t5-small")
# model = T5ForConditionalGeneration.from_pretrained("t5-small")

# # Context passage (retrieved from your KB)
# context = "Gravity was discovered by Isaac Newton in the 17th century."

# question = "Who discovered gravity?"

# input_text = f"question: {question} context: {context}"
# inputs = tokenizer(input_text, return_tensors="pt")

# outputs = model.generate(**inputs, min_length=20, max_length=50)
# answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print("Answer:", answer)

from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------- documents --------
documents = [
    "Diabetes is a chronic disease caused by high blood sugar levels.",
    "Fever is a temporary increase in body temperature due to infection.",
    "Headache can be caused by stress, dehydration, or illness.",
]

# -------- build vectorizer --------
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents)

# -------- retriever --------
def retrieve(query):
    query_vec = vectorizer.transform([query])
    
    scores = cosine_similarity(query_vec, doc_vectors)[0]
    best_idx = scores.argmax()

    # threshold (important)
    if scores[best_idx] < 0.1:
        return None

    return documents[best_idx]

def ask(question):
    context = retrieve(question)

    if context is None:
        return "I don't know", None

    prompt = f"question: {question}  context: {context}"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        inputs.input_ids,
        max_length=50,
        do_sample=False
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer, context

# test
while True:
    q = input("Ask: ")
    if q == "exit" : break
    ans, ctx = ask(q)

    print("\nContext:", ctx)
    print("Answer:", ans)
    print("-" * 50)