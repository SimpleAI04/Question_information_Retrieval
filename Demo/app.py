import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset

@st.cache_resource
def load_data_and_model():
    dataset = load_dataset('ms_marco', 'v1.1')
    subset = dataset['test']

    corpus = []
    for sample in subset:
        if sample['query_type'] != 'entity':
            continue
        passage_text_lst = sample['passages']['passage_text']
        corpus += passage_text_lst

    model = SentenceTransformer('all-MiniLM-L6-v2')
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

    return corpus, model, corpus_embeddings

def search(query, corpus, model, corpus_embeddings, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(scores, k=top_k)
    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        results.append((corpus[idx], float(score)))
    return results

# Giao diện Streamlit
st.title("🔎Text Retrieval Demo")
st.write("Nhập truy vấn để tìm các đoạn thông tin từ tập dữ liệu MS MARCO.")

query = st.text_input("Nhập thông tin truy vấn của bạn:")
top_k = st.slider("Chọn Top K kết quả muốn hiển thị:", min_value=1, max_value=10, value=5)

if query:
    corpus, model, corpus_embeddings = load_data_and_model()
    results = search(query, corpus, model, corpus_embeddings, top_k)

    st.success(f"Kết quả truy xuất cho truy vấn: `{query}`")
    for i, (text, score) in enumerate(results):
        st.markdown(f"**Top{i+1} (Score: {score:.4f})**\n\n{text}\n\n---")