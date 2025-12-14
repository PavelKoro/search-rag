
from MilvusSingleton_impl import MilvusSingleton
from TextEncoder_impl import TextEmbedding
import numpy as np

emb = TextEmbedding()

def poisk(query, name_db = "rag_db", collec = "docs"):
    milvus = MilvusSingleton(host="localhost", port="19530")
    milvus.setup_database(name_db)

    query_vec = np.asarray(emb.embedding_model.encode(query), dtype=np.float32).tolist()
    milv_id = milvus.search_by_vector(query_vec, collec, limit=15)
    
    while not milv_id['id']:
        print("[INFO]: Milvus no results found, retrying...")
        milv_id = milvus.search_milvus(query, collec, limit=10)
    print("[INFO]: Search results milvus:", milv_id['id'])

    print("[INFO]: Relevant chunks found:", milv_id['id'])

    res_chunks = []
    for i in range(len(milv_id['id'])):
        res_chunks.append({"text": milv_id['content'][i], "source": milv_id['source'][i]})

    print("[INFO]: Promt successfully:\n", res_chunks)
    return res_chunks

