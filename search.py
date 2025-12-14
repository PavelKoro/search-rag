import json
import numpy as np
from typing import List
from pathlib import Path
from datetime import datetime

from TextChunker_impl import TextChunker
from TextEncoder_impl import TextEmbedding
from MilvusSingleton_impl import MilvusSingleton

emb = TextEmbedding()
text_docs = TextChunker()

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

def parser(files: List[str]):
    existing_records = []
    if Path("files_chunks.json").exists():
        raw = Path("files_chunks.json").read_text(encoding="utf-8").strip()
        if raw:
            try:
                data = json.loads(raw)
                if isinstance(data, list):
                    existing_records = data
                else:
                    raise ValueError("JSON is not an array")
            except Exception:
                backup = Path("files_chunks.json").with_name(
                    f"files_chunks_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                backup.write_text(raw, encoding="utf-8")
                existing_records = []
    
    # 2) Считаем следующий id
    max_id = 0
    for item in existing_records:
        if isinstance(item, dict) and "id" in item:
            try:
                max_id = max(max_id, int(item["id"]))
            except Exception:
                pass
    next_id = max_id + 1

    # 3) Генерируем новые записи и добавляем
    new_records = []

    for file_name in files:
        file_path = Path(file_name)

        docs = text_docs.load_pdf_documents(file_path)
        chunks = text_docs.splitting(docs)
        print("[INFO]: Векторизуем текст")
        vectors = emb.vectorize_text(chunks)

        for i, chunk in enumerate(chunks):
            vec = vectors["emb"][i]
            if hasattr(vec, "tolist"):
                vec = vec.tolist()

            new_records.append(
                {
                    "id": next_id,
                    "source": chunk.metadata.get("source", str(file_path)),
                    "embeddings": vec,
                    "content": chunk.page_content,
                }
            )
            next_id += 1


    existing_records.extend(new_records)

    # 4) Сохраняем обратно (валидный JSON-массив)
    Path("files_chunks.json").write_text(
        json.dumps(existing_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] Добавлено чанков: {len(new_records)}. Всего записей: {len(existing_records)}")
    
    push_milv()



def push_milv(name_db = "rag_db", collec = "docs"):
    json_path = Path("files_chunks.json")
    rows = json.loads(json_path.read_text(encoding="utf-8"))

    milvus = MilvusSingleton(host="localhost", port="19530")
    milvus.setup_database(name_db)

    DIM = int(np.asarray(rows[0]["embeddings"]).size)
    milvus.create_collection(collec, size_vec=DIM, drop_if_exists=True)

    max_bytes = 40 * 1024 * 1024

    ids, sources, embs, contents = [], [], [], []
    batch_bytes = 0
    total = 0

    def send():
        nonlocal ids, sources, embs, contents, batch_bytes, total
        if not ids:
            return
        milvus.insert_data(
            collec,
            {"id": ids, "source": sources, "embeddings": embs, "content": contents},
            flush=False
        )
        total += len(ids)
        ids, sources, embs, contents = [], [], [], []
        batch_bytes = 0

    for r in rows:
        src = str(r.get("source", ""))
        txt = str(r.get("content", ""))

        emb = np.asarray(r["embeddings"], dtype=np.float32).reshape(-1).tolist()

        row_bytes = (len(emb) * 4) + len(src.encode("utf-8")) + len(txt.encode("utf-8")) + 256

        if ids and (batch_bytes + row_bytes > max_bytes):
            send()

        ids.append(int(r["id"]))
        sources.append(src)
        embs.append(emb)
        contents.append(txt)
        batch_bytes += row_bytes

    send()

    col = milvus.get_collection(collec)
    col.flush()
    col.load()

    return total