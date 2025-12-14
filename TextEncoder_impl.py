import torch
from sentence_transformers import SentenceTransformer

class TextEmbedding:
    def __init__(self):
        model_name='intfloat/multilingual-e5-large-instruct'
        self.embedding_model = SentenceTransformer(
            model_name, 
            device="cuda:0", 
            model_kwargs={"torch_dtype": torch.float16}
        )
        
    def vectorize_text(self, chunks):
        Data_db = {
            'id': [i for i in range(1, len(chunks) + 1)],
            'source': [chunk.metadata['source'] for chunk in chunks],
            'emb': [self.embedding_model.encode(chunk.page_content) for chunk in chunks],
            'content': [chunk.page_content for chunk in chunks]
        }
        return Data_db
    
    def model_emb(self):
        return self.embedding_model