import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.Client()
embedder = SentenceTransformer('all-MiniLM-L6-v2')

collection = client.get_or_create_collection(name='papers')
if collection.count() == 0:
    papers = pd.read_csv("hf://datasets/somosnlp-hackathon-2022/scientific_papers_en/scientific_paper_en.csv", 
                     index_col=0)
    papers = papers.iloc[:200]
    for paper in papers:
        id = paper['id']
        inx = id[id.find('.') + 1 :]
    
