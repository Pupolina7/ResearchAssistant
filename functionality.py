# import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from happytransformer import HappyTextToText, TTSettings
from styleformer import Styleformer
from sentence_transformers import SentenceTransformer
import chromadb
import pandas as pd
import logging
import re
from threading import Thread

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, # filename="py_log.log",filemode="w",
                    format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


# For chromadb collection
MAX_TOKENS = 512
client = chromadb.Client()
embedder = SentenceTransformer('all-MiniLM-L6-v2')
collection_name = 'papers'

# For grammar checker
happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")

# For academic style checks
sf = Styleformer(style=0) 

# For text generation
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
model.generation_config.max_new_tokens = 2048
tokenizer = AutoTokenizer.from_pretrained(model_name)

# from nltk import sent_tokenize

# def split_into_chunks(text, max_tokens=MAX_TOKENS):
#     sentences = sent_tokenize(text)
#     chunks, current = [], ""
#     current_tokens = 0

#     for sentence in sentences:
#         sentence_tokens = len(sentence.split())
#         if current_tokens + sentence_tokens <= max_tokens:
#             current += sentence + ' '
#             current_tokens += sentence_tokens
#         else:
#             chunks.append(current.strip())
#             current, current_tokens = sentence + ' ', sentence_tokens
#     if current:
#         chunks.append(current.strip())
#     return chunks


def split_into_chunks(text, max_tokens=MAX_TOKENS):
    sentences = text.split(". ")
    chunks = []
    current = ""
    for sentence in sentences:
        if len(current.split()) + len(sentence.split()) <= max_tokens:
            current += sentence + '. '
        else:
            chunks.append(current.strip())
            current = sentence + '. '
    if current:
        chunks.append(current.strip())
    return chunks

def clean_text(text):
    # Remove newlines within sentences but keep paragraph breaks
    text = re.sub(r'\n(?!\n)', ' ', text)
    
    # Remove multiple newlines, keeping only double newlines for paragraphs
    text = re.sub(r'\n{2,}', '\n\n', text)
    
    # Rejoin hyphenated words split across lines
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
    
    # Remove citation brackets and figure numbers
    text = re.sub(r'\[\d+\]', '', text)  # Removes [7], [6], etc.
    text = re.sub(r'Fig\.|Figure', '', text)  # Removes "Fig." or "Figure" references

    # Strip leading/trailing spaces from each paragraph
    paragraphs = text.split('\n')
    cleaned_paragraphs = [para.strip() for para in paragraphs if para.strip()]
    
    # Join cleaned paragraphs back with double newlines for readability
    cleaned_text = '\n\n'.join(cleaned_paragraphs)
    
    return cleaned_text

def get_collection() -> chromadb.Collection:
    collection_names = [collection.name for collection in client.list_collections()]
    logging.info(f"Client collection names: {collection_names}")
    if collection_name not in collection_names:
        logging.info(f"Creation of a collection...")
        collection = client.create_collection(name=collection_name)
        papers = pd.read_csv("hf://datasets/somosnlp-hackathon-2022/scientific_papers_en/scientific_paper_en.csv")
        logging.info(f"The data downloaded from url.")
        papers = papers.drop(['id'], axis=1)
        papers = papers.iloc[:200]

        for i in range(200):
            paper = papers.iloc[i]
            idx = paper.name

            full_text = clean_text('Abstract ' + paper['abstract'] + ' ' + paper['text_no_abstract'])
            chunks = split_into_chunks(full_text)
            
            for id, chunk in enumerate(chunks):
                embeddings = embedder.encode([chunk])
                collection.upsert(ids=f"paper{idx}_chunk_{id}", 
                                documents=[chunk], 
                                embeddings=embeddings,)
            logging.info(f"Collection upsert: The content of paper_{idx} was chunked and collected in vector db!")
        
        logging.info(f"Collection is filled!\n")
    else:
        collection = client.get_collection(name=collection_name)
        logging.info(f"Collection '{collection_name}' already exists!")
    return collection

def fix_grammar(text: str) -> str:
    logging.info(f"\n---Fix Grammar input:---\n{text}")
    args = TTSettings(num_beams=5, min_length=1)
    try:
        result = happy_tt.generate_text(f"grammar: {text}", args=args)
        corrected_text = result.text
    except Exception as e:
        logging.error(f"Error correcting grammar: {e}")
        corrected_text = text
    logging.info(f"\n---Grammar corrected:---\n{corrected_text}\n")
    return corrected_text

def fix_academic_style(informal_text: str) -> str:
    logging.info(f"\n---Fix Academic Style input:---\n{informal_text}")
    try:
        formal_text = sf.transfer(informal_text)
        if formal_text is None:
            formal_text = informal_text
            logging.warning("---COULD NOT FIX ACADEMIC STYLE!\n")
        else:
            logging.info(f"\n---Academic style corrected:---\n {formal_text}\n")
    except Exception as e:
        logging.error(f"Error in academic style transformation: {e}")
        formal_text = informal_text

    return formal_text

def _chat_stream(initial_text: str, parts: list):
    logging.info(f"\n---Generate Article input:---\n{initial_text}")
    parts = ", ".join(parts).lower()

    text_embedding = embedder.encode([initial_text])
    chroma_collection = get_collection()
    results = chroma_collection.query(
        query_embeddings=text_embedding,
        n_results=1
    )
    context = results['documents'][0] if results['documents'] else ""
    if context == "":
        logging.warning(f"COLLECTION QUERY: No context was found in the database!")

    messages = [
    {"role": "system", "content": """You are helpful Academic Research Assistant which helps to generate 
                                necessary parts of the reserch based on the provided context.
                                The context is the following: 'written text' - this is the text that user
                                has for now and want to complete, 'parts' - those are the parts of paper 
                                user needs to complete (it could be the abstract, introduction, methodology,
                                discussion, conclusion, or full text), 'context' - the similar article 
                                the structure of which can be used as a base for the text (it can be empty
                                in case of absence of similar papers in the database.). The output should be
                                only generated article (or parts of it). The responce must be provided as a text."""},
    {"role": "user", "content": f"'written text': {initial_text}\n 'parts': {parts}\n 'context': {context}"},
    ]
    input_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True
    )
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
    }
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    response = ""
    for new_text in streamer:
        response += new_text
        yield response

def predict(goal: str, parts: list, context: str):
        # print(f"User: {_query}")
        if goal == 'Check Academic Style':
            yield fix_academic_style(context)
        elif goal == 'Check Grammar':
            yield fix_grammar(context)
        else:
            full_response = ""
            for new_text in _chat_stream(context, parts):
                full_response = new_text
                yield full_response

            logging.info(f"\nThe text was generated!\n{full_response}")


# def generate_article(initial_text: str, parts: list) -> str:
#     logging.info(f"\n---Generate Article input:---\n{initial_text}")
#     parts = ", ".join(parts).lower()

#     text_embedding = embedder.encode([initial_text])
#     chroma_collection = get_collection()
#     results = chroma_collection.query(
#         query_embeddings=text_embedding,
#         n_results=1
#     )
#     context = results['documents'][0] if results['documents'] else ""
#     if context == "":
#         logging.warning(f"COLLECTION QUERY: No context was found in the database!")

#     messages = [
#     {"role": "system", "content": """You are helpful Academic Research Assistant which helps to generate 
#                                 necessary parts of the reserch based on the provided context.
#                                 The context is the following: 'written text' - this is the text that user
#                                 has for now and want to complete, 'parts' - those are the parts of paper 
#                                 user needs to complete (it could be the abstract, introduction, methodology,
#                                 discussion, conclusion, or full text), 'context' - the similar article 
#                                 the structure of which can be used as a base for the text (it can be empty
#                                 in case of absence of similar papers in the database.). The output should be
#                                 only generated article (or parts of it). The responce must be provided as a text."""},
#     {"role": "user", "content": f"'written text': {initial_text}\n 'parts': {parts}\n 'context': {context}"},
#     ]
#     text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True
#     )
#     model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

#     generated_ids = model.generate(
#         **model_inputs,
#         max_new_tokens=512
#     )
#     generated_ids = [
#         output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#     ]

#     response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     logging.info("The text was generated!")
#     return response
    
# def handle_user_prompt(goal: str, parts: list, context: str) -> str:
#     if goal == 'Check Academic Style':
#         return fix_academic_style(context)
#     elif goal == 'Check Grammar':
#         return fix_grammar(context)
#     elif goal == 'Write Text (Part)':
#         return generate_article(context, parts)
    
    
