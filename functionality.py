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
import hashlib
import diskcache as dc
import nltk 
nltk.download('punkt_tab')

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
grammar_cache = dc.Cache('grammar_cache')

# For academic style checks
sf = Styleformer(style=0) 
style_cache = dc.Cache('style_cache')

# For text generation
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
model.generation_config.max_new_tokens = 2048
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_cache = dc.Cache('model_cache')

def generate_key(text):
    return hashlib.md5(text.encode()).hexdigest()


def split_into_chunks(text, max_tokens=MAX_TOKENS):
    sentences = nltk.sent_tokenize(text)
    chunks, current = [], ""
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(sentence.split())
        if current_tokens + sentence_tokens <= max_tokens:
            current += sentence + ' '
            current_tokens += sentence_tokens
        else:
            chunks.append(current.strip())
            current, current_tokens = sentence + ' ', sentence_tokens
    if current:
        chunks.append(current.strip())
    return chunks


# def split_into_chunks(text, max_tokens=MAX_TOKENS):
#     sentences = text.split(". ")
#     chunks = []
#     current = ""
#     for sentence in sentences:
#         if len(current.split()) + len(sentence.split()) <= max_tokens:
#             current += sentence + '. '
#         else:
#             chunks.append(current.strip())
#             current = sentence + '. '
#     if current:
#         chunks.append(current.strip())
#     return chunks

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

def fix_grammar(text: str):
    logging.info(f"\n---Fix Grammar input:---\n{text}")
    key = generate_key(text)
    if key in  grammar_cache:
        logging.info(f"Similar request was found in 'grammar_cache' and retrieved from it!")
        yield grammar_cache[key]

    else:
        args = TTSettings(num_beams=5, min_length=1)
        chunks = split_into_chunks(text=text, max_tokens=40)
        corrected_text = ""
        error_flag = False
        for chunk in chunks:
            try:
                result = happy_tt.generate_text(f"grammar: {chunk}", args=args)
                corrected_part = f"{result.text} "
            except Exception as e:
                error_flag = True
                logging.error(f"Error correcting grammar: {e}")
                corrected_part = f"{chunk} "
            corrected_text += corrected_part
            yield corrected_text

        if not error_flag:
            grammar_cache.set(key, corrected_text, expire=86400)
            logging.info(f"The result was cached in 'grammar_cache'!")

def fix_academic_style(informal_text: str):
    logging.info(f"\n---Fix Academic Style input:---\n{informal_text}")
    key = generate_key(informal_text)
    if key in style_cache:
        logging.info(f"Similar request was found in 'style_cache' and retrieved from it!")
        yield style_cache[key]
    
    else:
        chunks = split_into_chunks(text=informal_text, max_tokens=25)
        formal_text = ""
        error_flag = False
        for chunk in chunks:
            try:
                corrected_part = sf.transfer(chunk)
                if corrected_part is None:
                    error_flag = True
                    corrected_part = f"{chunk} "
                    logging.warning("---COULD NOT FIX ACADEMIC STYLE!\n")
                else:
                    corrected_part = f"{corrected_part} "
            except Exception as e:
                error_flag = True
                logging.error(f"Error in academic style transformation: {e}")
                corrected_part = f"{chunk} "
            formal_text += corrected_part
            yield formal_text

        if not error_flag:
            style_cache.set(key, formal_text, expire=86400)
            logging.info(f"The result was cached in 'style_cache'!")

def _chat_stream(initial_text: str, parts: list):
    logging.info(f"\n---Generate Article input:---\n{initial_text}")
    parts = ", ".join(parts).lower()
    for_cache = initial_text + ' ' + parts
    key = generate_key(for_cache)
    if key in model_cache:
        logging.info(f"Similar request was found in 'model_cache' and retrieved from it!")
        yield model_cache[key]
    else: 
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
                                    only generated article (or parts of it). The responce must be provided as a 
                                    raw text. Be precise and follow the structure of academic papers parts."""},
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
        model_cache.set(key, response, expire=86400)
        logging.info(f"The result was cached in 'model_cache'!")

def predict(goal: str, parts: list, context: str):
        if context == "":
            yield "Write your text first!"
            logging.info("No context was provided!")
        elif goal == 'Fix Academic Style':
            formal_text = ""
            for new_text in fix_academic_style(context):
                formal_text = new_text
                yield formal_text

            logging.info(f"\n---Academic style corrected:---\n {formal_text}\n")
        elif goal == 'Fix Grammar':
            full_response = ""
            for new_text in fix_grammar(context):
                full_response = new_text
                yield full_response
            
            logging.info(f"\n---Grammar corrected:---\n{full_response}\n")
        else:
            full_response = ""
            for new_text in _chat_stream(context, parts):
                full_response = new_text
                yield full_response

            logging.info(f"\nThe text was generated!\n{full_response}")
