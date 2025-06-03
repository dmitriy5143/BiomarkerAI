import os
import glob
import json
import logging
import redis
import re
import time
from typing import List
import pubmed_parser as pp
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import config

current_config = config[os.environ.get("FLASK_CONFIG", "production")]
logger = logging.getLogger("RAGBuilder")

FILTERED_DIR = current_config.FILTERED_DIR
CHUNK_THRESHOLD = current_config.CHUNK_THRESHOLD
MAX_CHUNK_SIZE = current_config.MAX_CHUNK_SIZE
CHUNK_OVERLAP = current_config.CHUNK_OVERLAP
EMBEDDINGS = HuggingFaceEmbeddings(model_name=current_config.EMBEDDINGS_MODEL, model_kwargs={"device": current_config.EMBEDDINGS_DEVICE})
redis_client = redis.Redis(host=current_config.REDIS_HOST, port=current_config.REDIS_PORT, db=current_config.REDIS_DB, decode_responses=True)

def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[[^\]]+\]', '', text)
    return text

def dynamic_chunking(text: str, threshold: int = CHUNK_THRESHOLD, max_chunk: int = MAX_CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = clean_text(text)
    if len(text) <= threshold:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chunk
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end >= len(text):
            break
        start = end - overlap
    return chunks

def load_documents_from_filtered(filtered_dir: str) -> List[Document]:
    documents = []
    xml_files = glob.glob(os.path.join(filtered_dir, "**", "*.xml"), recursive=True)
    logger.info("Найдено %d XML файлов для обработки", len(xml_files))
    for xml_path in xml_files:
        try:
            article_data = pp.parse_pubmed_xml(xml_path)
            title = clean_text(article_data.get("full_title", ""))
            doi = article_data.get("doi", "").strip()
            pmid = article_data.get("pmid", "").strip()
            publication_date = article_data.get("publication_date", "").strip()
            journal = article_data.get("journal", "").strip()
            subjects = article_data.get("subjects", "").strip()
            file_id = os.path.basename(xml_path)
            file_metadata = {
                "file_id": file_id,
                "title": title,
                "doi": doi,
                "pmid": pmid,
                "publication_date": publication_date,
                "journal": journal,
                "subjects": subjects
            }
            redis_client.set(f"file:{file_id}", json.dumps(file_metadata))

            paragraphs = pp.parse_pubmed_paragraph(xml_path)
            if not paragraphs or all(not p.get("text", "").strip() for p in paragraphs):
                logger.info("Файл %s: отсутствуют параграфы полного текста – пропускаем", xml_path)
                continue

            sections = {}
            for para in paragraphs:
                sec = para.get("section", "").strip() or "Unlabeled"
                text = clean_text(para.get("text", ""))
                if text:
                    sections.setdefault(sec, []).append(text)

            for sec, para_list in sections.items():
                aggregated_text = " ".join(para_list).strip()
                if not aggregated_text:
                    continue

                if len(aggregated_text) > CHUNK_THRESHOLD:
                    chunks = dynamic_chunking(aggregated_text, threshold=CHUNK_THRESHOLD,
                                            max_chunk=MAX_CHUNK_SIZE,
                                            overlap=CHUNK_OVERLAP)
                else:
                    chunks = [aggregated_text]

                for idx, chunk in enumerate(chunks):
                    doc_id = f"{file_id}:{sec}:{idx}"
                    doc_metadata = {"doc_id": doc_id, "file_id": file_id, "section": sec}
                    redis_client.set(f"doc:{doc_id}", json.dumps(doc_metadata))
                    documents.append(Document(page_content=chunk, metadata=doc_metadata))
        except Exception as ex:
            logger.error("Ошибка при обработке файла %s: %s", xml_path, ex)
            continue

    logger.info("Общее количество Document объектов: %d", len(documents))
    return documents

def build_rag_system():
    logger.info("Загружаем документы из директории %s", FILTERED_DIR)
    docs = load_documents_from_filtered(FILTERED_DIR)
    if not docs:
        logger.warning("Нет документов для построения RAG системы.")
        return None

    logger.info("Вычисление эмбеддингов для %d документов...", len(docs))
    start_time = time.time()

    vectorstore = FAISS.from_documents(docs, EMBEDDINGS)

    elapsed = time.time() - start_time
    logger.info("Векторное хранилище построено с %d документами за %.2f секунд.", len(docs), elapsed)
    vectorstore.save_local(current_config.FAISS_INDEX_PATH)

    return vectorstore