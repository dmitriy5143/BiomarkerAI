from Bio import Entrez
import pubmed_parser as pp
import time
import pandas as pd
from bs4 import BeautifulSoup
import re
import os
import requests
import tarfile
import shutil
import spacy
import subprocess
import logging
import datetime
from config import config

current_config = config[os.environ.get("FLASK_CONFIG", "production")]
logger = logging.getLogger("RAGBuilder")
logger.setLevel(logging.INFO)
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
lemma_cache = {}

def cached_lemmatize(token):
    txt = token.text.lower()
    if txt in lemma_cache:
        return lemma_cache[txt]
    lemma = token.lemma_
    lemma_cache[txt] = lemma
    return lemma

def normalize_text(text):
    doc = nlp(text.lower())
    lemmas = [cached_lemmatize(token) for token in doc if not token.is_punct and not token.is_stop]
    return set(lemmas)

DOWNLOAD_DIR = current_config.DOWNLOAD_DIR
PROCESSED_DIR = os.path.join(DOWNLOAD_DIR, "processed")
FILTERED_DIR = os.path.join(DOWNLOAD_DIR, "filtered")
BASE_URL = current_config.BASE_URL
MAX_MEMORY_LIMIT_BYTES = getattr(current_config, "MAX_MEMORY_LIMIT_BYTES", 1 * 1024 * 1024 * 1024)
AVERAGE_XML_SIZE_BYTES = getattr(current_config, "AVERAGE_XML_SIZE_BYTES", 100 * 1024)
MAX_XML_FILES = MAX_MEMORY_LIMIT_BYTES // AVERAGE_XML_SIZE_BYTES
MAX_ARCHIVE_SIZE_MB = getattr(current_config, "MAX_ARCHIVE_SIZE_MB", 500)

for directory in [DOWNLOAD_DIR, PROCESSED_DIR, FILTERED_DIR]:
    os.makedirs(directory, exist_ok=True)

KEYWORDS = current_config.KEYWORDS

def update_keywords(new_keywords):
    global KEYWORDS
    if new_keywords:
        KEYWORDS.clear()
        if isinstance(new_keywords, dict):
            KEYWORDS.update(new_keywords)
        else:
            try:
                import json
                KEYWORDS.update(json.loads(new_keywords))
            except Exception as ex:
                logger.error("Ошибка при разборе ключевых слов: %s", ex)
    logger.info("Ключевые слова обновлены: %s", KEYWORDS)
    return KEYWORDS

def get_file_list(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            file_links = []
            for link in soup.find_all('a'):
                href = link.get('href')
                if href and href.endswith('.tar.gz'):
                    file_links.append(href)
            logger.info("Найдено %d архивов", len(file_links))
            return file_links
        else:
            logger.error("Ошибка при получении списка файлов: статус %s", response.status_code)
            return []
    except Exception as e:
        logger.error("Ошибка при получении списка файлов: %s", e)
        return []

def download_file(url, dest_path, retry_count=3, max_size_mb=MAX_ARCHIVE_SIZE_MB):
    for attempt in range(retry_count):
        try:
            response = requests.get(url, stream=True, timeout=30)
            total_size_in_bytes = int(response.headers.get('content-length', 0))
            if total_size_in_bytes > max_size_mb * 1024 * 1024:
                logger.warning("Архив %s слишком большой (%.2f МБ). Максимально допустимый размер: %d МБ. Пропускаем.",
                               os.path.basename(dest_path), total_size_in_bytes/1024/1024, max_size_mb)
                return False
            free_space = shutil.disk_usage(os.path.dirname(dest_path)).free
            if total_size_in_bytes > free_space:
                logger.error("Недостаточно места на диске для %s. Требуется: %.2f МБ, доступно: %.2f МБ",
                             dest_path, total_size_in_bytes/1024/1024, free_space/1024/1024)
                return False
            block_size = 1024 * 1024
            bytes_downloaded = 0
            with open(dest_path, 'wb') as file:
                for data in response.iter_content(block_size):
                    file.write(data)
                    bytes_downloaded += len(data)
            if total_size_in_bytes != 0 and bytes_downloaded != total_size_in_bytes:
                logger.warning("Размер скачанного файла (%d байт) не соответствует ожидаемому (%d байт). Попытка %d/%d",
                               bytes_downloaded, total_size_in_bytes, attempt+1, retry_count)
                if attempt < retry_count - 1:
                    time.sleep(5)
                    continue
                return False
            logger.info("Файл успешно скачан: %s", dest_path)
            return True
        except Exception as e:
            logger.error("Ошибка при скачивании %s: %s. Попытка %d/%d", url, e, attempt+1, retry_count)
            if attempt < retry_count - 1:
                time.sleep(5)
            else:
                return False
    return False

def stream_extract_and_filter_archive(archive_path, filtered_dir, update_progress_fn=None):
    saved_files = 0
    try:
        with tarfile.open(archive_path, 'r:gz') as tar:
            members = tar.getmembers()
            total_members = len(members)
            for idx, member in enumerate(members, 1):
                if not member.isfile():
                    continue
                if not member.name.endswith('.xml'):
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                try:
                    content = f.read()
                    temp_filename = os.path.basename(member.name)
                    temp_path = os.path.join(DOWNLOAD_DIR, "temp.xml")
                    with open(temp_path, "wb") as temp_file:
                        temp_file.write(content)
                    if is_relevant_article(temp_path):
                        if temp_filename not in processed_file_names:
                            destination_path = os.path.join(filtered_dir, temp_filename)
                            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                            with open(destination_path, "wb") as out_file:
                                out_file.write(content)
                            processed_file_names.add(temp_filename)
                            saved_files += 1
                            if update_progress_fn:
                                update_progress_fn({
                                    "phase": "extract",
                                    "status": "progress",
                                    "message": f"Сохранён релевантный файл: {temp_filename}.",
                                    "current_file": saved_files
                                })
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception as e:
                    logger.error("Ошибка при обработке файла %s: %s", member.name, e)
                    continue
        if os.path.exists(archive_path):
            os.remove(archive_path)
        return saved_files
    except Exception as e:
        logger.error("Ошибка при потоковой распаковке архива %s: %s", archive_path, e)
        return saved_files

processed_file_names = set()

def cleanup_archive(dirs_to_clean):
    for d in dirs_to_clean:
        if os.path.exists(d):
            try:
                shutil.rmtree(d)
                logger.info("Директория %s удалена для освобождения места.", d)
            except Exception as e:
                logger.error("Ошибка при удалении директории %s: %s", d, e)

def extract_date_from_filename(filename):
    match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    return match.group(1) if match else "0000-00-00"

def is_relevant_article(xml_path):
    try:
        article = pp.parse_pubmed_xml(xml_path)
        title = article.get("full_title", "")
        abstract = article.get("abstract", "")
        mesh_terms = article.get("mesh_terms", "") or article.get("subjects", "")
        combined_text = " ".join([title, abstract, mesh_terms])
        normalized_tokens = normalize_text(combined_text)
        for group in KEYWORDS.values():
            norm_group = set([kw.lower() for kw in group])
            if normalized_tokens.intersection(norm_group):
                return True
        return False
    except Exception as e:
        logger.error("Ошибка при обработке XML файла %s: %s", xml_path, e)
        return False

def process_archives_with_progress(max_archives=None, update_progress_fn=None):
    archive_list = get_file_list(BASE_URL)
    if not archive_list:
        msg = "Не удалось получить список архивов"
        logger.error(msg)
        if update_progress_fn:
            update_progress_fn({"phase": "archive", "status": "error", "message": msg})
        return

    baseline_archives = [a for a in archive_list if "baseline" in a]
    incr_archives = [a for a in archive_list if "incr" in a]

    baseline_archives.sort(key=lambda x: extract_date_from_filename(x))
    incr_archives.sort(key=lambda x: extract_date_from_filename(x), reverse=True)

    selected_archives = []
    if max_archives and isinstance(max_archives, int) and max_archives > 0:
        if max_archives == 1:
            if baseline_archives:
                selected_archives.append(baseline_archives[-1])
        else:
            if baseline_archives:
                selected_archives.append(baseline_archives[-1])
            selected_archives.extend(incr_archives[:max_archives - 1])
    else:
        selected_archives = archive_list

    total_archives = len(selected_archives)
    logger.info("Будет обработано %d архивов", total_archives)
    if update_progress_fn:
        update_progress_fn({
            "phase": "archive_overall",
            "status": "start",
            "total_archives": total_archives,
            "message": f"Начинается обработка {total_archives} архивов."
        })

    processed_xml_count = 0
    for idx, archive in enumerate(selected_archives, 1):
        meta = {
            "phase": "archive",
            "archive_number": idx,
            "total_archives": total_archives,
            "status": "started",
            "message": f"Обработка архива {archive} (дата: {extract_date_from_filename(archive)})"
        }
        if update_progress_fn:
            update_progress_fn(meta)

        url = BASE_URL + archive
        dest_path = os.path.join(DOWNLOAD_DIR, archive)

        if not download_file(url, dest_path):
            meta.update({
                "status": "download_failed",
                "message": f"Не удалось скачать архив {archive}. Пропускаем."
            })
            if update_progress_fn:
                update_progress_fn(meta)
            continue

        meta.update({
            "status": "extracting",
            "message": f"Обработка архива {archive}..."
        })
        if update_progress_fn:
            update_progress_fn(meta)

        filtered_archive_dir = os.path.join(FILTERED_DIR, os.path.splitext(os.path.splitext(archive)[0])[0])
        os.makedirs(filtered_archive_dir, exist_ok=True)
        saved = stream_extract_and_filter_archive(dest_path, filtered_archive_dir, update_progress_fn)
        processed_xml_count += saved

        meta.update({
            "status": "completed",
            "message": f"Архив {archive} успешно обработан. Сохранено {saved} релевантных статей."
        })
        if update_progress_fn:
            update_progress_fn(meta)

        if processed_xml_count >= MAX_XML_FILES:
            if update_progress_fn:
                update_progress_fn({
                    "phase": "archive_overall",
                    "status": "limiting",
                    "message": f"Достигнут лимит памяти — обработано {processed_xml_count} релевантных статей.",
                    "processed": processed_xml_count
                })
            break

        time.sleep(1)

    if update_progress_fn:
        update_progress_fn({
            "phase": "archive_overall",
            "status": "completed",
            "message": "Обработка архивов завершена."
        })