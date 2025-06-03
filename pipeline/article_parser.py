from Bio import Entrez
import pubmed_parser as pp
import time
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
import re
import os
import requests
import tarfile
import shutil
import spacy
import subprocess
import logging
from config import config

current_config = config[os.environ.get("FLASK_CONFIG", "production")]
logger = logging.getLogger("RAGBuilder")
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
EXTRACTED_DIR = os.path.join(DOWNLOAD_DIR, "extracted")
PROCESSED_DIR = os.path.join(DOWNLOAD_DIR, "processed")
FILTERED_DIR = os.path.join(DOWNLOAD_DIR, "filtered")
BASE_URL = current_config.BASE_URL

for directory in [DOWNLOAD_DIR, EXTRACTED_DIR, PROCESSED_DIR, FILTERED_DIR]:
    os.makedirs(directory, exist_ok=True)

KEYWORDS = current_config.KEYWORDS

def update_keywords(new_keywords):
    global KEYWORDS
    if new_keywords:
        KEYWORDS.clear()
        KEYWORDS.update(new_keywords)
        logger.info("Ключевые слова обновлены: %s", KEYWORDS)
    return KEYWORDS

def get_file_list(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            file_links = []
            for link in soup.find_all('a'):
                href = link.get('href')
                if href and href.endswith('.tar.gz'):
                    file_links.append(href)
            print(f"Найдено {len(file_links)} архивов")
            return file_links
        else:
            print(f"Ошибка при получении списка файлов: {response.status_code}")
            return []
    except Exception as e:
        print(f"Ошибка при получении списка файлов: {e}")
        return []

def download_file(url, dest_path, retry_count=3):
    for attempt in range(retry_count):
        try:
            response = requests.get(url, stream=True)
            total_size_in_bytes = int(response.headers.get('content-length', 0))
            free_space = shutil.disk_usage(os.path.dirname(dest_path)).free
            if total_size_in_bytes > free_space:
                print(f"Недостаточно места на диске. Требуется: {total_size_in_bytes/1024/1024:.2f} МБ, доступно: {free_space/1024/1024:.2f} МБ")
                return False
            block_size = 1024 * 1024
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc=os.path.basename(dest_path))
            with open(dest_path, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()
            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                print(f"Предупреждение: размер скачанного файла не соответствует ожидаемому. Попытка {attempt+1}/{retry_count}")
                if attempt < retry_count - 1:
                    time.sleep(5)
                    continue
                return False
            return True
        except Exception as e:
            print(f"Ошибка при скачивании {url}: {e}. Попытка {attempt+1}/{retry_count}")
            if attempt < retry_count - 1:
                time.sleep(5)
            else:
                return False
    return False

def extract_archive(archive_path, extract_dir):
    try:
        print(f"Распаковка архива {os.path.basename(archive_path)}...")
        os.makedirs(extract_dir, exist_ok=True)
        try:
            subprocess.run(['tar', '-xzf', archive_path, '-C', extract_dir],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            print(f"Ошибка при использовании внешней утилиты tar: {e}\nИспользуем встроенную библиотеку tarfile...")
            with tarfile.open(archive_path, 'r:gz') as tar:
                for member in tar:
                    if not (member.isreg() or member.isdir()):
                        continue
                    target_path = os.path.join(extract_dir, member.name)
                    if not os.path.normpath(target_path).startswith(os.path.normpath(extract_dir)):
                        continue
                    tar.extract(member, extract_dir)
                tar.members = []
        xml_count = sum(1 for root, dirs, files in os.walk(extract_dir) for file in files if file.endswith('.xml'))
        if os.path.exists(archive_path):
            os.remove(archive_path)
            print(f"Архив {os.path.basename(archive_path)} удален")
        print(f"Распаковано {xml_count} XML-файлов")
        return True
    except Exception as e:
        print(f"Ошибка при распаковке архива {archive_path}: {e}")
        return False

def is_relevant_article(xml_path):
    try:
        article = pp.parse_pubmed_xml(xml_path)
        title = article.get("full_title", "")
        abstract = article.get("abstract", "")
        mesh_terms = article.get("mesh_terms", "") or article.get("subjects", "")
        combined_text = " ".join([title, abstract, mesh_terms])
        normalized_tokens = normalize_text(combined_text)
        relevant = False
        for group in KEYWORDS.values():
            norm_group = set([kw.lower() for kw in group])
            if normalized_tokens.intersection(norm_group):
                relevant = True
                break
        return relevant
    except Exception as e:
        print(f"Ошибка при обработке XML файла {xml_path}: {e}")
        return False

def filter_extracted_xml(processed_dir, filtered_dir):
    for root, dirs, files in os.walk(processed_dir):
        for file in files:
            if file.endswith('.xml'):
                xml_path = os.path.join(root, file)
                if is_relevant_article(xml_path):
                    rel_path = os.path.relpath(root, processed_dir)
                    target_dir = os.path.join(filtered_dir, rel_path)
                    os.makedirs(target_dir, exist_ok=True)
                    shutil.copy2(xml_path, os.path.join(target_dir, file))

def process_archives_sequentially(max_archives=None):
    archive_list = get_file_list(BASE_URL)
    if not archive_list:
        print("Не удалось получить список архивов")
        return
    dated_archives = []
    for archive in archive_list:
        match = re.search(r'(\d{4}-\d{2}-\d{2})', archive)
        date = match.group(1) if match else ""
        dated_archives.append((date, archive))
    dated_archives.sort()
    if max_archives and max_archives > 0:
        dated_archives = dated_archives[-max_archives:]
    print(f"Будет обработано {len(dated_archives)} архивов")
    for date, archive in dated_archives:
        url = BASE_URL + archive
        dest_path = os.path.join(DOWNLOAD_DIR, archive)
        print(f"\n{'='*50}")
        print(f"Обработка архива: {archive} (дата: {date})")
        print(f"{'='*50}")
        archive_name = os.path.splitext(os.path.splitext(archive)[0])[0]
        processed_dir_for_archive = os.path.join(PROCESSED_DIR, archive_name)
        filtered_dir_for_archive = os.path.join(FILTERED_DIR, archive_name)
        if os.path.exists(filtered_dir_for_archive) and os.listdir(filtered_dir_for_archive):
            print(f"Архив {archive} уже был обработан. Пропускаем.")
            continue
        print(f"Скачивание архива {archive}...")
        if not download_file(url, dest_path):
            print(f"Не удалось скачать архив {archive}. Пропускаем.")
            continue
        if not extract_archive(dest_path, processed_dir_for_archive):
            print(f"Ошибка при распаковке архива {archive}")
            if os.path.exists(dest_path):
                os.remove(dest_path)
            continue
        filter_extracted_xml(processed_dir_for_archive, filtered_dir_for_archive)
        time.sleep(2)
    print("\nОбработка архивов завершена")