import os
import re
import json
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import redis
from config import config

current_config = config[os.environ.get("FLASK_CONFIG", "production")]
EMBEDDINGS = HuggingFaceEmbeddings(model_name=current_config.EMBEDDINGS_MODEL, model_kwargs={"device": current_config.EMBEDDINGS_DEVICE})
redis_client = redis.Redis(host=current_config.REDIS_HOST, port=current_config.REDIS_PORT, db=current_config.REDIS_DB, decode_responses=True)

class DocumentUtils:
    @staticmethod
    def retrieve_documents(query: str, top_k: int = 2) -> List[dict]:
        try:
            vectorstore = FAISS.load_local(current_config.FAISS_INDEX_PATH, EMBEDDINGS, allow_dangerous_deserialization=True)
        except Exception as ex:
            return []
        try:
            docs = vectorstore.similarity_search(query, k=top_k)
        except Exception as ex:
            return []
        enriched_results = []
        for doc in docs:
            doc_id = doc.metadata.get("doc_id")
            if not doc_id:
                continue
            doc_metadata_json = redis_client.get(f"doc:{doc_id}")
            if not doc_metadata_json:
                continue
            doc_metadata = json.loads(doc_metadata_json)
            file_id = doc.metadata.get("file_id")
            file_metadata_json = redis_client.get(f"file:{file_id}")
            file_metadata = json.loads(file_metadata_json) if file_metadata_json else {}
            enriched_results.append({
                "content": doc.page_content,
                "doc_metadata": doc_metadata,
                "file_metadata": file_metadata
            })
        return enriched_results

    @staticmethod
    def extract_json_from_code_block(raw_response: str) -> str:
        code_block_pattern = r"```(?:json)?(.+?)```"
        code_blocks = re.findall(code_block_pattern, raw_response, flags=re.DOTALL | re.IGNORECASE)

        if code_blocks:
            last_block = code_blocks[-1].strip()
            start_idx = last_block.find('{')
            end_idx = last_block.rfind('}')

            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                json_text = last_block[start_idx:end_idx+1]
                try:
                    parsed = json.loads(json_text)
                    return json.dumps(parsed)
                except json.JSONDecodeError as e:
                    fixed_json = re.sub(r'"\s+("(?!,))', '", \1', json_text)
                    fixed_json = re.sub(r',\s*\}', '}', fixed_json)
                    try:
                        parsed = json.loads(fixed_json)
                        return json.dumps(parsed)
                    except json.JSONDecodeError:
                      pass
        start_idx = raw_response.find('{')
        end_idx = raw_response.rfind('}')

        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            json_text = raw_response[start_idx:end_idx+1]
            try:
                parsed = json.loads(json_text)
                return json.dumps(parsed)
            except json.JSONDecodeError as e:
                fixed_json = re.sub(r'"\s+("(?!,))', '", \1', json_text)
                fixed_json = re.sub(r',\s*\}', '}', fixed_json)

                try:
                    parsed = json.loads(fixed_json)
                    return json.dumps(parsed)
                except json.JSONDecodeError:
                    pass
        json_pattern = r"\{(?:.|\n)*?\}"
        json_matches = re.findall(json_pattern, raw_response, flags=re.DOTALL)

        if json_matches:
            sorted_matches = sorted(json_matches, key=len, reverse=True)

            for potential_json in sorted_matches:
                fixed_json = re.sub(r'"\s+("(?!,))', '", \1', potential_json)
                fixed_json = re.sub(r',\s*\}', '}', fixed_json)
                try:
                    parsed = json.loads(fixed_json)
                    return json.dumps(parsed)
                except json.JSONDecodeError:
                    continue

        return None