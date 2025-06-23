import os
import re
import json
import time
import logging
import torch
import redis
from typing import List, Optional, Tuple
from transformers import (AutoTokenizer, AutoModelForCausalLM, pipeline)
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator
from pipeline.document_utils import DocumentUtils
from config import config
import warnings

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

logger = logging.getLogger("RAGBuilder")
current_config = config[os.environ.get("FLASK_CONFIG", "production")]
EMBEDDINGS = HuggingFaceEmbeddings(model_name=current_config.EMBEDDINGS_MODEL, model_kwargs={"device": current_config.EMBEDDINGS_DEVICE})
redis_client = redis.Redis(host=current_config.REDIS_HOST, port=current_config.REDIS_PORT, db=current_config.REDIS_DB, decode_responses=True)

class Source(BaseModel):
    source: str
    doi: Optional[str] = None
    pmid: Optional[str] = None

    @field_validator("pmid", "doi", mode="before")
    @classmethod
    def convert_to_str(cls, v):
        if v is None:
            return v
        return str(v)

class MetaboliteAnalysis(BaseModel):
    answer: str = Field(..., description="Name of the metabolite")
    reasoning: str = Field(..., description="Provide a detailed, comprehensive analysis including biochemical properties, mechanisms of action, and specific relationship to the disease.")
    sources: List[Source] = Field(default_factory=list, description="List of source objects with metadata")

class CustomPydanticOutputParser(PydanticOutputParser):
    def get_format_instructions(self) -> str:
        instructions = (
            "IMPORTANT: Format your response as a JSON code block using triple backticks:\n"
            "```json\n"
            "{\n"
            '  "answer": "Name of the metabolite",\n'
            '  "reasoning": "Detailed analysis...",\n'
            '  "sources": [\n'
            '     {\n'
            '       "source": "Source title",\n'
            '       "doi": "[DOI]",\n'
            '       "pmid": "[PMID]"\n'
            '     }\n'
            "  ]\n"
            "}\n"
            "```\n"
            "Ensure fields are present. If there is no data for a field, use an empty string or an empty array (for sources)."
        )
        return instructions

class OptimizedRAGLLMAgent:
    def __init__(self,
                 model_name: str = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
                 max_tokens: int = 1500, 
                 temperature: float = 0.6,
                 retry_attempts: int = 3,
                 max_total_tokens: int = 4000): 

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retry_attempts = retry_attempts
        self.max_total_tokens = max_total_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

        self.gen_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=0.95,
            repetition_penalty=1.15,
        )
        self.llm = HuggingFacePipeline(pipeline=self.gen_pipeline)
        self.output_parser = CustomPydanticOutputParser(pydantic_object=MetaboliteAnalysis)

    def count_tokens(self, text: str) -> int:
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        return len(tokens)

    def extract_documents(self, context: str) -> List[Tuple[str, str]]:
        pattern = re.compile(r"(Sources:\s*.+?\nSection:\s*.+?\nContent:\s*)(.+?)(?=\n\nSources:|$)", re.DOTALL)
        docs = pattern.findall(context)
        return docs

    def truncate_content_for_document(self, content: str, max_tokens: int, tokenizer) -> str:
        tokens = tokenizer.encode(content, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return content
        truncated_tokens = tokens[:max_tokens]
        return tokenizer.decode(truncated_tokens, skip_special_tokens=True)

    def intelligent_truncate_context(self, full_context: str, max_allowed_tokens: int, tokenizer) -> str:
        prefix = ""
        context_body = full_context
        prefix_pattern = r"^(Additional context from the literature:\s*)"
        prefix_match = re.match(prefix_pattern, full_context, re.IGNORECASE)
        if prefix_match:
            prefix = prefix_match.group(1).strip()
            context_body = full_context[len(prefix_match.group(0)) :].strip()

        documents = self.extract_documents(context_body)

        metadata_tokens_counts = []
        for metadata, _ in documents:
            count = tokenizer.encode(metadata, add_special_tokens=False)
            metadata_tokens_counts.append(len(count))

        total_metadata_tokens = sum(metadata_tokens_counts)
        prefix_tokens = len(tokenizer.encode(prefix, add_special_tokens=False)) if prefix else 0

        available_for_contents = max_allowed_tokens - total_metadata_tokens - prefix_tokens
        if available_for_contents <= 0:
            available_for_contents = 0

        num_docs = len(documents)
        base_allocation = available_for_contents // num_docs
        extra_tokens = available_for_contents - (base_allocation * num_docs)

        truncated_docs = []
        for i, (metadata, content) in enumerate(documents):
            allocated_tokens = base_allocation
            if i < extra_tokens:
                allocated_tokens += 1

            truncated_content = self.truncate_content_for_document(content, allocated_tokens, tokenizer)
            doc_text = f"{metadata}{truncated_content}"
            truncated_docs.append(doc_text)

        final_context = ""
        if prefix:
            final_context += prefix + "\n\n"
        final_context += "\n\n".join(truncated_docs)
        return final_context

    def preprocess_metabolite_text(self, full_text: str) -> str:
        def extract_block(text: str, header: str) -> str:
            pattern = re.compile(rf"{header}\s*(.*?)\s*(?=###|$)", re.DOTALL)
            match = pattern.search(text)
            return match.group(1).strip() if match else ""
        name_block = extract_block(full_text, r"##\s*Metabolite:")
        description_block = extract_block(full_text, r"###\s*Description")
        description_sentences = re.split(r'(?<=[.!?])\s+', description_block)
        short_description = " ".join(description_sentences[:10])
        preprocessed = (
            f"{name_block}\n\n"
            f"Description: {short_description}\n\n"
        )
        return preprocessed

    def formulate_query_base(self, metabolite_text: str, disease_info: str, experiment_conditions: str) -> str:
        format_instructions = self.output_parser.get_format_instructions()
        base_query = (
            "[INST] <<SYS>>\n"
            f"{format_instructions}\n"
            "<</SYS>>\n\n"
            "Below is information about a metabolite obtained from database matching:\n\n"
            f"{metabolite_text}\n\n"
            "Information about the disease under study:\n"
            f"{disease_info}\n\n"
            "Experimental conditions:\n"
            f"{experiment_conditions}\n\n"
            "Analyze this data and provide a detailed annotation of the metabolite, including at least 300-500 words of scientific explanation.\n"
            "[/INST]\n"
        )
        return base_query

    def enrich_sources_with_metadata(self, sources: List[Source], source_docs) -> List[Source]:
        if not source_docs:
            return sources
        source_metadata_map = {}
        for doc in source_docs:
            doc_metadata = doc.get("doc_metadata", {})
            file_id = doc_metadata.get("file_id")
            if file_id:
                file_metadata_json = redis_client.get(f"file:{file_id}")
                if file_metadata_json:
                    file_metadata = json.loads(file_metadata_json)
                    title = file_metadata.get("title", "")
                    if title:
                        source_metadata_map[title] = {
                            "doi": file_metadata.get("doi", "N/A"),
                            "pmid": file_metadata.get("pmid", "N/A")
                        }
        enriched_sources = []
        for source in sources:
            source_title = source.source
            if source_title in source_metadata_map:
                metadata = source_metadata_map[source_title]
                source.doi = metadata["doi"]
                source.pmid = metadata["pmid"]
            else:
                for title in source_metadata_map:
                    if title.lower() in source_title.lower() or source_title.lower() in title.lower():
                        metadata = source_metadata_map[title]
                        source.doi = metadata["doi"]
                        source.pmid = metadata["pmid"]
                        break
            enriched_sources.append(source)
        return enriched_sources

    def process_metabolite_annotation(self, full_metabolite_text: str, disease_info: str, experiment_conditions: str) -> Tuple[dict, list]:
        simple_text = self.preprocess_metabolite_text(full_metabolite_text)
        query_base = self.formulate_query_base(simple_text, disease_info, experiment_conditions)
        base_token_count = self.count_tokens(query_base)
        source_docs = DocumentUtils.retrieve_documents(query_base, top_k=2)

        docs_context = "\n\n".join([
            f"Sources: {doc['file_metadata'].get('title', 'N/A')}\n"
            f"Section: {doc['doc_metadata'].get('section', 'N/A')}\n"
            f"Content: {doc['content']}" 
            for doc in source_docs
        ])
        extracted_context = f"Additional context from the literature:\n{docs_context}"
        full_query = query_base.replace("[/INST]", f"{extracted_context}\n[/INST]")
        total_tokens = self.count_tokens(full_query)

        if total_tokens > self.max_total_tokens:
            allowed_context_tokens = self.max_total_tokens - base_token_count + self.count_tokens("[/INST]")
            truncated_context = self.intelligent_truncate_context(extracted_context, allowed_context_tokens, self.tokenizer)
            final_prompt = query_base.replace("[/INST]", f"{truncated_context}\n[/INST]")
        else:
            final_prompt = full_query

        attempt = 0
        raw_response = ""
        while attempt < self.retry_attempts:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raw_response = self.llm.invoke(final_prompt).strip()
                if raw_response:
                    logger.info("Received response from LLM.")
                    break
                else:
                    logger.warning("Empty response, retrying...")
            except Exception as ex:
                logger.error("Error during LLM call: %s", ex)
            attempt += 1
            time.sleep(1)

        final_json_text = DocumentUtils.extract_json_from_code_block(raw_response)
        if not final_json_text:
            logger.error("Failed to extract JSON from LLM response.")
            return {"answer": -1, "reasoning": "", "sources": []}, source_docs
        try:
            structured_output = self.output_parser.parse(final_json_text)
            enriched_sources = self.enrich_sources_with_metadata(structured_output.sources, source_docs)
            structured_output.sources = enriched_sources
            structured_output.reasoning = structured_output.reasoning.strip() + f"\nModel: {self.model_name}"
            return structured_output.dict(), source_docs
        except Exception as parse_err:
            logger.error("Error parsing structured output: %s. Returning empty response.", parse_err)
            return {"answer": -1, "reasoning": "", "sources": []}, source_docs