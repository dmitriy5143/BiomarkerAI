import os
import time
import json
import re
import pandas as pd
import torch
import logging
from pipeline.data_loader import DataLoader
from pipeline.genetic import GeneticFeatureSelector
from pipeline.visualizer import Visualizer
from pipeline.metabolite_formatter import MetaboliteFormatter
from pipeline.llm_agent import OptimizedRAGLLMAgent
from pipeline.rag import build_rag_system
from pipeline.article_parser import process_archives_sequentially, update_keywords
from config import config

logger = logging.getLogger("RAGBuilder")
current_config = config[os.environ.get("FLASK_CONFIG", "production")]

def execute_pipeline(params: dict):

    try:
        hmdb_df = pd.read_pickle(current_config.HMDB_DATABASE_PATH)
        logger.info("База данных загружена из %s", current_config.HMDB_DATABASE_PATH)
    except Exception as ex:
        logger.error("Ошибка загрузки HMDB базы: %s", ex)
        raise ex

    excel_path = params.get("excel_path")
    class_zero = params.get("class_zero")
    disease_info = params.get("disease_info")
    experiment_conditions = params.get("experiment_conditions")
    population_size = params.get("population_size", 100)
    n_generations = params.get("n_generations", 10)
    rebuild_rag = params.get("rebuild_rag", False)
    new_keywords = params.get("keywords") 
    max_archives = params.get("max_archives") 

    if not excel_path or class_zero is None:
        raise ValueError("Не заданы обязательные параметры: excel_path и class_zero")

    if new_keywords:
        update_keywords(new_keywords)
        logger.info("Ключевые слова обновлены пользователем")

    if rebuild_rag:
        logger.info("Запущен режим пересборки RAG-системы")
        process_archives_sequentially(max_archives=max_archives)
        vs = build_rag_system()
        if vs is None:
            logger.warning("Не удалось пересобрать RAG-систему. Продолжаем с текущим индексом.")
    else:
        logger.info("Используется существующее векторное хранилище")

    data_loader = DataLoader()
    X, y, hmdb_ids = data_loader.load_data(excel_path, class_zero)
    logger.info("Данные для анализа загружены")

    selector = GeneticFeatureSelector(X.to_numpy(), y.to_numpy(), population_size=population_size, n_generations=n_generations) 
    best_features_mask, final_loss = selector.select_features()
    logger.info(f"Значение лосс функции:{final_loss}")

    X = X.iloc[:, best_features_mask]
    hmdb_ids = hmdb_ids.iloc[best_features_mask]
    logger.info(f"Индексы выбранных переменных:{X.columns.tolist()}")

    viz = Visualizer(X, y)
    plots = []
    predictions_plot = viz.plot_predictions()
    coefficients_plot = viz.plot_coefficients() 
    plots.append(predictions_plot)
    plots.append(coefficients_plot)

    metabolites_df = data_loader.get_metabolite_info(hmdb_df, hmdb_ids)
    if metabolites_df.empty:
        err_msg = "Нет данных для заданных HMDBID."
        logger.error(err_msg)
        return {"error": err_msg}, None

    formatter = MetaboliteFormatter()
    formatted_info = formatter.format_metabolite_info_for_llm(metabolites_df)
    logger.info("Данные для LLM отформатированы.")
    metabolite_blocks = formatter.split_metabolite_blocks(formatted_info)
    if not metabolite_blocks:
        err_msg = "Не удалось выделить блоки метаболитов"
        logger.error(err_msg)
        return {"error": err_msg}, None

    hmdb_to_column_index = {}
    for col in X.columns:
        col_idx = int(col)
        hmdb_id_value = hmdb_ids.get(col, None)
        if pd.isna(hmdb_id_value) or str(hmdb_id_value).strip() == "":
            continue
        parts = [part.strip() for part in re.split(r"[,;\s]+", str(hmdb_id_value)) if part.strip()]
        for part in parts:
            normalized_id = part if part.startswith("HMDB") else f"HMDB{part}"
            hmdb_to_column_index[normalized_id] = col_idx

    agent = OptimizedRAGLLMAgent()
    detailed_results = []
    for idx, block in enumerate(metabolite_blocks, 1):
        hmdb_ids_in_block = re.findall(r"\(?(HMDB\d+)\)?", block)
        indices = []
        for hmdb in hmdb_ids_in_block:
            indices.append(hmdb_to_column_index[hmdb])
        indices = sorted(set(indices))
        logger.info(f"Обработка метаболита {indices}:")
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if idx > 1:
                time.sleep(2)
            result, _ = agent.process_metabolite_annotation(block, disease_info, experiment_conditions)
            result["column_index"] = indices
            detailed_results.append(result)
            logger.info(f"Результат для метаболита {indices} обработан успешно.")
        except Exception as ex:
            logger.error("Ошибка при обработке блока: %s", ex)
            continue

    output_file = "detailed_metabolite_annotations.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)
    logger.info("Аннотации сохранены в файл %s", output_file)
    return {"detailed_results": detailed_results, "final_loss": final_loss.tolist(), "selected_indices": X.columns.tolist(), "plots": plots }, None