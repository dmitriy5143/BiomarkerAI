import os
import time
import json
import re
import logging
import pandas as pd
import torch

from pipeline.data_loader import DataLoader
from pipeline.genetic import GeneticFeatureSelector
from pipeline.dmbde import DMBDE
from pipeline.psova2 import PSOVA2 
from pipeline.visualizer import Visualizer
from pipeline.metabolite_formatter import MetaboliteFormatter
from pipeline.llm_agent import OptimizedRAGLLMAgent
from pipeline.rag import build_rag_system
from pipeline.article_parser import process_archives_with_progress, update_keywords
from report import generate_pdf_report
from config import config

logger = logging.getLogger("RAGBuilder")
current_config = config[os.environ.get("FLASK_CONFIG", "production")]

def execute_pipeline(params: dict, update_progress_fn=None, llm_progress_fn=None):
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
        if update_progress_fn:
            update_progress_fn({
                "phase": "archive_overall",
                "status": "start",
                "message": "Начинается обработка архивов."
            })
        process_archives_with_progress(max_archives=max_archives, update_progress_fn=update_progress_fn)

        if update_progress_fn:
            update_progress_fn({
                "phase": "embeddings",
                "status": "start",
                "message": "Начинается создание векторного хранилища (эмбеддингов)..."
            })
        start_time = time.time()
        vs = build_rag_system()
        elapsed = time.time() - start_time
        if vs is None:
            logger.warning("Не удалось пересобрать RAG-систему. Продолжаем с текущим индексом.")
        else:
            logger.info("Векторное хранилище успешно создано за %.2f секунд.", elapsed)
            if update_progress_fn:
                update_progress_fn({
                    "phase": "embeddings",
                    "status": "completed",
                    "message": f"Векторное хранилище создано за {elapsed:.2f} секунд."
                })
    else:
        logger.info("Используется существующее векторное хранилище")

    if update_progress_fn:
        update_progress_fn({
            "phase": "rag_completed",
            "status": "success",
            "message": "Данные для векторного хранилища подготовлены."
        })
    time.sleep(1)
    data_loader = DataLoader()
    X, y, hmdb_ids = data_loader.load_data(excel_path, class_zero)
    logger.info("Данные для анализа загружены")

    if update_progress_fn:
        update_progress_fn({
            "phase": "feature_selection",
            "status": "start",
            "message": "Проводится выбор релевантных переменных..."
        })
        
    feature_selector_param = params.get("feature_selector", "vipga").lower()
    if feature_selector_param == "dmbde":
        try:
            SelectorClass = DMBDE
            logger.info("Используется алгоритм DMBDE")
        except ImportError:
            logger.error("Модуль DMBDE не найден. Используется GeneticFeatureSelector.")
            SelectorClass = GeneticFeatureSelector
    elif feature_selector_param == "psova2":
        try:
            SelectorClass = PSOVA2
            logger.info("Используется алгоритм PSOVA2")
        except ImportError:
            logger.error("Модуль PSOVA2 не найден. Используется GeneticFeatureSelector.")
            SelectorClass = GeneticFeatureSelector
    else:
        SelectorClass = GeneticFeatureSelector
        logger.info("Используется алгоритм VIP-GA (GeneticFeatureSelector)")

    selector = SelectorClass(X.to_numpy(), y.to_numpy(), population_size=population_size, n_generations=n_generations)
    result_dict = selector.select_features()
    best_features_mask = result_dict["mask"]
    final_loss = result_dict["loss"]
    entropy_history = result_dict.get("entropy_history")
    algorithm_name = result_dict.get("algorithm_name", "Unknown")
    logger.info(f"Алгоритм: {algorithm_name}, Лосс функции: {final_loss}")

    if update_progress_fn:
        update_progress_fn({
            "phase": "feature_selection",
            "status": "completed",
            "message": "Переменные отобраны.",
        })

    X = X.iloc[:, best_features_mask]
    hmdb_ids = hmdb_ids.iloc[best_features_mask]
    logger.info("Выбранные переменные: %s", X.columns.tolist())

    if update_progress_fn:
        update_progress_fn({
            "phase": "visualization",
            "status": "start",
            "message": "Создание визуализаций...",
        })

    viz = Visualizer(X, y)
    plots = []
    predictions_plot = viz.plot_predictions()
    coefficients_plot = viz.plot_coefficients()
    plots.append(predictions_plot)
    plots.append(coefficients_plot)
    if entropy_history is not None and len(entropy_history) > 0:
        entropy_plot = viz.plot_entropy_evolution(entropy_history)
        plots.append(entropy_plot)
        logger.info("Добавлен график эволюции энтропии популяции")

    if update_progress_fn:
        update_progress_fn({
            "phase": "visualization",
            "status": "completed",
            "phase_id": "visualization_completed", 
            "message": f"Переменные отобраны. Значение лосс функции: {final_loss}. Графики построены.",
            "final_loss": final_loss.tolist() if hasattr(final_loss, "tolist") else final_loss,
            "plots": plots
        })
    time.sleep(1.5) 

    metabolites_df = data_loader.get_metabolite_info(hmdb_df, hmdb_ids)
    if metabolites_df.empty:
        err_msg = "Нет данных для заданных HMDBID."
        logger.error(err_msg)
        return {"error": err_msg}, None

    formatter = MetaboliteFormatter()
    formatted_info = formatter.format_metabolite_info_for_llm(metabolites_df)
    logger.info("Данные для LLM отформатированы")
    metabolite_blocks = formatter.split_metabolite_blocks(formatted_info)
    if not metabolite_blocks:
        err_msg = "Не удалось выделить блоки метаболитов"
        logger.error(err_msg)
        return {"error": err_msg}, None

    if update_progress_fn:
        update_progress_fn({
            "phase": "llm_analysis",
            "status": "start",
            "message": "Начинается анализ метаболитов...",
            "total_metabolites": len(metabolite_blocks)
        })

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
        indices = [hmdb_to_column_index[hmdb] for hmdb in hmdb_ids_in_block if hmdb in hmdb_to_column_index]
        indices = sorted(set(indices))
        logger.info("Обработка метаболита: %s", indices)
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if idx > 1:
                time.sleep(2)
            result_llm, _ = agent.process_metabolite_annotation(block, disease_info, experiment_conditions)
            result_llm["column_index"] = indices
            result_llm["index"] = idx - 1 
            if llm_progress_fn:
                llm_progress_fn(result_llm, idx - 1)
            detailed_results.append(result_llm)
            logger.info("Метаболит %s успешно обработан.", indices)
            if update_progress_fn:
                update_progress_fn({
                    "phase": "llm_analysis",
                    "status": "progress",
                    "message": f"Обработано метаболитов: {idx}/{len(metabolite_blocks)}",
                    "current": idx,
                    "total": len(metabolite_blocks),
                    "result": result_llm  
                })
        except Exception as ex:
            logger.error("Ошибка обработки блока: %s", ex)
            continue

    with open("detailed_metabolite_annotations.json", "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)
    logger.info("Аннотации сохранены в файл %s", "detailed_metabolite_annotations.json")

    time.sleep(1.5)
    if update_progress_fn:
        update_progress_fn({
            "phase": "llm_analysis",
            "status": "completed",
            "message": "Анализ метаболитов завершен."
        })

    result = {
        "detailed_results": detailed_results,
        "final_loss": final_loss.tolist() if hasattr(final_loss, "tolist") else final_loss,
        "selected_indices": X.columns.tolist(),
        "plots": plots
    }

    report_dir = os.path.join("static", "reports")
    os.makedirs(report_dir, exist_ok=True)
    pdf_report_path = os.path.join(report_dir, "analysis_report.pdf")
    if generate_pdf_report(result, pdf_report_path):
        result["pdf_report_url"] = "/" + pdf_report_path.replace(os.path.sep, "/")
    else:
        result["pdf_report_url"] = None

    return result, None