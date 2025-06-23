import os
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory
from config import config

def create_app(config_name="production"):
    app = Flask(__name__,
                instance_relative_config=True,
                template_folder="templates",
                static_folder="static")
    app.config.from_object(config[config_name])

    directories = [
        app.config.get("LOG_DIR", "logs"),
        app.config.get("DOWNLOAD_DIR", "oa_noncomm_xml_archives"),
        os.path.join(app.config.get("DOWNLOAD_DIR", "oa_noncomm_xml_archives"), "extracted"),
        os.path.join(app.config.get("DOWNLOAD_DIR", "oa_noncomm_xml_archives"), "processed"),
        os.path.join(app.config.get("DOWNLOAD_DIR", "oa_noncomm_xml_archives"), "filtered"),
        os.path.join("static", "plots"),
        os.path.join("static", "reports")
    ]
    if app.config.get("FAISS_INDEX_PATH"):
        faiss_dir = os.path.dirname(app.config.get("FAISS_INDEX_PATH"))
        if faiss_dir and faiss_dir != ".":
            directories.append(faiss_dir)
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Создана директория: {directory}")
    os.makedirs(app.instance_path, exist_ok=True)

    log_dir = app.config.get("LOG_DIR", "logs")
    logging.basicConfig(level=app.config.get("LOG_LEVEL"))
    logger = logging.getLogger("RAGBuilder")
    file_handler = logging.FileHandler(os.path.join(log_dir, "app.log"))
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    app.logger = logger

    app.config.setdefault('CELERY_BROKER_URL', 'redis://localhost:6379/0')
    app.config.setdefault('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

    register_routes(app)
    return app

def register_routes(app):
    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html")

    @app.route("/run", methods=["POST"])
    def run_pipeline():
        try:
            excel_file = request.files.get("excel_file")
            class_zero = request.form.get("class_zero", type=int)
            disease_info = request.form.get("disease_info")
            experiment_conditions = request.form.get("experiment_conditions")
            population_size = request.form.get("population_size", default=100, type=int)
            n_generations = request.form.get("n_generations", default=10, type=int)
            rebuild_rag = request.form.get("rebuild_rag") == "on"
            max_archives = request.form.get("max_archives", type=int)
            keywords_input = request.form.get("keywords")
            feature_selector = request.form.get("feature_selector", "vipga")
            new_keywords = None
            if keywords_input:
                new_keywords = {}
                for group in keywords_input.split(";"):
                    if ":" in group:
                        key, words = group.split(":", 1)
                        new_keywords[key.strip()] = [w.strip() for w in words.split(",") if w.strip()]
            if not excel_file or excel_file.filename == "":
                return jsonify({"error": "Не выбран файл с данными."}), 400
            if class_zero is None:
                return jsonify({"error": "Не указано количество образцов класса 0."}), 400
            excel_path = os.path.join(app.instance_path, excel_file.filename)
            excel_file.save(excel_path)
            params = {
                "excel_path": excel_path,
                "class_zero": class_zero,
                "disease_info": disease_info,
                "experiment_conditions": experiment_conditions,
                "population_size": population_size,
                "n_generations": n_generations,
                "rebuild_rag": rebuild_rag,
                "keywords": new_keywords,
                "max_archives": max_archives,
                "feature_selector": feature_selector
            }
            from celery_app import celery  
            task = celery.send_task("tasks.process_pipeline_task", args=[params])
            return jsonify({"task_id": task.id}), 202
        except Exception as ex:
            app.logger.error("Ошибка при запуске пайплайна: %s", ex)
            return jsonify({"error": str(ex)}), 500

    @app.route("/task_status/<task_id>", methods=["GET"])
    def task_status(task_id):
        from celery_app import celery
        task = celery.AsyncResult(task_id)
        if task.state == 'PENDING':
            response = {'state': task.state, 'status': 'Задача в очереди...'}
        elif task.state == 'PROGRESS':
            response = {'state': task.state, 'info': task.info}
        elif task.state == 'SUCCESS':
            response = {'state': task.state, 'result': task.result}
        else:
            response = {'state': task.state, 'status': str(task.info)}
        return jsonify(response)

    @app.route('/static/reports/<filename>', methods=['GET'])
    def download_report(filename):
        reports_dir = os.path.join(app.static_folder, 'reports')
        return send_from_directory(reports_dir, filename, as_attachment=True)

if __name__ == "__main__":
    import os
    config_name = os.environ.get("FLASK_CONFIG", "production")
    app = create_app(config_name)
    app.run(host="0.0.0.0", port=8080, debug=app.config.get("DEBUG"))