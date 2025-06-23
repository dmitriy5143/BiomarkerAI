import os
import base64
import pdfkit
import logging

logger = logging.getLogger("Report")
logger.setLevel(logging.INFO)

def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/png;base64,{encoded}"
    except Exception as ex:
        logger.error("Ошибка при конвертации изображения %s: %s", image_path, ex)
        return None

def url_to_absolute_path(url):
    if url.startswith('/static/'):
        rel_path = url[len('/static/'):]  
        possible_paths = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", rel_path),
            os.path.join("/app", "static", rel_path),
            os.path.join("static", rel_path)]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return possible_paths[0]
    return url

def generate_pdf_report(result, pdf_path):
    plots_html = ""
    for plot in result.get("plots", []):
        img_url = plot.get("url", "")
        if not img_url:
            continue
        absolute_path = url_to_absolute_path(img_url)
        logger.info("Преобразование изображения с пути: %s", absolute_path)
        base64_img = image_to_base64(absolute_path)
        if base64_img:
            img_tag = (
                f'<div class="plot">'
                f'<img src="{base64_img}" alt="{plot.get("title", "График")}" style="max-width:100%;">'
                f'</div>'
            )
        else:
            img_tag = (
                f'<div class="plot">'
                f'<p>Изображение не найдено: {plot.get("title", "График")}</p>'
                f'</div>'
            )
        plots_html += img_tag

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Отчёт анализа</title>
<style>
    body {{ font-family: Arial, sans-serif; margin: 30px; }}
    h1 {{ text-align: center; }}
    .section {{ margin-bottom: 20px; }}
    .section-title {{ font-weight: bold; font-size: 1.2em; border-bottom: 1px solid #ccc; margin-bottom: 5px; padding-bottom: 3px; }}
    .plot {{ margin-bottom: 10px; }}
    .metabolite {{ margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid #eee; }}
</style>
</head>
<body>
<h1>Отчёт анализа обзорной метаболомики</h1>
<div class="section">
    <div class="section-title">Функция потерь</div>
    <p>{result.get("final_loss", "N/A")}</p>
</div>
<div class="section">
    <div class="section-title">Выбранные переменные</div>
    <p>{', '.join(map(str, result.get("selected_indices", [])))}</p>
</div>
<div class="section">
    <div class="section-title">Визуализация</div>
    {plots_html}
</div>
<div class="section">
    <div class="section-title">Результаты LLM анализа</div>
    {"".join([
        f'<div class="metabolite"><strong>Метаболит {i+1}:</strong><br>Ответ: {res.get("answer", "N/A")}<br>Размышления: {res.get("reasoning", "N/A")}</div>'
        for i, res in enumerate(result.get("detailed_results", []))
    ])}
</div>
</body>
</html>
"""

    try:
        options = {
            'encoding': "UTF-8",
            'enable-local-file-access': True,      
            'disable-smart-shrinking': True,          
            'quiet': ''                              
        }

        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        pdfkit.from_string(html, pdf_path, options=options)
        logger.info("PDF отчёт успешно сгенерирован и сохранён: %s", pdf_path)
        return True

    except Exception as e:
        logger.error("Ошибка генерации PDF отчёта: %s", e)
        return False