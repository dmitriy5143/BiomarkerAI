#!/bin/bash
set -e

echo "Запуск скрипта для загрузки и обработки данных HMDB..."
python scripts/download_hmdb.py

echo "Запуск приложения через docker-compose..."
docker-compose up --build -d

echo "Установка завершена. Откройте браузер и перейдите по адресу: http://localhost:8080"