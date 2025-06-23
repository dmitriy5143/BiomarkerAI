#!/bin/bash
set -e
echo "Запуск BiomarkerAI..."
exec gunicorn --bind 0.0.0.0:8080 --timeout 1200 "app:create_app()"