# BiomarkerAI
BiomarkerAI — инновационная платформа для комплексного анализа, автоматического отбора и предварительной аннотации биомаркеров заболеваний. Решение предоставляет исследователям мощный инструментарий для эффективной обработки экспериментальных данных в области обзорной метаболомики, поддерживая широкий спектр методов спектрального анализа, включая GC-MS, LC-MS и MS/MS.

Модульная архитектура платформы обеспечивает возможность гибкого расширения функциональности для интеграции с другими физико-химическими методами, такими как NIR и потенциометрические мультисенсорные системы, что позволяет адаптировать приложение под специфические требования различных исследовательских проектов.

# Ключевые возможности
- **Интеграция с базами данных**: В текущей версии BiomarkerAI интегрирован с базой данных HMDB (Human Metabolome Database), что соответствует классическим практикам идентификации метаболитов.
- **Алгоритмы выделения биомаркеров**: Выделение биомаркеров исследуемого заболевания проводится с использованием встроенных методов эволюционной оптимизации. Для оценки качества различения групп здоровых пациентов от больных применяется метод PLS-DA.
- **Автоматический анализ литературы**: Приложение оснащено встроенным механизмом парсинга и фильтрации релевантных публикаций с PubMed, что позволяет автоматически формировать векторное хранилище структурированной информации. Механизм оснащен этапом управления памяти.
- **Семантический анализ данных**: Для систематизации и семантического анализа собранных данных применяется современная языковая модель Mistral-7B.
- **Автоматическая генерация отчетов**: Результаты анализа автоматически генерируются в виде структурированного PDF-отчета, что позволяет удобно изучать информацию о найденной литературе и первичную аннотацию идентифицированных метаболитов.

# Технологические особенности
Разработанное на базе Flask и упакованное в Docker-контейнеры, BiomarkerAI обеспечивает простое развертывание и использование для пользователей без специальных технических навыков.

Особую ценность представляют инновационные алгоритмы отбора биомаркеров:
- **VIP-GA**: Гибридный генетический алгоритм, разработанный автором данной работы
- **PSOVA2**: Алгоритм, описанный в литературе (https://doi.org/10.3390/s21051816)
- **DMBDE**: Алгоритм, описанный в литературе (https://doi.org/10.1016/j.ins.2022.12.117)

Уникальность BiomarkerAI заключается в гармоничном сочетании классических практик аннотации потенциально релевантных метаболитов с современными технологиями автоматизации, что позволяет использовать приложение без предъявления высоких требований к программному обеспечению пользователя. BiomarkerAI является первой попыткой автоматизации анализа экспериментальных данных непосредственно в лабораторных условиях, что значительно ускоряет исследовательский процесс. Для базового использования приложения не требуются специализированные знания в области анализа данных и программирования, что делает его доступным для широкого круга исследователей и специалистов в области химии, медицины и биологии.

# Предварительные требования
- Docker Desktop вы можете установить по ссылке: (https://www.docker.com/products/docker-desktop/).
- Git Вы можете установить по ссылке: (https://git-scm.com/downloads).

# Шаги установки
1. Клонируйте репозиторий при помощи команды: git clone https://github.com/dmitriy5143/BiomarkerAI.git
2. Перейдите в созданную директорию проекта: cd BiomarkerAI
3. Запустите скрипт установки:
   - Для Windows (PowerShell): bash install.sh
   - Для Linux/macOS: ./install.sh

Данный скрипт выполнит скачивание и обработку содержимого базы данных HMDB и поднимет контейнеры через docker-compose. После ввода команд доступ к приложению можно получить по адерсу: http://localhost:8080.

# Использование
После запуска приложения вы можете:

1. Экспортировать результаты вашего исследования из программного обеспечения прибора в формате Excel и загрузить файл в веб-интерфейс приложения.
2. Собрать векторную базу данных специализированную под вашу исследовательскую задачу, указав набор ключевых слов для фильтрации статей.
3. Выполнить поиск потенциальных биомаркеров заболевания при помощи алгоритмов VIP-GA, DMBDE, PSOVA2.
4. Визуализировать точность дискриминации групп на отобранных переменных с помощью встроенных инструментов.
5. Экспортировать результаты отбора и предварительной аннотации метаболитов в pdf отчет, ссылка на скачивание которого будет доступна в веб-интерфейсе по окончании анализа.

# Обновление приложения
Для обновления приложения до последней версии введите следующие команды:
- git pull
- bash install.sh
 
# Кастомизация
- Для добавления дополнительной информации о метаболитах из HMDB, вам необходимо включить соответвующие заголовки в файл metabolite_formatter.py, а также в файле llm.agent.py обновить функцию extract_block.
- Если вам необходимо использовать данное приложение для анализа данных не медицинской направленности, вам необходимо обеспечить доступ к базе данных (вместо текущей HMDB) специализируемой на вашем классе задач.
- Параметры выделенной памяти для парсера PubMed пбуликаций вы можете задать в конфигурации приложения, по умолчанию лишь 1гб вашего диска будет заполнен релевантными файлами статей.
- Приложение содержит ограничение на размер скачиваемого с PubMed архива при необходимости его можно изменить в конфигурации приложения. 

# Рекомендации по использованию 
- **VIP-GA**: Рекомендуется устанавливать 100 итераций и 200 популяций.
- **DMBDE**: В соответствии с рекомендациями авторов, оптимальные параметры: 100 итераций и 100 популяций.
- **PSOVA2**: В соответствии с рекомендациями авторов, оптимальные параметры: 30 популяций и 200 итераций.
В случае избыточности данных настроек сработает критерий предварительной остановки.
