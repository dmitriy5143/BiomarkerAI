document.addEventListener("DOMContentLoaded", function () {
  const rebuildCheckbox = document.getElementById("rebuild_rag");
  const testModeFields = document.getElementById("test-mode-fields");
  rebuildCheckbox.addEventListener("change", function () {
    testModeFields.style.display = rebuildCheckbox.checked ? "block" : "none";
  });

  let finalResultsDisplayed = false;
  let llmResultBuffer = [];
  let finalPhaseTimeout;

  const form = document.getElementById("pipeline-form");
  form.addEventListener("submit", function (event) {
    event.preventDefault();
    document.getElementById("loading-spinner").style.display = "block";
    document.querySelector("button[type=submit]").disabled = true;
    resetUI();
    finalResultsDisplayed = false;
    llmResultBuffer = [];
    clearTimeout(finalPhaseTimeout);

    const formData = new FormData(form);
    console.log("Отправка формы на сервер...");
    fetch(form.action, {
      method: "POST",
      body: formData
    })
      .then(response => response.json())
      .then(data => {
        console.log("Получен ответ от сервера:", data);
        if (data.task_id) {
          console.log("Запуск отслеживания задачи:", data.task_id);
          pollTaskStatus(data.task_id);
        } else {
          console.log("Получен финальный результат без задачи Celery");
          renderFinalResult(data);
          document.getElementById("loading-spinner").style.display = "none";
          document.querySelector("button[type=submit]").disabled = false;
        }
      })
      .catch(error => {
        console.error("Ошибка при отправке формы:", error);
        document.getElementById("loading-spinner").style.display = "none";
        document.querySelector("button[type=submit]").disabled = false;
        alert("Ошибка: " + error);
      });
  });

  function resetUI() {
    document.getElementById("overall-progress-bar").style.width = "0%";
    document.getElementById("overall-progress-text").textContent = "0%";
    document.getElementById("current-archive-name").textContent = "";
    document.getElementById("current-archive-status").textContent = "";
    document.getElementById("llm-content").innerHTML = "";
    document.getElementById("loss-result").style.display = "none";
    document.getElementById("visualization-results").style.display = "none";
    document.getElementById("plots-container").innerHTML = "";
    document.getElementById("download-report").style.display = "none";
    document.getElementById("progress-container").style.display = "none";
  }

  function pollTaskStatus(taskID) {
    console.log("Начало опроса статуса задачи:", taskID);
    let currentPhase = null;
    let phaseProgress = {
      archive: { weight: 0.7, completed: 0 },
      embeddings: { weight: 0.3, completed: 0 }
    };

    const pollInterval = setInterval(() => {
      fetch(`/task_status/${taskID}`)
        .then(response => {
          if (!response.ok) {
            throw new Error("Статус ответа не OK: " + response.status);
          }
          return response.json();
        })
        .then(data => {
          console.log("Получено обновление статуса:", data);
          if (data.state === "PROGRESS" && data.info) {
            console.log("Обработка PROGRESS состояния, фаза:", data.info.phase);
            const progressContainer = document.getElementById("progress-container");
            if (progressContainer.style.display === "none") {
              progressContainer.style.display = "block";
            }
            if (data.info.phase !== currentPhase) {
              currentPhase = data.info.phase;
              console.log("Переход к фазе:", currentPhase);
            }

            if (data.info.phase === "archive" || data.info.phase === "archive_overall") {
              updateArchiveProgressUI(data.info, phaseProgress);
            }
            else if (data.info.phase === "extract") {
              document.getElementById("current-archive-status").textContent =
                `Извлечение файлов: ${data.info.current_file || 0} релевантных файлов найдено`;
            }
            else if (data.info.phase === "embeddings") {
              updateEmbeddingsProgressUI(data.info, phaseProgress);
            }
            else if (data.info.phase === "rag_completed") {
              showMessage("Данные для векторного хранилища подготовлены.", "green", false);
            }
            else if (data.info.phase === "feature_selection") {
              if (data.info.status === "start") {
                showMessage("Проводится выбор релевантных переменных...", "black", true);
              } else if (data.info.status === "completed") {
                showMessage("Переменные отобраны.", "green", true);
              }
            }
            else if (data.info.phase === "visualization") {
              if (data.info.status === "start") {
                showMessage("Создаются визуализации...", "black", true);
              } else if (data.info.status === "completed") {
                if (data.info.phase_id && data.info.phase_id === "visualization_completed") {
                  showMessage("Графики построены.", "green", true);
                  console.log("Структура данных фазы visualization:", JSON.stringify(data.info));
                  const lossValue = data.info.final_loss !== undefined ? data.info.final_loss : null;
                  const plots = Array.isArray(data.info.plots) ? data.info.plots : [];
                  if (lossValue !== null || (plots.length > 0 && plots[0].url)) {
                    try {
                      console.log("Рендеринг финальных результатов из фазы visualization");
                      renderFinalResult({
                        final_loss: lossValue,
                        plots: plots
                      });
                      finalResultsDisplayed = true;
                      while (llmResultBuffer.length > 0) {
                        let bufferedResult = llmResultBuffer.shift();
                        renderStreamingLLMResult(bufferedResult);
                      }
                    } catch (e) {
                      console.error("Ошибка отображения финальных результатов:", e);
                    }
                  } else {
                    console.warn("Получены данные визуализации, но loss или plots не соответствуют ожидаемому формату.");
                  }
                } else {
                  console.log("Обновление visualization получено, но phase_id не соответствует завершённой фазе.");
                }
              }
            }
            else if (data.info.phase === "llm_analysis") {
              if (data.info.status === "start") {
                showMessage("Начинается анализ метаболитов...", "black", true);
              }
              else if (data.info.status === "progress") {
                if (finalResultsDisplayed) {
                  if (data.info.result !== undefined) {
                    renderStreamingLLMResult(data.info.result);
                  }
                } else {
                  if (data.info.result !== undefined) {
                    llmResultBuffer.push(data.info.result);
                    console.log("LLM результаты получены, но ожидается завершение предыдущей фазы.");
                  }
                }
              }
              else if (data.info.status === "completed") {
                showMessage("Анализ метаболитов завершен.", "green", true);
                finalResultsDisplayed = true;
                while (llmResultBuffer.length > 0) {
                  let bufferedResult = llmResultBuffer.shift();
                  renderStreamingLLMResult(bufferedResult);
                }
              }
            }
            else {
              console.warn("Неизвестная фаза:", data.info.phase);
            }
          }

          if (data.state === "SUCCESS") {
            console.log("Задача успешно завершена, результат:", data.result);
            clearInterval(pollInterval);
            if (data.result) {
              try {
                renderFinalResult(data.result);
                // PDF-отчет обновляем, если pdf_report_url присутствует в финальном результате
                if (data.result.pdf_report_url &&
                    data.result.pdf_report_url !== "null" &&
                    data.result.pdf_report_url !== "None") {
                  updatePdfReportBlock(data.result);
                }
              } catch (e) {
                console.error("Ошибка рендеринга финального результата:", e);
              }
              finalResultsDisplayed = true;
              while (llmResultBuffer.length > 0) {
                let bufferedResult = llmResultBuffer.shift();
                renderStreamingLLMResult(bufferedResult);
              }
            }
            document.getElementById("loading-spinner").style.display = "none";
            document.querySelector("button[type=submit]").disabled = false;
          }

          if (data.state === "FAILURE") {
            console.error("Задача завершилась с ошибкой:", data.info);
            clearInterval(pollInterval);
            alert("Произошла ошибка при выполнении анализа.");
            document.getElementById("loading-spinner").style.display = "none";
            document.querySelector("button[type=submit]").disabled = false;
          }

        })
        .catch(err => {
          console.error("Ошибка опроса состояния: " + err);
          clearInterval(pollInterval);
          document.getElementById("loading-spinner").style.display = "none";
          document.querySelector("button[type=submit]").disabled = false;
        });
    }, 1000);
  }

  function updatePdfReportBlock(result) {
    // Обновляем блок только если pdf_report_url корректна
    if (result.pdf_report_url && result.pdf_report_url !== "null" && result.pdf_report_url !== "None") {
      const downloadBlock = document.getElementById("download-report");
      if (downloadBlock) {
        downloadBlock.style.display = "block";
        const link = document.getElementById("pdf-download-link");
        if (link) {
          link.href = result.pdf_report_url;
          console.log("PDF отчет обновлён, ссылка установлена:", link.href);
        } else {
          console.error("Элемент с ID 'pdf-download-link' не найден в DOM");
        }
      } else {
        console.error("Элемент с ID 'download-report' не найден в DOM");
      }
    } else {
      document.getElementById("download-report").style.display = "none";
      console.log("PDF отчет недоступен или ссылка некорректна:", result.pdf_report_url);
    }
  }

  function updateArchiveProgressUI(info, phaseProgress) {
    if (info.phase === "archive") {
      const archiveName = (info.message && info.message.includes("Обработка архива"))
        ? info.message.split("Обработка архива ")[1].split(" (дата:")[0]
        : `Архив ${info.archive_number || ''}`;
      console.log("Обновление текущего архива:", archiveName, info.message);
      document.getElementById("current-archive-name").textContent = archiveName;
      document.getElementById("current-archive-status").textContent = info.message || "Обработка архива...";
      if (info.archive_number && info.total_archives) {
        phaseProgress.archive.completed = info.archive_number / info.total_archives;
        if (info.archive_number === info.total_archives && info.status === "completed") {
          phaseProgress.archive.completed = 1.0;
        }
      }
    }
    if (info.phase === "archive_overall") {
      if (info.status === "start") {
        document.getElementById("current-archive-name").textContent = "Обработка архивов";
        document.getElementById("current-archive-status").textContent = "Начинается обработка архивов...";
        phaseProgress.archive.completed = 0;
      } else if (info.status === "completed" || info.status === "limiting") {
        document.getElementById("current-archive-name").textContent = "Обработка архивов";
        document.getElementById("current-archive-status").textContent = info.message;
        if (info.status === "completed")
          phaseProgress.archive.completed = 1.0;
      }
    }
    updateTotalProgress(
      phaseProgress.archive.completed * phaseProgress.archive.weight +
      phaseProgress.embeddings.completed * phaseProgress.embeddings.weight
    );
  }

  function updateEmbeddingsProgressUI(info, phaseProgress) {
    document.getElementById("current-archive-name").textContent = "Векторное хранилище";
    document.getElementById("current-archive-status").textContent = info.message || "Создание векторного хранилища...";
    if (info.status === "start") {
      phaseProgress.embeddings.completed = 0.1;
    } else if (info.status === "progress") {
      phaseProgress.embeddings.completed = 0.5;
    } else if (info.status === "completed") {
      phaseProgress.embeddings.completed = 1.0;
    }
    updateTotalProgress(
      phaseProgress.archive.completed * phaseProgress.archive.weight +
      phaseProgress.embeddings.completed * phaseProgress.embeddings.weight
    );
  }

  function updateTotalProgress(progressValue) {
    progressValue = Math.max(0, Math.min(1, progressValue));
    const percent = Math.floor(progressValue * 100);
    console.log("Обновление общего прогресса:", percent, "%");
    document.getElementById("overall-progress-bar").style.width = `${percent}%`;
    document.getElementById("overall-progress-text").textContent = `${percent}%`;
  }

  function showMessage(message, color, keep) {
    const progressContainer = document.getElementById("progress-container");
    progressContainer.style.display = "block";
    progressContainer.innerHTML = `<h4 style="text-align:center; color: ${color};">${message}</h4>`;
    // Если не требуется держать сообщение на экране долго, можно его убрать через некоторое время
    if (!keep) {
      setTimeout(() => {
        progressContainer.innerHTML = "";
      }, 3000);
    }
  }

  function renderStreamingLLMResult(result) {
    if (result === undefined) return;
    console.log("Рендеринг стриминг-результата LLM:", result);
    try {
      const llmContent = document.getElementById("llm-content");
      document.getElementById("llm-results").style.display = "block";
      const existingItem = document.querySelector(`.llm-result-item[data-index="${result.index}"]`);
      if (existingItem) {
        updateLLMResultItem(existingItem, result);
      } else {
        const resultItem = createLLMResultItem(result);
        llmContent.insertBefore(resultItem, llmContent.firstChild);
      }
    } catch (e) {
      console.error("Ошибка отображения LLM результата:", e);
    }
  }

  function createLLMResultItem(result) {
    const container = document.createElement('div');
    container.className = 'llm-result-item';
    container.setAttribute('data-index', result.index);
    const header = document.createElement('h4');
    header.textContent = `Метаболит ${result.index + 1}`;
    container.appendChild(header);
    if (result.answer && result.answer.trim() !== "") {
      const answerContainer = document.createElement('div');
      answerContainer.className = 'metabolite-answer';
      const answerLabel = document.createElement('strong');
      answerLabel.textContent = 'Ответ: ';
      answerContainer.appendChild(answerLabel);
      const answerText = document.createElement('span');
      answerText.textContent = result.answer;
      answerContainer.appendChild(answerText);
      container.appendChild(answerContainer);
    }
    if (result.reasoning && result.reasoning.trim() !== "") {
      const reasoningContainer = document.createElement('div');
      reasoningContainer.className = 'metabolite-analysis';
      const reasoningLabel = document.createElement('strong');
      reasoningLabel.textContent = 'Размышления:';
      reasoningContainer.appendChild(reasoningLabel);
      const reasoningContent = document.createElement('div');
      reasoningContent.className = 'analysis-content';
      result.reasoning.split('\n').forEach(paragraph => {
        if (paragraph.trim()) {
          const p = document.createElement('p');
          p.textContent = paragraph.trim();
          reasoningContent.appendChild(p);
        }
      });
      reasoningContainer.appendChild(reasoningContent);
      container.appendChild(reasoningContainer);
    }
    return container;
  }

  function updateLLMResultItem(element, result) {
    if (result.answer && result.answer.trim() !== "") {
      let answerContainer = element.querySelector('.llm-result-item .metabolite-answer');
      if (!answerContainer) {
        answerContainer = document.createElement('div');
        answerContainer.className = 'metabolite-answer';
        const answerLabel = document.createElement('strong');
        answerLabel.textContent = 'Ответ: ';
        answerContainer.appendChild(answerLabel);
        const answerText = document.createElement('span');
        answerContainer.appendChild(answerText);
        element.appendChild(answerContainer);
      }
      const answerText = element.querySelector('.metabolite-answer span');
      answerText.textContent = result.answer;
    }
    if (result.reasoning && result.reasoning.trim() !== "") {
      let reasoningContainer = element.querySelector('.llm-result-item .metabolite-analysis');
      if (!reasoningContainer) {
        reasoningContainer = document.createElement('div');
        reasoningContainer.className = 'metabolite-analysis';
        const reasoningLabel = document.createElement('strong');
        reasoningLabel.textContent = 'Размышления:';
        reasoningContainer.appendChild(reasoningLabel);
        const reasoningContent = document.createElement('div');
        reasoningContent.className = 'analysis-content';
        reasoningContainer.appendChild(reasoningContent);
        element.appendChild(reasoningContainer);
      }
      const reasoningContent = element.querySelector('.analysis-content');
      reasoningContent.innerHTML = '';
      result.reasoning.split('\n').forEach(paragraph => {
        if (paragraph.trim()) {
          const p = document.createElement('p');
          p.textContent = paragraph.trim();
          reasoningContent.appendChild(p);
        }
      });
    }
  }

  function renderFinalResult(result) {
    console.log("Рендеринг финального результата:", result);
    try {
      if (result.final_loss !== undefined) {
        const lossBlock = document.getElementById("loss-result");
        lossBlock.style.display = "block";
        document.getElementById("loss-value").textContent = Number(result.final_loss).toFixed(4);
      } else {
        document.getElementById("loss-result").style.display = "none";
      }
    } catch (e) {
      console.error("Ошибка отображения loss:", e);
    }
    try {
      if (result.plots && result.plots.length > 0) {
        document.getElementById("visualization-results").style.display = "block";
        const plotsContainer = document.getElementById("plots-container");
        plotsContainer.innerHTML = '';
        result.plots.forEach(plot => {
          const plotContainer = document.createElement('div');
          plotContainer.className = 'plot-container';
          if (plot.title) {
            const title = document.createElement('h5');
            title.textContent = plot.title;
            title.className = 'plot-title';
            plotContainer.appendChild(title);
          }
          const img = document.createElement('img');
          img.src = plot.url;
          img.alt = plot.title || 'График';
          img.className = 'result-plot';
          img.onerror = function () {
            console.error("Не удалось загрузить изображение:", plot.url);
          };
          plotContainer.appendChild(img);
          plotsContainer.appendChild(plotContainer);
        });
      } else {
        document.getElementById("visualization-results").style.display = "none";
      }
    } catch (e) {
      console.error("Ошибка отображения графиков:", e);
    }
    finalResultsDisplayed = true;
    while (llmResultBuffer.length > 0) {
      let bufferedResult = llmResultBuffer.shift();
      renderStreamingLLMResult(bufferedResult);
    }
    if (result.detailed_results && result.detailed_results.length > 0) {
      const llmContent = document.getElementById("llm-content");
      if (!llmContent.innerHTML.trim()) {
        result.detailed_results.forEach((res, index) => {
          if (typeof res.index === "undefined") res.index = index;
          renderStreamingLLMResult(res);
        });
      }
      document.getElementById("llm-results").style.display = "block";
    } else {
      document.getElementById("llm-results").style.display = "none";
    }
    // Обновление PDF-отчёта задержано до полного завершения анализа
    if (result.pdf_report_url &&
        result.pdf_report_url !== "null" &&
        result.pdf_report_url !== "None") {
      updatePdfReportBlock(result);
    }
  }
});