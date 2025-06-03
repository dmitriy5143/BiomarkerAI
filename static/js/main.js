document.getElementById("pipeline-form").addEventListener("submit", function(event) {
  event.preventDefault();
  document.getElementById("loading-spinner").style.display = "block";
  document.querySelector("button[type=submit]").disabled = true;

  var formData = new FormData(this);

  fetch(this.action, {
    method: "POST",
    body: formData
  })
    .then(response => response.json())
    .then(data => {
      document.getElementById("loading-spinner").style.display = "none";
      document.querySelector("button[type=submit]").disabled = false;

      if (data.final_loss !== undefined) {
        const lossBlock = document.getElementById("loss-result");
        lossBlock.style.display = "block";
        document.getElementById("loss-value").textContent = Number(data.final_loss).toFixed(4);
      } else {
        document.getElementById("loss-result").style.display = "none";
      }

      if (data.plots && data.plots.length > 0) {
        document.getElementById("visualization-results").style.display = "block";
        const plotsContainer = document.getElementById("plots-container");
        plotsContainer.innerHTML = '';
        data.plots.forEach(plot => {
          const img = document.createElement('img');
          img.src = plot.url;
          img.alt = plot.title || 'График';
          img.className = 'result-plot';
          plotsContainer.appendChild(img);
        });
      } else {
        document.getElementById("visualization-results").style.display = "none";
      }

      if (data.detailed_results && data.detailed_results.length > 0) {
        const llmContent = document.getElementById("llm-content");
        llmContent.innerHTML = '';

        data.detailed_results.forEach((result, index) => {
          const resultDiv = document.createElement('div');
          resultDiv.className = 'llm-result-item';

          const header = document.createElement('h4');
          header.textContent = `Метаболит ${index + 1}`;
          resultDiv.appendChild(header);

          if (result.answer !== undefined && result.answer !== "") {
            const answerContainer = document.createElement('div');
            answerContainer.className = 'metabolite-answer';
            const answerLabel = document.createElement('strong');
            answerLabel.textContent = 'Ответ: ';
            answerContainer.appendChild(answerLabel);
            const answerText = document.createElement('span');
            answerText.textContent = result.answer;
            answerContainer.appendChild(answerText);
            resultDiv.appendChild(answerContainer);
          }

          if (result.reasoning && result.reasoning.trim() !== "") {
            const reasoningContainer = document.createElement('div');
            reasoningContainer.className = 'metabolite-analysis';
            const reasoningLabel = document.createElement('strong');
            reasoningLabel.textContent = 'Размышления:';
            reasoningContainer.appendChild(reasoningLabel);
            const reasoningContent = document.createElement('div');
            reasoningContent.className = 'analysis-content';
            const paragraphs = result.reasoning.split('\n');
            paragraphs.forEach(paragraph => {
              if (paragraph.trim()) {
                const p = document.createElement('p');
                p.textContent = paragraph.trim();
                reasoningContent.appendChild(p);
              }
            });
            reasoningContainer.appendChild(reasoningContent);
            resultDiv.appendChild(reasoningContainer);
          }

          if (result.sources && result.sources.length > 0) {
            const refsContainer = document.createElement('div');
            refsContainer.className = 'metabolite-references';
            const refsLabel = document.createElement('strong');
            refsLabel.textContent = 'Источники:';
            refsContainer.appendChild(refsLabel);
            const refsList = document.createElement('ul');
            refsList.className = 'references-list';
            result.sources.forEach(src => {
              const li = document.createElement('li');
              let srcText = src.source;
              if (src.doi && src.doi.trim() !== "") {
                srcText += ` (DOI: ${src.doi})`;
              }
              if (src.pmid && src.pmid.trim() !== "") {
                srcText += ` (PMID: ${src.pmid})`;
              }
              li.textContent = srcText;
              refsList.appendChild(li);
            });
            refsContainer.appendChild(refsList);
            resultDiv.appendChild(refsContainer);
          }

          if (result.column_index && result.column_index.length > 0) {
            const columnContainer = document.createElement('div');
            columnContainer.className = 'metabolite-columns';
            const colLabel = document.createElement('strong');
            colLabel.textContent = 'Индекс столбца: ';
            columnContainer.appendChild(colLabel);
            const colText = document.createElement('span');
            colText.textContent = result.column_index.join(', ');
            columnContainer.appendChild(colText);
            resultDiv.appendChild(columnContainer);
          }

          llmContent.appendChild(resultDiv);
        });
        document.getElementById("llm-results").style.display = "block";
      } else {
        document.getElementById("llm-results").style.display = "none";
      }
    })
    .catch(error => {
      document.getElementById("loading-spinner").style.display = "none";
      document.querySelector("button[type=submit]").disabled = false;
      alert("Ошибка: " + error);
    });
});