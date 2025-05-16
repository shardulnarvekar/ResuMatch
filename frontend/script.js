document.addEventListener("DOMContentLoaded", function() {
  // Tab functionality
  const tabs = document.querySelectorAll('.tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      // Remove active class from all tabs and content
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      
      // Add active class to clicked tab
      tab.classList.add('active');
      
      // Show corresponding content
      const tabId = tab.getAttribute('data-tab');
      document.getElementById(tabId).classList.add('active');
    });
  });
  
  // Form submission
  document.getElementById("uploadForm").addEventListener("submit", async function(e) {
    e.preventDefault();

    const resumeFile = document.getElementById("resume").files[0];
    const jobDescription = document.getElementById("jobDescription").value.trim();
    const loader = document.getElementById("loader");
    const resultBox = document.getElementById("result");

    if (!resumeFile || !jobDescription) {
      alert("Please upload a resume and paste the job description.");
      return;
    }

    const formData = new FormData();
    formData.append("file", resumeFile);
    formData.append("job_description", jobDescription);

    try {
      // Show loader
      loader.style.display = "block";
      resultBox.style.display = "none";

      const response = await fetch("http://127.0.0.1:8000/api/upload-resume", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Failed to fetch results.");
      }

      const result = await response.json();
      
      // Also get AI suggestions
      const suggestionsResponse = await fetch("http://127.0.0.1:8000/api/suggest-improvements", {
        method: "POST",
        body: formData,
      });
      
      let aiSuggestions = [];
      if (suggestionsResponse.ok) {
        const suggestionsData = await suggestionsResponse.json();
        aiSuggestions = suggestionsData.suggestions || [];
      }

      // Hide loader
      loader.style.display = "none";
      
      // Update UI with results
      updateResultsUI(result, aiSuggestions);
      
      // Show results
      resultBox.style.display = "block";
      
      // Make sure we're on the first tab
      document.querySelector('.tab[data-tab="match-results"]').click();

    } catch (error) {
      loader.style.display = "none";
      alert("❌ Error: " + error.message);
    }
  });
});

function updateResultsUI(result, aiSuggestions) {
  // Update overall score
  const overallScore = document.getElementById("overallScore");
  overallScore.textContent = result.overall_score;
  
  // Create charts
  createScoreChart(result.overall_score);
  createMetricChart('tfidfChart', 'TF-IDF Similarity', result.tfidf_similarity);
  createMetricChart('semanticChart', 'Semantic Similarity', result.semantic_similarity);
  createMetricChart('keywordChart', 'Keyword Match', result.keyword_match_ratio);
  
  // Update keywords lists
  updateKeywordsList('matchedKeywords', result.matched_keywords);
  updateKeywordsList('missingKeywords', result.missing_keywords, true);
  
  // Update skills lists
  updateKeywordsList('matchedSkills', result.matched_skills);
  updateKeywordsList('missingSkills', result.missing_skills, true);
  
  // Update suggestions
  const suggestionsList = document.getElementById("suggestionsList");
  suggestionsList.innerHTML = '';
  
  // First add the TF suggestions
  if (result.suggestions && result.suggestions.length > 0) {
    result.suggestions.forEach(suggestion => {
      const suggestionItem = document.createElement('div');
      suggestionItem.className = 'suggestion-item';
      suggestionItem.textContent = suggestion;
      suggestionsList.appendChild(suggestionItem);
    });
  }
  
  // Then add AI suggestions
  if (aiSuggestions && aiSuggestions.length > 0) {
    aiSuggestions.forEach(suggestion => {
      const suggestionItem = document.createElement('div');
      suggestionItem.className = 'suggestion-item';
      suggestionItem.innerHTML = `<strong>AI Suggestion:</strong> ${suggestion}`;
      suggestionsList.appendChild(suggestionItem);
    });
  }
  
  // If no suggestions, add a message
  if (suggestionsList.children.length === 0) {
    const noSuggestions = document.createElement('div');
    noSuggestions.className = 'suggestion-item';
    noSuggestions.textContent = "Your resume already matches well with this job description!";
    suggestionsList.appendChild(noSuggestions);
  }
}

function updateKeywordsList(elementId, keywords, isMissing = false) {
  const element = document.getElementById(elementId);
  element.innerHTML = '';
  
  if (!keywords || keywords.length === 0) {
    const noneItem = document.createElement('div');
    noneItem.className = 'keyword';
    noneItem.textContent = 'None';
    element.appendChild(noneItem);
    return;
  }
  
  keywords.forEach(keyword => {
    const keywordElement = document.createElement('div');
    keywordElement.className = 'keyword';
    if (isMissing) {
      keywordElement.classList.add('missing-keyword');
    }
    keywordElement.textContent = keyword;
    element.appendChild(keywordElement);
  });
}

function createScoreChart(score) {
  const ctx = document.getElementById('scoreChart').getContext('2d');
  
  // Clear any existing chart
  if (window.scoreChart) {
    window.scoreChart.destroy();
  }
  
  // Create new chart
  window.scoreChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      datasets: [{
        data: [score, 100 - score],
        backgroundColor: [
          getScoreColor(score),
          '#f1f1f1'
        ],
        borderWidth: 0
      }]
    },
    options: {
      cutout: '70%',
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        tooltip: {
          enabled: false
        },
        legend: {
          display: false
        }
      },
      animation: {
        animateRotate: true,
        animateScale: true
      }
    }
  });
}

function createMetricChart(elementId, label, value) {
  const ctx = document.getElementById(elementId).getContext('2d');
  
  // Clear any existing chart
  if (window[elementId + 'Chart']) {
    window[elementId + 'Chart'].destroy();
  }
  
  // Create new chart
  window[elementId + 'Chart'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: [label],
      datasets: [{
        data: [value],
        backgroundColor: getScoreColor(value),
        barThickness: 40
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: true,
      scales: {
        x: {
          beginAtZero: true,
          max: 100,
          grid: {
            display: false
          }
        },
        y: {
          grid: {
            display: false
          }
        }
      },
      plugins: {
        legend: {
          display: false
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              return context.raw.toFixed(1) + '%';
            }
          }
        }
      }
    }
  });
}

function getScoreColor(score) {
  if (score >= 80) {
    return '#4caf50'; // Green
  } else if (score >= 60) {
    return '#8bc34a'; // Light green
  } else if (score >= 40) {
    return '#ffc107'; // Yellow
  } else if (score >= 20) {
    return '#ff9800'; // Orange
  } else {
    return '#f44336'; // Red
  }
}