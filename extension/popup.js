document.addEventListener('DOMContentLoaded', function() {
  const captureBtn = document.getElementById('captureBtn');
  const loading = document.getElementById('loading');
  const predictionDiv = document.getElementById('prediction'); // Renamed for clarity
  const historyList = document.getElementById('history-list');
  const noHistoryLi = document.getElementById('no-history');

  async function displayHistory() {
    try {
      let { history = [] } = await chrome.storage.local.get('history');

      // Clear previous dynamic items, keep noHistoryLi template
      historyList.innerHTML = '';
      historyList.appendChild(noHistoryLi); // Add it back temporarily

      if (history.length === 0) {
        noHistoryLi.style.display = 'block';
      } else {
        noHistoryLi.style.display = 'none';
        history.forEach(item => {
          const li = document.createElement('li');
          const itemDate = new Date(item.timestamp);
          // More robust time formatting, ensuring locale-specific and handling potential invalid dates
          let timeString;
          try {
            timeString = itemDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
          } catch (e) {
            timeString = "Invalid time";
          }

          li.innerHTML = `
            <strong>${item.prediction}</strong> - ${(item.confidence * 100).toFixed(1)}%
            <br>
            <span style="font-size: 0.8em; color: #777;">${itemDate.toLocaleDateString()} ${timeString}</span>
          `;
          historyList.appendChild(li);
        });
      }
    } catch (e) {
      console.error("Error displaying history:", e);
      noHistoryLi.textContent = 'Error loading history.';
      noHistoryLi.style.display = 'block';
    }
  }

  captureBtn.addEventListener('click', async () => {
    try {
      loading.style.display = 'block';
      predictionDiv.style.display = 'none';
      captureBtn.disabled = true;

      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      
      if (!tab) {
        throw new Error("No active tab found. Please ensure you are on a webpage.");
      }
      if (!tab.id) {
        throw new Error("Active tab has no ID. Cannot send message.");
      }

      const response = await chrome.tabs.sendMessage(tab.id, { action: 'captureScreen' });
      
      if (response && response.prediction) {
        const confidence = (response.confidence * 100).toFixed(1);
        predictionDiv.innerHTML = `
          <div>Predicted Country: <strong>${response.prediction}</strong></div>
          <div class="confidence">Confidence: ${confidence}%</div>
        `;
        predictionDiv.style.display = 'block';
      } else if (response && response.error) {
        predictionDiv.textContent = 'Error: ' + response.error;
        predictionDiv.style.display = 'block';
      } else {
        predictionDiv.textContent = 'Failed to get prediction. Please try again or check the content script.';
        predictionDiv.style.display = 'block';
      }
    } catch (error) {
      console.error('Error in capture button click:', error);
      predictionDiv.textContent = 'Error: ' + error.message;
      predictionDiv.style.display = 'block';
    } finally {
      loading.style.display = 'none';
      captureBtn.disabled = false;
      displayHistory(); // Refresh history view regardless of success or failure of current prediction
    }
  });

  // Initial display of history when popup opens
  displayHistory();
}); 