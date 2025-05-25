document.addEventListener('DOMContentLoaded', function() {
  const captureBtn = document.getElementById('captureBtn');
  const loading = document.getElementById('loading');
  const prediction = document.getElementById('prediction');

  captureBtn.addEventListener('click', async () => {
    try {
      // Show loading state
      loading.style.display = 'block';
      prediction.style.display = 'none';
      captureBtn.disabled = true;

      // Get the active tab
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      
      // Send message to content script to capture the screen
      const response = await chrome.tabs.sendMessage(tab.id, { action: 'captureScreen' });
      
      if (response && response.prediction) {
        const confidence = (response.confidence * 100).toFixed(1);
        prediction.innerHTML = `
          <div>Predicted Country: <strong>${response.prediction}</strong></div>
          <div class="confidence">Confidence: ${confidence}%</div>
        `;
        prediction.style.display = 'block';
      } else {
        prediction.textContent = 'Failed to get prediction. Please try again.';
        prediction.style.display = 'block';
      }
    } catch (error) {
      console.error('Error:', error);
      prediction.textContent = 'Error: ' + error.message;
      prediction.style.display = 'block';
    } finally {
      loading.style.display = 'none';
      captureBtn.disabled = false;
    }
  });
}); 