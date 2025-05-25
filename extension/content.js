// Listen for messages from the popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'captureScreen') {
    captureAndPredict()
      .then(result => sendResponse(result))
      .catch(error => sendResponse({ error: error.message }));
    return true; // Required for async response
  }
});

async function captureAndPredict() {
  try {
    // Find the main game view element
    const gameView = document.querySelector('.game-layout__content');
    if (!gameView) {
      throw new Error('Game view not found. Make sure you are on a GeoGuessr game page.');
    }

    // Create a canvas to capture the game view
    const canvas = await html2canvas(gameView, {
      useCORS: true,
      allowTaint: true,
      backgroundColor: null,
      scale: 1.0 // Ensure we get the full resolution
    });

    // Convert canvas to base64 image
    const imageData = canvas.toDataURL('image/jpeg', 0.95);

    // Send the image to the background script for prediction
    const response = await chrome.runtime.sendMessage({
      action: 'predictCountry',
      imageData: imageData
    });

    if (response.error) {
      throw new Error(response.error);
    }

    return response;
  } catch (error) {
    console.error('Error in captureAndPredict:', error);
    throw error;
  }
}

// Add html2canvas library
const script = document.createElement('script');
script.src = 'https://html2canvas.hertzen.com/dist/html2canvas.min.js';
document.head.appendChild(script); 