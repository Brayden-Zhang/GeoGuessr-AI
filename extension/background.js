// background.js

// Listen for keyboard shortcut command
chrome.commands.onCommand.addListener(async (command) => {
  if (command === "trigger_prediction") {
    console.log("trigger_prediction command received");
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (tab && tab.id) {
        // Check if the tab is a GeoGuessr tab before sending the message
        if (tab.url && tab.url.startsWith("https://www.geoguessr.com/")) {
          chrome.tabs.sendMessage(tab.id, { action: 'captureScreen' }, (response) => {
            if (chrome.runtime.lastError) {
              console.error('Error sending captureScreen message from command listener:', chrome.runtime.lastError.message);
              // Potentially, try to notify the user via an alert on the page if context permits,
              // or log to a storage area that the popup could read.
              // For now, just logging.
            } else {
              // Response from content script (prediction details or error)
              console.log('Prediction triggered by shortcut, content script response:', response);
              // The content script handles its own UI (overlay).
              // The popup, if open, would independently trigger and update.
              // If response contains an error, it's already been handled by content.js for overlay
            }
          });
        } else {
          console.log("Not a GeoGuessr tab. Command 'trigger_prediction' ignored.");
          // Optionally, provide feedback to the user if possible (e.g., a brief notification)
          // This is hard from a service worker without specific notification permissions/setup.
        }
      } else {
        console.error('No active tab found to send command to.');
      }
    } catch (e) {
      console.error('Error in command listener:', e);
    }
  }
});

// Listen for messages from content script (e.g., when popup initiates capture)
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'predictCountry') {
    predictCountry(request.imageData)
      .then(result => sendResponse(result))
      .catch(error => {
        console.error('Error in predictCountry message listener:', error);
        sendResponse({ error: error.message });
      });
    return true; // Required for async response
  }
});

async function predictCountry(imageData) {
  console.log('Received image data for prediction in background.js');
  try {
    const response = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ image_data: imageData }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`Server error: ${response.status} ${response.statusText}. Details: ${errorText}`);
      throw new Error(`Server error: ${response.status} ${response.statusText}. Details: ${errorText}`);
    }

    const data = await response.json();
    console.log('Prediction received from backend:', data);
    
    if (data.error) {
        console.error('Backend returned an error:', data.error);
        throw new Error(data.error); // This will be caught by the catch block below
    }

    // If no error in data, proceed to save to history
    try {
      const maxHistoryItems = 5;
      let { history = [] } = await chrome.storage.local.get('history');
      const newItem = {
        prediction: data.prediction,
        confidence: data.confidence,
        timestamp: new Date().toISOString()
      };
      history.unshift(newItem);
      history = history.slice(0, maxHistoryItems);
      await chrome.storage.local.set({ history });
      console.log('Prediction history updated:', history);
    } catch (storageError) {
      console.error('Error saving prediction to history:', storageError);
      // Don't let storage error stop returning the prediction
    }

    return data; // Should be {'prediction': 'SomeCountry', 'confidence': 0.95}

  } catch (error) { // Catches errors from fetch, response.json(), or data.error throw
    console.error('Error in predictCountry function:', error);
    // Return an error object compatible with how the popup expects it
    return { error: error.message };
  }
}