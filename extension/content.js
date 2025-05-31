const OVERLAY_ID = 'geoguessr-ai-overlay';
const COUNTRY_NAME_ID = 'ai-country-name';
const CONFIDENCE_SCORE_ID = 'ai-confidence-score';
const OVERLAY_ERROR_ID = 'ai-overlay-error';
const OVERLAY_CLOSE_ID = 'ai-overlay-close';
const LOADING_TEXT_ID = 'ai-loading-text'; // For specific loading text element

function createOrUpdateOverlay(prediction, confidence, errorMsg = null, options = {}) {
  let overlay = document.getElementById(OVERLAY_ID);
  let predictionDiv, confidenceDiv, errorDisplayEl, loadingTextEl, countryNameEl, confidenceScoreEl;

  if (!overlay) {
    overlay = document.createElement('div');
    overlay.id = OVERLAY_ID;
    overlay.style.position = 'fixed';
    overlay.style.top = '20px';
    overlay.style.right = '20px';
    overlay.style.backgroundColor = 'white';
    overlay.style.color = 'black';
    overlay.style.padding = '15px';
    overlay.style.border = '1px solid #ccc';
    overlay.style.borderRadius = '8px';
    overlay.style.zIndex = '9999';
    overlay.style.boxShadow = '0px 2px 10px rgba(0,0,0,0.15)';
    overlay.style.fontFamily = 'Arial, sans-serif';
    overlay.style.fontSize = '14px';
    overlay.style.minWidth = '200px'; // Ensure a decent width
    overlay.style.display = 'none'; // Initially hidden

    const title = document.createElement('div');
    title.textContent = 'GeoGuessr AI Prediction';
    title.style.fontWeight = 'bold';
    title.style.marginBottom = '10px';
    title.style.borderBottom = '1px solid #eee';
    title.style.paddingBottom = '5px';
    overlay.appendChild(title);

    predictionDiv = document.createElement('div');
    predictionDiv.innerHTML = `Predicted: <strong id="${COUNTRY_NAME_ID}"></strong>`;
    predictionDiv.style.marginBottom = '5px';
    overlay.appendChild(predictionDiv);

    confidenceDiv = document.createElement('div');
    confidenceDiv.innerHTML = `Confidence: <span id="${CONFIDENCE_SCORE_ID}"></span>%`;
    overlay.appendChild(confidenceDiv);

    errorDisplayEl = document.createElement('div');
    errorDisplayEl.id = OVERLAY_ERROR_ID;
    errorDisplayEl.style.color = 'red';
    errorDisplayEl.style.marginTop = '5px';
    errorDisplayEl.style.display = 'none'; // Hidden by default
    overlay.appendChild(errorDisplayEl);

    const closeButton = document.createElement('button');
    closeButton.id = OVERLAY_CLOSE_ID;
    closeButton.textContent = 'X';
    closeButton.style.position = 'absolute';
    closeButton.style.top = '5px';
    closeButton.style.right = '5px';
    closeButton.style.background = 'transparent';
    closeButton.style.border = 'none';
    closeButton.style.fontSize = '16px';
    closeButton.style.cursor = 'pointer';
    closeButton.style.color = '#aaa';
    closeButton.onmouseover = () => closeButton.style.color = '#000';
    closeButton.onmouseout = () => closeButton.style.color = '#aaa';
    closeButton.addEventListener('click', () => {
      overlay.style.display = 'none';
    });
    overlay.appendChild(closeButton);

    document.body.appendChild(overlay);
  } else {
    // Elements already exist, get references
    predictionDiv = document.getElementById(COUNTRY_NAME_ID).parentElement;
    confidenceDiv = document.getElementById(CONFIDENCE_SCORE_ID).parentElement;
    errorDisplayEl = document.getElementById(OVERLAY_ERROR_ID);
  }

  // Ensure elements are correctly identified or created
  if (overlay) {
    countryNameEl = document.getElementById(COUNTRY_NAME_ID);
    confidenceScoreEl = document.getElementById(CONFIDENCE_SCORE_ID);
    predictionDiv = countryNameEl ? countryNameEl.parentElement : null;
    confidenceDiv = confidenceScoreEl ? confidenceScoreEl.parentElement : null;
    errorDisplayEl = document.getElementById(OVERLAY_ERROR_ID);
    loadingTextEl = document.getElementById(LOADING_TEXT_ID);
  } else {
    // Overlay needs to be created (logic for this is in the original function, ensure it's run first)
  }

  // If overlay still not found after creation attempt (should be rare, defensive coding)
  if (!overlay) {
      console.error("Overlay element could not be created or found.");
      return;
  }


  if (options.isLoading) {
    if (countryNameEl) countryNameEl.textContent = 'Predicting...'; else if (predictionDiv) predictionDiv.innerHTML = `<strong id="${COUNTRY_NAME_ID}">Predicting...</strong>`;
    if (confidenceScoreEl) confidenceScoreEl.textContent = ' '; else if (confidenceDiv) confidenceDiv.innerHTML = `<span id="${CONFIDENCE_SCORE_ID}"> </span>%`;

    if (predictionDiv) predictionDiv.style.display = 'block'; // Show the div containing "Predicting..."
    if (confidenceDiv) confidenceDiv.style.display = 'block'; // Show the div for confidence (empty or with a placeholder)
    if (errorDisplayEl) errorDisplayEl.style.display = 'none'; // Hide error message

    overlay.style.display = 'block';
    return; // Exit early for loading state
  }

  // Fallback to existing logic if not loading
  countryNameEl = document.getElementById(COUNTRY_NAME_ID); // Re-fetch in case it was just created
  confidenceScoreEl = document.getElementById(CONFIDENCE_SCORE_ID); // Re-fetch
  predictionDiv = countryNameEl ? countryNameEl.parentElement : null; // Re-fetch
  confidenceDiv = confidenceScoreEl ? confidenceScoreEl.parentElement : null; // Re-fetch
  errorDisplayEl = document.getElementById(OVERLAY_ERROR_ID); // Re-fetch


  if (errorMsg) {
    if (countryNameEl) countryNameEl.textContent = 'Error';
    if (confidenceScoreEl) confidenceScoreEl.textContent = 'N/A';
    if (errorDisplayEl) {
        errorDisplayEl.textContent = errorMsg.length > 100 ? errorMsg.substring(0,97) + "..." : errorMsg;
        errorDisplayEl.style.display = 'block';
    }
    if (predictionDiv) predictionDiv.style.display = 'block'; // Ensure "Error" is visible
    if (confidenceDiv) confidenceDiv.style.display = 'block'; // Ensure "N/A" is visible for confidence
    // errorDisplayEl is handled below (shown)

  } else if (prediction && typeof confidence === 'number') {
    if (countryNameEl) countryNameEl.textContent = prediction;
    if (confidenceScoreEl) confidenceScoreEl.textContent = (confidence * 100).toFixed(1);
    if (errorDisplayEl) errorDisplayEl.style.display = 'none';
    if (predictionDiv) predictionDiv.style.display = 'block';
    if (confidenceDiv) confidenceDiv.style.display = 'block';
  } else {
    // Case where it's not error, not prediction, maybe initial call or unexpected state
    // Potentially hide all or show a default message. For now, ensure error is hidden.
     if (errorDisplayEl) errorDisplayEl.style.display = 'none';
  }

  overlay.style.display = 'block';
}


// Listen for messages from the popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'captureScreen') {
    // No loading indicator here, as captureAndPredict will handle it.
    captureAndPredict()
      .then(result => sendResponse(result)) // Send full result back to popup
      .catch(error => {
        // captureAndPredict already calls createOrUpdateOverlay for errors.
        // This sendResponse is for the popup.
        sendResponse({ error: error.message });
      });
    return true; // Required for async response
  }
});

async function captureAndPredict() {
  // Show loading indicator immediately
  createOrUpdateOverlay(null, null, null, { isLoading: true });

  try {
    // Find the main game view element
    const gameView = document.querySelector('.game-layout__content');
    if (!gameView) {
      const errorMsg = 'Game view not found. Are you on a game page?';
      createOrUpdateOverlay(null, null, errorMsg);
      throw new Error(errorMsg);
    }

    // Ensure html2canvas is loaded
    if (typeof html2canvas === 'undefined') {
        console.error('html2canvas not loaded yet!');
        const errorMsg = 'Error: html2canvas not loaded.';
        createOrUpdateOverlay(null, null, errorMsg);
        throw new Error(errorMsg);
    }

    const canvas = await html2canvas(gameView, {
      useCORS: true,
      allowTaint: true,
      backgroundColor: null, // Important for transparency if game view has rounded corners etc.
      scale: 1.0
    });

    // Convert canvas to base64 image
    const imageData = canvas.toDataURL('image/jpeg', 0.95);

    // Send the image to the background script for prediction.
    // The loading indicator is already showing.
    const response = await chrome.runtime.sendMessage({
      action: 'predictCountry',
      imageData: imageData
    });

    if (response && response.error) {
      createOrUpdateOverlay(null, null, response.error);
      throw new Error(response.error);
    }

    if (response && response.prediction) {
      createOrUpdateOverlay(response.prediction, response.confidence);
    } else {
      const errorMsg = 'Unexpected response from backend.';
      createOrUpdateOverlay(null, null, errorMsg);
      throw new Error(errorMsg);
    }

    return response;
  } catch (error) {
    console.error('Error in captureAndPredict:', error);
    // If createOrUpdateOverlay wasn't called with error yet by a more specific catch,
    // call it here. Error object 'error' should have 'message' property.
    createOrUpdateOverlay(null, null, error.message || "An unknown error occurred during prediction.");
    throw error;
  }
}

// Add html2canvas library if not already added by another part of the extension
if (!document.querySelector('script[src="https://html2canvas.hertzen.com/dist/html2canvas.min.js"]')) {
    const script = document.createElement('script');
    script.src = 'https://html2canvas.hertzen.com/dist/html2canvas.min.js';
    script.onload = () => {
        console.log('html2canvas loaded successfully.');
    };
    script.onerror = () => {
        console.error('Failed to load html2canvas.');
        // Potentially inform user via an alert or a basic overlay message
        createOrUpdateOverlay(null, null, "Critical error: Failed to load html2canvas. Predictions won't work.");
    };
    document.head.appendChild(script);
}