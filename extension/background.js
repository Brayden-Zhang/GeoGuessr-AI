// Load ONNX Runtime
let session = null;
let classNames = null;
let isModelLoading = false;

// Initialize the model
async function initModel() {
  if (isModelLoading) {
    return;
  }
  
  isModelLoading = true;
  try {
    // Load the ONNX model
    session = await ort.InferenceSession.create(chrome.runtime.getURL('model/model.onnx'));
    
    // Load class names
    const response = await fetch(chrome.runtime.getURL('model/class_names.json'));
    classNames = await response.json();
    
    console.log('Model loaded successfully');
    console.log(`Loaded ${classNames.length} country classes`);
  } catch (error) {
    console.error('Error loading model:', error);
    session = null;
    classNames = null;
  } finally {
    isModelLoading = false;
  }
}

// Initialize the model when the extension starts
initModel();

// Listen for messages from content script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'predictCountry') {
    predictCountry(request.imageData)
      .then(result => sendResponse(result))
      .catch(error => sendResponse({ error: error.message }));
    return true; // Required for async response
  }
});

async function predictCountry(imageData) {
  try {
    // Ensure model is loaded
    if (!session || !classNames) {
      await initModel();
      if (!session || !classNames) {
        throw new Error('Failed to load model');
      }
    }

    // Convert base64 image to tensor
    const img = await loadImage(imageData);
    const tensor = preprocessImage(img);

    // Run inference
    const results = await session.run({ input: tensor });
    const predictions = results.output.data;
    
    // Get the index of the highest probability
    const maxIndex = predictions.indexOf(Math.max(...predictions));
    const confidence = predictions[maxIndex];
    
    return {
      prediction: classNames[maxIndex],
      confidence: confidence
    };
  } catch (error) {
    console.error('Error in predictCountry:', error);
    throw error;
  }
}

// Helper function to load image from base64
function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

// Helper function to preprocess image
function preprocessImage(img) {
  const canvas = document.createElement('canvas');
  canvas.width = 224;
  canvas.height = 224;
  const ctx = canvas.getContext('2d');
  
  // Draw and resize image
  ctx.drawImage(img, 0, 0, 224, 224);
  
  // Get image data
  const imageData = ctx.getImageData(0, 0, 224, 224);
  const data = imageData.data;
  
  // Convert to float32 array and normalize
  const tensor = new Float32Array(1 * 3 * 224 * 224);
  for (let i = 0; i < data.length / 4; i++) {
    tensor[i] = data[i * 4] / 255.0; // R
    tensor[i + 224 * 224] = data[i * 4 + 1] / 255.0; // G
    tensor[i + 2 * 224 * 224] = data[i * 4 + 2] / 255.0; // B
  }
  
  return new ort.Tensor('float32', tensor, [1, 3, 224, 224]);
} 