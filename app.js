// Main App State
let model = null;
let currentImage = null; // Store the loaded image element
let latestDetections = []; // Store raw predictions
let currentThreshold = 0.50; // Current confidence threshold (0.0 - 1.0)

// Vehicle classes we want to highlight specifically
const VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle'];

// DOM Elements
const modelStatusEl = document.getElementById('model-status');
const modelStatusTextEl = modelStatusEl.querySelector('.status-text');
const metricsEl = document.getElementById('metrics');
const inferenceTimeEl = document.getElementById('inference-time');
const thresholdSlider = document.getElementById('threshold-slider');
const thresholdValEl = document.getElementById('threshold-val');
const detectionListEl = document.getElementById('detection-list');
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const browseBtn = document.getElementById('browse-btn');
const previewContainer = document.getElementById('preview-container');
const canvas = document.getElementById('detection-canvas');
const ctx = canvas.getContext('2d');
const processingOverlay = document.getElementById('processing-overlay');
const resetBtn = document.getElementById('reset-btn');
const galleryCards = document.querySelectorAll('.gallery-card');

// 1. Initialize TensorFlow.js and Load COCO-SSD Model
async function initModel() {
    try {
        console.log('Loading COCO-SSD model...');
        modelStatusEl.className = 'status-indicator loading';
        modelStatusTextEl.textContent = 'Loading COCO-SSD Model (20MB)...';
        
        // Wait for tfjs to be ready, then load cocoSsd
        await tf.ready();
        model = await cocoSsd.load({
            base: 'lite_mobilenet_v2' // Load a lighter model for faster browser performance
        });
        
        console.log('Model loaded successfully.');
        modelStatusEl.className = 'status-indicator ready';
        modelStatusTextEl.textContent = 'AutoDetect Engine Active';
    } catch (error) {
        console.error('Error loading the ML model:', error);
        modelStatusEl.className = 'status-indicator loading';
        modelStatusTextEl.textContent = 'Failed to load model. Refresh page.';
        alert('Could not load the TensorFlow model. Please verify your internet connection.');
    }
}

// Start model loading on script execution
initModel();

// 2. Setup Event Listeners

// File browser button
browseBtn.addEventListener('click', () => fileInput.click());

// File input selection
fileInput.addEventListener('change', handleFileSelect);

// Drag & Drop handlers
dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
});

dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
});

dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        processImageFile(e.dataTransfer.files[0]);
    }
});

// Slider inputs
thresholdSlider.addEventListener('input', (e) => {
    currentThreshold = parseInt(e.target.value) / 100;
    thresholdValEl.textContent = `${e.target.value}%`;
    
    // Re-render detections instantly if we have an image and previous detections
    if (currentImage && latestDetections.length > 0) {
        renderDetections();
    }
});

// Reset application
resetBtn.addEventListener('click', resetApp);

// Quick test gallery click handler
galleryCards.forEach(card => {
    card.addEventListener('click', () => {
        const url = card.getAttribute('data-url');
        if (url) {
            loadAndProcessFromUrl(url);
        }
    });
});

// 3. Image Processing Functions

function handleFileSelect(e) {
    if (e.target.files && e.target.files[0]) {
        processImageFile(e.target.files[0]);
    }
}

// Process local uploaded file
function processImageFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file (PNG, JPG, JPEG, WebP).');
        return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
            currentImage = img;
            runDetection(img);
        };
        img.src = event.target.result;
    };
    reader.readAsDataURL(file);
}

// Process remote URL from gallery
function loadAndProcessFromUrl(url) {
    showLoadingState();
    
    const img = new Image();
    img.crossOrigin = 'anonymous'; // Crucial for reading pixels via canvas (CORS protection)
    
    img.onload = () => {
        currentImage = img;
        runDetection(img);
    };
    
    img.onerror = () => {
        hideLoadingState();
        alert('Failed to load image from gallery. CORS restrictions or network issue.');
    };
    
    img.src = url;
}

// Prepare UI for processing
function showLoadingState() {
    dropzone.classList.add('hidden');
    previewContainer.classList.remove('hidden');
    processingOverlay.classList.remove('hidden');
}

function hideLoadingState() {
    processingOverlay.classList.add('hidden');
}

function resetApp() {
    currentImage = null;
    latestDetections = [];
    fileInput.value = '';
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Reset UI
    previewContainer.classList.add('hidden');
    dropzone.classList.remove('hidden');
    metricsEl.classList.add('hidden');
    
    // Reset list
    detectionListEl.innerHTML = '<p class="no-detections-text">Upload an image to run vehicle detection.</p>';
}

// 4. ML Inference and Rendering Detections

async function runDetection(img) {
    showLoadingState();
    
    // Wait slightly to let browser render the overlay/spinner
    await new Promise(resolve => setTimeout(resolve, 100));

    // Ensure model is loaded, if not, wait
    if (!model) {
        modelStatusTextEl.textContent = 'Waiting for COCO-SSD to finish loading...';
        let retries = 10;
        while (!model && retries > 0) {
            await new Promise(resolve => setTimeout(resolve, 500));
            retries--;
        }
        if (!model) {
            hideLoadingState();
            alert('Model loading took too long. Please refresh the page and try again.');
            return;
        }
    }

    try {
        // Set canvas bounds based on image aspect ratio while keeping maximum limits
        const maxWidth = 800;
        const maxHeight = 550;
        let width = img.naturalWidth || img.width;
        let height = img.naturalHeight || img.height;
        
        // Scale image proportionally to fit canvas constraints
        if (width > maxWidth) {
            height = Math.round((height * maxWidth) / width);
            width = maxWidth;
        }
        if (height > maxHeight) {
            width = Math.round((width * maxHeight) / height);
            height = maxHeight;
        }
        
        canvas.width = width;
        canvas.height = height;
        
        // Draw the base image
        ctx.drawImage(img, 0, 0, width, height);

        // Run object detection model and measure time taken
        const startTime = performance.now();
        const predictions = await model.detect(canvas);
        const endTime = performance.now();
        const inferenceDuration = Math.round(endTime - startTime);
        
        // Save detections globally
        latestDetections = predictions;
        
        // Display metrics
        metricsEl.classList.remove('hidden');
        inferenceTimeEl.textContent = `${inferenceDuration} ms`;
        
        // Render bounding boxes & stats
        renderDetections();
        
    } catch (error) {
        console.error('Error during image detection:', error);
        alert('An error occurred during object detection.');
    } finally {
        hideLoadingState();
    }
}

function renderDetections() {
    if (!currentImage) return;
    
    // 1. Redraw base image to clear previous boxes
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);
    
    // 2. Filter detections by score AND class
    const vehiclePredictions = latestDetections.filter(pred => {
        const isVehicle = VEHICLE_CLASSES.includes(pred.class.toLowerCase());
        const isAboveThreshold = pred.score >= currentThreshold;
        return isVehicle && isAboveThreshold;
    });

    // 3. Draw bounding boxes
    vehiclePredictions.forEach(pred => {
        const [x, y, width, height] = pred.bbox;
        const score = Math.round(pred.score * 100);
        const className = pred.class.toUpperCase();
        
        // Style variables
        const primaryColor = '#4f46e5'; // Indigo
        const glowColor = 'rgba(79, 70, 229, 0.2)';
        
        // Draw box outline with glowing edges
        ctx.strokeStyle = primaryColor;
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, width, height);
        
        // Semi-transparent fill overlay inside the box
        ctx.fillStyle = glowColor;
        ctx.fillRect(x, y, width, height);
        
        // Draw details badge background
        const tagText = `${className} ${score}%`;
        ctx.font = 'bold 12px "Outfit", sans-serif';
        const textWidth = ctx.measureText(tagText).width;
        
        ctx.fillStyle = primaryColor;
        // Position tag inside the box, or above if box is too close to top
        const tagY = y > 20 ? y - 24 : y + 4;
        const tagX = x;
        const tagHeight = 20;
        
        // Draw rounded rectangle for label tag
        drawRoundedRect(ctx, tagX, tagY, textWidth + 14, tagHeight, 4);
        ctx.fill();
        
        // Draw tag text
        ctx.fillStyle = '#ffffff';
        ctx.fillText(tagText, tagX + 7, tagY + 14);
    });

    // 4. Update the Detections Summary dashboard in sidebar
    updateSummaryList(vehiclePredictions);
}

// Utility to draw rounded rectangles
function drawRoundedRect(ctx, x, y, width, height, radius) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
}

// Helper to construct detection list with aggregated counts
function updateSummaryList(filteredDetections) {
    if (filteredDetections.length === 0) {
        detectionListEl.innerHTML = `
            <p class="no-detections-text">
                No vehicles detected above ${Math.round(currentThreshold * 100)}% confidence.
            </p>
        `;
        return;
    }

    // Count occurrences of each vehicle type
    const counts = {};
    filteredDetections.forEach(pred => {
        const cls = pred.class.toLowerCase();
        counts[cls] = (counts[cls] || 0) + 1;
    });

    // Render counts inside sidebar list
    let listHTML = '';
    
    // Sort keys alphabetically
    Object.keys(counts).sort().forEach(vehicleType => {
        const count = counts[vehicleType];
        const pluralName = count > 1 ? `${vehicleType}s` : vehicleType;
        const capitalizedName = pluralName.charAt(0).toUpperCase() + pluralName.slice(1);
        
        // Icon mapper
        let iconClass = 'fa-car';
        if (vehicleType === 'truck') iconClass = 'fa-truck';
        if (vehicleType === 'bus') iconClass = 'fa-bus';
        if (vehicleType === 'motorcycle') iconClass = 'fa-motorcycle';

        listHTML += `
            <div class="detection-item">
                <div class="detection-label-group">
                    <i class="fa-solid ${iconClass}"></i>
                    <span>${capitalizedName}</span>
                </div>
                <span class="detection-count">${count}</span>
            </div>
        `;
    });
    
    detectionListEl.innerHTML = listHTML;
}
