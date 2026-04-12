// Main Application JavaScript for Sawit Auto Labeling

// Global variables
let sessionId = null;
let currentImage = null;
let points = [];
let selectedPointId = null;
let currentTool = 'add';
let currentLabel = 'tree';
let imageInfo = null;
let zoomLevel = 1.0;
let panX = 0;
let panY = 0;
let isPanning = false;
let lastMouseX = 0;
let lastMouseY = 0;

// Canvas and ctx
let canvas = null;
let ctx = null;

// Color palette for labels
const labelColors = {
    'tree': '#22c55e',
    'building': '#ef4444', 
    'road': '#8b5cf6',
    'water': '#06b6d4',
    'default': '#6366f1'
};

// Initialize app on DOM ready
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    canvas = document.getElementById('main-canvas');
    ctx = canvas.getContext('2d');
    
    // Setup canvas
    resizeCanvas();
    setupEventListeners();
    
    // Initialize UI
    updatePointsCounter();
    updateStatus('Ready', 'ready');
    
    // Default labels
    const defaultLabels = ['tree', 'building', 'road', 'water'];
    updateLabelsList(defaultLabels);
    
    // Show help toast
    showToast('Upload an image to start labeling', 'info');
}

function resizeCanvas() {
    const container = document.getElementById('canvas-container');
    if (!currentImage) {
        canvas.width = container.clientWidth;
        canvas.height = container.clientHeight;
    } else {
        // Scale image to fit in container
        const scale = Math.min(container.clientWidth / currentImage.width, container.clientHeight / currentImage.height) * 0.9;
        canvas.width = currentImage.width * scale;  
        canvas.height = currentImage.height * scale;
    }
    redraw();
}

function setupEventListeners() {
    // Window resize
    window.addEventListener('resize', resizeCanvas);
    
    // Canvas events
    canvas.addEventListener('click', handleCanvasClick);
    canvas.addEventListener('mousemove', handleCanvasMouseMove);
    canvas.addEventListener('mousedown', handleCanvasMouseDown);
    canvas.addEventListener('mouseup', handleCanvasMouseUp);
    canvas.addEventListener('wheel', handleCanvasWheel);
    
    // Keyboard shortcuts
    document.addEventListener('keydown', handleKeyDown);
    
    // File input
    document.getElementById('imageInput').addEventListener('change', handleImageSelect);
}

// File Upload
function handleImageSelect(event) {
    const file = event.target.files[0];
    if (file) {
        // Preview first
        const reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('image-preview').src = e.target.result;
        };
        reader.readAsDataURL(file);
        
        // Auto-upload if shift key is pressed
        if (event.shiftKey) {
            uploadImage();
        }
    }
}

function uploadImage() {
    const fileInput = document.getElementById('imageInput');
    const file = fileInput.files[0];
    
    if (!file) {
        showToast('Please select an image file', 'warning');
        return;
    }
    
    const formData = new FormData();
    formData.append('image', file);
    
    // Show progress
    showProgress(true);
    updateStatus('Uploading...', 'working');
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        showProgress(false);
        
        if (data.success) {
            // Load image and setup canvas
            loadImageToCanvas(data.image_data, data.image_info);
            sessionId = data.session_id;
            imageInfo = data.image_info;
            
            // Update labels if provided
            if (data.labels) {
                updateLabelsList(data.labels);
            }
            
            updateStatus('Image loaded', 'ready');
            showToast('Image uploaded successfully! Click to add labels.', 'success');
            
            // Update image info display
            updateImageInfo(data.image_info, file.name);
            
        } else {
            showToast('Upload failed: ' + data.error, 'error');
            updateStatus('Upload failed', 'error');
        }
    })
    .catch(error => {
        showProgress(false);
        showToast('Network error: ' + error.message, 'error');
        updateStatus('Error', 'error');
    });
}

function loadImageToCanvas(imageDataUrl, info) {
    const img = new Image();
    img.onload = function() {
        currentImage = img;
        imageInfo = info;
        resizeCanvas();
        redraw();
        centerImage();
    };
    img.src = imageDataUrl;
}

function centerImage() {
    if (!currentImage) return;
    
    const container = document.getElementById('canvas-container');
    zoomLevel = 1.0;
    panX = 0;
    panY = 0;
    redraw();
}

// Canvas Drawing
function redraw() {
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw background
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Set transformation for pan/zoom
    ctx.save();
    ctx.translate(panX, panY);
    ctx.scale(zoomLevel, zoomLevel);
    
    // Draw image
    if (currentImage) {
        ctx.drawImage(currentImage, 0, 0);
    }
    
    // Draw points
    drawPoints();
    
    // Restore transformation
    ctx.restore();
    
    // Draw UI overlay (grid, selection box, etc.)
    drawUIOverlay();
}

function drawPoints() {
    points.forEach(point => {
        drawPoint(point);
    });
}

function drawPoint(point) {
    const color = labelColors[point.label] || labelColors['default'];
    const x = point.x;
    const y = point.y;
    
    // Draw marker circle
    ctx.beginPath();
    ctx.arc(x, y, 8 / zoomLevel, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2 / zoomLevel;
    ctx.stroke();
    
    // Draw border if selected
    if (point.id === selectedPointId) {
        ctx.beginPath();
        ctx.arc(x, y, 12 / zoomLevel, 0, 2 * Math.PI);
        ctx.strokeStyle = '#ffd700';
        ctx.lineWidth = 3 / zoomLevel;
        ctx.stroke();
    }
    
    // Draw label if visible
    if (document.getElementById('show-labels').checked) {
        ctx.font = `${12 / zoomLevel}px Inter`;
        ctx.fillStyle = '#ffffff';
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 1 / zoomLevel;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        
        // Draw text with shadow
        ctx.strokeText(point.label, x, y + 18 / zoomLevel);
        ctx.fillText(point.label, x, y + 18 / zoomLevel);
    }
}

function drawUIOverlay() {
    // Draw selection guidelines
    if (currentTool === 'delete') {
        ctx.fillStyle = 'rgba(220, 53, 69, 0.1)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    
    // Draw grid if enabled
    if (document.getElementById('snap-to-grid').checked) {
        drawGrid();
    }
}

function drawGrid() {
    const gridSize = 20;
    ctx.strokeStyle = 'rgba(0,0,0,0.1)';
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    
    for (let x = 0; x <= canvas.width; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
    }
    
    for (let y = 0; y <= canvas.height; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
    }
    
    ctx.setLineDash([]);
}

// Canvas Interaction
function handleCanvasClick(event) {
    const coords = getCanvasCoordinates(event);
    
    if (currentTool === 'add') {
        addPoint(coords.x, coords.y);
    } else if (currentTool === 'delete') {
        deletePointAt(coords.x, coords.y);
    }
}

function handleCanvasMouseMove(event) {
    const coords = getCanvasCoordinates(event);
    updateMouseCoords(coords.x, coords.y);
    
    // Change cursor based on tool and hover
    if (currentTool === 'add') {
        canvas.style.cursor = 'crosshair';
    } else if (currentTool === 'delete') {
        const point = findPointAt(coords.x, coords.y);
        canvas.style.cursor = point ? 'pointer' : 'not-allowed';
    } else if (currentTool === 'move') {
        canvas.style.cursor = isPanning ? 'grabbing' : 'grab';
    }
    
    if (isPanning) {
        const dx = event.clientX - lastMouseX;
        const dy = event.clientY - lastMouseY;
        panX += dx;
        panY += dy;
        lastMouseX = event.clientX;
        lastMouseY = event.clientY;
        redraw();
    }
}

function handleCanvasMouseDown(event) {
    if (currentTool === 'move') {
        isPanning = true;
        lastMouseX = event.clientX;
        lastMouseY = event.clientY;
        canvas.style.cursor = 'grabbing';
    }
}

function handleCanvasMouseUp(event) {
    isPanning = false;
    canvas.style.cursor = 'grab';
}

function handleCanvasWheel(event) {
    event.preventDefault();
    
    const zoomFactor = event.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(0.1, Math.min(10.0, zoomLevel * zoomFactor));
    
    setZoom(newZoom * 100);
}

function getCanvasCoordinates(event) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: (event.clientX - rect.left - panX) / zoomLevel,
        y: (event.clientY - rect.top - panY) / zoomLevel
    };
}

function updateMouseCoords(x, y) {
    document.getElementById('mouse-coords').textContent = `X: ${Math.round(x)}, Y: ${Math.round(y)}`;
}

// Point Management
function addPoint(x, y) {
    if (!currentImage) {
        showToast('Please upload an image first', 'warning');
        return;
    }
    
    const point = {
        id: points.length,
        x: x,
        y: y,
        label: currentLabel
    };
    
    points.push(point);
    redraw();
    updatePointsCounter();
    
    // Save to session
    if (sessionId) {
        fetch('/add_point', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                session_id: sessionId,
                x: x,
                y: y,
                label: currentLabel
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showToast(`Added ${currentLabel} at [${Math.round(x)}, ${Math.round(y)}]`, 'success');
            }
        });
    }
}

function deletePointAt(x, y) {
    const point = findPointAt(x, y);
    if (point) {
        confirmDeletePoint(point);
    }
}

function findPointAt(x, y) {
    const threshold = 10; // pixels
    return points.find(p => 
        Math.abs(p.x - x) < threshold && Math.abs(p.y - y) < threshold
    );
}

function findPointIndexAt(x, y) {
    return points.findIndex(p => 
        Math.abs(p.x - x) < 10 && Math.abs(p.y - y) < 10
    );
}

function confirmDeletePoint(point) {
    if (confirm(`Delete ${point.label} point #${point.id}?`)) {
        deletePoint(point.id);
    }
}

function deletePoint(pointId) {
    const pointIndex = points.findIndex(p => p.id === pointId);
    if (pointIndex === -1) return;
    
    const deletedPoint = points[pointIndex];
    points.splice(pointIndex, 1);
    
    // Update IDs
    for (let i = pointIndex; i < points.length; i++) {
        points[i].id = i;
    }
    
    redraw();
    updatePointsCounter();
    
    // Remove from session
    if (sessionId) {
        fetch('/delete_point', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                session_id: sessionId,
                point_id: pointId
            })
        });
    }
    
    showToast(`Deleted ${deletedPoint.label} point`, 'info');
}

// Labels Management
function updateLabelsList(labels) {
    const container = document.getElementById('labels-list');
    container.innerHTML = '';
    
    labels.forEach(label => {
        const color = labelColors[label] || labelColors['default'];
        const labelItem = document.createElement('div');
        labelItem.className = 'label-item';
        labelItem.setAttribute('data-label', label);
        labelItem.onclick = () => selectLabel(label);
        
        labelItem.innerHTML = `
            <div class="label-color" style="background-color: ${color}"></div>
            <span>${label}</span>
        `;
        
        if (label === currentLabel) {
            labelItem.classList.add('active');
        }
        
        container.appendChild(labelItem);
    });
}

function selectLabel(label) {
    currentLabel = label;
    
    // Update visual selection
    document.querySelectorAll('.label-item').forEach(item => {
        item.classList.remove('active');
    });
    document.querySelector(`[data-label="${label}"]`).classList.add('active');
    
    showToast(`Selected label: ${label}`, 'info');
}

function addLabel() {
    const input = document.getElementById('new-label-input');
    const newLabel = input.value.trim().toLowerCase();
    
    if (!newLabel) {
        showToast('Please enter a label name', 'warning');
        return;
    }
    
    if (sessionData && sessionData.labels && sessionData.labels.includes(newLabel)) {
        showToast('Label already exists', 'warning');
        return;
    }
    
    // Add new color for the label (if not exists)
    if (!labelColors[newLabel]) {
        // Generate random color
        const randomColor = '#' + Math.floor(Math.random()*16777215).toString(16);
        labelColors[newLabel] = randomColor;
    }
    
    // Add to current labels
    const currentLabels = Array.from(document.querySelectorAll('.label-item')).map(item => 
        item.getAttribute('data-label')
    );
    
    if (!currentLabels.includes(newLabel)) {
        currentLabels.push(newLabel);
        updateLabelsList(currentLabels);
        showToast(`Added new label: ${newLabel}`, 'success');
    }
    
    // Send to server
    if (sessionId) {
        fetch('/add_label', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                session_id: sessionId,
                label: newLabel
            })
        });
    }
    
    input.value = '';
}

// Tools
function setTool(tool) {
    currentTool = tool;
    
    // Update button visuals
    document.querySelectorAll('#add-tool, #delete-tool, #move-tool').forEach(btn => {
        btn.classList.remove('active');
    });
    document.getElementById(`${tool}-tool`).classList.add('active');
    
    showToast(`Tool: ${tool.charAt(0).toUpperCase() + tool.slice(1)}`, 'info');
    redraw();
}

// Zoom Controls
function zoomIn() {
    setZoom(Math.min(zoomLevel * 1.5 * 100, 500));
}

function zoomOut() {
    setZoom(Math.max(zoomLevel * 0.75 * 100, 10));
}

function fitImage() {
    centerImage();
    setZoom(100);
}

function setZoom(zoom) {
    zoomLevel = zoom / 100;
    document.getElementById('zoom-slider').value = zoom;
    document.getElementById('zoom-percent').textContent = `${zoom}%`;
    document.getElementById('zoom-level').textContent = `${zoom}%`;
    redraw();
}

// Export Functions
function exportData(format) {
    if (!sessionId) {
        showToast('Please upload an image first', 'warning');
        return;
    }
    
    if (points.length === 0) {
        showToast('No points to export', 'warning');
        return;
    }
    
    updateStatus('Exporting...', 'working');
    
    // Show loading modal
    document.getElementById('loading-text').textContent = `Exporting to ${format.toUpperCase()}...`;
    bootstrap.Modal.getOrCreateInstance(document.getElementById('loadingModal')).show();
    
    // Create download link
    const link = document.createElement('a');
    link.href = `/export/${format}/${sessionId}`;
    link.download = `sawit_labeling_${new Date().getTime()}.${format}`;
    link.click();
    
    // Hide loading modal and show success
    setTimeout(() => {
        bootstrap.Modal.getInstance(document.getElementById('loadingModal')).hide();
        updateStatus('Export complete', 'ready');
        showToast(`Exported ${points.length} points to ${format.toUpperCase()}`, 'success');
    }, 1000);
}

// UI Updates
function updatePointsCounter() {
    document.getElementById('points-count').textContent = points.length;
    document.getElementById('points-summary').textContent = `${points.length} points added`;
}

function updateStatus(text, type = 'ready') {
    const statusIndicator = document.getElementById('status-indicator');
    let icon, color;
    
    switch(type) {
        case 'working':
            icon = 'fas fa-circle text-warning';
            color = 'text-warning';
            break;
        case 'error':
            icon = 'fas fa-circle text-danger';
            color = 'text-danger';
            break;
        default:
            icon = 'fas fa-circle text-success';
            color = 'text-success';
    }
    
    statusIndicator.innerHTML = `<i class="${icon}"></i> ${text}`;
}

function updateImageInfo(info, filename) {
    let sizeText = '';
    if (info && info.width && info.height) {
        sizeText = `${info.width}×${info.height}`;
    }
    
    const crsWarning = info?.crs ? '' : 
        `<div class="warning-box mt-2">
            <p class="warning-text">
                <i class="fas fa-exclamation-triangle"></i> 
                No CRS metadata found. Outputs will use pixel coordinates only.
                For geographic coordinates, upload a georeferenced image.
            </p>
        </div>`;
    
    document.getElementById('image-info').innerHTML = `
        ${filename} (${sizeText})
        ${crsWarning}
    `;
}

// Progress Bar
function showProgress(show) {
    const progress = document.getElementById('upload-progress');
    progress.classList.toggle('d-none', !show);
    if (show) {
        document.getElementById('upload-progress-bar').style.width = '0%';
        setTimeout(() => {
            document.getElementById('upload-progress-bar').style.width = '100%';
        }, 100);
    }
}

// Toast Notifications
function showToast(message, type = 'info') {
    const toast = document.getElementById('mainToast');
    const toastBody = document.getElementById('toast-message');
    const toastHeader = toast.querySelector('.toast-header');
    
    // Update icon based on type
    let icon = 'fas fa-info-circle';
    let iconColor = 'text-primary';
    
    switch(type) {
        case 'success':
            icon = 'fas fa-check-circle';
            iconColor = 'text-success';
            break;
        case 'warning':
            icon = 'fas fa-exclamation-triangle';
            iconColor = 'text-warning';
            break;
        case 'error':
            icon = 'fas fa-times-circle';
            iconColor = 'text-danger';
            break;
    }
    
    toastHeader.innerHTML = `<i class="${icon} ${iconColor} me-2"></i> <strong class="me-auto">Sawit Labeling</strong>`;
    toastBody.textContent = message;
    
    // Show toast
    const bsToast = bootstrap.Toast.getOrCreateInstance(toast);
    bsToast.show();
}

// Keyboard Shortcuts
function handleKeyDown(event) {
    // Prevent shortcuts when typing in input fields
    if (event.target.tagName === 'INPUT') return;

    switch(event.key) {
        case 'a':
        case 'A':
            setTool('add');
            break;
        case 'd':
        case 'D':
            setTool('delete');
            break;
        case 'm':
        case 'M':
            setTool('move');
            break;
        case '+':
        case '=':
            zoomIn();
            break;
        case '-':
            zoomOut();
            break;
        case 'r':
        case 'R':
            fitImage();
            break;
        case 'u':
        case 'U':
            document.getElementById('imageInput').click();
            break;
        case 'l':
        case 'L':
            autoLabel();
            break;
    }
}

// Auto Label (AI Detection)
function autoLabel() {
    if (!sessionId) {
        showToast('Please upload an image first', 'warning');
        return;
    }

    const confThreshold = parseFloat(document.getElementById('conf-threshold').value);
    const iouThreshold = parseFloat(document.getElementById('iou-threshold').value);

    updateStatus('AI Auto-labeling...', 'working');
    showLoadingModal('Running AI detection...');

    fetch('/auto_label', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            session_id: sessionId,
            conf_threshold: confThreshold,
            iou_threshold: iouThreshold
        })
    })
    .then(response => response.json())
    .then(data => {
        hideLoadingModal();

        if (data.success) {
            // Refresh points from server
            fetch(`/get_points/${sessionId}`)
                .then(res => res.json())
                .then(pointsData => {
                    if (pointsData.success) {
                        points = pointsData.points;
                        redraw();
                        updatePointsCounter();

                        updateStatus(`AI found ${data.total} detections`, 'ready');
                        showToast(`AI detected ${data.total} objects (conf: ${confThreshold})`, 'success');
                    }
                });
        } else {
            updateStatus('AI detection failed', 'error');
            showToast('Auto-label error: ' + data.error, 'error');
        }
    })
    .catch(error => {
        hideLoadingModal();
        updateStatus('Network error', 'error');
        showToast('Network error: ' + error.message, 'error');
    });
}

function showLoadingModal(message) {
    document.getElementById('loading-text').textContent = message;
    bootstrap.Modal.getOrCreateInstance(document.getElementById('loadingModal')).show();
}

function hideLoadingModal() {
    bootstrap.Modal.getInstance(document.getElementById('loadingModal')).hide();
}