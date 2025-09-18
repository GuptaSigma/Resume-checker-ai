// Main JavaScript for AI Resume Analyzer

document.addEventListener('DOMContentLoaded', function() {
    initializeFileUpload();
    initializeFormValidation();
});

// File Upload Functionality
function initializeFileUpload() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('resume_files');
    const fileList = document.getElementById('fileList');
    
    if (!dropZone || !fileInput || !fileList) return;
    
    let selectedFiles = [];
    
    // Click to browse files
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });
    
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });
    
    // Highlight drop zone when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });
    
    // Handle dropped files
    dropZone.addEventListener('drop', handleDrop, false);
    
    // Handle file input change
    fileInput.addEventListener('change', function(e) {
        handleFiles(e.target.files);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    function highlight(e) {
        dropZone.classList.add('dragover');
    }
    
    function unhighlight(e) {
        dropZone.classList.remove('dragover');
    }
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }
    
    function handleFiles(files) {
        ([...files]).forEach(addFile);
        updateFileInput();
    }
    
    function addFile(file) {
        // Check file type
        const allowedTypes = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/msword', 'text/plain'];
        if (!allowedTypes.includes(file.type)) {
            showAlert('Invalid file type. Please upload PDF, DOCX, DOC, or TXT files only.', 'danger');
            return;
        }
        
        // Check file size (16MB)
        if (file.size > 16 * 1024 * 1024) {
            showAlert('File size too large. Maximum size is 16MB.', 'danger');
            return;
        }
        
        // Check if file already exists
        if (selectedFiles.some(f => f.name === file.name && f.size === file.size)) {
            showAlert('File already selected.', 'warning');
            return;
        }
        
        selectedFiles.push(file);
        displayFile(file);
    }
    
    function displayFile(file) {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <div class="file-info">
                <i class="fas fa-file-alt text-primary"></i>
                <div>
                    <div class="fw-medium">${file.name}</div>
                    <div class="file-size">${formatFileSize(file.size)}</div>
                </div>
            </div>
            <button type="button" class="remove-file" onclick="removeFile('${file.name}', ${file.size})">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        fileList.appendChild(fileItem);
    }
    
    window.removeFile = function(fileName, fileSize) {
        selectedFiles = selectedFiles.filter(f => !(f.name === fileName && f.size === fileSize));
        updateFileInput();
        renderFileList();
    };
    
    function updateFileInput() {
        const dt = new DataTransfer();
        selectedFiles.forEach(file => dt.items.add(file));
        fileInput.files = dt.files;
    }
    
    function renderFileList() {
        fileList.innerHTML = '';
        selectedFiles.forEach(displayFile);
    }
    
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

// Form Validation
function initializeFormValidation() {
    const form = document.getElementById('resumeForm');
    const submitBtn = document.getElementById('submitBtn');
    
    if (!form || !submitBtn) return;
    
    form.addEventListener('submit', function(e) {
        if (!validateForm()) {
            e.preventDefault();
            return false;
        }
        
        // Show loading state
        submitBtn.classList.add('btn-loading');
        submitBtn.disabled = true;
        
        // Show progress message
        showAlert('Processing resumes... This may take a few moments.', 'info');
    });
    
    function validateForm() {
        let isValid = true;
        
        // Check required fields
        const requiredFields = ['company_name', 'hr_name', 'hr_email', 'email_password'];
        requiredFields.forEach(fieldName => {
            const field = document.getElementById(fieldName);
            if (!field.value.trim()) {
                showAlert(`Please fill in the ${field.labels[0].textContent}`, 'danger');
                field.focus();
                isValid = false;
                return;
            }
        });
        
        // Check email format
        const emailField = document.getElementById('hr_email');
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(emailField.value)) {
            showAlert('Please enter a valid email address.', 'danger');
            emailField.focus();
            isValid = false;
        }
        
        // Check if files are selected
        const fileInput = document.getElementById('resume_files');
        if (!fileInput.files.length) {
            showAlert('Please select at least one resume file.', 'danger');
            isValid = false;
        }
        
        return isValid;
    }
}

// Utility Functions
function showAlert(message, type = 'info') {
    // Remove existing alerts
    const existingAlerts = document.querySelectorAll('.alert');
    existingAlerts.forEach(alert => {
        if (alert.classList.contains('alert-dismissible')) {
            alert.remove();
        }
    });
    
    // Create new alert
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert after navbar
    const navbar = document.querySelector('.navbar');
    if (navbar && navbar.nextSibling) {
        navbar.parentNode.insertBefore(alertDiv, navbar.nextSibling);
    } else {
        document.body.insertBefore(alertDiv, document.body.firstChild);
    }
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

// Smooth scrolling for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Add loading states to buttons
function addButtonLoading(button, text = 'Loading...') {
    button.disabled = true;
    button.setAttribute('data-original-text', button.innerHTML);
    button.innerHTML = `<i class="fas fa-spinner fa-spin me-2"></i>${text}`;
}

function removeButtonLoading(button) {
    button.disabled = false;
    const originalText = button.getAttribute('data-original-text');
    if (originalText) {
        button.innerHTML = originalText;
        button.removeAttribute('data-original-text');
    }
}

// Export functions for global use
window.showAlert = showAlert;
window.addButtonLoading = addButtonLoading;
window.removeButtonLoading = removeButtonLoading;
