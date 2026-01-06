const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const predictBtn = document.getElementById('predictBtn');
const resetBtn = document.getElementById('resetBtn');
const previewImg = document.getElementById('previewImg');
const previewContainer = document.getElementById('previewContainer');
const fileName = document.getElementById('fileName');

const resultsPanel = document.getElementById('resultsPanel');
const emptyState = document.getElementById('emptyState');
const loading = document.getElementById('loading');

let currentOrigBase64 = "";
let currentOverlayBase64 = "";
let currentResults = [];

// --- Drag & Drop Logic ---
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        handleFile(fileInput.files[0]);
    }
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
});

function handleFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        previewContainer.classList.remove('d-none');
        dropZone.classList.add('d-none');
        fileName.innerText = file.name;

        predictBtn.disabled = false;
        predictBtn.classList.remove('d-none');
        resetBtn.classList.add('d-none');

        resultsPanel.classList.add('d-none');
        emptyState.classList.remove('d-none');
    };
    reader.readAsDataURL(file);
}

// --- RESET FUNCTION ---
window.resetApp = () => {
    fileInput.value = "";
    previewContainer.classList.add('d-none');
    previewImg.src = "";
    dropZone.classList.remove('d-none');
    resultsPanel.classList.add('d-none');
    loading.classList.add('d-none');
    emptyState.classList.remove('d-none');

    predictBtn.disabled = true;
    predictBtn.classList.remove('d-none');
    resetBtn.classList.add('d-none');

    currentOrigBase64 = "";
    currentOverlayBase64 = "";
    currentResults = [];
}

// --- Visualization Switchers ---
window.showOriginal = () => {
    document.getElementById('overlayImg').src = "data:image/png;base64," + currentOrigBase64;
    document.querySelector('.btn-group .btn:first-child').classList.add('active');
    document.querySelector('.btn-group .btn:last-child').classList.remove('active');
    document.querySelector('.img-label').innerText = "Ảnh Gốc";
}

window.showHeatmap = () => {
    document.getElementById('overlayImg').src = "data:image/png;base64," + currentOverlayBase64;
    document.querySelector('.btn-group .btn:first-child').classList.remove('active');
    document.querySelector('.btn-group .btn:last-child').classList.add('active');
    document.querySelector('.img-label').innerText = "Vùng AI Nghi Ngờ";
}

// --- PDF Export ---
window.exportPDF = () => {
    const element = document.getElementById('pdf-report-template');

    document.getElementById('pdf-date').innerText = new Date().toLocaleString('vi-VN');
    document.getElementById('pdf-orig-img').src = "data:image/png;base64," + currentOrigBase64;
    document.getElementById('pdf-heatmap-img').src = "data:image/png;base64," + currentOverlayBase64;

    const tbody = document.getElementById('pdf-table-body');
    tbody.innerHTML = "";

    currentResults.forEach(item => {
        if (item.prob < 0.01 && !item.is_normal) return;

        const percent = (item.prob * 100).toFixed(2) + "%";
        const color = item.is_normal ? "color: green;" : "color: red; font-weight: bold;";
        const status = item.is_normal ? "Nguy cơ thấp" : "PHÁT HIỆN";

        // CẬP NHẬT: Thêm tên tiếng Anh vào PDF
        const row = `
            <tr>
                <td>
                    ${item.label}<br>
                    <small style="color: #666; font-style: italic;">(${item.label_en})</small>
                </td>
                <td>${percent}</td>
                <td style="${color}">${status}</td>
            </tr>
        `;
        tbody.insertAdjacentHTML('beforeend', row);
    });

    const opt = {
        margin: 0.5,
        filename: `BaoCao_MedAI_${Date.now()}.pdf`,
        image: { type: 'jpeg', quality: 0.98 },
        html2canvas: { scale: 2 },
        jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' }
    };

    element.style.display = 'block';
    html2pdf().set(opt).from(element).save().then(() => {
        element.style.display = 'none';
    });
}

// --- Predict Logic ---
predictBtn.addEventListener('click', async () => {
    const file = fileInput.files[0];
    const form = new FormData();
    form.append("file", file);

    emptyState.classList.add('d-none');
    resultsPanel.classList.add('d-none');
    resetBtn.classList.add('d-none');
    loading.classList.remove('d-none');
    predictBtn.disabled = true;

    try {
        const resp = await fetch("/predict", { method: "POST", body: form });
        const data = await resp.json();

        currentOrigBase64 = data.orig;
        currentOverlayBase64 = data.overlay;
        currentResults = data.top;

        showHeatmap();

        const statusAlert = document.getElementById('statusAlert');
        const topResult = data.top[0];
        const isNormal = topResult.is_normal;

        if (isNormal) {
            statusAlert.className = "alert alert-success shadow-sm d-flex align-items-center";
            statusAlert.innerHTML = `
                        <i class="fa-solid fa-shield-heart fa-2x me-3"></i>
                        <div>
                            <h5 class="alert-heading mb-0">Kết quả: Bình thường</h5>
                            <small>Không phát hiện dấu hiệu bệnh lý vượt ngưỡng cảnh báo.</small>
                        </div>
                    `;
        } else {
            statusAlert.className = "alert alert-danger shadow-sm d-flex align-items-center";
            statusAlert.innerHTML = `
                        <i class="fa-solid fa-triangle-exclamation fa-2x me-3"></i>
                        <div>
                            <h5 class="alert-heading mb-0">Phát Hiện Dấu Hiệu Bất Thường</h5>
                            <small>AI đã xác định nguy cơ bệnh lý.</small>
                        </div>
                    `;
        }

        const probsList = document.getElementById('probsList');
        probsList.innerHTML = "";

        data.top.forEach(item => {
            const percent = (item.prob * 100).toFixed(2);
            let colorClass = "bg-secondary";
            let textClass = "text-muted";

            if (!item.is_normal) {
                if (item.prob > 0.7) colorClass = "bg-danger";
                else if (item.prob > 0.4) colorClass = "bg-warning text-dark";
                else colorClass = "bg-info text-dark";
                textClass = "text-dark";
            } else {
                colorClass = "bg-success";
                textClass = "text-success";
            }

            // CẬP NHẬT: Thêm tên tiếng Anh vào danh sách hiển thị Web
            const html = `
                        <div class="mb-3">
                            <div class="prob-label ${item.is_normal ? 'text-success' : ''}">
                                <span>
                                    ${item.label} 
                                    <span class="text-muted small fw-normal">(${item.label_en})</span>
                                </span>
                                <span class="prob-value">${percent}%</span>
                            </div>
                            <div class="progress">
                                <div class="progress-bar ${colorClass}" role="progressbar" 
                                     style="width: ${percent}%" aria-valuenow="${percent}" aria-valuemin="0" aria-valuemax="100">
                                </div>
                            </div>
                        </div>
                    `;
            probsList.insertAdjacentHTML('beforeend', html);
        });

        loading.classList.add('d-none');
        resultsPanel.classList.remove('d-none');

        predictBtn.classList.add('d-none');
        resetBtn.classList.remove('d-none');

    } catch (err) {
        alert("Lỗi: " + err);
        loading.classList.add('d-none');
        emptyState.classList.remove('d-none');

        predictBtn.disabled = false;
        predictBtn.classList.remove('d-none');
    }
});