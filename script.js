// 상단 변수 선언은 이전과 동일
const reportTypeSelect = document.getElementById('reportType');
const reportDateSelect = document.getElementById('reportDate');
const reportTimeSelect = document.getElementById('reportTime');
const reportFileSelect = document.getElementById('reportFile');
const reportFrame = document.getElementById('reportFrame');
const loadingIndicator = document.getElementById('loadingIndicator'); // 로딩 인디케이터 추가
let reportIndexData = {};

function showLoading() {
    loadingIndicator.style.display = 'flex';
    reportFrame.style.opacity = '0.5'; // iframe 흐리게
}

function hideLoading() {
    loadingIndicator.style.display = 'none';
    reportFrame.style.opacity = '1'; // iframe 원래대로
}

async function fetchReportIndex() {
    // Dropdowns 비활성화
    reportDateSelect.disabled = true;
    reportTimeSelect.disabled = true;
    reportFileSelect.disabled = true;
    reportDateSelect.innerHTML = '<option>로딩 중...</option>';
    reportTimeSelect.innerHTML = '<option>--</option>';
    reportFileSelect.innerHTML = '<option>--</option>';

    try {
        const response = await fetch('report_index.json');
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        reportIndexData = await response.json();
        populateReportTypes();
        populateDates();
    } catch (error) {
        console.error('Error fetching report index:', error);
        reportDateSelect.innerHTML = '<option>목록 로드 실패</option>';
        reportDateSelect.disabled = true; // 실패 시 계속 비활성화
        reportTimeSelect.disabled = true;
        reportFileSelect.disabled = true;
    }
}

function populateReportTypes() {
    // (이전과 동일)
    reportTypeSelect.value = 'daily';
}

function populateDates() {
    const selectedType = reportTypeSelect.value;
    reportDateSelect.innerHTML = '';
    reportTimeSelect.innerHTML = '<option>--</option>';
    reportFileSelect.innerHTML = '<option>--</option>';
    reportTimeSelect.disabled = true;
    reportFileSelect.disabled = true;
    reportFrame.src = 'about:blank';

    const dates = reportIndexData[selectedType] ? Object.keys(reportIndexData[selectedType]) : [];

    if (dates.length === 0) {
        reportDateSelect.innerHTML = '<option>선택 가능 날짜 없음</option>';
        reportDateSelect.disabled = true;
        return;
    }

    dates.forEach(date => { /* (이전과 동일) */
        const option = document.createElement('option');
        option.value = date;
        option.textContent = date;
        reportDateSelect.appendChild(option);
    });

    reportDateSelect.disabled = false; // 날짜 로드 완료 후 활성화
    if (dates.length > 0) {
        reportDateSelect.value = dates[0];
        populateTimes(); // 날짜 채우고 시간 채우기 호출
    }
}

function populateTimes() {
    const selectedType = reportTypeSelect.value;
    const selectedDate = reportDateSelect.value;
    reportTimeSelect.innerHTML = '';
    reportFileSelect.innerHTML = '<option>--</option>';
    reportFileSelect.disabled = true;
    reportFrame.src = 'about:blank';

    const timeEntries = reportIndexData[selectedType]?.[selectedDate] || [];

    if (timeEntries.length === 0) {
        reportTimeSelect.innerHTML = '<option>선택 가능 시간 없음</option>';
        reportTimeSelect.disabled = true;
        return;
    }

    timeEntries.forEach(entry => { /* (이전과 동일) */
        const option = document.createElement('option');
        option.value = entry.time;
        option.textContent = entry.time;
        reportTimeSelect.appendChild(option);
    });

    reportTimeSelect.disabled = false; // 시간 로드 완료 후 활성화
    if (timeEntries.length > 0) {
        reportTimeSelect.value = timeEntries[0].time;
        populateFiles(); // 시간 채우고 파일 채우기 호출
    }
}

function populateFiles() {
    const selectedType = reportTypeSelect.value;
    const selectedDate = reportDateSelect.value;
    const selectedTime = reportTimeSelect.value;
    reportFileSelect.innerHTML = '';
    reportFrame.src = 'about:blank';

    const timeEntries = reportIndexData[selectedType]?.[selectedDate] || [];
    const selectedEntry = timeEntries.find(entry => entry.time === selectedTime);
    const reports = selectedEntry?.reports || [];

    if (reports.length === 0) {
        reportFileSelect.innerHTML = '<option>선택 가능 리포트 없음</option>';
        reportFileSelect.disabled = true;
        return;
    }

    let defaultReportPath = '';
    reports.forEach(report => { /* (이전과 동일, 기본 리포트 경로 찾는 로직 포함) */
        const option = document.createElement('option');
        option.value = report.path;
        option.textContent = report.name;
        reportFileSelect.appendChild(option);
        if (!report.name.includes('commentary') && report.name.endsWith('.html')) {
            defaultReportPath = report.path;
        }
    });

    reportFileSelect.disabled = false; // 파일 목록 로드 완료 후 활성화
    reportFileSelect.value = defaultReportPath || reports[0]?.path; // 기본 리포트 우선 선택
    loadReport(); // 파일 채우고 리포트 로드 호출
}

function loadReport() {
    const selectedReportPath = reportFileSelect.value;
    if (selectedReportPath) {
        showLoading(); // 로딩 시작
        reportFrame.onload = hideLoading; // 로딩 완료 시 숨김
        reportFrame.onerror = () => { // 로딩 실패 시 처리
            hideLoading();
            console.error("Failed to load report:", selectedReportPath);
            // Optionally show an error message to the user in the iframe or elsewhere
            reportFrame.src = 'about:blank'; // Clear iframe on error
        };
        reportFrame.src = selectedReportPath; // iframe 소스 설정
    } else {
        reportFrame.src = 'about:blank';
        hideLoading(); // 선택할 리포트 없으면 로딩 숨김
    }
}

// Event listeners (이전과 동일)
reportTypeSelect.addEventListener('change', populateDates);
reportDateSelect.addEventListener('change', populateTimes);
reportTimeSelect.addEventListener('change', populateFiles);
reportFileSelect.addEventListener('change', loadReport);

// Initial load (이전과 동일)
fetchReportIndex();