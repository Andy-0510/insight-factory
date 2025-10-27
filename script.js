// --- Element References ---
const reportTypeSelect = document.getElementById('reportType');
const reportYearSelect = document.getElementById('reportYear'); // New
const reportMonthSelect = document.getElementById('reportMonth'); // New
const reportDaySelect = document.getElementById('reportDay');   // New
const reportFrame = document.getElementById('reportFrame');
const loadingIndicator = document.getElementById('loadingIndicator');
const darkModeToggle = document.getElementById('darkModeToggle'); // New
const themeLabel = document.getElementById('themeLabel'); // New
let reportIndexData = {};
let availableDates = {}; // Store dates per type { daily: {'2025': {'10': ['26', '27']...}}, weekly: {...} }

// --- Theme Handling ---
function applyTheme(isDark) {
    if (isDark) {
        document.body.classList.add('dark-mode');
        themeLabel.textContent = 'Dark Mode';
    } else {
        document.body.classList.remove('dark-mode');
        themeLabel.textContent = 'Light Mode';
    }
}

function toggleTheme() {
    const isDark = darkModeToggle.checked;
    applyTheme(isDark);
    localStorage.setItem('darkMode', isDark); // Persist preference
}

// Check saved theme on load
const savedDarkMode = localStorage.getItem('darkMode') === 'true';
darkModeToggle.checked = savedDarkMode;
applyTheme(savedDarkMode);

// Add listener for toggle
darkModeToggle.addEventListener('change', toggleTheme);

// --- Loading Indicator ---
function showLoading() { /* (Previous logic) */
    loadingIndicator.style.display = 'flex';
    reportFrame.style.opacity = '0.3'; // Make it more faded
}
function hideLoading() { /* (Previous logic) */
    loadingIndicator.style.display = 'none';
    reportFrame.style.opacity = '1';
}

// --- Data Fetching and Processing ---
async function fetchReportIndex() {
    // Disable all date selects initially
    reportYearSelect.disabled = true; reportYearSelect.innerHTML = '<option>로딩 중...</option>';
    reportMonthSelect.disabled = true; reportMonthSelect.innerHTML = '<option>--</option>';
    reportDaySelect.disabled = true; reportDaySelect.innerHTML = '<option>--</option>';

    try {
        const response = await fetch('report_index.json');
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        reportIndexData = await response.json();
        processAvailableDates(); // Process dates into hierarchical structure
        populateReportTypes();
        populateYears(); // Start population chain
    } catch (error) {
        console.error('Error fetching report index:', error);
        reportYearSelect.innerHTML = '<option>목록 로드 실패</option>';
    }
}

// Pre-process dates for easier lookup
function processAvailableDates() {
    availableDates = { daily: {}, weekly: {}, monthly: {} };
    for (const type in reportIndexData) {
        if (reportIndexData.hasOwnProperty(type)) {
            const dates = Object.keys(reportIndexData[type]);
            dates.forEach(dateStr => { // dateStr is YYYY-MM-DD
                const [year, month, day] = dateStr.split('-');
                if (!availableDates[type][year]) {
                    availableDates[type][year] = {};
                }
                if (!availableDates[type][year][month]) {
                    availableDates[type][year][month] = [];
                }
                if (!availableDates[type][year][month].includes(day)) {
                     availableDates[type][year][month].push(day);
                     availableDates[type][year][month].sort().reverse(); // Sort days descending
                }
            });
             // Sort months descending for each year
            for(const year in availableDates[type]){
                 availableDates[type][year] = Object.fromEntries(
                      Object.entries(availableDates[type][year]).sort(([m1],[m2]) => m2.localeCompare(m1))
                 );
            }
            // Sort years descending
             availableDates[type] = Object.fromEntries(
                 Object.entries(availableDates[type]).sort(([y1],[y2]) => y2.localeCompare(y1))
             );
        }
    }
}


// --- Dropdown Population ---
function populateReportTypes() {
    reportTypeSelect.value = 'daily'; // Default
}

function populateYears() {
    const selectedType = reportTypeSelect.value;
    reportYearSelect.innerHTML = '';
    reportMonthSelect.innerHTML = '<option>--</option>'; reportMonthSelect.disabled = true;
    reportDaySelect.innerHTML = '<option>--</option>'; reportDaySelect.disabled = true;
    reportFrame.src = 'about:blank';

    const years = availableDates[selectedType] ? Object.keys(availableDates[selectedType]) : [];

    if (years.length === 0) {
        reportYearSelect.innerHTML = '<option>연도 없음</option>'; reportYearSelect.disabled = true;
        return;
    }

    // Years are already sorted descending
    years.forEach(year => {
        const option = document.createElement('option');
        option.value = year;
        option.textContent = year;
        reportYearSelect.appendChild(option);
    });

    reportYearSelect.disabled = false;
    if (years.length > 0) {
        reportYearSelect.value = years[0]; // Select latest year
        populateMonths(); // Populate months for the selected year
    }
}

function populateMonths() {
    const selectedType = reportTypeSelect.value;
    const selectedYear = reportYearSelect.value;
    reportMonthSelect.innerHTML = '';
    reportDaySelect.innerHTML = '<option>--</option>'; reportDaySelect.disabled = true;
    reportFrame.src = 'about:blank';

    const months = availableDates[selectedType]?.[selectedYear] ? Object.keys(availableDates[selectedType][selectedYear]) : [];

    if (months.length === 0) {
        reportMonthSelect.innerHTML = '<option>월 없음</option>'; reportMonthSelect.disabled = true;
        return;
    }

    // Months are already sorted descending
    months.forEach(month => {
        const option = document.createElement('option');
        option.value = month;
        option.textContent = month;
        reportMonthSelect.appendChild(option);
    });

    reportMonthSelect.disabled = false;
    if (months.length > 0) {
        reportMonthSelect.value = months[0]; // Select latest month
        populateDays(); // Populate days for the selected month
    }
}

function populateDays() {
    const selectedType = reportTypeSelect.value;
    const selectedYear = reportYearSelect.value;
    const selectedMonth = reportMonthSelect.value;
    reportDaySelect.innerHTML = '';
    reportFrame.src = 'about:blank';

    const days = availableDates[selectedType]?.[selectedYear]?.[selectedMonth] || [];

    if (days.length === 0) {
        reportDaySelect.innerHTML = '<option>일 없음</option>'; reportDaySelect.disabled = true;
        return;
    }

    // Days are already sorted descending
    days.forEach(day => {
        const option = document.createElement('option');
        option.value = day;
        option.textContent = day;
        reportDaySelect.appendChild(option);
    });

    reportDaySelect.disabled = false;
    if (days.length > 0) {
        reportDaySelect.value = days[0]; // Select latest day
        loadReport(); // Load the report for the selected date
    }
}

// --- Report Loading ---
function loadReport() {
    const selectedType = reportTypeSelect.value;
    const selectedYear = reportYearSelect.value;
    const selectedMonth = reportMonthSelect.value;
    const selectedDay = reportDaySelect.value;

    // Check if all parts of the date are selected
    if (!selectedYear || !selectedMonth || !selectedDay || selectedYear === '--' || selectedMonth === '--' || selectedDay === '--') {
        reportFrame.src = 'about:blank';
        return; // Don't load if date is incomplete
    }

    const selectedDate = `${selectedYear}-${selectedMonth}-${selectedDay}`;

    // Find the latest time entry for the selected date
    const timeEntries = reportIndexData[selectedType]?.[selectedDate] || [];
    if (timeEntries.length === 0) {
        console.warn(`No time entries found for ${selectedType} on ${selectedDate}`);
        reportFrame.src = 'about:blank';
        return;
    }
    // Assume timeEntries are sorted descending by time in report_index.json
    const latestTimeEntry = timeEntries[0];
    const reports = latestTimeEntry.reports || [];

    // Find the default HTML report (e.g., not commentary)
    let reportToLoad = reports.find(r => r.path.endsWith('.html') && !r.path.includes('commentary'));
    // Fallback to the first available HTML report if default not found
    if (!reportToLoad) {
        reportToLoad = reports.find(r => r.path.endsWith('.html'));
    }
    // Fallback to the first report if no HTML found (though HTML is preferred)
    if (!reportToLoad && reports.length > 0) {
        reportToLoad = reports[0];
    }

    const selectedReportPath = reportToLoad ? reportToLoad.path : null;

    if (selectedReportPath) {
        showLoading();
        reportFrame.onload = hideLoading;
        reportFrame.onerror = () => {
            hideLoading();
            console.error("Failed to load report:", selectedReportPath);
            reportFrame.src = 'about:blank';
        };
        // Use the path directly from report_index.json
        reportFrame.src = selectedReportPath;
        console.log("Loading report:", selectedReportPath);
    } else {
        console.warn(`No suitable report found for ${selectedType} on ${selectedDate} at time ${latestTimeEntry.time}`);
        reportFrame.src = 'about:blank';
        hideLoading();
    }
}

// --- Event Listeners ---
reportTypeSelect.addEventListener('change', populateYears);
reportYearSelect.addEventListener('change', populateMonths);
reportMonthSelect.addEventListener('change', populateDays);
reportDaySelect.addEventListener('change', loadReport); // Load report when day changes

// --- Initial Load ---
fetchReportIndex();