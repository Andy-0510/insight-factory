(function(){
  // DOM
  const reportTypeSelect = document.getElementById('reportType');
  const reportYearSelect = document.getElementById('reportYear');
  const reportMonthSelect = document.getElementById('reportMonth');
  const reportDaySelect = document.getElementById('reportDay');
  const reportFrame = document.getElementById('reportFrame');
  const loadingIndicator = document.getElementById('loadingIndicator');
  const themeToggle = document.getElementById('themeToggle');
  const themeText = document.getElementById('themeText');

  // 상태
  let reportIndexData = {};
  const REPORT_INDEX_PATH = './report_index.json'; // 필요 시 경로 변경

  // 테마 초기화 (localStorage)
  const savedTheme = localStorage.getItem('ir_theme') || 'light';
  setTheme(savedTheme);

  function setTheme(mode){
    if(mode === 'dark'){
      document.documentElement.setAttribute('data-theme','dark');
      themeToggle?.classList.add('active');
      themeToggle?.setAttribute('aria-checked','true');
      if(themeText) themeText.textContent = '다크';
    } else {
      document.documentElement.removeAttribute('data-theme');
      themeToggle?.classList.remove('active');
      themeToggle?.setAttribute('aria-checked','false');
      if(themeText) themeText.textContent = '라이트';
    }
    try{ localStorage.setItem('ir_theme', mode); }catch(e){}
  }

  themeToggle?.addEventListener('click', ()=> {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    setTheme(isDark ? 'light' : 'dark');
    // 뷰어 폭 다시 맞춤 (테마 변경으로 레이아웃 영향 가능성 대비)
    setTimeout(fitViewerWidth, 120);
  });
  themeToggle?.addEventListener('keydown', (e)=>{
    if(e.key === 'Enter' || e.key === ' '){ e.preventDefault(); themeToggle.click(); }
  });

  // 로딩 인디케이터 제어
  function showLoading(){ if(loadingIndicator) loadingIndicator.style.display = 'flex'; }
  function hideLoading(){ if(loadingIndicator) loadingIndicator.style.display = 'none'; }

  // 연/월/일 채우기
  function populateYears(range = 10){
    const cur = new Date().getFullYear();
    if(!reportYearSelect) return;
    reportYearSelect.innerHTML = '';
    for(let y = cur; y >= cur - range; y--){
      const opt = document.createElement('option');
      opt.value = String(y);
      opt.textContent = `${y}년`;
      reportYearSelect.appendChild(opt);
    }
    reportYearSelect.value = String(cur);
  }
  function populateMonths(){
    if(!reportMonthSelect) return;
    reportMonthSelect.innerHTML = '';
    for(let m=1;m<=12;m++){
      const opt = document.createElement('option');
      opt.value = String(m).padStart(2,'0');
      opt.textContent = `${m}월`;
      reportMonthSelect.appendChild(opt);
    }
    reportMonthSelect.value = String(new Date().getMonth()+1).padStart(2,'0');
  }
  function populateDays(y, m){
    if(!reportDaySelect) return;
    const daysInMonth = new Date(Number(y), Number(m), 0).getDate();
    reportDaySelect.innerHTML = '';
    for(let d=1; d<=daysInMonth; d++){
      const opt = document.createElement('option');
      opt.value = String(d).padStart(2,'0');
      opt.textContent = `${d}일`;
      reportDaySelect.appendChild(opt);
    }
    const today = new Date();
    if(Number(y) === today.getFullYear() && Number(m) === (today.getMonth()+1)){
      reportDaySelect.value = String(today.getDate()).padStart(2,'0');
    } else {
      reportDaySelect.value = '01';
    }
  }

  // --- Report Loading ---
  function loadReport() {
    if(!reportFrame || !reportTypeSelect || !reportYearSelect || !reportMonthSelect || !reportDaySelect) return;

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
    let reportToLoad = reports.find(r => r.path && r.path.endsWith('.html') && !r.path.includes('commentary'));
    // Fallback to the first available HTML report if default not found
    if (!reportToLoad) {
      reportToLoad = reports.find(r => r.path && r.path.endsWith('.html'));
    }
    // Fallback to the first report if no HTML found
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
      console.warn(`No suitable report found for ${selectedType} on ${selectedDate}` + (latestTimeEntry?.time ? ` at time ${latestTimeEntry.time}` : ''));
      reportFrame.src = 'about:blank';
      hideLoading();
    }
  }

  // --- Event Listeners ---
  reportTypeSelect?.addEventListener('change', ()=> loadReport());
  reportYearSelect?.addEventListener('change', ()=> { populateDays(reportYearSelect.value, reportMonthSelect.value); });
  reportMonthSelect?.addEventListener('change', ()=> { populateDays(reportYearSelect.value, reportMonthSelect.value); });
  reportDaySelect?.addEventListener('change', loadReport);

  // --- Initial populate ---
  populateYears(8);
  populateMonths();
  populateDays(reportYearSelect.value, reportMonthSelect.value);

  // --- fit viewer width to controls-inner ---
  function fitViewerWidth(){
    const viewer = document.querySelector('.viewer');
    const inner = document.querySelector('.controls-inner');
    if(!viewer || !inner) return;
    const w = Math.round(inner.getBoundingClientRect().width);
    viewer.style.maxWidth = w + 'px';
    viewer.style.margin = '0 auto';
  }
  fitViewerWidth();
  window.addEventListener('resize', fitViewerWidth);

  // --- fetch report index and initial load ---
  async function fetchReportIndex(){
    try{
      const res = await fetch(REPORT_INDEX_PATH, { cache: 'no-cache' });
      if(!res.ok) throw new Error(`Failed to fetch report index: ${res.status}`);
      const data = await res.json();
      reportIndexData = data || {};
      console.log('Loaded report index');
      // try to load report after index is ready
      loadReport();
    }catch(err){
      console.error('Error loading report index:', err);
      reportIndexData = {};
      reportFrame.src = 'about:blank';
      hideLoading();
    }
  }
  fetchReportIndex();

  // expose for debugging
  window.IR = window.IR || {};
  window.IR.loadReport = loadReport;
  window.IR.fetchReportIndex = fetchReportIndex;

})();