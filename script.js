(function(){
  // DOM 요소
  const reportTypeSelect = document.getElementById('reportType');
  const reportYearSelect = document.getElementById('reportYear');
  const reportMonthSelect = document.getElementById('reportMonth');
  const reportDaySelect = document.getElementById('reportDay');
  const reportTimeSelect = document.getElementById('reportTime'); // optional: not shown in UI, only for backward compat if needed
  const reportFileSelect = document.createElement('select'); // hidden select used internally if needed
  // BUT our UI uses file selection automatically: we'll keep reportFileSelect as virtual; instead we insert actual selection via JS when needed
  // For compatibility, create an invisible select element attached to DOM (optional)
  reportFileSelect.id = 'reportFile';
  reportFileSelect.style.display = 'none';
  document.body.appendChild(reportFileSelect);

  const reportFrame = document.getElementById('reportFrame');
  const loadingIndicator = document.getElementById('loadingIndicator');
  const themeToggle = document.getElementById('themeToggle');
  const themeText = document.getElementById('themeText');

  // 상태
  let reportIndexData = {};
  let availableDatesByType = {}; // { type: [ "YYYY-MM-DD", ... ] }
  const REPORT_INDEX_PATH = './report_index.json';

  // 테마 초기화
  const savedTheme = localStorage.getItem('ir_theme') || 'light';
  setTheme(savedTheme);
  function setTheme(mode){
    if(mode === 'dark'){
      document.documentElement.setAttribute('data-theme','dark');
      themeToggle?.classList.add('active');
      themeToggle?.setAttribute('aria-checked','true');
      if(themeText) themeText.textContent = '다크';
      applyIframeDarkMode(true);
    } else {
      document.documentElement.removeAttribute('data-theme');
      themeToggle?.classList.remove('active');
      themeToggle?.setAttribute('aria-checked','false');
      if(themeText) themeText.textContent = '라이트';
      applyIframeDarkMode(false);
    }
    try{ localStorage.setItem('ir_theme', mode); }catch(e){}
  }
  themeToggle?.addEventListener('click', ()=> {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    setTheme(isDark ? 'light' : 'dark');
    setTimeout(fitViewerWidth, 120);
  });
  themeToggle?.addEventListener('keydown', (e)=>{
    if(e.key === 'Enter' || e.key === ' '){ e.preventDefault(); themeToggle.click(); }
  });

  // 로딩 인디케이터
  function showLoading(){
    if(loadingIndicator) loadingIndicator.style.display = 'flex';
    if(reportFrame) reportFrame.style.opacity = '0.6';
  }
  function hideLoading(){
    if(loadingIndicator) loadingIndicator.style.display = 'none';
    if(reportFrame) reportFrame.style.opacity = '1';
  }

  // 유틸: parse date key YYYY-MM-DD
  function parseDateKey(dateKey){
    const parts = String(dateKey).split('-');
    if(parts.length !== 3) return null;
    return { year: parts[0], month: parts[1].padStart(2,'0'), day: parts[2].padStart(2,'0') };
  }

  // buildAvailableDatesMap
  function buildAvailableDatesMap(){
    availableDatesByType = {};
    Object.keys(reportIndexData || {}).forEach(type => {
      const dates = Object.keys(reportIndexData[type] || {});
      // 최신 우선으로 정렬(내림차순)
      dates.sort((a,b) => b.localeCompare(a));
      availableDatesByType[type] = dates;
    });
  }

  // 인덱스 기반으로 연/월/일 채우기
  function populateYearsFromIndex(type){
    if(!reportYearSelect) return;
    reportYearSelect.innerHTML = '';
    const dates = availableDatesByType[type] || [];
    if(dates.length === 0){
      reportYearSelect.innerHTML = '<option>선택 가능 연도 없음</option>';
      reportYearSelect.disabled = true;
      return;
    }
    const years = Array.from(new Set(dates.map(d=>parseDateKey(d)?.year).filter(Boolean)));
    years.sort((a,b)=> b.localeCompare(a));
    years.forEach(y => {
      const opt = document.createElement('option'); opt.value = String(y); opt.textContent = `${y}년`; reportYearSelect.appendChild(opt);
    });
    reportYearSelect.disabled = false;
    reportYearSelect.value = years[0];
    populateMonthsFromIndex(type, reportYearSelect.value);
  }
  function populateMonthsFromIndex(type, year){
    if(!reportMonthSelect) return;
    reportMonthSelect.innerHTML = '';
    const dates = availableDatesByType[type] || [];
    const monthsSet = new Set();
    dates.forEach(d => {
      const p = parseDateKey(d);
      if(p && p.year === String(year)) monthsSet.add(p.month);
    });
    const months = Array.from(monthsSet);
    if(months.length === 0){
      reportMonthSelect.innerHTML = '<option>선택 가능 월 없음</option>';
      reportMonthSelect.disabled = true;
      return;
    }
    months.sort((a,b)=> b.localeCompare(a));
    months.forEach(m => {
      const opt = document.createElement('option'); opt.value = m; opt.textContent = `${parseInt(m,10)}월`; reportMonthSelect.appendChild(opt);
    });
    reportMonthSelect.disabled = false;
    reportMonthSelect.value = months[0];
    populateDaysFromIndex(type, year, reportMonthSelect.value);
  }
  function populateDaysFromIndex(type, year, month){
    if(!reportDaySelect) return;
    reportDaySelect.innerHTML = '';
    const dates = availableDatesByType[type] || [];
    const daysSet = new Set();
    dates.forEach(d => {
      const p = parseDateKey(d);
      if(p && p.year === String(year) && p.month === String(month).padStart(2,'0')) daysSet.add(p.day);
    });
    const days = Array.from(daysSet);
    if(days.length === 0){
      reportDaySelect.innerHTML = '<option>선택 가능 일 없음</option>';
      reportDaySelect.disabled = true;
      return;
    }
    days.sort((a,b)=> b.localeCompare(a));
    days.forEach(day => {
      const opt = document.createElement('option'); opt.value = day; opt.textContent = `${parseInt(day,10)}일`; reportDaySelect.appendChild(opt);
    });
    reportDaySelect.disabled = false;
    reportDaySelect.value = days[0];
    // 시간 채우기
    populateTimes();
  }

  // 선택된 연/월/일 -> YYYY-MM-DD
  function getSelectedDateString(){
    const y = reportYearSelect?.value;
    const m = reportMonthSelect?.value;
    const d = reportDaySelect?.value;
    if(!y || !m || !d) return null;
    return `${y}-${String(m).padStart(2,'0')}-${String(d).padStart(2,'0')}`;
  }

  // populateTimes (type + date -> time entries)
  function populateTimes(){
    const selectedType = reportTypeSelect?.value;
    const selectedDate = getSelectedDateString();
    // create or use hidden selects for time/file if needed
    // We'll use internal flow: populate time select (create on the fly if not present)
    // ensure a time select exists in DOM for events (we can keep it hidden)
    let timeSelect = document.getElementById('internalTimeSelect');
    if(!timeSelect){
      timeSelect = document.createElement('select');
      timeSelect.id = 'internalTimeSelect';
      timeSelect.style.display = 'none';
      document.body.appendChild(timeSelect);
    }
    timeSelect.innerHTML = '';
    // reset file hidden select
    reportFileSelect.innerHTML = '';
    reportFileSelect.disabled = true;
    if(!selectedType || !selectedDate){
      timeSelect.innerHTML = '<option>--</option>';
      timeSelect.disabled = true;
      reportFrame.src = 'about:blank';
      return;
    }
    const timeEntries = reportIndexData[selectedType]?.[selectedDate] || [];
    if(!Array.isArray(timeEntries) || timeEntries.length === 0){
      timeSelect.innerHTML = '<option>선택 가능 시간 없음</option>';
      timeSelect.disabled = true;
      reportFrame.src = 'about:blank';
      return;
    }
    timeEntries.forEach(entry => {
      const opt = document.createElement('option'); opt.value = entry.time; opt.textContent = entry.time; timeSelect.appendChild(opt);
    });
    timeSelect.disabled = false;
    timeSelect.value = timeEntries[0].time;
    // populate files for the chosen time
    populateFilesFromEntry(selectedType, selectedDate, timeSelect.value);
    // attach change handler so if we ever expose internalTimeSelect it updates files
    timeSelect.onchange = () => populateFilesFromEntry(selectedType, selectedDate, timeSelect.value);
  }

  // populateFilesFromEntry (type,date,time) -> fills hidden reportFileSelect and triggers loadReport
  function populateFilesFromEntry(type, date, time){
    reportFileSelect.innerHTML = '';
    reportFileSelect.disabled = true;
    reportFrame.src = 'about:blank';
    if(!type || !date || !time) return;
    const timeEntries = reportIndexData[type]?.[date] || [];
    const selectedEntry = timeEntries.find(e => e.time === time);
    const reports = (selectedEntry && Array.isArray(selectedEntry.reports)) ? selectedEntry.reports : [];
    if(reports.length === 0){
      reportFileSelect.innerHTML = '<option>선택 가능 리포트 없음</option>';
      reportFileSelect.disabled = true;
      return;
    }
    let defaultReportPath = '';
    reports.forEach(rep => {
      const opt = document.createElement('option'); opt.value = rep.path; opt.textContent = rep.name || rep.path; reportFileSelect.appendChild(opt);
      if(!defaultReportPath && typeof rep.path === 'string' && rep.path.endsWith('.html') && !(rep.name || '').toLowerCase().includes('commentary')){
        defaultReportPath = rep.path;
      }
    });
    reportFileSelect.disabled = false;
    reportFileSelect.value = defaultReportPath || reports[0].path;
    // auto load
    loadReportFromPath(reportFileSelect.value);
  }

  // loadReportFromPath: 실제 iframe 로드 (이전 loadReport와 동일)
  function loadReportFromPath(path){
    if(!reportFrame) return;
    if(!path){
      reportFrame.src = 'about:blank';
      hideLoading();
      return;
    }
    showLoading();
    // onload: try to inject styles if same-origin (handled in applyIframeDarkMode)
    reportFrame.onload = () => {
      hideLoading();
      // after load, also reapply dark-mode styling if active
      if(document.documentElement.getAttribute('data-theme') === 'dark') applyIframeDarkMode(true);
    };
    reportFrame.onerror = () => {
      hideLoading();
      console.error('Failed to load report:', path);
      reportFrame.src = 'about:blank';
    };
    reportFrame.src = path;
  }

  // === iframe dark-mode handling ===
  // Attempt to inject CSS into iframe if same-origin; if cross-origin, fall back to CSS filter
  function applyIframeDarkMode(enable){
    if(!reportFrame) return;
    // clear any previously applied filter first
    reportFrame.style.filter = '';
    // try to inject style into iframe document
    try {
      const doc = reportFrame.contentDocument || reportFrame.contentWindow.document;
      if(!doc) throw new Error('no doc');
      // create/replace style id
      const STYLE_ID = 'injected-dark-style';
      let s = doc.getElementById(STYLE_ID);
      if(enable){
        if(!s){
          s = doc.createElement('style');
          s.id = STYLE_ID;
          s.innerHTML = `
            :root, body { background: #0b1220 !important; color: #e6eef9 !important; }
            body, p, div, span, td, th, li, a { color: #e6eef9 !important; background: transparent !important; }
            table, pre, code { color: #e6eef9 !important; }
            a { color: #7ea2ff !important; }
            img, svg, video { filter: none !important; } /* 이미지 색 보정 원치 않음 */
          `;
          doc.head ? doc.head.appendChild(s) : doc.documentElement.appendChild(s);
        } else {
          s.innerHTML = `
            :root, body { background: #0b1220 !important; color: #e6eef9 !important; }
            body, p, div, span, td, th, li, a { color: #e6eef9 !important; background: transparent !important; }
            table, pre, code { color: #e6eef9 !important; }
            a { color: #7ea2ff !important; }
            img, svg, video { filter: none !important; }
          `;
        }
      } else {
        if(s) s.remove();
      }
      // success -> ensure no filter on iframe
      reportFrame.style.filter = '';
      return;
    } catch (e) {
      // cross-origin or access denied -> fallback
      if(enable){
        // apply CSS filter to iframe to invert dark text -> light
        // note: this inverts images too; we try to mitigate by double-inverting if necessary (complex), but keep simple
        reportFrame.style.filter = 'invert(1) hue-rotate(180deg) contrast(1.02)';
      } else {
        reportFrame.style.filter = '';
      }
      return;
    }
  }

  // Attaching event listeners for the visible selects (type/year/month/day)
  reportTypeSelect?.addEventListener('change', () => {
    populateYearsFromIndex(reportTypeSelect.value);
  });
  reportYearSelect?.addEventListener('change', () => {
    populateMonthsFromIndex(reportTypeSelect.value, reportYearSelect.value);
  });
  reportMonthSelect?.addEventListener('change', () => {
    populateDaysFromIndex(reportTypeSelect.value, reportYearSelect.value, reportMonthSelect.value);
  });
  reportDaySelect?.addEventListener('change', populateTimes);

  // fit viewer width
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

  // fetch index
  async function fetchReportIndex(){
    // initial UI lock
    if(reportYearSelect) reportYearSelect.disabled = true;
    if(reportMonthSelect) reportMonthSelect.disabled = true;
    if(reportDaySelect) reportDaySelect.disabled = true;
    if(reportTimeSelect) { reportTimeSelect.disabled = true; reportTimeSelect.innerHTML = '<option>로딩...</option>'; }

    try{
      const res = await fetch(REPORT_INDEX_PATH, { cache: 'no-cache' });
      if(!res.ok) throw new Error(`Failed to fetch report index: ${res.status}`);
      const data = await res.json();
      reportIndexData = data || {};
      buildAvailableDatesMap();
      // ensure reportTypeSelect exists and has value
      const selType = reportTypeSelect?.value || Object.keys(reportIndexData)[0];
      if(selType) populateYearsFromIndex(selType);
      console.log('Loaded report index and populated date selects.');
    }catch(err){
      console.error('Error loading report index:', err);
      reportIndexData = {};
      if(reportTimeSelect) reportTimeSelect.innerHTML = '<option>목록 로드 실패</option>';
      reportFrame.src = 'about:blank';
      hideLoading();
    }
  }
  fetchReportIndex();

  // expose for debugging
  window.IR = window.IR || {};
  window.IR.loadReportFromPath = loadReportFromPath;
  window.IR.fetchReportIndex = fetchReportIndex;
  window.IR.reportIndexData = reportIndexData;

})();