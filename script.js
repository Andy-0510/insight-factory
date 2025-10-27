(function(){
  // -----------------------
  // DOM 요소
  // -----------------------
  const reportTypeSelect = document.getElementById('reportType'); // daily / weekly / monthly
  const reportYearSelect = document.getElementById('reportYear');
  const reportMonthSelect = document.getElementById('reportMonth');
  const reportDaySelect = document.getElementById('reportDay');
  // internal hidden selects for time/file handling (keeps UI simple)
  let internalTimeSelect = document.getElementById('internalTimeSelect');
  if(!internalTimeSelect){
    internalTimeSelect = document.createElement('select');
    internalTimeSelect.id = 'internalTimeSelect';
    internalTimeSelect.style.display = 'none';
    document.body.appendChild(internalTimeSelect);
  }
  let internalFileSelect = document.getElementById('internalFileSelect');
  if(!internalFileSelect){
    internalFileSelect = document.createElement('select');
    internalFileSelect.id = 'internalFileSelect';
    internalFileSelect.style.display = 'none';
    document.body.appendChild(internalFileSelect);
  }

  const reportFrame = document.getElementById('reportFrame');
  const loadingIndicator = document.getElementById('loadingIndicator');
  const themeToggle = document.getElementById('themeToggle');
  const themeText = document.getElementById('themeText');

  // -----------------------
  // 상태 및 설정
  // -----------------------
  let reportIndexData = {};            // 원본 JSON 데이터
  let availableDatesByType = {};       // { type: ['YYYY-MM-DD', ...], ... }
  const REPORT_INDEX_PATH = './report_index.json'; // 필요시 경로 수정

  // -----------------------
  // 유틸리티 함수
  // -----------------------
  function normalizeDateKey(raw) {
    if(!raw) return null;
    const s = String(raw).trim();
    const parts = s.split('-').map(p => p.trim());
    if(parts.length !== 3) return null;
    const y = parts[0].padStart(4,'0');
    const m = parts[1].padStart(2,'0');
    const d = parts[2].padStart(2,'0');
    if(!/^\d{4}$/.test(y) || !/^\d{2}$/.test(m) || !/^\d{2}$/.test(d)) return null;
    return `${y}-${m}-${d}`;
  }

  function parseDateKey(dateKey){
    const parts = String(dateKey).split('-');
    if(parts.length !== 3) return null;
    return { year: parts[0], month: parts[1].padStart(2,'0'), day: parts[2].padStart(2,'0') };
  }

  // -----------------------
  // 테마 처리 (로컬 저장)
  // -----------------------
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

  // -----------------------
  // 로딩 인디케이터
  // -----------------------
  function showLoading(){
    if(loadingIndicator) loadingIndicator.style.display = 'flex';
    if(reportFrame) reportFrame.style.opacity = '0.6';
  }
  function hideLoading(){
    if(loadingIndicator) loadingIndicator.style.display = 'none';
    if(reportFrame) reportFrame.style.opacity = '1';
  }

  // -----------------------
  // availableDates 맵 구성
  // -----------------------
  function buildAvailableDatesMap(){
    availableDatesByType = {};
    if(!reportIndexData || typeof reportIndexData !== 'object') return;
    Object.keys(reportIndexData).forEach(type => {
      const rawDates = Object.keys(reportIndexData[type] || {});
      const normalized = rawDates.map(d => normalizeDateKey(d)).filter(Boolean);
      const uniq = Array.from(new Set(normalized));
      // 최신순(내림차순)
      uniq.sort((a,b) => b.localeCompare(a));
      availableDatesByType[type] = uniq;
    });
    console.log('buildAvailableDatesMap ->', availableDatesByType);
  }

  // -----------------------
  // 인덱스 기반 드롭다운 채우기 (연/월/일)
  // -----------------------
  function populateYearsFromIndex(type){
    if(!reportYearSelect) return;
    reportYearSelect.innerHTML = '';
    const dates = availableDatesByType[type] || [];
    if(dates.length === 0){
      reportYearSelect.innerHTML = '<option>선택 가능 연도 없음</option>';
      reportYearSelect.disabled = true;
      // clear downstream selects
      reportMonthSelect.innerHTML = '<option>--</option>'; reportMonthSelect.disabled = true;
      reportDaySelect.innerHTML = '<option>--</option>'; reportDaySelect.disabled = true;
      return;
    }
    const years = Array.from(new Set(dates.map(d => parseDateKey(d)?.year).filter(Boolean)));
    years.sort((a,b) => b.localeCompare(a));
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
      reportDaySelect.innerHTML = '<option>--</option>'; reportDaySelect.disabled = true;
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
    // 시간 목록 채우기
    populateTimes();
  }

  // -----------------------
  // 시간/파일 채우기 (인덱스 기반)
  // -----------------------
  function getSelectedDateString(){
    const y = reportYearSelect?.value;
    const m = reportMonthSelect?.value;
    const d = reportDaySelect?.value;
    if(!y || !m || !d) return null;
    return `${y}-${String(m).padStart(2,'0')}-${String(d).padStart(2,'0')}`;
  }

  function populateTimes(){
    const type = reportTypeSelect?.value;
    const dateStr = getSelectedDateString();
    internalTimeSelect.innerHTML = '';
    internalFileSelect.innerHTML = '';
    internalFileSelect.disabled = true;
    if(!type || !dateStr){
      internalTimeSelect.innerHTML = '<option>--</option>';
      internalTimeSelect.disabled = true;
      reportFrame.src = 'about:blank';
      return;
    }
    const timeEntries = reportIndexData[type]?.[dateStr] || [];
    if(!Array.isArray(timeEntries) || timeEntries.length === 0){
      internalTimeSelect.innerHTML = '<option>선택 가능 시간 없음</option>';
      internalTimeSelect.disabled = true;
      reportFrame.src = 'about:blank';
      return;
    }
    // timeEntries assumed to be in desired order (we kept latest first when building map)
    timeEntries.forEach(entry => {
      const opt = document.createElement('option'); opt.value = entry.time; opt.textContent = entry.time; internalTimeSelect.appendChild(opt);
    });
    internalTimeSelect.disabled = false;
    internalTimeSelect.value = timeEntries[0].time;
    // populate files for this selected time
    populateFilesFromEntry(type, dateStr, internalTimeSelect.value);
    // attach onchange to keep behavior consistent if internalTimeSelect changes
    internalTimeSelect.onchange = () => populateFilesFromEntry(type, dateStr, internalTimeSelect.value);
  }

  function populateFilesFromEntry(type, dateStr, time){
    internalFileSelect.innerHTML = '';
    internalFileSelect.disabled = true;
    reportFrame.src = 'about:blank';
    if(!type || !dateStr || !time) return;
    const timeEntries = reportIndexData[type]?.[dateStr] || [];
    const selectedEntry = timeEntries.find(e => e.time === time) || {};
    const reports = Array.isArray(selectedEntry.reports) ? selectedEntry.reports : [];
    if(reports.length === 0){
      internalFileSelect.innerHTML = '<option>선택 가능 리포트 없음</option>';
      internalFileSelect.disabled = true;
      return;
    }
    let defaultReportPath = '';
    reports.forEach(rep => {
      const opt = document.createElement('option'); opt.value = rep.path; opt.textContent = rep.name || rep.path; internalFileSelect.appendChild(opt);
      if(!defaultReportPath && typeof rep.path === 'string' && rep.path.endsWith('.html') && !((rep.name||'').toLowerCase().includes('commentary'))){
        defaultReportPath = rep.path;
      }
    });
    internalFileSelect.disabled = false;
    internalFileSelect.value = defaultReportPath || (reports[0] && reports[0].path) || '';
    // 자동 로드
    loadReportFromPath(internalFileSelect.value);
  }

  // -----------------------
  // 실제 리포트 로드
  // -----------------------
  function loadReportFromPath(path){
    if(!reportFrame) return;
    if(!path){
      reportFrame.src = 'about:blank';
      hideLoading();
      return;
    }
    showLoading();
    reportFrame.onload = () => {
      hideLoading();
      // after load, re-apply iframe dark mode if active
      if(document.documentElement.getAttribute('data-theme') === 'dark') applyIframeDarkMode(true);
    };
    reportFrame.onerror = () => {
      hideLoading();
      console.error('Failed to load report:', path);
      reportFrame.src = 'about:blank';
    };
    console.log('Setting iframe.src =', path);
    reportFrame.src = path;
  }

  // -----------------------
  // iframe 다크모드 처리 (same-origin이면 스타일 주입, 아니면 filter)
  // -----------------------
  function applyIframeDarkMode(enable){
    if(!reportFrame) return;
    // clear previous filter
    reportFrame.style.filter = '';
    try {
      const doc = reportFrame.contentDocument || reportFrame.contentWindow.document;
      if(!doc) throw new Error('no doc');
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
            img, svg, video { filter: none !important; }
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
      reportFrame.style.filter = '';
      return;
    } catch (e) {
      // cross-origin -> fallback to filter
      if(enable){
        reportFrame.style.filter = 'invert(1) hue-rotate(180deg) contrast(1.02)';
      } else {
        reportFrame.style.filter = '';
      }
      return;
    }
  }

  // -----------------------
  // 이벤트 연결 (UI selects)
  // -----------------------
  reportTypeSelect?.addEventListener('change', () => {
    const newType = reportTypeSelect.value;
    populateYearsFromIndex(newType);
  });
  reportYearSelect?.addEventListener('change', () => {
    populateMonthsFromIndex(reportTypeSelect.value, reportYearSelect.value);
  });
  reportMonthSelect?.addEventListener('change', () => {
    populateDaysFromIndex(reportTypeSelect.value, reportYearSelect.value, reportMonthSelect.value);
  });
  reportDaySelect?.addEventListener('change', populateTimes);

  // -----------------------
  // 뷰어 너비 맞춤
  // -----------------------
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

  // -----------------------
  // report_index.json 로드
  // -----------------------
  async function fetchReportIndex(){
    // disable selects during load
    if(reportYearSelect) reportYearSelect.disabled = true;
    if(reportMonthSelect) reportMonthSelect.disabled = true;
    if(reportDaySelect) reportDaySelect.disabled = true;
    if(internalTimeSelect) { internalTimeSelect.disabled = true; internalTimeSelect.innerHTML = '<option>로딩...</option>'; }
    try{
      const res = await fetch(REPORT_INDEX_PATH, { cache: 'no-cache' });
      if(!res.ok) throw new Error(`Failed to fetch report index: ${res.status}`);
      const data = await res.json();
      reportIndexData = data || {};
      buildAvailableDatesMap();
      // ensure reportTypeSelect has a valid value
      const selType = reportTypeSelect?.value || Object.keys(reportIndexData)[0];
      if(selType) populateYearsFromIndex(selType);
      console.log('reportIndexData loaded:', Object.keys(reportIndexData));
    }catch(err){
      console.error('Error loading report index:', err);
      reportIndexData = {};
      if(internalTimeSelect) internalTimeSelect.innerHTML = '<option>목록 로드 실패</option>';
      if(internalFileSelect) internalFileSelect.innerHTML = '<option>--</option>';
      reportFrame.src = 'about:blank';
      hideLoading();
    }
  }
  fetchReportIndex();

  // expose for debug
  window.IR = window.IR || {};
  window.IR.fetchReportIndex = fetchReportIndex;
  window.IR.reportIndexData = reportIndexData;
  window.IR.availableDatesByType = availableDatesByType;
})();