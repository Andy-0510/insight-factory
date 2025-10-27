(function(){
  // -----------------------
  // DOM 요소 (index.html 구조와 일치해야 함)
  // -----------------------
  const reportTypeSelect = document.getElementById('reportType');
  const reportYearSelect = document.getElementById('reportYear');
  const reportMonthSelect = document.getElementById('reportMonth');
  const reportDaySelect = document.getElementById('reportDay');

  // hidden internal selects (시간/파일 관리를 위해 내부적으로 사용)
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

  const reportContainer = document.getElementById('reportContainer'); // 리포트가 들어갈 컨테이너
  const loadingIndicator = document.getElementById('loadingIndicator');
  const themeToggle = document.getElementById('themeToggle');
  const themeText = document.getElementById('themeText');

  // -----------------------
  // 상태 및 설정
  // -----------------------
  let reportIndexData = {};
  let availableDatesByType = {};
  const REPORT_INDEX_PATH = './report_index.json'; // 필요시 경로 수정

  // -----------------------
  // 유틸 함수
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
  // 테마 초기화 (localStorage)
  // -----------------------
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
    if(reportContainer) reportContainer.style.opacity = '0.6';
  }
  function hideLoading(){
    if(loadingIndicator) loadingIndicator.style.display = 'none';
    if(reportContainer) reportContainer.style.opacity = '1';
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
  // 시간/파일 채우기
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
      if(reportContainer) reportContainer.innerHTML = '';
      return;
    }
    const timeEntries = reportIndexData[type]?.[dateStr] || [];
    if(!Array.isArray(timeEntries) || timeEntries.length === 0){
      internalTimeSelect.innerHTML = '<option>선택 가능 시간 없음</option>';
      internalTimeSelect.disabled = true;
      if(reportContainer) reportContainer.innerHTML = '';
      return;
    }
    timeEntries.forEach(entry => {
      const opt = document.createElement('option'); opt.value = entry.time; opt.textContent = entry.time; internalTimeSelect.appendChild(opt);
    });
    internalTimeSelect.disabled = false;
    internalTimeSelect.value = timeEntries[0].time;
    populateFilesFromEntry(type, dateStr, internalTimeSelect.value);
    internalTimeSelect.onchange = () => populateFilesFromEntry(type, dateStr, internalTimeSelect.value);
  }

  function populateFilesFromEntry(type, dateStr, time){
    internalFileSelect.innerHTML = '';
    internalFileSelect.disabled = true;
    if(!type || !dateStr || !time){
      if(reportContainer) reportContainer.innerHTML = '';
      return;
    }
    const timeEntries = reportIndexData[type]?.[dateStr] || [];
    const selectedEntry = timeEntries.find(e => e.time === time) || {};
    const reports = Array.isArray(selectedEntry.reports) ? selectedEntry.reports : [];
    if(reports.length === 0){
      internalFileSelect.innerHTML = '<option>선택 가능 리포트 없음</option>';
      internalFileSelect.disabled = true;
      if(reportContainer) reportContainer.innerHTML = '';
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
    loadReportFromPath(internalFileSelect.value);
  }

  // -----------------------
  // 안전 실행: 샌드박스 iframe으로 리포트 HTML 로드
  // -----------------------
  function createSandboxIframeForHtml(htmlText) {
    // 이전 blob URL 정리
    while(reportContainer.firstChild) {
      const node = reportContainer.firstChild;
      if(node.dataset && node.dataset.blobUrl) {
        try{ URL.revokeObjectURL(node.dataset.blobUrl); }catch(e){}
      }
      reportContainer.removeChild(node);
    }

    const blob = new Blob([htmlText], { type: 'text/html' });
    const blobUrl = URL.createObjectURL(blob);

    const iframe = document.createElement('iframe');
    iframe.setAttribute('sandbox', 'allow-scripts allow-forms allow-popups'); // 부모와 분리
    iframe.style.width = '100%';
    iframe.style.height = '100%';
    iframe.style.border = '0';
    iframe.style.minHeight = '360px';
    iframe.dataset.blobUrl = blobUrl;
    iframe.src = blobUrl;

    reportContainer.appendChild(iframe);
    return iframe;
  }

  async function loadReportFromPath(path){
    if(!reportContainer) return;
    if(!path){
      reportContainer.innerHTML = '';
      hideLoading();
      return;
    }
    showLoading();
    try{
      const res = await fetch(path, { cache: 'no-cache' });
      if(!res.ok) throw new Error('Failed to fetch: ' + res.status);
      let html = await res.text();

      // base 태그 삽입: 상대경로 보정
      try{
        const baseUrl = new URL(path, location.href).href.replace(/\/[^/]*$/, '/');
        if(/<base[^>]*>/i.test(html)){
          html = html.replace(/<base[^>]*>/i, `<base href="${baseUrl}">`);
        } else if(/<head[^>]*>/i.test(html)){
          html = html.replace(/<head([^>]*)>/i, `<head$1>\n<base href="${baseUrl}">`);
        } else {
          html = `<base href="${baseUrl}">` + html;
        }
      }catch(e){ console.warn('base inject failed', e); }

      // (선택) 스크립트 제거를 원하면 아래 주석 해제
      // html = html.replace(/<script[\s\S]*?<\/script>/gi, '<!-- script removed -->');

      // create sandbox iframe
      createSandboxIframeForHtml(html);

      hideLoading();
    }catch(err){
      console.error('report fetch/load error', err);
      reportContainer.innerHTML = `<div style="color:#e00;padding:12px;">리포트를 불러오지 못했습니다. (${err.message})</div>`;
      hideLoading();
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
      const selType = reportTypeSelect?.value || Object.keys(reportIndexData)[0];
      if(selType) populateYearsFromIndex(selType);
      console.log('reportIndexData loaded:', Object.keys(reportIndexData));
    }catch(err){
      console.error('Error loading report index:', err);
      reportIndexData = {};
      if(internalTimeSelect) internalTimeSelect.innerHTML = '<option>목록 로드 실패</option>';
      reportContainer.innerHTML = '';
      hideLoading();
    }
  }
  fetchReportIndex();

  // expose debug helpers
  window.IR = window.IR || {};
  window.IR.fetchReportIndex = fetchReportIndex;
  window.IR.reportIndexData = reportIndexData;
  window.IR.availableDatesByType = availableDatesByType;
  window.IR.loadReportFromPath = loadReportFromPath;

})();