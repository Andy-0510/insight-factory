(function(){
  // -----------------------
  // DOM 요소
  // -----------------------
  const reportTypeSelect = document.getElementById('reportType');
  const reportYearSelect = document.getElementById('reportYear');
  const reportMonthSelect = document.getElementById('reportMonth');
  const reportDaySelect = document.getElementById('reportDay');

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

  const reportContainer = document.getElementById('reportContainer');
  const loadingIndicator = document.getElementById('loadingIndicator');
  const themeToggle = document.getElementById('themeToggle');
  const themeText = document.getElementById('themeText');

  // -----------------------
  // 상태 및 설정
  // -----------------------
  let reportIndexData = {};
  let availableDatesByType = {};
  const REPORT_INDEX_PATH = './report_index.json';

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
  // 테마 초기화
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
  // 드롭다운 채우기
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
    populateTimes();
  }

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
    loadReportFromPath_seamless(internalFileSelect.value);
  }

  function createSeamlessSandboxIframe(htmlText) {
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
    // allow-same-origin 추가: 동일 출처 접근이 필요할 때만 허용하세요 (보안 주의)
    iframe.setAttribute('sandbox', 'allow-scripts allow-forms allow-popups allow-same-origin');
    iframe.src = blobUrl;
    iframe.dataset.blobUrl = blobUrl;

    iframe.style.width = '100%';
    iframe.style.height = '400px';
    iframe.style.border = '0';
    iframe.style.boxShadow = 'none';
    iframe.style.background = 'transparent';
    iframe.style.display = 'block';
    iframe.style.margin = '0';
    iframe.style.padding = '0';
    iframe.style.overflow = 'visible';
    iframe.style.minHeight = '200px';

    iframe.addEventListener('load', () => {
      try {
        const doc = iframe.contentDocument || iframe.contentWindow.document;
        const body = doc.body;
        const htmlEl = doc.documentElement;
        const newHeight = Math.max(
          body ? body.scrollHeight : 0,
          htmlEl ? htmlEl.scrollHeight : 0
        );
        if(newHeight && newHeight > 0) {
          iframe.style.height = newHeight + 'px';
        } else {
          iframe.style.height = '600px';
        }

        // 동일 출처면 내부 스타일 보정: 글꼴 크기, 중앙 정렬
        try {
          const injectedCss = `
            body { font-size: ${getComputedStyle(document.documentElement).getPropertyValue('--body-font-size') || '16px'} !important; text-align: center !important; line-height:1.6 !important; }
            h1,h2,h3 { text-align:center !important; }
            table, pre, code { font-size: 15px !important; }
          `;
          const docHead = doc.head || doc.getElementsByTagName('head')[0] || doc.documentElement;
          let s = doc.getElementById('injected-size-style');
          if(!s){
            s = doc.createElement('style'); s.id = 'injected-size-style'; s.innerHTML = injectedCss; docHead.appendChild(s);
          } else {
            s.innerHTML = injectedCss;
          }
          // --- 다크 텍스트 흰색 강제 주입 ---
          if(document.documentElement.getAttribute('data-theme') === 'dark'){
            const darkCss = `
              html, body, p, div, span, li, a, td, th {
                color: #ffffff !important;
                background: transparent !important;
              }
              thead th, table thead th, .table-header, .thead-dark {
                color: #ffffff !important;
                background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)) !important;
                border-bottom: 1px solid rgba(255,255,255,0.06) !important;
              }
              pre, code {
                color: #f7fafc !important;
                background: rgba(255,255,255,0.02) !important;
              }
              *[style] { color: #ffffff !important; background: transparent !important; }
              a { color: #9cc2ff !important; }
              img, svg { filter: none !important; }
            `;
            let s2 = doc.getElementById('injected-dark-style');
            if(!s2){
              s2 = doc.createElement('style');
              s2.id = 'injected-dark-style';
              s2.innerHTML = darkCss;
              docHead.appendChild(s2);
            } else {
              s2.innerHTML = darkCss;
            }

            // inline 스타일 제거 시도 (같은 출처일 때만 가능)
            try {
              const walker = doc.createTreeWalker(doc.documentElement, NodeFilter.SHOW_ELEMENT, null, false);
              const root = doc.documentElement;
              if(root && root.style){
                root.style.color = '';
                root.style.backgroundColor = '';
              }
              let node = walker.nextNode();
              while(node){
                try{
                  if(node.style){
                    if(node.style.color) node.style.color = '';
                    if(node.style.backgroundColor) node.style.backgroundColor = '';
                  }
                  if(node.hasAttribute){
                    if(node.getAttribute('color')) node.removeAttribute('color');
                    if(node.getAttribute('bgcolor')) node.removeAttribute('bgcolor');
                  }
                }catch(e){
                  // ignore node-level errors
                }
                node = walker.nextNode();
              }
            } catch(e){
              console.warn('inline color removal failed:', e);
            }

            // 안전망: 모든 요소에 강제 스타일
            try {
              const FORCE_ID = 'injected-force-dark';
              const forceCss = `* { color: #ffffff !important; background: transparent !important; }`;
              let f = doc.getElementById(FORCE_ID);
              if(!f){
                f = doc.createElement('style');
                f.id = FORCE_ID;
                f.innerHTML = forceCss;
                docHead.appendChild(f);
              } else {
                f.innerHTML = forceCss;
              }
            } catch(e){
              console.warn('force style injection failed:', e);
            }

          } else {
            // 라이트 모드면 이전에 주입한 다크 스타일 제거
            const prev = doc.getElementById('injected-dark-style'); if(prev) prev.remove();
            const prevForce = doc.getElementById('injected-force-dark'); if(prevForce) prevForce.remove();
          }
        } catch(e) {
          console.warn('inner-doc styling failed (maybe cross-origin):', e);
        }

      } catch(e) {
        console.warn('자동 높이 실패(크로스오리진):', e);
        iframe.style.height = '640px';
        iframe.style.overflow = 'auto';
        // 크로스오리진이면 폴백: 필터로 다크 보정
        if(document.documentElement.getAttribute('data-theme') === 'dark'){
          try{ iframe.style.filter = 'invert(1) hue-rotate(180deg) contrast(1.02) brightness(0.96)'; }catch(e){}
        } else {
          iframe.style.filter = '';
        }
      }
      hideLoading();
    });

    iframe.addEventListener('error', () => {
      hideLoading();
      reportContainer.innerHTML = `<div style="color:#e00;padding:12px;">리포트를 불러오지 못했습니다.</div>`;
    });

    reportContainer.appendChild(iframe);
    return iframe;
  }

  async function loadReportFromPath_seamless(path){
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

      try{
        const baseUrl = new URL(path, location.href).href.replace(/\/[^/]*$/, '/');
        if(/<base[^>]*>/i.test(html)){
          html = html.replace(/<base[^/]*>/i, `<base href="${baseUrl}">`);
        } else if(/<head[^>]*>/i.test(html)){
          html = html.replace(/<head([^>]*)>/i, `<head$1>\n<base href="${baseUrl}">`);
        } else {
          html = `<base href="${baseUrl}">` + html;
        }
      }catch(e){ console.warn('base inject failed', e); }

      createSeamlessSandboxIframe(html);
    }catch(err){
      console.error('report fetch/load error', err);
      reportContainer.innerHTML = `<div style="color:#e00;padding:12px;">리포트를 불러오지 못했습니다. (${err.message})</div>`;
      hideLoading();
    }
  }

  // 이벤트 연결
  reportTypeSelect?.addEventListener('change', () => populateYearsFromIndex(reportTypeSelect.value));
  reportYearSelect?.addEventListener('change', () => populateMonthsFromIndex(reportTypeSelect.value, reportYearSelect.value));
  reportMonthSelect?.addEventListener('change', () => populateDaysFromIndex(reportTypeSelect.value, reportYearSelect.value, reportMonthSelect.value));
  reportDaySelect?.addEventListener('change', populateTimes);

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

  // fetch report_index
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

  // expose debug
  window.IR = window.IR || {};
  window.IR.fetchReportIndex = fetchReportIndex;
  window.IR.reportIndexData = reportIndexData;
  window.IR.availableDatesByType = availableDatesByType;
  window.IR.loadReportFromPath_seamless = loadReportFromPath_seamless;

})();
