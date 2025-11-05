(function(){
  // -----------------------
  // DOM 요소
  // -----------------------
  const reportTypeSelect = document.getElementById('reportType'); // daily / weekly / monthly
  const reportYearSelect = document.getElementById('reportYear');
  const reportMonthSelect = document.getElementById('reportMonth');
  const reportDaySelect = document.getElementById('reportDay');
  // internal hidden selects for time/file handling
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
      if(themeText) themeText.textContent = '🌜(Dark Mode)';
      applyIframeDarkMode(true);
    } else {
      document.documentElement.removeAttribute('data-theme');
      themeToggle?.classList.remove('active');
      themeToggle?.setAttribute('aria-checked','false');
      if(themeText) themeText.textContent = '🌞(Light Mode)';
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
  // 여기서 표 헤더 음영(헤더 셀 배경/box-shadow) 강제 덮어쓰기 규칙 포함
  // -----------------------
  function applyIframeDarkMode(enable){
    if(!reportFrame) return;
    reportFrame.style.filter = '';
    try {
      const doc = reportFrame.contentDocument || reportFrame.contentWindow.document;
      if(!doc) throw new Error('no doc');
      const STYLE_ID = 'injected-dark-style';
      let s = doc.getElementById(STYLE_ID);
      const injectedCss = `
        :root, body { background: #0b1220 !important; color: #e6eef9 !important; }
        body, p, div, span, td, th, li, a { color: #e6eef9 !important; background: transparent !important; }
        table, pre, code { color: #e6eef9 !important; }
        a { color: #7ea2ff !important; }

        /* ========== 표 헤더 보강 (다크모드에서 음영/배경 덮어쓰기) ========== */
        thead, thead tr, thead th, table thead th {
          background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)) !important;
          color: #e6eef9 !important;
          border-bottom: 1px solid rgba(255,255,255,0.06) !important;
        }
        thead th, table thead th {
          box-shadow: inset 0 -6px 12px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.02) !important;
        }
        /* 특정 테이블 라이브러리 클래스들도 덮어쓰기 */
        .table-header, .thead-dark, .table .header-row, .tbl-header {
          background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)) !important;
          color: #e6eef9 !important;
        }
        /* inline style로 들어간 경우에도 강제 적용(광범위, 주의) */
        *[style] thead, *[style] thead th, *[style] .table-header {
          background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)) !important;
          color: #e6eef9 !important;
        }
      `;
      if(enable){
        if(!s){
          s = doc.createElement('style');
          s.id = STYLE_ID;
          s.innerHTML = injectedCss;
          doc.head ? doc.head.appendChild(s) : doc.documentElement.appendChild(s);
        } else {
          s.innerHTML = injectedCss;
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
//  function fitViewerWidth(){
//    const viewer = document.querySelector('.viewer');
//    const inner = document.querySelector('.controls-inner');
//    if(!viewer || !inner) return;
//    const w = Math.round(inner.getBoundingClientRect().width);
//    viewer.style.maxWidth = w + 'px';
//    viewer.style.margin = '0 auto';
//  }
//  fitViewerWidth();
//  window.addEventListener('resize', fitViewerWidth);

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
      if(internalFileSelect) internalFileSelect.innerHTML = '<option>--</option>';
      reportFrame.src = 'about:blank';
      hideLoading();
    }
  }
  fetchReportIndex();

  // ===== 챗봇 JS 시작 =====
  
  // 2단계에서 만든 HTML 요소들을 JS 변수로 가져옵니다.
  const chatbotWindow = document.getElementById('chatbot-window');
  const chatbotToggleButton = document.getElementById('chatbot-toggle-btn');
  const chatbotCloseButton = document.getElementById('chatbot-close-btn');
  const chatMessages = document.getElementById('chat-messages');
  const chatInput = document.getElementById('chat-input');
  const chatSendButton = document.getElementById('chat-send-btn');
  
  // 1단계에서 만든 '비밀 통로' 주소입니다. (※※※ 실제 주소로 변경 필요 ※※※)
  const CHATBOT_API_URL = 'https://chatbot-api.thecki003.workers.dev/'; 
  
  // '초인종(💬)' 버튼을 클릭했을 때
  chatbotToggleButton.addEventListener('click', () => {
      // 3단계 CSS에서 만든 'visible' 클래스를 붙여서 창을 '짠!' 하고 보이게 함
      chatbotWindow.classList.toggle('visible'); 
  });
  
  // 챗봇 창의 'X' 닫기 버튼을 클릭했을 때
  chatbotCloseButton.addEventListener('click', () => {
      // 'visible' 클래스를 제거해서 창을 숨김
      chatbotWindow.classList.remove('visible');
  });
  
  /**
   * 현재 iframe에 로드된 리포트의 텍스트를 읽어옵니다.
   * [span_0](start_span)(다크 모드 로직 [cite: 91-95]과 동일한 원리로 iframe에 접근합니다.)
   */
  /**
 * (수정된 버전)
 * 현재 iframe에 로드된 리포트의 텍스트를 읽어옵니다.
 */
function getCurrentReportText() {
    // 1. index.html에 있는 <iframe id="reportFrame">을 찾습니다.
    const reportFrame = document.getElementById('reportFrame');

    if (!reportFrame) {
        console.error("챗봇 오류: reportFrame 요소를 찾을 수 없습니다.");
        return "오류: 리포트 프레임을 찾을 수 없습니다.";
    }

    try {
        // 2. iframe 내부의 내용물(document)에 접근합니다.
        const iframeDoc = reportFrame.contentDocument || reportFrame.contentWindow.document;

        // 3. (안전 장치) iframe의 body가 완전히 로드되었는지 확인합니다.
        if (!iframeDoc || !iframeDoc.body) {
            console.warn("챗봇 경고: 리포트가 아직 로드되지 않았거나, 내용에 접근할 수 없습니다.");
            return "리포트 내용이 아직 로드되지 않았습니다. 잠시 후 다시 시도해주세요.";
        }

        // 4. 스타일 대신, 화면에 보이는 '모든 텍스트'를 복사합니다.
        const reportText = iframeDoc.body.innerText;

        // 5. 텍스트가 너무 길면 AI가 힘들어하므로, 앞부분 5000자 정도만 잘라서 줍니다.
        return reportText.substring(0, 5000); 

    } catch (e) {
        console.error("챗봇이 리포트 내용을 읽는 데 실패했습니다:", e);
        // 5단계(CORS) 문제일 가능성이 높습니다.
        return "오류: 현재 리포트 내용에 접근할 수 없습니다. (보안 정책[CORS] 문제일 수 있습니다.)";
    }
}
  
  // '전송' 버튼 클릭 이벤트
  chatSendButton.addEventListener('click', handleSendMessage);
  
  // '입력창'에서 Enter 키 누름 이벤트
  chatInput.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
          handleSendMessage();
      }
  });
  
  /**
   * 메시지를 전송하는 핵심 함수
   */
  async function handleSendMessage() {
      const userQuestion = chatInput.value.trim(); // 사용자가 입력한 질문
      if (!userQuestion) return; // 질문이 없으면 무시
  
      // 1. 내 질문을 채팅창에 먼저 표시
      addMessageToChat('user', userQuestion);
      chatInput.value = ''; // 입력창 비우기
  
      // 2. (핵심) 현재 리포트 텍스트 읽어오기 (4-3 함수 호출)
      const reportContext = getCurrentReportText();
  
      // 3. '비밀 통로'로 "질문"과 "리포트 텍스트"를 함께 전송
      try {
          const response = await fetch(CHATBOT_API_URL, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                  question: userQuestion,  // 사용자의 질문
                  context: reportContext // 리포트 텍스트
              })
          });
  
          if (!response.ok) {
              throw new Error(`API 서버 오류: ${response.statusText}`);
          }
  
          const data = await response.json();
          const aiAnswer = data.answer || "답변을 받는 데 실패했습니다.";
  
          // 4. AI 답변을 채팅창에 표시
          addMessageToChat('bot', aiAnswer);
  
      } catch (error) {
          console.error('챗봇 API 통신 오류:', error);
          addMessageToChat('bot', `오류가 발생했습니다: ${error.message}`);
      }
  }
  
  /**
   * 채팅창에 말풍선을 추가하는 도우미 함수
   */
  function addMessageToChat(sender, text) {
      const messageElement = document.createElement('div');
      messageElement.classList.add('message', sender); // 'message'와 'user' 또는 'bot' 스타일 적용
      messageElement.innerText = text;
      
      chatMessages.appendChild(messageElement); // 채팅창에 추가
      
      // 새 메시지가 오면 항상 스크롤을 맨 아래로 내림
      chatMessages.scrollTop = chatMessages.scrollHeight; 
  }
  
  
  // expose for debug
  window.IR = window.IR || {};
  window.IR.fetchReportIndex = fetchReportIndex;
  window.IR.reportIndexData = reportIndexData;
  window.IR.availableDatesByType = availableDatesByType;
})();
