(function(){
  // DOM 요소 (현재 UI에 맞춘 id)
  const reportTypeSelect = document.getElementById('reportType'); // 리포트 종류
  const reportYearSelect = document.getElementById('reportYear'); // 연도
  const reportMonthSelect = document.getElementById('reportMonth'); // 월
  const reportDaySelect = document.getElementById('reportDay'); // 일
  const reportTimeSelect = document.getElementById('reportTime'); // 시간(기존 인덱스 기반)
  const reportFileSelect = document.getElementById('reportFile'); // 파일 리스트(경로 값)
  const reportFrame = document.getElementById('reportFrame');
  const loadingIndicator = document.getElementById('loadingIndicator');
  const themeToggle = document.getElementById('themeToggle');
  const themeText = document.getElementById('themeText');

  // 상태 및 설정
  let reportIndexData = {};
  const REPORT_INDEX_PATH = './report_index.json'; // 필요시 경로 변경

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

  // 연/월/일 기본 채우기 (UI 유지)
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

  // Helper: 선택된 연/월/일을 "YYYY-MM-DD" 형식으로 반환
  function getSelectedDateString(){
    const y = reportYearSelect?.value;
    const m = reportMonthSelect?.value;
    const d = reportDaySelect?.value;
    if(!y || !m || !d) return null;
    return `${y}-${m}-${d}`;
  }

  // --- 기존 로직을 연/월/일 UI에 맞춰 재구성 ---
  // populateTimes: 선택한 타입 + 날짜 -> 해당 날짜의 time entries로 select 채움
  function populateTimes(){
    if(!reportTimeSelect || !reportFileSelect) return;
    const selectedType = reportTypeSelect?.value;
    const selectedDate = getSelectedDateString();
    reportTimeSelect.innerHTML = '';
    reportFileSelect.innerHTML = '<option>--</option>';
    reportFileSelect.disabled = true;
    reportFrame.src = 'about:blank';

    if(!selectedType || !selectedDate){
      reportTimeSelect.innerHTML = '<option>--</option>';
      reportTimeSelect.disabled = true;
      return;
    }

    const timeEntries = reportIndexData[selectedType]?.[selectedDate] || [];
    if(!Array.isArray(timeEntries) || timeEntries.length === 0){
      reportTimeSelect.innerHTML = '<option>선택 가능 시간 없음</option>';
      reportTimeSelect.disabled = true;
      return;
    }

    // timeEntries 배열에서 각 entry.time을 option으로 추가
    timeEntries.forEach(entry => {
      const opt = document.createElement('option');
      opt.value = entry.time;
      opt.textContent = entry.time;
      reportTimeSelect.appendChild(opt);
    });

    reportTimeSelect.disabled = false;
    // 기본으로 최신(첫 항목) 선택
    reportTimeSelect.value = timeEntries[0].time;
    populateFiles(); // 시간 채운 뒤 파일 목록 채우기
  }

  // populateFiles: 선택된 타입/날짜/시간 -> reports 목록 채워서 파일 select에 넣음
  function populateFiles(){
    if(!reportFileSelect || !reportTimeSelect) return;
    const selectedType = reportTypeSelect?.value;
    const selectedDate = getSelectedDateString();
    const selectedTime = reportTimeSelect?.value;
    reportFileSelect.innerHTML = '';
    reportFrame.src = 'about:blank';

    if(!selectedType || !selectedDate || !selectedTime){
      reportFileSelect.innerHTML = '<option>--</option>';
      reportFileSelect.disabled = true;
      return;
    }

    const timeEntries = reportIndexData[selectedType]?.[selectedDate] || [];
    const selectedEntry = timeEntries.find(e => e.time === selectedTime) || {};
    const reports = selectedEntry.reports || [];

    if(!Array.isArray(reports) || reports.length === 0){
      reportFileSelect.innerHTML = '<option>선택 가능 리포트 없음</option>';
      reportFileSelect.disabled = true;
      return;
    }

    let defaultReportPath = '';
    // reports: { name, path, ... } 구조 가정
    reports.forEach(rep => {
      const opt = document.createElement('option');
      opt.value = rep.path;
      opt.textContent = rep.name || rep.path;
      reportFileSelect.appendChild(opt);
      if(!defaultReportPath && typeof rep.path === 'string' && rep.path.endsWith('.html') && !(rep.name || '').toLowerCase().includes('commentary')){
        defaultReportPath = rep.path;
      }
    });

    reportFileSelect.disabled = false;
    // 기본값 설정: 우선 defaultReportPath, 없으면 첫 리포트
    reportFileSelect.value = defaultReportPath || (reports[0] && reports[0].path) || '';
    // 자동으로 로드
    loadReport();
  }

  // --- loadReport (파일 선택 시 iframe 로드) ---
  function loadReport() {
    if(!reportFileSelect || !reportFrame) return;
    const selectedReportPath = reportFileSelect.value;
    if (selectedReportPath) {
      showLoading();
      reportFrame.onload = hideLoading;
      reportFrame.onerror = () => {
        hideLoading();
        console.error("Failed to load report:", selectedReportPath);
        reportFrame.src = 'about:blank';
      };
      reportFrame.src = selectedReportPath;
    } else {
      reportFrame.src = 'about:blank';
      hideLoading();
    }
  }

  // --- Event listeners 연결 ---
  // 리포트 종류 바뀌면 날짜(연/월/일) 기반으로 다시 time/file 채우기 시도
  reportTypeSelect?.addEventListener('change', () => {
    // 기존 인덱스에서 사용 가능한 날짜 중 현재 선택 연/월/일이 없을 수 있으므로 검증 필요
    // 간단히 populateTimes 호출해서 가능한 시간/파일 목록 갱신
    populateTimes();
  });

  // 연/월/일 변경 시에는 날짜 문자열 다시 계산해서 시간 목록 갱신
  reportYearSelect?.addEventListener('change', () => {
    populateDays(reportYearSelect.value, reportMonthSelect.value);
    // populateDays 내부에서 day값 셋팅 되므로 그 후에 시간 목록 갱신
    setTimeout(populateTimes, 0);
  });
  reportMonthSelect?.addEventListener('change', () => {
    populateDays(reportYearSelect.value, reportMonthSelect.value);
    setTimeout(populateTimes, 0);
  });
  reportDaySelect?.addEventListener('change', populateTimes);

  // 시간/파일 변경 리스너
  reportTimeSelect?.addEventListener('change', populateFiles);
  reportFileSelect?.addEventListener('change', loadReport);

  // --- 초기화: 연/월/일 채우기 및 인덱스 fetch ---
  populateYears(8);
  populateMonths();
  populateDays(reportYearSelect.value, reportMonthSelect.value);

  // fit viewer width (controls-inner과 동일 너비 유지)
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

  // --- report_index.json 불러오기 (기존 fetch 로직 재사용) ---
  async function fetchReportIndex(){
    // 비활성화 상태 표시: 시간/파일 셀렉트 (초기 로딩 UX)
    if(reportTimeSelect) { reportTimeSelect.disabled = true; reportTimeSelect.innerHTML = '<option>로딩...</option>'; }
    if(reportFileSelect) { reportFileSelect.disabled = true; reportFileSelect.innerHTML = '<option>--</option>'; }
    try{
      const res = await fetch(REPORT_INDEX_PATH, { cache: 'no-cache' });
      if(!res.ok) throw new Error(`Failed to fetch report index: ${res.status}`);
      const data = await res.json();
      reportIndexData = data || {};
      console.log('Loaded report index');
      // 인덱스 로드 후 현재 선택된 연/월/일(또는 기본 값)으로 시간/파일 채우기 시도
      populateTimes();
    }catch(err){
      console.error('Error loading report index:', err);
      reportIndexData = {};
      if(reportTimeSelect) reportTimeSelect.innerHTML = '<option>목록 로드 실패</option>';
      if(reportFileSelect) reportFileSelect.innerHTML = '<option>--</option>';
      if(reportTimeSelect) reportTimeSelect.disabled = true;
      if(reportFileSelect) reportFileSelect.disabled = true;
      reportFrame.src = 'about:blank';
      hideLoading();
    }
  }
  fetchReportIndex();

  // 디버깅용 노출
  window.IR = window.IR || {};
  window.IR.loadReport = loadReport;
  window.IR.fetchReportIndex = fetchReportIndex;
  window.IR.reportIndexData = reportIndexData;

})();