(function(){
  // DOM 요소
  const reportTypeSelect = document.getElementById('reportType');
  const reportYearSelect = document.getElementById('reportYear');
  const reportMonthSelect = document.getElementById('reportMonth');
  const reportDaySelect = document.getElementById('reportDay');
  const reportFrame = document.getElementById('reportFrame');
  const loadingIndicator = document.getElementById('loadingIndicator');
  const themeToggle = document.getElementById('themeToggle');
  const themeText = document.getElementById('themeText');

  // 내부 상태
  let reportIndexData = {}; // report_index.json 로드 결과 저장
  const REPORT_INDEX_PATH = './report_index.json'; // 필요시 경로 변경

  // 토글 테마 (기존 코드 유지)
  const savedTheme = localStorage.getItem('ir_theme') || 'light';
  setTheme(savedTheme);
  function setTheme(mode){
    if(mode === 'dark'){
      document.documentElement.setAttribute('data-theme','dark');
      themeToggle.classList.add('active');
      themeToggle.setAttribute('aria-checked','true');
      themeText.textContent = '다크';
    } else {
      document.documentElement.removeAttribute('data-theme');
      themeToggle.classList.remove('active');
      themeToggle.setAttribute('aria-checked','false');
      themeText.textContent = '라이트';
    }
    localStorage.setItem('ir_theme', mode);
  }
  themeToggle?.addEventListener('click', ()=> {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    setTheme(isDark ? 'light' : 'dark');
  });
  themeToggle?.addEventListener('keydown', (e)=>{
    if(e.key === 'Enter' || e.key === ' '){ e.preventDefault(); themeToggle.click(); }
  });

  // 로딩 표시 제어
  function showLoading(){
    if(loadingIndicator) loadingIndicator.style.display = 'flex';
  }
  function hideLoading(){
    if(loadingIndicator) loadingIndicator.style.display = 'none';
  }

  // 연/월/일 채우기 함수
  function populateYears(range = 10){
    const cur = new Date().getFullYear();
    reportYearSelect.innerHTML = '';
    for(let y = cur; y >= cur - range; y--){
      const opt = document.createElement('option');
      opt.value = y;
      opt.textContent = `${y}년`;
      reportYearSelect.appendChild(opt);
    }
    reportYearSelect.value = cur;
  }
  function populateMonths(){
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

  // --- 네가 주신 실제 로드 로직을 반영한 loadReport ---
  function loadReport() {
    const selectedType = reportTypeSelect.value;
    const selectedYear = reportYearSelect.value;
    const selectedMonth = reportMonthSelect.value;
    const selectedDay = reportDaySelect.value;

    // 날짜가 완전한지 체크
    if (!selectedYear || !selectedMonth || !selectedDay || selectedYear === '--' || selectedMonth === '--' || selectedDay === '--') {
      reportFrame.src = 'about:blank';
      return; // 날짜 불완전 시 로드 중단
    }

    const selectedDate = `${selectedYear}-${selectedMonth}-${selectedDay}`;

    // 인덱스에서 해당 타입/날짜의 시간 항목 찾기
    const timeEntries = reportIndexData[selectedType]?.[selectedDate] || [];
    if (!Array.isArray(timeEntries) || timeEntries.length === 0) {
      console.warn(`No time entries found for ${selectedType} on ${selectedDate}`);
      reportFrame.src = 'about:blank';
      return;
    }

    // report_index.json이 시간순(내림차순) 정렬되어 있다고 가정 -> 첫 항목이 최신
    const latestTimeEntry = timeEntries[0];
    const reports = latestTimeEntry.reports || [];

    // 우선순위: HTML 리포트(주석 제외) -> HTML 리포트(첫번째) -> 첫 리포트(어떤 형식이든)
    let reportToLoad = reports.find(r => typeof r.path === 'string' && r.path.endsWith('.html') && !r.path.includes('commentary'));
    if (!reportToLoad) {
      reportToLoad = reports.find(r => typeof r.path === 'string' && r.path.endsWith('.html'));
    }
    if (!reportToLoad && reports.length > 0) {
      reportToLoad = reports[0];
    }

    const selectedReportPath = reportToLoad ? reportToLoad.path : null;

    if (selectedReportPath) {
      showLoading();
      // onload / onerror 핸들러 설정
      reportFrame.onload = () => {
        hideLoading();
        console.log('Report loaded:', selectedReportPath);
      };
      reportFrame.onerror = () => {
        hideLoading();
        console.error('Failed to load report:', selectedReportPath);
        reportFrame.src = 'about:blank';
      };
      // 실제 경로 사용
      reportFrame.src = selectedReportPath;
      console.log('Loading report:', selectedReportPath);
    } else {
      console.warn(`No suitable report found for ${selectedType} on ${selectedDate}` + (latestTimeEntry?.time ? ` at time ${latestTimeEntry.time}` : ''));
      reportFrame.src = 'about:blank';
      hideLoading();
    }
  }

  // --- report_index.json 불러오기 ---
  async function fetchReportIndex(){
    try{
      const res = await fetch(REPORT_INDEX_PATH, { cache: 'no-cache' });
      if(!res.ok) throw new Error(`Failed to fetch report index: ${res.status}`);
      const data = await res.json();
      reportIndexData = data;
      console.log('Loaded report index:', Object.keys(reportIndexData));
      // 인덱스 로드 후 필요하면 UI 초기값 조정(예: 연도/월/일 옵션을 인덱스 기반으로 만들려면 여기서 처리)
      // 현재는 기본 populate 후 loadReport 호출
      loadReport();
    }catch(err){
      console.error('Error loading report index:', err);
      reportIndexData = {};
      // 인덱스 없을 때 기본 동작: 빈 프레임
      reportFrame.src = 'about:blank';
      hideLoading();
    }
  }

  // 이벤트 연결
  reportTypeSelect.addEventListener('change', ()=> {
    // 타입 변경 시, 필요하면 연/월/일을 인덱스 기반으로 조정 가능
    // 지금은 기존 날짜 유지하고 리포트 로드
    loadReport();
  });
  reportYearSelect.addEventListener('change', ()=> {
    populateDays(reportYearSelect.value, reportMonthSelect.value);
  });
  reportMonthSelect.addEventListener('change', ()=> {
    populateDays(reportYearSelect.value, reportMonthSelect.value);
  });
  reportDaySelect.addEventListener('change', loadReport);

  // 초기화
  populateYears(8);
  populateMonths();
  populateDays(reportYearSelect.value, reportMonthSelect.value);

  // 인덱스 fetch 및 초기 로드
  fetchReportIndex();

})();