(function(){
  const root = document.documentElement;
  const app = document.getElementById('app');
  const themeToggle = document.getElementById('themeToggle');
  const themeText = document.getElementById('themeText');
  const yearSelect = document.getElementById('reportYear');
  const monthSelect = document.getElementById('reportMonth');
  const daySelect = document.getElementById('reportDay');
  const typeSelect = document.getElementById('reportType');
  const iframe = document.getElementById('reportFrame');
  const loading = document.getElementById('loadingIndicator');

  // 초기 테마(로컬스토리지)
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

  themeToggle.addEventListener('click', ()=> {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    setTheme(isDark ? 'light' : 'dark');
  });
  themeToggle.addEventListener('keydown', (e)=>{
    if(e.key === 'Enter' || e.key === ' '){ e.preventDefault(); themeToggle.click(); }
  });

  // 연/월/일 옵션 채우기
  function populateYears(range = 10){
    const cur = new Date().getFullYear();
    yearSelect.innerHTML = '';
    for(let y = cur; y >= cur - range; y--){
      const opt = document.createElement('option');
      opt.value = y;
      opt.textContent = y + '년';
      yearSelect.appendChild(opt);
    }
    yearSelect.value = cur;
  }
  function populateMonths(){
    monthSelect.innerHTML = '';
    for(let m=1;m<=12;m++){
      const opt = document.createElement('option');
      opt.value = m;
      opt.textContent = m + '월';
      monthSelect.appendChild(opt);
    }
    monthSelect.value = (new Date().getMonth()+1);
  }
  function populateDays(y, m){
    const daysInMonth = new Date(y, m, 0).getDate();
    daySelect.innerHTML = '';
    for(let d=1; d<=daysInMonth; d++){
      const opt = document.createElement('option');
      opt.value = d;
      opt.textContent = d + '일';
      daySelect.appendChild(opt);
    }
    const today = new Date();
    if(y == today.getFullYear() && m == (today.getMonth()+1)){
      daySelect.value = today.getDate();
    } else {
      daySelect.value = 1;
    }
  }

  // 초기화
  populateYears(8);
  populateMonths();
  populateDays(yearSelect.value, monthSelect.value);

  // 바뀔 때마다 리포트 로딩 시뮬레이션 (실제 URL 로드하려면 iframe.src 설정)
  function loadReport(){
    const y = yearSelect.value;
    const m = monthSelect.value.toString().padStart(2,'0');
    const d = daySelect.value.toString().padStart(2,'0');
    const type = typeSelect.value;

    loading.style.display = 'flex';
    iframe.src = 'about:blank';

    setTimeout(()=> {
      const html = `
        <html><head><meta charset="utf-8"><style>
          body{font-family: -apple-system,BlinkMacSystemFont,"Pretendard",Segoe UI,Roboto; padding:20px; color:#222}
          h2{margin-top:0}
          .meta{color:#6b7280; font-size:13px}
        </style></head><body>
          <h2>리포트: ${type} / ${y}-${m}-${d}</h2>
          <div class="meta">샘플 리포트 내용입니다. 실제 리포트 URL을 iframe.src에 넣어주세요.</div>
          <p>데이터 미리보기...</p>
        </body></html>
      `;
      try{
        const doc = iframe.contentWindow.document;
        doc.open();
        doc.write(html);
        doc.close();
      }catch(e){
        iframe.src = 'data:text/html;charset=utf-8,' + encodeURIComponent(html);
      }
      loading.style.display = 'none';
    }, 700);
  }

  yearSelect.addEventListener('change', ()=> populateDays(yearSelect.value, monthSelect.value));
  monthSelect.addEventListener('change', ()=> populateDays(yearSelect.value, monthSelect.value));
  [yearSelect, monthSelect, daySelect, typeSelect].forEach(el => {
    el.addEventListener('change', loadReport);
  });

  loadReport();
})();