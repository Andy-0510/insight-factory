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
  iframe.setAttribute('sandbox', 'allow-scripts allow-forms allow-popups');
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

      // 테마에 따른 텍스트 색상 조정
      const injectedCss = `
        body { 
          font-size: \${getComputedStyle(document.documentElement).getPropertyValue('--body-font-size') || '16px'} !important;
          text-align: center !important;
          line-height:1.6 !important;
          color: \${document.documentElement.getAttribute('data-theme') === 'dark' ? '#fff' : '#000'} !important; /* 다크모드일 때 텍스트 흰색 */
        }
        h1,h2,h3 { text-align:center !important; color: \${document.documentElement.getAttribute('data-theme') === 'dark' ? '#fff' : '#000'} !important; }
        table, pre, code { 
          font-size: 15px !important;
          color: \${document.documentElement.getAttribute('data-theme') === 'dark' ? '#fff' : '#000'} !important;
          background-color: \${document.documentElement.getAttribute('data-theme') === 'dark' ? '#1a1a1a' : '#fff'} !important;
        }
      `;
      
      const docHead = doc.head || doc.getElementsByTagName('head')[0] || doc.documentElement;
      let s = doc.getElementById('injected-size-style');
      if(!s){
        s = doc.createElement('style'); s.id = 'injected-size-style'; s.innerHTML = injectedCss; docHead.appendChild(s);
      } else {
        s.innerHTML = injectedCss;
      }
    } catch(e) {
      // cross-origin이면 접근 불가
      console.warn('자동 높이 실패(크로스오리진):', e);
      iframe.style.height = '640px';
      iframe.style.overflow = 'auto';
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
