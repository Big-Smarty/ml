(() => {
  const storage = {get(k,fallback){try{return JSON.parse(localStorage.getItem(k))??fallback;}catch{return fallback;}},set(k,v){try{localStorage.setItem(k,JSON.stringify(v));}catch{/* Reading works with storage disabled. */}}};
  const savedTheme=storage.get('fp-theme',null);
  document.body.classList.toggle('dark',savedTheme ? savedTheme==='dark' : matchMedia('(prefers-color-scheme: dark)').matches);
  document.querySelector('#theme-button').addEventListener('click',()=>{document.body.classList.toggle('dark');storage.set('fp-theme',document.body.classList.contains('dark')?'dark':'light');});
  const menu=document.querySelector('#menu-button'),sidebar=document.querySelector('#sidebar');
  menu.addEventListener('click',()=>{const open=sidebar.classList.toggle('open');menu.setAttribute('aria-expanded',String(open));});
  document.addEventListener('keydown',e=>{if(e.key==='Escape'){sidebar.classList.remove('open');menu.setAttribute('aria-expanded','false');hideTerm();}});
  const progress=storage.get('fp-progress',{}),chapter=document.body.dataset.chapter,checkbox=document.querySelector('#mark-complete');
  for(const link of document.querySelectorAll('[data-chapter].chapter-link'))link.classList.toggle('finished',!!progress[link.dataset.chapter]);
  if(checkbox){checkbox.checked=!!progress[chapter];checkbox.addEventListener('change',()=>{progress[chapter]=checkbox.checked;storage.set('fp-progress',progress);document.querySelector('.chapter-link.current')?.classList.toggle('finished',checkbox.checked);});}
  const current=document.querySelector('.chapter-link.current');if(current)current.scrollIntoView({block:'center'});
  for(const pre of document.querySelectorAll('pre')){const code=pre.querySelector('code');if(!code)continue;const button=document.createElement('button');button.className='copy-code';button.textContent='Copy';button.setAttribute('aria-label','Copy code');button.addEventListener('click',async()=>{try{await navigator.clipboard.writeText(code.textContent);button.textContent='Copied';}catch{button.textContent='Select to copy';}setTimeout(()=>button.textContent='Copy',1800);});pre.append(button);}
  const search=document.querySelector('#course-search'),results=document.querySelector('#search-results');let index=[];
  fetch('/assets/search.json').then(r=>{if(!r.ok)throw Error('search');return r.json();}).then(data=>{index=data;if(search.value)runSearch();}).catch(()=>{search.placeholder='Search unavailable; browse below';});
  function runSearch(){const query=search.value.trim().toLowerCase();results.replaceChildren();if(query.length<2)return;const words=query.split(/\s+/);const matches=index.map(item=>({...item,score:words.reduce((n,w)=>n+(item.title.toLowerCase().includes(w)?6:0),0)})).filter(item=>words.every(w=>(item.title+' '+item.text).toLowerCase().includes(w))).sort((a,b)=>b.score-a.score).slice(0,12);if(!matches.length){results.textContent='No matching chapters or terms.';return;}for(const item of matches){const a=document.createElement('a');a.href=item.url;a.textContent=item.title;const small=document.createElement('small');small.textContent=item.url.includes('/glossary/')?'Glossary explanation':item.url.includes('/chapters/')?'Chapter':'Course guide';a.append(small);results.append(a);}}
  search.addEventListener('input',runSearch);
  let terms={};fetch('/assets/terms.json').then(r=>r.json()).then(data=>terms=data).catch(()=>{/* Linked glossary pages still work. */});
  const preview=document.querySelector('#term-preview');let activeTerm=null,touchOpenedAnchor=null;
  function hideTerm(){preview.hidden=true;if(activeTerm)activeTerm.removeAttribute('aria-describedby');activeTerm=null;touchOpenedAnchor=null;}
  function showTerm(anchor){const term=terms[anchor.dataset.term];if(!term)return;if(activeTerm&&activeTerm!==anchor)activeTerm.removeAttribute('aria-describedby');activeTerm=anchor;preview.replaceChildren();const title=document.createElement('strong');title.textContent=term.name;preview.append(title,document.createTextNode(term.definition+' Open the link for a fuller explanation; on touch screens, tap again.'));preview.hidden=false;anchor.setAttribute('aria-describedby','term-preview');const r=anchor.getBoundingClientRect();const width=preview.offsetWidth;preview.style.left=Math.max(10,Math.min(innerWidth-width-10,r.left))+'px';const below=r.bottom+10;preview.style.top=(below+preview.offsetHeight<innerHeight?below:Math.max(10,r.top-preview.offsetHeight-10))+'px';}
  let touchPreviewAnchor=null;
  for(const anchor of document.querySelectorAll('a.term')){
    anchor.addEventListener('mouseenter',()=>showTerm(anchor));
    anchor.addEventListener('mouseleave',()=>{if(document.activeElement!==anchor)hideTerm();});
    anchor.addEventListener('focus',()=>showTerm(anchor));
    anchor.addEventListener('blur',hideTerm);
    // First touch opens the definition; a second touch follows the ordinary glossary link.
    anchor.addEventListener('pointerdown',e=>{touchPreviewAnchor=e.pointerType==='touch'&&terms[anchor.dataset.term]&&touchOpenedAnchor!==anchor?anchor:null;});
    anchor.addEventListener('click',e=>{if(touchPreviewAnchor===anchor){e.preventDefault();touchPreviewAnchor=null;anchor.focus();showTerm(anchor);touchOpenedAnchor=anchor;}});
  }
  addEventListener('scroll',()=>{if(activeTerm&&document.activeElement===activeTerm)showTerm(activeTerm);else hideTerm();},{passive:true});
})();
