/* The full course is readable without JS. These controls add focus and local records. */
(() => {
  'use strict';
  document.documentElement.classList.add('js');
  const key = 'first-principles-v2';
  let saved = {};
  let storageAvailable = true;
  try { saved = JSON.parse(localStorage.getItem(key) || '{}'); } catch (_) { storageAvailable = false; }
  if (!saved || typeof saved !== 'object' || Array.isArray(saved)) saved = {};
  const persist = () => {
    try { localStorage.setItem(key, JSON.stringify(saved)); }
    catch (_) { storageAvailable = false; }
  };
  const theme = document.querySelector('#theme-button');
  if (saved.theme === 'dark' || saved.theme === 'light') document.documentElement.dataset.theme = saved.theme;
  theme?.addEventListener('click', () => {
    const dark = document.documentElement.dataset.theme === 'dark' || (!document.documentElement.dataset.theme && matchMedia('(prefers-color-scheme: dark)').matches);
    saved.theme = dark ? 'light' : 'dark';
    document.documentElement.dataset.theme = saved.theme;
    persist();
  });
  const menu = document.querySelector('#menu-button');
  const sidebar = document.querySelector('#sidebar');
  const mobile = matchMedia('(max-width: 820px)');
  const syncMenu = () => {
    const open = document.body.classList.contains('nav-open');
    if (sidebar) sidebar.inert = mobile.matches && !open;
    document.querySelector('#main').inert = mobile.matches && open;
  };
  const closeMenu = () => { document.body.classList.remove('nav-open'); menu?.setAttribute('aria-expanded', 'false'); syncMenu(); };
  mobile.addEventListener('change', syncMenu);
  syncMenu();
  menu?.addEventListener('click', () => {
    const open = document.body.classList.toggle('nav-open');
    menu.setAttribute('aria-expanded', String(open));
    syncMenu();
    if (open) document.querySelector('#course-search')?.focus();
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && document.body.classList.contains('nav-open')) { closeMenu(); menu.focus(); }
  });
  document.querySelector('#main')?.addEventListener('click', closeMenu);

  const chapterNode = document.querySelector('#chapter-data');
  if (chapterNode) {
    const chapter = JSON.parse(chapterNode.textContent);
    const nodes = [...document.querySelectorAll('#chapter-content > .learning-step')];
    const steps = chapter.steps || [];
    let index = 0;
    let full = false;
    const params = new URLSearchParams(location.search);
    const stored = saved.reading?.[chapter.id];
    const findStep = value => steps.findIndex(s => s.id === value);
    const targetIndex = () => {
      let hash = '';
      try { hash = decodeURIComponent(location.hash.slice(1)); } catch (_) { /* Invalid fragment: use the requested step. */ }
      const target = document.getElementById(hash);
      const parent = target?.closest('.learning-step');
      return findStep(parent?.id || params.get('step') || stored?.step);
    };
    index = Math.max(0, targetIndex());
    full = params.get('view') === 'full';
    const select = document.querySelector('#step-select');
    const toggle = document.querySelector('#view-toggle');
    const status = document.querySelector('#step-status');
    const back = document.querySelector('#step-back');
    const next = document.querySelector('#step-next');
    const draw = (updateUrl = true, focus = false) => {
      if (!steps.length || nodes.length !== steps.length) return;
      nodes.forEach((node, i) => { node.hidden = !full && i !== index; });
      select.value = steps[index].id;
      toggle.textContent = full ? 'Focused steps' : 'Full chapter';
      toggle.setAttribute('aria-pressed', String(full));
      const step = steps[index];
      const session = chapter.sessions.find(s => String(s.id) === String(step.session));
      status.textContent = full ? `${steps.length} steps · full chapter` : `Session ${step.session}${session ? `: ${session.title}` : ''} · about ${step.minutes} minutes for this step`;
      document.querySelector('#step-count').textContent = `${index + 1} / ${steps.length}`;
      back.disabled = index === 0;
      next.disabled = index === steps.length - 1;
      document.querySelector('#session-goal').textContent = session ? `This session: ${session.goal}` : '';
      document.querySelector('.step-controls').hidden = full;
      if (updateUrl) {
        const url = new URL(location.href);
        url.searchParams.set('step', step.id);
        if (full) url.searchParams.set('view', 'full'); else url.searchParams.delete('view');
        url.hash = '';
        history.replaceState(null, '', url);
      }
      saved.reading ||= {};
      saved.reading[chapter.id] = { step: step.id, at: Date.now() };
      saved.last = { chapter: chapter.id, step: step.id, title: chapter.title, stepTitle: step.title };
      persist();
      if (focus) {
        const heading = nodes[index].querySelector('h2');
        heading?.setAttribute('tabindex', '-1');
        heading?.focus({ preventScroll: true });
        nodes[index].scrollIntoView({ block: 'start', behavior: 'auto' });
      }
    };
    if (steps.length && nodes.length === steps.length) {
      draw(false);
      select.addEventListener('change', () => { index = findStep(select.value); draw(true, true); });
      toggle.addEventListener('click', () => { full = !full; draw(); });
      back.addEventListener('click', () => { index--; draw(true, true); });
      next.addEventListener('click', () => { index++; draw(true, true); });
      window.addEventListener('hashchange', () => {
        const target = targetIndex();
        if (target >= 0) { index = target; draw(false); document.getElementById(decodeURIComponent(location.hash.slice(1)))?.scrollIntoView(); }
      });
    }
    saved.practice ||= {};
    const record = saved.practice[chapter.id] || {};
    const practiceStatus = document.querySelector('#practice-status');
    const recordStatus = () => {
      if (practiceStatus) practiceStatus.textContent = storageAvailable ? 'Saved in this browser. Rust checks are self-reported.' : 'Browser storage is unavailable. Your notes will not survive closing this page.';
    };
    const saveRecord = () => { saved.practice[chapter.id] = { ...record, at: Date.now() }; persist(); recordStatus(); };
    document.querySelectorAll('[data-practice]').forEach(input => {
      input.checked = record[input.dataset.practice] === true;
      input.addEventListener('change', () => { record[input.dataset.practice] = input.checked; saveRecord(); });
    });
    const note = document.querySelector('#practice-note');
    if (note) { note.value = typeof record.note === 'string' ? record.note : ''; note.addEventListener('input', () => { record.note = note.value; saveRecord(); }); recordStatus(); }
  }
  if (saved.last && /^\d{2}$/.test(saved.last.chapter) && typeof saved.last.step === 'string' && document.querySelector('#continue-link')) {
    document.querySelector('#continue-label').textContent = `Continue · Chapter ${saved.last.chapter}`;
    document.querySelector('#continue-title').textContent = saved.last.title;
    document.querySelector('#continue-description').textContent = saved.last.stepTitle;
    const link = document.querySelector('#continue-link');
    link.href = `/chapters/${saved.last.chapter}.html?step=${encodeURIComponent(saved.last.step)}`;
    link.textContent = 'Continue learning →';
  }

  const searchInput = document.querySelector('#course-search');
  const results = document.querySelector('#search-results');
  let searchData;
  let queryNumber = 0;
  searchInput?.addEventListener('input', async () => {
    const current = ++queryNumber;
    const query = searchInput.value.trim().toLowerCase();
    results.replaceChildren();
    document.querySelector('#search-status').textContent = '';
    if (query.length < 2) return;
    try {
      if (!searchData) { const response = await fetch('/assets/search.json'); if (!response.ok) throw new Error('search'); searchData = await response.json(); }
      if (current !== queryNumber) return;
      const words = query.split(/\s+/);
      const found = searchData.map(item => {
        const title = item.title.toLowerCase();
        const haystack = `${title} ${item.text}`.toLowerCase();
        return { item, score: words.every(word => haystack.includes(word)) ? words.reduce((score, word) => score + (title.includes(word) ? 5 : 1), 0) : 0 };
      }).filter(row => row.score).sort((a, b) => b.score - a.score).slice(0, 10);
      document.querySelector('#search-status').textContent = found.length ? `${found.length} matching steps or definitions` : 'No matches. Try a shorter term.';
      found.forEach(({ item }) => {
        const link = document.createElement('a'); link.href = item.url;
        const title = document.createElement('strong'); title.textContent = item.title;
        const context = document.createElement('small'); context.textContent = item.context;
        link.append(title, context); results.append(link);
      });
    } catch (_) { document.querySelector('#search-status').textContent = 'Search could not load. Use the course map or glossary.'; }
  });
  // Focus/hover definitions supplement ordinary glossary links, which always remain usable.
  const preview = document.querySelector('#term-preview');
  let terms;
  document.querySelectorAll('a.term[data-term]').forEach(link => {
    const show = async () => {
      try {
        terms ||= await (await fetch('/assets/terms.json')).json();
        if (!link.matches(':hover, :focus')) return;
        const term = terms[link.dataset.term]; if (!term) return;
        preview.textContent = `${term.name}: ${term.definition}`;
        preview.hidden = false; link.setAttribute('aria-describedby', 'term-preview');
        const rect = link.getBoundingClientRect();
        preview.style.left = `${Math.max(12, Math.min(rect.left, innerWidth - 332))}px`;
        preview.style.top = `${Math.max(12, Math.min(rect.bottom + 8, innerHeight - preview.offsetHeight - 12))}px`;
      } catch (_) { /* The glossary link remains available. */ }
    };
    const hide = () => { preview.hidden = true; link.removeAttribute('aria-describedby'); };
    link.addEventListener('mouseenter', show); link.addEventListener('focus', show);
    link.addEventListener('mouseleave', hide); link.addEventListener('blur', hide);
    link.addEventListener('keydown', event => { if (event.key === 'Escape') hide(); });
  });
})();
