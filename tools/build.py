#!/usr/bin/env python3
"""Build the offline course into dist without writing any learner/source files."""
from pathlib import Path
from html import escape
from html.parser import HTMLParser
import hashlib
import json
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'dist'
ASSET_VERSIONS = {}


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def plain(content):
    parser = Text()
    parser.feed(content)
    return ' '.join(' '.join(parser.parts).split())


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def asset(name):
    if name not in ASSET_VERSIONS:
        ASSET_VERSIONS[name] = hashlib.sha256((ROOT / 'site/assets' / name).read_bytes()).hexdigest()[:12]
    return f'/assets/{name}?v={ASSET_VERSIONS[name]}'


def json_html(value):
    return json.dumps(value).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')


def shell(title, body, chapters, active='', extra=''):
    body = re.sub(r'href="(/code/[^"#]+\.(?:rs|toml|wgsl|md|py|json))"', r'href="\1.html"', body)
    nav = ''
    for index, part in enumerate(dict.fromkeys(c['part'] for c in chapters), 1):
        group = [c for c in chapters if c['part'] == part]
        opened = ' open' if active in [c['id'] for c in group] or active == f'section-{index:02}' else ''
        links = ''.join(f'<a class="chapter-link" {"aria-current=page" if active == c["id"] else ""} href="/chapters/{c["id"]}.html"><span>{c["id"]}</span>{escape(c["title"])}</a>' for c in group)
        nav += f'<details class="nav-section"{opened}><summary><span>{index:02}</span>{escape(part)}</summary><a class="section-overview" href="/sections/{index:02}.html">Project overview</a>{links}</details>'
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Learn machine learning in Rust through explained examples, working experiments, and nine connected projects.">
<title>{escape(title)} · First Principles</title><link rel="icon" href="/assets/favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="{asset('style.css')}"><script defer src="{asset('prism.js')}"></script><script defer src="{asset('app.js')}"></script>{extra}</head>
<body data-chapter="{active}"><a class="skip-link" href="#main">Skip to content</a>
<header class="topbar"><button id="menu-button" class="js-only" aria-label="Open course navigation" aria-expanded="false" aria-controls="sidebar">☰</button><a class="brand" href="/"><span class="brand-mark">ƒ</span><span>First Principles<small>Machine learning in Rust</small></span></a><nav aria-label="Reference"><a href="/start.html">Setup</a><a href="/glossary.html">Glossary</a><button id="theme-button" class="js-only" aria-label="Switch color theme">◐</button></nav></header>
<aside class="sidebar" id="sidebar"><div class="search-box js-only"><label for="course-search">Find an idea</label><input type="search" id="course-search" placeholder="Try gradient or attention" autocomplete="off"><div id="search-status" role="status"></div><div id="search-results"></div></div><a class="map-link" href="/#course-map">The course map</a><nav aria-label="Course chapters">{nav}</nav><div class="sidebar-footer"><a href="/start.html">How to learn here</a><a href="/conventions.html">Code &amp; notation</a><a href="/git.html">Save your experiments</a></div></aside>
<main id="main" class="main">{body}<footer class="page-footer"><a href="/about.html">About the course</a><a href="/downloads/first-principles-source.zip" download>Download the course</a><span>Small experiments. Explanations you can check.</span></footer></main><div id="term-preview" role="tooltip" hidden></div></body></html>'''


def step_text(content, step_id):
    # Search chunks end at the next step, retaining nested sections and worked answers.
    match = re.search(r'<section\b[^>]*\bid="' + re.escape(step_id) + r'"[^>]*>', content)
    if not match:
        return plain(content)
    tail = content[match.end():]
    next_step = re.search(r'<section\b[^>]*class="[^"]*learning-step', tail)
    return plain(tail[:next_step.start()] if next_step else tail)


def course_sources():
    for folder in ('labs', 'projects', 'chapters', 'site', 'guidance', 'tools', 'datasets', '.helix', '.openai'):
        for path in sorted((ROOT / folder).rglob('*')):
            if path.is_file() and not any(part in ('target', 'downloads', '.git', '__pycache__', 'starter', 'generated') for part in path.relative_to(ROOT).parts) and path.suffix not in ('.checkpoint', '.pyc'):
                yield path
    for name in ('README.md', 'CHAPTERS.md', 'course.json', 'sections.json', 'justfile', 'rust-toolchain.toml', '.gitignore'):
        if (ROOT / name).is_file():
            yield ROOT / name


def main():
    chapters = json.loads((ROOT / 'course.json').read_text())
    sections = json.loads((ROOT / 'sections.json').read_text())
    # Only generated output is replaced. Neither a build nor a download edits lab sources.
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    shutil.copytree(ROOT / 'site/assets', OUT / 'assets')
    metadata = {c['id']: json.loads((ROOT / 'chapters' / c['id'] / 'meta.json').read_text()) for c in chapters}
    terms, search, catalog = {}, [], []
    for c in chapters:
        for slug, term in metadata[c['id']].get('terms', {}).items():
            terms.setdefault(slug, {**term, 'chapter': c['id']})
    for index, ch in enumerate(chapters):
        number = ch['id']
        folder = ROOT / 'chapters' / number
        meta = metadata[number]
        content = (folder / 'lesson.html').read_text()
        section = next(s for s in sections if number in s['chapters'])
        steps = meta.get('steps', [])
        sessions = meta.get('sessions', [])
        lab = meta.get('lab', {})
        data = dict(id=number, title=ch['title'], steps=steps, sessions=sessions, lab=lab)
        catalog.append(data)
        body = f'<article class="lesson"><header class="chapter-heading"><p class="breadcrumb"><a href="/sections/{section["id"]}.html">{escape(section["title"])}</a> <span>/ Chapter {number}</span></p><h1>{escape(ch["title"])}</h1>'
        body += f'<p class="chapter-summary">{escape(meta.get("outcomes", [ch["project"]])[0])}</p>'
        if sessions:
            body += f'<p class="session-count">{len(sessions)} sessions · {len(steps)} learning steps</p>'
        body += '</header>'
        if steps:
            choices = ''.join(f'<option value="{escape(s["id"])}">{i+1}. {escape(s["title"])}</option>' for i, s in enumerate(steps))
            body += f'<nav class="reader-controls js-only" aria-label="Chapter reading mode"><div><label for="step-select">Current step</label><select id="step-select">{choices}</select></div><button id="view-toggle" aria-pressed="false">Full chapter</button><p id="step-status" role="status"></p></nav><div class="session-goal js-only" id="session-goal"></div>'
        body += '<div id="chapter-content">' + content + '</div>'
        if steps:
            body += '<nav class="step-controls js-only" aria-label="Learning steps"><button id="step-back">← Back</button><span id="step-count"></span><button id="step-next">Next step →</button></nav>'
        if lab:
            manual = ''
            if lab.get('manual_checks'):
                manual = '<div class="manual-checks"><h3>Also review the implementation</h3><p>Numerical agreement alone cannot establish these parts of the goal:</p><ul>' + ''.join(f'<li>{escape(item)}</li>' for item in lab['manual_checks']) + '</ul></div>'
            body += f'<aside class="practice-card"><p class="eyebrow">Your Rust experiment</p><h2>Make the idea work</h2><p>{escape(lab["goal"])}</p><p><code>just lab {number}</code> runs your baseline. <code>just lab-check {number}</code> checks the learning goal.</p><p><a href="/code/{escape(lab["entry"])}">Open the learner source</a> · <a href="/code/labs/{escape(lab["package"])}/README.md">Lab instructions &amp; checkpoints</a></p>{manual}<details class="practice-record js-only"><summary>Record your practice</summary><p>These are your own records. The browser does not run or verify your Rust code.</p><label><input type="checkbox" data-practice="attempted"> I tried the implementation</label><label><input type="checkbox" data-practice="checked"> I met the numerical checks and any implementation-review criteria</label><label><input type="checkbox" data-practice="explained"> I explained the result and tried a new case</label><label for="practice-note">What did you notice?</label><textarea id="practice-note" rows="3" placeholder="A result, a mistake, or a question to return to"></textarea><p id="practice-status" role="status"></p></details></aside>'
        links = f'<a href="/chapters/{chapters[index-1]["id"]}.html"><small>Previous chapter</small>{escape(chapters[index-1]["title"])}</a>' if index else '<a href="/start.html"><small>Before you begin</small>Set up your workspace</a>'
        links += f'<a href="/chapters/{chapters[index+1]["id"]}.html"><small>Next chapter</small>{escape(chapters[index+1]["title"])}</a>' if index+1 < len(chapters) else '<a href="/"><small>Your next experiment</small>Return to the course map</a>'
        body += f'<nav class="chapter-pagination" aria-label="Adjacent chapters">{links}</nav></article><script type="application/json" id="chapter-data">{json_html(data)}</script>'
        extra = ''
        if (folder / 'demo.js').exists():
            shutil.copy2(folder / 'demo.js', OUT / 'assets' / f'ch{number}.js')
            version = hashlib.sha256((folder / 'demo.js').read_bytes()).hexdigest()[:12]
            extra = f'<script defer src="/assets/ch{number}.js?v={version}"></script>'
        write(OUT / 'chapters' / f'{number}.html', shell(ch['title'], body, chapters, number, extra))
        for step in steps or [dict(id='', title=ch['title'])]:
            url = f'/chapters/{number}.html' + (f'?step={step["id"]}#{step["id"]}' if step['id'] else '')
            search.append(dict(title=f'{number} · {step["title"]}', context=ch['title'], url=url, text=step_text(content, step['id'])))
    cards = ''
    for section in sections:
        number = section['id']
        group = [c for c in chapters if c['id'] in section['chapters']]
        cards += f'<a class="course-card" href="/sections/{number}.html"><span class="card-number">{number}</span><div><p class="eyebrow">Chapters {group[0]["id"]}–{group[-1]["id"]}</p><h3>{escape(section["title"])}</h3><p>{escape(section["project"])}</p><span class="card-action">Explore the project →</span></div></a>'
        pre = 'Everyday Rust: functions, structs, slices and basic error handling. We introduce the ML and mathematics as you need them.' if not section['prerequisites'] else 'Builds on ' + ', '.join(f'<a href="/sections/{p}.html">Section {int(p)}</a>' for p in section['prerequisites']) + '. Earlier ideas are recalled in each chapter.'
        body = f'<article class="lesson section-page"><p class="eyebrow">Section {number} · Chapters {group[0]["id"]}–{group[-1]["id"]}</p><h1>{escape(section["title"])}</h1><p class="lead">{escape(section["project"])}</p><div class="project-outcome"><h2>What you will build</h2><p>{escape(section["artifact"])}</p></div><h2>Before you start</h2><p>{pre}</p><h2>The project milestones</h2><ol class="milestones">' + ''.join(f'<li>{escape(m)}</li>' for m in section['milestones']) + '</ol><h2>Your chapters</h2><ol class="section-chapters">'
        body += ''.join(f'<li><a href="/chapters/{c["id"]}.html"><span>{c["id"]}</span><div><strong>{escape(c["title"])}</strong><p>{escape(metadata[c["id"]].get("lab", {}).get("goal", c["project"]))}</p></div><b aria-hidden="true">→</b></a></li>' for c in group)
        body += f'</ol><p><a class="button" href="/chapters/{group[0]["id"]}.html">Begin this section →</a></p><p>Local package: <code>labs/{section["package"]}</code>. <a href="/start.html#labs">How the labs work</a>.</p></article>'
        write(OUT / 'sections' / f'{number}.html', shell(section['title'], body, chapters, f'section-{number}'))
    home = f'''<div class="home"><section class="home-intro"><p class="eyebrow">A practical course for programmers new to ML</p><h1>Understand the idea.<br>Then make it work.</h1><p class="lead">Learn machine learning through small Rust experiments. Follow the numbers, change the algorithm, and find out why the result changed.</p><div class="start-card"><div><p class="eyebrow" id="continue-label">Your first session</p><h2 id="continue-title">One neuron learns from data</h2><p id="continue-description">Start with a sensor that needs calibrating. No calculus required.</p></div><a class="button" id="continue-link" href="/chapters/01.html">Start learning →</a></div><p class="setup-link">New workspace? <a href="/start.html">Set up Rust and the course files</a></p></section><section class="learning-rhythm"><div><span>01</span><h2>Follow a worked example</h2><p>Meet each idea with actual numbers and explanations.</p></div><div><span>02</span><h2>Run, change, compare</h2><p>Start from working code. Implement the algorithm that matters.</p></div><div><span>03</span><h2>Try an unfamiliar case</h2><p>Check what you learned by predicting and explaining a new result.</p></div></section><section id="course-map"><p class="eyebrow">56 chapters · Nine connected projects</p><h2>Your path through machine learning</h2><p class="map-intro">Work in sessions of roughly 30–45 minutes. Each chapter has short steps and a full reading view; take as much time as an idea needs.</p><div class="course-grid">{cards}</div></section></div>'''
    write(OUT / 'index.html', shell('Learn machine learning through experiments', home, chapters))
    for slug, term in sorted(terms.items()):
        body = f'<article class="lesson glossary-entry"><p class="eyebrow"><a href="/glossary.html">Glossary</a></p><h1>{escape(term["name"])}</h1><p class="lead">{escape(term["definition"])}</p><p>{escape(term.get("explanation", term["definition"]))}</p><p><a class="button" href="/chapters/{term["chapter"]}.html">Meet the idea in Chapter {term["chapter"]} →</a></p></article>'
        write(OUT / 'glossary' / f'{slug}.html', shell(term['name'], body, chapters))
        search.append(dict(title=term['name'], context='Glossary', url=f'/glossary/{slug}.html', text=term['definition'] + ' ' + term.get('explanation', '')))
    glossary = '<article class="lesson"><p class="eyebrow">A reference to return to</p><h1>Glossary</h1><p class="lead">A short definition, a fuller explanation, and a path back to the lesson.</p><dl class="glossary-list">' + ''.join(f'<dt><a href="/glossary/{slug}.html">{escape(term["name"])}</a></dt><dd>{escape(term["definition"])}</dd>' for slug, term in sorted(terms.items(), key=lambda x: x[1]['name'].lower())) + '</dl></article>'
    write(OUT / 'glossary.html', shell('Glossary', glossary, chapters))
    for name, title in (('start', 'Set up and start learning'), ('about', 'About the course'), ('git', 'Save your experiments'), ('conventions', 'Code and notation')):
        content = (ROOT / 'site' / f'{name}.html').read_text()
        write(OUT / f'{name}.html', shell(title, content, chapters))
        search.append(dict(title=title, context='Course guide', url=f'/{name}.html', text=plain(content)))
    for name, data in [('terms', terms), ('search', search), ('course', catalog)]:
        write(OUT / 'assets' / f'{name}.json', json.dumps(data))
    sources = list(course_sources())
    for path in sources:
        relative = path.relative_to(ROOT)
        dest = OUT / 'code' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        if path.suffix in ('.rs', '.toml', '.wgsl', '.md', '.py', '.json'):
            language = {'.rs': 'rust', '.wgsl': 'wgsl', '.toml': 'toml', '.py': 'python', '.json': 'json'}.get(path.suffix, 'none')
            body = f'<article class="lesson source-page"><p class="eyebrow">Course source</p><h1>{escape(path.name)}</h1><p><code>{escape(str(relative))}</code></p><p><a href="/code/{escape(str(relative))}" download>Download this file</a> · <a href="/downloads/first-principles-source.zip" download>Download the complete course</a></p><pre><code class="language-{language}">{escape(path.read_text())}</code></pre></article>'
            # Preserve actual download targets rather than rewriting them to source viewers.
            rendered = shell(str(relative), body, chapters)
            rendered = rendered.replace(f'href="/code/{escape(str(relative))}.html" download', f'href="/code/{escape(str(relative))}" download')
            write(Path(str(dest) + '.html'), rendered)
    (OUT / 'downloads').mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT / 'downloads/first-principles-source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, 'first-principles/' + str(path.relative_to(ROOT)))
    redesigned = sum(m.get('redesign_version') == 2 for m in metadata.values())
    print(f'Built {len(chapters)} chapters ({redesigned} redesigned), {len(sections)} section pages, {len(terms)} terms into {OUT}')


if __name__ == '__main__':
    main()
