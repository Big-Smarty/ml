#!/usr/bin/env python3
"""Assemble original HTML chapters into an offline static course; Python stdlib only."""
from pathlib import Path
from html import escape
from html.parser import HTMLParser
import hashlib, json, re, shutil
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'site/generated'
ASSET_VERSIONS = {}
def asset(name):
    if name not in ASSET_VERSIONS:
        ASSET_VERSIONS[name] = hashlib.sha256((ROOT/'site/assets'/name).read_bytes()).hexdigest()[:12]
    return f'/assets/{name}?v={ASSET_VERSIONS[name]}'
class Text(HTMLParser):
    def __init__(self): super().__init__(); self.parts=[]
    def handle_data(self,data): self.parts.append(data)
def plain(html):
    p=Text(); p.feed(html); return ' '.join(' '.join(p.parts).split())
def write(path,text):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(text)
def shell(title, body, chapters, active='', toc='', extra=''):
    # Browsers may download raw source instead of showing it. Keep source reading HTML too.
    body=re.sub(r'href="(/code/[^"#]+\.(?:rs|toml|wgsl|md|py))"',r'href="\1.html"',body)
    nav=''; last=None
    for ch in chapters:
        if ch['part']!=last:
            nav+=f'<div class="nav-part">{escape(ch["part"])}</div>'; last=ch['part']
        nav+=f'<a class="chapter-link {"current" if active==ch["id"] else ""}" href="/chapters/{ch["id"]}.html" data-chapter="{ch["id"]}"><span>{ch["id"]}</span>{escape(ch["title"])}</a>'
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="A hands-on Rust course: one neuron to language models, with original lectures, exercises and reference implementations."><title>{escape(title)} · First Principles</title><link rel="icon" href="/assets/favicon.svg" type="image/svg+xml"><link rel="stylesheet" href="{asset('style.css')}"><script src="{asset('prism.js')}" defer></script><script src="{asset('app.js')}" defer></script>{extra}</head><body data-chapter="{active}"><a class="skip-link" href="#main">Skip to content</a><header class="topbar"><button id="menu-button" class="icon-button" aria-label="Toggle chapter navigation" aria-expanded="false">☰</button><a class="brand" href="/"><span class="brand-mark">ƒ</span> First Principles <span class="brand-sub">Machine learning, in Rust</span></a><div class="top-actions"><a href="/start.html">Start here</a><a href="/git.html">Git guide</a><a href="/conventions.html">Code guide</a><a href="/glossary.html">Glossary</a><button id="theme-button" class="icon-button" aria-label="Switch color theme">◐</button></div></header><aside class="sidebar" id="sidebar"><label class="search-label" for="course-search">Find a chapter or concept</label><input type="search" id="course-search" placeholder="Search the course…" autocomplete="off"><div id="search-results" aria-live="polite"></div><nav class="mobile-guides" aria-label="Course guides"><a href="/start.html">Start here</a><a href="/git.html">Git guide</a><a href="/conventions.html">Code guide</a><a href="/glossary.html">Glossary</a></nav><nav aria-label="Course chapters">{nav}</nav><div class="sidebar-footer">Learn the idea.<br>Build it. Check it. Make it yours.</div></aside><main id="main" class="main">{body}<footer class="page-footer">First Principles · Original lessons, executable ideas.<br><a href="/about.html">About this course</a> · <a href="/git.html">Track your learning with Git</a> · <a href="/start.html#offline">Using the course offline</a></footer></main>{toc}<div id="term-preview" role="tooltip" hidden></div></body></html>'''
def main():
    chapters=json.loads((ROOT/'course.json').read_text()); OUT.mkdir(parents=True,exist_ok=True)
    shutil.copytree(ROOT/'site/assets',OUT/'assets',dirs_exist_ok=True)
    terms={}; metadata={}; search=[]; completed=[]
    for ch in chapters:
        p=ROOT/'chapters'/ch['id']; meta=json.loads((p/'meta.json').read_text()) if (p/'meta.json').exists() else {}
        metadata[ch['id']]=meta
        for slug,term in meta.get('terms',{}).items():
            if slug not in terms: terms[slug]={**term,'chapter':ch['id']}
    for page in (OUT/'glossary').glob('*.html'):
        if page.stem not in terms:
            page.unlink()
    for idx,ch in enumerate(chapters):
        p=ROOT/'chapters'/ch['id']; meta=metadata[ch['id']]
        if not (p/'lesson.html').exists():
            content='<p class="lead">This chapter is being authored. It is not yet part of a completed course release.</p>'
        else: content=(p/'lesson.html').read_text(); completed.append(ch['id'])
        hs=re.findall(r'<h2\b[^>]*\bid="([^"]+)"[^>]*>(.*?)</h2>',content,re.S)
        # Most authors put anchors on the surrounding section.
        hs+=re.findall(r'<section\b[^>]*\bid="([^"]+)"[^>]*>\s*<h2[^>]*>(.*?)</h2>',content,re.S)
        seen=set(); hs=[(k,v) for k,v in hs if not (k in seen or seen.add(k))]
        toc='<aside class="toc"><span class="eyebrow">On this page</span>'+''.join(f'<a href="#{escape(k)}">{escape(plain(v))}</a>' for k,v in hs)+'</aside>'
        prev=chapters[idx-1] if idx else None; nxt=chapters[idx+1] if idx+1<len(chapters) else None
        links=(f'<a href="/chapters/{prev["id"]}.html"><small>← Previous</small>{escape(prev["title"])}</a>' if prev else '<a href="/start.html"><small>← Start here</small>Prepare your workspace</a>')
        links+=(f'<a href="/chapters/{nxt["id"]}.html"><small>Next →</small>{escape(nxt["title"])}</a>' if nxt else '<a href="/"><small>Keep exploring →</small>Return to the course map</a>')
        outcomes=''.join('<li>'+escape(x)+'</li>' for x in meta.get('outcomes',[]))
        body=f'<article class="lesson"><div class="breadcrumb">{escape(ch["part"])} <span>/</span> Chapter {ch["id"]}</div><h1>{escape(ch["title"])}</h1><div class="lesson-meta"><span>Rust · Learn by building</span><a href="/code/projects/ch{ch["id"]}/src/main.rs">Reference code ↗</a></div>'
        prerequisites=meta.get('prerequisites',ch['prerequisites'])
        if prerequisites:
            body+='<p class="prerequisites">Builds on '+', '.join(f'<a href="/chapters/{str(pre).zfill(2)}.html">Chapter {int(pre)}</a>' for pre in prerequisites)+'.</p>'
        else:
            body+='<p class="prerequisites">Start here. Programming basics are enough; no ML or calculus required.</p>'
        if outcomes: body+='<details class="objectives"><summary>What you’ll be able to do</summary><ul>'+outcomes+'</ul></details>'
        body+=content+f'<div class="completion"><label><input type="checkbox" id="mark-complete"> I can explain and apply this chapter</label><p>Reading progress stays in this browser. Exercise checks stay in Rustlings.</p></div><nav class="chapter-pagination" aria-label="Adjacent chapters">{links}</nav></article>'
        extra=''
        if (p/'demo.js').exists():
            shutil.copy2(p/'demo.js',OUT/'assets'/f'ch{ch["id"]}.js')
            version=hashlib.sha256((p/'demo.js').read_bytes()).hexdigest()[:12]
            extra=f'<script defer src="/assets/ch{ch["id"]}.js?v={version}"></script>'
        write(OUT/'chapters'/f'{ch["id"]}.html',shell(ch['title'],body,chapters,ch['id'],toc,extra))
        search.append(dict(title=f'{ch["id"]} · {ch["title"]}',url=f'/chapters/{ch["id"]}.html',text=plain(content)))
    for slug,term in sorted(terms.items()):
        body=f'<article class="lesson glossary-entry"><div class="breadcrumb">Field notes / Glossary</div><h1>{escape(term["name"])}</h1><p class="lead">{escape(term["definition"])}</p><p>{escape(term.get("explanation",term["definition"]))}</p><a class="button" href="/chapters/{term["chapter"]}.html">See it in Chapter {term["chapter"]} →</a><p><a href="/glossary.html">Browse all terms</a></p></article>'
        write(OUT/'glossary'/f'{slug}.html',shell(term['name'],body,chapters)); search.append(dict(title=term['name'],url=f'/glossary/{slug}.html',text=term['definition']+' '+term.get('explanation','')))
    glossary='<article class="lesson"><div class="breadcrumb">A reference to return to</div><h1>Make the vocabulary familiar.</h1><p class="lead">Short definitions with a path back to the idea in context. Search from the sidebar, or explore the terms below.</p><dl class="glossary-list">'+''.join(f'<dt><a href="/glossary/{k}.html">{escape(v["name"])}</a></dt><dd>{escape(v["definition"])}</dd>' for k,v in sorted(terms.items(),key=lambda x:x[1]['name'].lower()))+'</dl></article>'
    write(OUT/'glossary.html',shell('Glossary',glossary,chapters))
    parts=[]
    for ch in chapters:
        if ch['part'] not in parts: parts.append(ch['part'])
    cards=''
    for i,part in enumerate(parts):
        group=[c for c in chapters if c['part']==part]
        cards+=f'<section class="part-card"><div class="part-number">{i+1:02}</div><div><span class="eyebrow">Chapters {group[0]["id"]}–{group[-1]["id"]}</span><h3>{escape(part)}</h3><ul>'+''.join(f'<li><a href="/chapters/{c["id"]}.html"><span>{c["id"]}</span>{escape(c["title"])}</a></li>' for c in group)+'</ul></div></section>'
    home='''<div class="home"><section class="hero"><div class="eyebrow">A practical course · Rust from the ground up</div><h1>One neuron.<br>A whole new way<br>to <em>think.</em></h1><p class="lead">Build machine learning from first principles. Follow the numbers, write the code, and grow a single neuron into your own language model.</p><div class="hero-actions"><a class="button" href="/chapters/01.html">Start with one neuron <span>→</span></a><a href="/start.html">Set up your workspace ↗</a></div><div class="hero-formula" aria-label="prediction equals weight times input plus bias"><span>input</span><b>x</b><i>×</i><b>w</b><i>+</i><b>b</b><i>→</i><b class="output">ŷ</b><span>prediction</span></div></section><section class="how-to"><div><span class="eyebrow">01 / Understand</span><h2>Follow every step.</h2><p>Plain explanations, worked arithmetic and interactive diagrams. New terms are always one click away.</p></div><div><span class="eyebrow">02 / Build</span><h2>Make the idea run.</h2><p>Small Rustlings exercises lead into complete Rust projects. Write the maths yourself before reaching for a framework.</p></div><div><span class="eyebrow">03 / Investigate</span><h2>Know why it works.</h2><p>Check gradients, break assumptions, measure speed. Carry the same habits from your first model to the GPU.</p></div></section><section class="course-map" id="course-map"><div class="section-heading"><div><span class="eyebrow">Your path through the course</span><h2>Small steps. Substantial projects.</h2></div><p>56 chapters · Work at your own pace</p></div>'''+cards+'''</section><section class="end-note"><h2>Start small. Stay curious.</h2><p>You don’t need to know calculus to begin. You do need to pause, experiment, and explain the results to yourself. We’ll build the mathematics as we go.</p><a href="/start.html">How to use this course →</a></section></div>'''
    write(OUT/'index.html',shell('Machine learning from scratch',home,chapters))
    for name,title in (('start','Start here'),('about','About the course'),('git','Track your learning with Git'),('conventions','Code and terminology')):
        content=(ROOT/'site'/f'{name}.html').read_text()
        write(OUT/f'{name}.html',shell(title,content,chapters))
        search.append(dict(title=title,url=f'/{name}.html',text=plain(content)))
    write(OUT/'assets/terms.json',json.dumps(terms)); write(OUT/'assets/search.json',json.dumps(search))
    index=['# Chapters\n','This map is generated from course.json and chapter metadata. Lectures are HTML, not Markdown.\n']
    for ch in chapters:
        m=metadata[ch['id']]; pre=m.get('prerequisites',ch['prerequisites'])
        index += [f'## {ch["id"]}. {ch["title"]}\n',f'Part: {ch["part"]}. Prerequisites: {", ".join(pre) or "Programming basics only"}.\n',f'Project: {ch["project"]}. Dataset: {ch["dataset"]}.\n','Lessons: '+ '; '.join(m.get('lessons',ch['topics']))+'.\n','Introduced concepts: '+ '; '.join(ch['topics'])+'.\n','Outcomes:\n']
        index += ['- '+x for x in m.get('outcomes',[f'Explain and implement {t.lower()}' for t in ch['topics']])]
        index += ['\nCompletion checks:\n']+['- '+x for x in m.get('checks',['Run the reference tests, finish the guided exercise, explain the worked example, and solve the independent variation.'])]+['\n']
    write(ROOT/'CHAPTERS.md','\n'.join(index))
    # Rustlings community pack, based on its documented format.
    info=['format_version = 1','welcome_message = "First Principles: read the HTML lesson, then make the idea run."','final_message = "Keep building: complete the chapter projects and transfer exercises."']
    bins=['bin = [']; cargo=['[package]','name = "first_principles_exercises"','version = "0.1.0"','edition = "2021"','publish = false','autobins = false','[workspace]']
    for path in sorted((ROOT/'exercises/cpu/exercises').glob('*.rs')):
        chapter=path.stem[2:4]
        info += ['','[[exercises]]',f'name = "{path.stem}"','test = true',f'hint = "Review Chapter {chapter} in the local website. Reveal its progressive hints, then compare with the solution only after trying."']
        bins += [f'  {{ name = "{path.stem}", path = "exercises/{path.name}" }},',f'  {{ name = "{path.stem}_sol", path = "solutions/{path.name}" }},']
    write(ROOT/'exercises/cpu/info.toml','\n'.join(info)+'\n');write(ROOT/'exercises/cpu/Cargo.toml','\n'.join(bins+["]",""]+cargo)+'\n')
    # Downloadable sources; omit all build artifacts and caches.
    for folder in ('projects','exercises','datasets','guidance','tools'):
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and not any(x in path.parts for x in ('target','downloads','.git','__pycache__')) and path.suffix not in ('.checkpoint',):
                dest=OUT/'code'/path.relative_to(ROOT); dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,dest)
                if path.suffix in ('.rs','.toml','.wgsl','.md','.py'):
                    relative=str(path.relative_to(ROOT))
                    language={'.rs':'rust','.wgsl':'wgsl','.toml':'toml','.py':'python'}.get(path.suffix,'none')
                    source_body=f'<article class="lesson source-page"><div class="breadcrumb">Course files</div><h1>{escape(path.name)}</h1><p>Local file: <code>{escape(relative)}</code></p><p>Open this file in your editor to work on it. This page displays the bundled course source.</p><pre><code class="language-{language}">{escape(path.read_text())}</code></pre></article>'
                    write(Path(str(dest)+'.html'),shell(relative,source_body,chapters))
    print(f'Built {len(completed)}/{len(chapters)} authored chapters, {len(terms)} terms into {OUT}')
if __name__=='__main__': main()
