#!/usr/bin/env python3
"""Content/link audit and scoped Rust reference/exercise gates. No external Python packages."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import argparse, json, re, subprocess, tempfile, time
from build import ROOT, OUT, plain
class Page(HTMLParser):
    def __init__(self): super().__init__(); self.ids=set();self.links=[];self.h2=0;self.details=0;self.errors=[]
    def handle_starttag(self,tag,attrs):
        d=dict(attrs)
        if 'id' in d:
            if d['id'] in self.ids:self.errors.append('duplicate id '+d['id'])
            self.ids.add(d['id'])
        if tag=='h2':self.h2+=1
        if tag=='details':self.details+=1
        if tag in ('a','link','script','img'):
            value=d.get('href') or d.get('src')
            if value:self.links.append(value)
def audit(ids):
    findings=[]; stats=[]; pages={}
    for file in OUT.rglob('*.html'):
        p=Page();p.feed(file.read_text());pages[file.resolve()]=p
    for id in ids:
        folder=ROOT/'chapters'/id; errors=[]
        required=[folder/'lesson.html',folder/'meta.json',folder/'research.md',ROOT/'projects'/f'ch{id}'/'Cargo.toml',ROOT/'projects'/f'ch{id}'/'src/main.rs',ROOT/'projects'/f'ch{id}'/'starter/Cargo.toml',ROOT/'exercises/cpu/exercises'/f'ch{id}_01.rs',ROOT/'exercises/cpu/solutions'/f'ch{id}_01.rs']
        for f in required:
            if not f.exists():errors.append('missing '+str(f.relative_to(ROOT)))
        if (folder/'lesson.html').exists():
            content=(folder/'lesson.html').read_text();p=Page();p.feed(content);words=len(plain(content).split())
            if words<1500:errors.append(f'only {words} words; needs substantive explanation')
            if p.h2<4:errors.append(f'only {p.h2} h2 lesson sections')
            if p.details<3:errors.append('missing progressive hints/answers')
            if 'being authored' in content:errors.append('placeholder text')
            if re.search(r'\\(?:\[|\(|frac\{|begin\{)|\$\$',content):errors.append('unrendered LaTeX markup')
            errors+=p.errors
            stats.append(dict(chapter=id,words=words,sections=p.h2,answer_disclosures=p.details))
        if (folder/'meta.json').exists():
            m=json.loads((folder/'meta.json').read_text())
            for key in ('outcomes','lessons','terms','sources','checks','limitations'):
                if key not in m:errors.append('missing metadata '+key)
            for pre in m.get('prerequisites',[]):
                if int(pre)>=int(id):errors.append('forward prerequisite '+str(pre))
        findings += [dict(chapter=id,error=e) for e in errors]
    # Entire generated site's local resources and fragment targets.
    for file,p in pages.items():
        if '/code/' in str(file):continue
        for link in p.links:
            url=urlsplit(link)
            if url.scheme or url.netloc:continue
            path=unquote(url.path)
            target=((OUT/path.lstrip('/')) if path.startswith('/') else (file.parent/path)) if path else file
            if target.is_dir():target=target/'index.html'
            if not target.exists():findings.append(dict(page=str(file.relative_to(OUT)),error='missing link '+link));continue
            if url.fragment and target.suffix=='.html' and target.resolve() in pages and unquote(url.fragment) not in pages[target.resolve()].ids:
                findings.append(dict(page=str(file.relative_to(OUT)),error='missing anchor '+link))
    return stats,findings

def run(cmd,timeout=180):
    start=time.monotonic()
    try:
        p=subprocess.run(cmd,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
        return dict(command=' '.join(cmd),returncode=p.returncode,seconds=round(time.monotonic()-start,3),output=p.stdout[-14000:])
    except subprocess.TimeoutExpired as e:return dict(command=' '.join(cmd),returncode=-1,seconds=timeout,output='TIMEOUT '+str(e))
def rust_checks(ids):
    reports=[]
    for id in ids:
        manifest=ROOT/'projects'/f'ch{id}'/'Cargo.toml'
        if not manifest.exists():continue
        commands=[['cargo','fmt','--manifest-path',str(manifest),'--check'],['cargo','clippy','--offline','--manifest-path',str(manifest),'--all-targets','--','-D','warnings'],['cargo','test','--offline','--manifest-path',str(manifest)]]
        if id not in ('29','30','31','32'):
            commands.append(['cargo','run','--release','--offline','--manifest-path',str(manifest)])
        results=[run(cmd) for cmd in commands]
        starter=manifest.parent/'starter/Cargo.toml'
        if starter.exists():
            results.append(run(['cargo','fmt','--manifest-path',str(starter),'--check']))
            results.append(run(['cargo','clippy','--offline','--manifest-path',str(starter),'--all-targets','--','-D','warnings']))
            results.append(run(['cargo','run','--offline','--manifest-path',str(starter)]))
            test=run(['cargo','test','--offline','--manifest-path',str(starter)]);test['expected_failure']=True
            test['expected_failure_observed']=test['returncode']!=0 and any(marker in test['output'] for marker in ('not yet implemented', 'guided repair:')) and 'error[E' not in test['output'];results.append(test)
        with tempfile.TemporaryDirectory(prefix='ml-check-') as tmp:
            for solution in sorted((ROOT/'exercises/cpu/solutions').glob(f'ch{id}_*.rs')):
                for kind,path in [('solution',solution),('exercise',ROOT/'exercises/cpu/exercises'/solution.name)]:
                    binary=Path(tmp)/(kind+solution.stem)
                    compile=run(['rustc','--edition=2021','--test',str(path),'-o',str(binary)]);results.append(compile)
                    if compile['returncode']==0:
                        test=run([str(binary)])
                        if kind=='exercise':test.update(expected_failure=True,expected_failure_observed=test['returncode']!=0 and ('not yet implemented' in test['output']))
                        results.append(test)
        ok=all(r.get('expected_failure_observed',r['returncode']==0) for r in results)
        reports.append(dict(chapter=id,passed=ok,results=results));print(f'Chapter {id}: {"PASS" if ok else "FAIL"}',flush=True)
        (ROOT/'guidance/validation').mkdir(exist_ok=True)
        (ROOT/'guidance/validation'/f'ch{id}.json').write_text(json.dumps(reports[-1],indent=2)+'\n')
    return reports
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--chapters',nargs='*');parser.add_argument('--rust',action='store_true');args=parser.parse_args()
    ids=args.chapters or [f'{n:02}' for n in range(1,57)];stats,findings=audit(ids)
    (ROOT/'guidance/validation').mkdir(exist_ok=True)
    (ROOT/'guidance/validation/content.json').write_text(json.dumps(dict(stats=stats,findings=findings),indent=2)+'\n')
    print(json.dumps(dict(chapters=len(stats),words=sum(s['words'] for s in stats),findings=findings[:50]),indent=2))
    results=rust_checks(ids) if args.rust else []
    raise SystemExit(bool(findings) or any(not r['passed'] for r in results))
