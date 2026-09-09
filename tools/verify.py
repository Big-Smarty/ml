#!/usr/bin/env python3
"""Audit the new course and verify delivered labs; not a learner-completion grade."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import argparse
import json
import re
import subprocess
import time

from build import ROOT, OUT, plain


class Page(HTMLParser):
    def __init__(self, top_level_steps=False):
        super().__init__()
        self.top_level_steps = top_level_steps
        self.stack = []
        self.ids = set()
        self.links = []
        self.steps = []
        self.h2 = 0
        self.details = 0
        self.errors = []
        self.source_excerpts = []
        self.current_code = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            if attrs['id'] in self.ids:
                self.errors.append('duplicate id ' + attrs['id'])
            self.ids.add(attrs['id'])
        if tag == 'section' and 'learning-step' in attrs.get('class', '').split():
            self.steps.append(attrs.get('id', ''))
            if self.top_level_steps and self.stack:
                self.errors.append('learning-step must be a top-level lesson section: ' + attrs.get('id', ''))
        self.h2 += tag == 'h2'
        self.details += tag == 'details'
        if tag in ('a', 'link', 'script', 'img', 'iframe'):
            value = attrs.get('href') or attrs.get('src')
            if value:
                self.links.append(value)
        if tag == 'code' and attrs.get('data-source'):
            self.current_code = [attrs['data-source'], []]
        if tag not in ('area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'):
            self.stack.append(tag)

    def handle_data(self, data):
        if self.current_code is not None:
            self.current_code[1].append(data)

    def handle_endtag(self, tag):
        if tag in self.stack:
            self.stack = self.stack[:len(self.stack) - 1 - self.stack[::-1].index(tag)]
        if tag == 'code' and self.current_code is not None:
            self.source_excerpts.append(self.current_code)
            self.current_code = None


def compact(text):
    return re.sub(r'\s+', '', text)


def chapter_audit(number, course, sections):
    folder = ROOT / 'chapters' / number
    content = (folder / 'lesson.html').read_text()
    meta = json.loads((folder / 'meta.json').read_text())
    page = Page(top_level_steps=True)
    page.feed(content)
    errors = list(page.errors)
    for key in ('outcomes', 'terms', 'sources', 'checks', 'limitations', 'steps', 'sessions', 'lab', 'interactives', 'topic_coverage'):
        if key not in meta:
            errors.append('missing metadata ' + key)
    if meta.get('redesign_version') != 2:
        errors.append('chapter has not completed the redesign')
    steps = meta.get('steps', [])
    step_ids = [s.get('id') for s in steps]
    if not steps or step_ids != page.steps:
        errors.append('ordered metadata steps differ from learning-step sections')
    if len(set(step_ids)) != len(step_ids):
        errors.append('repeated step ids')
    for step in steps:
        if not step.get('title') or not isinstance(step.get('minutes'), (int, float)) or step.get('minutes', 0) <= 0:
            errors.append('step needs a title and positive time estimate')
    sessions = meta.get('sessions', [])
    session_ids = [s.get('id') for s in sessions]
    if not sessions or len(set(session_ids)) != len(session_ids):
        errors.append('missing or duplicate sessions')
    for step in steps:
        if step.get('session') not in session_ids:
            errors.append('step references an unknown session')
    for session in sessions:
        if not session.get('title') or not session.get('goal'):
            errors.append('session needs a title and observable goal')
        minutes = sum(s.get('minutes', 0) for s in steps if s.get('session') == session['id'])
        if not 20 <= minutes <= 60:
            errors.append(f'session {session["id"]} estimate {minutes}min needs review against the approximate 30–45min plan')
    topics = set(course['topics'])
    coverage = meta.get('topic_coverage', {})
    if set(coverage) != topics:
        errors.append(f'topic coverage differs: missing {sorted(topics-set(coverage))}; unexpected {sorted(set(coverage)-topics)}')
    for topic, step in coverage.items():
        if step not in step_ids:
            errors.append(f'topic {topic!r} points to an unknown step')
    if re.search(r'Rustlings|just (?:starter|exercises)\b|projects/ch\d+/starter', content, re.I):
        errors.append('retired exercise workflow in active lesson')
    if re.search(r'\\(?:\[|\(|frac\{|begin\{)|\$\$', content):
        errors.append('unrendered LaTeX')
    if page.details < 2:
        errors.append('missing meaningful hints/worked answers')
    for pre in meta.get('prerequisites', course['prerequisites']):
        if not str(pre).isdigit() or int(pre) >= int(number) or int(pre) < 1:
            errors.append('invalid or forward prerequisite ' + str(pre))
    section = next(s for s in sections if number in s['chapters'])
    lab = meta.get('lab', {})
    if 'manual_checks' in lab and (not isinstance(lab['manual_checks'], list) or not all(isinstance(item, str) and item.strip() for item in lab['manual_checks'])):
        errors.append('manual_checks must be a list of concrete review criteria')
    if lab.get('package') != section['package'] or lab.get('chapter') != number:
        errors.append('lab mapping differs from section registry')
    entry = ROOT / lab.get('entry', 'missing')
    if not entry.is_relative_to(ROOT / 'labs' / section['package']) or not entry.is_file():
        errors.append('missing learner source entry in assigned package')
    required = [ROOT / 'labs' / section['package'] / p for p in ('Cargo.toml', 'README.md', f'src/solutions/ch{number}.rs')]
    required += [folder / 'research.md', ROOT / 'guidance/redesign' / f'section-{section["id"]}.md']
    for path in required:
        if not path.is_file():
            errors.append('missing ' + str(path.relative_to(ROOT)))
    for source, chunks in page.source_excerpts:
        if source.startswith('exercises/'):
            errors.append('retired exercise source in active lesson ' + source)
        path = (ROOT / source).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            errors.append('invalid source excerpt path ' + source)
        elif compact(''.join(chunks)) not in compact(path.read_text()):
            errors.append('excerpt differs from ' + source)
    for demo in meta.get('interactives', []):
        if not demo.get('id') or not demo.get('title') or demo.get('illustration') is not True:
            errors.append('interactive needs id/title and explicit illustration label')
        if demo.get('anchor') not in page.ids:
            errors.append('missing interactive anchor ' + str(demo.get('anchor')))
        if not (folder / demo.get('file', 'missing')).is_file():
            errors.append('missing interactive script ' + str(demo.get('file')))
    stats = dict(chapter=number, words=len(plain(content).split()), steps=len(steps), sessions=len(sessions), topics=len(topics), interactives=len(meta.get('interactives', [])))
    return stats, errors


def audit(ids):
    chapters = json.loads((ROOT / 'course.json').read_text())
    sections = json.loads((ROOT / 'sections.json').read_text())
    stats, findings, pages = [], [], {}
    for file in OUT.rglob('*.html'):
        parser = Page()
        parser.feed(file.read_text())
        pages[file.resolve()] = parser
    if not (OUT / 'index.html').is_file():
        findings.append(dict(error='dist is missing; run python3 tools/build.py first'))
    for number in ids:
        try:
            stat, errors = chapter_audit(number, next(c for c in chapters if c['id'] == number), sections)
            stats.append(stat)
            findings += [dict(chapter=number, error=error) for error in errors]
        except (OSError, ValueError, KeyError, TypeError) as error:
            findings.append(dict(chapter=number, error=str(error)))
    # Check all rendered local links and both fragment and guided-step targets.
    for file, page in pages.items():
        if file.is_relative_to(OUT / 'code'):
            continue
        findings += [dict(page=str(file.relative_to(OUT)), error=error) for error in page.errors]
        for link in page.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            path = unquote(url.path)
            target = (OUT / path.lstrip('/') if path.startswith('/') else file.parent / path) if path else file
            if target.is_dir():
                target /= 'index.html'
            target = target.resolve()
            if not target.is_relative_to(OUT.resolve()) or not target.exists():
                findings.append(dict(page=str(file.relative_to(OUT)), error='missing local link ' + link))
                continue
            if url.fragment and target.suffix == '.html' and target in pages and unquote(url.fragment) not in pages[target].ids:
                findings.append(dict(page=str(file.relative_to(OUT)), error='missing anchor ' + link))
            if 'step=' in url.query and target in pages:
                from urllib.parse import parse_qs
                for step in parse_qs(url.query).get('step', []):
                    if step not in pages[target].steps:
                        findings.append(dict(page=str(file.relative_to(OUT)), error='unknown learning step ' + link))
    if len(ids) == 56:
        tools = {demo['id'] for c in chapters for demo in json.loads((ROOT / 'chapters' / c['id'] / 'meta.json').read_text()).get('interactives', [])}
        if len(tools) != 18:
            findings.append(dict(error=f'expected 18 focused interactive tools; found {len(tools)}'))
    return stats, findings


def run(command, timeout=300):
    start = time.monotonic()
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        return dict(command=command, returncode=result.returncode, seconds=round(time.monotonic()-start, 3), output=result.stdout[-18000:])
    except subprocess.TimeoutExpired as error:
        return dict(command=command, returncode=-1, seconds=timeout, output='TIMEOUT ' + str(error))


def goal_delivery_ok(result):
    if re.search(r'panicked at|error\[E\d|could not compile', result['output']):
        return False
    return result['returncode'] == 0 or (result['returncode'] == 1 and 'GOAL_NOT_MET:' in result['output']) or (result['returncode'] == 3 and 'GOAL_REVIEW_REQUIRED:' in result['output'])


def rust_checks(ids):
    sections = json.loads((ROOT / 'sections.json').read_text())
    reports = []
    for section in sections:
        numbers = [n for n in section['chapters'] if n in ids]
        if not numbers:
            continue
        manifest = f'labs/{section["package"]}/Cargo.toml'
        base = ['--offline', '--manifest-path', manifest]
        checks = [['cargo', 'fmt', '--manifest-path', manifest, '--check'], ['cargo', 'clippy', *base, '--all-targets', '--', '-D', 'warnings'], ['cargo', 'test', *base]]
        results = [run(command) for command in checks]
        for number in numbers:
            command = ['cargo', 'run', '--release', *base, '--', number]
            results.append(run(command))
            results.append(run([*command, '--solution', '--check']))
            goal = run([*command, '--check'])
            goal['learner_goal_met'] = goal['returncode'] == 0
            # Only explicit expected learning statuses can substitute for success.
            goal['delivery_ok'] = goal_delivery_ok(goal)
            goal['manual_review_required'] = goal['returncode'] == 3 and 'GOAL_REVIEW_REQUIRED:' in goal['output']
            if goal['manual_review_required']:
                meta = json.loads((ROOT / 'chapters' / number / 'meta.json').read_text())
                goal['delivery_ok'] = goal['delivery_ok'] and bool(meta.get('lab', {}).get('manual_checks'))
            results.append(goal)
        passed = all(r.get('delivery_ok', r['returncode'] == 0) for r in results)
        report = dict(section=section['id'], package=section['package'], chapters=numbers, passed=passed, results=results)
        reports.append(report)
        folder = ROOT / 'guidance/validation-redesign'
        folder.mkdir(exist_ok=True)
        (folder / f'section-{section["id"]}.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f'Section {section["id"]}: {"PASS" if passed else "FAIL"} ({section["package"]})', flush=True)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chapters', nargs='+')
    parser.add_argument('--rust', action='store_true')
    args = parser.parse_args()
    from chapter import chapter
    ids = [chapter(value) for value in args.chapters] if args.chapters else [f'{n:02}' for n in range(1, 57)]
    stats, findings = audit(ids)
    for finding in findings:
        print(json.dumps(finding))
    rust = rust_checks(ids) if args.rust else []
    print(f'Audit: {len(stats)} chapters, {sum(s["steps"] for s in stats)} steps, {len(findings)} findings')
    report = dict(stats=stats, findings=findings, rust=rust)
    folder = ROOT / 'guidance/validation-redesign'
    folder.mkdir(exist_ok=True)
    (folder / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    return bool(findings) or any(not s['passed'] for s in rust)


if __name__ == '__main__':
    raise SystemExit(main())
