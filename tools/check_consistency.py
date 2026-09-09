#!/usr/bin/env python3
"""Check new learning-step mappings and literal source excerpts without running labs."""
import json
from build import ROOT
from verify import chapter_audit


def check():
    course = json.loads((ROOT / 'course.json').read_text())
    sections = json.loads((ROOT / 'sections.json').read_text())
    errors = []
    for chapter in course:
        try:
            _, findings = chapter_audit(chapter['id'], chapter, sections)
            errors += [f'Chapter {chapter["id"]}: {finding}' for finding in findings]
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f'Chapter {chapter["id"]}: {error}')
    return errors


if __name__ == '__main__':
    findings = check()
    for finding in findings:
        print(finding)
    print(f'Consistency: {len(findings)} findings across 56 chapters')
    raise SystemExit(bool(findings))
