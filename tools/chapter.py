#!/usr/bin/env python3
"""Map chapter numbers to the nine labs, forwarding arguments without a shell."""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def chapter(value):
    if not re.fullmatch(r'(?:0?[1-9]|[1-4][0-9]|5[0-6])', value):
        raise ValueError('chapter must be 1..56 (1 and 01 are both accepted)')
    return f'{int(value):02}'


def lab_manifest(number):
    sections = json.loads((ROOT / 'sections.json').read_text())
    package = next(section['package'] for section in sections if number in section['chapters'])
    return f'labs/{package}/Cargo.toml'


def commands(action, args):
    if action in ('check', 'verify'):
        ids = [chapter(value) for value in args]
        audit = [sys.executable, 'tools/verify.py']
        if action == 'verify':
            audit.append('--rust')
        if ids:
            audit += ['--chapters', *ids]
        return [[sys.executable, 'tools/build.py'], audit]
    if not args:
        raise ValueError('a chapter number is required')
    number, *extra = args
    number = chapter(number)
    manifest = lab_manifest(number)
    if action in ('deps', 'lab-build', 'lint', 'asm'):
        if extra:
            raise ValueError(f'{action} accepts one chapter only')
        operation = {'deps': 'fetch', 'lab-build': 'build', 'lint': 'clippy', 'asm': 'rustc'}[action]
        flags = [] if action == 'deps' else ['--offline', '--release']
        if action == 'lint':
            flags += ['--all-targets']
        trailing = ['--', '-D', 'warnings'] if action == 'lint' else ['--', '--emit=asm'] if action == 'asm' else []
        return [['cargo', operation, *flags, '--manifest-path', manifest, *trailing]]
    if action in ('fmt', 'fmt-check'):
        if extra:
            raise ValueError('format recipes accept one chapter only')
        return [['cargo', 'fmt', '--manifest-path', manifest, *(['--check'] if action == 'fmt-check' else [])]]
    if action in ('lab', 'run', 'lab-check', 'solution', 'gpu'):
        flags = []
        if action == 'lab-check':
            flags.append('--check')
        if action == 'solution':
            flags.append('--solution')
        if action == 'gpu':
            if number not in ('29', '30', '31', '32'):
                raise ValueError('the GPU lab recipe supports chapters 29–32; see other chapters for explicit backend commands')
            flags.append('--gpu')
        return [['cargo', 'run', '--offline', '--release', '--manifest-path', manifest, '--', number, *flags, *extra]]
    if action in ('lab-test', 'test'):
        return [['cargo', 'test', '--offline', '--release', '--all-targets', '--manifest-path', manifest, '--', *extra]]
    if action == 'gpu-test':
        if number not in ('29', '30', '31', '32'):
            raise ValueError('gpu-test supports chapters 29–32')
        return [['cargo', 'test', '--offline', '--release', '--manifest-path', manifest, '--', '--ignored', '--nocapture', *extra]]
    if action == 'reference-gpu-test':
        if number != '36':
            raise ValueError('the preserved decoder GPU test is chapter 36')
        return [['cargo', 'test', '--offline', '--release', '--features', 'gpu', '--manifest-path', 'projects/ch36/Cargo.toml', '--', '--ignored', '--nocapture', *extra]]
    if action in ('reference', 'reference-test'):
        operation = 'test' if action == 'reference-test' else 'run'
        return [['cargo', operation, '--offline', '--release', '--manifest-path', f'projects/ch{number}/Cargo.toml', '--', *extra]]
    raise ValueError('unknown action; use just --list for the current lab commands')


def self_test():
    assert chapter('1') == chapter('01') == '01'
    assert chapter('56') == '56'
    for value in ('00', '0', '57', '001', '../path', '1; echo bad', '１', '-1'):
        try:
            chapter(value)
        except ValueError:
            continue
        raise AssertionError(f'accepted invalid chapter {value!r}')
    literal = ['a path with spaces', '$(touch should-not-exist)', 'x; echo nope', '--steps', '2']
    assert commands('lab', ['1', *literal])[0][-len(literal):] == literal
    assert commands('lab-check', ['01'])[0][-2:] == ['01', '--check']
    assert commands('solution', ['56', '--check'])[0][-3:] == ['56', '--solution', '--check']
    assert commands('verify', ['1', '02'])[1][-3:] == ['--chapters', '01', '02']
    assert commands('gpu', ['29'])[0][-2:] == ['29', '--gpu']
    assert '--offline' not in commands('deps', ['29'])[0]
    assert commands('lint', ['01'])[0][-3:] == ['--', '-D', 'warnings']
    assert commands('asm', ['28'])[0][-2:] == ['--', '--emit=asm']
    assert commands('lab-test', ['53', 'serving_monitor_and_rollback_are_real', '--ignored'])[0][-3:] == ['--', 'serving_monitor_and_rollback_are_real', '--ignored']
    assert commands('reference-gpu-test', ['36'])[0][4:6] == ['--features', 'gpu']
    for action, number in [('gpu-test', '01'), ('reference-gpu-test', '35')]:
        try:
            commands(action, [number])
        except ValueError:
            continue
        raise AssertionError(f'accepted unsupported hardware recipe {action} {number}')
    assert lab_manifest('06') == lab_manifest('01')
    assert lab_manifest('07') != lab_manifest('06')
    assert len({lab_manifest(f'{i:02}') for i in range(1, 57)}) == 9
    print('Recipe checks passed: chapter bounds, nine-package mapping, goal checks and literal argument forwarding.')


def main():
    if len(sys.argv) < 2:
        raise ValueError('use a just recipe, or pass a chapter action')
    action, *args = sys.argv[1:]
    if action == 'self-test':
        self_test()
        return 0
    for command in commands(action, args):
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from error
