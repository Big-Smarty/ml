#!/usr/bin/env python3
"""Small argv-only adapter for chapter recipes; existing tools own all course checks."""
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def chapter(value):
    if not re.fullmatch(r"(?:0?[1-9]|[1-4][0-9]|5[0-6])", value):
        raise ValueError("chapter must be 1..56 (1 and 01 are both accepted)")
    return f"{int(value):02}"


def commands(action, args):
    if action in ("check", "verify"):
        ids = [chapter(value) for value in args]
        check = [sys.executable, "tools/verify.py"]
        if action == "verify":
            check.append("--rust")
        if ids:
            check += ["--chapters", *ids]
        return [[sys.executable, "tools/build.py"], check]
    if not args:
        raise ValueError("a chapter number is required")
    number, *extra = args
    number = chapter(number)
    manifest = f"projects/ch{number}/Cargo.toml"
    if action in ("starter", "starter-test"):
        manifest = f"projects/ch{number}/starter/Cargo.toml"
    if action in ("fmt", "fmt-check"):
        if extra:
            raise ValueError("format recipes accept one chapter only")
        return [["cargo", "fmt", "--manifest-path", manifest, *(["--check"] if action == "fmt-check" else [])]]
    features = []
    if action in ("gpu", "gpu-test"):
        supported = {"29", "30", "31", "32", "39" if action == "gpu" else "36"}
        if number not in supported:
            raise ValueError(f"{action} supports chapters {', '.join(sorted(supported))}")
        if number in ("36", "39"):
            features = ["--features", "gpu"]
        if action == "gpu-test":
            extra = ["--ignored", *extra]
        elif number == "39":
            extra = ["--gpu", *extra]
    elif action == "run" and number in {"29", "30", "31", "32"}:
        raise ValueError(f"chapter {number} runs on a GPU; opt in with just gpu {number}")
    if action not in ("run", "test", "starter", "starter-test", "gpu", "gpu-test"):
        raise ValueError("unknown chapter action")
    operation = "test" if action in ("test", "starter-test", "gpu-test") else "run"
    return [["cargo", operation, "--offline", *(["--release"] if operation == "run" else []),
             "--manifest-path", manifest, *features, "--", *extra]]


def self_test():
    assert chapter("1") == chapter("01") == "01"
    assert chapter("56") == "56"
    for value in ("00", "0", "57", "001", "../path", "1; echo bad", "１", "-1"):
        try:
            chapter(value)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid chapter {value!r}")
    literal = ["a path with spaces", "$(touch should-not-exist)", "x; echo nope", "--steps", "2"]
    assert commands("run", ["1", *literal])[0][-len(literal):] == literal
    assert commands("starter-test", ["01"])[0][4] == "projects/ch01/starter/Cargo.toml"
    assert commands("verify", ["1", "02"])[1][-3:] == ["--chapters", "01", "02"]
    assert commands("gpu", ["39"])[0][-3:] == ["gpu", "--", "--gpu"]
    assert commands("gpu-test", ["36"])[0][-3:] == ["gpu", "--", "--ignored"]
    print("Recipe checks passed: chapter bounds, normalized paths, and literal argv forwarding.")


def main():
    if len(sys.argv) < 2:
        raise ValueError("use a just recipe, or pass a chapter action")
    action, *args = sys.argv[1:]
    if action == "self-test":
        self_test()
        return 0
    if action == "exercises":
        bundled = ROOT / ".tools/bin/rustlings"
        runner = str(bundled) if bundled.is_file() else shutil.which("rustlings")
        if not runner:
            raise ValueError("Rustlings is not installed; see the README setup instructions")
        return subprocess.run([runner, *args], cwd=ROOT / "exercises/cpu").returncode
    for command in commands(action, args):
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from error
