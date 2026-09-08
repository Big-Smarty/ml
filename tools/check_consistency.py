#!/usr/bin/env python3
"""Check the course's shared neuron API, vocabulary ownership, and chapter bridges."""
from html.parser import HTMLParser
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[1]


class Lesson(HTMLParser):
    def __init__(self):
        super().__init__()
        self.has_bridge = False
        self.has_guide = False
        self.code = []
        self.current_code = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.has_bridge |= tag == "section" and attrs.get("id") == "continuity"
        self.has_guide |= tag == "a" and attrs.get("href") == "/conventions.html"
        if tag == "code" and attrs.get("data-source"):
            self.current_code = [attrs["data-source"], []]

    def handle_data(self, data):
        if self.current_code is not None:
            self.current_code[1].append(data)

    def handle_endtag(self, tag):
        if tag == "code" and self.current_code is not None:
            self.code.append(self.current_code)
            self.current_code = None


def compact(text):
    return re.sub(r"\s+", "", text)


def check():
    errors = []
    owners = {}
    for number in range(1, 57):
        chapter = f"{number:02}"
        folder = ROOT / "chapters" / chapter
        lesson = Lesson()
        lesson.feed((folder / "lesson.html").read_text())
        if not lesson.has_bridge or not lesson.has_guide:
            errors.append(f"Chapter {chapter}: missing continuity section or code guide link")
        if not lesson.code:
            errors.append(f"Chapter {chapter}: missing source-checked implementation excerpt")
        for source, chunks in lesson.code:
            path = (ROOT / source).resolve()
            if not path.is_relative_to(ROOT) or not path.is_file():
                errors.append(f"Chapter {chapter}: invalid snippet source {source}")
            elif compact("".join(chunks)) not in compact(path.read_text()):
                errors.append(f"Chapter {chapter}: excerpt differs from {source}")
        for slug in json.loads((folder / "meta.json").read_text()).get("terms", {}):
            if slug in owners:
                errors.append(f"Glossary {slug}: defined by both {owners[slug]} and {chapter}")
            owners[slug] = chapter
        if not (ROOT / "guidance/consistency" / f"ch{chapter}.md").is_file():
            errors.append(f"Chapter {chapter}: missing author review record")
        for base in (ROOT / "projects" / f"ch{chapter}",):
            for path in base.rglob("*.rs"):
                if "target" in path.parts:
                    continue
                if re.search(r"\bfn\s+loss_(?:grad|and_grad)(?:\s*\(|_(?:with_gpu|backend)\s*\()", path.read_text()):
                    errors.append(f"{path.relative_to(ROOT)}: use loss_and_gradient")

    expected = {
        "predict": "fn predict(&self, input: f64) -> f64",
        "loss": "fn loss(&self, data: &[(f64, f64)]) -> Result<f64, &'static str>",
        "numerical_gradient": "fn numerical_gradient(&self, data: &[(f64, f64)]) -> Result<Gradient, &'static str>",
        "step": "fn step(self, data: &[(f64, f64)], learning_rate: f64) -> Result<Self, &'static str>",
        "train": "fn train(self, data: &[(f64, f64)], steps: usize, learning_rate: f64) -> Result<Self, &'static str>",
    }
    for chapter in ("01", "02"):
        source = (ROOT / f"projects/ch{chapter}/src/main.rs").read_text()
        if "struct Neuron" not in source or "impl Neuron" not in source:
            errors.append(f"Chapter {chapter}: first model must be Neuron")
        for name, signature in expected.items():
            match = re.search(rf"\bfn\s+{name}\s*\([^{{]+?(?=\{{)", source)
            actual = compact(match[0]).replace("mutself,", "self,").replace(",)", ")") if match else ""
            if actual != compact(signature):
                errors.append(f"Chapter {chapter}: {name} differs from shared neuron contract")
    # These are the same neuron; Chapter 2 changes only the gradient used by step.
    first = (ROOT / "projects/ch01/src/main.rs").read_text()
    second = (ROOT / "projects/ch02/src/main.rs").read_text()
    sixth = (ROOT / "projects/ch06/src/main.rs").read_text()
    for name in expected:
        pattern = rf"^    fn {name}\b.*?^    }}\n"
        one = re.search(pattern, first, re.S | re.M)
        two = re.search(pattern, second, re.S | re.M)
        if one and two:
            body = one[0].replace(".numerical_gradient(data)", ".gradient(data)") if name == "step" else one[0]
            if compact(body) != compact(two[0]):
                errors.append(f"Chapters 01/02: shared {name} implementation drifted")
        if name in ("predict", "loss"):
            six = re.search(pattern, sixth, re.S | re.M)
            if not one or not six or compact(one[0]) != compact(six[0]):
                errors.append(f"Chapters 01/06: shared {name} implementation drifted")
    return errors


if __name__ == "__main__":
    findings = check()
    for finding in findings:
        print(finding)
    print(f"Consistency: {len(findings)} findings across 56 chapters")
    raise SystemExit(bool(findings))
