#!/usr/bin/env python3
"""Small regression checks for reader extraction, safe source links and lab dispatch."""
import hashlib
from pathlib import Path
import tempfile
import zipfile

import build
from chapter import self_test
from verify import Page, goal_delivery_ok, math_errors


def main():
    self_test()
    assert not math_errors('<math><mfrac><mn>1</mn><mi>N</mi></mfrac></math>')
    assert not math_errors('<math><mi>x</mi><mo>&lt;</mo><mn>2</mn></math>')
    assert math_errors('<math><mfrac><mn>1</mn></mfrac></math>')
    assert math_errors('<math><mi>x</mn></math>')
    assert math_errors('<math><mo><mrow><mo>+</mo></mrow></mo></math>')
    assert math_errors('<math><mn>2</mn>')
    assert math_errors('<math><mo>(</mo><mi>x</mi><msup><mo>)</mo><mn>2</mn></msup></math>')
    assert not math_errors('<math><msup><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow><mn>2</mn></msup></math>')
    table = '<table><caption>Numeric trace</caption><tr><td>3</td></tr></table>'
    wrapped = build.scrollable_tables(table)
    assert wrapped.count('class="table-wrap"') == 1 and 'aria-label="Numeric trace"' in wrapped
    assert build.scrollable_tables('<div class="table-wrap">' + table + '</div>') == wrapped
    assert build.inline_math('<p><math><mi>x</mi></math></p>') == '<p><span class="inline-math"><math><mi>x</mi></math></span></p>'
    display = '<div class="equation"><math display="block"><mi>x</mi></math></div>'
    assert build.inline_math(display) == display
    assert not goal_delivery_ok(dict(returncode=1, output="missing data"))
    assert not goal_delivery_ok(dict(returncode=101, output="GOAL_NOT_MET: panicked at x"))
    assert goal_delivery_ok(dict(returncode=1, output="GOAL_NOT_MET: expected MSE below0.01; got0.4"))
    assert goal_delivery_ok(dict(returncode=3, output="GOAL_REVIEW_REQUIRED: verify shared-memory reuse using the trace rubric"))
    assert goal_delivery_ok(dict(returncode=0, output="goal checks passed"))
    fragment = '<section class="learning-step" id="one"><h2>First</h2><section id="detail"><p>Worked value: 7.</p></section></section><section class="learning-step" id="two"><h2>Second</h2><p>Transfer value: 9.</p></section>'
    one = build.step_text(fragment, 'one')
    assert 'Worked value: 7.' in one and 'Transfer value: 9.' not in one
    two = build.step_text(fragment, 'two')
    assert 'Transfer value: 9.' in two and 'Worked value: 7.' not in two
    parser = Page()
    parser.feed(fragment)
    assert parser.steps == ['one', 'two']
    assert 'detail' in parser.ids
    parser.feed('<p id="one">duplicate</p>')
    assert parser.errors == ['duplicate id one']
    nested = Page(top_level_steps=True)
    nested.feed('<div><section class="learning-step" id="nested"><h2>Hidden from reader controls</h2></section></div>')
    assert nested.errors == ['learning-step must be a top-level lesson section: nested']
    assert '<' not in build.json_html({'title': '</script><script>bad()</script>'})
    source = build.ROOT / 'labs/s01-foundations/src/ch01.rs'
    before = hashlib.sha256(source.read_bytes()).digest()
    original_out = build.OUT
    with tempfile.TemporaryDirectory(prefix='ml-build-check-') as folder:
        build.OUT = Path(folder) / 'dist'
        build.main()
        assert hashlib.sha256(source.read_bytes()).digest() == before
        assert (build.OUT / 'index.html').is_file()
        assert (build.OUT / 'sections/09.html').is_file()
        with zipfile.ZipFile(build.OUT / 'downloads/first-principles-source.zip') as archive:
            names = archive.namelist()
            assert 'first-principles/labs/s01-foundations/src/ch01.rs' in names
            assert not any('/starter/' in name or '/exercises/' in name or '/target/' in name or '/.codex/' in name for name in names)
        page = (build.OUT / 'chapters/01.html').read_text()
        assert 'id="chapter-data"' in page and 'id="view-toggle"' in page
    build.OUT = original_out
    print('Reader extraction, archive exclusions, safe embedded metadata and non-mutating build checks passed.')


if __name__ == '__main__':
    main()
