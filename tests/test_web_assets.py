"""Contract checks for the dependency-free GitHub Pages application."""
from html.parser import HTMLParser
from pathlib import Path
import re
import shutil
import subprocess
import json


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.tabs = {}
        self.panels = {}
        self.local_assets = []
        self.audio_assets = []
        self.audio_count = 0
        self.modules = 0
        self.phrase_modules = 0
        self.module_grips = 0
        self.phrases = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if attrs.get("role") == "tab":
            self.tabs[attrs.get("id")] = attrs.get("aria-controls")
        if attrs.get("role") == "tabpanel":
            self.panels[attrs.get("id")] = attrs.get("aria-labelledby")
        if tag == "script" and attrs.get("src"):
            self.local_assets.append(attrs["src"])
        if tag == "audio":
            self.audio_count += 1
            if attrs.get("src"):
                self.audio_assets.append(attrs["src"])
        if tag == "article" and attrs.get("role") == "listitem":
            self.modules += 1
            if "phrase-module" in attrs.get("class", "").split():
                self.phrase_modules += 1
        if tag == "button" and "module-grab" in attrs.get("class", "").split() and attrs.get("draggable") == "true":
            self.module_grips += 1
        if tag == "button" and "analyze-phrase-btn" in attrs.get("class", "").split():
            self.phrases.append({key: attrs.get(key) for key in ("data-text", "data-seed", "data-kps")})
        if tag == "link" and attrs.get("rel") == "stylesheet":
            self.local_assets.append(attrs["href"])


def test_pages_assets_are_local_and_present():
    parser = PageParser()
    parser.feed((WEB / "index.html").read_text())
    assert parser.audio_count == 1  # one review player shared by every phrase module
    assert parser.local_assets
    assert all(not asset.startswith(("http:", "https:", "//")) for asset in parser.local_assets)
    assert all((WEB / asset.removeprefix("./")).is_file() for asset in parser.local_assets)
    assert set(parser.audio_assets) == {"./audio/synthetic-demo.wav"}
    assert all((WEB / asset.removeprefix("./")).is_file() for asset in parser.audio_assets)


def test_audio_examples_are_independent_reorderable_modules():
    parser = PageParser()
    parser.feed((WEB / "index.html").read_text())
    assert parser.modules == 4  # three examples plus the upload module
    assert parser.phrase_modules == 3
    assert parser.module_grips == parser.modules
    assert 'id="keyboardType"' in (WEB / "index.html").read_text()
    assert 'value="laptop"' in (WEB / "index.html").read_text()
    assert 'value="membrane"' in (WEB / "index.html").read_text()
    assert 'value="mechanical"' in (WEB / "index.html").read_text()
    source = (WEB / "js/app.js").read_text()
    assert 'addEventListener("dragover"' in source
    assert 'querySelector(".module-up")' in source
    assert 'querySelector(".module-down")' in source
    assert 'id="moduleOrderStatus" class="sr-only" aria-live="polite"' in (WEB / "index.html").read_text()
    assert 'document.querySelectorAll(".analyze-phrase-btn")' in source
    assert 'K.SYNTH_KEYBOARDS[keyboardType]' in source
    configured = json.loads((ROOT / "scripts/example_wavs.json").read_text())
    assert parser.phrases == [
        {"data-text": item["text"], "data-seed": str(item["seed"]), "data-kps": str(item["keysPerSecond"])}
        for item in configured
    ]


def test_tabs_and_panels_have_matching_accessible_relationships():
    parser = PageParser()
    parser.feed((WEB / "index.html").read_text())
    assert parser.tabs == {"tab-lab": "lab", "tab-live": "live", "tab-about": "about"}
    assert parser.panels == {"lab": "tab-lab", "live": "tab-live", "about": "tab-about"}


def test_app_bindings_resolve_to_page_controls():
    parser = PageParser()
    parser.feed((WEB / "index.html").read_text())
    source = (WEB / "js/app.js").read_text()
    bound_ids = set(re.findall(r'\$\("([A-Za-z][A-Za-z0-9_-]*)"\)', source))
    assert bound_ids <= parser.ids
    assert {"runBtn", "micBtn", "trainBtn", "testBox", "maskToggle"} <= parser.ids
    assert {"wavFileInput", "expectedSequence", "analyzeUploadBtn", "sampleSequence"} <= parser.ids


def test_wav_results_are_rendered_in_timestamped_sequence_order():
    html = (WEB / "index.html").read_text()
    source = (WEB / "js/app.js").read_text()
    assert 'id="audioTimeline"' in html
    assert 'id="sampleAudio"' in html
    assert 'formatTimestamp(onsets[item.predIndex])' in source
    assert 'time.dateTime = `PT${sec.toFixed(3)}S`' in source
    assert 'result.textContent = expected === null' in source
    assert 'file.arrayBuffer()' in source
    assert 'renderAudioTimeline(samples, detected.onsets, summary.items)' in source
    assert 'timelineAudio.currentTime = fraction * timelineState.duration' in source


def test_page_explains_scope_and_local_audio_handling():
    html = (WEB / "index.html").read_text().lower()
    assert "not a keytap3 port" in html
    assert "no analytics or upload endpoint" in html
    assert "synthetic keyboard model" in html


def test_example_audio_is_a_small_pcm_wav():
    wav = (WEB / "audio/synthetic-demo.wav").read_bytes()
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    assert int.from_bytes(wav[22:24], "little") == 1  # mono
    assert int.from_bytes(wav[24:28], "little") == 16000
    assert int.from_bytes(wav[34:36], "little") == 16
    assert len(wav) < 100_000


def test_example_wav_recognizes_end_to_end_when_node_is_available():
    node = shutil.which("node")
    if not node:
        import pytest
        pytest.skip("Node.js is needed to exercise the browser recognizer")
    subprocess.run([node, str(ROOT / "scripts/check_example_wav.js")], check=True, cwd=ROOT)
