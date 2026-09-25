"""Contract checks for the dependency-free GitHub Pages application."""
from html.parser import HTMLParser
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.tabs = {}
        self.panels = {}
        self.local_assets = []

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
        if tag == "link" and attrs.get("rel") == "stylesheet":
            self.local_assets.append(attrs["href"])


def test_pages_assets_are_local_and_present():
    parser = PageParser()
    parser.feed((WEB / "index.html").read_text())
    assert parser.local_assets
    assert all(not asset.startswith(("http:", "https:", "//")) for asset in parser.local_assets)
    assert all((WEB / asset.removeprefix("./")).is_file() for asset in parser.local_assets)


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


def test_page_explains_scope_and_local_audio_handling():
    html = (WEB / "index.html").read_text().lower()
    assert "not a keytap3 port" in html
    assert "no analytics or upload endpoint" in html
    assert "synthetic keyboard model" in html
