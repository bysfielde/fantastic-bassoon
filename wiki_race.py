#!/usr/bin/env python3
"""
Wikipedia Race - Singleplayer
Navigate from a start article to a target article using only Wikipedia links.
Requires: pip install PyQt6 PyQt6-WebEngine
"""

import sys
import time
import threading
import urllib.request
import json
from collections import deque
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QLineEdit, QDialog, QDialogButtonBox,
    QListWidget, QListWidgetItem, QFrame, QMessageBox, QSizePolicy
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile, QWebEngineScript
from PyQt6.QtCore import QUrl, Qt, QTimer, pyqtSignal, QObject, QThread
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon
import urllib.parse
from sentence_transformers import SentenceTransformer, SimilarityFunction
import numpy as np

# Semantic Textual Similarity
WIKI_API = "https://en.wikipedia.org/w/api.php"
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
hubData = np.load("hub_embeddings.npz")

def get_article_text(title):
    params = urllib.parse.urlencode({
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": 1,
        "exintro": 1
    })
    req = urllib.request.Request(
        f"{WIKI_API}?{params}",
        headers={"User-Agent": "WikiRaceGame/1.0"}
    )
    data = json.loads(urllib.request.urlopen(req).read())
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        return page.get("extract", None)
    return None
def best_hub(title):
    text = get_article_text(title)
    data = list(zip(hubData["titles"].tolist(), hubData["matrix"]))
    closeHub = ""
    closest = 0.0
    if text:
        embedding = model.encode(text)
        # Calculates cosine similarity of article with each hub
        for t, m in data:
            s = model.similarity(embedding, m).numpy()[0, 0]
            if s > closest:
                closest = s
                closeHub = t
    return closeHub
# ─── Bidirectional BFS Solver ─────────────────────────────────────────────────
#
# WIKI_API = "https://en.wikipedia.org/w/api.php"
# MAX_BFS_DEPTH = 3       # max hops per side before giving up
# LINKS_PER_PAGE = 500    # Wikipedia API max
#
#
# def _wiki_links(titles):
#     """Fetch outgoing links for up to 50 articles in one API call.
#     Returns {title: [linked_title, ...]}."""
#     result = {}
#     # API accepts up to 50 titles at once
#     for i in range(0, len(titles), 50):
#         chunk = titles[i:i+50]
#         params = urllib.parse.urlencode({
#             "action": "query",
#             "format": "json",
#             "titles": "|".join(chunk),
#             "prop": "links",
#             "pllimit": LINKS_PER_PAGE,
#             "plnamespace": 0,       # main namespace only
#             "redirects": 1,
#         })
#         url = f"{WIKI_API}?{params}"
#         req = urllib.request.Request(url, headers={"User-Agent": "WikiRaceGame/1.0"})
#         with urllib.request.urlopen(req, timeout=8) as resp:
#             data = json.loads(resp.read())
#
#         pages = data.get("query", {}).get("pages", {})
#         for page in pages.values():
#             title = page.get("title", "")
#             links = [l["title"] for l in page.get("links", [])]
#             result[title] = links
#
#         # Handle continues (paginated links) — fetch remaining pages
#         while "continue" in data:
#             cont_params = dict(urllib.parse.parse_qsl(params))
#             cont_params.update(data["continue"])
#             cont_url = f"{WIKI_API}?{urllib.parse.urlencode(cont_params)}"
#             req = urllib.request.Request(cont_url, headers={"User-Agent": "WikiRaceGame/1.0"})
#             with urllib.request.urlopen(req, timeout=8) as resp:
#                 data = json.loads(resp.read())
#             pages = data.get("query", {}).get("pages", {})
#             for page in pages.values():
#                 title = page.get("title", "")
#                 existing = result.get(title, [])
#                 existing.extend(l["title"] for l in page.get("links", []))
#                 result[title] = existing
#
#     return result
#
#
# def _resolve_redirect(title):
#     """Resolve a Wikipedia redirect to the canonical title."""
#     params = urllib.parse.urlencode({
#         "action": "query",
#         "format": "json",
#         "titles": title,
#         "redirects": 1,
#     })
#     url = f"{WIKI_API}?{params}"
#     req = urllib.request.Request(url, headers={"User-Agent": "WikiRaceGame/1.0"})
#     with urllib.request.urlopen(req, timeout=8) as resp:
#         data = json.loads(resp.read())
#     pages = data.get("query", {}).get("pages", {})
#     for page in pages.values():
#         return page.get("title", title)
#     return title
#
#
# def _wiki_backlinks(titles):
#     """Fetch incoming links (pages that link TO each title) using 'What links here'.
#     Returns {title: [source_title, ...]} — i.e. which articles point to each title.
#     Only processes one title at a time (API limitation for linkshere)."""
#     result = {}
#     for title in titles:
#         sources = []
#         params = urllib.parse.urlencode({
#             "action": "query",
#             "format": "json",
#             "list": "backlinks",
#             "bltitle": title,
#             "bllimit": LINKS_PER_PAGE,
#             "blnamespace": 0,
#             "blredirect": 0,
#         })
#         url = f"{WIKI_API}?{params}"
#         req = urllib.request.Request(url, headers={"User-Agent": "WikiRaceGame/1.0"})
#         with urllib.request.urlopen(req, timeout=8) as resp:
#             data = json.loads(resp.read())
#         sources.extend(bl["title"] for bl in data.get("query", {}).get("backlinks", []))
#
#         while "continue" in data:
#             cont_params = dict(urllib.parse.parse_qsl(params))
#             cont_params.update(data["continue"])
#             cont_url = f"{WIKI_API}?{urllib.parse.urlencode(cont_params)}"
#             req = urllib.request.Request(cont_url, headers={"User-Agent": "WikiRaceGame/1.0"})
#             with urllib.request.urlopen(req, timeout=8) as resp:
#                 data = json.loads(resp.read())
#             sources.extend(bl["title"] for bl in data.get("query", {}).get("backlinks", []))
#
#         result[title] = sources
#     return result
#
#
# def find_shortest_path(start, target, stop_event):
#     """
#     True bidirectional BFS using directed Wikipedia links:
#       - Forward frontier expands using outgoing links (prop=links)
#         — the links a player can actually click on a page.
#       - Backward frontier expands using incoming links (list=backlinks / "What links here")
#         — articles that link TO a given page, so the backward path is
#         always one that a forward player could have followed.
#     Paths are only joined when a forward neighbor is found in the backward
#     frontier (i.e. the forward side reaches an article the backward side
#     has confirmed is reachable-to-target via real outgoing links).
#     """
#     start = _resolve_redirect(start)
#     target = _resolve_redirect(target)
#
#     if start.lower() == target.lower():
#         return [start]
#
#     # {title: path_from_that_origin}
#     fwd = {start: [start]}       # paths from start →
#     bwd = {target: [target]}     # paths from target ← (stored in reverse: target…node)
#
#     fwd_visited = {start}
#     bwd_visited = {target}
#
#     for _ in range(MAX_BFS_DEPTH):
#         if stop_event.is_set():
#             return None
#
#         # ── Expand forward frontier (outgoing links) ──────────────────────
#         if not fwd:
#             return None
#         try:
#             fwd_links = _wiki_links(list(fwd.keys()))
#         except Exception:
#             return None
#         if stop_event.is_set():
#             return None
#
#         next_fwd = {}
#         for title, path in fwd.items():
#             for neighbor in fwd_links.get(title, []):
#                 if neighbor in fwd_visited:
#                     continue
#                 new_path = path + [neighbor]
#                 if neighbor in bwd_visited:
#                     # neighbor is reachable from target going backwards →
#                     # bwd[neighbor] is stored as [target, …, neighbor], reverse it
#                     return new_path + list(reversed(bwd[neighbor][:-1]))
#                 fwd_visited.add(neighbor)
#                 next_fwd[neighbor] = new_path
#         fwd = next_fwd
#
#         if stop_event.is_set():
#             return None
#
#         # ── Expand backward frontier (incoming links / What links here) ───
#         if not bwd:
#             return None
#         try:
#             bwd_links = _wiki_backlinks(list(bwd.keys()))
#         except Exception:
#             return None
#         if stop_event.is_set():
#             return None
#
#         next_bwd = {}
#         for title, path in bwd.items():
#             for neighbor in bwd_links.get(title, []):
#                 if neighbor in bwd_visited:
#                     continue
#                 new_path = path + [neighbor]   # stored as [target, …, neighbor]
#                 if neighbor in fwd_visited:
#                     fwd_path = fwd.get(neighbor) or next_fwd.get(neighbor, [neighbor])
#                     return fwd_path + list(reversed(new_path[:-1]))
#                 bwd_visited.add(neighbor)
#                 next_bwd[neighbor] = new_path
#         bwd = next_bwd
#
#     return None  # Exceeded depth limit
#

# ─── Stylesheet ────────────────────────────────────────────────────────────────

DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #0f0f0f;
    color: #e8e8e8;
    font-family: 'Courier New', monospace;
}

#toolbar {
    background-color: #0f0f0f;
    border-bottom: 2px solid #00ff88;
    padding: 6px 12px;
    min-height: 54px;
    max-height: 54px;
}

#gameLabel {
    color: #00ff88;
    font-size: 15px;
    font-weight: bold;
    letter-spacing: 2px;
}

#startLabel {
    color: #888;
    font-size: 11px;
    letter-spacing: 1px;
}

#articleLabel {
    color: #fff;
    font-size: 13px;
    font-weight: bold;
}

#targetLabel {
    color: #ff4444;
    font-size: 13px;
    font-weight: bold;
}

#arrowLabel {
    color: #00ff88;
    font-size: 16px;
    font-weight: bold;
}

#clicksLabel {
    color: #ffcc00;
    font-size: 13px;
    font-weight: bold;
    min-width: 80px;
}

#timerLabel {
    color: #00aaff;
    font-size: 13px;
    font-weight: bold;
    min-width: 70px;
}

QPushButton {
    background-color: #1a1a1a;
    color: #00ff88;
    border: 1px solid #00ff88;
    border-radius: 3px;
    padding: 5px 12px;
    font-family: 'Courier New', monospace;
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 1px;
    min-height: 26px;
}

QPushButton:hover {
    background-color: #00ff88;
    color: #0f0f0f;
}

QPushButton:pressed {
    background-color: #00cc66;
    color: #0f0f0f;
}

QPushButton#dangerBtn {
    color: #ff4444;
    border-color: #ff4444;
}

QPushButton#dangerBtn:hover {
    background-color: #ff4444;
    color: #0f0f0f;
}

QPushButton:disabled {
    color: #333;
    border-color: #333;
    background-color: #0f0f0f;
}

QDialog {
    background-color: #111;
    color: #e8e8e8;
    font-family: 'Courier New', monospace;
}

QLineEdit {
    background-color: #1a1a1a;
    color: #e8e8e8;
    border: 1px solid #444;
    border-radius: 3px;
    padding: 6px 10px;
    font-family: 'Courier New', monospace;
    font-size: 13px;
}

QLineEdit:focus {
    border-color: #00ff88;
}

QListWidget {
    background-color: #111;
    color: #e8e8e8;
    border: 1px solid #333;
    border-radius: 3px;
    font-family: 'Courier New', monospace;
    font-size: 12px;
}

QListWidget::item {
    padding: 4px 8px;
    border-bottom: 1px solid #1a1a1a;
}

QListWidget::item:selected {
    background-color: #00ff8822;
    color: #00ff88;
}

QLabel {
    color: #e8e8e8;
}

QDialogButtonBox QPushButton {
    min-width: 80px;
}

#separator {
    color: #333;
    font-size: 18px;
}
"""


# ─── BFS Worker Thread ────────────────────────────────────────────────────────
#
# class BFSWorker(QThread):
#     """Runs the bidirectional BFS in a background thread."""
#     finished = pyqtSignal(object)   # emits list[str] or None
#
#     def __init__(self, start, target):
#         super().__init__()
#         self.start_article = start
#         self.target_article = target
#         self._stop = threading.Event()
#
#     def run(self):
#         result = find_shortest_path(self.start_article, self.target_article, self._stop)
#         if not self._stop.is_set():
#             self.finished.emit(result)
#
#     def cancel(self):
#         self._stop.set()
#

# ─── Setup Dialog ───────────────────────────────────────────────────────────────

PRESET_RACES = [
    ("Cleopatra", "Kevin Bacon"),
    ("Banana", "World War II"),
    ("Pluto (planet)", "Pizza"),
    ("Genghis Khan", "Taylor Swift"),
    ("Photosynthesis", "Arnold Schwarzenegger"),
    ("Ancient Rome", "Internet"),
    ("Black hole", "William Shakespeare"),
    ("Octopus", "United States Constitution"),
]

class SetupDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("WIKIPEDIA RACE — NEW GAME")
        self.setMinimumWidth(480)
        self.setMinimumHeight(400)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        title = QLabel("[ NEW RACE ]")
        title.setStyleSheet("color: #00ff88; font-size: 18px; font-weight: bold; letter-spacing: 3px;")
        layout.addWidget(title)

        # Start
        layout.addWidget(QLabel("START ARTICLE:"))
        self.start_input = QLineEdit()
        self.start_input.setPlaceholderText("e.g. Cleopatra")
        layout.addWidget(self.start_input)

        # Target
        layout.addWidget(QLabel("TARGET ARTICLE:"))
        self.target_input = QLineEdit()
        self.target_input.setPlaceholderText("e.g. Kevin Bacon")
        layout.addWidget(self.target_input)

        # Presets
        preset_label = QLabel("— OR PICK A PRESET —")
        preset_label.setStyleSheet("color: #555; font-size: 11px; letter-spacing: 2px;")
        layout.addWidget(preset_label)

        self.preset_list = QListWidget()
        self.preset_list.setMaximumHeight(140)
        for start, end in PRESET_RACES:
            item = QListWidgetItem(f"{start}  →  {end}")
            item.setData(Qt.ItemDataRole.UserRole, (start, end))
            self.preset_list.addItem(item)
        self.preset_list.itemClicked.connect(self._on_preset_click)
        layout.addWidget(self.preset_list)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_preset_click(self, item):
        start, end = item.data(Qt.ItemDataRole.UserRole)
        self.start_input.setText(start)
        self.target_input.setText(end)

    def get_values(self):
        return self.start_input.text().strip(), self.target_input.text().strip()


# ─── History Dialog ─────────────────────────────────────────────────────────────

class HistoryDialog(QDialog):
    def __init__(self, path, elapsed, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PATH HISTORY")
        self.setMinimumWidth(400)
        self._build_ui(path, elapsed)

    def _build_ui(self, path, elapsed):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        title = QLabel(f"[ YOUR PATH — {len(path)} clicks — {elapsed:.1f}s ]")
        title.setStyleSheet("color: #00ff88; font-size: 14px; font-weight: bold;")
        layout.addWidget(title)

        lst = QListWidget()
        for i, article in enumerate(path):
            item = QListWidgetItem(f"  {i:02d}.  {article}")
            if i == 0:
                item.setForeground(QColor("#00ff88"))
            elif i == len(path) - 1:
                item.setForeground(QColor("#ff4444"))
            lst.addItem(item)
        layout.addWidget(lst)

        btn = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        btn.accepted.connect(self.accept)
        layout.addWidget(btn)


# ─── Result Dialog (win/give-up with best path) ──────────────────────────────

class ResultDialog(QDialog):
    """Shows final stats, your path, and the optimal path side by side."""

    play_again = pyqtSignal()

    def __init__(self, won: bool, user_path: list[str], elapsed: float,
                 start: str, target: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YOU WIN!" if won else "GAME OVER")
        self.setMinimumWidth(640)
        self.setMinimumHeight(520)
        self._won = won
        self._user_path = user_path
        self._best_path = None
        self._worker = None
        self._build_ui(won, user_path, elapsed, start, target)
        # self._start_bfs(start, target)

    def _build_ui(self, won, user_path, elapsed, start, target):
        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(24, 24, 24, 24)

        # Header
        if won:
            header_text = "🏆  TARGET REACHED"
            header_color = "#00ff88"
        else:
            header_text = "💀  BETTER LUCK NEXT TIME"
            header_color = "#ff4444"

        header = QLabel(header_text)
        header.setStyleSheet(f"color: {header_color}; font-size: 18px; font-weight: bold; letter-spacing: 2px;")
        layout.addWidget(header)

        # Stats row
        clicks = len(user_path) - 1
        mins, secs = divmod(int(elapsed), 60)
        stats = QLabel(f"Clicks: {clicks}    Time: {mins}:{secs:02d}    Articles visited: {len(user_path)}")
        stats.setStyleSheet("color: #888; font-size: 12px;")
        layout.addWidget(stats)

        # Divider
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #333;")
        layout.addWidget(line)

        # Two-column path display
        cols_widget = QWidget()
        cols = QHBoxLayout(cols_widget)
        cols.setSpacing(16)

        # Your path
        left = QVBoxLayout()
        your_lbl = QLabel("YOUR PATH")
        your_lbl.setStyleSheet("color: #ffcc00; font-size: 12px; font-weight: bold; letter-spacing: 2px;")
        left.addWidget(your_lbl)
        self.your_list = QListWidget()
        self._populate_path_list(self.your_list, user_path, "#ffcc00")
        left.addWidget(self.your_list)
        cols.addLayout(left)

        # Best path
        right = QVBoxLayout()
        best_hdr = QHBoxLayout()
        best_lbl = QLabel("OPTIMAL PATH")
        best_lbl.setStyleSheet("color: #00aaff; font-size: 12px; font-weight: bold; letter-spacing: 2px;")
        best_hdr.addWidget(best_lbl)
        self.searching_lbl = QLabel("  ⟳ searching…")
        self.searching_lbl.setStyleSheet("color: #555; font-size: 11px;")
        best_hdr.addWidget(self.searching_lbl)
        best_hdr.addStretch()
        right.addLayout(best_hdr)
        self.best_list = QListWidget()
        self.best_list.addItem(QListWidgetItem("  Calculating…"))
        right.addWidget(self.best_list)
        cols.addLayout(right)

        layout.addWidget(cols_widget)

        # Score comparison (populated once BFS finishes)
        self.score_lbl = QLabel("")
        self.score_lbl.setStyleSheet("color: #888; font-size: 12px; text-align: center;")
        self.score_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.score_lbl)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_again = QPushButton("PLAY AGAIN")
        self.btn_again.clicked.connect(self._on_play_again)
        btn_row.addWidget(self.btn_again)
        btn_close = QPushButton("CLOSE")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

    def _populate_path_list(self, lst: QListWidget, path: list[str], accent: str):
        lst.clear()
        for i, article in enumerate(path):
            item = QListWidgetItem(f"  {i:02d}.  {article}")
            if i == 0:
                item.setForeground(QColor("#00ff88"))
            elif i == len(path) - 1:
                item.setForeground(QColor("#ff4444"))
            else:
                item.setForeground(QColor(accent))
            lst.addItem(item)

    # def _start_bfs(self, start, target):
    #     self._worker = BFSWorker(start, target)
    #     self._worker.finished.connect(self._on_bfs_done)
    #     self._worker.start()
    #
    # def _on_bfs_done(self, path):
    #     self._best_path = path
    #     self.searching_lbl.setText("")
    #     self.best_list.clear()
    #
    #     user_clicks = len(self._user_path) - 1
    #
    #     if path is None:
    #         item = QListWidgetItem("  Could not find path within search depth.")
    #         item.setForeground(QColor("#555"))
    #         self.best_list.addItem(item)
    #         self.score_lbl.setText("Optimal path search exceeded depth limit.")
    #     else:
    #         self._populate_path_list(self.best_list, path, "#00aaff")
    #         optimal_clicks = len(path) - 1
    #         diff = user_clicks - optimal_clicks
    #         if diff == 0:
    #             verdict = "🎯 Perfect! You matched the optimal path!"
    #             color = "#00ff88"
    #         elif diff == 1:
    #             verdict = f"⭐ So close! Just 1 click over optimal."
    #             color = "#ffcc00"
    #         elif diff > 0:
    #             verdict = f"📊 You used {diff} more clicks than optimal ({optimal_clicks} clicks)."
    #             color = "#ff8844"
    #         else:
    #             # user_clicks < optimal_clicks — shouldn't happen but handle gracefully
    #             verdict = f"🤔 Interesting route! ({user_clicks} vs {optimal_clicks} optimal)"
    #             color = "#aaaaff"
    #         self.score_lbl.setText(verdict)
    #         self.score_lbl.setStyleSheet(f"color: {color}; font-size: 13px; font-weight: bold;")

    def _on_play_again(self):
        self._play_again_requested = True
        self.accept()

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(2000)
        self._worker = None
        super().closeEvent(event)
        # Emit only after dialog is fully torn down
        if getattr(self, "_play_again_requested", False):
            self.play_again.emit()



class WikiPage(QWebEnginePage):
    """Intercepts navigation to detect Wikipedia link clicks."""
    wiki_link_clicked = pyqtSignal(str, str)  # (title, url)

    def __init__(self, profile, parent=None):
        super().__init__(profile, parent)

    def acceptNavigationRequest(self, url, nav_type, is_main_frame):
        if not is_main_frame:
            return True  # Allow sub-frame loads (images, etc.)

        url_str = url.toString()

        # Only intercept clicks (not form submissions or redirects we initiated)
        if nav_type == QWebEnginePage.NavigationType.NavigationTypeLinkClicked:
            # Must be a Wikipedia /wiki/ article link
            if "wikipedia.org/wiki/" in url_str:
                # Exclude special pages
                path = url.path()
                if any(path.startswith(f"/wiki/{ns}:") for ns in [
                    "Special", "Help", "Wikipedia", "Talk", "User",
                    "Template", "Category", "Portal", "File", "MOS"
                ]):
                    return True  # Let special pages through normally
                title = urllib.parse.unquote(path.replace("/wiki/", "").replace("_", " "))
                self.wiki_link_clicked.emit(title, url_str)
                return True  # Allow navigation

        return True


# ─── Main Window ─────────────────────────────────────────────────────────────────

class WikiRace(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wikipedia Race")
        self.resize(1200, 850)

        self.start_article = ""
        self.target_article = ""
        self.click_path = []
        self.click_count = 0
        self.start_time = None
        self.game_active = False

        self._build_ui()
        self._setup_timer()
        self.setStyleSheet(DARK_STYLE)

        # Prompt setup on launch
        QTimer.singleShot(300, self._new_game)

    # ── UI Construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Toolbar
        toolbar = self._build_toolbar()
        main_layout.addWidget(toolbar)

        # Web view
        profile = QWebEngineProfile("WikiRace", self)
        self.page = WikiPage(profile, self)
        self.page.wiki_link_clicked.connect(self._on_wiki_link_clicked)

        self.web = QWebEngineView()
        self.web.setPage(self.page)
        self.web.loadFinished.connect(self._on_load_finished)
        main_layout.addWidget(self.web)

    def _build_toolbar(self):
        toolbar = QWidget()
        toolbar.setObjectName("toolbar")
        toolbar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        row = QHBoxLayout(toolbar)
        row.setContentsMargins(12, 4, 12, 4)
        row.setSpacing(10)

        # Game title
        game_lbl = QLabel("WIKI RACE")
        game_lbl.setObjectName("gameLabel")
        row.addWidget(game_lbl)

        sep1 = QLabel("│")
        sep1.setObjectName("separator")
        row.addWidget(sep1)

        # Route display
        route_widget = QWidget()
        route_layout = QHBoxLayout(route_widget)
        route_layout.setContentsMargins(0, 0, 0, 0)
        route_layout.setSpacing(6)

        start_col = QVBoxLayout()
        start_col.setSpacing(1)
        start_lbl = QLabel("FROM")
        start_lbl.setObjectName("startLabel")
        self.start_display = QLabel("—")
        self.start_display.setObjectName("articleLabel")
        start_col.addWidget(start_lbl)
        start_col.addWidget(self.start_display)
        route_layout.addLayout(start_col)

        arrow = QLabel("→")
        arrow.setObjectName("arrowLabel")
        route_layout.addWidget(arrow)

        target_col = QVBoxLayout()
        target_col.setSpacing(1)
        target_lbl = QLabel("TARGET")
        target_lbl.setObjectName("startLabel")
        self.target_display = QLabel("—")
        self.target_display.setObjectName("targetLabel")
        target_col.addWidget(target_lbl)
        target_col.addWidget(self.target_display)
        route_layout.addLayout(target_col)

        row.addWidget(route_widget)
        row.addStretch()

        # Stats
        sep2 = QLabel("│")
        sep2.setObjectName("separator")
        row.addWidget(sep2)

        clicks_col = QVBoxLayout()
        clicks_col.setSpacing(1)
        clicks_hdr = QLabel("CLICKS")
        clicks_hdr.setObjectName("startLabel")
        self.clicks_display = QLabel("0")
        self.clicks_display.setObjectName("clicksLabel")
        clicks_col.addWidget(clicks_hdr)
        clicks_col.addWidget(self.clicks_display)
        row.addLayout(clicks_col)

        sep3 = QLabel("│")
        sep3.setObjectName("separator")
        row.addWidget(sep3)

        timer_col = QVBoxLayout()
        timer_col.setSpacing(1)
        timer_hdr = QLabel("TIME")
        timer_hdr.setObjectName("startLabel")
        self.timer_display = QLabel("0:00")
        self.timer_display.setObjectName("timerLabel")
        timer_col.addWidget(timer_hdr)
        timer_col.addWidget(self.timer_display)
        row.addLayout(timer_col)

        sep4 = QLabel("│")
        sep4.setObjectName("separator")
        row.addWidget(sep4)

        # Buttons
        self.btn_new = QPushButton("NEW GAME")
        self.btn_new.clicked.connect(self._new_game)
        row.addWidget(self.btn_new)

        self.btn_history = QPushButton("PATH")
        self.btn_history.clicked.connect(self._show_history)
        row.addWidget(self.btn_history)

        self.btn_back = QPushButton("← BACK")
        self.btn_back.clicked.connect(self._go_back)
        self.btn_back.setEnabled(False)
        row.addWidget(self.btn_back)

        self.btn_give_up = QPushButton("GIVE UP")
        self.btn_give_up.setObjectName("dangerBtn")
        self.btn_give_up.clicked.connect(self._give_up)
        self.btn_give_up.setEnabled(False)
        row.addWidget(self.btn_give_up)

        return toolbar

    def _setup_timer(self):
        self.timer = QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._tick_timer)

    # ── Game Logic ───────────────────────────────────────────────────────────

    def _new_game(self):
        dlg = SetupDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        start, target = dlg.get_values()
        if not start or not target:
            QMessageBox.warning(self, "Missing Input", "Please enter both a start and target article.")
            return

        self.timer.stop()
        self.start_article = start
        self.target_article = target
        self.click_path = [start]
        self.click_count = 0
        self.game_active = True
        self.start_time = time.time()

        self.start_display.setText(start)
        self.target_display.setText(target)
        self.clicks_display.setText("0")
        self.timer_display.setText("0:00")

        self.btn_give_up.setEnabled(True)
        self.btn_back.setEnabled(False)

        self.timer.start()
        self._load_article(start)

    def _load_article(self, title):
        encoded = urllib.parse.quote(title.replace(" ", "_"))
        url = f"https://en.wikipedia.org/wiki/{encoded}"
        self.web.load(QUrl(url))

    def _on_wiki_link_clicked(self, title, url):
        if not self.game_active:
            return

        self.click_count += 1
        self.click_path.append(title)
        self.clicks_display.setText(str(self.click_count))
        self.btn_back.setEnabled(len(self.click_path) > 1)

        # Check win condition
        def normalize(s):
            return s.lower().replace("_", " ").strip()

        if normalize(title) == normalize(self.target_article):
            self._win()

    def _on_load_finished(self, ok):
        if ok:
            self._inject_highlight_css()

    def _inject_highlight_css(self):
        """Inject CSS to subtly highlight the toolbar is separate from Wikipedia."""
        js = """
        (function() {
            // Remove Wikipedia's own top bar to reduce confusion
            var siteNotice = document.getElementById('siteNotice');
            if (siteNotice) siteNotice.style.display = 'none';

            // Scroll to top
            window.scrollTo(0, 0);
        })();
        """
        self.page.runJavaScript(js)

    def _go_back(self):
        if len(self.click_path) <= 1:
            return
        # Remove current from path
        self.click_path.pop()
        self.click_count = max(0, self.click_count - 1)  # optional: penalize or not
        self.clicks_display.setText(str(self.click_count))
        self.btn_back.setEnabled(len(self.click_path) > 1)

        prev = self.click_path[-1]
        self._load_article(prev)

    def _tick_timer(self):
        if self.start_time:
            elapsed = int(time.time() - self.start_time)
            mins, secs = divmod(elapsed, 60)
            self.timer_display.setText(f"{mins}:{secs:02d}")

    def _elapsed(self):
        if self.start_time:
            return time.time() - self.start_time
        return 0

    def _win(self):
        self.game_active = False
        self.timer.stop()
        elapsed = self._elapsed()

        dlg = ResultDialog(
            won=True,
            user_path=self.click_path,
            elapsed=elapsed,
            start=self.start_article,
            target=self.target_article,
            parent=self,
        )
        dlg.play_again.connect(self._new_game)
        self.btn_give_up.setEnabled(False)
        dlg.exec()

    def _give_up(self):
        self.game_active = False
        self.timer.stop()
        elapsed = self._elapsed()

        dlg = ResultDialog(
            won=False,
            user_path=self.click_path,
            elapsed=elapsed,
            start=self.start_article,
            target=self.target_article,
            parent=self,
        )
        dlg.play_again.connect(self._new_game)
        self.btn_give_up.setEnabled(False)
        dlg.exec()

    def _show_history(self):
        if not self.click_path:
            return
        dlg = HistoryDialog(self.click_path, self._elapsed(), self)
        dlg.exec()


# ─── Entry Point ─────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Wikipedia Race")
    app.setStyle("Fusion")

    # Dark palette base
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#0f0f0f"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#e8e8e8"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#111111"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#1a1a1a"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#e8e8e8"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#1a1a1a"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#00ff88"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#00ff8833"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#00ff88"))
    app.setPalette(palette)

    window = WikiRace()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()