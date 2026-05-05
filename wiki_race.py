#!/usr/bin/env python3
"""
Wikipedia Race - Singleplayer
Navigate from a start article to a target article using only Wikipedia links.
Requires: pip install PyQt6 PyQt6-WebEngine numpy sentence-transformers
Also needs: hub_embeddings.npz (run the hub embedding script once to generate)
"""

import sys
import time
import threading
import urllib.request
import urllib.parse
import json
import numpy as np
from sentence_transformers import SentenceTransformer

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QLineEdit, QDialog, QDialogButtonBox,
    QListWidget, QListWidgetItem, QFrame, QMessageBox, QSizePolicy
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
from PyQt6.QtCore import QUrl, Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QColor, QPalette

# ─── NLP setup ────────────────────────────────────────────────────────────────

WIKI_API = "https://en.wikipedia.org/w/api.php"
model    = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
hubData  = np.load("hub_embeddings.npz", allow_pickle=True)

# ─── Wikipedia API helpers ────────────────────────────────────────────────────

def _api(params):
    p   = {**params, "format": "json"}
    url = f"{WIKI_API}?{urllib.parse.urlencode(p)}"
    req = urllib.request.Request(url, headers={"User-Agent": "WikiRaceGame/1.0"})
    with urllib.request.urlopen(req, timeout=12) as r:
        return json.loads(r.read())


def _resolve_redirect(title):
    try:
        data = _api({"action": "query", "titles": title, "redirects": 1})
        for page in data.get("query", {}).get("pages", {}).values():
            return page.get("title", title)
    except Exception:
        pass
    return title


def _wiki_links(titles):
    """Outgoing links for a batch of titles. Returns {title: [neighbors]}."""
    out = {}
    for i in range(0, len(titles), 50):
        chunk = titles[i:i+50]
        try:
            data = _api({
                "action": "query", "titles": "|".join(chunk),
                "prop": "links", "pllimit": 500, "plnamespace": 0,
                "redirects": 1,
            })
            for page in data.get("query", {}).get("pages", {}).values():
                t = page.get("title", "")
                out[t] = [lk["title"] for lk in page.get("links", [])]
        except Exception as e:
            print(f"  links batch failed: {e}")
    return out


def _wiki_backlinks(title):
    """Incoming links for a single title ('What links here')."""
    sources = []
    try:
        data = _api({
            "action": "query", "list": "backlinks",
            "bltitle": title, "bllimit": 500,
            "blnamespace": 0, "blredirect": 0,
        })
        sources = [bl["title"] for bl in data.get("query", {}).get("backlinks", [])]
    except Exception as e:
        print(f"  backlinks failed for '{title}': {e}")
    return sources


def get_article_text(title):
    """Fetch intro paragraph as plain text."""
    try:
        data = _api({
            "action": "query", "titles": title,
            "prop": "extracts", "explaintext": 1, "exintro": 1,
        })
        for page in data.get("query", {}).get("pages", {}).values():
            return page.get("extract", None)
    except Exception:
        pass
    return None


# ─── NLP: hub similarity ──────────────────────────────────────────────────────

def best_hub(title):
    """
    Return the hub article most semantically similar to title.
    Uses cosine similarity between sentence-transformer embeddings.
    """
    text = get_article_text(title)
    if not text:
        return None
    embedding  = model.encode(text)
    hub_titles = hubData["titles"].tolist()
    hub_matrix = hubData["matrix"]
    best_t, best_s = None, -1.0
    for ht, he in zip(hub_titles, hub_matrix):
        s = float(model.similarity(embedding, he).numpy()[0, 0])
        if s > best_s:
            best_s, best_t = s, ht
    return best_t


def get_hint(current_article, target_article):
    """
    NLP hint: compare the current article's embedding against the target's.
    Suggests which hub to navigate toward to get semantically closer.
    Returns a hint string shown in the toolbar.
    """
    try:
        current_hub = best_hub(current_article)
        target_hub  = best_hub(target_article)
        if current_hub is None or target_hub is None:
            return "No hint available"
        if current_hub == target_hub:
            return f"You're on track — stay near '{current_hub}' topics"
        return f"Hint: navigate toward '{target_hub}' topics"
    except Exception:
        return "No hint available"


# ─── Bidirectional BFS ────────────────────────────────────────────────────────

MAX_DEPTH    = 3
LINKS_PER_PG = 500


def _reconstruct(fwd_parents, bwd_parents, meet):
    # Forward half: meet → start
    fwd = []
    node = meet
    while node is not None:
        fwd.append(node)
        node = fwd_parents[node]
    fwd.reverse()
    # Backward half: meet → target (skip meet itself)
    bwd = []
    node = bwd_parents[meet]
    while node is not None:
        bwd.append(node)
        node = bwd_parents[node]
    return fwd + bwd


def find_shortest_path(start, target, stop_event):
    """
    True bidirectional BFS using:
      - Forward frontier: outgoing links  (what a player can click)
      - Backward frontier: incoming links (What links here)
    Paths meet when a forward neighbor appears in the backward visited set.
    """
    start  = _resolve_redirect(start)
    target = _resolve_redirect(target)

    if start.lower() == target.lower():
        return [start]

    fwd_visited = {start}
    bwd_visited = {target}
    fwd_parents = {start: None}
    bwd_parents = {target: None}
    fwd_frontier = [start]
    bwd_frontier = [target]

    for _ in range(MAX_DEPTH):
        if stop_event.is_set():
            return None

        # ── Forward step (outgoing links) ─────────────────────────────────
        if not fwd_frontier:
            return None
        try:
            fwd_links = _wiki_links(fwd_frontier)
        except Exception as e:
            print(f"  fwd links failed: {e}")
            return None
        if stop_event.is_set():
            return None

        next_fwd = []
        for node in fwd_frontier:
            for nb in fwd_links.get(node, []):
                if nb in fwd_visited:
                    continue
                fwd_visited.add(nb)
                fwd_parents[nb] = node
                if nb in bwd_visited:
                    return _reconstruct(fwd_parents, bwd_parents, nb)
                next_fwd.append(nb)
        fwd_frontier = next_fwd

        if stop_event.is_set():
            return None

        # ── Backward step (incoming links / What links here) ──────────────
        if not bwd_frontier:
            return None
        next_bwd = []
        for node in bwd_frontier:
            try:
                for nb in _wiki_backlinks(node):
                    if nb in bwd_visited:
                        continue
                    bwd_visited.add(nb)
                    bwd_parents[nb] = node
                    if nb in fwd_visited:
                        return _reconstruct(fwd_parents, bwd_parents, nb)
                    next_bwd.append(nb)
            except Exception as e:
                print(f"  bwd links failed for '{node}': {e}")
            if stop_event.is_set():
                return None
        bwd_frontier = next_bwd

    return None


# ─── BFS Worker thread ────────────────────────────────────────────────────────

class BFSWorker(QThread):
    finished = pyqtSignal(object)
    TIMEOUT  = 90

    def __init__(self, start, target):
        super().__init__()
        self.start_article  = start
        self.target_article = target
        self._stop = threading.Event()

    def run(self):
        timer  = threading.Timer(self.TIMEOUT, self._stop.set)
        result = None
        timer.start()
        try:
            result = find_shortest_path(
                self.start_article, self.target_article, self._stop
            )
        except Exception as e:
            print(f"[BFSWorker] {e}")
        finally:
            timer.cancel()
            self.finished.emit(result)

    def cancel(self):
        self._stop.set()


# ─── Hint Worker thread ───────────────────────────────────────────────────────

class HintWorker(QThread):
    """Computes NLP hint in background so it never blocks the UI."""
    finished = pyqtSignal(str)

    def __init__(self, current, target):
        super().__init__()
        self.current = current
        self.target  = target

    def run(self):
        try:
            hint = get_hint(self.current, self.target)
        except Exception:
            hint = "No hint available"
        self.finished.emit(hint)


# ─── Stylesheet ───────────────────────────────────────────────────────────────

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
    min-height: 60px;
    max-height: 60px;
}
#gameLabel   { color: #00ff88; font-size: 15px; font-weight: bold; letter-spacing: 2px; }
#startLabel  { color: #888;    font-size: 11px; letter-spacing: 1px; }
#articleLabel{ color: #fff;    font-size: 13px; font-weight: bold; }
#targetLabel { color: #ff4444; font-size: 13px; font-weight: bold; }
#arrowLabel  { color: #00ff88; font-size: 16px; font-weight: bold; }
#clicksLabel { color: #ffcc00; font-size: 13px; font-weight: bold; min-width: 80px; }
#timerLabel  { color: #00aaff; font-size: 13px; font-weight: bold; min-width: 70px; }
#hintLabel   { color: #aa88ff; font-size: 11px; font-style: italic; }
QPushButton {
    background-color: #1a1a1a; color: #00ff88; border: 1px solid #00ff88;
    border-radius: 3px; padding: 5px 12px; font-family: 'Courier New', monospace;
    font-size: 11px; font-weight: bold; letter-spacing: 1px; min-height: 26px;
}
QPushButton:hover   { background-color: #00ff88; color: #0f0f0f; }
QPushButton:pressed { background-color: #00cc66; color: #0f0f0f; }
QPushButton#dangerBtn       { color: #ff4444; border-color: #ff4444; }
QPushButton#dangerBtn:hover { background-color: #ff4444; color: #0f0f0f; }
QPushButton:disabled { color: #333; border-color: #333; background-color: #0f0f0f; }
QDialog   { background-color: #111; color: #e8e8e8; font-family: 'Courier New', monospace; }
QLineEdit {
    background-color: #1a1a1a; color: #e8e8e8; border: 1px solid #444;
    border-radius: 3px; padding: 6px 10px; font-size: 13px;
}
QLineEdit:focus { border-color: #00ff88; }
QListWidget {
    background-color: #111; color: #e8e8e8; border: 1px solid #333;
    border-radius: 3px; font-size: 12px;
}
QListWidget::item          { padding: 4px 8px; border-bottom: 1px solid #1a1a1a; }
QListWidget::item:selected { background-color: #00ff8822; color: #00ff88; }
QLabel { color: #e8e8e8; }
QDialogButtonBox QPushButton { min-width: 80px; }
#separator { color: #333; font-size: 18px; }
"""

# ─── Setup Dialog ─────────────────────────────────────────────────────────────

PRESET_RACES = [
    ("Cleopatra",      "Kevin Bacon"),
    ("Banana",         "World War II"),
    ("Pluto (planet)", "Pizza"),
    ("Genghis Khan",   "Taylor Swift"),
    ("Photosynthesis", "Arnold Schwarzenegger"),
    ("Ancient Rome",   "Internet"),
    ("Black hole",     "William Shakespeare"),
    ("Octopus",        "Constitution of the United States"),
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

        layout.addWidget(QLabel("START ARTICLE:"))
        self.start_input = QLineEdit()
        self.start_input.setPlaceholderText("e.g. Cleopatra")
        layout.addWidget(self.start_input)

        layout.addWidget(QLabel("TARGET ARTICLE:"))
        self.target_input = QLineEdit()
        self.target_input.setPlaceholderText("e.g. Kevin Bacon")
        layout.addWidget(self.target_input)

        preset_label = QLabel("— OR PICK A PRESET —")
        preset_label.setStyleSheet("color: #555; font-size: 11px; letter-spacing: 2px;")
        layout.addWidget(preset_label)

        self.preset_list = QListWidget()
        self.preset_list.setMaximumHeight(140)
        for s, e in PRESET_RACES:
            item = QListWidgetItem(f"{s}  →  {e}")
            item.setData(Qt.ItemDataRole.UserRole, (s, e))
            self.preset_list.addItem(item)
        self.preset_list.itemClicked.connect(self._on_preset_click)
        layout.addWidget(self.preset_list)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_preset_click(self, item):
        s, e = item.data(Qt.ItemDataRole.UserRole)
        self.start_input.setText(s)
        self.target_input.setText(e)

    def get_values(self):
        return self.start_input.text().strip(), self.target_input.text().strip()


# ─── History Dialog ───────────────────────────────────────────────────────────

class HistoryDialog(QDialog):
    def __init__(self, path, elapsed, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PATH HISTORY")
        self.setMinimumWidth(400)
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)
        title = QLabel(f"[ YOUR PATH — {len(path)} articles — {elapsed:.1f}s ]")
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


# ─── Result Dialog ────────────────────────────────────────────────────────────

class ResultDialog(QDialog):
    play_again = pyqtSignal()

    def __init__(self, won, user_path, elapsed, start, target, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YOU WIN!" if won else "GAME OVER")
        self.setMinimumWidth(640)
        self.setMinimumHeight(520)
        self._user_path = user_path
        self._worker    = None
        self._build_ui(won, user_path, elapsed)
        self._start_bfs(start, target)

    def _build_ui(self, won, user_path, elapsed):
        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(24, 24, 24, 24)

        header = QLabel("🏆  TARGET REACHED" if won else "💀  BETTER LUCK NEXT TIME")
        header.setStyleSheet(
            f"color: {'#00ff88' if won else '#ff4444'}; "
            "font-size: 18px; font-weight: bold; letter-spacing: 2px;"
        )
        layout.addWidget(header)

        clicks     = len(user_path) - 1
        mins, secs = divmod(int(elapsed), 60)
        stats = QLabel(f"Clicks: {clicks}    Time: {mins}:{secs:02d}    Articles: {len(user_path)}")
        stats.setStyleSheet("color: #888; font-size: 12px;")
        layout.addWidget(stats)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #333;")
        layout.addWidget(line)

        cols_widget = QWidget()
        cols = QHBoxLayout(cols_widget)
        cols.setSpacing(16)

        left = QVBoxLayout()
        lbl  = QLabel("YOUR PATH")
        lbl.setStyleSheet("color: #ffcc00; font-size: 12px; font-weight: bold; letter-spacing: 2px;")
        left.addWidget(lbl)
        self.your_list = QListWidget()
        self._fill_list(self.your_list, user_path, "#ffcc00")
        left.addWidget(self.your_list)
        cols.addLayout(left)

        right    = QVBoxLayout()
        best_hdr = QHBoxLayout()
        blbl     = QLabel("OPTIMAL PATH")
        blbl.setStyleSheet("color: #00aaff; font-size: 12px; font-weight: bold; letter-spacing: 2px;")
        best_hdr.addWidget(blbl)
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

        self.score_lbl = QLabel("")
        self.score_lbl.setStyleSheet("color: #888; font-size: 12px;")
        self.score_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.score_lbl)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_again = QPushButton("PLAY AGAIN")
        btn_again.clicked.connect(self._on_play_again)
        btn_row.addWidget(btn_again)
        btn_close = QPushButton("CLOSE")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

    def _fill_list(self, lst, path, accent):
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

    def _start_bfs(self, start, target):
        self._worker = BFSWorker(start, target)
        self._worker.finished.connect(self._on_bfs_done)
        self._worker.start()

    def _on_bfs_done(self, path):
        self.searching_lbl.setText("")
        self.best_list.clear()
        user_clicks = len(self._user_path) - 1

        if path is None:
            item = QListWidgetItem("  Could not find path within search depth.")
            item.setForeground(QColor("#555"))
            self.best_list.addItem(item)
            self.score_lbl.setText("Search exceeded depth limit.")
            self.score_lbl.setStyleSheet("color: #555; font-size: 12px;")
        else:
            self._fill_list(self.best_list, path, "#00aaff")
            optimal = len(path) - 1
            diff    = user_clicks - optimal
            if diff == 0:
                verdict, color = "🎯 Perfect! You matched the optimal path!", "#00ff88"
            elif diff == 1:
                verdict, color = "⭐ So close! Just 1 click over optimal.", "#ffcc00"
            elif diff > 0:
                verdict, color = f"📊 You used {diff} more clicks than optimal ({optimal} clicks).", "#ff8844"
            else:
                verdict, color = f"🤔 Interesting route! ({user_clicks} vs {optimal} optimal)", "#aaaaff"
            self.score_lbl.setText(verdict)
            self.score_lbl.setStyleSheet(f"color: {color}; font-size: 13px; font-weight: bold;")

    def _on_play_again(self):
        self._play_again_requested = True
        self.accept()

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(2000)
        self._worker = None
        super().closeEvent(event)
        if getattr(self, "_play_again_requested", False):
            self.play_again.emit()


# ─── Web Page ─────────────────────────────────────────────────────────────────

class WikiPage(QWebEnginePage):
    wiki_link_clicked = pyqtSignal(str, str)

    def __init__(self, profile, parent=None):
        super().__init__(profile, parent)

    def acceptNavigationRequest(self, url, nav_type, is_main_frame):
        if not is_main_frame:
            return True
        if nav_type == QWebEnginePage.NavigationType.NavigationTypeLinkClicked:
            if "wikipedia.org/wiki/" in url.toString():
                path = url.path()
                if any(path.startswith(f"/wiki/{ns}:") for ns in [
                    "Special", "Help", "Wikipedia", "Talk", "User",
                    "Template", "Category", "Portal", "File", "MOS"
                ]):
                    return True
                title = urllib.parse.unquote(
                    path.replace("/wiki/", "").replace("_", " ")
                )
                self.wiki_link_clicked.emit(title, url.toString())
        return True


# ─── Main Window ──────────────────────────────────────────────────────────────

class WikiRace(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wikipedia Race")
        self.resize(1200, 850)
        self.start_article  = ""
        self.target_article = ""
        self.click_path     = []
        self.click_count    = 0
        self.start_time     = None
        self.game_active    = False
        self._hint_worker   = None
        self._build_ui()
        self._setup_timer()
        self.setStyleSheet(DARK_STYLE)
        QTimer.singleShot(300, self._new_game)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout  = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_toolbar())

        profile = QWebEngineProfile("WikiRace", self)
        self.page = WikiPage(profile, self)
        self.page.wiki_link_clicked.connect(self._on_wiki_link_clicked)
        self.web = QWebEngineView()
        self.web.setPage(self.page)
        self.web.loadFinished.connect(self._on_load_finished)
        layout.addWidget(self.web)

    def _build_toolbar(self):
        toolbar = QWidget()
        toolbar.setObjectName("toolbar")
        toolbar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        outer = QVBoxLayout(toolbar)
        outer.setContentsMargins(12, 4, 12, 2)
        outer.setSpacing(2)

        # Top row
        row = QHBoxLayout()
        row.setSpacing(10)

        lbl = QLabel("WIKI RACE")
        lbl.setObjectName("gameLabel")
        row.addWidget(lbl)
        row.addWidget(self._sep())

        route = QWidget()
        rl    = QHBoxLayout(route)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(6)

        sc = QVBoxLayout()
        sc.setSpacing(1)
        sc.addWidget(self._small("FROM"))
        self.start_display = QLabel("—")
        self.start_display.setObjectName("articleLabel")
        sc.addWidget(self.start_display)
        rl.addLayout(sc)

        arrow = QLabel("→")
        arrow.setObjectName("arrowLabel")
        rl.addWidget(arrow)

        tc = QVBoxLayout()
        tc.setSpacing(1)
        tc.addWidget(self._small("TARGET"))
        self.target_display = QLabel("—")
        self.target_display.setObjectName("targetLabel")
        tc.addWidget(self.target_display)
        rl.addLayout(tc)

        row.addWidget(route)
        row.addStretch()
        row.addWidget(self._sep())

        cc = QVBoxLayout()
        cc.setSpacing(1)
        cc.addWidget(self._small("CLICKS"))
        self.clicks_display = QLabel("0")
        self.clicks_display.setObjectName("clicksLabel")
        cc.addWidget(self.clicks_display)
        row.addLayout(cc)

        row.addWidget(self._sep())

        tc2 = QVBoxLayout()
        tc2.setSpacing(1)
        tc2.addWidget(self._small("TIME"))
        self.timer_display = QLabel("0:00")
        self.timer_display.setObjectName("timerLabel")
        tc2.addWidget(self.timer_display)
        row.addLayout(tc2)

        row.addWidget(self._sep())

        self.btn_new = QPushButton("NEW GAME")
        self.btn_new.clicked.connect(self._new_game)
        row.addWidget(self.btn_new)

        self.btn_history = QPushButton("PATH")
        self.btn_history.clicked.connect(self._show_history)
        row.addWidget(self.btn_history)

        self.btn_hint = QPushButton("💡 HINT")
        self.btn_hint.clicked.connect(self._request_hint)
        self.btn_hint.setEnabled(False)
        row.addWidget(self.btn_hint)

        self.btn_back = QPushButton("← BACK")
        self.btn_back.clicked.connect(self._go_back)
        self.btn_back.setEnabled(False)
        row.addWidget(self.btn_back)

        self.btn_give_up = QPushButton("GIVE UP")
        self.btn_give_up.setObjectName("dangerBtn")
        self.btn_give_up.clicked.connect(self._give_up)
        self.btn_give_up.setEnabled(False)
        row.addWidget(self.btn_give_up)

        outer.addLayout(row)

        # Hint row
        self.hint_display = QLabel("")
        self.hint_display.setObjectName("hintLabel")
        self.hint_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self.hint_display)

        return toolbar

    def _sep(self):
        s = QLabel("│")
        s.setObjectName("separator")
        return s

    def _small(self, text):
        l = QLabel(text)
        l.setObjectName("startLabel")
        return l

    def _setup_timer(self):
        self.timer = QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._tick_timer)

    # ── Game logic ────────────────────────────────────────────────────────

    def _new_game(self):
        dlg = SetupDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        start, target = dlg.get_values()
        if not start or not target:
            QMessageBox.warning(self, "Missing Input",
                                "Please enter both a start and target article.")
            return
        self.timer.stop()
        self.start_article  = start
        self.target_article = target
        self.click_path     = [start]
        self.click_count    = 0
        self.game_active    = True
        self.start_time     = time.time()
        self.start_display.setText(start)
        self.target_display.setText(target)
        self.clicks_display.setText("0")
        self.timer_display.setText("0:00")
        self.hint_display.setText("")
        self.btn_give_up.setEnabled(True)
        self.btn_hint.setEnabled(True)
        self.btn_back.setEnabled(False)
        self.timer.start()
        self._load_article(start)

    def _load_article(self, title):
        encoded = urllib.parse.quote(title.replace(" ", "_"))
        self.web.load(QUrl(f"https://en.wikipedia.org/wiki/{encoded}"))

    def _on_wiki_link_clicked(self, title, url):
        if not self.game_active:
            return
        self.click_count += 1
        self.click_path.append(title)
        self.clicks_display.setText(str(self.click_count))
        self.btn_back.setEnabled(len(self.click_path) > 1)
        self.hint_display.setText("")   # clear hint on each click
        if self._normalize(title) == self._normalize(self.target_article):
            self._win()

    def _on_load_finished(self, ok):
        if ok:
            self.page.runJavaScript("""
                (function() {
                    var n = document.getElementById('siteNotice');
                    if (n) n.style.display = 'none';
                    window.scrollTo(0, 0);
                })();
            """)

    def _normalize(self, s):
        return s.lower().replace("_", " ").strip()

    def _go_back(self):
        if len(self.click_path) <= 1:
            return
        self.click_path.pop()
        self.click_count = max(0, self.click_count - 1)
        self.clicks_display.setText(str(self.click_count))
        self.btn_back.setEnabled(len(self.click_path) > 1)
        self.hint_display.setText("")
        self._load_article(self.click_path[-1])

    def _tick_timer(self):
        if self.start_time:
            e = int(time.time() - self.start_time)
            self.timer_display.setText(f"{e // 60}:{e % 60:02d}")

    def _elapsed(self):
        return time.time() - self.start_time if self.start_time else 0.0

    def _request_hint(self):
        """
        NLP hint: compares the semantic hub of the current article against
        the hub of the target. Runs in a background thread so the UI
        stays responsive while embeddings are computed.
        """
        if not self.game_active:
            return
        current = self.click_path[-1]
        self.hint_display.setText("💡 computing hint…")
        self.btn_hint.setEnabled(False)

        self._hint_worker = HintWorker(current, self.target_article)
        self._hint_worker.finished.connect(self._on_hint_done)
        self._hint_worker.start()

    def _on_hint_done(self, hint):
        self.hint_display.setText(f"💡 {hint}")
        self.btn_hint.setEnabled(self.game_active)

    def _win(self):
        if not self.game_active:
            return
        self.game_active = False
        self.timer.stop()
        self.btn_give_up.setEnabled(False)
        self.btn_hint.setEnabled(False)
        dlg = ResultDialog(True, self.click_path, self._elapsed(),
                           self.start_article, self.target_article, self)
        dlg.play_again.connect(self._new_game)
        dlg.exec()

    def _give_up(self):
        self.game_active = False
        self.timer.stop()
        self.btn_give_up.setEnabled(False)
        self.btn_hint.setEnabled(False)
        dlg = ResultDialog(False, self.click_path, self._elapsed(),
                           self.start_article, self.target_article, self)
        dlg.play_again.connect(self._new_game)
        dlg.exec()

    def _show_history(self):
        if self.click_path:
            HistoryDialog(self.click_path, self._elapsed(), self).exec()


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Wikipedia Race")
    app.setStyle("Fusion")

    p = QPalette()
    p.setColor(QPalette.ColorRole.Window,          QColor("#0f0f0f"))
    p.setColor(QPalette.ColorRole.WindowText,      QColor("#e8e8e8"))
    p.setColor(QPalette.ColorRole.Base,            QColor("#111111"))
    p.setColor(QPalette.ColorRole.AlternateBase,   QColor("#1a1a1a"))
    p.setColor(QPalette.ColorRole.Text,            QColor("#e8e8e8"))
    p.setColor(QPalette.ColorRole.Button,          QColor("#1a1a1a"))
    p.setColor(QPalette.ColorRole.ButtonText,      QColor("#00ff88"))
    p.setColor(QPalette.ColorRole.Highlight,       QColor("#00ff8833"))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor("#00ff88"))
    app.setPalette(p)

    w = WikiRace()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
