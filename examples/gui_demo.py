#!/usr/bin/env python3
"""T9 GUI — authentic predictive phone-style text input powered by ai-t9.

Run:
    # Use pre-built data files (recommended — instant startup):
    python examples/gui_demo.py \\
        --vocab data/vocab.json --dict data/dict.json \\
        --model data/model.npz --ngram data/bigram.json

    # Fall back to NLTK (downloads ~15 MB on first run):
    python examples/gui_demo.py

Keyboard shortcuts:
    2-9         T9 digit input
    0 / Space   Confirm word + space
    Backspace   Delete last digit (or undo last word if buffer empty)
    Tab / #     Cycle through candidates
    1 / .       Punctuation cycling
    M           Toggle T9 ↔ ABC (multi-tap) mode
    Escape      Clear all
    Enter       Confirm word (same as Space)
    Ctrl+C      Copy composed text to clipboard
"""

from __future__ import annotations

import argparse
import sys

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QGuiApplication, QKeySequence, QPalette, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

# ─── Colour palette ──────────────────────────────────────────────────────────

C: dict[str, str] = {
    "phone_bg":         "#111318",
    "display_bg":       "#0d1a0f",
    "display_text":     "#98d898",
    "display_active":   "#00ff88",
    "display_dim":      "#3a6a3a",
    "bar_bg":           "#0a0d0f",
    "bar_border":       "#1e2226",
    "cand_text":        "#7ec8e3",
    "cand_sel_bg":      "#0c3650",
    "cand_sel_text":    "#ffffff",
    "key_bg":           "#1e2230",
    "key_top":          "#282c40",
    "key_border":       "#3a3e52",
    "key_text":         "#dde0f0",
    "key_letters":      "#7880a8",
    "key_hover":        "#2e3860",
    "key_press":        "#0d1530",
    "key_special_bg":   "#1a2a1a",
    "key_special_text": "#55bb77",
    "status_text":      "#565a6e",
    "ctx_chip_bg":      "#162016",
    "ctx_chip_text":    "#55aa66",
    "mode_T9":          "#00cc77",
    "mode_ABC":         "#ffaa33",
    "punct_text":       "#cc9944",
}

# ─── T9 / phone keypad data ───────────────────────────────────────────────────

# Letters printed on each key
KEY_SUB: dict[str, str] = {
    "1": ".,?!",
    "2": "ABC",
    "3": "DEF",
    "4": "GHI",
    "5": "JKL",
    "6": "MNO",
    "7": "PQRS",
    "8": "TUV",
    "9": "WXYZ",
    "*": "BACK",
    "0": "SPACE",
    "#": "NEXT",
}

# Physical keypad layout (row × col)
KEYPAD_ROWS: list[list[str]] = [
    ["1", "2", "3"],
    ["4", "5", "6"],
    ["7", "8", "9"],
    ["*", "0", "#"],
]

# Multi-tap letters per key
MULTITAP: dict[str, str] = {
    "1": ".,?!-':\"()",
    "2": "abc",
    "3": "def",
    "4": "ghi",
    "5": "jkl",
    "6": "mno",
    "7": "pqrs",
    "8": "tuv",
    "9": "wxyz",
}

PUNCT_CYCLE = [".", ",", "!", "?", ";", ":", "-", "…", "'", "\""]

# ─── Custom widgets ────────────────────────────────────────────────────────────


class T9KeyButton(QPushButton):
    """A phone keypad button.

    Displays a large digit (or symbol) on top and a small letter-set below.
    Emits ``long_pressed`` after the button is held for 650 ms.
    """

    long_pressed = pyqtSignal(str)

    # Special icon labels
    _TOP_OVERRIDES = {"*": "★", "0": "0", "#": "#"}

    def __init__(self, key: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._key = key
        self._held = False

        self._hold_timer = QTimer(self)
        self._hold_timer.setSingleShot(True)
        self._hold_timer.setInterval(650)
        self._hold_timer.timeout.connect(lambda: self.long_pressed.emit(self._key))

        self._build()

    # -- construction --------------------------------------------------

    def _build(self) -> None:
        top_text = self._TOP_OVERRIDES.get(self._key, self._key)
        sub_text = KEY_SUB.get(self._key, "")

        top = QLabel(top_text, self)
        top.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        top.setStyleSheet(
            f"color:{C['key_text']};background:transparent;"
            f"font-size:22px;font-weight:bold;"
        )

        sub = QLabel(sub_text, self)
        sub.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        is_special = self._key in ("*", "0", "#", "1")
        sub_color = C["key_special_text"] if is_special else C["key_letters"]
        sub.setStyleSheet(
            f"color:{sub_color};background:transparent;font-size:8px;"
        )

        vlay = QVBoxLayout(self)
        vlay.setContentsMargins(2, 6, 2, 6)
        vlay.setSpacing(1)
        vlay.addWidget(top)
        vlay.addWidget(sub)

        self.setMinimumSize(72, 64)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._paint_normal()

    # -- style helpers -------------------------------------------------

    def _paint_normal(self) -> None:
        self.setStyleSheet(
            f"QPushButton{{"
            f"background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            f"stop:0 {C['key_top']},stop:1 {C['key_bg']});"
            f"border:1px solid {C['key_border']};"
            f"border-radius:8px;}}"
            f"QPushButton:hover{{background:{C['key_hover']};}}"
        )

    def _paint_pressed(self) -> None:
        self.setStyleSheet(
            f"QPushButton{{"
            f"background:{C['key_press']};"
            f"border:2px solid {C['display_active']};"
            f"border-radius:8px;}}"
        )

    # -- events --------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        self._held = True
        self._hold_timer.start()
        self._paint_pressed()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        self._hold_timer.stop()
        self._held = False
        self._paint_normal()
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        if self._held:
            self._hold_timer.stop()
            self._held = False
            self._paint_normal()
        super().leaveEvent(event)


class CandidateButton(QPushButton):
    """A selectable predicted-word chip in the candidate bar."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._word = ""
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setVisible(False)

    def set_candidate(
        self, word: str, tooltip: str = "", selected: bool = False
    ) -> None:
        self._word = word
        self.setText(word)
        self.setToolTip(tooltip)
        self.setVisible(bool(word))
        if selected:
            self.setStyleSheet(
                f"QPushButton{{background:{C['cand_sel_bg']};"
                f"color:{C['cand_sel_text']};"
                f"border:1px solid {C['display_active']};"
                f"border-radius:5px;padding:3px 10px;"
                f"font-weight:bold;font-size:13px;}}"
                f"QPushButton:hover{{background:#174a70;}}"
            )
        else:
            self.setStyleSheet(
                f"QPushButton{{background:#13161e;"
                f"color:{C['cand_text']};"
                f"border:1px solid #222535;"
                f"border-radius:5px;padding:3px 10px;"
                f"font-size:13px;}}"
                f"QPushButton:hover{{background:#1c2030;color:white;}}"
            )

    def clear_candidate(self) -> None:
        self._word = ""
        self.setText("")
        self.setToolTip("")
        self.setVisible(False)

    @property
    def word(self) -> str:
        return self._word


class TextDisplay(QFrame):
    """The phone LCD screen showing composed text and the in-progress word."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.Box)
        self.setStyleSheet(
            f"QFrame{{background:{C['display_bg']};"
            f"border:2px solid #1e3a1e;border-radius:6px;}}"
        )

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 6)
        lay.setSpacing(4)

        self._browser = QTextBrowser()
        self._browser.setReadOnly(True)
        self._browser.setFrameStyle(0)
        self._browser.setStyleSheet(
            f"QTextBrowser{{background:transparent;"
            f"color:{C['display_text']};"
            f"font-size:15px;font-family:'Courier New',monospace;"
            f"border:none;}}"
        )
        self._browser.setMinimumHeight(90)
        lay.addWidget(self._browser)

        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)

        self._hint = QLabel("")
        self._hint.setStyleSheet(
            f"color:{C['display_dim']};font-size:11px;background:transparent;"
        )
        bottom.addWidget(self._hint)
        bottom.addStretch()

        self._mode_lbl = QLabel("T9")
        self._mode_lbl.setStyleSheet(
            f"color:{C['mode_T9']};font-size:11px;"
            f"font-weight:bold;background:transparent;"
        )
        bottom.addWidget(self._mode_lbl)
        lay.addLayout(bottom)

    def refresh(
        self,
        committed: str,
        active_word: str,
        digit_hint: str,
        mode: str,
    ) -> None:
        def _esc(s: str) -> str:
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")

        comm_html = _esc(committed)
        if active_word:
            act_html = _esc(active_word)
            body = (
                f'<span style="color:{C["display_text"]}">{comm_html}</span>'
                f'<span style="color:{C["display_active"]};'
                f'text-decoration:underline;font-weight:bold">{act_html}</span>'
                f'<span style="color:{C["display_active"]}">▌</span>'
            )
        elif committed:
            body = (
                f'<span style="color:{C["display_text"]}">{comm_html}</span>'
                f'<span style="color:{C["display_active"]}">▌</span>'
            )
        else:
            body = f'<span style="color:{C["display_dim"]}">Start typing…</span>'

        self._browser.setHtml(
            f'<div style="font-family:\'Courier New\',monospace;font-size:15px">'
            f"{body}</div>"
        )
        sb = self._browser.verticalScrollBar()
        sb.setValue(sb.maximum())

        self._hint.setText(f"[{digit_hint}]" if digit_hint else "")

        if mode == "T9":
            self._mode_lbl.setText("T9")
            self._mode_lbl.setStyleSheet(
                f"color:{C['mode_T9']};font-size:11px;"
                f"font-weight:bold;background:transparent;"
            )
        else:
            self._mode_lbl.setText("ABC")
            self._mode_lbl.setStyleSheet(
                f"color:{C['mode_ABC']};font-size:11px;"
                f"font-weight:bold;background:transparent;"
            )


class ContextStrip(QWidget):
    """A thin strip showing the recent context words as chips."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(
            f"background:{C['bar_bg']};border-top:1px solid {C['bar_border']};"
        )

        outer = QHBoxLayout(self)
        outer.setContentsMargins(6, 3, 6, 3)
        outer.setSpacing(4)

        lbl = QLabel("ctx:")
        lbl.setFixedWidth(28)
        lbl.setStyleSheet(
            f"color:{C['status_text']};font-size:11px;background:transparent;"
        )
        outer.addWidget(lbl)

        self._chips_box = QHBoxLayout()
        self._chips_box.setSpacing(4)
        self._chips_box.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(self._chips_box)
        outer.addStretch()

    def set_context(self, words: list[str]) -> None:
        # Clear existing chips
        while self._chips_box.count():
            item = self._chips_box.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not words:
            empty = QLabel("—")
            empty.setStyleSheet(
                f"color:{C['status_text']};font-size:11px;background:transparent;"
            )
            self._chips_box.addWidget(empty)
        else:
            for w in words[-7:]:
                chip = QLabel(w)
                chip.setStyleSheet(
                    f"background:{C['ctx_chip_bg']};"
                    f"color:{C['ctx_chip_text']};"
                    f"border:1px solid #203820;"
                    f"border-radius:3px;padding:1px 5px;"
                    f"font-size:11px;"
                )
                self._chips_box.addWidget(chip)


class SettingsDialog(QDialog):
    """Settings dialog for top-k and predictor signal info."""

    def __init__(self, predictor, top_k: int, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(340)
        self.setStyleSheet(
            f"QDialog{{background:{C['phone_bg']};color:#ccc;}}"
            f"QLabel{{color:#aaa;}}"
            f"QGroupBox{{color:#77aa77;border:1px solid #2a2a3a;"
            f"border-radius:4px;padding-top:10px;margin-top:6px;}}"
            f"QGroupBox::title{{subcontrol-origin:margin;left:8px;}}"
        )

        lay = QVBoxLayout(self)

        # Predictor signals
        grp = QGroupBox("Active signals")
        glay = QFormLayout(grp)
        for sig, w in predictor.weights.items():
            col = C["display_active"] if w > 0 else C["status_text"]
            icon = "●" if w > 0 else "○"
            val_lbl = QLabel(f"{w:.3f}" if w > 0 else "disabled")
            val_lbl.setStyleSheet(f"color:{col};")
            glay.addRow(f"{icon} {sig}", val_lbl)
        lay.addWidget(grp)

        # Top-k
        grp2 = QGroupBox("Prediction")
        g2lay = QFormLayout(grp2)
        self._topk = QSpinBox()
        self._topk.setRange(1, 10)
        self._topk.setValue(top_k)
        self._topk.setStyleSheet(
            "background:#1e2030;color:#ddd;border:1px solid #40405a;"
        )
        g2lay.addRow("Candidates (top-k):", self._topk)
        lay.addWidget(grp2)

        # Help blurb
        help_lbl = QLabel(
            "<b>Keyboard:</b> 2-9 digits · Backspace=delete · "
            "Space/0=confirm · Tab/#=next · M=mode · Esc=clear"
        )
        help_lbl.setWordWrap(True)
        help_lbl.setStyleSheet(
            f"color:{C['status_text']};font-size:11px;background:transparent;"
        )
        lay.addWidget(help_lbl)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        btns.setStyleSheet(
            "QPushButton{background:#1e2830;color:#ccc;"
            "border:1px solid #354535;border-radius:4px;padding:4px 12px;}"
            "QPushButton:hover{background:#263630;}"
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def get_top_k(self) -> int:
        return self._topk.value()


# ─── Main window ──────────────────────────────────────────────────────────────


class T9PhoneWindow(QMainWindow):
    """The T9 phone UI window.

    Modes
    -----
    T9  — predictive: digits accumulate into a sequence and the predictor
          ranks all dictionary words whose T9 code matches.  # cycles
          through alternates; Space/0 confirms the highlighted word.

    ABC — multi-tap: pressing the same key rapidly cycles through its
          letters (like classic non-predictive T9).  A 600 ms timer
          commits the pending letter automatically.
    """

    _MODE_T9 = "T9"
    _MODE_ABC = "ABC"

    def __init__(self, predictor, top_k: int = 5) -> None:
        super().__init__()
        from ai_t9 import T9Session

        self._predictor = predictor
        self._session = T9Session(predictor, context_window=7)
        self._top_k = top_k

        # ── T9 state ──────────────────────────────────────────────────
        self._digit_buf: str = ""       # digits entered for current word
        self._candidates: list = []     # RankedCandidate objects
        self._cand_idx: int = 0         # currently highlighted candidate
        self._committed: str = ""       # confirmed text (words + spaces)
        self._mode: str = self._MODE_T9

        # ── Multi-tap state ───────────────────────────────────────────
        self._mt_digit: str = ""        # last digit pressed
        self._mt_idx: int = 0           # current letter index on that key
        self._mt_pending: str = ""      # letter being assembled
        self._mt_timer = QTimer(self)
        self._mt_timer.setSingleShot(True)
        self._mt_timer.setInterval(600)
        self._mt_timer.timeout.connect(self._mt_commit)

        # ── Punctuation cycling ────────────────────────────────────────
        self._punct_idx: int = -1

        self._build_ui()
        self._install_shortcuts()
        self._refresh()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()

    # ─────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.setWindowTitle("ai-t9  •  T9 Phone")
        self.setMinimumSize(390, 700)
        self.resize(420, 760)

        central = QWidget()
        self.setCentralWidget(central)
        central.setStyleSheet(f"background:{C['phone_bg']};")

        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 6)
        root.setSpacing(5)

        # ── Title bar ─────────────────────────────────────────────────
        tbar = QHBoxLayout()
        tbar.setContentsMargins(4, 0, 4, 0)

        title = QLabel("ai-t9")
        title.setStyleSheet(
            f"color:{C['display_active']};font-size:15px;"
            f"font-weight:bold;background:transparent;"
        )
        tbar.addWidget(title)

        self._signals_lbl = QLabel("")
        self._signals_lbl.setStyleSheet(
            f"color:{C['status_text']};font-size:11px;background:transparent;"
        )
        tbar.addWidget(self._signals_lbl)
        tbar.addStretch()

        settings_btn = QToolButton()
        settings_btn.setText("⚙")
        settings_btn.setToolTip("Settings")
        settings_btn.setStyleSheet(
            "color:#666;background:transparent;border:none;font-size:17px;"
        )
        settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        settings_btn.clicked.connect(self._open_settings)
        tbar.addWidget(settings_btn)
        root.addLayout(tbar)

        # ── Text display ──────────────────────────────────────────────
        self._display = TextDisplay()
        self._display.setMinimumHeight(120)
        root.addWidget(self._display)

        # ── Candidate bar ─────────────────────────────────────────────
        bar = QFrame()
        bar.setStyleSheet(
            f"QFrame{{background:{C['bar_bg']};"
            f"border:1px solid {C['bar_border']};"
            f"border-radius:4px;}}"
        )
        bar_lay = QHBoxLayout(bar)
        bar_lay.setContentsMargins(6, 4, 6, 4)
        bar_lay.setSpacing(5)

        self._cand_btns: list[CandidateButton] = []
        for _ in range(8):
            cb = CandidateButton()
            cb.clicked.connect(lambda _checked, b=cb: self._pick_candidate(b))
            self._cand_btns.append(cb)
            bar_lay.addWidget(cb)

        bar_lay.addStretch()

        clr = QToolButton()
        clr.setText("✕")
        clr.setToolTip("Clear all  (Esc or long-press ★)")
        clr.setStyleSheet(
            "color:#884444;background:transparent;border:none;font-size:15px;"
        )
        clr.setCursor(Qt.CursorShape.PointingHandCursor)
        clr.clicked.connect(self._clear_all)
        bar_lay.addWidget(clr)

        cpy = QToolButton()
        cpy.setText("⎘")
        cpy.setToolTip("Copy to clipboard  (Ctrl+C)")
        cpy.setStyleSheet(
            "color:#448844;background:transparent;border:none;font-size:15px;"
        )
        cpy.setCursor(Qt.CursorShape.PointingHandCursor)
        cpy.clicked.connect(self._copy_to_clipboard)
        bar_lay.addWidget(cpy)

        root.addWidget(bar)

        # ── Context strip ─────────────────────────────────────────────
        self._ctx_strip = ContextStrip()
        root.addWidget(self._ctx_strip)

        # ── Keypad ────────────────────────────────────────────────────
        kpad = QFrame()
        kpad.setStyleSheet(
            "QFrame{background:#0b0d14;border:1px solid #1a1d26;border-radius:10px;}"
        )
        grid = QGridLayout(kpad)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setSpacing(5)

        self._key_btns: dict[str, T9KeyButton] = {}
        for r, row in enumerate(KEYPAD_ROWS):
            for c, key in enumerate(row):
                btn = T9KeyButton(key)
                btn.clicked.connect(lambda _chk, k=key: self._on_key(k))
                btn.long_pressed.connect(self._on_long_press)
                self._key_btns[key] = btn
                grid.addWidget(btn, r, c)

        root.addWidget(kpad)

        # ── Bottom row: mode toggle + send ────────────────────────────
        brow = QHBoxLayout()
        brow.setContentsMargins(4, 2, 4, 2)

        mode_btn = QPushButton("⇄  T9 / ABC")
        mode_btn.setToolTip("Toggle predictive / multi-tap mode  (M)")
        mode_btn.setStyleSheet(
            f"QPushButton{{background:#16201a;color:{C['ctx_chip_text']};"
            f"border:1px solid #2a4a2a;border-radius:4px;"
            f"padding:5px 12px;font-size:12px;}}"
            f"QPushButton:hover{{background:#1e2e22;}}"
        )
        mode_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        mode_btn.clicked.connect(self._toggle_mode)
        brow.addWidget(mode_btn)

        brow.addStretch()

        send_btn = QPushButton("Send ↵")
        send_btn.setToolTip("Confirm current word and clear display")
        send_btn.setStyleSheet(
            "QPushButton{background:#102810;color:#77dd77;"
            "border:1px solid #224422;border-radius:4px;"
            "padding:5px 14px;font-size:12px;font-weight:bold;}"
            "QPushButton:hover{background:#163816;}"
        )
        send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        send_btn.clicked.connect(self._send)
        brow.addWidget(send_btn)

        root.addLayout(brow)

        # ── Status bar ────────────────────────────────────────────────
        self.statusBar().setStyleSheet(
            f"color:{C['status_text']};background:{C['phone_bg']};font-size:11px;"
        )

        self._refresh_signals_label()

    def _install_shortcuts(self) -> None:
        QShortcut(QKeySequence("Ctrl+C"), self, self._copy_to_clipboard)

    # ─────────────────────────────────────────────────────────────────
    # Key dispatch
    # ─────────────────────────────────────────────────────────────────

    def _on_key(self, key: str) -> None:
        if self._mode == self._MODE_T9:
            self._t9_key(key)
        else:
            self._abc_key(key)

    def _on_long_press(self, key: str) -> None:
        if key == "*":
            self._clear_all()
        elif key == "#":
            # Newline
            if self._digit_buf:
                self._t9_confirm()
            self._committed += "\n"
            self._refresh()
        elif key == "0":
            # Double space → new paragraph
            if self._digit_buf:
                self._t9_confirm()
            self._committed += "\n\n"
            self._refresh()

    # ─────────────────────────────────────────────────────────────────
    # T9 (predictive) mode
    # ─────────────────────────────────────────────────────────────────

    def _t9_key(self, key: str) -> None:
        if key in "23456789":
            self._digit_buf += key
            self._cand_idx = 0
            self._t9_predict()

        elif key == "1":
            # Punctuation: confirm any pending word first
            if self._digit_buf:
                self._t9_confirm()
            self._punct_idx = (self._punct_idx + 1) % len(PUNCT_CYCLE)
            self._committed += PUNCT_CYCLE[self._punct_idx]
            self._refresh()

        elif key == "*":
            self._t9_backspace()

        elif key == "#":
            # Cycle to next candidate
            if self._candidates:
                self._cand_idx = (self._cand_idx + 1) % len(self._candidates)
            self._refresh()

        elif key == "0":
            # Space — confirm current candidate (if any) + add space
            if self._digit_buf:
                self._t9_confirm()
            if self._committed and self._committed[-1] != "\n":
                self._committed += " "
            self._refresh()

    def _t9_predict(self) -> None:
        if self._digit_buf:
            try:
                self._candidates = self._session.dial(
                    self._digit_buf, top_k=self._top_k, return_details=True
                )
            except ValueError:
                self._candidates = []
        else:
            self._candidates = []
        self._refresh()

    def _t9_confirm(self) -> None:
        """Confirm the currently highlighted candidate."""
        if not self._candidates:
            return
        cand = self._candidates[self._cand_idx]
        word = cand.word if hasattr(cand, "word") else str(cand)
        self._session.confirm(word)
        self._committed += word
        self._digit_buf = ""
        self._cand_idx = 0
        self._candidates = []

    def _t9_backspace(self) -> None:
        if self._digit_buf:
            # Delete last digit
            self._digit_buf = self._digit_buf[:-1]
            self._cand_idx = 0
            self._t9_predict()
            return

        if not self._committed:
            return

        # If the last committed character is a space, try to restore the
        # preceding word back into the digit buffer so the user can correct it.
        if self._committed.endswith(" "):
            self._committed = self._committed[:-1]
            # Find the last word
            if " " in self._committed or "\n" in self._committed:
                # Split on whitespace manually to get the tail
                import re
                parts = re.split(r"[\s]", self._committed)
                last = parts[-1] if parts else ""
            else:
                last = self._committed
            if last:
                from ai_t9.t9_map import word_to_digits
                d = word_to_digits(last)
                if d:
                    self._committed = self._committed[: -len(last)]
                    self._session.undo_confirm()
                    self._digit_buf = d
                    self._cand_idx = 0
                    self._t9_predict()
                    return
        else:
            # Just delete the last character
            self._committed = self._committed[:-1]

        self._refresh()

    # ─────────────────────────────────────────────────────────────────
    # ABC (multi-tap) mode
    # ─────────────────────────────────────────────────────────────────

    def _abc_key(self, key: str) -> None:
        if key in MULTITAP:
            letters = MULTITAP[key]
            if key == self._mt_digit and self._mt_timer.isActive():
                # Same key again: advance to next letter
                self._mt_timer.stop()
                self._mt_idx = (self._mt_idx + 1) % len(letters)
            else:
                # Different key: commit previous pending letter
                self._mt_commit()
                self._mt_digit = key
                self._mt_idx = 0
            self._mt_pending = letters[self._mt_idx]
            self._mt_timer.start()

        elif key == "*":
            self._mt_timer.stop()
            if self._mt_pending:
                self._mt_pending = ""
                self._mt_digit = ""
                self._mt_idx = 0
            elif self._committed:
                self._committed = self._committed[:-1]

        elif key == "0":
            self._mt_commit()
            if self._committed and self._committed[-1] not in (" ", "\n"):
                self._committed += " "

        elif key == "#":
            # Uppercase toggle for the pending letter
            if self._mt_pending:
                self._mt_timer.stop()
                if self._mt_pending.islower():
                    self._mt_pending = self._mt_pending.upper()
                else:
                    self._mt_pending = self._mt_pending.lower()
                self._mt_timer.start()

        elif key == "1":
            self._mt_commit()
            self._punct_idx = (self._punct_idx + 1) % len(PUNCT_CYCLE)
            self._committed += PUNCT_CYCLE[self._punct_idx]

        self._refresh()

    def _mt_commit(self) -> None:
        """Flush the pending multi-tap letter into committed text."""
        if self._mt_pending:
            self._committed += self._mt_pending
            self._mt_pending = ""
        self._mt_digit = ""
        self._mt_idx = 0

    # ─────────────────────────────────────────────────────────────────
    # Candidate selection
    # ─────────────────────────────────────────────────────────────────

    def _pick_candidate(self, btn: CandidateButton) -> None:
        if not btn.word:
            return
        self._session.confirm(btn.word)
        self._committed += btn.word
        self._digit_buf = ""
        self._cand_idx = 0
        self._candidates = []
        self._refresh()

    # ─────────────────────────────────────────────────────────────────
    # Mode toggle
    # ─────────────────────────────────────────────────────────────────

    def _toggle_mode(self) -> None:
        # Flush any in-progress input
        if self._mode == self._MODE_T9:
            if self._digit_buf:
                self._t9_confirm()
        else:
            self._mt_timer.stop()
            self._mt_commit()

        self._mode = (
            self._MODE_ABC if self._mode == self._MODE_T9 else self._MODE_T9
        )
        self._digit_buf = ""
        self._candidates = []
        self._cand_idx = 0
        self._mt_pending = ""
        self._mt_digit = ""
        self._refresh()

    # ─────────────────────────────────────────────────────────────────
    # Utility actions
    # ─────────────────────────────────────────────────────────────────

    def _clear_all(self) -> None:
        self._mt_timer.stop()
        self._digit_buf = ""
        self._candidates = []
        self._cand_idx = 0
        self._committed = ""
        self._mt_pending = ""
        self._mt_digit = ""
        self._mt_idx = 0
        self._session.reset()
        self._refresh()

    def _copy_to_clipboard(self) -> None:
        text = self._committed
        if self._mode == self._MODE_T9 and self._candidates:
            cand = self._candidates[self._cand_idx]
            text += cand.word if hasattr(cand, "word") else str(cand)
        elif self._mode == self._MODE_ABC and self._mt_pending:
            text += self._mt_pending
        QGuiApplication.clipboard().setText(text)
        self.statusBar().showMessage("Copied to clipboard", 1500)

    def _send(self) -> None:
        # Flush buffer
        if self._mode == self._MODE_T9 and self._digit_buf and self._candidates:
            self._t9_confirm()
        elif self._mode == self._MODE_ABC:
            self._mt_timer.stop()
            self._mt_commit()
        msg = self._committed.strip()
        if msg:
            self.statusBar().showMessage(f"✓ Sent: {msg!r}", 3000)
        self._clear_all()

    def _open_settings(self) -> None:
        dlg = SettingsDialog(self._predictor, self._top_k, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._top_k = dlg.get_top_k()
            if self._digit_buf:
                self._t9_predict()
            else:
                self._refresh()

    # ─────────────────────────────────────────────────────────────────
    # Display refresh
    # ─────────────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        # Active (in-progress) word for display
        if self._mode == self._MODE_T9:
            if self._candidates:
                c = self._candidates[self._cand_idx]
                active = c.word if hasattr(c, "word") else str(c)
            else:
                active = ""
            hint = self._digit_buf

        else:
            active = self._mt_pending
            hint = f"{self._mt_digit}×{self._mt_idx + 1}" if self._mt_digit else ""

        self._display.refresh(
            committed=self._committed,
            active_word=active,
            digit_hint=hint,
            mode=self._mode,
        )

        self._refresh_candidate_bar()
        self._ctx_strip.set_context(self._session.context)
        self._refresh_status()

    def _refresh_candidate_bar(self) -> None:
        if self._mode == self._MODE_T9 and self._candidates:
            for i, cb in enumerate(self._cand_btns):
                if i < len(self._candidates):
                    c = self._candidates[i]
                    word = c.word if hasattr(c, "word") else str(c)
                    if hasattr(c, "freq_score"):
                        tip = (
                            f"freq={c.freq_score:.3f}  "
                            f"model={c.model_score:.3f}  "
                            f"ngram={c.ngram_score:.3f}  "
                            f"final={c.final_score:.3f}"
                        )
                    else:
                        tip = ""
                    cb.set_candidate(word, tip, selected=(i == self._cand_idx))
                else:
                    cb.clear_candidate()
        else:
            for cb in self._cand_btns:
                cb.clear_candidate()

    def _refresh_status(self) -> None:
        ctx = self._session.context
        ctx_str = " » ".join(ctx) if ctx else "—"
        chars = len(self._committed)
        words = len(self._committed.split()) if self._committed.strip() else 0
        self.statusBar().showMessage(
            f"chars: {chars}  words: {words}  context: [{ctx_str}]"
        )

    def _refresh_signals_label(self) -> None:
        active = [s for s, w in self._predictor.weights.items() if w > 0]
        self._signals_lbl.setText("  " + " + ".join(active))

    # ─────────────────────────────────────────────────────────────────
    # Keyboard events (physical keyboard → T9 keys)
    # ─────────────────────────────────────────────────────────────────

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        k = event.key()
        _digit = {
            Qt.Key.Key_1: "1",
            Qt.Key.Key_2: "2",
            Qt.Key.Key_3: "3",
            Qt.Key.Key_4: "4",
            Qt.Key.Key_5: "5",
            Qt.Key.Key_6: "6",
            Qt.Key.Key_7: "7",
            Qt.Key.Key_8: "8",
            Qt.Key.Key_9: "9",
            Qt.Key.Key_0: "0",
        }
        if k == Qt.Key.Key_Backspace:
            self._on_key("*")
        elif k in (Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._on_key("0")
        elif k == Qt.Key.Key_Tab:
            self._on_key("#")
        elif k == Qt.Key.Key_Period:
            self._on_key("1")
        elif k == Qt.Key.Key_M:
            self._toggle_mode()
        elif k == Qt.Key.Key_Escape:
            self._clear_all()
        elif k in _digit:
            self._on_key(_digit[k])
        else:
            super().keyPressEvent(event)


# ─── CLI / entry point ────────────────────────────────────────────────────────


def _build_predictor(args):
    from ai_t9 import T9Predictor

    if args.vocab and args.dict:
        print(f"Loading predictor from:  {args.vocab}")
        predictor = T9Predictor.from_files(
            vocab_path=args.vocab,
            dict_path=args.dict,
            model_path=args.model,
            ngram_path=args.ngram,
        )
    else:
        print("Building default predictor from NLTK (first run may download data)…")
        predictor = T9Predictor.build_default(verbose=True)

    print("Signals active:")
    for sig, w in predictor.weights.items():
        status = f"{w:.3f}" if w > 0 else "disabled"
        print(f"  {sig:8s}: {status}")
    return predictor


def main() -> int:
    parser = argparse.ArgumentParser(description="ai-t9 Qt GUI demo")
    parser.add_argument("--vocab", metavar="FILE", default=None, help="vocab.json path")
    parser.add_argument("--dict",  metavar="FILE", default=None, help="dict.json path")
    parser.add_argument("--model", metavar="FILE", default=None, help="model.npz path (optional)")
    parser.add_argument("--ngram", metavar="FILE", default=None, help="bigram.json path (optional)")
    parser.add_argument("--top-k", type=int, default=5, help="Candidates to show (default 5)")
    args = parser.parse_args()

    predictor = _build_predictor(args)

    app = QApplication(sys.argv)
    app.setApplicationName("ai-t9")
    app.setStyle("Fusion")

    # Apply a global dark Fusion palette
    pal = QPalette()
    for role, hex_col in (
        (QPalette.ColorRole.Window,         C["phone_bg"]),
        (QPalette.ColorRole.WindowText,     "#cccccc"),
        (QPalette.ColorRole.Base,           "#0d0f14"),
        (QPalette.ColorRole.AlternateBase,  "#111318"),
        (QPalette.ColorRole.Text,           "#cccccc"),
        (QPalette.ColorRole.Button,         C["key_bg"]),
        (QPalette.ColorRole.ButtonText,     C["key_text"]),
        (QPalette.ColorRole.Highlight,      C["cand_sel_bg"]),
        (QPalette.ColorRole.HighlightedText,"#ffffff"),
        (QPalette.ColorRole.ToolTipBase,    "#1a1d26"),
        (QPalette.ColorRole.ToolTipText,    "#cccccc"),
    ):
        pal.setColor(role, QColor(hex_col))
    app.setPalette(pal)

    win = T9PhoneWindow(predictor, top_k=args.top_k)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
