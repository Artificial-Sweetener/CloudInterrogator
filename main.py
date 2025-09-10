#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import os
import random
import sys
from dataclasses import dataclass
from io import BytesIO
from typing import List, Optional, Tuple

from PIL import Image
from PySide6.QtCore import QPoint, QSettings
from PySide6.QtCore import QSize
from PySide6.QtCore import QSize as QSizeObj
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QGuiApplication, QIcon, QImage, QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QSpinBox,
    QSplitter,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    ComboBox,
    Dialog,
    FluentIcon,
    InfoBar,
    InfoBarPosition,
    LineEdit,
    Pivot,
    PivotItem,
    PrimaryPushButton,
    PushButton,
    Theme,
    setTheme,
    setThemeColor,
    themeColor,
)
from qframelesswindow import AcrylicWindow
from qframelesswindow.titlebar import TitleBar

import resources_rc

APP_ORG = "CloudInterrogator"
APP_NAME = "CloudInterrogator"

if (
    getattr(sys, "frozen", False)
    or os.path.splitext(sys.argv[0])[1].lower() == ".exe"
    or os.environ.get("NUITKA_ONEFILE_PARENT")
):
    app_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
else:
    app_dir = os.path.dirname(os.path.abspath(__file__))

ENDPOINTS_FILE = os.path.join(app_dir, "endpoints.json")

UI_GAP = 12  # single knob for section spacing


@dataclass
class Endpoint:
    name: str
    base_url: str
    api_key: str
    models: List[str]


class EndpointStore:
    def __init__(self, path: str = ENDPOINTS_FILE):
        self.path = path
        self._endpoints: List[Endpoint] = []
        self.load()

    def load(self) -> None:
        if not os.path.exists(self.path):
            self._endpoints = []
            self.save()
            return
        with open(self.path, "r", encoding="utf-8") as f:
            raw = json.load(f) or []
        self._endpoints = [Endpoint(**e) for e in raw]

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump([e.__dict__ for e in self._endpoints], f, indent=2)

    def display_names(self) -> List[str]:
        out: List[str] = []
        for ep in self._endpoints:
            if ep.models:
                for m in ep.models:
                    out.append(f"{ep.name} / {m}")
            else:
                out.append(f"{ep.name} / (no models)")
        return out

    def details_from_display(
        self, display: str
    ) -> Optional[Tuple[str, str, Optional[str]]]:
        for ep in self._endpoints:
            prefix = f"{ep.name} / "
            if display.startswith(prefix):
                model = display[len(prefix) :]
                if model == "(no models)":
                    return (ep.base_url, ep.api_key, None)
                if model in ep.models:
                    return (ep.base_url, ep.api_key, model)
        return None

    def upsert(self, ep: Endpoint) -> None:
        i = next((i for i, e in enumerate(self._endpoints) if e.name == ep.name), -1)
        if i >= 0:
            self._endpoints[i] = ep
        else:
            self._endpoints.append(ep)
        self.save()

    def delete(self, name: str) -> None:
        self._endpoints = [e for e in self._endpoints if e.name != name]
        self.save()


class ChatStreamer(QThread):
    token = Signal(str)
    done = Signal()
    failed = Signal(str)

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        image_data_url: Optional[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: Optional[int],
        parent=None,
    ):
        super().__init__(parent)
        self.base_url, self.api_key, self.model = base_url, api_key, model
        self.system_prompt, self.user_prompt = (
            system_prompt.strip(),
            user_prompt.strip(),
        )
        self.image_data_url = image_data_url
        self.max_tokens, self.temperature, self.top_p = max_tokens, temperature, top_p
        self.seed = seed
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _messages(self):
        msgs = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        content = [
            {
                "type": "text",
                "text": self.user_prompt or "Describe the image in detail.",
            }
        ]
        if self.image_data_url:
            content.append(
                {"type": "image_url", "image_url": {"url": self.image_data_url}}
            )
        msgs.append({"role": "user", "content": content})
        return msgs

    def run(self):
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            kwargs = dict(
                model=self.model,
                messages=self._messages(),
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                stream=True,
            )
            if self.seed is not None:
                kwargs["seed"] = int(self.seed)
            stream = client.chat.completions.create(**kwargs)
            for event in stream:
                if self._cancel:
                    break
                try:
                    delta = event.choices[0].delta
                    if delta and getattr(delta, "content", None):
                        self.token.emit(delta.content)
                except Exception:
                    choice = event.get("choices", [{}])[0]
                    delta = choice.get("delta", {})
                    content = delta.get("content")
                    if content:
                        self.token.emit(content)
            self.done.emit()
        except Exception as e:
            self.failed.emit(str(e))


def to_pixmap(pil: Image.Image, target: QSize) -> QPixmap:
    # Use a 32bpp format to ensure 4-byte row alignment on Windows
    if pil.mode != "RGBA":
        pil = pil.convert("RGBA")
    w, h = pil.size
    buf = pil.tobytes()
    # Force deep copy via .copy() to detach from Python buffer
    qimg = QImage(buf, w, h, w * 4, QImage.Format_RGBA8888).copy()
    pm = QPixmap.fromImage(qimg)
    return pm.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def encode_image(pil: Image.Image, short_side: int = 512, quality: int = 85) -> str:
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    w, h = pil.size
    scale = short_side / min(w, h)
    img = pil.resize((int(round(w * scale)), int(round(h * scale))), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


class ImagePreview(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pil = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(260)
        self.setText("No image")
        self.setStyleSheet(
            "background:#181818; border:1px solid #2a2a2a; border-radius:10px;"
        )

    def set_image(self, pil: Image.Image):
        self._pil = pil.convert("RGB")
        self._update_pixmap()

    def clear_image(self):
        self._pil = None
        self.setPixmap(QPixmap())
        self.setText("No image")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._pil:
            self._update_pixmap()

    def _update_pixmap(self):
        w = self.width()
        h = self.height()
        if w < 2 or h < 2:
            return
        pm = to_pixmap(self._pil, QSize(w, h))
        self.setPixmap(pm)


class EndpointEditorDialog(QDialog):
    def __init__(self, parent=None, endpoint: Optional[Endpoint] = None):
        super().__init__(parent)
        self.setWindowTitle("Edit Endpoint")
        self.setMinimumWidth(520)
        self.name_edit, self.base_url_edit, self.api_key_edit = (
            LineEdit(),
            LineEdit(),
            LineEdit(),
        )
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.models_edit = QPlainTextEdit()
        self.models_edit.setPlaceholderText("One model per line")
        form = QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        form.addRow("Name", self.name_edit)
        form.addRow("Base URL", self.base_url_edit)
        form.addRow("API Key", self.api_key_edit)
        form.addRow("Models", self.models_edit)
        ok_btn, cancel_btn = PrimaryPushButton(FluentIcon.SAVE, "Save"), PushButton(
            "Cancel"
        )
        btns = QHBoxLayout()
        btns.addStretch(1)
        btns.addWidget(ok_btn)
        btns.addWidget(cancel_btn)
        root = QVBoxLayout(self)
        root.addLayout(form)
        root.addLayout(btns)
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        if endpoint:
            self.name_edit.setText(endpoint.name)
            self.base_url_edit.setText(endpoint.base_url)
            self.api_key_edit.setText(endpoint.api_key)
            self.models_edit.setPlainText("\n".join(endpoint.models))

    def value(self) -> Endpoint:
        models = [
            ln.strip()
            for ln in self.models_edit.toPlainText().splitlines()
            if ln.strip()
        ]
        return Endpoint(
            self.name_edit.text().strip(),
            self.base_url_edit.text().strip(),
            self.api_key_edit.text().strip(),
            models,
        )


class EndpointManagerDialog(QDialog):
    def __init__(self, store: EndpointStore, parent=None):
        super().__init__(parent)
        self.store = store
        self.setWindowTitle("Endpoints Manager")
        self.setMinimumSize(700, 420)
        self.list = QListWidget()
        for ep in self.store._endpoints:
            self.list.addItem(QListWidgetItem(ep.name))
        self.add_btn, self.edit_btn, self.del_btn, self.close_btn = (
            PrimaryPushButton(FluentIcon.ADD, "Add"),
            PrimaryPushButton(FluentIcon.EDIT, "Edit"),
            PrimaryPushButton(FluentIcon.DELETE, "Delete"),
            PushButton("Close"),
        )
        left = QVBoxLayout()
        left.addWidget(self.list)
        right = QVBoxLayout()
        [right.addWidget(b) for b in (self.add_btn, self.edit_btn, self.del_btn)]
        right.addStretch(1)
        right.addWidget(self.close_btn)
        body = QHBoxLayout(self)
        body.addLayout(left, 4)
        body.addLayout(right, 1)
        self.add_btn.clicked.connect(self.add_endpoint)
        self.edit_btn.clicked.connect(self.edit_endpoint)
        self.del_btn.clicked.connect(self.delete_endpoint)
        self.close_btn.clicked.connect(self.accept)

    def _selected_name(self) -> Optional[str]:
        it = self.list.currentItem()
        return it.text() if it else None

    def add_endpoint(self):
        dlg = EndpointEditorDialog(self)
        if dlg.exec() == QDialog.Accepted:
            ep = dlg.value()
            if not ep.name:
                InfoBar.error("Invalid endpoint", "Name is required", parent=self)
                return
            self.store.upsert(ep)
            self.list.addItem(QListWidgetItem(ep.name))
            InfoBar.success("Saved", f"Endpoint '{ep.name}' added.", parent=self)

    def edit_endpoint(self):
        name = self._selected_name()
        if not name:
            return
        ep = next((e for e in self.store._endpoints if e.name == name), None)
        dlg = EndpointEditorDialog(self, ep)
        if dlg.exec() == QDialog.Accepted:
            new_ep = dlg.value()
            self.store.upsert(new_ep)
            self._refresh()
            InfoBar.success("Saved", f"Endpoint '{new_ep.name}' updated.", parent=self)

    def delete_endpoint(self):
        name = self._selected_name()
        if not name:
            return
        if Dialog.confirm("Delete endpoint", f"Delete '{name}'?"):
            self.store.delete(name)
            self._refresh()
            InfoBar.success("Deleted", f"Endpoint '{name}' removed.", parent=self)

    def _refresh(self):
        self.list.clear()
        [self.list.addItem(QListWidgetItem(ep.name)) for ep in self.store._endpoints]


class CustomTitleBar(TitleBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.maxBtn.hide()
        if hasattr(self, "menuContainer"):
            self.menuContainer.setFixedWidth(0)
        title = QLabel("Cloud Interrogator", self)
        title.setStyleSheet(
            "color: white; font-size: 15px; margin-left: 14px; font-weight: 500;"
        )
        self.layout().insertWidget(0, title, 0, Qt.AlignLeft)
        self.mouseDoubleClickEvent = lambda event: None


class FluentWindow(AcrylicWindow):
    def __init__(self):
        super().__init__()
        tb = CustomTitleBar(self)
        for btn in [tb.minBtn, tb.closeBtn]:
            btn.setNormalColor(Qt.white)
        self.setTitleBar(tb)
        self.titleBar = tb
        try:
            self.windowEffect.setMicaEffect(self.winId(), isDarkMode=True, isAlt=False)
        except Exception:
            pass

    def closeEvent(self, e):
        if hasattr(self, "on_close") and callable(self.on_close):
            try:
                self.on_close()
            except Exception:
                pass
        super().closeEvent(e)


class MainPanel(QWidget):
    KEY_EP_MODEL = "state/endpoint_model"
    KEY_EP_INDEX = "state/endpoint_index"
    KEY_USER = "state/user_prompt"
    KEY_SYSTEM = "state/system_prompt"
    KEY_OUTPUT = "state/output_text"
    KEY_IMAGE = "state/image_path"
    KEY_IMG_DIR = "state/last_image_dir"
    KEY_TOKENS = "state/max_tokens"
    KEY_TEMP = "state/temperature"
    KEY_TOPP = "state/top_p"
    KEY_SEED = "state/seed"
    KEY_RANDOM = "state/randomize"
    KEY_SPLIT = "ui/splitter_sizes"
    KEY_PROMPT_ROUTE = "ui/prompt_route"
    KEY_PAIR_DIR = "state/last_pair_dir"

    def __init__(self, store: EndpointStore):
        super().__init__()
        self.store = store
        self.current_image: Optional[Image.Image] = None
        self.current_image_path: Optional[str] = None
        self.streamer: Optional[ChatStreamer] = None
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(400)
        self._save_timer.timeout.connect(self.save_app_state)

        self.setStyleSheet(
            """
            QFrame#Section { background: #121212; border: 1px solid #2a2a2a; border-radius: 12px; }
            QPlainTextEdit, QLineEdit { background: #141414; border: 1px solid #2a2a2a; border-radius: 10px; padding: 8px; }
            QLabel#status { color: #a0a0a0; }
            """
        )

        self.endpoint_combo = ComboBox()
        self.refresh_endpoints()
        self.manage_btn = PushButton("Manage Endpoints")
        self.run_btn = PrimaryPushButton(FluentIcon.PLAY, "Run")
        self.stop_btn = PrimaryPushButton(FluentIcon.CLOSE, "Stop")
        self.stop_btn.setEnabled(False)
        self.status_lbl = QLabel("Idle")
        self.status_lbl.setObjectName("status")
        top_row = QHBoxLayout()
        top_row.setSpacing(12)
        top_row.addWidget(QLabel("Endpoint / Model"))
        top_row.addWidget(self.endpoint_combo, 1)
        top_row.addWidget(self.manage_btn)
        top_row.addStretch(1)
        top_row.addWidget(self.status_lbl)
        top_row.addWidget(self.run_btn)
        top_row.addWidget(self.stop_btn)

        # Image (2)
        image_panel = QFrame()
        image_panel.setObjectName("Section")
        iv = QVBoxLayout(image_panel)
        iv.setContentsMargins(12, 12, 12, 12)
        iv.setSpacing(8)
        self.preview = ImagePreview()
        self.open_img_btn, self.clear_img_btn = PrimaryPushButton(
            FluentIcon.PHOTO, "Load Image"
        ), PushButton("Clear")
        self.save_pair_btn, self.load_pair_btn = PushButton("Save Pair…"), PushButton(
            "Load Pair…"
        )
        img_btns = QHBoxLayout()
        img_btns.addWidget(self.open_img_btn)
        img_btns.addWidget(self.clear_img_btn)
        img_btns.addStretch(1)
        img_btns.addWidget(self.save_pair_btn)
        img_btns.addWidget(self.load_pair_btn)
        iv.addWidget(self.preview, 1)
        iv.addLayout(img_btns)

        # Settings (1) stacked
        settings_panel = QFrame()
        settings_panel.setObjectName("Section")
        form = QFormLayout(settings_panel)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        form.setContentsMargins(12, 12, 12, 12)
        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(1, 4096)
        self.max_tokens.setValue(1024)
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.0, 2.0)
        self.temperature.setSingleStep(0.1)
        self.temperature.setValue(1.0)
        self.top_p = QDoubleSpinBox()
        self.top_p.setRange(0.0, 1.0)
        self.top_p.setSingleStep(0.05)
        self.top_p.setValue(1.0)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2**31 - 1)
        self.randomize = QCheckBox("Randomize seed on run")
        form.addRow("Max tokens", self.max_tokens)
        form.addRow("Temperature", self.temperature)
        form.addRow("Top-p", self.top_p)
        form.addRow("Seed", self.seed)
        form.addRow(self.randomize)

        # Borderless container so cards float over Mica
        top_cluster = QWidget()
        tc = QHBoxLayout(top_cluster)
        tc.setContentsMargins(0, 0, 0, 0)  # no outer padding
        tc.setSpacing(UI_GAP)
        tc.addWidget(image_panel, 2)
        tc.addWidget(settings_panel, 1)
        top_cluster.setFixedHeight(360)

        # Prompts
        self.user_edit = QPlainTextEdit()
        self.user_edit.setPlaceholderText("User prompt…")

        self.system_edit = QPlainTextEdit()
        self.system_edit.setPlaceholderText("System prompt…")

        self.prompt_pivot = Pivot(self)
        self.prompt_pivot.setContentsMargins(0, 0, 0, 0)
        self.prompt_pivot.setItemFontSize(12)  # ≈ default label size on Windows
        self.prompt_pivot.setFixedHeight(20)  # matches label height; keeps bar visible

        self.prompt_stack = QStackedLayout()

        def _wrap(editor):
            w = QWidget()
            l = QVBoxLayout(w)
            l.setContentsMargins(0, 0, 0, 0)  # bar should visually touch the editor
            l.setSpacing(0)
            l.addWidget(editor)
            return w

        self.prompt_stack.addWidget(_wrap(self.user_edit))  # index 0
        self.prompt_stack.addWidget(_wrap(self.system_edit))  # index 1

        # Short labels read fine at this size
        self.prompt_pivot.insertWidget(-1, "User", PivotItem("User", self.prompt_pivot))
        self.prompt_pivot.insertWidget(
            -1, "System", PivotItem("System", self.prompt_pivot)
        )

        # Nudge per-item padding down so the bar stays slim
        for item in self.prompt_pivot.items.values():
            item.setStyleSheet(
                "QPushButton{padding:0 8px 3px 8px; margin:0; min-height:18px;}"
            )

        # Header keeps the labels left, like the old tabs
        pivot_header = QHBoxLayout()
        pivot_header.setContentsMargins(0, 0, 0, 0)
        pivot_header.setSpacing(0)
        pivot_header.addWidget(self.prompt_pivot, 0, Qt.AlignLeft)
        pivot_header.addStretch(1)

        self.prompt_region = QWidget()
        pr = QVBoxLayout(self.prompt_region)
        pr.setContentsMargins(0, 0, 0, 0)
        pr.setSpacing(2)
        pr.addLayout(pivot_header)

        stack_host = QWidget()
        stack_host.setLayout(self.prompt_stack)
        pr.addWidget(stack_host, 1)

        self.prompt_pivot.currentItemChanged.connect(
            lambda key: self.prompt_stack.setCurrentIndex(0 if key == "User" else 1)
        )
        self.prompt_pivot.setCurrentItem("User")

        # Output
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setPlaceholderText("Model output will stream here…")
        self.copy_btn, self.clear_out_btn = PushButton("Copy Output"), PushButton(
            "Clear Output"
        )
        out_head = QHBoxLayout()
        out_head.addWidget(QLabel("Model Output"))
        out_head.addStretch(1)
        out_head.addWidget(self.copy_btn)
        out_head.addWidget(self.clear_out_btn)
        out_frame = QFrame()
        out_frame.setObjectName("Section")
        out_layout = QVBoxLayout(out_frame)
        out_layout.setContentsMargins(12, 8, 12, 12)
        out_layout.addLayout(out_head)
        out_layout.addWidget(self.output)

        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.addWidget(self.prompt_region)
        self.splitter.addWidget(out_frame)
        self.splitter.setSizes([700, 500])

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 16, 24, 16)
        root.setSpacing(UI_GAP)
        root.addLayout(top_row)
        root.addWidget(top_cluster)
        root.addWidget(self.splitter, 1)

        self.open_img_btn.clicked.connect(self.on_open_image)
        self.clear_img_btn.clicked.connect(self.on_clear_image)
        self.save_pair_btn.clicked.connect(self.on_save_pair)
        self.load_pair_btn.clicked.connect(self.on_load_pair)
        self.manage_btn.clicked.connect(self.on_manage_endpoints)
        self.run_btn.clicked.connect(self.on_run)
        self.stop_btn.clicked.connect(self.on_stop)
        self.clear_out_btn.clicked.connect(self.on_clear_output)
        self.copy_btn.clicked.connect(self.copy_output)

        for w, sig in [
            (self.endpoint_combo, self.endpoint_combo.currentIndexChanged),
            (self.max_tokens, self.max_tokens.valueChanged),
            (self.temperature, self.temperature.valueChanged),
            (self.top_p, self.top_p.valueChanged),
            (self.seed, self.seed.valueChanged),
            (self.randomize, self.randomize.toggled),
            (self.prompt_pivot, self.prompt_pivot.currentItemChanged),
        ]:
            sig.connect(self.schedule_save)
        self.user_edit.textChanged.connect(self.schedule_save)
        self.system_edit.textChanged.connect(self.schedule_save)

        self.load_app_state()

    def schedule_save(self):
        self._save_timer.start()

    def save_app_state(self):
        s = QSettings(APP_ORG, APP_NAME)
        s.setValue(self.KEY_EP_MODEL, self.endpoint_combo.currentText())
        s.setValue(self.KEY_EP_INDEX, int(self.endpoint_combo.currentIndex()))
        s.setValue(self.KEY_USER, self.user_edit.toPlainText())
        s.setValue(self.KEY_SYSTEM, self.system_edit.toPlainText())
        s.setValue(self.KEY_OUTPUT, self.output.toPlainText())
        s.setValue(self.KEY_IMAGE, self.current_image_path or "")
        s.setValue(self.KEY_TOKENS, int(self.max_tokens.value()))
        s.setValue(self.KEY_TEMP, float(self.temperature.value()))
        s.setValue(self.KEY_TOPP, float(self.top_p.value()))
        s.setValue(self.KEY_SEED, int(self.seed.value()))
        s.setValue(self.KEY_RANDOM, bool(self.randomize.isChecked()))
        s.setValue(self.KEY_SPLIT, self.splitter.sizes())
        s.setValue(
            self.KEY_PROMPT_ROUTE,
            "User" if self.prompt_stack.currentIndex() == 0 else "System",
        )
        s.setValue(
            self.KEY_IMG_DIR,
            (
                os.path.dirname(self.current_image_path)
                if self.current_image_path
                else s.value(self.KEY_IMG_DIR, os.path.expanduser("~"))
            ),
        )

    def load_app_state(self):
        s = QSettings(APP_ORG, APP_NAME)
        idx = s.value(self.KEY_EP_INDEX, None)
        if isinstance(idx, int) and 0 <= idx < self.endpoint_combo.count():
            self.endpoint_combo.setCurrentIndex(idx)
        else:
            epm = s.value(self.KEY_EP_MODEL, "", str)
            if epm:
                j = self.endpoint_combo.findText(epm)
                if j >= 0:
                    self.endpoint_combo.setCurrentIndex(j)
        self.user_edit.setPlainText(s.value(self.KEY_USER, "", str))
        self.system_edit.setPlainText(s.value(self.KEY_SYSTEM, "", str))
        self.output.setPlainText(s.value(self.KEY_OUTPUT, "", str))
        mt = s.value(self.KEY_TOKENS, None)
        self.max_tokens.setValue(int(mt) if mt is not None else self.max_tokens.value())
        tv = s.value(self.KEY_TEMP, None)
        self.temperature.setValue(
            float(tv) if tv is not None else self.temperature.value()
        )
        pv = s.value(self.KEY_TOPP, None)
        self.top_p.setValue(float(pv) if pv is not None else self.top_p.value())
        sd = s.value(self.KEY_SEED, None)
        self.seed.setValue(int(sd) if sd is not None else self.seed.value())
        rnd = s.value(self.KEY_RANDOM, None)
        if rnd is not None:
            self.randomize.setChecked(str(rnd).lower() in ("1", "true", "yes"))
        sizes = s.value(self.KEY_SPLIT)
        if isinstance(sizes, list) and sizes:
            try:
                self.splitter.setSizes([int(x) for x in sizes])
            except Exception:
                pass
        route = s.value(self.KEY_PROMPT_ROUTE, "User", str)
        self.prompt_pivot.setCurrentItem("User" if route != "System" else "System")
        self.prompt_stack.setCurrentIndex(0 if route != "System" else 1)
        path = s.value(self.KEY_IMAGE, "", str)
        if path:
            self.load_image_from_path(path)

    def refresh_endpoints(self):
        self.endpoint_combo.clear()
        names = self.store.display_names()
        if not names:
            self.endpoint_combo.addItem("(No endpoints configured)")
            self.endpoint_combo.setEnabled(False)
        else:
            self.endpoint_combo.addItems(names)
            self.endpoint_combo.setEnabled(True)

    def load_image_from_path(self, fp: str):
        try:
            img = Image.open(fp)
            self.current_image, self.current_image_path = img, fp
            self.preview.set_image(img)
            InfoBar.success(
                "Loaded image",
                os.path.basename(fp),
                parent=self,
                position=InfoBarPosition.TOP,
            )
            s = QSettings(APP_ORG, APP_NAME)
            s.setValue(self.KEY_IMAGE, fp)
            s.setValue(self.KEY_IMG_DIR, os.path.dirname(fp))
        except Exception as e:
            InfoBar.error(
                "Failed to load image",
                str(e),
                parent=self,
                position=InfoBarPosition.TOP,
            )

    def on_open_image(self):
        s = QSettings(APP_ORG, APP_NAME)
        start_dir = s.value(self.KEY_IMG_DIR, os.path.expanduser("~"), str)
        fp, _ = QFileDialog.getOpenFileName(
            self,
            "Select image",
            start_dir,
            "Images (*.png *.jpg *.jpeg *.bmp);;All files (*.*)",
        )
        if not fp:
            return
        self.load_image_from_path(fp)

    def on_clear_image(self):
        self.current_image = None
        self.current_image_path = None
        self.preview.clear_image()
        s = QSettings(APP_ORG, APP_NAME)
        s.setValue(self.KEY_IMAGE, "")

    def on_save_pair(self):
        s = QSettings(APP_ORG, APP_NAME)
        last_dir = s.value(self.KEY_PAIR_DIR, os.path.expanduser("~"), str)
        fp, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image+Prompt Pair",
            last_dir,
            "Cloud Interrogator Pair (*.ci.json);;JSON (*.json)",
        )
        if not fp:
            return
        try:
            data = {
                "image_path": self.current_image_path or "",
                "user_prompt": self.user_edit.toPlainText(),
                "system_prompt": self.system_edit.toPlainText(),
            }
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            s.setValue(self.KEY_PAIR_DIR, os.path.dirname(fp))
            InfoBar.success(
                "Saved", os.path.basename(fp), parent=self, position=InfoBarPosition.TOP
            )
            self.save_app_state()
        except Exception as e:
            InfoBar.error(
                "Save failed", str(e), parent=self, position=InfoBarPosition.TOP
            )

    def on_load_pair(self):
        s = QSettings(APP_ORG, APP_NAME)
        last_dir = s.value(self.KEY_PAIR_DIR, os.path.expanduser("~"), str)
        fp, _ = QFileDialog.getOpenFileName(
            self,
            "Load Image+Prompt Pair",
            last_dir,
            "Cloud Interrogator Pair (*.ci.json);;JSON (*.json)",
        )
        if not fp:
            return
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.user_edit.setPlainText(data.get("user_prompt", ""))
            self.system_edit.setPlainText(data.get("system_prompt", ""))

            path = data.get("image_path") or ""
            if path and os.path.exists(path):
                self.load_image_from_path(path)
            else:
                if path:
                    InfoBar.info(
                        "Missing image",
                        f"File not found: {path}",
                        parent=self,
                        position=InfoBarPosition.TOP,
                    )

            s.setValue(self.KEY_PAIR_DIR, os.path.dirname(fp))
            InfoBar.success(
                "Loaded",
                os.path.basename(fp),
                parent=self,
                position=InfoBarPosition.TOP,
            )
            self.save_app_state()
        except Exception as e:
            InfoBar.error(
                "Load failed", str(e), parent=self, position=InfoBarPosition.TOP
            )

    def on_manage_endpoints(self):
        dlg = EndpointManagerDialog(self.store, self)
        if dlg.exec() == QDialog.Accepted:
            self.refresh_endpoints()
            self.save_app_state()

    def _resolve_endpoint(self) -> Optional[Tuple[str, str, str]]:
        det = self.store.details_from_display(self.endpoint_combo.currentText())
        if not det:
            return None
        base_url, api_key, model = det
        if not model:
            InfoBar.error(
                "Invalid model",
                "Select an endpoint with a model.",
                parent=self,
                position=InfoBarPosition.TOP,
            )
            return None
        return base_url, api_key, model

    def on_run(self):
        if self.streamer and self.streamer.isRunning():
            InfoBar.info(
                "Busy",
                "Already running; stop first.",
                parent=self,
                position=InfoBarPosition.TOP,
            )
            return
        resolved = self._resolve_endpoint()
        if not resolved:
            return
        base_url, api_key, model = resolved
        image_url = (
            encode_image(self.current_image, 512, 85)
            if self.current_image is not None
            else None
        )
        seed = self.seed.value()
        if self.randomize.isChecked():
            seed = random.randint(0, 2**31 - 1)
            self.seed.setValue(seed)
        self.output.clear()
        self.set_busy(True, "Sending…")
        InfoBar.info(
            "Request", "Sent to server.", parent=self, position=InfoBarPosition.TOP
        )
        self.streamer = ChatStreamer(
            base_url,
            api_key,
            model,
            self.system_edit.toPlainText(),
            self.user_edit.toPlainText(),
            image_url,
            int(self.max_tokens.value()),
            float(self.temperature.value()),
            float(self.top_p.value()),
            int(seed),
        )
        self.streamer.token.connect(self.on_stream_token)
        self.streamer.done.connect(self.on_stream_done)
        self.streamer.failed.connect(self.on_stream_failed)
        self.streamer.start()

    def on_stop(self):
        if self.streamer and self.streamer.isRunning():
            self.streamer.cancel()
            self.streamer.wait(1500)
            self.set_busy(False, "Stopped")
            InfoBar.info(
                "Stopped",
                "Streaming stopped.",
                parent=self,
                position=InfoBarPosition.TOP,
            )
            self.save_app_state()

    def on_stream_token(self, text: str):
        sb = self.output.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 4
        c = QTextCursor(self.output.document())
        c.movePosition(QTextCursor.End)
        c.insertText(text)
        if at_bottom:
            sb.setValue(sb.maximum())
        if self.status_lbl.text() != "Streaming…":
            self.status_lbl.setText("Streaming…")

    def on_stream_done(self):
        self.set_busy(False, "Done")
        InfoBar.success(
            "Done", "Streaming complete.", parent=self, position=InfoBarPosition.TOP
        )
        self.save_app_state()

    def on_stream_failed(self, err: str):
        self.set_busy(False, "Failed")
        InfoBar.error("Request failed", err, parent=self, position=InfoBarPosition.TOP)
        self.save_app_state()

    def on_clear_output(self):
        self.output.clear()
        self.save_app_state()

    def copy_output(self):
        QGuiApplication.clipboard().setText(self.output.toPlainText())
        InfoBar.success(
            "Copied",
            "Output copied to clipboard.",
            parent=self,
            position=InfoBarPosition.TOP,
        )

    def set_busy(self, busy: bool, status: str):
        self.status_lbl.setText(status)
        self.run_btn.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)


def get_accent_color() -> QColor:
    if sys.platform == "win32":
        try:
            import winaccent

            return QColor(winaccent.accent_dark)
        except Exception:
            pass
    return QColor("#0099FF")


def main():
    app = QApplication(sys.argv)
    setTheme(Theme.DARK)
    setThemeColor(get_accent_color())
    win = FluentWindow()
    win.setWindowTitle("Cloud Interrogator")
    exe_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    icon_path = os.path.join(exe_dir, "icon.ico")

    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(":/icon.ico"))
        win.setWindowIcon(QIcon(":/icon.ico"))
    s = QSettings(APP_ORG, APP_NAME)
    size = s.value("window/size")
    pos = s.value("window/pos")
    if isinstance(size, QSizeObj):
        win.resize(size)
    else:
        win.resize(900, 1200)
    if isinstance(pos, QPoint):
        win.move(pos)
    store = EndpointStore(ENDPOINTS_FILE)
    panel = MainPanel(store)

    def on_close():
        panel.save_app_state()
        s.setValue("window/size", win.size())
        s.setValue("window/pos", win.pos())

    win.on_close = on_close
    layout = QVBoxLayout(win)
    layout.setContentsMargins(0, win.titleBar.height(), 0, 0)
    layout.addWidget(panel)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
