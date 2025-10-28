import shutil
import traceback
import tempfile
import math
import random
import sys, os
from types import SimpleNamespace
from functools import partial

# Windows系统：设置AppUserModelID，确保任务栏图标正确分组和显示
if sys.platform.startswith("win"):
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('MultiscaleDesign.V1.0')
    except Exception:
        pass

# PyInstaller打包后，stdout/stderr可能为None，重定向到os.devnull避免崩溃
if getattr(sys, 'frozen', False):
    if sys.stdout is None:
        sys.stdout = open(os.devnull, 'w')
    if sys.stderr is None:
        sys.stderr = open(os.devnull, 'w')

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from PyQt6.QtCore import Qt, QSize, QTimer, QUrl, pyqtSignal, QThread
from PyQt6.QtGui import (
    QAction, QIcon, QPixmap, QImage, QDesktopServices, QBrush, QColor
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QListWidget, QListWidgetItem,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QPushButton, QTabWidget,
    QFormLayout, QLineEdit, QComboBox, QPlainTextEdit, QDialog, QDialogButtonBox,
    QSplitter, QStackedWidget, QToolBar, QFileDialog,
    QProgressBar, QStatusBar, QSpinBox, QDoubleSpinBox, QToolButton, QMenu,
    QMessageBox, QSizePolicy, QStyle, QToolTip, QFileIconProvider,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView
)
# 所有消息框以“主界面所在屏幕中心”居中
from PyQt6.QtCore import QObject, QEvent
from PyQt6.QtGui import QGuiApplication

class _CenterOnMainScreenFilter(QObject):
    """在显示和首次调整大小/布局请求时，将对话框移动到主界面所在屏幕的中心。
    说明：某些原生样式会在显示后二次调整布局导致尺寸变化，单次居中可能造成视觉偏差。
    因此这里在 Show 后用 singleShot(0, ...) 再居中一次，并在第一次 Resize/LayoutRequest 时再做一次（至多2次），避免抖动。
    """
    def __init__(self, main_window):
        super().__init__(main_window)
        self._mw = main_window  # 期望是 MainWindow 或其子部件

    def _center_once(self, w: QWidget):
        try:
            _move_window_to_screen_center(w, self._mw)
            # 标记次数
            n = int(w.property("_centered_passes") or 0)
            w.setProperty("_centered_passes", n + 1)
        except Exception:
            pass

    def eventFilter(self, obj, e):
        et = e.type()
        if et == QEvent.Type.Show:
            try:
                obj.setProperty("_centered_passes", 0)
            except Exception:
                pass
            # 立即和下一轮事件循环各居中一次
            self._center_once(obj)
            QTimer.singleShot(0, lambda: self._center_once(obj))
        elif et in (QEvent.Type.Resize, QEvent.Type.LayoutRequest):
            # 若尺寸/布局第一次变化，再居中一次以适配自动换行/原生样式调整
            try:
                n = int(obj.property("_centered_passes") or 0)
            except Exception:
                n = 0
            if n < 2:
                self._center_once(obj)
        return False

def _move_window_to_screen_center(window: QWidget, base: Optional[QWidget]=None):
    """将窗口移动到base窗口所在屏幕的中心；若base为None则使用系统主屏。"""
    try:
        s = None
        if base is not None and getattr(base, 'windowHandle', None):
            wh = base.windowHandle()
            if wh is not None:
                s = wh.screen()
        if s is None:
            center = base.frameGeometry().center() if base is not None else None
            s = (QGuiApplication.screenAt(center) if center is not None else None) or QGuiApplication.primaryScreen()
        pr = s.availableGeometry()
        frame = window.frameGeometry()
        if frame.width() <= 0 or frame.height() <= 0:
            sz = window.sizeHint()
            frame.setWidth(max(1, sz.width()))
            frame.setHeight(max(1, sz.height()))
        frame.moveCenter(pr.center())
        window.move(frame.topLeft())
    except Exception:
        pass

def _exec_centered_message_box_on_main(box, main_window):
    """在显示前安装“主界面屏幕中心”过滤器."""
    try:
        box.installEventFilter(_CenterOnMainScreenFilter(main_window))
    except Exception:
        pass
    return box.exec()

def _show_centered_message(parent, icon, title, text, *args, **kwargs):
    """通用居中消息框辅助函数"""
    buttons = args[0] if len(args) >= 1 else kwargs.get('buttons', QMessageBox.StandardButton.Ok)
    default_btn = args[1] if len(args) >= 2 else kwargs.get('defaultButton', QMessageBox.StandardButton.NoButton)
    box = QMessageBox(icon, title, text, buttons, parent)
    if default_btn and isinstance(default_btn, QMessageBox.StandardButton):
        try: box.setDefaultButton(default_btn)
        except Exception: pass
    return _exec_centered_message_box_on_main(box, parent)

def msg_information(parent, title, text, *args, **kwargs):
    """显示一个居中于父窗口屏幕的“信息”对话框。"""
    return _show_centered_message(parent, QMessageBox.Icon.Information, title, text, *args, **kwargs)

def msg_warning(parent, title, text, *args, **kwargs):
    """显示一个居中于父窗口屏幕的“警告”对话框。"""
    return _show_centered_message(parent, QMessageBox.Icon.Warning, title, text, *args, **kwargs)

def msg_critical(parent, title, text, *args, **kwargs):
    """显示一个居中于父窗口屏幕的“严重错误”对话框。"""
    return _show_centered_message(parent, QMessageBox.Icon.Critical, title, text, *args, **kwargs)

def msg_question(parent, title, text, *args, **kwargs):
    """显示一个居中于父窗口屏幕的“提问”对话框。"""
    # 确保 'buttons' 默认为 Yes|No
    kwargs.setdefault('buttons', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
    return _show_centered_message(parent, QMessageBox.Icon.Question, title, text, *args, **kwargs)

# 机器学习/深度学习功能集成
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
_ORIG_LOAD_MODEL = load_model

# 模型缓存机制：根据文件路径和修改时间避免重复加载
import threading, weakref

class _ModelCache:
    """
    Keras/TensorFlow模型缓存。
    使用weakref和文件元数据（路径、大小、修改时间）来缓存已加载的模型，
    避免在同一会话中重复从磁盘加载相同的模型。
    """
    _lock = threading.Lock()
    _cache = {}

    @classmethod
    def get(cls, path: str, custom_objects=None):
        if not path:
            return None
        ap = os.path.abspath(path)
        try:
            st = os.stat(ap)
            mtime_ns = st.st_mtime_ns
            size = st.st_size
        except Exception:
            mtime_ns = 0
            size = 0
        key = (ap, size, mtime_ns, bool(custom_objects))
        with cls._lock:
            ref = cls._cache.get(key)
            mdl = ref() if ref else None
            if mdl is not None:
                return mdl
        # 真实加载
        mdl = _ORIG_LOAD_MODEL(ap, custom_objects=custom_objects) if custom_objects else _ORIG_LOAD_MODEL(ap)
        with cls._lock:
            cls._cache[key] = weakref.ref(mdl)
        return mdl

from tensorflow.keras import backend as K
import pandas as pd
import time
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import joblib
import torch
import torch.nn as nn

# 集成matplotlib绘图库
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import rcParams
import json

# 抑制已知的 PyQt6/sip 废弃警告，保持控制台输出干净
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='Tight layout not applied.*', category=UserWarning)

# Matplotlib 字体与负号显示设置（确保中文与负号正常显示）
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# 全局变量定义
APP_NAME = '航空复合材料多层级智能设计平台'  # 应用程序名称
APP_DIR = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))  # 程序根目录（PyInstaller 打包后为临时目录 _MEIPASS；否则为当前脚本目录）
ASSETS_DIR = os.path.join(APP_DIR, 'assets')  # 资源文件目录（图标、图片等）
USER_DOCS_DIR = os.path.join(APP_DIR, 'Manuals and Tutorials')  # 用户文档目录（手册/教程）
INTRO_DIR = os.path.join(USER_DOCS_DIR, 'Example')  # 入门示例目录
SAMPLES_DIR = os.path.join(USER_DOCS_DIR, 'Example of fine-tuning data')  # 微调数据示例目录
MODEL_DIR = os.path.join(APP_DIR, 'Model')  # 机器学习/深度学习模型文件目录
INTERNAL_MS_DB = os.path.join(APP_DIR, 'Microstructure-database')  # 内置微结构数据库目录
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')  # 统一可识别图片扩展名（大小写兼容）

# 在 assets 目录中按关键词优先级选择一张最合适的横幅图片，并返回其路径。
def find_banner_image():
    if not os.path.isdir(ASSETS_DIR):
        return ''
    files = []
    for fn in os.listdir(ASSETS_DIR):
        low = fn.lower()
        if low.endswith(('.png', '.jpg', '.jpeg')):
            files.append(fn)
    priority = ['多层级', 'multiscale', 'multi', 'design', 'banner', 'arrow']

    def score(fn):
        low = fn.lower()
        for i, k in enumerate(priority):
            if k in low:
                return i
        return 999
    files.sort(key=lambda fn: (score(fn), fn.lower()))
    return os.path.join(ASSETS_DIR, files[0]) if files else ''

# 从系统样式中获取标准图标；作为资源缺失或未提供自定义图标时的回退方案。
def get_std_icon(name: str, style_widget=None):
    sw = style_widget or (QApplication.instance().activeWindow() if QApplication.instance() else None)
    st = sw.style() if sw else QApplication.instance().style() if QApplication.instance() else None
    if st is None:
        return QIcon()
    mp_map = {
        "document-new": QStyle.StandardPixmap.SP_FileIcon,
        "document-open": QStyle.StandardPixmap.SP_DialogOpenButton,
        "document-save": QStyle.StandardPixmap.SP_DialogSaveButton,
        "help-contents": QStyle.StandardPixmap.SP_MessageBoxInformation,
        "folder": QStyle.StandardPixmap.SP_DirIcon,
        "applications-engineering": QStyle.StandardPixmap.SP_ComputerIcon,
        "edit-clear": QStyle.StandardPixmap.SP_LineEditClearButton,
        "go-home": QStyle.StandardPixmap.SP_DirHomeIcon,
        "go-previous": QStyle.StandardPixmap.SP_ArrowBack,
        "view-list-details": QStyle.StandardPixmap.SP_FileDialogDetailedView,
        "image-x-generic": QStyle.StandardPixmap.SP_FileIcon,
    }
    mp = mp_map.get(name, QStyle.StandardPixmap.SP_FileIcon)
    try:
        return st.standardIcon(mp)
    except Exception:
        return QIcon()
try:
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as _BaseToolbar
except Exception:
    try:
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as _BaseToolbar
    except Exception:
        _BaseToolbar = None

# Matplotlib导航工具栏包装类：统一图标尺寸、隐藏坐标显示、自适应窗口宽度
class AutoNavToolbar(_BaseToolbar if _BaseToolbar is not None else object):
    """
    自定义Matplotlib导航工具栏。
    - 统一图标尺寸
    - 隐藏坐标显示（2D图），保留（3D图）
    - 移除分隔符和边框
    """
    def __init__(self, canvas, parent=None):
        if _BaseToolbar is None:
            return
        super().__init__(canvas, parent)
        self.setIconSize(QSize(18, 18))  # 设置工具栏图标统一尺寸，确保按钮正常显示
        self.setContentsMargins(0, 0, 0, 0)  # 去除工具栏外边距，使布局更紧凑
        # 检查画布是否包含3D坐标轴
        self._is_3d_plot = False
        try:
            # 检查画布上的任何一个 axes 的 'name' 是否为 '3d'
            self._is_3d_plot = any(getattr(ax, 'name', '') == '3d' for ax in self.canvas.figure.get_axes())
        except Exception:
            self._is_3d_plot = False  # 出错时默认为 2D
        for act in list(self.actions()):
            if act.isSeparator():
                self.removeAction(act)
        self.setStyleSheet('QToolBar{border:none;spacing:6px;} '
                           'QToolBar::separator{width:0px;height:0px;} '
                           'QToolButton{border:0;margin:0;padding:0 2px;}')  # 通过样式表隐藏工具栏分隔线/边框，避免占位影响
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_message(self, s):
        # 如果是 3D 绘图 (如层合板)，则正常显示坐标
        if self._is_3d_plot:
            # 调用基类 (NavigationToolbar2QT) 的 set_message
            _BaseToolbar.set_message(self, s)
        else:
            # 如果是 2D 绘图 (已有自定义悬浮标签)，则传递空字符串
            # 这样既能清除旧消息，也不会显示新消息（包括颜色值）
            _BaseToolbar.set_message(self, "")

    def resizeEvent(self, e):
        return super().resizeEvent(e)

# 横幅图片标签：在给定高度内自适应等比例缩放并居中显示横幅图片。
class BannerLabel(QLabel):

    def __init__(self, img_path: str, fixed_height: int=220, parent=None):
        super().__init__(parent)
        self.setObjectName('bannerImage')
        self._src_path = img_path or ''
        self._base = QPixmap(self._src_path) if self._src_path and os.path.exists(self._src_path) else QPixmap()
        self.is_valid = not self._base.isNull()
        # 样式设置
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._fixed_h = int(fixed_height)
        if self._fixed_h > 0:
            self.setMinimumHeight(self._fixed_h)
            self.setMaximumHeight(self._fixed_h)
        if self.is_valid:
            self._update_scaled()
        # 设置一个柔和的卡片式背景
        self.setStyleSheet('#bannerImage { background: #F7FAFF; border: 1px solid #E5E7EB; border-radius: 8px; }')

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.is_valid:
            self._update_scaled()

    def _update_scaled(self):
        if self._base.isNull():
            return
        w = max(10, self.width() - 12)
        h = self._fixed_h if self._fixed_h > 0 else max(10, self.height() - 12)
        pm = self._base.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(pm)

# 优先从 assets 目录加载中文命名的 PNG 图标（如“微结构”），若不存在则回退到系统标准图标。
def get_asset_icon(ch_name: str, fallback_name: str='file'):
    p = os.path.join(ASSETS_DIR, f'{ch_name}.png')
    if os.path.exists(p):
        return QIcon(p)
    # 回退到标准图标
    try:
        return get_std_icon(fallback_name)
    except Exception:
        return QIcon()

# 为 Matplotlib 画布设置保守的尺寸与上限（最大宽度、最小高度），避免布局反馈造成的无限放大或工具栏拥挤。
def protect_canvas(canvas_widget):
    _MAX_CANVAS_W = 1960  # 画布最大宽度上限，防止布局反馈导致无限变宽（保护渲染器）
    _MIN_CANVAS_H = 220  # 画布最小高度，保证工具栏与坐标标注显示空间
    canvas_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    canvas_widget.setMinimumHeight(_MIN_CANVAS_H)
    canvas_widget.setMaximumWidth(_MAX_CANVAS_W)

# 在坐标轴区域内绑定一个随鼠标移动的悬浮坐标标签，用于显示 (x, y) 值，替代工具栏右上角的坐标显示。
def attach_hover_coords(ax, canvas, fmt='({x:.3g}, {y:.3g})'):
    ctx = SimpleNamespace(ax=ax, canvas=canvas, fmt=fmt, ann=None, renderer=None)
    import time as _t
    ctx.last_ts=0.0
    ann_list = [None]

    def ensure_ann(ctx):
        a = ctx.ann
        needs_new = False
        try:
            if a is None:
                needs_new = True
            else:
                ax_texts = getattr(ctx.ax, 'texts', [])
                ax_artists = getattr(ctx.ax, 'artists', [])
                if getattr(a, 'axes', None) is None or (a not in ax_texts and a not in ax_artists):
                    needs_new = True
        except Exception:
            needs_new = True
        if needs_new:
            a = ctx.ax.annotate('', xy=(0, 0), xytext=(8, 8), textcoords='offset points', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.75, ec='#666'), fontsize=9, color='#111', zorder=10)
            a.set_visible(False)
            a.set_in_layout(False)
            a.set_clip_on(True)  # 遵循坐标轴边界
            a.set_ha('left')
            a.set_va('bottom')
            ctx.ann = a
            ann_list[0] = a
        return ctx.ann

    # 缓存渲染器
    def _get_renderer(ctx):
        if ctx.renderer is None:
            try:
                ctx.renderer = ctx.canvas.get_renderer()
            except Exception:
                try:
                    ctx.canvas.draw()
                    ctx.renderer = ctx.canvas.get_renderer()
                except Exception:
                    ctx.renderer = None
        return ctx.renderer

    def _flip_to_fit(a, ev, ctx):
        # 在显示/像素坐标系中工作
        try:
            ax_box = ctx.ax.get_window_extent()
        except Exception:
            try:
                ctx.canvas.draw()
                ax_box = ctx.ax.get_window_extent()
            except Exception:
                return
        rnd = _get_renderer(ctx)
        try:
            bbox = a.get_window_extent(renderer=rnd)
            tw, th = (bbox.width, bbox.height)
        except Exception:
            tw, th = (80, 18)  # 安全回退
        # 从右上方偏移开始
        off_x, off_y = (8, 8)
        halign, valign = ('left', 'bottom')
        # 鼠标在画布内的像素位置
        px, py = (ev.x, ev.y)
        # 若右侧越界则翻转到左侧
        if px + off_x + tw > ax_box.x1 - 2:
            off_x = -8
            halign = 'right'
        # 若左侧越界则保持右侧
        if px + off_x < ax_box.x0 + 2:
            off_x = 8
            halign = 'left'
        # 若顶部越界则翻转到底部
        if py + off_y + th > ax_box.y1 - 2:
            off_y = -8
            valign = 'top'
        # 若底部越界则保持顶部
        if py + off_y < ax_box.y0 + 2:
            off_y = 8
            valign = 'bottom'
        a.set_ha(halign)
        a.set_va(valign)
        a.set_position((off_x, off_y))  # 更新xy文本

    def on_move(event, ctx):
        if event.inaxes is not ctx.ax or event.xdata is None or event.ydata is None:
            a = ctx.ann
            if a is not None and a.get_visible():
                a.set_visible(False)
                ctx.canvas.draw_idle()
            return
        a = ensure_ann(ctx)
        a.xy = (event.xdata, event.ydata)
        try:
            a.set_text(ctx.fmt.format(x=float(event.xdata), y=float(event.ydata)))
        except Exception:
            try:
                a.set_text(f'({event.xdata:.3g}, {event.ydata:.3g})')
            except Exception:
                a.set_text('')
        _flip_to_fit(a, event, ctx)
        if not a.get_visible():
            a.set_visible(True)
        # 节流绘制：~60 FPS 上限
        now = _t.perf_counter()
        if (not a.get_visible()) or (now - ctx.last_ts) >= 0.016:
            ctx.last_ts = now
            ctx.canvas.draw_idle()

    def on_leave(event, ctx):
        a = ctx.ann
        if a is not None and a.get_visible():
            a.set_visible(False)
            ctx.canvas.draw_idle()
    canvas.mpl_connect('motion_notify_event', partial(on_move, ctx=ctx))  # 绑定鼠标移动事件以更新悬浮坐标标签
    canvas.mpl_connect('figure_leave_event', partial(on_leave, ctx=ctx))  # 鼠标离开坐标轴后隐藏悬浮坐标标签
    return ann_list

# 根据层级名称打开对应的“入门示例”PDF，若缺失则给出提示。
def open_intro_pdf(level: str, parent=None):
    mapping = {'微结构': '微结构入门示例.pdf', '层合板': '层合板入门示例.pdf', '加筋结构': '加筋结构入门示例.pdf', '机身段': '机身段入门示例.pdf'}
    fname = mapping.get(level)
    if not fname:
        msg_warning(parent, '未配置', f'未为层级 “{level}” 配置入门示例。')
        return
    path = os.path.join(INTRO_DIR, fname)
    if not os.path.exists(path):
        msg_warning(parent, '未找到文件', '未在入门示例目录找到：\n' + path)
        return
    ok = QDesktopServices.openUrl(QUrl.fromLocalFile(path))
    if not ok:
        msg_warning(parent, '打开失败', '系统未能打开：\n' + path)

# 将预置的“微调数据示例”保存到用户选定目录（并处理覆盖确认与异常提示）。
def save_samples_xlsx(level: str, parent=None):
    mapping = {'微结构': '微结构模型微调数据示例.xlsx', '层合板': '层合板模型微调数据示例.xlsx', '加筋结构': '加筋结构模型微调数据示例.xlsx', '机身段': '机身段模型微调数据示例.xlsx'}
    fname = mapping.get(level)
    if not fname:
        msg_warning(parent, '未配置', f'未为层级 “{level}” 配置微调数据示例。')
        return
    src_path = os.path.join(SAMPLES_DIR, fname)
    if not os.path.exists(src_path):
        msg_warning(parent, '文件不存在', f'未在“Example of fine-tuning data”目录找到：\n{src_path}')
        return
    default_dir = os.path.expanduser('~')
    save_dir = get_existing_directory(parent, f'选择保存目录（{level} 微调数据示例）', default_dir)
    if not save_dir:
        return
    dst_path = os.path.join(save_dir, fname)
    try:
        if os.path.exists(dst_path):
            reply = msg_question(parent, '文件已存在', f'目标文件已存在：\n{dst_path}\n是否覆盖？', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                return
        shutil.copy2(src_path, dst_path)
        msg_information(parent, '保存成功', f'已保存到：\n{dst_path}')
    except Exception as e:
        msg_warning(parent, '保存失败', f'无法保存示例：{e}')

# 打开帮助文档 PDF，若缺失或系统关联失败则提示。
def open_help_pdf(parent=None):
    path = os.path.join(USER_DOCS_DIR, '帮助文档.pdf')
    if not os.path.exists(path):
        msg_warning(parent, '未找到帮助文档', '未在以下位置找到帮助文档：\n' + path)
        return
    ok = QDesktopServices.openUrl(QUrl.fromLocalFile(path))
    if not ok:
        msg_warning(parent, '打开失败', '系统未能打开：\n' + path)

# 稳健读取图像：支持中文/长路径读取，输出为 uint8 BGR 图像；失败返回 None。
def safe_imread(path: str, as_color: bool=True):
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except Exception:
        return None
    flag = cv2.IMREAD_COLOR if as_color else cv2.IMREAD_UNCHANGED
    img = cv2.imdecode(data, flag)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

# 微结构图像统一预处理：resize→灰度→阈值二值化→[-0.5, 0.5] 归一化→堆叠为 3 通道 float32。
def preprocess_ms_image(path: str, IMG_W: int=224, IMG_H: int=224):
    img = safe_imread(path, as_color=True)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 5.1, 255, cv2.THRESH_BINARY)
    dst = bw.astype('float32') / 255.0 - 0.5
    return np.stack([dst, dst, dst], axis=-1).astype('float32')

# 收集指定目录中的图像路径（可递归/随机打乱），返回排序稳定的绝对路径列表。
def gather_images(folder: str, exts=IMG_EXTS, recursive: bool=False, shuffle: bool=False, seed: Optional[int]=None):
    paths = []
    if not folder or not os.path.isdir(folder):
        return paths
    if recursive:
        for root, _, files in os.walk(folder):
            for fn in files:
                lower = fn.lower()
                if any((lower.endswith(e.lower()) for e in exts)):
                    paths.append(os.path.abspath(os.path.join(root, fn)))
    else:
        for fn in os.listdir(folder):
            lower = fn.lower()
            if any((lower.endswith(e.lower()) for e in exts)):
                paths.append(os.path.abspath(os.path.join(folder, fn)))
    paths.sort()
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(paths)
    return paths

# --- 机器学习模型辅助代码 ---
# Keras 自定义 RMSE 指标函数（用于模型加载或训练监控）。
def rmse_metric(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# Keras 数据生成器：统一训练/预测的数据批组织方式；训练时每图像 4 个样本，预测时每图像 100 个样本。
class CustomDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, images, x_true=None, y_true=None, batch_size=None, shuffle=True, is_training=True, input_names=None):
        # 规范化为 4D 图像张量 (N, H, W, C)
        imgs = np.asarray(images)
        if imgs.ndim == 3:
            imgs = imgs[None, ...]
        self.images = imgs.astype(np.float32)
        self.n_images = self.images.shape[0]
        self.x_true = None if x_true is None else np.asarray(x_true, dtype=np.float32).reshape(-1, 1)
        self.y_true = None if y_true is None else np.asarray(y_true, dtype=np.float32)
        self.batch_size = int(batch_size) if batch_size else len(self.images)
        self.shuffle = shuffle
        self.is_training = bool(is_training)
        self.input_names = list(input_names) if input_names is not None else None
        total = self.n_images * (4 if self.is_training else 100)
        self.indexes = np.arange(total, dtype=np.int64)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(math.ceil(len(self.indexes) / float(self.batch_size)))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.indexes))
        idx = self.indexes[start:end]
        real_index = idx % self.n_images
        batch_images = self.images[real_index].astype(np.float32)
        batch_x = None if self.x_true is None else self.x_true[start:end].astype(np.float32)
        if self.is_training:
            batch_y = None if self.y_true is None else self.y_true[start:end].astype(np.float32)
            return ({self.input_names[0] if self.input_names else 0: batch_images, self.input_names[1] if self.input_names else 1: batch_x}, batch_y)
        else:
            return {self.input_names[0] if self.input_names else 0: batch_images, self.input_names[1] if self.input_names else 1: batch_x}
# --- 辅助代码结束 ---

# 工具：长文本断行（富文本）
class _TextUtil:

    @staticmethod
    def html_escape(s: str) -> str:
        return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    @staticmethod
    def wrap_breakable(s: str) -> str:
        # pre-wrap 保留空格/换行；break-all 允许任意处断开（适合长路径/无空格）
        return f"<div style='white-space:pre-wrap; word-break:break-all; overflow-wrap:anywhere'>{_TextUtil.html_escape(s)}</div>"

    @staticmethod
    def elide_text(text: str, max_chars: int=60) -> str:
        """
        返回"text"的缩短版本，适合"max_chars"字符。如果字符串太长，则将其中间替换为省略号。
        此方法主要用于截断标签中的文件系统路径。调用者应该将完整路径分配给小部件的工具提示以供参考。
        """
        try:
            if not isinstance(text, str):
                text = str(text)
            if len(text) <= max_chars:
                return text
            avail = max_chars - 3
            end_len = max(10, avail // 3)
            start_len = avail - end_len
            return text[:start_len] + '...' + text[-end_len:]
        except Exception:
            return text

# 可自动“中间省略”的标签：根据当前宽度截断中间部分，完整文本放在 tooltip 中，同时减少不必要重绘避免闪烁。
class ElidedLabel(QLabel):

    def __init__(self, text: str='', parent: Optional[QWidget]=None):
        super().__init__(parent)
        self.full_text = text
        self.setWordWrap(False)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._cached_width: int = -1
        self._cached_display: str = ''
        self.update_elided()

    def set_full_text(self, text: str):
        """
        设置标签的全文。如果新文本与现有全文相同，则不会更新任何内容。这个保护可以防止不必要的调用"update_elided"。
        这可能会导致重复的重画和可见的闪烁，当相同的路径被多次分配（例如，当响应重复的配置变化信号）。传递空值或“None”值将清除全文。
        """
        try:
            new_text = text or ''
            # 避免在文本未实际更改时触发重绘
            if new_text == getattr(self, 'full_text', ''):
                return
            self.full_text = new_text
            self.update_elided()
        except Exception:
            # 出错时回退到简单赋值
            self.full_text = text or ''
            self.update_elided()

    def update_elided(self):
        """根据当前标签宽度，计算并设置中间省略的文本。"""
        try:
            fm = self.fontMetrics()
            available_width = max(0, self.width() - 4)  # 减去一个小边距
            '''
            如果宽度为 0，则退化为固定长度截断。
            当小部件的宽度尚未确定（例如在初始布局时为0或1）时，基于该宽度计算省略文本会生成一个短字符串，一旦知道实际宽度，该字符串将立即被替换。
            这会产生分散注意力的闪烁。在这种情况下，退回到一个固定的字符计数省略，并避免更新标签，如果实际上没有什么变化。
            '''
            if available_width <= 0:
                display = _TextUtil.elide_text(self.full_text, 60)
            else:
                display = fm.elidedText(self.full_text or '', Qt.TextElideMode.ElideMiddle, available_width)
            # 仅在可用宽度或显示文本发生更改时更新
            if available_width == self._cached_width and display == self._cached_display:
                return
            self._cached_width = available_width
            self._cached_display = display
            super().setText(display)
            super().setToolTip(self.full_text if self.full_text else '')
        except Exception:
            super().setText(self.full_text)
            super().setToolTip(self.full_text)

    def _wrap_tooltip_text(self, text: str) -> str:
        """为工具提示（Tooltip）生成自动换行的富文本HTML。"""
        try:
            fm = self.fontMetrics()
            max_px = max(120, self.width() - 12)
            lines = []
            cur = ''
            for ch in text:
                if fm.horizontalAdvance(cur + ch) > max_px and cur:
                    lines.append(cur)
                    cur = ch
                else:
                    cur += ch
            if cur:
                lines.append(cur)
            html = '<div style="white-space:pre-wrap; word-break:break-all;">' + _TextUtil.html_escape('\n'.join(lines)) + '</div>'
            return html
        except Exception:
            return _TextUtil.wrap_breakable(text)

    def event(self, e):
        """在光标附近显示一个包装的工具提示，绑定到当前标签宽度。"""
        if e.type() == QEvent.Type.ToolTip:
            full = getattr(self, 'full_text', '') or ''
            if full:
                QToolTip.showText(e.globalPos(), self._wrap_tooltip_text(full), self)
                return True
        return super().event(e)

    def resizeEvent(self, e):
        self.update_elided()
        super().resizeEvent(e)

# 将 QFileDialog（非原生对话框）居中到屏幕，并设置图标提供器以保证工具按钮图标显示正常。
def _center_dialog_on_parent(dialog: QFileDialog, parent: QWidget) -> None:
    dialog.setIconProvider(QFileIconProvider())
    dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
    # 以“主界面所在屏幕”为准居中
    s = None
    try:
        if parent is not None and getattr(parent, 'windowHandle', None):
            wh = parent.windowHandle()
            if wh is not None:
                s = wh.screen()
    except Exception:
        s = None
    if s is None:
        center = parent.frameGeometry().center() if parent is not None else None
        s = (QGuiApplication.screenAt(center) if center is not None else None) or QGuiApplication.primaryScreen()
    pr = s.availableGeometry()
    sz = dialog.sizeHint()
    w, h = (max(1, sz.width()), max(1, sz.height()))
    x = pr.center().x() - w // 2
    y = pr.center().y() - h // 2
    dialog.move(int(x), int(y))

def get_open_file(parent: QWidget, title: str, filter: str='', default_dir: str='') -> Tuple[str, str]:
    """弹出“打开文件”对话框（居中），返回选中文件路径与过滤器；取消返回 ('', '')。"""
    dlg = QFileDialog(parent, title, default_dir, filter)
    dlg.setFileMode(QFileDialog.FileMode.ExistingFile)
    _center_dialog_on_parent(dlg, parent)
    if dlg.exec() == QFileDialog.DialogCode.Accepted:
        files = dlg.selectedFiles()
        filt = dlg.selectedNameFilter()
        return (files[0] if files else '', filt or '')
    return ('', '')

def get_save_file(parent: QWidget, title: str, filter: str='', default_dir: str='') -> Tuple[str, str]:
    """弹出“保存文件”对话框（居中），返回保存路径与过滤器；取消返回 ('', '')。"""
    dlg = QFileDialog(parent, title, default_dir, filter)
    dlg.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
    dlg.setFileMode(QFileDialog.FileMode.AnyFile)
    _center_dialog_on_parent(dlg, parent)
    if dlg.exec() == QFileDialog.DialogCode.Accepted:
        files = dlg.selectedFiles()
        filt = dlg.selectedNameFilter()
        return (files[0] if files else '', filt or '')
    return ('', '')

def get_existing_directory(parent: QWidget, title: str, default_dir: str='') -> str:
    """弹出“选择文件夹”对话框（居中），返回所选目录；取消返回空字符串。"""
    dlg = QFileDialog(parent, title, default_dir)
    dlg.setFileMode(QFileDialog.FileMode.Directory)
    dlg.setOption(QFileDialog.Option.ShowDirsOnly, True)
    _center_dialog_on_parent(dlg, parent)
    if dlg.exec() == QFileDialog.DialogCode.Accepted:
        files = dlg.selectedFiles()
        return files[0] if files else ''
    return ''

# -----------------------------
# 全局配置（Dashboard -> 各层级页共享）
# -----------------------------
@dataclass
class ProjectConfig:
    """
    存储整个应用程序会话的全局配置。
    包含项目信息、当前层级、模型模式、预训练和微调模型的路径等。
    """
    # 每层级独立的模型模式（未设置视为“未选择”）
    model_mode_map: Dict[str, str] = field(default_factory=dict)  # 每层级独立的模型模式（未设置视为“未选择”）
    name: str = 'Demo_Project'
    level: str = '微结构'  # 当前选择层级
    units: str = 'SI'  # SI / Imperial
    model_mode: str = '预训练'  # 预训练 / 微调
    # 预训练模型（支持微结构双模型，其它层级单模型 default）
    pretrained_models: Dict[str, Dict[str, str]] = field(default_factory=dict)  # 预训练模型（支持微结构双模型，其它层级单模型 default）
    # 微调数据
    finetune_image_paths: List[str] = field(default_factory=list)
    finetune_param_file: str = ''
    # 各层级微调模型（微结构含 CNN/ICNN）
    finetuned_models: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def get_mode(self, level: str) -> str:
        try:
            return self.model_mode_map.get(level, '未选择')
        except Exception:
            return '未选择'

    def set_mode(self, level: str, mode: str):
        self.model_mode_map[level] = mode or '未选择'

    # 便捷方法
    def get_pt(self, level: str, sub: str='default') -> str:
        """
        获取指定层级的预训练模型路径。对于微结构层级，会将用户选择的模型文件名映射到``MODEL_DIR`` 目录下的相应子目录中：
        * CNN 模型放在 ``Model/微结构/CNN``
        * ICNN 模型放在 ``Model/微结构/ICNN``
        其它层级的模型默认放在 ``Model/<层级名>`` 目录下。 如果配置中已经存储了绝对路径，则直接返回该路径。
        """
        name = self.pretrained_models.get(level, {}).get(sub, '') or ''
        # 如果已经是绝对路径，直接返回
        if name and os.path.isabs(name):
            return name
        # 构造相对路径。仅当有名称时才拼接
        if not name:
            return ''
        try:
            # 微结构层级有 CNN/ICNN 二级子目录
            if level == '微结构':
                subdir = sub if sub else 'default'
                return os.path.join(MODEL_DIR, level, subdir, name)
            else:
                # 其他层级直接在其文件夹下查找模型名
                return os.path.join(MODEL_DIR, level, name)
        except Exception:
            # 发生错误时返回名称本身
            return name

    def set_pt(self, level: str, name: str, sub: str='default'):
        self.pretrained_models.setdefault(level, {})[sub] = name or ''

    def get_ft(self, level: str, sub: str='default') -> str:
        return self.finetuned_models.get(level, {}).get(sub, '')

    def set_ft(self, level: str, path: str, sub: str='default'):
        if path:
            self.finetuned_models.setdefault(level, {})[sub] = path

# =============================
# 训练监控窗口（模型微调）
# =============================
# Keras 回调到 Qt 信号的桥接器：在 epoch 结束时发出损失指标。
class _KerasEpochBridge(keras.callbacks.Callback):
    """Keras回调，用于在每个epoch结束时捕获损失，并通过Qt信号发送出去。"""
    def __init__(self, signal):
        super().__init__()
        self._signal = signal

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self._signal.emit(int(epoch) + 1, float(logs.get('loss', 0.0)), float(logs.get('val_loss', logs.get('loss', 0.0))))


# 基于 QThread 的训练后台线程：支持早停、保存最佳模型、发出 epoch/完成/错误信号。
class TrainWorker(QThread):
    """
    Keras模型训练后台线程（适用于标准Numpy数组输入）。

    通过手动循环 epoch 和 train_on_batch 来实现精细控制，
    支持在 batch 之间检查暂停/停止信号。
    """
    epoch_signal = pyqtSignal(int, float, float)  # epoch, loss, val_loss
    done_signal = pyqtSignal(str, float)  # msg, best_val
    error_signal = pyqtSignal(str)  # traceback text

    def __init__(self, model, X_tr, y_tr, X_te, y_te, epochs, batch, monitor, patience, min_delta, lr,
                 parent=None):
        super().__init__(parent)
        self.model = model
        self.X_tr, self.y_tr = (X_tr, y_tr)
        self.X_te, self.y_te = (X_te, y_te)
        self.epochs = int(epochs)
        self.batch = int(batch)
        self.monitor = str(monitor or 'val_loss')
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.lr = float(lr)
        self._best_val = float('inf')
        self._best_weights = None  # 用于在内存中存储最佳权重
        self._stop = False
        self._pause = False

    def request_pause(self):
        self._pause = True

    def request_resume(self):
        self._pause = False

    def request_stop(self):
        if not self._stop:
            self._stop = True

    def run(self):
        try:
            # 编译一次
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), loss='mse', metrics=['mae'])
            n = int(self.X_tr.shape[0])
            bsz = max(1, int(self.batch))
            steps = (n + bsz - 1) // bsz
            patience_count = 0
            rng = np.random.RandomState(42)
            for ep in range(int(self.epochs)):
                if self._stop:
                    break
                # 每个 epoch 开始时打乱
                idx = np.arange(n)
                rng.shuffle(idx)
                # 运行中的损失累加器
                epoch_losses = []
                for s in range(steps):
                    if self._stop:
                        break
                    # 批次间的暂停门
                    while self._pause and (not self._stop):
                        time.sleep(0.1)
                    lo = s * bsz
                    hi = min(lo + bsz, n)
                    bidx = idx[lo:hi]
                    xb = self.X_tr[bidx]
                    yb = self.y_tr[bidx]
                    # train_on_batch 返回 loss (如果包含 metrics 则是列表；第一项是 loss)
                    out = self.model.train_on_batch(xb, yb, reset_metrics=False)
                    loss_val = float(out[0] if isinstance(out, (list, tuple)) else out)
                    epoch_losses.append(loss_val)
                # epoch 结束后或提前中断后评估验证损失（防止立即停止时 epoch 为空）
                if len(epoch_losses) == 0:
                    tr_mean = 0.0
                else:
                    tr_mean = float(np.mean(epoch_losses))
                ev = self.model.evaluate(self.X_te, self.y_te, batch_size=bsz, verbose=0)
                val_loss = float(ev[0] if isinstance(ev, (list, tuple)) else ev)

                # --- 保存检查点 -> 保存内存权重 ---
                improved = val_loss < self._best_val - self.min_delta
                if improved:
                    self._best_val = val_loss
                    self._best_weights = self.model.get_weights()  # 保存最佳权重到内存
                    patience_count = 0
                else:
                    patience_count += 1

                # 发出进度信号
                self.epoch_signal.emit(ep + 1, tr_mean, val_loss)
                if patience_count >= self.patience:
                    break

            # --- 训练结束，恢复最佳权重 ---
            if self._best_weights is not None:
                self.model.set_weights(self._best_weights)

            # --- 确定完成原因 ---
            msg = "OK"
            if self._stop:
                msg = "STOPPED"
            elif patience_count >= self.patience:
                msg = "EARLY_STOP"

            self.done_signal.emit(msg, float(self._best_val))
        except Exception as e:
            self.error_signal.emit(traceback.format_exc())

# 基于 QThread 的生成器训练后台线程：对接自定义生成器，支持早停与模型保存。
class GenTrainWorker(QThread):
    """
    Keras模型训练后台线程（适用于Keras DataGenerator输入）。

    通过手动循环 epoch 和 train_on_batch 来实现精细控制，
    支持在 batch 之间检查暂停/停止信号。
    """
    epoch_signal = pyqtSignal(int, float, float)
    done_signal = pyqtSignal(str, float)
    error_signal = pyqtSignal(str)

    def __init__(self, model, train_gen, val_gen, epochs, monitor, patience, min_delta, lr, parent=None):
        super().__init__(parent)
        self.model = model
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.epochs = int(epochs)
        self.monitor = str(monitor or 'val_loss')
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.lr = float(lr)
        self._best_val = float('inf')
        self._best_weights = None  # 用于在内存中存储最佳权重
        self._stop = False
        self._pause = False

    def request_pause(self):
        self._pause = True

    def request_resume(self):
        self._pause = False

    def request_stop(self):
        if not self._stop:
            self._stop = True

    def run(self):
        try:
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), loss='mse', metrics=['mae'])
            patience_count = 0
            for ep in range(self.epochs):
                if self._stop:
                    break
                epoch_losses = []
                # 训练循环
                steps = len(self.train_gen)
                for s in range(steps):
                    if self._stop:
                        break
                    while self._pause and (not self._stop):
                        time.sleep(0.1)
                    xb, yb = self.train_gen[s]
                    out = self.model.train_on_batch(xb, yb, reset_metrics=False)
                    loss_val = float(out[0] if isinstance(out, (list, tuple)) else out)
                    epoch_losses.append(loss_val)
                tr_mean = float(np.mean(epoch_losses)) if epoch_losses else 0.0
                # 评估
                ev = self.model.evaluate(self.val_gen, verbose=0)
                val_loss = float(ev[0] if isinstance(ev, (list, tuple)) else ev)

                # --- 保存检查点 -> 保存内存权重 ---
                improved = val_loss < self._best_val - self.min_delta
                if improved:
                    self._best_val = val_loss
                    self._best_weights = self.model.get_weights()  # 保存最佳权重到内存
                    patience_count = 0
                else:
                    patience_count += 1

                self.epoch_signal.emit(ep + 1, tr_mean, val_loss)
                if patience_count >= self.patience:
                    break

            # --- 训练结束，恢复最佳权重 ---
            if self._best_weights is not None:
                self.model.set_weights(self._best_weights)

            # --- 确定完成原因 ---
            msg = "OK"
            if self._stop:
                msg = "STOPPED"
            elif patience_count >= self.patience:
                msg = "EARLY_STOP"

            self.done_signal.emit(msg, float(self._best_val))
        except Exception:
            self.error_signal.emit(traceback.format_exc())

class SklearnTrainWorker(QThread):
    """
    用于Scikit-learn模型（如MLPRegressor）的后台训练线程。
    使用 'adam' 求解器和 'partial_fit' 来支持实时进度报告和epoch间控制（暂停/停止）。
    """
    # 信号
    epoch_signal = pyqtSignal(int, float, float)  # epoch, loss, val_loss
    done_signal = pyqtSignal(str, object, object, float)  # (status, model, scaler, best_val_loss)
    error_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str)

    def __init__(self, X_all, y_all, epochs, lr, patience, min_delta, parent=None):
        super().__init__(parent)
        self.X_all = X_all
        self.y_all = y_all
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self._stop_requested = False
        self._pause_requested = False

    def request_stop(self):
        # 添加哨兵检查
        if not self._stop_requested:
            self.log_signal.emit('[INFO] 收到停止请求...')
            self._stop_requested = True

    def request_pause(self):
        self._pause_requested = True

    def request_resume(self):
        self._pause_requested = False

    def run(self):
        # --- 在 try 块外部定义 final_model 和 scaler ---
        # --- 确保它们在任何情况下都能被 finally 块访问 ---
        final_model = None
        scaler = None
        best_model_state = None
        model = None  # 确保 model 变量存在
        best_val_loss = float('inf')  # 在 try 块外部初始化

        try:
            self.log_signal.emit('[INFO] 正在准备数据 (80/20 划分)...')
            X_tr, X_te, y_tr, y_te = train_test_split(self.X_all, self.y_all, test_size=0.2, random_state=42,
                                                      shuffle=True)

            if self._stop_requested:
                raise InterruptedError("Stop requested during data split")

            self.log_signal.emit('[INFO] 正在拟合 StandardScaler...')
            scaler = StandardScaler()  # 赋值给外部 scaler
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_te_scaled = scaler.transform(X_te)

            if self._stop_requested:
                raise InterruptedError("Stop requested during scaler fit")

            self.log_signal.emit(f'[INFO] 正在创建 MLPRegressor (solver=adam, max_iter=1, lr={self.lr})...')
            model = MLPRegressor(
                solver='adam',
                alpha=1e-1,
                max_iter=1,
                warm_start=True,
                learning_rate_init=self.lr,
                hidden_layer_sizes=(60, 60, 60),
                random_state=10,
                verbose=False
            )
            final_model = model  # 确保 final_model 至少是初始化的模型

            self.log_signal.emit('[INFO] 训练开始... (使用 adam 求解器)')

            patience_count = 0

            for ep in range(self.epochs):
                if self._stop_requested:
                    self.log_signal.emit('[INFO] 训练被用户停止。')
                    break

                # 在 epoch 开始前检查暂停标志
                while self._pause_requested and not self._stop_requested:
                    # 如果请求了暂停，则在此处等待
                    if ep % 50 == 0:  # 避免日志刷屏
                        self.log_signal.emit('[INFO] 训练已暂停... 等待恢复...')
                    time.sleep(0.1)  # 避免CPU空转

                # 如果在暂停时点击了停止，则退出
                if self._stop_requested:
                    self.log_signal.emit('[INFO] 训练在暂停期间被终止。')
                    break

                # 运行一轮 (epoch) - 这是阻塞点
                model.fit(X_tr_scaled, y_tr)

                # 如果刚跑完一个epoch就收到了停止信号，则立即停止
                if self._stop_requested:
                    self.log_signal.emit('[INFO] 训练在 Epoch 完成后立即终止。')
                    break

                # 计算损失
                tr_loss = mean_squared_error(y_tr, model.predict(X_tr_scaled))
                val_loss = mean_squared_error(y_te, model.predict(X_te_scaled))

                # 发送进度信号
                self.epoch_signal.emit(ep + 1, tr_loss, val_loss)

                # 早停和模型保存
                improved = val_loss < best_val_loss - self.min_delta
                if improved:
                    best_val_loss = val_loss
                    patience_count = 0
                    try:
                        best_model_state = (model.coefs_, model.intercepts_)
                    except Exception:
                        best_model_state = None
                else:
                    patience_count += 1

                if patience_count >= self.patience:
                    self.log_signal.emit(f'[INFO] 早停触发于 epoch {ep + 1} (val_loss 未改善)。')
                    break

            # --- 循环结束 ---
            self.log_signal.emit('[INFO] 训练循环结束。')

            # --- 无论如何结束，都尝试恢复最佳模型 ---
            final_model = model  # 默认使用最后状态的模型
            if best_model_state:
                try:
                    model.coefs_ = best_model_state[0]
                    model.intercepts_ = best_model_state[1]
                    final_model = model  # 确认 final_model 指向最佳状态
                except Exception as e:
                    self.log_signal.emit(f'[WARN] 恢复最佳模型状态失败: {e}，将使用最后状态的模型。')
                    # final_model 保持为 model 的最后状态

        # --- 异常处理块 ---
        except InterruptedError as ie:
            # 这是我们自己抛出的停止异常
            self.log_signal.emit(f'[INFO] 训练停止: {ie}')
            # 确保 final_model 指向当前模型状态
            if final_model is None and model is not None:
                final_model = model
        except Exception as e:
            # 捕获 try 块中的任何异常
            if not self._stop_requested:
                self.error_signal.emit(traceback.format_exc())
            else:
                self.log_signal.emit(f'[WARN] 用户停止过程中发生异常（可能无害）: {e}')

            # 关键：如果异常发生在循环后，但 model 和 scaler 已定义，
            # 确保 final_model 被赋值
            if final_model is None and model is not None:
                final_model = model

        # --- 最终信号发送 (移出 try...except) ---
        # 无论训练如何结束（正常、早停、用户停止、异常停止），
        # 只要 scaler 和 final_model (至少被初始化过) 存在，就发送它们。
        if scaler is not None:
            if final_model is not None:
                status = "STOPPED" if self._stop_requested else "COMPLETED"
                self.done_signal.emit(status, final_model, scaler, best_val_loss)
            else:
                # Model 都没初始化成功
                if not self._stop_requested:
                    self.error_signal.emit("Model initialization failed.")
        else:
            # Scaler 都没初始化成功
            self.log_signal.emit('[ERROR] Scaler 未初始化，无法发送最终模型。')
            if not self._stop_requested:
                self.error_signal.emit("Scaler initialization failed.")

# =============================
# 预测线程（避免阻塞 UI）
# =============================
class EnvelopePredictWorker(QThread):
    """
    后台线程，用于预测单个微结构图像的失效包络。
    1. 预处理图像。
    2. 加载CNN模型预测Tsai-Wu参数。
    3. 构造x网格。
    4. 加载ICNN模型预测修正项。
    5. 组合计算最终的失效包络(x, y)曲线。
    """
    done = pyqtSignal(object, object)  # x_curve, y_curve (np.ndarray or list)
    error = pyqtSignal(str)
    log_signal = pyqtSignal(str)

    def __init__(self, image_path: str, cnn_model_path: str, icnn_model_path: str, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.cnn_model_path = cnn_model_path or ''
        self.icnn_model_path = icnn_model_path or ''
        # 停止标志，用于优雅地终止线程
        self._stop = False

    def request_stop(self):
        """请求线程优雅地停止。"""
        # 检查是否已请求停止，防止重复记录
        if not self._stop:
            self.log_signal.emit('[INFO] 正在停止失效包络预测...')
            self._stop = True

    def run(self):
        try:
            if not os.path.exists(self.image_path):
                self.log_signal.emit(f'[ERROR] 图像不存在：{self.image_path}')
                self.error.emit('图像不存在')
                return

            if self._stop:
                # 移除日志
                return

            # 读入&预处理单张微结构图像
            image = preprocess_ms_image(self.image_path)
            if image is None:
                self.log_signal.emit(f'[ERROR] 读取/预处理失败：{self.image_path}')
                self.error.emit('图像预处理失败')
                return

            if self._stop:
                # 移除日志
                return

            # 加载 CNN（参数预测）
            if not self.cnn_model_path or not os.path.exists(self.cnn_model_path):
                self.log_signal.emit(f'[ERROR] CNN 模型文件不存在：{self.cnn_model_path}')
                self.error.emit('CNN 模型缺失')
                return

            if self._stop:
                # 移除日志
                return

            self.log_signal.emit(f'[INFO] 正在加载 CNN 模型: {self.cnn_model_path}')
            try:
                cnn_model = _ModelCache.get(self.cnn_model_path)
            except Exception as e:
                if self._stop:
                    # 移除日志
                    return
                self.log_signal.emit(f'[ERROR] 加载 CNN 模型失败：{e}')
                self.error.emit('CNN 模型加载失败')
                return

            if self._stop:
                # 移除日志
                return

            preds = cnn_model.predict(image[None, ...], verbose=0)

            if self._stop:
                # 移除日志
                return

            # 保护维度
            try:
                p0, p1, p2 = (float(preds[0, 0]), float(preds[0, 1]), float(preds[0, 2]))
            except Exception:
                self.log_signal.emit('[ERROR] CNN 输出维度异常，无法构造 x 网格。')
                self.error.emit('CNN 输出异常')
                return
            # 构造 x 网格（100 点）
            x = np.zeros(100, dtype=np.float32)
            x[0] = p0
            x[-1] = p2
            inc = (x[0] - x[-1]) / max(1, len(x) - 1)
            for i in range(len(x)):
                x[i] = x[0] - i * inc

            if self._stop:
                # 移除日志
                return

            # 加载 ICNN 修正模型
            if not self.icnn_model_path or not os.path.exists(self.icnn_model_path):
                self.log_signal.emit(f'[ERROR] ICNN 模型文件不存在：{self.icnn_model_path}')
                self.error.emit('ICNN 模型缺失')
                return

            if self._stop:
                # 移除日志
                return

            self.log_signal.emit(f'[INFO] 正在加载 ICNN 模型: {self.icnn_model_path}')
            try:
                icnn_model = _ModelCache.get(self.icnn_model_path, custom_objects={'rmse_metric': rmse_metric})
            except Exception as e:
                if self._stop:
                    # 移除日志
                    return
                self.log_signal.emit(f'[ERROR] 加载 ICNN 模型失败：{e}')
                self.error.emit('ICNN 模型加载失败')
                return

            if self._stop:
                # 移除日志
                return

            gen = CustomDataGenerator(image, x, batch_size=32, shuffle=False, is_training=False,
                                      input_names=getattr(icnn_model, 'input_names', None))
            y_corr = icnn_model.predict(gen, verbose=0)

            if self._stop:
                # 移除日志
                return

            # Tsai-Wu 构造
            p0 = p0 if p0 != 0 else 1e-09
            p1 = p1 if p1 != 0 else 1e-09
            p2 = p2 if p2 != 0 else 1e-09
            term1 = (1.0 / p0 + 1.0 / p2) * x
            term2 = -1.0 / (p0 * p2) * x ** 2
            term3 = 1.0 / p1 ** 2
            inside = (1.0 - term1 - term2) / term3
            inside = np.where(inside < 0.0, 0.0, inside)
            y_mod = np.sqrt(inside).astype(np.float32)
            y_corr = y_corr.ravel('F').astype(np.float32)
            m = min(len(y_mod), len(y_corr))  # 对齐 y_mod 和 y_corr 的长度
            y = y_mod[:m] - y_corr[:m]
            if len(y) >= 1:
                y[0] = 0.0
                y[-1] = 0.0
            y = np.where(y < 0.0, 0.0, y)

            if self._stop:
                # 移除日志
                return

            self.done.emit(x[:m], y)
        except Exception as e:
            if self._stop:
                # 移除日志
                return  # 忽略停止时引发的异常
            tb = traceback.format_exc()
            self.log_signal.emit(f'[ERROR] 预测线程异常：{e}')
            self.error.emit(tb)

class BandPredictWorker(QThread):
    """
    后台线程，用于批量预测多张微结构图像的失效包络，并计算“强度带”。
    1. 一次性加载CNN和ICNN模型。
    2. 循环处理每张图像，调用 _predict_one。
    3. 发出 partial 信号更新实时曲线。
    4. 聚合所有曲线，计算公共x轴上的y_min和y_max包络。
    5. 发出 done 信号返回最终结果。
    """
    progress = pyqtSignal(int, int)  # i, total
    partial = pyqtSignal(object, object)  # x_curve, y_curve
    done = pyqtSignal(dict)  # {'x_common','y_min','y_max','xs_list','ys_list'}
    error = pyqtSignal(str)
    log_signal = pyqtSignal(str)

    def __init__(self, image_paths: list, cnn_model_path: str, icnn_model_path: str, parent=None):
        super().__init__(parent)
        self.image_paths = list(image_paths or [])
        self.cnn_model_path = cnn_model_path or ''
        self.icnn_model_path = icnn_model_path or ''
        self._stop = False

    def request_stop(self):
        # 添加哨兵检查和日志
        if not self._stop:
            self.log_signal.emit('[INFO] 正在停止强度带预测...')
            self._stop = True

    def _predict_one(self, icnn_model, cnn_model, img_path: str):
        image = preprocess_ms_image(img_path)
        if image is None:
            self.log_signal.emit(f'[WARN] 预处理失败，跳过：{img_path}')
            return (None, None)
        preds = cnn_model.predict(image[None, ...], verbose=0)
        try:
            p0, p1, p2 = (float(preds[0, 0]), float(preds[0, 1]), float(preds[0, 2]))
        except Exception:
            self.log_signal.emit(f'[WARN] CNN 输出异常，跳过：{img_path}')
            return (None, None)
        x = np.zeros(100, dtype=np.float32)
        x[0] = p0
        x[-1] = p2
        inc = (x[0] - x[-1]) / max(1, len(x) - 1)
        for i in range(len(x)):
            x[i] = x[0] - i * inc
        gen = CustomDataGenerator(image, x, batch_size=32, shuffle=False, is_training=False, input_names=getattr(icnn_model, 'input_names', None))
        y_corr = icnn_model.predict(gen, verbose=0)
        # Tsai-Wu
        p0 = p0 if p0 != 0 else 1e-09
        p1 = p1 if p1 != 0 else 1e-09
        p2 = p2 if p2 != 0 else 1e-09
        term1 = (1.0 / p0 + 1.0 / p2) * x
        term2 = -1.0 / (p0 * p2) * x ** 2
        term3 = 1.0 / p1 ** 2
        inside = (1.0 - term1 - term2) / term3
        inside = np.where(inside < 0.0, 0.0, inside)
        y_mod = np.sqrt(inside).astype(np.float32)
        y_corr = y_corr.ravel('F').astype(np.float32)
        m = min(len(y_mod), len(y_corr))
        y = y_mod[:m] - y_corr[:m]
        if len(y) >= 1:
            y[0] = 0.0
            y[-1] = 0.0
        y = np.where(y < 0.0, 0.0, y)
        return (x[:m], y)

    def run(self):
        try:
            total = len(self.image_paths)
            if total == 0:
                self.error.emit('没有可用的图像进行强度带预测')
                return
            # 一次性加载模型
            if not self.cnn_model_path or not os.path.exists(self.cnn_model_path):
                self.log_signal.emit(f'[ERROR] CNN 模型文件不存在：{self.cnn_model_path}')
                self.error.emit('CNN 模型缺失')
                return
            if not self.icnn_model_path or not os.path.exists(self.icnn_model_path):
                self.log_signal.emit(f'[ERROR] ICNN 模型文件不存在：{self.icnn_model_path}')
                self.error.emit('ICNN 模型缺失')
                return
            self.log_signal.emit(f'[INFO] 正在加载 CNN 模型: {self.cnn_model_path}')
            cnn_model = _ModelCache.get(self.cnn_model_path)
            self.log_signal.emit(f'[INFO] 正在加载 ICNN 模型: {self.icnn_model_path}')
            icnn_model = _ModelCache.get(self.icnn_model_path, custom_objects={'rmse_metric': rmse_metric})
            xs, ys = ([], [])
            for i, p in enumerate(self.image_paths, 1):
                if self._stop:
                    # 移除日志
                    break
                self.log_signal.emit(f'[INFO] 处理 {i}/{total}：{os.path.basename(p)}')
                x_curve, y_curve = self._predict_one(icnn_model, cnn_model, p)
                if x_curve is not None and y_curve is not None:
                    xs.append(x_curve.astype(np.float32))
                    ys.append(y_curve.astype(np.float32))
                    self.partial.emit(x_curve, y_curve)
                else:
                    self.log_signal.emit(f'[WARN] 跳过图像 {p}')
                self.progress.emit(i, total)

            if self._stop:  # 检查循环是否因为停止而退出
                return

            if len(xs) == 0:
                self.error.emit('未得到任何有效曲线')
                return
            # 聚合：从全局范围构造公共 x 网格并计算包络
            x_left = float(np.min([np.min(xi) for xi in xs]))
            x_right = float(np.max([np.max(xi) for xi in xs]))
            x_common = np.linspace(x_left, x_right, 200).astype(np.float32)
            Y = []
            for xi, yi in zip(xs, ys):
                if xi[0] > xi[-1]:
                    xi, yi = (xi[::-1], yi[::-1])
                Y.append(np.interp(x_common, xi, yi))
            Y = np.stack(Y, axis=0)
            y_min = np.min(Y, axis=0)
            y_max = np.max(Y, axis=0)
            y_min[0] = y_max[0] = 0.0
            y_min[-1] = y_max[-1] = 0.0
            self.done.emit({'x_common': x_common, 'y_min': y_min, 'y_max': y_max, 'xs_list': xs, 'ys_list': ys})
        except Exception as e:
            if self._stop:
                # 移除日志
                return
            tb = traceback.format_exc()
            self.log_signal.emit(f'[ERROR] 强度带预测线程异常：{e}')
            self.error.emit(tb)

# =============================
# 通用预测线程（用于非微结构层级）
# =============================
class GenericLevelPredictWorker(QThread):
    """
    通用的后台预测线程，用于执行一个耗时的计算函数（callable）。
    用于机身段等目前使用模拟计算的层级。
    """
    done = pyqtSignal(object)  # result payload (e.g., summary text or dict)
    error = pyqtSignal(str)
    log_signal = pyqtSignal(str)

    def __init__(self, compute_callable, parent=None):
        super().__init__(parent)
        self._compute = compute_callable

    def run(self):
        try:
            if callable(self._compute):
                res = self._compute()
            else:
                res = None
            self.done.emit(res)
        except Exception as e:
            tb = traceback.format_exc()
            self.log_signal.emit(f'[ERROR] 预测线程异常：{e}')
            self.error.emit(tb)


# =============================
# 层合板预测线程
# =============================
class LaminatePredictWorker(QThread):
    """
    层合板预测线程:
    1. 加载模型
    2. 加载 'Laminate_train_data.csv' (用于拟合Scaler)
    3. 加载 'kongjian.csv' (用于设计空间)
    4. 使用输入参数填充设计空间
    5. 缩放、预测
    6. 整理结果 DataFrame
    7. 准备3D绘图数据 (ANGLE1, ANGLE2, Z)
    8. 返回 (ANGLE1, ANGLE2, Z, 预测结果DataFrame)
    """
    # 返回绘图数据
    done = pyqtSignal(object, object, object, object)  # (ANGLE1, ANGLE2, Z, pd.DataFrame)
    error = pyqtSignal(str)
    log_signal = pyqtSignal(str)

    def __init__(self, model_path: str, input_params: list, parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.input_params = input_params
        self._stop_requested = False

    def request_stop(self):
        """请求线程优雅地停止。"""
        # 检查是否已请求停止，防止重复记录
        if not self._stop_requested:
            self.log_signal.emit('[INFO] 正在停止层合板预测...')
            self._stop_requested = True

    def run(self):
        try:
            self.log_signal.emit('[INFO] 层合板预测线程启动...')
            # --- 1. 验证模型路径 ---
            if not self.model_path or not os.path.exists(self.model_path):
                self.log_signal.emit(f'[ERROR] 找不到模型文件：{self.model_path}')
                self.error.emit('模型文件缺失')
                return
            if self._stop_requested: return

            # --- 2. 准备依赖文件路径 ---
            model_dir = os.path.dirname(self.model_path)
            train_csv_path = os.path.join(model_dir, 'Laminate_train_data.csv')
            space_csv_path = os.path.join(model_dir, 'kongjian.csv')
            if not os.path.exists(train_csv_path):
                self.log_signal.emit(f'[ERROR] 依赖文件缺失: {train_csv_path}')
                self.error.emit(f'依赖文件 {os.path.basename(train_csv_path)} 缺失')
                return
            if not os.path.exists(space_csv_path):
                self.log_signal.emit(f'[ERROR] 依赖文件缺失: {space_csv_path}')
                self.error.emit(f'依赖文件 {os.path.basename(space_csv_path)} 缺失')
                return

            if self._stop_requested: return

            # --- 3. 获取输入参数 (7个用于填充) ---
            try:
                p = self.input_params
                test_features = [p[0], p[1], p[2], p[5], p[6], p[7], p[8]]
            except IndexError:
                self.log_signal.emit(f'[ERROR] 输入参数列表不完整，需要9个参数')
                self.error.emit('输入参数不足9个')
                return

            if self._stop_requested: return

            # --- 4. 拟合Scaler (使用 Laminate_train_data.csv) ---
            self.log_signal.emit(f'[INFO] 加载 {train_csv_path} 用于拟合缩放器...')
            train_data = pd.read_csv(train_csv_path)
            X_train = train_data.iloc[:, 1:10]
            scaler = StandardScaler().fit(X_train)
            self.log_signal.emit('[INFO] 缩放器拟合完毕。')

            if self._stop_requested: return

            # --- 5. 准备设计空间数据 ---
            self.log_signal.emit(f'[INFO] 加载 {space_csv_path} (设计空间)...')
            kongjian_df = pd.read_csv(space_csv_path)
            fill_columns = ['L', 'W', 'D', 'mt', 'sl', 'gg12', 'ee2']
            kongjian_df[fill_columns] = test_features
            X_kongjian = kongjian_df[['L', 'W', 'D', 'angle1', 'angle2', 'mt', 'sl', 'gg12', 'ee2']]
            X_kongjian_scaled = scaler.transform(X_kongjian)
            self.log_signal.emit('[INFO] 设计空间数据准备完毕。')

            if self._stop_requested: return

            # --- 6. 加载模型并预测 ---
            self.log_signal.emit(f'[INFO] 正在加载模型: {self.model_path}')
            model = _ModelCache.get(self.model_path)

            if self._stop_requested: return

            self.log_signal.emit('[INFO] 正在预测...')
            predictions = model.predict(X_kongjian_scaled).flatten()

            # --- 在记录“完成”之前立即检查停止标志 ---
            if self._stop_requested:
                # 移除日志
                return

            self.log_signal.emit('[INFO] 预测完成。')
            if self._stop_requested: return

            # --- 7. 整理结果 ---
            result_df = kongjian_df[['angle1', 'angle2']].copy()
            result_df['prediction'] = predictions
            result_df.columns = ['铺层角度𝛷', '铺层角度𝛹', '初始失效因子𝑅']

            if self._stop_requested: return

            # --- 8. 准备3D绘图数据 (不在此处绘图) ---
            self.log_signal.emit('[INFO] 正在准备3D绘图数据...')
            angle1 = result_df['铺层角度𝛷'].unique()
            angle2 = result_df['铺层角度𝛹'].unique()
            ANGLE1, ANGLE2 = np.meshgrid(angle1, angle2)
            try:
                Z = result_df['初始失效因子𝑅'].values.reshape(len(angle1), len(angle2)).T
            except ValueError:
                self.log_signal.emit(
                    f'[WARN] 预测结果维度 ({len(result_df)}) 与设计空间维度 ({len(angle1)}x{len(angle2)}) 不匹配，尝试重塑。')
                Z = result_df['初始失效因子𝑅'].values.reshape(len(angle2), len(angle1))

            if self._stop_requested:
                # 移除日志
                return

            # --- 9. 发送信号 (发送原始数据) ---
            self.done.emit(ANGLE1, ANGLE2, Z, result_df)

        except Exception as e:
            if self._stop_requested: return  # 忽略停止时引发的异常
            tb = traceback.format_exc()
            self.log_signal.emit(f'[ERROR] 层合板预测线程异常：{e}\n{tb}')
            self.error.emit(tb)

# =============================
# 加筋结构 - PyTorch 图像生成模型定义
# =============================
if nn is not None:
    class ImageGenerator(nn.Module):
        """
        加筋结构：PyTorch图像生成模型。
        一个简单的全连接网络，将16个特征映射为 (3, 300, 300) 的图像。
        """
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(16, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 300 * 300 * 3),
                nn.Tanh()
            )

        def forward(self, x):
            x = self.fc(x)
            x = x.view(-1, 3, 300, 300)
            return x

# =============================
# 加筋结构 - 性能预测线程
# =============================
class StiffenedPanelPredictWorker(QThread):
    """
    加筋结构性能预测线程：
    1. 加载回归模型 (joblib) 和图像生成模型 (pth)。
    2. 接收11个输入参数。
    3. 运行回归模型，预测5个屈曲载荷。
    4. 组合16个特征 (11输入 + 5预测)。
    5. 运行图像模型，生成一阶屈曲模态图。
    6. 返回 (屈曲载荷列表, 模态图像(numpy array))
    """
    done = pyqtSignal(object, object)  # (list: 5个屈曲载荷, np.ndarray: 模态图像)
    error = pyqtSignal(str)
    log_signal = pyqtSignal(str)

    def __init__(self, reg_model_path: str, reg_scaler_path: str, img_model_path: str, input_params: list,
                 parent=None):
        super().__init__(parent)
        self.reg_model_path = reg_model_path
        self.reg_scaler_path = reg_scaler_path
        self.img_model_path = img_model_path
        self.input_params = input_params
        self._stop_requested = False

        # 图像生成模型所需的均值和标准差
        self.scaler_mean = np.array([6.85460818e+02, 2.74467364e+02, 4.09587273e+01, 3.06477273e+01,
                                     1.89156288e+05, 2.15939978e+05, 2.41090395e+05, 2.54475731e+05,
                                     2.68100341e+05, 1.75398657e+05, 1.34614462e+04, 2.99350814e-01,
                                     6.24666182e+03, 6.24666182e+03, 4.75065818e+03, 7.86649091e+01])
        self.scaler_std = np.array([4.16750577e+02, 1.29565259e+02, 2.64881058e+01, 2.13555399e+01,
                                    1.16494228e+05, 1.27383382e+05, 1.37657222e+05, 1.42527418e+05,
                                    1.48249371e+05, 4.31530754e+04, 3.74247615e+03, 5.74553192e-02,
                                    1.59604225e+03, 1.59604225e+03, 1.31310127e+03, 4.52498125e+01])

    def request_stop(self):
        """请求线程优雅地停止。"""
        # --- 防止重复记录日志 ---
        if not self._stop_requested:
            self.log_signal.emit('[INFO] 正在停止性能预测...')
            self._stop_requested = True

    def run(self):
        try:
            self.log_signal.emit('[INFO] 加筋结构性能预测线程启动...')
            # --- 检查 PyTorch 是否导入 ---
            if torch is None or nn is None:
                error_msg = "PyTorch 模块未加载。请检查 PyTorch 是否已正确安装。"
                self.log_signal.emit(f'[ERROR] {error_msg}')
                self.error.emit(error_msg)
                return  # 提前退出线程

            if self._stop_requested:
                # 移除日志
                return

            # --- 1. 验证模型路径 (新) ---
            missing_files = []
            if not os.path.exists(self.reg_model_path): missing_files.append(os.path.basename(self.reg_model_path))
            if not os.path.exists(self.reg_scaler_path): missing_files.append(os.path.basename(self.reg_scaler_path))
            if not os.path.exists(self.img_model_path): missing_files.append(os.path.basename(self.img_model_path))

            if missing_files:
                error_msg = f'缺少一个或多个模型文件: {", ".join(missing_files)}。'
                self.log_signal.emit(f'[ERROR] {error_msg}')
                self.error.emit(error_msg)
                return

            if self._stop_requested:
                # 移除日志
                return

            # --- 2. 加载回归模型 (Joblib) ---
            self.log_signal.emit('[INFO] 加载回归模型 (Joblib)...')
            try:
                # 捕获 joblib 加载错误，特别是库版本不兼容时
                reg_model = joblib.load(self.reg_model_path)
                if self._stop_requested:
                    # 移除日志
                    return
                reg_scaler = joblib.load(self.reg_scaler_path)
                # --- 获取Scaler的特征名 ---
                try:
                    scaler_feature_names = reg_scaler.get_feature_names_out()
                except AttributeError:
                    try:
                        scaler_feature_names = reg_scaler.feature_names_in_
                    except AttributeError:
                        self.log_signal.emit('[WARN] 无法从Scaler获取特征名, 将使用硬编码猜测 (11个特征)。')
                        # 必须与 reg_input 的顺序一致
                        scaler_feature_names = ['L', 'W', 'H1', 'W1', 'E1', 'E2', 'Nu12', 'G12', 'G13', 'G23', 'd']
                if len(scaler_feature_names) != 11:
                    self.log_signal.emit(f'[WARN] Scaler期望 {len(scaler_feature_names)} 个特征名, 但代码将提供11个。')
                    # 如果加载的特征名数量不匹配，则强制使用硬编码的11个
                    scaler_feature_names = ['L', 'W', 'H1', 'W1', 'E1', 'E2', 'Nu12', 'G12', 'G13', 'G23', 'd']

            except TypeError as te:
                # 捕获特定的 TypeError
                tb = traceback.format_exc()
                error_msg = (f"加载 Joblib 模型时发生 TypeError (很可能由库版本不兼容引起): {te}\n"
                             f"请确保创建模型的环境与当前运行环境的 numpy/scikit-learn 版本兼容。\n"
                             f"建议在当前环境中重新保存模型文件。\n"
                             f"详细信息:\n{tb}")
                self.log_signal.emit(f'[ERROR] {error_msg}')
                self.error.emit(error_msg)
                return  # 加载失败，退出
            except Exception as e:
                # 捕获其他可能的加载错误
                tb = traceback.format_exc()
                error_msg = f"加载 Joblib 模型时发生未知错误: {e}\n详细信息:\n{tb}"
                self.log_signal.emit(f'[ERROR] {error_msg}')
                self.error.emit(error_msg)
                return  # 加载失败，退出

            if self._stop_requested:
                # 移除日志
                return

            # --- 3. 运行回归预测 ---
            self.log_signal.emit('[INFO] 正在预测屈曲载荷...')
            if len(self.input_params) != 11:
                error_msg = f"输入参数数量错误，需要11个，实际收到 {len(self.input_params)} 个。"
                self.log_signal.emit(f'[ERROR] {error_msg}')
                self.error.emit(error_msg)
                return
            reg_input = np.array(self.input_params).reshape(1, -1)
            try:
                # --- 使用DataFrame进行transform ---
                input_df = pd.DataFrame(reg_input, columns=scaler_feature_names)
                input_scaled = reg_scaler.transform(input_df)

                if self._stop_requested:
                    # 移除日志
                    return
                # 预测结果是 (1, N) 数组，N >= 5
                # 注意：如果 reg_model (MLPRegressor) 也需要特征名，
                # 则应传递 input_scaled (它现在是一个DataFrame)
                # 如果它接受Numpy，则传递 input_scaled.to_numpy()
                # 我们假设模型也需要带特征名的输入
                predictions = reg_model.predict(input_scaled)
                if predictions.shape[1] < 5:
                    error_msg = f"回归模型预测输出维度不足5个 (实际为 {predictions.shape[1]})。"
                    self.log_signal.emit(f'[ERROR] {error_msg}')
                    self.error.emit(error_msg)
                    return
                buckling_loads = predictions[0, :5].tolist()  # 取前5个载荷
            except Exception as e:
                if self._stop_requested:
                    # 移除日志
                    return  # 如果是停止导致的异常，则忽略
                tb = traceback.format_exc()
                error_msg = f"运行回归预测时出错: {e}\n详细信息:\n{tb}"
                self.log_signal.emit(f'[ERROR] {error_msg}')
                self.error.emit(error_msg)
                return

            if self._stop_requested:
                # 移除日志
                return

            # --- 4. 准备图像生成模型的输入 (16个特征) ---
            self.log_signal.emit('[INFO] 准备图像生成器输入...')
            # 顺序:[L, W, H1, W1] + [5个预测载荷] + [E1, E2, Nu12, G12, G13, G23, d]
            reg_input_1d = reg_input.flatten()
            reg_results_1d = np.array(buckling_loads)
            combined_features = np.concatenate([
                reg_input_1d[:4],  # L, W, H1, W1
                reg_results_1d,  # 5个预测载荷
                reg_input_1d[4:]  # E1, E2, Nu12, G12, G13, G23, d
            ])
            if len(combined_features) != 16:
                error_msg = f"特征组合错误，需要16个特征，但得到 {len(combined_features)} 个"
                self.log_signal.emit(f'[ERROR] {error_msg}')
                self.error.emit(error_msg)
                return
            img_input = combined_features.reshape(1, -1)

            if self._stop_requested:
                # 移除日志
                return

            # --- 5. 加载图像模型 (PyTorch) ---
            self.log_signal.emit('[INFO] 加载图像生成模型 (PyTorch)...')
            # --- 自动检测设备 (GPU或CPU) ---
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.log_signal.emit(f'[INFO] PyTorch 将使用: {device}')
            except Exception as e:
                self.log_signal.emit(f'[WARN] PyTorch 设备检测失败: {e}，回退到 CPU。')
                device = torch.device('cpu')

            if self._stop_requested:
                # 移除日志
                return

            try:
                img_model = ImageGenerator().to(device)  # 将模型实例移至设备
                # 确保在检测到的设备上加载
                # 使用 self.img_model_path 访问在 __init__ 中保存的路径
                img_model.load_state_dict(torch.load(self.img_model_path, map_location=device))
                img_model.eval()  # 设置为评估模式
            except Exception as e:
                if self._stop_requested:
                    # 移除日志
                    return
                tb = traceback.format_exc()
                error_msg = f"加载 PyTorch 模型时出错: {e}\n详细信息:\n{tb}"
                self.log_signal.emit(f'[ERROR] {error_msg}')
                self.error.emit(error_msg)
                return

            if self._stop_requested:
                # 移除日志
                return

            # --- 6. 运行图像生成 ---
            self.log_signal.emit('[INFO] 正在生成屈曲模态图...')
            try:
                # 标准化 (使用类中定义的均值/标准差)
                # 添加检查以确保 scaler_mean 和 scaler_std 的形状正确
                if self.scaler_mean.shape[0] != 16 or self.scaler_std.shape[0] != 16:
                    error_msg = "内部定义的 scaler 均值/标准差形状不为 16。"
                    self.log_signal.emit(f'[ERROR] {error_msg}')
                    self.error.emit(error_msg)
                    return
                input_normalized = (img_input - self.scaler_mean) / self.scaler_std
                # 将输入张量移至设备
                input_tensor = torch.tensor(input_normalized, dtype=torch.float32).to(device)
                with torch.no_grad():
                    output = img_model(input_tensor)  # (B, C, H, W), B=1
                    # 调整维度顺序, C,H,W -> H,W,C
                    # 确保数据移回CPU以便numpy转换
                    image_raw = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    # 从 Tanh 的 [-1, 1] 范围转换到 [0, 1]
                    image_rgb = (image_raw + 1) / 2.0
                    image_rgb = np.clip(image_rgb, 0, 1)  # 确保范围
            except Exception as e:
                if self._stop_requested:
                    # 移除日志
                    return
                tb = traceback.format_exc()
                error_msg = f"生成图像时出错: {e}\n详细信息:\n{tb}"
                self.log_signal.emit(f'[ERROR] {error_msg}')
                self.error.emit(error_msg)
                return

            if self._stop_requested:
                # 移除日志
                return

            self.log_signal.emit('[INFO] 性能预测完成。')
            self.done.emit(buckling_loads, image_rgb)  # 发送 NumPy 数组

        except Exception as e:
            # 捕获 run 方法中的任何其他未预料到的异常
            if self._stop_requested:
                # 移除日志
                return
            tb = traceback.format_exc()
            self.log_signal.emit(f'[CRITICAL] 加筋结构预测线程发生严重错误：{e}\n{tb}')
            self.error.emit(f"预测线程发生严重错误: {e}")

# =============================
# 加筋结构 - 优化设计线程 (GA)
# =============================
class StiffenedPanelOptimizeWorker(QThread):
    """
    加筋结构优化设计线程
    """
    # 信号
    update_progress = pyqtSignal(int, float, list, object, object, object, object, object)
    finished = pyqtSignal(list, list, list, list, list)
    error = pyqtSignal(str)
    log_signal = pyqtSignal(str)

    def __init__(self, model_path: str, scaler_path: str, params: list, parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.params = params
        self.running = True

    def stop(self):
        # 检查是否已在运行，防止重复记录
        if self.running:
            self.log_signal.emit('[INFO] 正在停止优化设计...')
            self.running = False

    def run(self):
        try:
            self.log_signal.emit('[INFO] 加筋结构优化设计线程启动...')

            # --- 1. 加载模型 (使用传入的完整路径) ---
            model_path = self.model_path
            scaler_path = self.scaler_path

            if not all(os.path.exists(p) for p in [model_path, scaler_path]):
                self.log_signal.emit(f'[ERROR] 缺少一个或多个优化模型文件。')
                self.error.emit('优化模型文件缺失')
                return
            self.log_signal.emit('[INFO] 加载优化模型 (Joblib)...')
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            # --- 1b. 获取 Scaler 的特征名 ---
            try:
                # 尝试从加载的scaler获取特征名
                scaler_feature_names = scaler.get_feature_names_out()
            except AttributeError:
                try:
                    scaler_feature_names = scaler.feature_names_in_
                except AttributeError:
                    # 如果失败，硬编码特征名以匹配输入顺序
                    self.log_signal.emit('[WARN] 无法从Scaler获取特征名, 将使用硬编码猜测。')
                    scaler_feature_names = ['L', 'W', 'W1', 'H1', 'E1', 'E2', 'Nu12', 'G12', 'G13', 'G23', 'd']
            except Exception as e:
                self.log_signal.emit(f'[WARN] 获取Scaler特征名时出错: {e}, 将使用硬编码猜测。')
                scaler_feature_names = ['L', 'W', 'W1', 'H1', 'E1', 'E2', 'Nu12', 'G12', 'G13', 'G23', 'd']

            if len(scaler_feature_names) != 11:
                self.log_signal.emit(f'[ERROR] Scaler特征名数量({len(scaler_feature_names)})与预期(11)不符。')
                # 尽管如此，还是尝试使用硬编码的11个
                scaler_feature_names = ['L', 'W', 'W1', 'H1', 'E1', 'E2', 'Nu12', 'G12', 'G13', 'G23', 'd']

            R, L, W, E1, E2, Nu12, G12, G13, G23 = self.params

            # --- 2. 遗传算法参数 ---
            population_size = 100
            generations = 200
            mutation_rate = 0.3
            elite_percentage = 0.01

            # 初始化数据记录
            generation_list = []
            best_W1_list = []
            best_H1_list = []
            best_d_list = []
            best_perf_list = []

            # 初始化种群
            population = [self.generate_individual(W) for _ in range(population_size)]

            for generation in range(generations):
                # (检查点 1: 每代开始时检查)
                if not self.running:
                    # 移除日志
                    break
                # --- 添加短暂休眠以允许UI响应和绘图 ---
                # 减慢循环 (100ms/代)，使绘图更新和停止按钮可见
                # 总共约 30 * 0.1 = 3 秒
                time.sleep(0.1)
                # 计算适应度
                fitness_values = []
                # (W1, H1, d) 是我们要优化的三个变量
                for W1, H1, d in population:
                    # (检查点 2: 每个个体计算前检查 - 这确保了快速停止)
                    # --- 在适应度计算循环中添加停止检查 ---
                    if not self.running:
                        break
                    # 检查约束
                    if not (0.05 * W <= W1 <= 0.25 * W and 0.5 <= H1 / W1 <= 1.0 and W1 / (W - W1) <= d / (
                            W - W1) <= 0.5):
                        fitness = float('-inf')
                    else:
                        # 11个输入 (L, W, W1, H1, E1, E2, Nu12, G12, G13, G23, d)
                        input_data = np.array([L, W, W1, H1, E1, E2, Nu12, G12, G13, G23, d])

                        # --- 使用DataFrame来避免特征名警告 ---
                        input_df = pd.DataFrame(input_data.reshape(1, -1), columns=scaler_feature_names)
                        input_scaled = scaler.transform(input_df)
                        prediction_result = model.predict(input_scaled)

                        if prediction_result is None or prediction_result.size == 0:
                            self.log_signal.emit(f'[WARN] 模型预测对输入返回了空结果: {input_data}')
                            Eigenvalue1_scalar = float('-inf')  # 赋一个不会通过检查的值
                        else:
                            Eigenvalue1_scalar = prediction_result.flatten()[0]

                        if Eigenvalue1_scalar > R:
                            denominator = W1 + 2 * H1
                            if denominator > 1e-9:
                                fitness = Eigenvalue1_scalar / denominator
                            else:
                                fitness = float('-inf')
                        else:
                            fitness = float('-inf')  # 不满足约束
                    fitness_values.append(fitness)

                # (检查点 3: 检查内部循环是否因为停止而中断)
                if not self.running:
                    # 移除日志
                    break

                # 种群排序和选择
                sorted_population = [x for _, x in
                                     sorted(zip(fitness_values, population), key=lambda item: item[0], reverse=True)]
                best_individual = sorted_population[0]
                W1_best, H1_best, d_best = best_individual
                perf_best = max(fitness_values)

                if perf_best == float('-inf'):
                    self.log_signal.emit(f'[WARN] 第 {generation + 1} 代：未找到满足约束 (>{R}) 的解。')
                    pass

                # 记录数据
                generation_list.append(generation + 1)
                best_W1_list.append(W1_best)
                best_H1_list.append(H1_best)
                best_d_list.append(d_best)
                best_perf_list.append(perf_best)

                # 实时曲线更新点: 每代结束时发送信号
                self.update_progress.emit(generation + 1, perf_best, best_individual,
                                          generation_list, best_W1_list, best_H1_list, best_d_list, best_perf_list)

                # 选择精英
                elite_count = int(population_size * elite_percentage)
                elite_individuals = sorted_population[:elite_count]

                # 生成下一代
                parents = sorted_population[:population_size // 2]
                next_generation = self.crossover(parents, population_size)
                next_generation = self.mutate(next_generation, W, mutation_rate)

                # 加入精英
                next_generation = next_generation[:population_size - elite_count]
                next_generation.extend(elite_individuals)
                population = next_generation[:population_size]

            if not self.running:
                # 如果循环是因为停止而中断的，确保日志记录
                # 移除日志
                pass
            else:
                self.log_signal.emit('[INFO] 优化设计完成。')

            self.finished.emit(generation_list, best_W1_list, best_H1_list, best_d_list, best_perf_list)

        except Exception as e:
            if not self.running:
                # 移除日志
                return
            tb = traceback.format_exc()
            self.log_signal.emit(f'[ERROR] 优化线程异常：{e}\n{tb}')
            self.error.emit(tb)

    def generate_individual(self, W):
        # a1 = H1/W1, a2 = d/(W-W1)
        # W1, H1, d
        while True:
            W1_min = 0.05 * W
            W1_max = 0.25 * W
            W1 = random.uniform(W1_min, W1_max)

            a1_min = 0.5
            a1_max = 1.0
            a1 = random.uniform(a1_min, a1_max)
            H1 = a1 * W1

            # 确保 W - W1 > 0，避免除零
            if W - W1 <= 1e-9:
                 continue # 如果 W1 太接近 W，重新生成

            a2_min = W1 / (W - W1)
            a2_max = 0.5

            if a2_min > a2_max: # 约束冲突
                continue # 重新生成 W1

            a2 = random.uniform(a2_min, a2_max)
            d = a2 * (W - W1)

            # 添加额外的边界检查确保值在合理范围内 (可选但推荐)
            if not (W1_min <= W1 <= W1_max and a1_min <= (H1 / W1 if W1 > 1e-9 else 0) <= a1_max and a2_min <= a2 <= a2_max):
                 continue # 如果交叉/变异后超出范围，重新生成 (虽然这里是初始生成，但也加上)

            return [W1, H1, d]

    def crossover(self, parents, target_size):
        next_generation = parents[:] # 复制精英或父代
        while len(next_generation) < target_size:
            p1, p2 = random.sample(parents, 2)
            child1 = [0.0] * 3
            child2 = [0.0] * 3

            # BLX-a 交叉 (更适合连续变量)
            alpha = 0.5
            for i in range(3):
                d = abs(p1[i] - p2[i])
                min_v, max_v = min(p1[i], p2[i]), max(p1[i], p2[i])
                # 限制交叉范围，避免生成极端值
                lower_bound = min_v - alpha * d
                upper_bound = max_v + alpha * d
                child1[i] = random.uniform(lower_bound, upper_bound)
                child2[i] = random.uniform(lower_bound, upper_bound)
                # 可以在这里添加约束检查，确保交叉结果仍在合理范围内，但这通常在适应度计算时处理

            next_generation.extend([child1, child2])
        return next_generation[:target_size]

    def mutate(self, population, W, mutation_rate):
        mutated_population = []
        for ind in population: # ind 是 [W1, H1, d]
            new_ind = list(ind) # 复制个体以进行变异
            if random.random() < mutation_rate:
                mutate_point = random.randint(0, 2)
                # 采用更简单的变异：在当前值附近小范围扰动
                # 或者直接重新生成该维度，但要确保约束
                # 这里我们尝试重新生成以保证约束符合性
                temp_individual = self.generate_individual(W)
                new_ind[mutate_point] = temp_individual[mutate_point]
                # 重新验证整个个体的约束可能更安全
                W1, H1, d = new_ind
                if not (0.05 * W <= W1 <= 0.25 * W and 0.5 <= (H1 / W1 if W1 > 1e-9 else 0) <= 1.0 and (W1 / (W - W1) if W - W1 > 1e-9 else float('inf')) <= (d / (W - W1) if W - W1 > 1e-9 else float('inf')) <= 0.5):
                     # 如果变异导致不满足约束，可以选择放弃变异或重新尝试
                     # 这里我们放弃本次变异，保留原个体的一个副本
                     mutated_population.append(list(ind))
                     continue # 跳过添加 new_ind
            mutated_population.append(new_ind) # 添加（可能变异过的）个体
        return mutated_population

# 独立微调训练窗口：曲线、日志过滤、早停控制、暂停/恢复/停止。
class FineTuneWindow(QMainWindow):
    """
    一个独立的窗口，用于模型微调。
    显示训练/验证损失曲线、训练日志，并提供训练控制（启动、暂停、停止、保存）。
    支持 Keras (TrainWorker, GenTrainWorker) 和 Scikit-learn (SklearnTrainWorker)。
    """

    def __init__(self, config: ProjectConfig, level: str, write_log_to_main, on_cfg_changed,
                 fixed_model_type: Optional[str] = None):
        super().__init__()
        self.cfg = config
        self.level = level
        self.write_log_to_main = write_log_to_main
        self.on_cfg_changed = on_cfg_changed
        self.setWindowTitle(f'微调训练 - {self.level}')
        self.resize(1080, 720)
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        # 顶部简要信息：使用多个标签分别显示层级、图像目录、材料参数数据和输出数据，每个路径独立设置悬停提示
        top = QHBoxLayout()
        # 层级显示（不截断）
        lbl_level = QLabel(f'层级：{self.level}')
        lbl_level.setTextFormat(Qt.TextFormat.PlainText)
        top.addWidget(lbl_level)
        # 提取数据路径
        try:
            imgs_full_list = self.cfg.finetune_image_paths if self.cfg.finetune_image_paths else []
        except Exception:
            imgs_full_list = []
        imgs_full = ', '.join(imgs_full_list) if imgs_full_list else ''
        imgs_short = ', '.join([os.path.basename(p) for p in imgs_full_list]) if imgs_full_list else ''
        tbl_full = getattr(self.cfg, 'finetune_param_file', '') or ''
        tbl_short = os.path.basename(tbl_full) if tbl_full else ''
        out_full = getattr(self.cfg, 'finetune_output_file', '') or ''
        out_short = os.path.basename(out_full) if out_full else ''
        # 微结构需要显示图像目录
        if self.level == '微结构':
            lbl_img_desc = QLabel('\u3000图像目录：')
            lbl_img_desc.setTextFormat(Qt.TextFormat.PlainText)
            top.addWidget(lbl_img_desc)
            lbl_img_val = ElidedLabel(imgs_short if imgs_short else '未选择')
            # 允许水平扩展以防止在启动时由于宽度=0而闪烁
            lbl_img_val.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            '''
            保留最小宽度，这样标签就不会以零宽度开始。在初始布局时，当可用宽度为0或1像素时，ElidedLabel将反复重新计算省略的字符串并导致可见闪烁。
            通过给标签一个合理的最小宽度，我们避免了病态的0→1→2像素增长序列，减少了启动时的闪烁。最小约150px保持摘要紧凑，同时仍然适合典型的路径。
            '''
            lbl_img_val.setMinimumWidth(150)
            lbl_img_val.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            lbl_img_val.set_full_text(imgs_full if imgs_full else '未选择图像目录')
            top.addWidget(lbl_img_val)
        # 材料参数数据
        lbl_tbl_desc = QLabel('\u3000材料参数数据：')
        lbl_tbl_desc.setTextFormat(Qt.TextFormat.PlainText)
        top.addWidget(lbl_tbl_desc)
        lbl_tbl_val = ElidedLabel(tbl_short if tbl_short else '未选择')
        lbl_tbl_val.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        # 提供与图像目录相似的最小宽度，以避免重复的0→1像素调整，这可能导致小部件首次出现时闪烁。确保基线宽度也保持随后的省略稳定。
        lbl_tbl_val.setMinimumWidth(150)
        lbl_tbl_val.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lbl_tbl_val.set_full_text(tbl_full if tbl_full else '未选择材料参数数据')
        top.addWidget(lbl_tbl_val)
        # 输出数据
        lbl_out_desc = QLabel('\u3000输出数据：')
        lbl_out_desc.setTextFormat(Qt.TextFormat.PlainText)
        top.addWidget(lbl_out_desc)
        lbl_out_val = ElidedLabel(out_short if out_short else '未选择')
        lbl_out_val.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        # 与上面相同的原理：保留一个小的最小宽度来稳定初始布局，并减少宽度从零到最终大小时的闪烁。
        lbl_out_val.setMinimumWidth(150)
        lbl_out_val.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lbl_out_val.set_full_text(out_full if out_full else '未选择输出数据')
        top.addWidget(lbl_out_val)
        top.addStretch(1)
        root.addLayout(top)
        # 中间：图表 + 工具栏
        fig = Figure(figsize=(5, 3), tight_layout=True)
        self.ax = fig.add_subplot(111)
        self.ax.set_title('Training Curve')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')
        self.line_train, = self.ax.plot([], [], label='loss')
        self.line_val, = self.ax.plot([], [], label='val_loss')
        self.ax.legend(loc='best')
        self.canvas = FigureCanvas(fig)
        protect_canvas(self.canvas)
        self.toolbar = AutoNavToolbar(self.canvas, self)
        # 右侧：控制与参数
        right_box = QGroupBox('训练控制与参数')
        right = QFormLayout(right_box)
        self.spn_epochs = QSpinBox()
        self.spn_epochs.setRange(1, 2000)
        self.spn_epochs.setValue(50)
        self.spn_batch = QSpinBox()
        self.spn_batch.setRange(1, 4096)
        self.spn_batch.setValue(32)
        self.dsb_lr = QDoubleSpinBox()
        self.dsb_lr.setDecimals(6)
        self.dsb_lr.setRange(1e-06, 1.0)
        self.dsb_lr.setSingleStep(0.0001)
        self.dsb_lr.setValue(0.001)
        self.cmb_monitor = QComboBox()
        self.cmb_monitor.addItems(['val_loss', 'loss'])
        self.spn_patience = QSpinBox()
        self.spn_patience.setRange(1, 100)
        self.spn_patience.setValue(10)
        self.dsb_min_delta = QDoubleSpinBox()
        self.dsb_min_delta.setDecimals(6)
        self.dsb_min_delta.setRange(0.0, 1.0)
        self.dsb_min_delta.setSingleStep(0.0001)
        self.dsb_min_delta.setValue(0.0001)
        # 微结构：由导航页传入固定模型类型（CNN/ICNN），移除界面下拉
        self.fixed_model_type = fixed_model_type
        if self.level == '微结构' and isinstance(self.fixed_model_type, str) and self.fixed_model_type:
            self.setWindowTitle(f'微调训练 - {self.level}（{self.fixed_model_type}）')
        self.btn_start = QPushButton('Start')
        self.btn_pause = QPushButton('Pause')
        self.btn_resume = QPushButton('Resume')
        self.btn_stop = QPushButton('Stop')
        self.btn_save = QPushButton('保存模型…')
        self.btn_start.clicked.connect(self._start_training)
        self.btn_pause.clicked.connect(self._pause_training)
        self.btn_resume.clicked.connect(self._resume_training)
        self.btn_stop.clicked.connect(self._stop_training)
        self.btn_save.clicked.connect(self._save_model)
        right.addRow('Epochs', self.spn_epochs)
        right.addRow('Batch Size', self.spn_batch)
        right.addRow('Learning Rate', self.dsb_lr)
        right.addRow('Monitor', self.cmb_monitor)
        right.addRow('EarlyStop Patience', self.spn_patience)
        right.addRow('EarlyStop MinDelta', self.dsb_min_delta)
        ctrl_row = QWidget()
        ctrl_l = QHBoxLayout(ctrl_row)
        ctrl_l.setContentsMargins(0, 0, 0, 0)  # 去除工具栏外边距，使布局更紧凑
        for b in (self.btn_start, self.btn_pause, self.btn_resume, self.btn_stop, self.btn_save):
            ctrl_l.addWidget(b)
        right.addRow('', ctrl_row)
        # 下方：日志 + 过滤器
        log_box = QGroupBox('训练日志')
        log_layout = QVBoxLayout(log_box)
        filt_row = QHBoxLayout()
        filt_row.addWidget(QLabel('过滤：'))
        self.cmb_filter = QComboBox()
        self.cmb_filter.addItems(['ALL', 'INFO', 'WARN', 'ERROR'])
        self.cmb_filter.currentTextChanged.connect(self._refresh_log_view)
        filt_row.addWidget(self.cmb_filter)
        filt_row.addStretch(1)
        log_layout.addLayout(filt_row)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        log_layout.addWidget(self.log_view)
        # 进度
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        log_layout.addWidget(self.progress)
        # 左右分栏
        mid = QSplitter(Qt.Orientation.Horizontal)
        left_col = QWidget()
        left_layout = QVBoxLayout(left_col)
        left_layout.addWidget(self.toolbar)
        left_layout.addWidget(self.canvas)
        mid.addWidget(left_col)
        mid.addWidget(right_box)
        # 使用伸缩因子代替固定像素尺寸，左侧:右侧=4:1
        mid.setStretchFactor(0, 4)
        mid.setStretchFactor(1, 1)
        root.addWidget(mid)
        root.addWidget(log_box)
        # 训练状态（模拟器/占位）
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick_simulate)
        self.running = False
        self.paused = False
        self.cur_epoch = 0
        self.max_epochs = self.spn_epochs.value()
        self.loss_hist: List[float] = []
        self.val_loss_hist: List[float] = []
        self.best_val = float('inf')
        self.no_improve = 0
        self.all_logs: List[Tuple[str, str]] = []
        self._sklearn_success = False  # 添加标志位
        self._append_log('INFO', '训练窗口就绪。')

    def _data_summary_text(self) -> str:
        # 更新简要摘要以包含输出数据
        if self.level == '微结构':
            imgs = ', '.join(self.cfg.finetune_image_paths) if self.cfg.finetune_image_paths else '未选择图像目录'
            tbl = self.cfg.finetune_param_file or '未选择材料参数数据'
            out = getattr(self.cfg, 'finetune_output_file', '') or '未选择输出数据'
            return f'<b>层级：</b>{self.level}\u3000<b>图像目录：</b>{imgs}\u3000<b>材料参数数据：</b>{tbl}\u3000<b>输出数据：</b>{out}'
        else:
            tbl = self.cfg.finetune_param_file or '未选择材料参数数据'
            out = getattr(self.cfg, 'finetune_output_file', '') or '未选择输出数据'
            return f'<b>层级：</b>{self.level}\u3000<b>材料参数数据：</b>{tbl}\u3000<b>输出数据：</b>{out}'

    def _append_log(self, level: str, text: str):
        self.all_logs.append((level, text))
        self._refresh_log_view()
        self.write_log_to_main(f'[{level}] {text}')

    def _refresh_log_view(self):
        level = self.cmb_filter.currentText()
        self.log_view.clear()
        for lv, tx in self.all_logs:
            if level == 'ALL' or level == lv:
                self.log_view.appendPlainText(f'[{lv}] {tx}')

    def _stop_simulation(self, reason: str):
        """停止模拟训练（占位功能）。"""
        # --- 根据停止原因和 best_val 构造日志 ---
        if reason == '用户请求':
            msg = f'模拟训练中断，最优模型val_loss={self.best_val:.6f}'
        else:
            msg = f'模拟训练完成，最优模型val_loss={self.best_val:.6f}'
        self._append_log('INFO', msg)
        self.running = False
        self.paused = False
        self.timer.stop()
        self.progress.setValue(100)
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_save.setEnabled(False)  # 无法保存模拟模型

    def _start_training(self):
        try:
            if self.level == '微结构':
                model_type = self.fixed_model_type or 'CNN'
                if not self.cfg.finetune_image_paths:
                    self._append_log('ERROR', '未选择训练图像目录。')
                    return
                img_dir = self.cfg.finetune_image_paths[0]
                if not os.path.isdir(img_dir):
                    self._append_log('ERROR', f'图像目录不存在：{img_dir}')
                    return
                # 材料参数数据与输出数据必选
                param_file = self.cfg.finetune_param_file or ''
                out_file = getattr(self.cfg, 'finetune_output_file', '')
                if model_type == 'ICNN':
                    if not param_file:
                        self._append_log('ERROR', 'ICNN 微调需要选择材料参数数据（N×4，含表头）。')
                        return
                if not out_file:
                    self._append_log('ERROR', '未选择输出数据。')
                    return
                # 读取图像列表
                img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if
                             os.path.isfile(os.path.join(img_dir, f)) and f.lower().endswith(
                                 ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
                img_paths.sort(key=lambda p: os.path.basename(p).lower())
                n_imgs = len(img_paths)
                if n_imgs == 0:
                    self._append_log('ERROR', '训练目录中未发现任何支持格式的图像。')
                    return

                # 读取数据
                def read_table(fp):
                    if fp.lower().endswith(('.xlsx', '.xls')):
                        return pd.read_excel(fp)
                    return pd.read_csv(fp)

                y_df = read_table(out_file)
                if y_df.shape[0] == 0:
                    self._append_log('ERROR', '输出数据为空。')
                    return
                # 统一只保留数值列
                y_vals = y_df.select_dtypes(include='number').to_numpy(dtype='float32')
                if model_type == 'CNN':
                    # CNN：y 为 (N, d)
                    if y_vals.ndim == 1:
                        y_vals = y_vals.reshape(-1, 1)
                    if y_vals.shape[0] != n_imgs:
                        self._append_log('WARN',
                                         f'图像数量({n_imgs})与输出标签数量({y_vals.shape[0]})不一致，将按最小数量对齐。')
                        n = min(n_imgs, y_vals.shape[0])
                        img_paths = img_paths[:n]
                        y_vals = y_vals[:n, :]
                    else:
                        n = n_imgs
                    self._append_log('INFO', f'[CNN] 加载数据：样本数={n}，输出维度={y_vals.shape[1]}（即三个单轴强度）。')
                    # 预处理图像
                    X = []
                    for p in img_paths:
                        arr = preprocess_ms_image(p)
                        if arr is None:
                            self._append_log('ERROR', f'预处理失败：{p}')
                            return
                        X.append(arr)
                    X = np.asarray(X, dtype='float32')
                    # 数据集划分
                    X_tr, X_te, y_tr, y_te = train_test_split(X, y_vals, test_size=0.2, random_state=42, shuffle=True)
                    # 载入预训练 CNN
                    pt_path = self.cfg.get_pt('微结构', 'CNN') or ''
                    if not pt_path or not os.path.exists(pt_path):
                        self._append_log('ERROR', f'未选择或找不到预训练 CNN 模型：{pt_path}')
                        return
                    self._append_log('INFO', f'加载预训练 CNN 模型：{pt_path}')
                    try:
                        base = _ModelCache.get(pt_path)
                    except Exception as e:
                        self._append_log('ERROR', f'加载预训练模型失败：{e}')
                        return
                    try:
                        current_out = int(base.output_shape[-1])
                    except Exception:
                        current_out = None
                    if current_out is None or current_out != y_tr.shape[1]:
                        self._append_log('WARN',
                                         f'预训练CNN输出维度为 {current_out}，与标签维度 {y_tr.shape[1]} 不一致，自动添加适配层。')
                        out = layers.Dense(y_tr.shape[1], name='ft_out')(base.output)
                        model = Model(inputs=base.input, outputs=out)
                    else:
                        model = base
                    lr = float(self.dsb_lr.value())
                    monitor = self.cmb_monitor.currentText() or 'val_loss'
                    patience = int(self.spn_patience.value())
                    min_delta = float(self.dsb_min_delta.value())

                    # UI 状态
                    self.loss_hist.clear()
                    self.val_loss_hist.clear()
                    self._redraw()
                    self.progress.setValue(0)
                    self.btn_start.setEnabled(False)
                    self.btn_pause.setEnabled(True)
                    self.btn_resume.setEnabled(False)
                    self.btn_stop.setEnabled(True)
                    self.btn_save.setEnabled(False)
                    batch = int(self.spn_batch.value())
                    epochs = int(self.spn_epochs.value())
                    self._worker = TrainWorker(model, X_tr, y_tr, X_te, y_te, epochs, batch, monitor, patience,
                                               min_delta, lr, self)
                    self._worker.epoch_signal.connect(self._on_epoch_progress)
                    self._worker.done_signal.connect(self._on_train_done)
                    self._worker.error_signal.connect(self._on_train_error)
                    self._worker.start()
                else:
                    # === ICNN 部分 ===
                    x_df = read_table(param_file)
                    if x_df.shape[0] == 0:
                        self._append_log('ERROR', '材料参数数据为空。')
                        return
                    x_vals = x_df.select_dtypes(include='number').to_numpy(dtype='float32')
                    # 保留前4列
                    if x_vals.ndim == 1:
                        x_vals = x_vals.reshape(-1, 1)
                    if x_vals.shape[1] < 4 or y_vals.shape[1] < 4:
                        self._append_log('ERROR', '材料数据和输出数据都应至少包含4列数值数据。')
                        return
                    x_vals = x_vals[:, :4]
                    y4 = y_vals[:, :4]
                    # 对齐N
                    if x_vals.shape[0] != n_imgs or y4.shape[0] != n_imgs:
                        self._append_log('WARN', f'图像数量({n_imgs})与材料参数/输出数据行数不一致，将按最小数量对齐。')
                        n = min(n_imgs, x_vals.shape[0], y4.shape[0])
                        img_paths = img_paths[:n]
                        x_vals = x_vals[:n, :]
                        y4 = y4[:n, :]
                    else:
                        n = n_imgs
                    self._append_log('INFO',
                                     f'[ICNN] 加载数据：样本数={n}，每个样本包含一张图像和4个强度点(x,y)，将按列堆叠复用图像。')
                    # 预处理图像
                    Ximg = []
                    for p in img_paths:
                        arr = preprocess_ms_image(p)
                        if arr is None:
                            self._append_log('ERROR', f'预处理失败：{p}')
                            return
                        Ximg.append(arr)
                    Ximg = np.asarray(Ximg, dtype='float32')
                    # 划分索引（按图像）
                    idx_all = np.arange(n)
                    np.random.seed(42)
                    np.random.shuffle(idx_all)
                    n_tr = max(1, int(round(n * 0.8)))
                    tr_idx = np.sort(idx_all[:n_tr])
                    te_idx = np.sort(idx_all[n_tr:])
                    # 构造 x_true / y_true （列堆叠，Fortran 顺序）
                    x_tr = x_vals[tr_idx, :].reshape(-1, order='F').astype('float32').reshape(-1, 1)
                    y_tr = y4[tr_idx, :].reshape(-1, order='F').astype('float32').reshape(-1, 1)
                    x_te = x_vals[te_idx, :].reshape(-1, order='F').astype('float32').reshape(-1, 1)
                    y_te = y4[te_idx, :].reshape(-1, order='F').astype('float32').reshape(-1, 1)
                    X_tr = Ximg[tr_idx]
                    X_te = Ximg[te_idx]
                    # 载入预训练 ICNN
                    pt_path = self.cfg.get_pt('微结构', 'ICNN') or ''
                    if not pt_path or not os.path.exists(pt_path):
                        self._append_log('ERROR', f'未选择或找不到预训练 ICNN 模型：{pt_path}')
                        return
                    self._append_log('INFO', f'加载预训练 ICNN 模型：{pt_path}')
                    try:
                        custom_objects = {'rmse_metric': rmse_metric}
                        model = _ModelCache.get(pt_path, custom_objects=custom_objects)
                    except Exception as e:
                        self._append_log('ERROR', f'加载预训练模型失败：{e}')
                        return
                    # 若输出维度不为1，添加适配层
                    try:
                        out_dim = int(model.output_shape[-1])
                    except Exception:
                        out_dim = None
                    if out_dim is None or out_dim != 1:
                        self._append_log('WARN', f'预训练ICNN输出维度为 {out_dim}，将追加 Dense(1) 适配层。')
                        out = layers.Dense(1, name='ft_out')(model.output)
                        model = Model(inputs=model.input, outputs=out)
                    # 生成器
                    batch = int(self.spn_batch.value())
                    train_gen = CustomDataGenerator(X_tr, x_tr, y_tr, batch_size=batch, shuffle=True, is_training=True,
                                                    input_names=getattr(model, 'input_names', None))
                    val_gen = CustomDataGenerator(X_te, x_te, y_te, batch_size=batch, shuffle=False, is_training=True,
                                                  input_names=getattr(model, 'input_names', None))
                    lr = float(self.dsb_lr.value())
                    monitor = self.cmb_monitor.currentText() or 'val_loss'
                    patience = int(self.spn_patience.value())
                    min_delta = float(self.dsb_min_delta.value())

                    # 更新 UI 状态
                    self.loss_hist.clear()
                    self.val_loss_hist.clear()
                    self._redraw()
                    self.progress.setValue(0)
                    self.btn_start.setEnabled(False)
                    self.btn_pause.setEnabled(True)
                    self.btn_resume.setEnabled(False)
                    self.btn_stop.setEnabled(True)
                    self.btn_save.setEnabled(False)
                    epochs = int(self.spn_epochs.value())
                    self._worker = GenTrainWorker(model, train_gen, val_gen, epochs, monitor, patience, min_delta, lr,
                                                  self)
                    self._worker.epoch_signal.connect(self._on_epoch_progress)
                    self._worker.done_signal.connect(self._on_train_done)
                    self._worker.error_signal.connect(self._on_train_error)
                    self._worker.start()

            # --- 层合板微调逻辑 ---
            elif self.level == '层合板':
                self._append_log('INFO', '[层合板] 开始微调...')
                # 1. 验证输入
                param_file = self.cfg.finetune_param_file or ''
                out_file = getattr(self.cfg, 'finetune_output_file', '')
                if not param_file or not out_file:
                    self._append_log('ERROR', '层合板微调需要同时选择“材料参数数据”(输入)和“输出数据”(标签)。')
                    return
                # 2. 获取预训练模型路径 (用于定位依赖文件)
                pt_path = self.cfg.get_pt('层合板', 'default') or ''
                if not pt_path or not os.path.exists(pt_path):
                    self._append_log('ERROR', f'未选择或找不到预训练模型：{pt_path}。无法继续微调。')
                    return
                # 3. 准备 Scaler (基于 Laminate_train_data.csv，该文件应与模型在同一目录)
                model_dir = os.path.dirname(pt_path)
                train_csv_path = os.path.join(model_dir, 'Laminate_train_data.csv')
                if not os.path.exists(train_csv_path):
                    self._append_log('ERROR', f'依赖文件缺失: {train_csv_path}。无法拟合标准化缩放器。')
                    return
                try:
                    self._append_log('INFO', f'加载 {train_csv_path} 用于拟合缩放器...')
                    train_data = pd.read_csv(train_csv_path)
                    # Laminite_train_data.csv 的 2-10 列 (索引) 是特征
                    X_orig_train = train_data.iloc[:, 1:10]
                    scaler = StandardScaler().fit(X_orig_train)
                    self._append_log('INFO', '缩放器拟合完毕。')
                except Exception as e:
                    self._append_log('ERROR', f'加载或拟合 {os.path.basename(train_csv_path)} 失败: {e}')
                    return

                # 4. 加载新数据 (用于微调)
                def read_table(fp):
                    if fp.lower().endswith(('.xlsx', '.xls')):
                        return pd.read_excel(fp)
                    return pd.read_csv(fp)

                X_df = read_table(param_file)
                y_df = read_table(out_file)
                # 提取数值 (包含表头)
                X_new = X_df.select_dtypes(include='number').to_numpy(dtype='float32')
                y_new = y_df.select_dtypes(include='number').to_numpy(dtype='float32')
                if X_new.shape[0] != y_new.shape[0]:
                    self._append_log('ERROR',
                                     f'输入数据 ({X_new.shape[0]}行) 和输出数据 ({y_new.shape[0]}行) 数量不匹配。')
                    return
                if X_new.shape[1] < 9:  # 至少需要9个特征
                    self._append_log('ERROR', f'输入数据特征不足9列 (当前{X_new.shape[1]}列)。')
                    return
                # 使用前9个特征
                X_new = X_new[:, :9]
                if y_new.ndim == 1:
                    y_new = y_new.reshape(-1, 1)
                self._append_log('INFO', f'加载微调数据：{X_new.shape[0]} 个样本。')
                # 5. 划分数据集并缩放
                X_tr, X_te, y_tr, y_te = train_test_split(X_new, y_new, test_size=0.2, random_state=42,
                                                          shuffle=True)
                X_tr = scaler.transform(X_tr)
                X_te = scaler.transform(X_te)
                # 6. 载入预训练模型 (路径已在步骤2获取)
                self._append_log('INFO', f'加载预训练模型：{pt_path}')
                try:
                    model = _ModelCache.get(pt_path)
                except Exception as e:
                    self._append_log('ERROR', f'加载预训练模型失败：{e}')
                    return
                # 7. 启动 TrainWorker (非生成器)
                lr = float(self.dsb_lr.value())
                monitor = self.cmb_monitor.currentText() or 'val_loss'
                patience = int(self.spn_patience.value())
                min_delta = float(self.dsb_min_delta.value())

                # UI 状态
                self.loss_hist.clear()
                self.val_loss_hist.clear()
                self._redraw()
                self.progress.setValue(0)
                self.btn_start.setEnabled(False)
                self.btn_pause.setEnabled(True)
                self.btn_resume.setEnabled(False)
                self.btn_stop.setEnabled(True)
                self.btn_save.setEnabled(False)
                batch = int(self.spn_batch.value())
                epochs = int(self.spn_epochs.value())
                self._worker = TrainWorker(model, X_tr, y_tr, X_te, y_te, epochs, batch, monitor, patience,
                                           min_delta,
                                           lr, self)
                self._worker.epoch_signal.connect(self._on_epoch_progress)
                self._worker.done_signal.connect(self._on_train_done)
                self._worker.error_signal.connect(self._on_train_error)
                self._worker.start()

            # 加筋结构 (re-training) 逻辑
            elif self.level == '加筋结构':
                self._append_log('INFO', '[加筋结构] 开始微调 (重新训练)...')

                # 1. 检查子模块 (占位)
                selected_pt_mode = self.cfg.get_pt('加筋结构', 'default')

                # 启动“优化设计”的模拟训练
                if '优化设计' in selected_pt_mode:
                    self._append_log('INFO', f'[{self.level} - 优化设计] 启动模拟微调... (此为占位功能)')
                    # --- UI 状态 ---
                    self.loss_hist.clear()
                    self.val_loss_hist.clear()
                    self._redraw()
                    self.progress.setRange(0, 100)  # 确保它不是在不确定模式
                    self.progress.setValue(0)
                    self.btn_start.setEnabled(False)
                    self.btn_pause.setEnabled(True)
                    self.btn_resume.setEnabled(False)
                    self.btn_stop.setEnabled(True)
                    self.btn_save.setEnabled(False)  # 无法保存模拟模型
                    # --- 启动模拟器 ---
                    self.running = True
                    self.paused = False
                    self.cur_epoch = 0
                    self.max_epochs = int(self.spn_epochs.value())
                    self.best_val = float('inf')
                    self.no_improve = 0
                    # 启动计时器
                    self.timer.start(100)  # 每个模拟epoch 100ms
                    return

                if '性能预测' not in selected_pt_mode:
                    self._append_log('ERROR', '未在Dashboard选择“性能预测”预训练模式，无法确定微调目标。')
                    return

                self._append_log('INFO', '正在为“性能预测”模块准备数据...')
                # 2. 验证输入 (用户已提供完整的X和Y)
                param_file = self.cfg.finetune_param_file or ''
                out_file = getattr(self.cfg, 'finetune_output_file', '')
                if not param_file or not out_file:
                    self._append_log('ERROR', '加筋结构微调需要同时选择“材料参数数据”(输入X)和“输出数据”(标签Y)。')
                    return

                def read_table(fp):
                    if fp.lower().endswith(('.xlsx', '.xls')):
                        return pd.read_excel(fp)
                    return pd.read_csv(fp)

                # 3. 加载数据
                try:
                    self._append_log('INFO', f'加载输入数据 (X): {param_file}')
                    X_df = read_table(param_file)
                    self._append_log('INFO', f'加载输出数据 (Y): {out_file}')
                    Y_df = read_table(out_file)
                    if len(X_df) != len(Y_df):
                        self._append_log('ERROR', f'输入({len(X_df)})和输出({len(Y_df)})行数不匹配。')
                        return
                    X_all = X_df.select_dtypes(include=np.number).to_numpy(dtype='float32')
                    y_all = Y_df.select_dtypes(include=np.number).to_numpy(dtype='float32')
                    if X_all.shape[1] < 11:
                        self._append_log('ERROR', f'输入数据 (X) 应至少有 11 列，但检测到 {X_all.shape[1]} 列。')
                        return
                    if y_all.shape[1] < 5:
                        self._append_log('ERROR', f'输出数据 (Y) 应至少有 5 列，但检测到 {y_all.shape[1]} 列。')
                        return
                    X_all = X_all[:, :11]
                    y_all = y_all[:, :5]
                    self._append_log('INFO',
                                     f'数据加载完成。总行数: {len(X_all)} (X={X_all.shape[1]}列, Y={y_all.shape[1]}列)')

                    lr = float(self.dsb_lr.value())
                    epochs = int(self.spn_epochs.value())
                    patience = int(self.spn_patience.value())
                    min_delta = float(self.dsb_min_delta.value())

                    # --- UI 状态 ---
                    self.loss_hist.clear()
                    self.val_loss_hist.clear()
                    self._redraw()
                    self.progress.setValue(0)
                    self.progress.setRange(0, 100)
                    self.btn_start.setEnabled(False)

                    # --- 启用 Pause/Resume 按钮 ---
                    self._append_log('INFO', 'Sklearn (adam) 求解器支持 Epoch 间暂停。')
                    self.btn_pause.setEnabled(True)
                    self.btn_resume.setEnabled(False)
                    self.btn_stop.setEnabled(True)
                    self.btn_save.setEnabled(False)

                    self._sklearn_success = False  # 重置标志

                    self._worker = SklearnTrainWorker(X_all, y_all, epochs, lr, patience, min_delta, self)

                    # 定义一个辅助函数来解析来自 SklearnTrainWorker 的单字符串日志
                    def parse_and_log(log_str: str):
                        level = 'INFO'  # 默认级别
                        text = log_str
                        if log_str.startswith('[INFO]'):
                            text = log_str[6:].strip()
                        elif log_str.startswith('[WARN]'):
                            level = 'WARN'
                            text = log_str[6:].strip()
                        elif log_str.startswith('[ERROR]'):
                            level = 'ERROR'
                            text = log_str[7:].strip()
                        elif log_str.startswith('[CRITICAL]'):
                            level = 'ERROR'  # 统一映射为 ERROR
                            text = log_str[10:].strip()

                        # 使用解析后的参数调用正确的 _append_log
                        self._append_log(level, text)

                    # 将 log_signal 连接到新的辅助函数
                    self._worker.log_signal.connect(parse_and_log)
                    self._worker.done_signal.connect(self._on_sklearn_train_done)
                    self._worker.error_signal.connect(self._on_train_error)
                    self._worker.finished.connect(self._on_sklearn_finished)
                    self._worker.epoch_signal.connect(self._on_epoch_progress)
                    self._worker.start()
                except Exception as e:
                    self._append_log('ERROR', f'处理加筋结构数据时失败: {e}\n{traceback.format_exc()}')
                    return

            # --- 其它层级 (机身段) ---
            else:
                self._append_log('INFO', f'[{self.level}] 启动模拟微调... (此为占位功能)')
                # --- UI 状态 ---
                self.loss_hist.clear()
                self.val_loss_hist.clear()
                self._redraw()
                self.progress.setRange(0, 100)  # 确保它不是在不确定模式
                self.progress.setValue(0)
                self.btn_start.setEnabled(False)
                self.btn_pause.setEnabled(True)
                self.btn_resume.setEnabled(False)
                self.btn_stop.setEnabled(True)
                self.btn_save.setEnabled(False)  # 无法保存模拟模型
                # --- 启动模拟器 ---
                self.running = True
                self.paused = False
                self.cur_epoch = 0
                self.max_epochs = int(self.spn_epochs.value())
                self.best_val = float('inf')
                self.no_improve = 0
                # 启动计时器
                self.timer.start(100)  # 每个模拟epoch 100ms

        except Exception as e:
            self._append_log('ERROR', f'训练过程中出现异常：{e}\n{traceback.format_exc()}')

    def _save_model(self):
        # 1. 确定保存的文件类型和子键
        is_stiffened_predict = False
        sub_key_model = 'default'
        sub_key_scaler = None
        save_filter = 'Keras Model (*.h5)'
        default_name = f'{self.level}_finetuned.h5'

        if self.level == '微结构':
            sub_key_model = self.fixed_model_type if self.fixed_model_type else 'CNN'
            save_filter = 'Keras Model (*.h5)'
            default_name = f'{self.level}_{sub_key_model}_finetuned.h5'

        elif self.level == '层合板':
            sub_key_model = 'default'
            save_filter = 'Keras Model (*.h5)'
            default_name = f'{self.level}_finetuned.h5'

        elif self.level == '加筋结构':
            selected_pt_mode = self.cfg.get_pt('加筋结构', 'default')
            if '性能预测' in selected_pt_mode:
                is_stiffened_predict = True
                save_filter = 'Scikit-learn Model (*.joblib)'
                default_name = 'model_shear&buckle_finetuned.joblib'
                sub_key_model = 'predict_model'  # 自定义键
                sub_key_scaler = 'predict_scaler'  # 自定义键
            else:
                self._append_log('ERROR', '无法保存：未识别的加筋结构微调模式。')
                return
        else:
            self._append_log('ERROR', f'无法保存：层级 {self.level} 的保存逻辑未定义。')
            return

        # 2. 弹出保存对话框 (只保存主模型)
        path, _ = get_save_file(self, f'保存 {self.level} 微调模型', filter=save_filter, default_dir=default_name)
        if not path:
            return

        # 确保文件扩展名
        if save_filter == 'Keras Model (*.h5)' and not path.lower().endswith('.h5'):
            path += '.h5'
        elif save_filter == 'Scikit-learn Model (*.joblib)' and not path.lower().endswith('.joblib'):
            path += '.joblib'

        saved_model_path = None
        saved_scaler_path = None

        # 3. 保存
        try:
            if is_stiffened_predict:
                # --- 保存 Joblib (Model + Scaler) ---
                if not joblib:
                    self._append_log('ERROR', 'Joblib 未加载，无法保存。')
                    return
                if hasattr(self, 'trained_model') and self.trained_model:
                    joblib.dump(self.trained_model, path)
                    self._append_log('INFO', f'已保存 Scikit-learn 模型到：{path}')
                    saved_model_path = path
                else:
                    self._append_log('ERROR', '未找到 "trained_model" (sklearn)，无法保存。')
                    return

                if hasattr(self, 'trained_scaler') and self.trained_scaler:
                    scaler_path = os.path.splitext(path)[0].replace('_new', '') + '_scaler_new.joblib'
                    # 兼容性命名：model_shear&buckle_scaler_new.joblib
                    if 'model_shear&buckle' in scaler_path:
                        scaler_path = os.path.join(os.path.dirname(path), 'model_shear&buckle_scaler_new.joblib')

                    joblib.dump(self.trained_scaler, scaler_path)
                    self._append_log('INFO', f'已保存 Scikit-learn Scaler 到：{scaler_path}')
                    saved_scaler_path = scaler_path
                else:
                    self._append_log('ERROR', '未找到 "trained_scaler" (sklearn)，无法保存 Scaler。')

            else:
                # --- 保存 Keras (.h5) (微结构/层合板) ---
                src = getattr(self, '_best_ckpt_path', None)
                if src and os.path.exists(src):
                    shutil.copy2(src, path)
                    self._append_log('INFO', f'已保存最佳 Keras 模型到：{path}')
                    saved_model_path = path
                elif hasattr(self, 'trained_model') and self.trained_model:
                    self.trained_model.save(path)
                    self._append_log('INFO', f'已保存当前 Keras 模型到：{path}')
                    saved_model_path = path
                else:
                    self._append_log('WARN', '尚无可保存的 Keras 模型。')
                    return

        except Exception as e:
            self._append_log('ERROR', f'保存失败：{e}\n{traceback.format_exc()}')
            return

        # 4. 更新全局 Config
        if saved_model_path:
            self.cfg.set_ft(self.level, saved_model_path, sub=sub_key_model)
        if saved_scaler_path and sub_key_scaler:
            self.cfg.set_ft(self.level, saved_scaler_path, sub=sub_key_scaler)

        self.on_cfg_changed()

    def _tick_simulate(self):
        """模拟训练的定时器滴答事件（用于占位）。"""
        if not self.running or self.paused:
            return
        self.cur_epoch += 1
        base = 1.0 / (self.cur_epoch ** 0.5 + 1e-06)
        loss = base + random.uniform(-0.02, 0.02) + 0.1
        val_loss = base + random.uniform(-0.03, 0.03) + 0.12
        self._push_metric(loss, val_loss)
        self.progress.setValue(int(100 * self.cur_epoch / self.max_epochs))
        monitor = self.cmb_monitor.currentText()
        cur = val_loss if monitor == 'val_loss' else loss
        if cur + self.dsb_min_delta.value() < self.best_val:
            self.best_val = cur
            self.no_improve = 0
            self._append_log('INFO', f'epoch {self.cur_epoch}: {monitor} improved to {self.best_val:.4f}')
        else:
            self.no_improve += 1
            self._append_log('INFO', f'epoch {self.cur_epoch}: {monitor} no improve ({self.no_improve}/{self.spn_patience.value()})')
            if self.no_improve >= self.spn_patience.value():
                # --- 移除日志，传递原因 ---
                self._stop_training(reason='EarlyStopping')
        if self.cur_epoch >= self.max_epochs:
            # --- 移除日志，传递原因 ---
            self._stop_training(reason='Completed')

    def _push_metric(self, loss: float, val_loss: float):
        self.loss_hist.append(max(loss, 0.0))
        self.val_loss_hist.append(max(val_loss, 0.0))
        self._redraw()

    def _redraw(self):
        xs = list(range(1, len(self.loss_hist) + 1))
        self.line_train.set_data(xs, self.loss_hist)
        self.line_val.set_data(xs, self.val_loss_hist)
        self.ax.relim()
        self.ax.autoscale()
        self.canvas.draw_idle()

    def _pause_training(self):
        if hasattr(self, '_worker') and self._worker is not None:
            # 检查工作线程是否支持暂停
            if hasattr(self._worker, 'request_pause'):
                self._worker.request_pause()
                if isinstance(self._worker, SklearnTrainWorker):
                    self._append_log('INFO', '已请求暂停训练 (将在当前Epoch结束后生效)')
                else:
                    self._append_log('INFO', '已请求暂停训练')
                self.btn_pause.setEnabled(False)
                self.btn_resume.setEnabled(True)
                self.btn_stop.setEnabled(True)
            else:
                self._append_log('WARN', '当前工作线程不支持暂停。')
        elif self.running and not self.paused: # 检查模拟器
            self.paused = True
            self._append_log('INFO', '已暂停模拟训练')
            self.btn_pause.setEnabled(False)
            self.btn_resume.setEnabled(True)
            self.btn_stop.setEnabled(True)

    def _resume_training(self):
        if hasattr(self, '_worker') and self._worker is not None:
            if hasattr(self._worker, 'request_resume'):
                self._worker.request_resume()
                self._append_log('INFO', '已恢复训练')
                self.btn_pause.setEnabled(True)
                self.btn_resume.setEnabled(False)
                self.btn_stop.setEnabled(True)
            else:
                self._append_log('WARN', '当前工作线程不支持恢复。')
        elif self.running and self.paused: # 检查模拟器
            self.paused = False
            self._append_log('INFO', '已恢复模拟训练')
            self.btn_pause.setEnabled(True)
            self.btn_resume.setEnabled(False)
            self.btn_stop.setEnabled(True)

    def _stop_training(self, reason='用户请求'):
        if hasattr(self, '_worker') and self._worker is not None:
            if isinstance(self._worker, (TrainWorker, GenTrainWorker)):
                self._worker.request_stop()
            # 停止 Sklearn 训练
            elif isinstance(self._worker, SklearnTrainWorker):
                self._worker.request_stop()
            else:
                # 备用
                if hasattr(self._worker, 'request_stop'):
                    self._worker.request_stop()
                else:
                    self._worker.terminate()
        elif self.running:  # 检查模拟器
            # --- 传递 reason ---
            self._stop_simulation(reason)

        self.btn_stop.setEnabled(False)  # 立即禁用停止按钮，防止重复点击

    def _on_epoch_progress(self, epoch, loss, val_loss):
        self.loss_hist.append(float(loss))
        self.val_loss_hist.append(float(val_loss))
        self._redraw()
        total = max(1, int(self.spn_epochs.value()))
        self.progress.setValue(int(100 * epoch / total))
        self._append_log('INFO', f'epoch {epoch}: loss={loss:.6f}, val_loss={val_loss:.6f}')

    def _on_train_done(self, msg, best_val):
        self.trained_model = getattr(self, '_worker', None).model if getattr(self, '_worker', None) else None
        if msg == "STOPPED":
            # --- 在停止时也显示最优 val_loss ---
            self._append_log('INFO', f'Keras 训练已终止。最优 val_loss={best_val:.6f}')
        else:
            self._append_log('INFO', f"训练完成。最优 val_loss={best_val:.6f}")
        self.progress.setValue(100)
        for b in (self.btn_start, self.btn_pause, self.btn_resume, self.btn_stop, self.btn_save):
            b.setEnabled(True)
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_save.setEnabled(True)

    def _on_train_error(self, tb_text):
        self._append_log('ERROR', f'训练线程异常：\n{tb_text}')
        for b in (self.btn_start, self.btn_pause, self.btn_resume, self.btn_stop, self.btn_save):
            b.setEnabled(True)
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_save.setEnabled(False)

    def _on_sklearn_train_done(self, status, model, scaler, best_val_loss):
        self._sklearn_success = True  # 标记为成功
        # 训练成功，保存模型和scaler到临时变量
        self.trained_model = model
        self.trained_scaler = scaler  # 存储scaler
        # --- 根据状态显示不同的日志 ---
        if status == "STOPPED":
            self._append_log('INFO', f'MLPRegressor 训练中断。最优 val_loss={best_val_loss:.6f}')
        else:  # "COMPLETED" (或任何其他非停止状态)
            self._append_log('INFO', f'MLPRegressor 训练完成。最优 val_loss={best_val_loss:.6f}')
        self.btn_save.setEnabled(True)  # 允许保存

    def _on_sklearn_finished(self):
        # 无论成功还是失败，都重置UI
        if not hasattr(self, 'btn_start'): return  # 窗口可能已关闭
        # --- 移除冗余日志 ---
        self.progress.setRange(0, 100)  # 恢复确定模式
        self.progress.setValue(100)
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(False)
        self.btn_stop.setEnabled(False)
        # btn_save 状态由 _on_sklearn_train_done 控制

    def closeEvent(self, e):
        if hasattr(self, '_worker') and self._worker is not None:
            # --- 只在线程仍在运行时才请求停止 ---
            if self._worker.isRunning():
                self._worker.request_stop()
                # 给线程一点时间，让它优雅地退出
                self._worker.wait(200)
        return super().closeEvent(e)

class DashboardPage(QWidget):
    # 统一的层级顺序
    LEVELS = ['微结构', '层合板', '加筋结构', '机身段']

    def _open_help(self):
        """打开帮助手册（仅：Manuals and Tutorials/帮助文档.pdf）"""
        return open_help_pdf(self)

    def __init__(self, config: ProjectConfig, on_go_predict, on_go_finetune, on_log, open_ft_window, on_cfg_changed):
        super().__init__()
        self.cfg = config
        self.on_go_predict = on_go_predict
        self.on_go_finetune = on_go_finetune
        self.log = on_log
        self.open_ft_window = open_ft_window
        self.on_cfg_changed = on_cfg_changed
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 16, 24, 16)
        root.setSpacing(16)
        title = QLabel('航空复合材料多层级智能设计平台')
        subtitle = QLabel('从微结构(μm)到机身段(m)，一站式建模 · 学习 · 预测 · 优化')
        title.setObjectName('h1')
        subtitle.setObjectName('sub')
        root.addWidget(title)
        root.addWidget(subtitle)
        # === 插入主图 Banner（从 ./assets 自动选择一张图片） ===
        self.banner = BannerLabel(find_banner_image(), fixed_height=220)
        # 如果图片缺失或加载失败，静默跳过，不影响其他功能
        if getattr(self.banner, 'is_valid', False):
            root.addWidget(self.banner)
        # -- 导航入口 --
        entry_box = QGroupBox('导航入口')
        form = QFormLayout(entry_box)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        self.edt_proj = QLineEdit(self.cfg.name)
        self.edt_proj.setPlaceholderText('Project')
        self.cmb_level = QComboBox()
        self.cmb_level.addItems(self.LEVELS)
        self.cmb_units = QComboBox()
        self.cmb_units.addItems(['SI', 'Imperial'])
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(['预训练', '微调'])
        # 动态预训练模型区：微结构（A = CNN / B = ICNN）或其它（default）
        self.pt_widget = QWidget()
        self.pt_layout = QGridLayout(self.pt_widget)
        self.pt_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.pt_layout.setContentsMargins(0, 0, 0, 0)  # 去除工具栏外边距，使布局更紧凑
        self.pt_layout.setVerticalSpacing(0)
        self.cmb_pretrained_single = None
        self.cmb_pretrained_A = None
        self.cmb_pretrained_B = None
        form.addRow('项目名称', self.edt_proj)
        form.addRow('选择层级', self.cmb_level)
        form.addRow('单位制', self.cmb_units)
        form.addRow('模型模式', self.cmb_mode)
        lbl_pt = QLabel('预训练模型')
        form.addRow(lbl_pt, self.pt_widget)
        # 微调数据区域（在“微调”模式下启用）
        ft_box = QGroupBox('微调训练数据（仅“微调”模式必填）')
        ft_box.setObjectName('ft_box')
        self.ft_layout = QGridLayout(ft_box)
        # 行：训练图像目录（仅微结构需要） -- 左对齐 + 同宽
        self.row_img = QWidget()
        row_img_layout = QHBoxLayout(self.row_img)
        row_img_layout.setContentsMargins(0, 0, 0, 0)  # 去除工具栏外边距，使布局更紧凑
        row_img_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.btn_pick_imgs = QPushButton('选择训练图像目录')
        # 用于显示选中训练图像目录；默认显示“未选择”，过长路径截断显示使用可自动截断的标签显示训练图像目录路径
        self.lbl_img_cnt = ElidedLabel('未选择')
        # 设置可选中文本
        self.lbl_img_cnt.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        # 使标签在布局中水平扩展，避免初始宽度为 0 导致闪烁
        self.lbl_img_cnt.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        # 设置完整路径提示
        self.lbl_img_cnt.set_full_text('未选择')
        self.btn_pick_imgs.clicked.connect(self._choose_image_dir)
        row_img_layout.addWidget(self.btn_pick_imgs)
        row_img_layout.addWidget(self.lbl_img_cnt)
        # 行：材料参数数据（所有层级微调都需要） -- 左对齐 + 同宽
        self.row_tbl = QWidget()
        row_tbl_layout = QHBoxLayout(self.row_tbl)
        row_tbl_layout.setContentsMargins(0, 0, 0, 0)  # 去除工具栏外边距，使布局更紧凑
        row_tbl_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.btn_pick_tbl = QPushButton('选择材料参数数据(Excel/CSV)')
        # 不对按钮设定固定宽度，允许布局自动分配大小，使用可自动截断的标签显示材料参数数据路径
        self.lbl_tbl_path = ElidedLabel('未选择')
        self.lbl_tbl_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_tbl_path.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.lbl_tbl_path.set_full_text('未选择')
        self.btn_pick_tbl.clicked.connect(self._choose_param_table)
        row_tbl_layout.addWidget(self.btn_pick_tbl)
        row_tbl_layout.addWidget(self.lbl_tbl_path)
        self.ft_layout.addWidget(self.row_img, 0, 0, 1, 2)
        self.ft_layout.addWidget(self.row_tbl, 1, 0, 1, 2)
        # 行：输出数据（所有层级微调都需要）
        self.row_out = QWidget()
        row_out_layout = QHBoxLayout(self.row_out)
        row_out_layout.setContentsMargins(0, 0, 0, 0)  # 去除工具栏外边距，使布局更紧凑
        row_out_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.btn_pick_out = QPushButton('选择输出数据(Excel/CSV)')
        # 不对按钮设定固定宽度，允许布局自动分配大小，使用可自动截断的标签显示输出数据路径
        self.lbl_out_path = ElidedLabel('未选择')
        self.lbl_out_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_out_path.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.lbl_out_path.set_full_text('未选择')
        self.btn_pick_out.clicked.connect(self._choose_output_table)
        row_out_layout.addWidget(self.btn_pick_out)
        row_out_layout.addWidget(self.lbl_out_path)
        self.ft_layout.addWidget(self.row_out, 2, 0, 1, 2)
        form.addRow(ft_box)
        # 操作
        op_row = QWidget()
        op_layout = QHBoxLayout(op_row)
        op_layout.setContentsMargins(0, 0, 0, 0)  # 去除工具栏外边距，使布局更紧凑
        self.btn_go_predict = QPushButton('开始设计 →')
        self.btn_go_finetune = QPushButton('模型微调 →')
        # 微结构微调列表按钮（CNN/ICNN）
        self.btn_go_finetune_menu = QToolButton()
        self.btn_go_finetune_menu.setText('模型微调 →')
        self.btn_go_finetune_menu.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.menu_ft = QMenu(self.btn_go_finetune_menu)
        self.act_ft_cnn = self.menu_ft.addAction('微调 CNN 模型')
        self.act_ft_icnn = self.menu_ft.addAction('微调 ICNN 模型')
        self.act_ft_cnn.triggered.connect(self._go_finetune_cnn)
        self.act_ft_icnn.triggered.connect(self._go_finetune_icnn)
        self.btn_go_finetune_menu.setMenu(self.menu_ft)
        # -- 统一“开始设计 / 模型微调”按钮尺寸 --
        self.btn_go_predict.setMinimumHeight(32)
        self.btn_go_finetune.setMinimumHeight(32)
        self.btn_go_finetune_menu.setMinimumHeight(32)
        sp = self.btn_go_predict.sizePolicy()
        sp.setHorizontalPolicy(QSizePolicy.Policy.Expanding)
        sp.setVerticalPolicy(QSizePolicy.Policy.Fixed)
        self.btn_go_predict.setSizePolicy(sp)
        sp2 = self.btn_go_finetune.sizePolicy()
        sp2.setHorizontalPolicy(QSizePolicy.Policy.Expanding)
        sp2.setVerticalPolicy(QSizePolicy.Policy.Fixed)
        self.btn_go_finetune.setSizePolicy(sp2)
        sp3 = self.btn_go_finetune_menu.sizePolicy()
        sp3.setHorizontalPolicy(QSizePolicy.Policy.Expanding)
        sp3.setVerticalPolicy(QSizePolicy.Policy.Fixed)
        self.btn_go_finetune_menu.setSizePolicy(sp3)
        self.btn_go_predict.clicked.connect(self._go_predict)
        # 默认仅显示普通“模型微调”，当层级=微结构且模式=微调时隐藏它并显示下拉菜单按钮
        self.btn_go_finetune.clicked.connect(self._go_finetune)
        op_layout.addWidget(self.btn_go_predict)
        op_layout.addWidget(self.btn_go_finetune)
        op_layout.addWidget(self.btn_go_finetune_menu)
        self.btn_go_finetune_menu.setVisible(False)
        # 直接将操作行作为单独一行添加，让其横跨整个表单宽度，避免在左侧留空的标签列。
        try:
            form.addRow(op_row)
        except Exception:
            # 如果单独添加失败，则回退到原来的方式
            form.addRow('', op_row)
        root.addWidget(entry_box, 2)
        # 最近项目
        recent_box = QGroupBox('最近项目')
        recent_layout = QVBoxLayout(recent_box)
        self.recent = QListWidget()
        for name in ['WingSkin_RVE_2024', 'AS4_8552_Transverse', 'Stiffened_Panel_Demo']:
            QListWidgetItem(get_std_icon('folder', self), name, self.recent)
        recent_layout.addWidget(self.recent)
        root.addWidget(recent_box, 1)
        # -- 统一主页按钮高度（以“微调数据”三个按钮为标准） --
        _H = 32
        for w in self.findChildren((QPushButton, QToolButton)):
            if w.text().strip():
                w.setMinimumHeight(_H)
                pol = w.sizePolicy()
                pol.setVerticalPolicy(QSizePolicy.Policy.Fixed)
                w.setSizePolicy(pol)
        root.addStretch(1)
        # 初始化
        self.cmb_level.currentTextChanged.connect(self._on_level_changed)
        self.cmb_mode.currentTextChanged.connect(self._on_mode_changed)
        self._on_level_changed(self.cmb_level.currentText())
        self._on_mode_changed(self.cmb_mode.currentText())
        self._update_ft_go_button()

    # -- 预训练模型区：按层级重建 --
    def _rebuild_pt_area(self, level: str):
        # 清空
        while self.pt_layout.count():
            item = self.pt_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.cmb_pretrained_single = None
        self.cmb_pretrained_A = None
        self.cmb_pretrained_B = None

        if level == '微结构':
            self.cmb_pretrained_A = QComboBox()
            self.cmb_pretrained_B = QComboBox()
            # 自动查找 Model/微结构/CNN 和 ICNN 目录下的模型文件
            try:
                cnn_dir = os.path.join(MODEL_DIR, '微结构', 'CNN')
                if os.path.isdir(cnn_dir):
                    self.cmb_pretrained_A.addItems([f for f in os.listdir(cnn_dir) if f.endswith('.h5')])
            except Exception:
                pass  # 目录不存在或读取失败
            try:
                icnn_dir = os.path.join(MODEL_DIR, '微结构', 'ICNN')
                if os.path.isdir(icnn_dir):
                    self.cmb_pretrained_B.addItems([f for f in os.listdir(icnn_dir) if f.endswith('.h5')])
            except Exception:
                pass

            # 如果未找到任何模型，添加占位符
            if self.cmb_pretrained_A.count() == 0:
                self.cmb_pretrained_A.addItems(['ResNet-101-3.h5'])  # 默认回退
            if self.cmb_pretrained_B.count() == 0:
                self.cmb_pretrained_B.addItems(['ICNN.h5'])  # 默认回退

            self.pt_layout.addWidget(QLabel('CNN 模型'), 0, 0)
            self.pt_layout.addWidget(self.cmb_pretrained_A, 0, 1)
            self.pt_layout.addWidget(QLabel('ICNN 模型'), 1, 0)
            self.pt_layout.addWidget(self.cmb_pretrained_B, 1, 1)

        else:
            self.cmb_pretrained_single = QComboBox()
            models = []

            if level == '层合板':
                # 自动查找 Model/层合板 目录下的模型
                try:
                    lvl_dir = os.path.join(MODEL_DIR, level)
                    if os.path.isdir(lvl_dir):
                        models = [f for f in os.listdir(lvl_dir) if f.endswith('.h5')]
                except Exception:
                    pass
                if not models:
                    models = ['Laminate-Strength-v1.h5', 'Laminate-Buckle-v1.h5']  # 默认回退

            elif level == '加筋结构':
                # 为加筋结构提供模式选择
                models = [
                    '性能预测 (屈曲载荷+模态)',
                    '优化设计 (遗传算法)'
                ]

            elif level == '机身段':
                # 自动查找 Model/机身段 目录下的模型
                try:
                    lvl_dir = os.path.join(MODEL_DIR, level)
                    if os.path.isdir(lvl_dir):
                        models = [f for f in os.listdir(lvl_dir) if f.endswith('.h5')]
                except Exception:
                    pass
                if not models:
                    models = ['Fuselage-Sizing-v1.h5']  # 默认回退

            self.cmb_pretrained_single.addItems(models)
            self.pt_layout.addWidget(self.cmb_pretrained_single, 0, 0, 1, 2)

    # -- 辅助函数 --
    def _on_level_changed(self, text: str):
        self._rebuild_pt_area(text)
        self._refresh_finetune_area_visibility()
        self._update_ft_go_button()

    def _on_mode_changed(self, text: str):
        self._refresh_finetune_area_visibility()
        self._update_ft_go_button()  # 由数据完整性决定启用/禁用

    def _refresh_finetune_area_visibility(self):
        """微调数据区域在任意模式下都显示；仅在“微调”模式下可编辑。"""
        is_ft = self.cmb_mode.currentText() == '微调'
        # 区域始终可见
        box = self.findChild(QGroupBox, 'ft_box')
        if box is not None:
            box.setVisible(True)
            # 控件启/禁
            for w in box.findChildren(QWidget):
                # 标签始终可读；按钮和输入在非微调时禁用
                if hasattr(w, 'setEnabled'):
                    w.setEnabled(is_ft)
        # 微结构行的可见性仍由层级决定（但在非微调时也保持显示，只是禁用）
        is_micro = self.cmb_level.currentText() == '微结构'
        self.row_img.setVisible(is_micro)

    def _update_ft_go_button(self):
        """
        根据选择的微调数据是否完整来启用/禁用“模型微调”按钮。
        - 微结构：图像目录 + 材料参数数据 + 输出数据 + 预训练CNN/ICNN均已选择。
        - 其他层级：材料参数数据 + 输出数据 + 预训练模型已选择。
        """
        is_ft_mode = self.cmb_mode.currentText() == '微调'
        if not is_ft_mode:
            self.btn_go_finetune.setEnabled(False)
            self.btn_go_finetune_menu.setEnabled(False)
            return
        level = self.cmb_level.currentText()
        # 数据就绪性：参数/输出（以及微结构图像目录）
        has_param = bool(self.cfg.finetune_param_file)
        has_out = bool(getattr(self.cfg, 'finetune_output_file', ''))
        if level == '微结构':
            has_imgs = bool(self.cfg.finetune_image_paths)
            data_ok = has_imgs and has_param and has_out
        else:
            data_ok = has_param and has_out
        # 预训练模型是否选择（“微调的模型 = 预训练模型”，因此强制要求）
        pretrained_ok = False
        if level == '微结构':
            # 直接读取当前下拉框状态，避免必须先点击“模型微调”才写入 cfg
            sel_a = (self.cmb_pretrained_A.currentText() if self.cmb_pretrained_A else '').strip()
            sel_b = (self.cmb_pretrained_B.currentText() if self.cmb_pretrained_B else '').strip()
            pretrained_ok = bool(sel_a) and bool(sel_b)
        else:
            sel_s = (self.cmb_pretrained_single.currentText() if self.cmb_pretrained_single else '').strip()
            pretrained_ok = bool(sel_s)
        ok = data_ok and pretrained_ok
        # 按层级调整微调区可见性
        is_micro = level == '微结构'
        self.row_img.setVisible(is_micro)
        self.row_tbl.setVisible(True)
        self.row_out.setVisible(True)
        # 统一按钮可见性：由层级决定，不因模式切换隐藏
        if is_micro:
            self.btn_go_finetune.setVisible(False)
            self.btn_go_finetune_menu.setVisible(True)
            self.btn_go_finetune_menu.setEnabled(ok if is_ft_mode else False)
        else:
            self.btn_go_finetune_menu.setVisible(False)
            self.btn_go_finetune.setVisible(True)
            self.btn_go_finetune.setEnabled(ok if is_ft_mode else False)

    def _choose_image_dir(self):
        # 使用自定义的居中目录对话框
        d = get_existing_directory(self, '选择训练图像目录')
        if d:
            self.cfg.finetune_image_paths = [d]
            # 设置完整路径并自动截断显示
            try:
                self.lbl_img_cnt.set_full_text(d)
            except Exception:
                self.lbl_img_cnt.set_full_text(d)
            self.on_cfg_changed()
            self._update_ft_go_button()

    def _choose_param_table(self):
        # 使用自定义的居中文件打开对话框
        f, _ = get_open_file(self, '选择材料参数数据', 'Tables (*.csv *.xlsx)')
        if f:
            self.cfg.finetune_param_file = f
            # 设置完整路径并自动截断显示
            try:
                self.lbl_tbl_path.set_full_text(f)
            except Exception:
                self.lbl_tbl_path.set_full_text(f)
            self.on_cfg_changed()
            self._update_ft_go_button()

    def _choose_output_table(self):
        # 使用自定义的居中文件打开对话框
        f, _ = get_open_file(self, '选择输出数据', 'Tables (*.csv *.xlsx)')
        if f:
            self.cfg.finetune_output_file = f
            if hasattr(self, 'lbl_out_path'):
                try:
                    self.lbl_out_path.set_full_text(f)
                except Exception:
                    self.lbl_out_path.set_full_text(f)
            self.on_cfg_changed()
            self._update_ft_go_button()

    # -- 教程：入门示例 --
    def _open_intro(self, level: str):
        """打开入门示例（统一入口）"""
        return open_intro_pdf(level, self)

    def _download_samples(self, level: str):
        """微调数据示例下载（统一入口）"""
        return save_samples_xlsx(level, self)

    def _collect_cfg(self):
        self.cfg.name = self.edt_proj.text().strip() or 'Project'
        new_level = self.cmb_level.currentText()
        # 仅当层级实际更改时记录日志/更新，避免冗余更新
        level_changed = (new_level != self.cfg.level)
        self.cfg.level = new_level
        self.cfg.units = self.cmb_units.currentText()
        new_mode_text = self.cmb_mode.currentText()
        mode_changed = (self.cfg.get_mode(self.cfg.level) != new_mode_text)
        self.cfg.set_mode(self.cfg.level, new_mode_text)

        # --- 预训练模型选择 ---
        pt_changed = False
        if self.cfg.level == '微结构':
            a = self.cmb_pretrained_A.currentText() if self.cmb_pretrained_A else ''
            b = self.cmb_pretrained_B.currentText() if self.cmb_pretrained_B else ''
            old_a = self.cfg.get_pt('微结构', 'CNN')
            old_b = self.cfg.get_pt('微结构', 'ICNN')
            if a != old_a or b != old_b:
                self.cfg.set_pt('微结构', a, 'CNN')
                self.cfg.set_pt('微结构', b, 'ICNN')
                pt_changed = True
        else:
            s = self.cmb_pretrained_single.currentText() if self.cmb_pretrained_single else ''
            old_s = self.cfg.get_pt(self.cfg.level, 'default')
            # 对于加筋结构, 's' 是模式名称 ("性能预测..." 或 "优化设计...")
            # 对于其他层级, 's' 是模型文件名
            if s != old_s:
                self.cfg.set_pt(self.cfg.level, s, 'default')
                pt_changed = True

        # --- 更新 Inspector ---
        # 如果层级、模式或预训练模型发生更改，则更新
        if level_changed or mode_changed or pt_changed:
            self.on_cfg_changed()  # 触发 Inspector 更新

    # -- 校验微调输入（仅在微调模式有效） --
    def _check_ft_inputs(self) -> bool:
        if self.cfg.get_mode(self.cfg.level) != '微调':
            return True
        # 所有层级：必须选择 材料参数数据 + 输出数据
        if not self.cfg.finetune_param_file:
            if hasattr(self, 'log'):
                self.log('[WARN] 微调需要选择材料参数数据。')
            return False
        if not getattr(self.cfg, 'finetune_output_file', ''):
            if hasattr(self, 'log'):
                self.log('[WARN] 微调需要选择输出数据。')
            return False
        level = self.cfg.level or self.cmb_level.currentText()
        # 微结构额外：必须选择训练图像目录
        if level == '微结构' and (not self.cfg.finetune_image_paths):
            if hasattr(self, 'log'):
                self.log('[WARN] 微结构微调需要选择训练图像目录。')
            return False
        # 必须先“选择预训练模型”，否则不能进行模型微调
        if level == '微结构':
            a = (self.cmb_pretrained_A.currentText() if getattr(self, 'cmb_pretrained_A', None) else '').strip()
            b = (self.cmb_pretrained_B.currentText() if getattr(self, 'cmb_pretrained_B', None) else '').strip()
            if not a or not b:
                if hasattr(self, 'log'):
                    self.log('[WARN] 微调前请先选择预训练模型（CNN 与 ICNN）。')
                return False
        else:
            s = (self.cmb_pretrained_single.currentText() if getattr(self, 'cmb_pretrained_single', None) else '').strip()
            if not s:
                if hasattr(self, 'log'):
                    self.log('[WARN] 微调前请先选择预训练模型。')
                return False
        return True

    # -- 开始设计 --
    def _go_predict(self):
        # 若项目名称为空，则自动填充为 Project（并写入文本框）
        if not self.edt_proj.text().strip():
            self.edt_proj.setText('Project')
        self._collect_cfg()
        if hasattr(self, 'log'):
            self.log(f'[INFO] 进入层级：{self.cfg.level}，单位：{self.cfg.units}，模式：{self.cfg.get_mode(self.cfg.level)}。')
        # 通过 MainWindow 注入的回调进行页面跳转
        self.on_go_predict(self.cfg.level)

    # -- 模型微调 --
    def _go_finetune(self):
        # 若项目名称为空，则自动填充为 Project（并写入文本框）
        if not self.edt_proj.text().strip():
            self.edt_proj.setText('Project')
        self._collect_cfg()
        if not self._check_ft_inputs():
            return
        if hasattr(self, 'log'):
            self.log(f'[INFO] 准备微调：{self.cfg.level}。')
        self.open_ft_window(self.cfg.level, None)

    def _go_finetune_cnn(self):
        # 若项目名称为空，则自动填充为 Project（并写入文本框）
        if not self.edt_proj.text().strip():
            self.edt_proj.setText('Project')
        self._collect_cfg()
        if not self._check_ft_inputs():
            return
        if hasattr(self, 'log'):
            self.log('[INFO] 准备微调：微结构 · CNN。')
        self.open_ft_window('微结构', 'CNN')

    def _go_finetune_icnn(self):
        # 若项目名称为空，则自动填充为 Project（并写入文本框）
        if not self.edt_proj.text().strip():
            self.edt_proj.setText('Project')
        self._collect_cfg()
        if not self._check_ft_inputs():
            return
        if hasattr(self, 'log'):
            self.log('[INFO] 准备微调：微结构 · ICNN。')
        self.open_ft_window('微结构', 'ICNN')

# =============================
# 右侧：项目概览 / 日志
# =============================
class RightInspector(QWidget):
    """
    主界面右侧边栏，包含“项目概览”和“日志”两个选项卡。
    用于显示全局配置的只读状态和接收所有层级的日志信息。
    """
    def __init__(self, config: ProjectConfig):
        super().__init__()
        self.cfg = config
        v = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tab_overview = QWidget()
        self.tab_logs = QWidget()
        # 项目概览（只读）
        of = QFormLayout(self.tab_overview)
        self.lb_proj = QLabel(self.cfg.name)
        self.lb_level = QLabel(self.cfg.level)
        self.lb_units = QLabel(self.cfg.units)
        self.lb_mode = QLabel(self.cfg.get_mode(self.cfg.level))
        # 使用 ElidedLabel 来显示预训练模型和微调模型的简要路径，自动根据可用宽度截断，并在悬停时显示完整文本。
        self.lb_pt = ElidedLabel('未选择')
        self.lb_ft = ElidedLabel('未加载')
        # 调整显示属性：允许水平扩展并提供最小宽度，支持文本选择。
        for _lbl in (self.lb_pt, self.lb_ft):
            _lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            # 提供一个合理的最小宽度以避免初始宽度为零导致反复重绘。
            _lbl.setMinimumWidth(150)
            _lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        of.addRow('项目名称', self.lb_proj)
        of.addRow('当前层级', self.lb_level)
        of.addRow('单位制', self.lb_units)
        of.addRow('模型模式', self.lb_mode)
        of.addRow('预训练模型', self.lb_pt)
        of.addRow('微调模型', self.lb_ft)
        # 日志
        lg = QVBoxLayout(self.tab_logs)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.appendPlainText('[INFO] 就绪。')
        lg.addWidget(self.log)
        self.tabs.addTab(self.tab_overview, '项目概览')
        self.tabs.addTab(self.tab_logs, '日志')
        v.addWidget(self.tabs)

    def _build_pt_summary(self) -> str:
        lv = self.cfg.level
        if lv == '微结构':
            c = self.cfg.get_pt('微结构', 'CNN')
            i = self.cfg.get_pt('微结构', 'ICNN')
            parts = []
            if c:
                parts.append(f'CNN={c}')
            if i:
                parts.append(f'ICNN={i}')
            return '；'.join(parts) if parts else '未选择'
        else:
            val = self.cfg.get_pt(lv, 'default')
            return val if val else '未选择'

    def _build_ft_summary(self) -> str:
        lv = self.cfg.level
        if lv == '微结构':
            c = self.cfg.get_ft('微结构', 'CNN')
            i = self.cfg.get_ft('微结构', 'ICNN')
            parts = []
            if c:
                parts.append(f'CNN={c}')
            if i:
                parts.append(f'ICNN={i}')
            return '；'.join(parts) if parts else '未加载'
        else:
            val = self.cfg.get_ft(lv, 'default')
            return val if val else '未加载'

    def update_from_config(self):
        self.lb_proj.setText(self.cfg.name)
        self.lb_level.setText(self.cfg.level)
        self.lb_units.setText(self.cfg.units)
        self.lb_mode.setText(self.cfg.get_mode(self.cfg.level))
        # 为预训练模型和微调模型摘要设置完整文本。ElidedLabel 会根据宽度自动截断并更新提示。
        try:
            self.lb_pt.set_full_text(self._build_pt_summary())
        except Exception:
            # 回退到简单文本赋值以防万一
            self.lb_pt.setText(self._build_pt_summary())
        try:
            self.lb_ft.set_full_text(self._build_ft_summary())
        except Exception:
            self.lb_ft.setText(self._build_ft_summary())

    def write_log(self, text: str):
        self.log.appendPlainText(text)
        # 日志变化时切换到“日志”页
        self.tabs.setCurrentWidget(self.tab_logs)

# =============================
# 层级页：公共基类（处理预测启停与校验）
# =============================
class _BaseLevelPage(QWidget):
    """
    所有层级页面（微结构、层合板等）的抽象基类。

    提供了配置(cfg)、日志(log)的统一访问，
    以及预测启停、按钮状态管理、必填项校验的公共逻辑。
    """
    # === 注入：所有层级页面均可用的必填字段辅助函数 ===
    def _collect_missing(self, name_widget_pairs):
        missing = []
        for item in name_widget_pairs or []:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                name, w = (item[0], item[1])
            else:
                w = item
                try:
                    name = getattr(w, 'placeholderText', lambda: '')() or getattr(w, 'toolTip', lambda: '')() or getattr(w, 'objectName', '') or '未命名输入'
                except Exception:
                    name = '未命名输入'
            empty = False
            if isinstance(w, QLineEdit):
                empty = w.text().strip() == ''
            elif isinstance(w, QComboBox):
                empty = w.currentIndex() < 0 and (not w.currentText().strip())
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                empty = False  # 数值类通常有默认值
            else:
                # 尝试通用的 text()
                try:
                    t = w.text().strip()
                    empty = t == ''
                except Exception:
                    empty = False
            if empty:
                missing.append(str(name))
        return missing

    def _warn_missing(self, missing_names, title='缺少输入'):
        if not missing_names:
            return True
        try:
            msg = '以下必填项缺失：\\n' + '\\n'.join((f'- {n}' for n in missing_names))
            msg_warning(self, title, msg)
        except Exception:
            self.log('[WARN] ' + '缺少输入：' + ', '.join(missing_names))
        return False

    def __init__(self, config: ProjectConfig, write_log, level_name: str, on_cfg_changed):
        super().__init__()
        self.cfg = config
        self.log = write_log
        self.level_name = level_name
        self.on_cfg_changed = on_cfg_changed
        self.predict_running = False  # 预测状态

    def _can_predict(self) -> bool:
        if self.cfg.get_mode(self.level_name) == '微调' and (not self.cfg.get_ft(self.level_name, 'default')):
            self.log('[WARN] 当前为“微调”模式，请先导入本层级的微调模型文件再开始预测。')
            return False
        return True

    def _make_mode_text(self) -> str:
        return f'当前模型模式：{self.cfg.get_mode(self.level_name)}'

    def _update_predict_buttons(self, btn_imports: List[QPushButton], btn_start: QPushButton, btn_stop: QPushButton,
                                lbl_mode: QLabel, extra_disable_when_ft_missing=None):
        for b in btn_imports:
            b.setEnabled(self.cfg.get_mode(self.level_name) == '微调' and (not self.predict_running))
        ft_ok = True
        mode = self.cfg.get_mode(self.level_name)
        if mode == '未选择':
            ft_ok = False
        if mode == '微调':
            if self.level_name == '微结构':
                try:
                    ft_ok = bool(self.cfg.get_ft('微结构', 'CNN')) and bool(self.cfg.get_ft('微结构', 'ICNN'))
                except Exception:
                    ft_ok = False
            else:
                try:
                    ft_ok = bool(self.cfg.get_ft(self.level_name, 'default'))
                except Exception:
                    ft_ok = False
        # 形式完整性和高光
        form_ok = True
        try:
            _req = getattr(self, '_required_fields', [])
            if _req:
                empty_widgets = []
                for _w in _req:
                    w = _w[1] if isinstance(_w, (tuple, list)) and len(_w) >= 2 else _w
                    if isinstance(w, QLineEdit):
                        if w.text().strip() == '':
                            empty_widgets.append(w)
                # 红色高亮表示空的
                for _w in _req:
                    w2 = _w[1] if isinstance(_w, (tuple, list)) and len(_w) >= 2 else _w
                    if isinstance(w2, QLineEdit):
                        if w2 in empty_widgets:
                            w2.setStyleSheet('border: 1px solid #d9534f;')
                        else:
                            w2.setStyleSheet('')  # 移除样式表（恢复默认）
                form_ok = len(empty_widgets) == 0
                if not form_ok:
                    empty_widgets[0].setFocus()
        except Exception:
            form_ok = True

        can_start = not self.predict_running and ft_ok and form_ok
        btn_start.setEnabled(can_start)
        # 停止按钮仅在有任务（self.predict_running）运行时启用
        btn_stop.setEnabled(self.predict_running)

        if extra_disable_when_ft_missing:
            for b in extra_disable_when_ft_missing:
                if b is not None:
                    b.setEnabled(can_start)
        if self.level_name == '微结构':
            for b in self.findChildren(QPushButton):
                try:
                    t = b.text()
                except Exception:
                    t = ''
                _norm = (t or '').replace(' ', '').strip()
                if '停止' in _norm:
                    continue
                if '预测' in _norm and ('强度带' in _norm or '失效包络' in _norm):
                    b.setEnabled(can_start)
        # 显示当前模型模式为普通文本，不进行富文本换行处理。
        lbl_mode.setTextFormat(Qt.TextFormat.PlainText)
        lbl_mode.setWordWrap(False)
        try:
            lbl_mode.setText(self._make_mode_text())
        except Exception:
            # 回退以避免潜在异常
            lbl_mode.setText(str(self._make_mode_text()))

    def load_from_config(self):
        """
        [虚方法] 当配置更改或页面加载时，更新此页面上的控件状态。
        子类应重写此方法。
        """
        # 示例：
        # self.lbl_mode.setText(self._make_mode_text())
        # self._update_predict_buttons(...)
        pass

    def _on_level_done(self, payload):
        """
        当特定层级的预测线程完成时调用默认处理程序。如果摘要标签可用，则使用有效负载（或通用完成消息）更新它，并记录完成日志条目。
        """
        if hasattr(self, 'res_summary'):
            if isinstance(payload, str):
                self.res_summary.setText(payload)
            else:
                # 如果负载不是字符串，则回退到通用消息。
                self.res_summary.setText('预测完成。')
        self.log('[INFO] 预测完成。')

    def _on_level_error(self, tb_text: str):
        """
        当层级预测线程引发异常时调用默认错误处理程序。记录追溯并更新摘要标签以指示失败。
        """
        self.log(f'[ERROR] 预测线程异常：\n{tb_text}')
        if hasattr(self, 'res_summary'):
            self.res_summary.setText('预测失败，请检查日志。')

    def _finish_level_predict(self):
        """
        当层级预测完成或终止时，重置状态标志并刷新按钮状态。
        """
        self.predict_running = False
        # 收集导入按钮
        import_btns: List[QPushButton] = []
        for attr in ('btn_import_ft_cnn', 'btn_import_ft_icnn', 'btn_import_ft'):
            b = getattr(self, attr, None)
            if b:
                import_btns.append(b)
        extra: List[QPushButton] = []
        # 额外的控件（例如强度带/包络按钮）
        for attr in ('btn_predict_band', 'btn_predict_envelope'):
            b = getattr(self, attr, None)
            if b:
                extra.append(b)
        btn_predict = getattr(self, 'btn_predict', None)
        btn_stop = getattr(self, 'btn_stop', None)
        lbl_mode = getattr(self, 'lbl_mode', None)
        if btn_predict and btn_stop and lbl_mode:
            self._update_predict_buttons(import_btns, btn_predict, btn_stop, lbl_mode, extra_disable_when_ft_missing=extra)

    def _stop_predict(self):
        """
        请求停止当前页面正在运行的预测任务。
        子类（如 MicrostructurePage）应重写此方法以停止其特定的工作线程。
        """
        # 尝试停止正在进行的强度带预测（如果存在）
        if hasattr(self, '_band_worker') and self._band_worker is not None:
            self._band_worker.request_stop()
            self.log('[WARN] 已请求停止强度带预测（线程将在当前样本结束后退出）')
        # 重置运行标志
        if getattr(self, '_band_running', False) and self.predict_running:
            self.predict_running = False
            self._band_running = False
        # 隐藏进度条（如果适用）
        if hasattr(self, 'band_progress'):
            self.band_progress.setVisible(False)
        # 刷新导入/开始/停止按钮
        import_btns: List[QPushButton] = []
        for attr in ('btn_import_ft_cnn', 'btn_import_ft_icnn', 'btn_import_ft'):
            b = getattr(self, attr, None)
            if b:
                import_btns.append(b)
        btn_predict = getattr(self, 'btn_predict', None)
        btn_stop = getattr(self, 'btn_stop', None)
        lbl_mode = getattr(self, 'lbl_mode', None)
        if btn_predict and btn_stop and lbl_mode:
            self._update_predict_buttons(import_btns, btn_predict, btn_stop, lbl_mode)

# =============================
# 层级页：微结构
# =============================
class MicrostructurePage(_BaseLevelPage):
    """
    负责单个RVE图像的失效包络预测和批量图像的强度带预测。
    管理CNN和ICNN模型的加载、微调导入和预测线程。
    """
    def _collect_missing(self, name_widget_pairs):
        missing = []
        for item in name_widget_pairs or []:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                name, w = (item[0], item[1])
            else:
                w = item
                try:
                    name = getattr(w, 'placeholderText', lambda: '')() or getattr(w, 'toolTip', lambda: '')() or getattr(w, 'objectName', '') or '未命名输入'
                except Exception:
                    name = '未命名输入'
            if isinstance(w, QLineEdit) and w.text().strip() == '':
                missing.append(str(name))
        return missing

    def _warn_missing(self, items, title='参数缺失'):
        if not items:
            return
        msg = '以下参数未填写：\n• ' + '\n• '.join(items)
        msg_warning(self, title, msg)

    def __init__(self, config: ProjectConfig, write_log, on_cfg_changed):
        super().__init__(config, write_log, '微结构', on_cfg_changed)
        # --- 项目状态缓存 ---
        self.last_env_curve = None  # {'x': [...], 'y': [...]}
        self.last_band_data = None  # {'x_common': [...], 'y_min': [...], 'y_max': [...], 'xs_list': [[...]], 'ys_list': [[...]]}
        # --- 机器学习实例变量 ---
        self.model_cnn = None
        self.model_icnn = None
        self.current_rve_path = ''  # 存储所选 RVE 图像的路径
        layout = QHBoxLayout(self)
        left = QGroupBox('单个微结构失效包络预测（模型-数据驱动算法）')
        form = QFormLayout(left)
        # -- RVE 图像（行1：Vf 下拉 + 内置“随机加载” + 外部选择 + 路径显示） --
        self.cmb_single_vf = QComboBox()
        self.cmb_single_vf.addItems(['0.30', '0.40', '0.50', '0.60'])
        self.btn_load_internal = QPushButton('加载图像')
        self.btn_pick_img_ext = QPushButton('选择外部图像')
        # 使用 ElidedLabel 显示选中的 RVE 图像路径，自动根据宽度截断
        self.lbl_img = ElidedLabel('未选择')
        self.lbl_img.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.lbl_img.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_img.set_full_text('未选择')
        self.btn_load_internal.clicked.connect(self._load_random_internal_image)
        self.btn_pick_img_ext.clicked.connect(self._choose_infer_image)
        row_img = QWidget()
        row_img_l = QHBoxLayout(row_img)
        row_img_l.setContentsMargins(0, 0, 0, 0)  # 去除工具栏外边距，使布局更紧凑
        row_img_l.addWidget(QLabel('Vf'))
        row_img_l.addWidget(self.cmb_single_vf)
        row_img_l.addWidget(self.btn_load_internal)
        row_img_l.addWidget(self.btn_pick_img_ext)
        row_img_l.addWidget(self.lbl_img, 1)
        form.addRow('RVE 图像', row_img)
        # -- 可视化预览（行2：独立一行） --
        self.rve_preview = QLabel()
        self.rve_preview.setFixedSize(128, 128)
        self.rve_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rve_preview.setStyleSheet('background:#fafafa; border:2px solid #94a3b8; border-radius:4px;')
        form.addRow('可视化预览', self.rve_preview)
        # 纤维参数
        grp_fiber = QGroupBox('纤维')
        fform = QFormLayout(grp_fiber)
        self.edt_fiber_E = QLineEdit('230e9')
        self.edt_fiber_nu = QLineEdit('0.3')
        fform.addRow('弹性模量 E_f (Pa)', self.edt_fiber_E)
        fform.addRow('泊松比 ν_f', self.edt_fiber_nu)
        form.addRow(grp_fiber)
        # 基体参数
        grp_matrix = QGroupBox('基体')
        mform = QFormLayout(grp_matrix)
        self.edt_matrix_E = QLineEdit('3.5e9')
        self.edt_matrix_nu = QLineEdit('0.3')
        self.edt_matrix_Xc = QLineEdit('150e6')
        self.edt_matrix_beta = QLineEdit('20')
        self.edt_matrix_upl = QLineEdit('1e-4')
        mform.addRow('弹性模量 E_m (Pa)', self.edt_matrix_E)
        mform.addRow('泊松比 ν_m', self.edt_matrix_nu)
        mform.addRow('压缩强度 X_c (Pa)', self.edt_matrix_Xc)
        mform.addRow('摩擦角 β (°)', self.edt_matrix_beta)
        mform.addRow('失效位移 u^pl', self.edt_matrix_upl)
        form.addRow(grp_matrix)
        # 界面参数
        grp_iface = QGroupBox('界面')
        iform = QFormLayout(grp_iface)
        self.edt_K = QLineEdit('1e8')
        self.edt_tn0 = QLineEdit('6.0e7')
        self.edt_ts0 = QLineEdit('9.0e7')
        self.edt_Gn = QLineEdit('0.005')
        self.edt_Gs = QLineEdit('0.025')
        iform.addRow('刚度 K', self.edt_K)
        iform.addRow('法向强度 t_n^0 (Pa)', self.edt_tn0)
        iform.addRow('切向强度 t_s^0 (Pa)', self.edt_ts0)
        iform.addRow('法向断裂能 G_n^c', self.edt_Gn)
        iform.addRow('切向断裂能 G_s^c', self.edt_Gs)
        form.addRow(grp_iface)
        # 模式 & 操作
        self.lbl_mode = QLabel()
        # 导入微调模型（CNN/ICNN）
        self.btn_import_ft_cnn = QPushButton('导入CNN微调模型')
        self.btn_import_ft_icnn = QPushButton('导入ICNN微调模型')
        self.btn_predict = QPushButton('预测失效包络')
        self.btn_band = QPushButton('预测强度带')
        self.btn_stop = QPushButton('停止预测')
        self.btn_import_ft_cnn.clicked.connect(lambda: self._import_finetuned_model('CNN'))
        self.btn_import_ft_icnn.clicked.connect(lambda: self._import_finetuned_model('ICNN'))
        self.btn_predict.clicked.connect(self._predict_envelope)
        self.btn_stop.clicked.connect(self._stop_predict)
        self.btn_band.clicked.connect(self._predict_band)
        form.addRow(self.lbl_mode)
        row_ops = QWidget()
        row_ops_l = QHBoxLayout(row_ops)
        row_ops_l.setContentsMargins(0, 0, 0, 0)  # 去除工具栏外边距，使布局更紧凑
        row_ops_l.addWidget(self.btn_import_ft_cnn)
        row_ops_l.addWidget(self.btn_import_ft_icnn)
        row_ops_l.addWidget(self.btn_predict)
        row_ops_l.addWidget(self.btn_band)
        row_ops_l.addWidget(self.btn_stop)
        row_ops_l.addStretch(1)
        form.addRow(row_ops)
        # 必填字段：全部填写后才启用预测
        self._required_fields = [self.edt_fiber_E, self.edt_fiber_nu, self.edt_matrix_E, self.edt_matrix_nu,
                                 self.edt_matrix_Xc, self.edt_matrix_beta, self.edt_matrix_upl, self.edt_K,
                                 self.edt_tn0, self.edt_ts0, self.edt_Gn, self.edt_Gs
                                 ]
        def __refresh_buttons_required__():
            self._update_predict_buttons([self.btn_import_ft_cnn, self.btn_import_ft_icnn],
                                         self.btn_predict, self.btn_stop, self.lbl_mode)
        for __w in self._required_fields:
            __target = __w[1] if isinstance(__w, (tuple, list)) and len(__w) >= 2 else __w
            __target.textChanged.connect(lambda *_: __refresh_buttons_required__())
        __refresh_buttons_required__()
        # 强度带设置
        grp_band = QGroupBox('失效带预测')
        bform = QFormLayout(grp_band)
        bform.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        # Vf：限定在 30%/40%/50%/60% -- 用 0.30/0.40 的数值文本，便于后续 float()
        self.cmb_vf = QComboBox()
        self.cmb_vf.addItems(['0.30', '0.40', '0.50', '0.60'])
        # N：100~1000 可选
        self.spn_n = QSpinBox()
        self.spn_n.setRange(100, 1000)
        self.spn_n.setSingleStep(50)
        self.spn_n.setValue(200)
        bform.addRow('纤维体积分数 Vf', self.cmb_vf)
        bform.addRow('微结构数量 N', self.spn_n)
        # 外部数据：用户自行提供一个微结构文件夹（优先用于强度带抽样）
        self.btn_user_micro = QPushButton('文件导入')
        self.btn_clear_user_micro = QPushButton('清除外部数据')
        # 使用 ElidedLabel 显示用户微结构目录路径，自动根据宽度截断
        self.lbl_user_micro = ElidedLabel('未选择')
        self.lbl_user_micro.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.lbl_user_micro.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_user_micro.set_full_text('未选择')
        self.btn_user_micro.clicked.connect(self._choose_user_micro_dir)
        self.btn_clear_user_micro.clicked.connect(self._clear_user_micro_dir)
        row_user = QWidget()
        row_user_l = QHBoxLayout(row_user)
        row_user_l.setContentsMargins(0, 0, 0, 0)  # 去除工具栏外边距，使布局更紧凑
        row_user_l.addWidget(self.btn_user_micro)
        row_user_l.addWidget(self.btn_clear_user_micro)
        row_user_l.addWidget(self.lbl_user_micro, 1)
        # 明确拉伸权重：标签多占空间
        row_user_l.setStretch(0, 0)
        row_user_l.setStretch(1, 0)
        row_user_l.setStretch(2, 1)
        bform.addRow('选择外部文件', row_user)
        form.addRow(grp_band)
        self.user_micro_dir = ''  # 外部微结构目录（可选）
        self.selected_micro_paths = []  # 本次随机抽样到的微结构路径
        # 初始时清除外部数据按钮禁用，仅在成功选择外部目录后启用
        self.btn_clear_user_micro.setEnabled(False)
        # 右侧：结果与可视化
        right = QGroupBox('结果与可视化')
        rv = QVBoxLayout(right)
        self.fig_env = Figure(figsize=(5, 2.6), tight_layout=True)
        self.ax_env = self.fig_env.add_subplot(111)
        self.canvas_env = FigureCanvas(self.fig_env)
        protect_canvas(self.canvas_env)
        self.toolbar_env = AutoNavToolbar(self.canvas_env, self)
        self._redraw_env()  # 初始化绘图
        self.fig_band = Figure(figsize=(5, 2.6), tight_layout=True)
        self.ax_band = self.fig_band.add_subplot(111)
        self.canvas_band = FigureCanvas(self.fig_band)
        protect_canvas(self.canvas_band)
        self.toolbar_band = AutoNavToolbar(self.canvas_band, self)
        self._redraw_band()  # 初始化绘图
        # 附着浮动光标坐标标签（无工具栏文本）
        attach_hover_coords(self.ax_env, self.canvas_env)
        attach_hover_coords(self.ax_band, self.canvas_band)
        rv.addWidget(self.toolbar_env)
        rv.addWidget(self.canvas_env)
        rv.addWidget(self.toolbar_band)
        rv.addWidget(self.canvas_band)
        self.res_summary = QLabel('尚无结果')
        rv.addWidget(self.res_summary)
        # 强度带专用进度条（独立）
        self.band_progress = QProgressBar()
        self.band_progress.setRange(0, 100)
        self.band_progress.setValue(0)
        self.band_progress.setVisible(False)
        rv.addWidget(self.band_progress)
        rv.addStretch(1)
        layout.addWidget(left, 1)
        layout.addWidget(right, 1)
        self.load_from_config()

    def _redraw_env(self, x_data=None, y_data=None):
        self.ax_env.clear()
        self.ax_env.set_title('失效包络')
        self.ax_env.set_xlabel('$\\sigma_{22}$ (MPa)')
        self.ax_env.set_ylabel('$\\tau_{23}$ (MPa)')
        self.ax_env.grid(True, alpha=0.4)
        if x_data is not None and y_data is not None:
            self.ax_env.plot(x_data, y_data, linestyle='-', linewidth=2, color='r', label='失效包络线')
            self.ax_env.legend(loc='best')
        self.canvas_env.draw_idle()

    def _redraw_band(self, x_data=None, all_y_data=None):
        """
        重绘强度带图表。
        - 如果提供了 x_data 和 all_y_data，则计算并绘制强度带。
        - 否则，清空图表并显示“尚无结果”。
        """
        self.ax_band.clear()

        # 检查是否有数据传入
        if x_data is not None and all_y_data is not None and (len(all_y_data) > 0):
            self.ax_band.set_title('强度带')
            all_y_data_np = np.array(all_y_data)
            y_min = np.min(all_y_data_np, axis=0)
            y_max = np.max(all_y_data_np, axis=0)
            # 绘制所有点的散点图
            for y_curve in all_y_data:
                self.ax_band.scatter(x_data, y_curve, color='gray', alpha=0.1, s=5)
            # 绘制边界
            self.ax_band.plot(x_data, y_min, color='blue', linewidth=1.5, label='内边界')
            self.ax_band.plot(x_data, y_max, color='red', linewidth=1.5, label='外边界')
            # 填充强度带
            self.ax_band.fill_between(x_data, y_min, y_max, color='skyblue', alpha=0.4, label='失效强度带')
            self.ax_band.legend(loc='best')
        else:
            # 清空或无数据时
            self.ax_band.set_title('强度带 (尚无结果)')

        # 统一设置坐标轴和网格
        self.ax_band.set_xlabel('$\\sigma_{22}$ (MPa)')
        self.ax_band.set_ylabel('$\\tau_{23}$ (MPa)')
        self.ax_band.grid(True, alpha=0.4)
        self.canvas_band.draw_idle()

    def _redraw_band_live(self):
        """根据已累积的部分曲线进行即时绘制（包络为当前部分数据的最小/最大）。"""
        xs_list = getattr(self, '_live_xs', [])
        ys_list = getattr(self, '_live_ys', [])
        if not xs_list or not ys_list:
            return
        # 构造公共x并插值
        x_left = float(np.min([np.min(xi) for xi in xs_list]))
        x_right = float(np.max([np.max(xi) for xi in xs_list]))
        x_common = np.linspace(x_left, x_right, 160).astype(np.float32)
        Y = []
        for xi, yi in zip(xs_list, ys_list):
            if xi[0] > xi[-1]:
                xi, yi = (xi[::-1], yi[::-1])
            Y.append(np.interp(x_common, xi, yi))
        Y = np.stack(Y, axis=0)
        y_min = np.min(Y, axis=0)
        y_max = np.max(Y, axis=0)
        y_min[0] = y_max[0] = 0.0
        y_min[-1] = y_max[-1] = 0.0
        # 绘制
        self.ax_band.clear()
        self.ax_band.set_title('强度带')
        self.ax_band.set_xlabel('$\\sigma_{22}$ (MPa)')
        self.ax_band.set_ylabel('$\\tau_{23}$ (MPa)')
        self.ax_band.grid(True, alpha=0.4)
        # 散点：轻量展示当前样本
        for xi, yi in zip(xs_list, ys_list):
            self.ax_band.scatter(xi, yi, alpha=0.08, s=5)
        self.ax_band.plot(x_common, y_min, color='blue', linewidth=1.0, label='内边界*')
        self.ax_band.plot(x_common, y_max, color='red', linewidth=1.0, label='外边界*')
        self.ax_band.fill_between(x_common, y_min, y_max, color='skyblue', alpha=0.25, label='强度带')
        self.ax_band.legend(loc='best')
        self.canvas_band.draw_idle()

    def _redraw_band_multi(self, xs_list, ys_list, x_common, y_min, y_max):
        """
        绘制最终的强度带结果。

        Args:
            xs_list (list): 原始x坐标数组列表。
            ys_list (list): 原始y坐标数组列表。
            x_common (np.ndarray): 统一插值后的x轴。
            y_min (np.ndarray): 下包络线。
            y_max (np.ndarray): 上包络线。
        """
        self.ax_band.clear()
        self.ax_band.set_title('强度带')
        self.ax_band.set_xlabel('$\\sigma_{22}$ (MPa)')
        self.ax_band.set_ylabel('$\\tau_{23}$ (MPa)')
        self.ax_band.grid(True, alpha=0.4)
        # 散点：能看到不同样本的端点
        for xi, yi in zip(xs_list, ys_list):
            self.ax_band.scatter(xi, yi, alpha=0.08, s=5)
        # 包络
        self.ax_band.plot(x_common, y_min, color='blue', linewidth=1.5, label='内边界')
        self.ax_band.plot(x_common, y_max, color='red', linewidth=1.5, label='外边界')
        self.ax_band.fill_between(x_common, y_min, y_max, color='skyblue', alpha=0.4, label='失效强度带')
        self.ax_band.legend(loc='best')
        self.canvas_band.draw_idle()

    # 微调模式对微结构：需要 CNN 与 ICNN 同时具备
    def _can_predict(self) -> bool:
        if self.cfg.get_mode(self.level_name) == '微调':
            has_cnn = bool(self.cfg.get_ft('微结构', 'CNN'))
            has_icnn = bool(self.cfg.get_ft('微结构', 'ICNN'))
            if not (has_cnn and has_icnn):
                self.log('[WARN] 微调模式下需同时导入 CNN 与 ICNN 微调模型后才能预测。')
                return False
        return True

    def load_from_config(self):
        self._update_predict_buttons(
            [self.btn_import_ft_cnn, self.btn_import_ft_icnn],
            self.btn_predict, self.btn_stop, self.lbl_mode
        )

    def _choose_infer_image(self):
        f, _ = get_open_file(self, '选择RVE图像', 'Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)')
        if f:
            norm = os.path.normpath(f)
            self.current_rve_path = norm
            try:
                self.lbl_img.set_full_text(norm)
            except Exception:
                self.lbl_img.set_full_text(norm)
            self._update_preview(norm)

    def _resolve_internal_vf_dir(self, vf: float) -> str:
        if not os.path.isdir(INTERNAL_MS_DB):
            return ''
        candidates = [f'{int(vf * 100)}%', f'{int(vf * 100)}', f'{vf:.2f}']
        for name in os.listdir(INTERNAL_MS_DB):
            p = os.path.join(INTERNAL_MS_DB, name)
            if not os.path.isdir(p):
                continue
            norm = name.strip().replace('％', '%')
            if norm in candidates or f'{int(vf * 100)}' in norm:
                return p
        return ''

    def _load_random_internal_image(self):
        vf = float(self.cmb_single_vf.currentText())
        vf_dir = self._resolve_internal_vf_dir(vf)
        if not vf_dir:
            msg_warning(self, '未找到数据库', '未找到与所选 Vf 对应的内置数据库子文件夹。')
            return
        imgs = [os.path.join(vf_dir, f) for f in os.listdir(vf_dir) if f.lower().endswith(IMG_EXTS)]
        if not imgs:
            msg_warning(self, '无图像', '内置数据库该子文件夹中未发现图像文件。')
            return
        f = random.choice(imgs)
        norm = os.path.normpath(f)
        self.current_rve_path = norm
        try:
            self.lbl_img.set_full_text(norm)
        except Exception:
            self.lbl_img.set_full_text(norm)
        self._update_preview(norm)
        self.log(f'[INFO] 已从内置数据库 {os.path.basename(vf_dir)} 随机加载图像：{f}')

    def _update_preview(self, path: str):
        try:
            pm = QPixmap(path)
            if pm.isNull():
                img = safe_imread(path, as_color=True)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w, ch = img.shape
                    qimg = QImage(img.data, w, h, ch * w, QImage.Format.Format_RGB888)
                    pm = QPixmap.fromImage(qimg)
            if not pm.isNull():
                pm = pm.scaled(128, 128, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                self.rve_preview.setPixmap(pm)
                self.rve_preview.setToolTip(_TextUtil.wrap_breakable(path))
            else:
                self.rve_preview.clear()
        except Exception:
            self.rve_preview.clear()

    def _choose_user_micro_dir(self):
        d = get_existing_directory(self, '选择需要导入的微结构文件夹')
        # 如果选择了目录才更新状态；否则保持现有设置
        if d:
            norm = os.path.normpath(d)
            self.user_micro_dir = norm
            # 更新显示
            try:
                self.lbl_user_micro.set_full_text(norm)
            except Exception:
                self.lbl_user_micro.set_full_text(norm)
            # 外部数据选中后，忽略 Vf 与 N，并禁用其控件
            self.cmb_vf.setEnabled(False)
            self.spn_n.setEnabled(False)
            # 启用清除按钮
            self.btn_clear_user_micro.setEnabled(True)
            self.log('[INFO] 已选择外部数据目录，直接对该目录内所有图像进行强度带预测。')

    def _clear_user_micro_dir(self):
        self.user_micro_dir = ''
        try:
            self.lbl_user_micro.set_full_text('未选择')
        except Exception:
            self.lbl_user_micro.set_full_text('未选择')
        self.log('[INFO] 已清除外部数据，将在预测强度带时优先尝试使用内置数据库。')
        # 恢复 Vf 与 N 控件可用
        self.cmb_vf.setEnabled(True)
        self.spn_n.setEnabled(True)
        # 禁用清除按钮，防止再次点击
        self.btn_clear_user_micro.setEnabled(False)

    def _import_finetuned_model(self, sub: str):
        f, _ = get_open_file(self, f'导入{sub}微调模型', 'Model (*.h5 *.pt *.pth)')
        if f:
            self.cfg.set_ft(self.level_name, f, sub=sub)
            self.log(f'[INFO] 微结构页：已导入{sub}微调模型 -> {f}')
            self._update_predict_buttons([self.btn_import_ft_cnn, self.btn_import_ft_icnn],
                                         self.btn_predict, self.btn_stop, self.lbl_mode)
            self.on_cfg_changed()  # 更新概览

    def _run_single_prediction(self, image_path):
        IMG_W, IMG_H, BATCH_SIZE = (224, 224, 4)
        # 1. 加载和预处理图像（稳健读取 + 统一到 3 通道）
        image_initial = safe_imread(image_path, as_color=True)
        if image_initial is None:
            self.log(f'[ERROR] 无法读取图像文件: {image_path}')
            return (None, None)
        image_dsize = cv2.resize(image_initial, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
        # 转灰度再阈值，避免对三通道图像直接 threshold 的不兼容
        gray = cv2.cvtColor(image_dsize, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
        dst = bw.astype(np.float32) / 255.0 - 0.5  # [-0.5, 0.5]
        # 扩展到三通道
        dst = np.stack([dst, dst, dst], axis=-1)
        image = dst[None, ...].astype(np.float32)  # (1, H, W, 3)
        # 2. 根据模式（预训练或微调）获取模型路径
        if self.cfg.get_mode(self.level_name) == '预训练':
            cnn_model_path = self.cfg.get_pt('微结构', 'CNN')
            icnn_model_path = self.cfg.get_pt('微结构', 'ICNN')
        else:  # "微调"
            cnn_model_path = self.cfg.get_ft('微结构', 'CNN')
            icnn_model_path = self.cfg.get_ft('微结构', 'ICNN')
        # 3. 加载 CNN 模型并预测
        if not self.model_cnn or not hasattr(self.model_cnn, '_path') or self.model_cnn._path != cnn_model_path:
            self.log(f'[INFO] 正在加载 CNN 模型: {cnn_model_path}')
            if not os.path.exists(cnn_model_path):
                self.log(f'[ERROR] CNN 模型文件不存在: {cnn_model_path}')
                return (None, None)
            self.model_cnn = _ModelCache.get(cnn_model_path)
            self.model_cnn._path = cnn_model_path  # 存储路径以避免重新加载
        predictions = self.model_cnn.predict(image)
        # 4. 定义输入 x 数组
        x = np.zeros(100, dtype=np.float32)
        x[0] = predictions[0, 0]
        x[99] = predictions[0, 2]
        increment = (x[0] - x[99]) / (len(x) - 1)
        for i in range(len(x)):
            x[i] = x[0] - i * increment
        # 5. 加载 ICNN 模型并预测
        if not self.model_icnn or not hasattr(self.model_icnn, '_path') or self.model_icnn._path != icnn_model_path:
            self.log(f'[INFO] 正在加载 ICNN 模型: {icnn_model_path}')
            if not os.path.exists(icnn_model_path):
                self.log(f'[ERROR] ICNN 模型文件不存在: {icnn_model_path}')
                return (None, None)
            custom_objects = {'rmse_metric': rmse_metric}
            self.model_icnn = _ModelCache.get(icnn_model_path, custom_objects=custom_objects)
            self.model_icnn._path = icnn_model_path  # 存储路径以避免重新加载
        test_generator_single = CustomDataGenerator(image, x, batch_size=BATCH_SIZE, shuffle=False, is_training=False, input_names=getattr(self.model_icnn, 'input_names', None))
        predictions_icnn = self.model_icnn.predict(test_generator_single, verbose=0)

        # 6. 计算 Tsai-Wu 包络
        def Tsai_Wu(x_in: np.ndarray):
            # 确保预测值不为零，以避免除法错误
            p_0 = predictions[0, 0] if predictions[0, 0] != 0 else 1e-09
            p_1 = predictions[0, 1] if predictions[0, 1] != 0 else 1e-09
            p_2 = predictions[0, 2] if predictions[0, 2] != 0 else 1e-09
            term1 = (1.0 / p_0 + 1.0 / p_2) * x_in
            term2 = -1.0 / (p_0 * p_2) * x_in ** 2
            term3 = 1.0 / p_1 ** 2
            inside_sqrt = (1.0 - term1 - term2) / term3
            inside_sqrt = np.where(inside_sqrt < 0.0, 0.0, inside_sqrt)
            return np.sqrt(inside_sqrt).astype(np.float32)
        y_mod = Tsai_Wu(x)
        # predictions_icnn 输出形状通常为 (N,1) 或 (N,)；按 Fortran 顺序展平与原实现保持一致
        y_corr = predictions_icnn.ravel('F').astype(np.float32)
        # 对齐长度（理论上两者都应为 100）
        m = min(len(y_mod), len(y_corr))
        y = y_mod[:m] - y_corr[:m]
        # 首尾置零并截断为非负
        if len(y) >= 1:
            y[0] = 0.0
            y[-1] = 0.0
        y = np.where(y < 0.0, 0.0, y)
        return (x[:m], y)

    # ---------- 预测与绘图 ----------
    def _finish_env_predict(self):
        """失效包络预测线程结束后的清理工作。"""
        # 检查是否是用户停止的 (即 predict_running 仍为 True)
        if self.predict_running and not self._band_running:
            self.log('[INFO] 失效包络预测已终止。')
        # 线程结束后的公共收尾
        self._band_running = False
        self.predict_running = False
        self._update_predict_buttons([self.btn_import_ft_cnn, self.btn_import_ft_icnn],
                                     self.btn_predict, self.btn_stop, self.lbl_mode)

    def _on_env_done(self, x_curve, y_curve):
        self.predict_running = False
        try:
            if x_curve is not None and y_curve is not None:
                x_curve = np.array(x_curve, dtype=np.float32)
                y_curve = np.array(y_curve, dtype=np.float32)
                self._redraw_env(x_curve, y_curve)
                self.res_summary.setText('失效包络预测完成。')
                try:
                    self.last_env_curve = {'x': x_curve.tolist(), 'y': y_curve.tolist()}
                except Exception:
                    self.last_env_curve = None
                self.log('[INFO] 失效包络预测成功。')
            else:
                self._redraw_env()
                self.res_summary.setText('预测失败，请检查日志。')
                self.log('[ERROR] 失效包络预测失败。')
        finally:
            self._finish_env_predict()

    def _on_env_error(self, tb_text: str):
        self.predict_running = False
        self.log(f'[ERROR] 预测线程异常：\n{tb_text}')
        self.res_summary.setText('预测失败，请检查日志。')
        self._redraw_env()
        self._finish_env_predict()

    def _predict_envelope(self):
        """启动单个失效包络的预测（使用后台线程）。"""
        # 使用线程避免阻塞 UI
        self._band_running = False
        if self.predict_running:
            return
        if not self._can_predict():
            self._update_predict_buttons([self.btn_import_ft_cnn, self.btn_import_ft_icnn],
                                         self.btn_predict, self.btn_stop, self.lbl_mode)
            return
        if not self.current_rve_path:
            msg_warning(self, '参数缺失', '请先选择待预测的 RVE 图像。')
            return
        # 可选的最小参数校验
        missing = self._collect_missing([('纤维 弹性模量 E_f', self.edt_fiber_E), ('基体 弹性模量 E_m', self.edt_matrix_E)])
        if missing:
            self._warn_missing(missing)
            return
        # 解析模型路径（预训练/微调）
        mode = self.cfg.get_mode('微结构')
        if mode == '微调':
            cnn_path = self.cfg.get_ft('微结构', 'CNN') or ''
            icnn_path = self.cfg.get_ft('微结构', 'ICNN') or ''
        else:
            cnn_path = self.cfg.get_pt('微结构', 'CNN') or ''
            icnn_path = self.cfg.get_pt('微结构', 'ICNN') or ''
        self.predict_running = True
        self._update_predict_buttons([self.btn_import_ft_cnn, self.btn_import_ft_icnn],
                                     self.btn_predict, self.btn_stop, self.lbl_mode)
        self.res_summary.setText('正在预测失效包络…')
        self.log('[INFO] 启动微结构失效包络预测')
        # 启动线程
        try:
            if hasattr(self, '_env_worker') and self._env_worker is not None:
                self._env_worker.terminate()
            self._env_worker = EnvelopePredictWorker(self.current_rve_path, cnn_path, icnn_path, parent=self)
            self._env_worker.done.connect(self._on_env_done)
            self._env_worker.error.connect(self._on_env_error)
            self._env_worker.log_signal.connect(self.log)
            self._env_worker.finished.connect(lambda: self._finish_env_predict())
            self._env_worker.start()
        except Exception as e:
            self.log(f'[ERROR] 无法启动预测线程：{e}')
            self.predict_running = False
            self._update_predict_buttons([self.btn_import_ft_cnn, self.btn_import_ft_icnn],
                                         self.btn_predict, self.btn_stop, self.lbl_mode)

    def _predict_band(self):
        """启动强度带的批量预测（使用后台线程）。"""
        if self.predict_running:
            return
        # 外部数据模式：如果选择了外部目录，则忽略 Vf 与 N，直接对该目录内全部图像进行预测
        ext_mode = bool(self.user_micro_dir and os.path.isdir(self.user_micro_dir))
        if ext_mode:
            msg = f'检测到已选择外部数据目录：{self.user_micro_dir}。将直接对该目录内的所有微结构图像进行强度带预测，是否继续？'
        else:
            msg = '将从内置数据库中按当前“纤维体积分数 Vf”和“微结构数量 N”进行不重复随机抽样并预测强度带。是否继续？'
        reply = msg_question(self, '确认预测', msg, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.No:
            return
        # 从 UI 读取（仅在内置数据库模式需要）
        if not ext_mode:
            vf = float(self.cmb_vf.currentText())
            n = int(self.spn_n.value())
        self.predict_running = True
        self._update_predict_buttons([self.btn_import_ft_cnn, self.btn_import_ft_icnn],
                                     self.btn_predict, self.btn_stop, self.lbl_mode)
        self.log('[INFO] 启动微结构预测（失效强度带）…')
        QApplication.processEvents()
        # 标记为强度带运行中，启用“停止预测”按钮
        self._band_running = True
        self._update_predict_buttons([self.btn_import_ft_cnn, self.btn_import_ft_icnn],
                                     self.btn_predict, self.btn_stop, self.lbl_mode)
        # 数据源选择
        self.selected_micro_paths = []
        used_source = ''
        if ext_mode:
            imgs = gather_images(self.user_micro_dir)
            # 去重并稳定排序（按文件名）
            imgs = sorted(set(imgs), key=lambda p: os.path.basename(p).lower())
            if not imgs:
                msg_warning(self, '无图像数据', '所选外部目录中未找到可用图像。')
                self.log('[ERROR] 强度带预测失败：外部目录内无可用图像。')
                self.predict_running = False
                self._update_predict_buttons([self.btn_import_ft_cnn, self.btn_import_ft_icnn],
                                             self.btn_predict, self.btn_stop, self.lbl_mode)
                return
            self.selected_micro_paths = imgs
            used_source = f'外部数据 (来自 {self.user_micro_dir})'
            self.log(f'[INFO] 已忽略 Vf 与 N，将处理外部目录内的全部图像：{len(imgs)} 张。')
        else:
            vf_dir = self._resolve_internal_vf_dir(vf)
            if vf_dir:
                imgs = gather_images(vf_dir)
                if imgs:
                    if len(imgs) >= n:
                        # 不放回抽样，天然不重复
                        self.selected_micro_paths = random.sample(imgs, n)
                    else:
                        self.selected_micro_paths = imgs[:]
                        self.log(f'[WARN] 内置数据库图像数({len(imgs)})少于所需数量({n})，将使用全部图像。')
                    used_source = f'内置数据库 (来自 {vf_dir})'
        if not self.selected_micro_paths:
            msg_warning(self, '无图像数据', '未找到可用于预测的图像。')
            self.log('[ERROR] 强度带预测失败：无可用图像。')
            self.predict_running = False
            self._update_predict_buttons([self.btn_import_ft_cnn, self.btn_import_ft_icnn],
                                         self.btn_predict, self.btn_stop, self.lbl_mode)
            return
        # 额外校验：随机抽样不应有重复
        if not ext_mode:
            if len(set(self.selected_micro_paths)) != len(self.selected_micro_paths):
                self.log('[ERROR] 发生重复抽样，这不应出现。将自动去重后继续。')
                # 去重但保持相对顺序
                seen = set()
                uniq = []
                for p in self.selected_micro_paths:
                    if p not in seen:
                        seen.add(p)
                        uniq.append(p)
                self.selected_micro_paths = uniq
        self.log(f'[INFO] 最终将处理 {len(self.selected_micro_paths)} 张图像；来源：{used_source}')
        # -- 使用后台线程执行批量预测 --
        mode = self.cfg.get_mode('微结构')
        if mode == '微调':
            cnn_path = self.cfg.get_ft('微结构', 'CNN') or ''
            icnn_path = self.cfg.get_ft('微结构', 'ICNN') or ''
        else:
            cnn_path = self.cfg.get_pt('微结构', 'CNN') or ''
            icnn_path = self.cfg.get_pt('微结构', 'ICNN') or ''
        self.predict_running = True
        self._band_running = True
        self._update_predict_buttons([self.btn_import_ft_cnn, self.btn_import_ft_icnn],
                                     self.btn_predict, self.btn_stop, self.lbl_mode)
        self.res_summary.setText(f'正在预测强度带（共 {len(self.selected_micro_paths)} 张）…')
        try:
            if hasattr(self, '_band_worker') and self._band_worker is not None:
                self._band_worker.request_stop()
                self._band_worker.terminate()
            self._band_worker = BandPredictWorker(self.selected_micro_paths, cnn_path, icnn_path, parent=self)
            self._band_worker.progress.connect(self._on_band_progress)
            self._band_worker.partial.connect(self._on_band_partial)
            self._band_worker.done.connect(self._on_band_done)
            self._band_worker.error.connect(self._on_band_error)
            self._band_worker.log_signal.connect(self.log)
            self._band_worker.finished.connect(lambda: self._finish_band_predict())
            self._band_worker.start()
        except Exception as e:
            self.log(f'[ERROR] 无法启动强度带预测线程：{e}')
            self.predict_running = False
            self._band_running = False
            self._update_predict_buttons([self.btn_import_ft_cnn, self.btn_import_ft_icnn],
                                         self.btn_predict, self.btn_stop, self.lbl_mode)

    # 线程将负责其余的聚合与绘图
    def _finish_band_predict(self):
        # 检查是否是用户停止的
        if self._band_running:
            self.log('[INFO] 强度带预测已终止。')
        self._band_running = False
        self.predict_running = False
        # 收尾进度条
        if hasattr(self, 'band_progress'):
            self.band_progress.setValue(self.band_progress.maximum())
            self.band_progress.setVisible(False)
        # 清空实时缓存
        if hasattr(self, '_live_xs'):
            self._live_xs, self._live_ys = ([], [])
        self._update_predict_buttons([self.btn_import_ft_cnn, self.btn_import_ft_icnn],
                                     self.btn_predict, self.btn_stop, self.lbl_mode)

    def _on_band_progress(self, i: int, total: int):
        # 初始化/更新进度条
        if hasattr(self, 'band_progress'):
            self.band_progress.setVisible(True)
            self.band_progress.setRange(0, max(1, int(total)))
            self.band_progress.setValue(int(i))
        self.res_summary.setText(f'正在预测强度带… {i}/{total}')

    def _on_band_partial(self, x_curve, y_curve):
        # 累积并即时绘制
        if not hasattr(self, '_live_xs'):
            self._live_xs, self._live_ys = ([], [])
        x_curve = np.array(x_curve, dtype=np.float32)
        y_curve = np.array(y_curve, dtype=np.float32)
        self._live_xs.append(x_curve)
        self._live_ys.append(y_curve)
        # 节流：每加入2条刷新一次
        n = len(self._live_xs)
        if n % 2 == 0:
            self._redraw_band_live()

    def _on_band_done(self, payload: dict):
        """强度带预测成功的回调函数。"""
        self.predict_running = False
        self._band_running = False
        try:
            x_common = np.array(payload.get('x_common', []), dtype=np.float32)
            y_min = np.array(payload.get('y_min', []), dtype=np.float32)
            y_max = np.array(payload.get('y_max', []), dtype=np.float32)
            xs_list = [np.array(x, dtype=np.float32) for x in payload.get('xs_list', [])]
            ys_list = [np.array(y, dtype=np.float32) for y in payload.get('ys_list', [])]
            if len(x_common) and len(y_min) and len(y_max):
                self._redraw_band_multi(xs_list, ys_list, x_common, y_min, y_max)
                self.res_summary.setText(f'强度带预测完成 ({len(ys_list)} 个样本)。')
                try:
                    self.last_band_data = {'x_common': x_common.tolist(), 'y_min': y_min.tolist(), 'y_max': y_max.tolist(),
                                           'xs_list': [xi.tolist() for xi in xs_list], 'ys_list': [yi.tolist() for yi in ys_list]}
                except Exception:
                    self.last_band_data = None
                self.log('[INFO] 强度带预测成功。')
            else:
                self._redraw_band()
                self.res_summary.setText('强度带预测失败，未得到有效数据。')
                self.log('[ERROR] 强度带预测失败。')
        finally:
            self._finish_band_predict()

    def _on_band_error(self, tb_text: str):
        self.predict_running = False
        self._band_running = False
        self.log(f'[ERROR] 强度带预测线程异常：\n{tb_text}')
        self.res_summary.setText('强度带预测失败，请检查日志。')
        self._redraw_band()
        self._finish_band_predict()

    def _stop_predict(self):
        """停止当前正在运行的微结构预测任务（包络或强度带）。"""
        # 立即禁用按钮，防止重复点击
        self.btn_stop.setEnabled(False)

        # 检查是否为强度带预测 (BandPredictWorker)
        if getattr(self, '_band_running', False) and hasattr(self, '_band_worker') and self._band_worker is not None:
            if self._band_worker.isRunning():
                self._band_worker.request_stop()

        # 检查是否为失效包络预测 (EnvelopePredictWorker)
        elif (not getattr(self, '_band_running', False)) and hasattr(self,
                                                                     '_env_worker') and self._env_worker is not None:
            if self._env_worker.isRunning():
                self._env_worker.request_stop()
        else:
            self.log('[INFO] 没有正在运行的微结构预测任务需要停止。')

    def export_state(self):
        # 收集关键输入与结果
        state = {
            'current_rve_path': self.current_rve_path,
            'user_micro_dir': self.user_micro_dir,
            'fiber': {
                'E': self.edt_fiber_E.text(),
                'nu': self.edt_fiber_nu.text(),
            },
            'matrix': {
                'E': self.edt_matrix_E.text(),
                'nu': self.edt_matrix_nu.text(),
                'Xc': self.edt_matrix_Xc.text(),
                'beta': self.edt_matrix_beta.text(),
                'upl': self.edt_matrix_upl.text(),
            },
            'interface': {
                'K': self.edt_K.text(),
                'tn0': self.edt_tn0.text(),
                'ts0': self.edt_ts0.text(),
                'Gn': self.edt_Gn.text(),
                'Gs': self.edt_Gs.text(),
            },
            'single_vf': self.cmb_single_vf.currentText(),
            'band_vf': self.cmb_vf.currentText(),
            'band_n': int(self.spn_n.value()),
            'last_env_curve': self.last_env_curve,
            'last_band_data': self.last_band_data,
        }
        return state

    def import_state(self, state: dict):
        try:
            self.current_rve_path = state.get('current_rve_path', '') or ''
            if self.current_rve_path:
                try:
                    self.lbl_img.set_full_text(self.current_rve_path)
                except Exception:
                    self.lbl_img.set_full_text(self.current_rve_path)
                self._update_preview(self.current_rve_path)
            self.user_micro_dir = state.get('user_micro_dir', '') or ''
            if self.user_micro_dir:
                try:
                    self.lbl_user_micro.set_full_text(self.user_micro_dir)
                except Exception:
                    self.lbl_user_micro.set_full_text(self.user_micro_dir)
            f = state.get('fiber', {})
            self.edt_fiber_E.setText(f.get('E', ''))
            self.edt_fiber_nu.setText(f.get('nu', ''))
            m = state.get('matrix', {})
            self.edt_matrix_E.setText(m.get('E', ''))
            self.edt_matrix_nu.setText(m.get('nu', ''))
            self.edt_matrix_Xc.setText(m.get('Xc', ''))
            self.edt_matrix_beta.setText(m.get('beta', ''))
            self.edt_matrix_upl.setText(m.get('upl', ''))
            itf = state.get('interface', {})
            self.edt_K.setText(itf.get('K', ''))
            self.edt_tn0.setText(itf.get('tn0', ''))
            self.edt_ts0.setText(itf.get('ts0', ''))
            self.edt_Gn.setText(itf.get('Gn', ''))
            self.edt_Gs.setText(itf.get('Gs', ''))
            sv = state.get('single_vf')
            if sv:
                idx = self.cmb_single_vf.findText(str(sv))
                if idx >= 0:
                    self.cmb_single_vf.setCurrentIndex(idx)
            bv = state.get('band_vf')
            if bv:
                idx = self.cmb_vf.findText(str(bv))
                if idx >= 0:
                    self.cmb_vf.setCurrentIndex(idx)
            bn = state.get('band_n')
            if isinstance(bn, int):
                self.spn_n.setValue(int(bn))
            # 恢复曲线
            self.last_env_curve = state.get('last_env_curve')
            if self.last_env_curve and self.last_env_curve.get('x') and self.last_env_curve.get('y'):
                x = np.array(self.last_env_curve['x'], dtype=np.float32)
                y = np.array(self.last_env_curve['y'], dtype=np.float32)
                self._redraw_env(x, y)
            self.last_band_data = state.get('last_band_data')
            if self.last_band_data:
                xc = np.array(self.last_band_data.get('x_common', []), dtype=np.float32)
                y_min = np.array(self.last_band_data.get('y_min', []), dtype=np.float32)
                y_max = np.array(self.last_band_data.get('y_max', []), dtype=np.float32)
                xs_list = [np.array(x, dtype=np.float32) for x in self.last_band_data.get('xs_list', [])]
                ys_list = [np.array(y, dtype=np.float32) for y in self.last_band_data.get('ys_list', [])]
                if len(xc) and len(y_min) and len(y_max):
                    self._redraw_band_multi(xs_list, ys_list, xc, y_min, y_max)
        except Exception as e:
            self.log(f'[WARN] 导入微结构页项目状态时发生错误：{e}')

# =============================
# 层级页：层合板
# =============================
class LaminatePage(_BaseLevelPage):
    """
    负责根据9个输入参数，预测并显示许用载荷空间的3D曲面图和结果表格。
    """

    def __init__(self, config: ProjectConfig, write_log, on_cfg_changed):
        super().__init__(config, write_log, '层合板', on_cfg_changed)
        layout = QHBoxLayout(self)

        left = QGroupBox('输入参数（层合板）')
        left_vbox = QVBoxLayout(left)
        left_vbox.setSpacing(10)

        # --- 使用 QTableWidget 作为输入 ---
        self.table_inputs = QTableWidget()
        self.table_inputs.setRowCount(9)
        self.table_inputs.setColumnCount(2)
        self.table_inputs.setHorizontalHeaderItem(0, QTableWidgetItem("设计参数"))
        self.table_inputs.setHorizontalHeaderItem(1, QTableWidgetItem("数值"))

        # 设置参数名称和参考数值
        param_names = [
            "长度：L", "宽度：W", "直径：D", "铺层角度：𝛷", "铺层角度：𝛹",
            "横向拉伸强度：mt", "面内剪切强度：sl", "G12", "E2"
        ]
        param_values = ["106", "197", "19", "68", "74", "80", "160", "5100", "8900"]

        for i, name in enumerate(param_names):
            item_name = QTableWidgetItem(name)
            item_name.setFlags(item_name.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table_inputs.setItem(i, 0, item_name)
            self.table_inputs.setItem(i, 1, QTableWidgetItem(param_values[i]))

        self.table_inputs.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_inputs.verticalHeader().setVisible(False)

        left_vbox.addWidget(self.table_inputs)

        # 计算并设置表格的固定高度
        row_height = self.table_inputs.rowHeight(0) if self.table_inputs.rowHeight(0) > 0 else 30
        header_height = self.table_inputs.horizontalHeader().height() if self.table_inputs.horizontalHeader().isVisible() else 30
        table_height = (row_height * self.table_inputs.rowCount()) + header_height + 4
        self.table_inputs.setFixedHeight(table_height)
        # --- 输入区结束 ---

        self.lbl_mode = QLabel()
        self.btn_import_ft = QPushButton('导入微调模型')
        self.btn_predict = QPushButton('开始预测')
        self.btn_stop = QPushButton('停止预测')
        self.btn_import_ft.clicked.connect(self._import_finetuned_model)
        self.btn_predict.clicked.connect(self._predict)
        self.btn_stop.clicked.connect(self._stop_predict)

        # 1. 创建按钮行 (ops)
        ops = QWidget()
        ops_l = QHBoxLayout(ops)
        ops_l.setContentsMargins(0, 0, 0, 0)
        ops_l.addWidget(self.btn_import_ft)
        ops_l.addWidget(self.btn_predict)
        ops_l.addWidget(self.btn_stop)

        # 2. 创建垂直容器 (controls_container)
        controls_container = QWidget()
        controls_vbox = QVBoxLayout(controls_container)
        controls_vbox.setContentsMargins(0, 0, 0, 0)
        controls_vbox.setSpacing(4)
        controls_vbox.addWidget(self.lbl_mode)
        controls_vbox.addWidget(ops)

        # 3. 将这个单一容器添加到 QVBoxLayout
        left_vbox.addWidget(controls_container)
        left_vbox.addStretch(1)

        # 必填字段检查 (在 _predict 中手动完成)
        self._required_fields = []
        # 用于保存和恢复3D绘图数据
        self.current_plot_data = None

        # --- 右侧结果区 ---
        right = QGroupBox('结果与可视化')
        rv = QVBoxLayout(right)
        # 1. Matplotlib 画布 (用于显示3D图)
        self.fig_res = Figure(figsize=(5, 4), tight_layout=True)
        self.ax_res = self.fig_res.add_subplot(111, projection='3d')
        self.canvas_res = FigureCanvas(self.fig_res)
        protect_canvas(self.canvas_res)
        self.toolbar_res = AutoNavToolbar(self.canvas_res, self)
        self._laminate_cbar = None

        # 2. 结果表格
        self.table_res = QTableWidget()
        self.table_res.setColumnCount(3)
        self.table_res.setHorizontalHeaderItem(0, QTableWidgetItem("铺层角度𝛷"))
        self.table_res.setHorizontalHeaderItem(1, QTableWidgetItem("铺层角度𝛹"))
        self.table_res.setHorizontalHeaderItem(2, QTableWidgetItem("初始失效因子𝑅"))
        self.table_res.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_res.verticalHeader().setVisible(False)
        self.table_res.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        rv.addWidget(self.toolbar_res)
        rv.addWidget(self.canvas_res, 1)
        rv.addWidget(self.table_res, 1)

        # 最终布局
        layout.addWidget(left, 2)
        layout.addWidget(right, 3)

        self.load_from_config()
        self._clear_results()  # 初始化绘图区

    def _clear_results(self):
        """清空结果和绘图（重建3D轴）"""
        try:
            # 清空保存的绘图数据
            self.current_plot_data = None

            # 移除可能存在的旧色条
            if hasattr(self, '_laminate_cbar') and self._laminate_cbar:
                try:
                    self._laminate_cbar.remove()
                except Exception:
                    pass
                self._laminate_cbar = None

            self.fig_res.clear()  # 清除整个Figure
            self.ax_res = self.fig_res.add_subplot(111, projection='3d')  # 重新创建3D轴
            # self.ax_res.set_title('许用载荷空间 (尚无结果)')
            self.fig_res.suptitle('许用载荷空间 (尚无结果)', y=0.9)

            # 添加坐标轴标签和旋转
            self.ax_res.set_xlabel('Angle1 (deg)')
            self.ax_res.set_ylabel('Angle2 (deg)')
            self.ax_res.set_zlabel('Predicted Factor (R)')

            # 调整视角，使Z轴在左侧
            self.ax_res.view_init(elev=20, azim=-135)

            self.canvas_res.draw_idle()
            self.table_res.setRowCount(0)
        except Exception as e:
            self.log(f'[WARN] 清理层合板绘图区失败: {e}')

    def load_from_config(self):
        """更新层合板页面的按钮状态"""
        mode = self.cfg.get_mode(self.level_name) # '预训练' 或 '微调'
        is_ft = (mode == '微调')
        ft_model_ok = bool(self.cfg.get_ft(self.level_name, 'default'))
        running = self.predict_running # 检查是否已有任务在运行

        self.lbl_mode.setText(self._make_mode_text())

        # 导入按钮：仅在微调模式且未运行时可用
        self.btn_import_ft.setEnabled(is_ft and not running)

        # 停止按钮：仅在任务运行时可用
        self.btn_stop.setEnabled(running)

        # 检查输入 (简单检查第一行是否有值)
        inputs_ok = True
        item = self.table_inputs.item(0, 1) # 检查第一个输入框
        if not item or not item.text().strip():
            inputs_ok = False
        else:
            try:
                float(item.text().strip()) # 检查是否为数字
            except ValueError:
                inputs_ok = False

        # 预测按钮：(未运行) 且 (输入OK) 且 ( (非微调模式) 或 (微调模式且模型OK) )
        can_predict = (not running) and inputs_ok and \
                      (not is_ft or (is_ft and ft_model_ok))
        self.btn_predict.setEnabled(can_predict)

    def _import_finetuned_model(self):
        # 使用自定义居中文件打开对话框
        f, _ = get_open_file(self, '导入微调模型', 'Model (*.h5 *.pt *.pth)')
        if f:
            self.cfg.set_ft(self.level_name, f, 'default')
            self.log(f'[INFO] 层合板页：已导入微调模型 -> {f}')
            self._update_predict_buttons([self.btn_import_ft], self.btn_predict, self.btn_stop, self.lbl_mode)
            self.on_cfg_changed()

    def _get_input_parameters(self) -> Optional[List[float]]:
        """从表格中获取并验证9个输入参数"""
        params = []
        for row in range(9):
            item = self.table_inputs.item(row, 1)
            if not item or item.text().strip() == '':
                msg_warning(self, "输入错误", f"第 {row + 1} 行参数 '{self.table_inputs.item(row, 0).text()}' 未输入！")
                return None
            try:
                params.append(float(item.text()))
            except ValueError:
                msg_warning(self, "输入错误", f"第 {row + 1} 行参数 '{item.text()}' 格式不正确，必须为数值！")
                return None
        return params

    def _predict(self):
        if not self._can_predict():
            self._update_predict_buttons([self.btn_import_ft], self.btn_predict, self.btn_stop, self.lbl_mode)
            return
        # 1. 获取和验证参数
        input_params = self._get_input_parameters()
        if input_params is None:
            return
        # 2. 获取模型路径
        mode = self.cfg.get_mode(self.level_name)
        if mode == '微调':
            model_path = self.cfg.get_ft(self.level_name, 'default')
        else:  # 预训练
            model_path = self.cfg.get_pt(self.level_name, 'default')
        # 3. 启动线程
        self.predict_running = True
        self._update_predict_buttons([self.btn_import_ft], self.btn_predict, self.btn_stop, self.lbl_mode)
        self.log('[INFO] 启动 LaminatePage 预测…')
        self._clear_results()  # 清空上次结果
        try:
            if hasattr(self, '_level_worker') and self._level_worker is not None:
                self._level_worker.terminate()
            self._level_worker = LaminatePredictWorker(model_path, input_params, parent=self)
            self._level_worker.done.connect(self._on_level_done)
            self._level_worker.error.connect(self._on_level_error)
            self._level_worker.log_signal.connect(self.log)
            self._level_worker.finished.connect(self._finish_level_predict)
            self._level_worker.start()
        except Exception as e:
            self.log(f'[ERROR] 无法启动层合板预测线程：{e}')
            self.predict_running = False
            self.load_from_config()  # 重置按钮

    def _on_level_done(self, ANGLE1, ANGLE2, Z, result_df: pd.DataFrame):
        """预测成功回调：显示交互式3D图像和表格"""
        self.predict_running = False
        try:
            # 缓存绘图数据，以便保存项目
            self.current_plot_data = {
                'angle1': ANGLE1,
                'angle2': ANGLE2,
                'z': Z
            }

            # 1. 绘制交互式 3D 图像
            self.log('[INFO] 正在绘制3D曲面...')
            self.ax_res.clear()  # 清除 "尚无结果"
            surf = self.ax_res.plot_surface(ANGLE1, ANGLE2, Z, cmap='viridis', rstride=1, cstride=1, linewidth=0.1)
            # self.ax_res.set_title('许用载荷空间')
            self.fig_res.suptitle('许用载荷空间', y=0.9)

            # 统一坐标轴标签和旋转
            self.ax_res.set_xlabel('Angle1 (deg)')
            self.ax_res.set_ylabel('Angle2 (deg)')
            self.ax_res.set_zlabel('Predicted Factor (R)')

            # 调整视角，使Z轴在左侧
            self.ax_res.view_init(elev=20, azim=-135)

            # 移除旧色条 (如果存在)
            if hasattr(self, '_laminate_cbar') and self._laminate_cbar:
                try:
                    self._laminate_cbar.remove()
                except Exception:
                    pass

            # 添加色条，并使用 pad 增加间距
            self._laminate_cbar = self.fig_res.colorbar(surf, ax=self.ax_res, shrink=0.5, aspect=10, pad=0.1)
            self.canvas_res.draw_idle()

            # 2. 填充结果表格
            self.log('[INFO] 正在填充结果表格...')
            self.table_res.setRowCount(len(result_df))
            for idx, row in result_df.iterrows():
                self.table_res.setItem(idx, 0, QTableWidgetItem(f"{row['铺层角度𝛷']:.1f}"))
                self.table_res.setItem(idx, 1, QTableWidgetItem(f"{row['铺层角度𝛹']:.1f}"))
                self.table_res.setItem(idx, 2, QTableWidgetItem(f"{row['初始失效因子𝑅']:.4f}"))
            self.log('[INFO] 层合板预测完成。')

        except Exception as e:
            self.log(f'[ERROR] 加载层合板结果失败: {e}')

        self._finish_level_predict()  # 调用finish

    def _on_level_error(self, tb_text: str):
        self.predict_running = False
        self._clear_results()  # 失败时也清空
        self.log(f'[ERROR] 预测线程异常：\n{tb_text}')
        self._finish_level_predict()  # 确保按钮被重置

    def _finish_level_predict(self):
        # 检查是否是用户停止的
        if self.predict_running:
            self.log('[INFO] 层合板预测已终止。')

        self.predict_running = False
        self.load_from_config()  # 重置所有按钮状态

    def _stop_predict(self):
        """停止层合板预测线程。"""
        self.btn_stop.setEnabled(False)  # 立即禁用

        # 层合板预测
        if hasattr(self, '_level_worker') and self._level_worker.isRunning():
            self._level_worker.request_stop()
        else:
            self.log('[INFO] 层合板预测任务未在运行。')

    def export_state(self):
        state = {
            'inputs': [],
            'res_table': [],
            'plot_data': None  # 存储3D绘图数据
        }
        try:
            # 保存输入
            for row in range(9):
                name = self.table_inputs.item(row, 0).text()
                val = self.table_inputs.item(row, 1).text()
                state['inputs'].append((name, val))
            # 保存输出
            if self.table_res.rowCount() > 0:
                # 仅保存少量数据作为预览
                for row in range(min(50, self.table_res.rowCount())):
                    r_data = [
                        self.table_res.item(row, 0).text(),
                        self.table_res.item(row, 1).text(),
                        self.table_res.item(row, 2).text()
                    ]
                    state['res_table'].append(r_data)

            # 保存绘图数据
            if hasattr(self, 'current_plot_data') and self.current_plot_data:
                try:
                    state['plot_data'] = {
                        'angle1': self.current_plot_data['angle1'].tolist(),
                        'angle2': self.current_plot_data['angle2'].tolist(),
                        'z': self.current_plot_data['z'].tolist()
                    }
                except Exception as e:
                    self.log(f'[WARN] 无法序列化3D绘图数据: {e}')
                    state['plot_data'] = None

        except Exception as e:
            self.log(f'[WARN] 导出层合板状态失败: {e}')
        return state

    def import_state(self, state: dict):
        try:
            # 恢复输入
            inputs = state.get('inputs', [])
            if len(inputs) == 9:
                for row, (name, val) in enumerate(inputs):
                    # 确保 item 存在
                    if self.table_inputs.item(row, 0) is None:
                        self.table_inputs.setItem(row, 0, QTableWidgetItem(name))
                    else:
                        self.table_inputs.item(row, 0).setText(name)
                    if self.table_inputs.item(row, 1) is None:
                        self.table_inputs.setItem(row, 1, QTableWidgetItem(val))
                    else:
                        self.table_inputs.item(row, 1).setText(val)

            # 恢复输出 (表格)
            res_table = state.get('res_table', [])
            self.table_res.setRowCount(len(res_table))
            for row, r_data in enumerate(res_table):
                self.table_res.setItem(row, 0, QTableWidgetItem(r_data[0]))
                self.table_res.setItem(row, 1, QTableWidgetItem(r_data[1]))
                self.table_res.setItem(row, 2, QTableWidgetItem(r_data[2]))

            # 恢复 3D 图像
            plot_data = state.get('plot_data')
            if plot_data and plot_data.get('angle1') and plot_data.get('angle2') and plot_data.get('z'):
                self.log('[INFO] 正在恢复3D图像...')
                ANGLE1 = np.array(plot_data['angle1'])
                ANGLE2 = np.array(plot_data['angle2'])
                Z = np.array(plot_data['z'])
                # 重新绘制
                self.ax_res.clear()
                surf = self.ax_res.plot_surface(ANGLE1, ANGLE2, Z, cmap='viridis', rstride=1, cstride=1, linewidth=0.1)
                # self.ax_res.set_title('许用载荷空间')
                self.fig_res.suptitle('许用载荷空间', y=0.9)
                self.ax_res.set_xlabel('Angle1 (deg)')
                self.ax_res.set_ylabel('Angle2 (deg)')
                self.ax_res.set_zlabel('Predicted Factor (R)')
                # 调整视角，使Z轴在左侧
                self.ax_res.view_init(elev=20, azim=-135)
                if hasattr(self, '_laminate_cbar') and self._laminate_cbar:
                    try:
                        self._laminate_cbar.remove()
                    except Exception:
                        pass
                self._laminate_cbar = self.fig_res.colorbar(surf, ax=self.ax_res, shrink=0.5, aspect=10, pad=0.1)
                self.canvas_res.draw_idle()
                # 缓存数据以便再次保存
                self.current_plot_data = {'angle1': ANGLE1, 'angle2': ANGLE2, 'z': Z}
                self.log('[INFO] 已恢复层合板表格和3D图像。')
            else:
                # 如果没有绘图数据，则只清空图像
                self._clear_results()
                if res_table:
                    self.log('[INFO] 已恢复层合板表格数据（图像需重新预测）。')

        except Exception as e:
            self.log(f'[WARN] 导入层合板页项目状态时发生错误：{e}')

# =============================
# 层级页：加筋结构
# =============================
class StiffenedPanelPage(_BaseLevelPage):
    """
    包含两个主要功能：
    1. 性能预测：输入11个参数，预测5阶屈曲载荷和1阶屈曲模态图。
    2. 优化设计：输入9个参数，使用遗传算法优化 W1, H1, d。
    """

    def __init__(self, config: ProjectConfig, write_log, on_cfg_changed):
        super().__init__(config, write_log, '加筋结构', on_cfg_changed)

        self._predict_worker = None
        self._optimize_worker = None
        self.opt_plot_data = {}
        self.current_buckling_loads = []
        self.current_mode_shape_image = None
        layout = QHBoxLayout(self)

        # --- 左侧布局 ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # 1. 创建左侧的 TabWidget
        self.left_tabs = QTabWidget()

        # 2. 创建“性能预测”输入页面
        page_predict_input = QWidget()
        predict_page_vbox = QVBoxLayout(page_predict_input)
        pred_form = QFormLayout()
        predict_page_vbox.addLayout(pred_form)

        self.table_predict_inputs = QTableWidget()
        self.table_predict_inputs.setRowCount(11)
        self.table_predict_inputs.setColumnCount(2)
        self.table_predict_inputs.setHorizontalHeaderItem(0, QTableWidgetItem("参数"))
        self.table_predict_inputs.setHorizontalHeaderItem(1, QTableWidgetItem("数值"))
        input_order = [
            ("L/mm", "327"), ("W/mm", "143"), ("H1/mm", "27"), ("W1/mm", "25"),
            ("E1/MPa", "159273"), ("E2/MPa", "17615"), ("Nu12", "0.39017"),
            ("G12/MPa", "3770"), ("G13/MPa", "3770.1"), ("G23/MPa", "4020"),
            ("d/mm", "55")
        ]
        for i, (name, value) in enumerate(input_order):
            item_name = QTableWidgetItem(name)
            item_name.setFlags(item_name.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table_predict_inputs.setItem(i, 0, item_name)
            self.table_predict_inputs.setItem(i, 1, QTableWidgetItem(value))
        self.table_predict_inputs.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_predict_inputs.verticalHeader().setVisible(False)

        # 为表格设置固定高度以显示所有行
        row_height_pred = self.table_predict_inputs.rowHeight(0) if self.table_predict_inputs.rowHeight(0) > 0 else 30
        header_height_pred = self.table_predict_inputs.horizontalHeader().height() if self.table_predict_inputs.horizontalHeader().isVisible() else 30
        table_height_pred = (row_height_pred * self.table_predict_inputs.rowCount()) + header_height_pred + 4
        self.table_predict_inputs.setFixedHeight(table_height_pred)
        pred_form.addRow(self.table_predict_inputs)

        # --- 为性能预测tab也添加stretch，使其顶部对齐 ---
        predict_page_vbox.addStretch(1)

        # 3. 创建“优化设计”输入页面
        page_optimize_input = QWidget()
        optimize_page_vbox = QVBoxLayout(page_optimize_input)
        opt_form = QFormLayout()
        optimize_page_vbox.addLayout(opt_form)

        self.table_optimize_inputs = QTableWidget()
        self.table_optimize_inputs.setRowCount(9)
        self.table_optimize_inputs.setColumnCount(2)
        self.table_optimize_inputs.setHorizontalHeaderItem(0, QTableWidgetItem("设计参数"))
        self.table_optimize_inputs.setHorizontalHeaderItem(1, QTableWidgetItem("数值"))
        opt_params = [
            ("Eigenvalue1 (约束)", "272788"), ("底板长度 L", "327"), ("底板宽度 W", "143"),
            ("E1", "159273"), ("E2", "17615"), ("Nu12", "0.39017"),
            ("G12", "3770"), ("G13", "3770.1"), ("G23", "4020")
        ]
        for i, (name, value) in enumerate(opt_params):
            item_name = QTableWidgetItem(name)
            item_name.setFlags(item_name.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table_optimize_inputs.setItem(i, 0, item_name)
            self.table_optimize_inputs.setItem(i, 1, QTableWidgetItem(value))
        self.table_optimize_inputs.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_optimize_inputs.verticalHeader().setVisible(False)

        # 为表格设置固定高度以显示所有行
        row_height_opt = self.table_optimize_inputs.rowHeight(0) if self.table_optimize_inputs.rowHeight(0) > 0 else 30
        header_height_opt = self.table_optimize_inputs.horizontalHeader().height() if self.table_optimize_inputs.horizontalHeader().isVisible() else 30
        table_height_opt = (row_height_opt * self.table_optimize_inputs.rowCount()) + header_height_opt + 4
        self.table_optimize_inputs.setFixedHeight(table_height_opt)
        opt_form.addRow(self.table_optimize_inputs)

        # --- 为优化设计tab添加stretch，使其顶部对齐 ---
        optimize_page_vbox.addStretch(1)

        # 4. 将页面添加到 TabWidget
        self.left_tabs.addTab(page_predict_input, "性能预测")
        self.left_tabs.addTab(page_optimize_input, "优化设计")

        # 5. 将 TabWidget 添加到左侧主布局
        left_layout.addWidget(self.left_tabs, 0)

        # 6. 控制区
        control_box = QWidget()
        # 使用 QHBoxLayout 来横向排列按钮
        control_hbox = QHBoxLayout(control_box)
        control_hbox.setContentsMargins(0, 0, 0, 0)  # 移除边距
        self.lbl_mode = QLabel(self._make_mode_text())
        # 创建新的统一按钮
        self.btn_import_ft = QPushButton('导入微调模型')
        self.btn_start = QPushButton('开始...')  # 文本将在 load_from_config 中设置
        self.btn_stop = QPushButton('停止任务')

        control_hbox.addWidget(self.btn_import_ft)
        control_hbox.addWidget(self.btn_start)
        control_hbox.addWidget(self.btn_stop)

        left_layout.addWidget(self.lbl_mode)
        left_layout.addWidget(control_box)

        # 添加一个最终的伸缩，将 TabWidget 和 ControlBox 推到顶部
        left_layout.addStretch(1)

        # --- 右侧结果区 ---
        right_box = QGroupBox('结果与可视化')
        right_layout = QVBoxLayout(right_box)
        self.tabs_results = QTabWidget()
        # -- 预测结果页 --
        page_predict = QWidget()
        layout_predict = QVBoxLayout(page_predict)
        pred_mode_box = QGroupBox("屈曲模态预测")
        layout_pred_mode = QVBoxLayout(pred_mode_box)
        self.fig_mode_shape = Figure(figsize=(5, 2.5), tight_layout=True)  # 保持缩小
        self.ax_mode_shape = self.fig_mode_shape.add_subplot(111)
        self.canvas_mode_shape = FigureCanvas(self.fig_mode_shape)
        protect_canvas(self.canvas_mode_shape)
        self.toolbar_mode_shape = AutoNavToolbar(self.canvas_mode_shape, self)
        layout_pred_mode.addWidget(self.toolbar_mode_shape)
        layout_pred_mode.addWidget(self.canvas_mode_shape)
        layout_predict.addWidget(pred_mode_box, 1)
        pred_load_box = QGroupBox("屈曲载荷预测")
        layout_pred_load = QVBoxLayout(pred_load_box)
        self.fig_load_curve = Figure(figsize=(4, 2.5), tight_layout=True)  # 保持缩小
        self.ax_load_curve = self.fig_load_curve.add_subplot(111)
        self.canvas_load_curve = FigureCanvas(self.fig_load_curve)
        protect_canvas(self.canvas_load_curve)
        self.toolbar_load_curve = AutoNavToolbar(self.canvas_load_curve, self)
        layout_pred_load.addWidget(self.toolbar_load_curve)
        layout_pred_load.addWidget(self.canvas_load_curve)
        layout_predict.addWidget(pred_load_box, 1)
        self.tabs_results.addTab(page_predict, "性能预测结果")
        # -- 优化结果页 --
        page_optimize = QWidget()
        layout_optimize = QVBoxLayout(page_optimize)
        opt_plot_box = QGroupBox("各项参数优化过程")
        layout_opt_plots = QGridLayout(opt_plot_box)
        fig_w, fig_h = 2.5, 1.8
        self.fig_opt_w1 = Figure(figsize=(fig_w, fig_h), tight_layout=True)
        self.ax_opt_w1 = self.fig_opt_w1.add_subplot(111)
        self.canvas_opt_w1 = FigureCanvas(self.fig_opt_w1)
        protect_canvas(self.canvas_opt_w1)
        self.fig_opt_h1 = Figure(figsize=(fig_w, fig_h), tight_layout=True)
        self.ax_opt_h1 = self.fig_opt_h1.add_subplot(111)
        self.canvas_opt_h1 = FigureCanvas(self.fig_opt_h1)
        protect_canvas(self.canvas_opt_h1)
        self.fig_opt_d = Figure(figsize=(fig_w, fig_h), tight_layout=True)
        self.ax_opt_d = self.fig_opt_d.add_subplot(111)
        self.canvas_opt_d = FigureCanvas(self.fig_opt_d)
        protect_canvas(self.canvas_opt_d)
        self.fig_opt_perf = Figure(figsize=(fig_w, fig_h), tight_layout=True)
        self.ax_opt_perf = self.fig_opt_perf.add_subplot(111)
        self.canvas_opt_perf = FigureCanvas(self.fig_opt_perf)
        protect_canvas(self.canvas_opt_perf)
        self.opt_canvases = {
            'W1': (self.fig_opt_w1, self.ax_opt_w1, self.canvas_opt_w1),
            'H1': (self.fig_opt_h1, self.ax_opt_h1, self.canvas_opt_h1),
            'd': (self.fig_opt_d, self.ax_opt_d, self.canvas_opt_d),
            'perf': (self.fig_opt_perf, self.ax_opt_perf, self.canvas_opt_perf),
        }
        layout_opt_plots.addWidget(self.canvas_opt_w1, 0, 0)
        layout_opt_plots.addWidget(self.canvas_opt_h1, 0, 1)
        layout_opt_plots.addWidget(self.canvas_opt_d, 1, 0)
        layout_opt_plots.addWidget(self.canvas_opt_perf, 1, 1)
        layout_opt_plots.setColumnStretch(0, 1)
        layout_opt_plots.setColumnStretch(1, 1)
        layout_opt_plots.setRowStretch(0, 1)
        layout_opt_plots.setRowStretch(1, 1)
        layout_optimize.addWidget(opt_plot_box, 2)
        opt_res_box = QGroupBox("参数优化结果")
        layout_opt_res = QVBoxLayout(opt_res_box)
        self.table_opt_results = QTableWidget()
        self.table_opt_results.setRowCount(5)
        self.table_opt_results.setColumnCount(2)
        self.table_opt_results.setHorizontalHeaderItem(0, QTableWidgetItem("设计参数"))
        self.table_opt_results.setHorizontalHeaderItem(1, QTableWidgetItem("数值"))
        opt_res_params = ["W1", "H1", "d", "Eigenvalue1", "性能指标 (轻量化)"]
        for i, name in enumerate(opt_res_params):
            item = QTableWidgetItem(name)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table_opt_results.setItem(i, 0, item)
        self.table_opt_results.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_opt_results.verticalHeader().setVisible(False)
        layout_opt_res.addWidget(self.table_opt_results)
        layout_optimize.addWidget(opt_res_box, 1)
        self.tabs_results.addTab(page_optimize, "优化设计结果")
        right_layout.addWidget(self.tabs_results)

        # --- 最终布局 ---
        layout.addWidget(left_widget, 2)
        layout.addWidget(right_box, 3)

        # --- 连接信号槽 ---
        self.btn_import_ft.clicked.connect(self._import_finetuned_model)
        self.btn_start.clicked.connect(self._start_task)
        self.btn_stop.clicked.connect(self._stop_task)

        # 表格内容变化时触发按钮状态更新
        self.table_predict_inputs.itemChanged.connect(self.load_from_config)
        self.table_optimize_inputs.itemChanged.connect(self.load_from_config)
        # Tab 切换时触发按钮状态和文本更新
        self.left_tabs.currentChanged.connect(self.load_from_config)

        # --- 为 2D 曲线附加悬浮坐标提示 ---
        attach_hover_coords(self.ax_mode_shape, self.canvas_mode_shape)
        attach_hover_coords(self.ax_load_curve, self.canvas_load_curve, fmt='(阶数 {x:.0f}, 载荷 {y:.3g})')
        fmt_w1 = '(迭代 {x:.0f}, W1 {y:.3g})'
        fmt_h1 = '(迭代 {x:.0f}, H1 {y:.3g})'
        fmt_d = '(迭代 {x:.0f}, d {y:.3g})'
        fmt_perf = '(迭代 {x:.0f}, Perf {y:.3g})'
        attach_hover_coords(self.opt_canvases['W1'][1], self.opt_canvases['W1'][2], fmt=fmt_w1)
        attach_hover_coords(self.opt_canvases['H1'][1], self.opt_canvases['H1'][2], fmt=fmt_h1)
        attach_hover_coords(self.opt_canvases['d'][1], self.opt_canvases['d'][2], fmt=fmt_d)
        attach_hover_coords(self.opt_canvases['perf'][1], self.opt_canvases['perf'][2], fmt=fmt_perf)

        self._clear_predict_results()
        self._clear_optimize_results()
        self.load_from_config()  # 初始加载按钮状态和文本

    def _validate_and_highlight_table(self, table: QTableWidget) -> bool:
        """
        验证表格的第二列（索引1）中的所有输入。
        - 检查是否为空
        - 检查是否为有效浮点数
        - 为无效单元格应用浅红色背景
        - 为有效单元格清除背景
        - 返回 True (全部有效) 或 False (至少一个无效)
        """
        all_valid = True
        # 红色背景 (来自 #d9534f)
        invalid_brush = QBrush(QColor(242, 222, 222))  # 浅红色
        valid_brush = QBrush(Qt.GlobalColor.white)  # 默认白色背景

        try:
            for row in range(table.rowCount()):
                item = table.item(row, 1)
                is_valid = False

                if item:
                    text = item.text().strip()
                    if text:
                        try:
                            float(text)
                            is_valid = True
                        except ValueError:
                            is_valid = False  # 非空但不是数字

                if is_valid:
                    if item:  # 确保 item 存在
                        item.setBackground(valid_brush)
                else:
                    all_valid = False
                    if not item:
                        # 如果单元格不存在 (例如, 第一次加载时)
                        item = QTableWidgetItem("")
                        table.setItem(row, 1, item)
                    item.setBackground(invalid_brush)  # 设置红色背景

            return all_valid
        except Exception as e:
            self.log(f'[WARN] 验证高亮表格时出错: {e}')
            return False  # 出错时默认为无效

    # --- 统一的导入微调模型方法 ---
    def _import_finetuned_model(self):
        """根据当前左侧 Tab 导入对应的微调模型 (预测或优化)。"""
        current_tab_index = self.left_tabs.currentIndex()
        is_predict_tab = (current_tab_index == 0)
        task_name = "性能预测" if is_predict_tab else "优化设计"
        model_key = "predict_model" if is_predict_tab else "optimize_model"
        scaler_key = "predict_scaler" if is_predict_tab else "optimize_scaler"
        joblib_filter = 'Scikit-learn Model (*.joblib)'

        self.log(f'[INFO] 1/2: 请选择微调后的 {task_name} 模型 ({joblib_filter})...')
        f_model, _ = get_open_file(self, f'1/2: 选择微调后的 {task_name} 模型', filter=joblib_filter)
        if not f_model:
            self.log('[INFO] 取消导入。')
            return

        self.log(f'[INFO] 2/2: 请选择配套的 {task_name} Scaler ({joblib_filter})...')
        default_dir = os.path.dirname(f_model)
        f_scaler, _ = get_open_file(self, f'2/2: 选择微调后的 {task_name} Scaler', default_dir=default_dir,
                                    filter=joblib_filter)
        if not f_scaler:
            self.log('[INFO] 取消导入。')
            return

        # 存储到 config
        self.cfg.set_ft(self.level_name, f_model, sub=model_key)
        self.cfg.set_ft(self.level_name, f_scaler, sub=scaler_key)

        self.log(f'[INFO] 加筋结构 ({task_name}) 微调模型导入完成。')
        self.load_from_config()  # 更新按钮状态
        self.on_cfg_changed()  # 更新 Inspector

    # --- 统一的开始任务方法 ---
    def _start_task(self):
        """根据当前左侧 Tab 启动预测或优化任务。"""
        if self.predict_running:  # 如果已有任务在运行，则不启动新任务
            return

        current_tab_index = self.left_tabs.currentIndex()
        if current_tab_index == 0:  # 性能预测 Tab
            self._run_prediction_task()  # 调用（可能重构后的）预测逻辑
        elif current_tab_index == 1:  # 优化设计 Tab
            self._run_optimization_task()  # 调用（可能重构后的）优化逻辑

    # --- 运行预测任务的逻辑 (从旧 _predict 提取) ---
    def _run_prediction_task(self):
        """包含启动性能预测工作线程的逻辑。"""
        inputs = self._get_predict_inputs()
        if inputs is None:
            return

        self.tabs_results.setCurrentIndex(0)  # 切换到预测结果 Tab
        self._clear_predict_results()

        mode = self.cfg.get_mode(self.level_name)
        img_model_dir = os.path.join(MODEL_DIR, self.level_name, "性能预测")
        img_model_path = os.path.join(img_model_dir, 'model_figure.pth')

        if not os.path.exists(img_model_path):
            self.log(f'[ERROR] 关键文件缺失：找不到图像生成模型: {img_model_path}')
            msg_critical(self, "文件缺失",
                         f"找不到必须的图像生成模型:\n{img_model_path}\n\n请确保 '性能预测' 预训练模型已正确安装。")
            return

        try:
            reg_model_path, reg_scaler_path = "", ""
            if mode == '微调':
                self.log('[INFO] 使用微调模型 (Joblib) 进行预测...')
                reg_model_path = self.cfg.get_ft(self.level_name, 'predict_model')
                reg_scaler_path = self.cfg.get_ft(self.level_name, 'predict_scaler')
                if not reg_model_path or not reg_scaler_path or not os.path.exists(
                        reg_model_path) or not os.path.exists(reg_scaler_path):
                    self.log('[ERROR] 微调模型或 Scaler 未导入，或文件不存在。')
                    msg_warning(self, "模型错误", "请先导入性能预测的微调模型和Scaler。")
                    return
            else:  # 预训练
                self.log('[INFO] 使用预训练模型 (Joblib) 进行预测...')
                model_base_dir = os.path.join(MODEL_DIR, self.level_name, "性能预测")
                reg_model_path = os.path.join(model_base_dir, 'model_shear&buckle.joblib')
                reg_scaler_path = os.path.join(model_base_dir, 'model_shear&buckle_scaler.joblib')
                if not os.path.exists(reg_model_path) or not os.path.exists(reg_scaler_path):
                    self.log(f'[ERROR] 预训练模型或 Scaler 文件不存在于: {model_base_dir}')
                    msg_critical(self, "文件缺失", f"性能预测预训练模型或 Scaler 缺失:\n{model_base_dir}")
                    return

            # --- 启动 Worker ---
            self.predict_running = True
            self.load_from_config()  # 更新按钮状态

            if self._predict_worker: self._predict_worker.deleteLater()  # 清理旧 worker

            self._predict_worker = StiffenedPanelPredictWorker(
                reg_model_path=reg_model_path, reg_scaler_path=reg_scaler_path,
                img_model_path=img_model_path, input_params=inputs, parent=self
            )
            self._predict_worker.done.connect(self._on_predict_done)
            self._predict_worker.error.connect(self._on_predict_error)
            self._predict_worker.log_signal.connect(self.log)
            self._predict_worker.finished.connect(self._finish_predict)  # 连接清理函数
            self._predict_worker.start()

        except Exception as e:
            self.log(f'[CRITICAL] 无法启动预测线程：{e}\n{traceback.format_exc()}')
            msg_critical(self, "线程错误", f"无法启动预测线程:\n{e}")
            self._finish_predict()  # 确保清理

    # --- 运行优化任务的逻辑 (从旧 _start_optimization 提取) ---
    def _run_optimization_task(self):
        """包含启动优化设计工作线程的逻辑。"""
        inputs = self._get_optimize_inputs()
        if inputs is None:
            return

        self.tabs_results.setCurrentIndex(1)  # 切换到优化结果 Tab
        self._clear_optimize_results()

        mode = self.cfg.get_mode(self.level_name)
        model_path, scaler_path = "", ""

        try:
            if mode == '微调':
                model_path = self.cfg.get_ft(self.level_name, 'optimize_model')
                scaler_path = self.cfg.get_ft(self.level_name, 'optimize_scaler')
                if not model_path or not scaler_path or not os.path.exists(model_path) or not os.path.exists(
                        scaler_path):
                    self.log('[ERROR] 微调模式下未找到优化设计模型或Scaler，或文件不存在。')
                    msg_warning(self, "模型错误", "请先导入优化设计的微调模型和Scaler。")
                    return
                self.log(f'[INFO] 使用优化设计微调模型: {model_path}')
            else:  # 预训练
                model_base_dir = os.path.join(MODEL_DIR, self.level_name, "优化设计")
                model_path = os.path.join(model_base_dir, 'model_shear&buckle_reverse_design.joblib')
                scaler_path = os.path.join(model_base_dir, 'model_shear&buckle_scaler_reverse_design.joblib')
                self.log(f'[INFO] 使用优化设计预训练模型: {model_path}')
                if not os.path.isdir(model_base_dir) or not os.path.exists(model_path) or not os.path.exists(
                        scaler_path):
                    self.log(f'[ERROR] 预训练模型目录或文件不存在: {model_base_dir}')
                    msg_critical(self, "目录错误", f"优化设计预训练模型文件缺失:\n{model_base_dir}")
                    return

            self.predict_running = True
            self.load_from_config()
            self.log('[INFO] 启动加筋结构优化设计...')

            if self._optimize_worker: self._optimize_worker.deleteLater()  # 清理旧 worker

            self._optimize_worker = StiffenedPanelOptimizeWorker(model_path, scaler_path, inputs, self)
            self._optimize_worker.update_progress.connect(self._on_optimize_update)
            self._optimize_worker.finished.connect(self._on_optimize_finished)
            self._optimize_worker.error.connect(self._on_optimize_error)
            self._optimize_worker.log_signal.connect(self.log)
            self._optimize_worker.finished.connect(self._finish_optimize)  # 连接清理函数
            self._optimize_worker.start()

        except Exception as e:
            self.log(f'[CRITICAL] 无法启动优化线程：{e}\n{traceback.format_exc()}')
            msg_critical(self, "线程错误", f"无法启动优化线程:\n{e}")
            self._finish_optimize()  # 确保清理

    def _clear_predict_results(self):
        """清空预测结果"""
        self.current_buckling_loads = []  # 重置屈曲荷载缓存
        self.current_mode_shape_image = None  # 重置模态图像缓存
        self.ax_mode_shape.clear()
        self.ax_mode_shape.set_title('屈曲模态 (尚无结果)')
        self.ax_mode_shape.axis('off')
        self.canvas_mode_shape.draw_idle()
        self.ax_load_curve.clear()
        self.ax_load_curve.set_title('屈曲荷载 (尚无结果)')
        self.ax_load_curve.set_xlabel('阶数')
        self.ax_load_curve.set_ylabel('载荷 (kN)')
        self.ax_load_curve.grid(True)
        self.canvas_load_curve.draw_idle()

    def _clear_optimize_results(self):
        """清空优化结果"""
        self.opt_plot_data = {}
        for key, (fig, ax, canvas) in self.opt_canvases.items():
            ax.clear()
            ax.set_title(f'{key} (尚无结果)')
            ax.set_xlabel('迭代次数')
            ax.grid(True)
            canvas.draw_idle()

        for i in range(5):
            self.table_opt_results.setItem(i, 1, QTableWidgetItem(""))

    def load_from_config(self):
        """更新加筋结构页面的按钮状态和文本，并根据 Dashboard 选择切换 Tab。"""
        mode = self.cfg.get_mode(self.level_name)
        is_ft = (mode == '微调')
        running = self.predict_running

        self.lbl_mode.setText(self._make_mode_text())

        # 统一的导入按钮：仅在微调模式且未运行时可用
        self.btn_import_ft.setEnabled(is_ft and not running)

        # 统一的停止按钮：仅在任务运行时可用
        self.btn_stop.setEnabled(running)

        # 检查微调模型是否已导入
        ft_pred_ok = bool(self.cfg.get_ft(self.level_name, 'predict_model')) and \
                     bool(self.cfg.get_ft(self.level_name, 'predict_scaler'))
        ft_opt_ok = bool(self.cfg.get_ft(self.level_name, 'optimize_model')) and \
                    bool(self.cfg.get_ft(self.level_name, 'optimize_scaler'))

        # 检查输入是否有效 (调用新的高亮函数)
        pred_inputs_ok = self._validate_and_highlight_table(self.table_predict_inputs)
        opt_inputs_ok = self._validate_and_highlight_table(self.table_optimize_inputs)

        # 获取 Dashboard 的控制选择
        selected_pt_mode = self.cfg.get_pt(self.level_name, 'default')
        is_predict_selected_pt = '性能预测' in selected_pt_mode
        is_optimize_selected_pt = '优化设计' in selected_pt_mode

        # 获取当前左侧 Tab
        current_tab_index = self.left_tabs.currentIndex()
        is_predict_tab = (current_tab_index == 0)
        is_optimize_tab = (current_tab_index == 1)

        # 更新 "开始..." 按钮的文本
        self.btn_start.setText("开始预测" if is_predict_tab else "优化设计")

        # 启用/禁用 "开始..." 按钮的逻辑
        can_start = False
        if not running:
            if is_predict_tab and is_predict_selected_pt: # 如果在预测 Tab 且 Dashboard 选了预测
                can_start = pred_inputs_ok and (not is_ft or ft_pred_ok)
            elif is_optimize_tab and is_optimize_selected_pt: # 如果在优化 Tab 且 Dashboard 选了优化
                can_start = opt_inputs_ok and (not is_ft or ft_opt_ok)

        self.btn_start.setEnabled(can_start)

        # 根据 Dashboard 的选择自动切换左侧 Tab (仅在非运行状态下)
        if not running:
            if is_predict_selected_pt and not is_predict_tab:
                # 阻止信号循环触发 load_from_config
                self.left_tabs.blockSignals(True)
                self.left_tabs.setCurrentIndex(0)
                self.left_tabs.blockSignals(False)
                # 手动更新一次按钮文本，因为 currentChanged 信号被阻塞了
                self.btn_start.setText("开始预测")
                # 重新计算 can_start 状态
                can_start = pred_inputs_ok and (not is_ft or ft_pred_ok)
                self.btn_start.setEnabled(can_start)

            elif is_optimize_selected_pt and not is_optimize_tab:
                self.left_tabs.blockSignals(True)
                self.left_tabs.setCurrentIndex(1)
                self.left_tabs.blockSignals(False)
                self.btn_start.setText("优化设计")
                can_start = opt_inputs_ok and (not is_ft or ft_opt_ok)
                self.btn_start.setEnabled(can_start)

    def _import_finetuned_model_predict(self):
        """导入性能预测所需的模型和Scaler (Joblib)。"""
        self.log('[INFO] 1/2: 请选择微调后的性能预测模型 (.joblib)...')
        f_model, _ = get_open_file(self, '1/2: 选择微调后的性能预测模型 (.joblib)', filter='Scikit-learn Model (*.joblib)')
        if not f_model:
            self.log('[INFO] 取消导入。')
            return

        self.log('[INFO] 2/2: 请选择配套的 Scaler 文件 (.joblib)...')
        default_dir = os.path.dirname(f_model)
        f_scaler, _ = get_open_file(self, '2/2: 选择微调后的 Scaler (.joblib)', default_dir=default_dir,
                                    filter='Scaler (*.joblib)')
        if not f_scaler:
            self.log('[INFO] 取消导入。')
            return

        # 存储到 config
        self.cfg.set_ft(self.level_name, f_model, sub='predict_model')
        self.cfg.set_ft(self.level_name, f_scaler, sub='predict_scaler')

        self.log(f'[INFO] 加筋结构(性能预测)微调模型导入完成。')
        self.load_from_config()  # 更新按钮状态
        self.on_cfg_changed()  # 更新 Inspector

    def _import_finetuned_model_optimize(self):
        """导入优化设计所需的模型和Scaler (Joblib)。"""
        self.log('[INFO] 1/2: 请选择微调后的优化设计模型 (.joblib)...')
        f_model, _ = get_open_file(self, '1/2: 选择微调后的优化设计模型 (.joblib)',
                                   filter='Scikit-learn Model (*.joblib)')
        if not f_model:
            self.log('[INFO] 取消导入。')
            return

        self.log('[INFO] 2/2: 请选择配套的 Scaler 文件 (.joblib)...')
        default_dir = os.path.dirname(f_model)
        f_scaler, _ = get_open_file(self, '2/2: 选择微调后的 Scaler (.joblib)', default_dir=default_dir,
                                    filter='Scaler (*.joblib)')
        if not f_scaler:
            self.log('[INFO] 取消导入。')
            return

        # 存储到 config
        self.cfg.set_ft(self.level_name, f_model, sub='optimize_model')
        self.cfg.set_ft(self.level_name, f_scaler, sub='optimize_scaler')

        self.log(f'[INFO] 加筋结构(优化设计)微调模型导入完成。')
        self.load_from_config()  # 更新按钮状态
        self.on_cfg_changed()  # 更新 Inspector

    def _get_predict_inputs(self) -> Optional[List[float]]:
        params = []
        for row in range(self.table_predict_inputs.rowCount()):
            item = self.table_predict_inputs.item(row, 1)
            try:
                params.append(float(item.text()))
            except (ValueError, TypeError):
                msg_warning(self, "输入错误",
                            f"性能预测参数 '{self.table_predict_inputs.item(row, 0).text()}' 格式错误或为空。")
                return None
        return params

    def _get_optimize_inputs(self) -> Optional[List[float]]:
        params = []
        for row in range(self.table_optimize_inputs.rowCount()):
            item = self.table_optimize_inputs.item(row, 1)
            try:
                params.append(float(item.text()))
            except (ValueError, TypeError):
                msg_warning(self, "输入错误",
                            f"优化设计参数 '{self.table_optimize_inputs.item(row, 0).text()}' 格式错误或为空。")
                return None
        return params

    def _predict(self):
        if self.predict_running:
            return

        inputs = self._get_predict_inputs()
        if inputs is None:
            return

        self.tabs_results.setCurrentIndex(0)
        self._clear_predict_results()

        mode = self.cfg.get_mode(self.level_name)

        # --- 必需：图像生成模型 (model_figure.pth) 始终来自预训练目录 ---
        img_model_dir = os.path.join(MODEL_DIR, self.level_name, "性能预测")
        img_model_path = os.path.join(img_model_dir, 'model_figure.pth')

        if not os.path.exists(img_model_path):
            self.log(f'[ERROR] 关键文件缺失：找不到图像生成模型: {img_model_path}')
            msg_critical(self, "文件缺失",
                         f"找不到必须的图像生成模型:\n{img_model_path}\n\n请确保 '性能预测' 预训练模型已正确安装。")
            return

        try:
            reg_model_path = ""
            reg_scaler_path = ""

            if mode == '微调':
                self.log('[INFO] 使用微调模型 (Joblib) 进行预测...')
                reg_model_path = self.cfg.get_ft(self.level_name, 'predict_model')
                reg_scaler_path = self.cfg.get_ft(self.level_name, 'predict_scaler')

                if not reg_model_path or not reg_scaler_path or not os.path.exists(
                        reg_model_path) or not os.path.exists(reg_scaler_path):
                    self.log('[ERROR] 微调模型或 Scaler 未导入，或文件不存在。')
                    msg_warning(self, "模型错误", "请先导入性能预测的微调模型和Scaler。")
                    return

            else:  # 预训练模式
                self.log('[INFO] 使用预训练模型 (Joblib) 进行预测...')
                selected_pt_mode = self.cfg.get_pt(self.level_name, 'default')
                if '性能预测' not in selected_pt_mode:
                    self.log('[ERROR] 预训练模式下未选择性能预测。')
                    msg_warning(self, "模式错误", "请在首页选择 '性能预测' 预训练模型模式。")
                    return

                model_base_dir = os.path.join(MODEL_DIR, self.level_name, "性能预测")

                reg_model_path = os.path.join(model_base_dir, 'model_shear&buckle.joblib')
                reg_scaler_path = os.path.join(model_base_dir, 'model_shear&buckle_scaler.joblib')

            # --- 启动 Worker ---
            self.predict_running = True
            self.load_from_config()  # 更新按钮

            if self._predict_worker:
                self._predict_worker.deleteLater()

            self._predict_worker = StiffenedPanelPredictWorker(
                reg_model_path=reg_model_path,
                reg_scaler_path=reg_scaler_path,
                img_model_path=img_model_path,  # 传入图像模型路径
                input_params=inputs,
                parent=self
            )

            # 连接信号
            self._predict_worker.done.connect(self._on_predict_done)
            self._predict_worker.error.connect(self._on_predict_error)
            self._predict_worker.log_signal.connect(self.log)
            self._predict_worker.finished.connect(self._finish_predict)
            self._predict_worker.start()

        except Exception as e:
            self.log(f'[CRITICAL] 无法启动预测线程：{e}\n{traceback.format_exc()}')
            msg_critical(self, "线程错误", f"无法启动预测线程:\n{e}")
            self._finish_predict()

    def _on_predict_done(self, buckling_loads, mode_shape_image):
        self.predict_running = False
        self.log('[INFO] 性能预测完成，正在更新界面...')
        self.current_buckling_loads = buckling_loads  # 缓存屈曲荷载数据 (N)
        self.current_mode_shape_image = mode_shape_image  # 缓存模态图像数据
        try:
            # 1. 更新曲线图
            self.ax_load_curve.clear()
            # 转换为 kN
            loads_kN = [load / 1000.0 for load in buckling_loads]
            self.ax_load_curve.plot(range(1, 6), loads_kN, 'bo-')
            self.ax_load_curve.set_xlabel('阶数')
            self.ax_load_curve.set_ylabel('屈曲载荷 (kN)')
            self.ax_load_curve.set_title('屈曲载荷')
            self.ax_load_curve.grid(True)
            self.canvas_load_curve.draw_idle()

            # 2. 更新模态图
            self.ax_mode_shape.clear()
            self.ax_mode_shape.imshow(mode_shape_image)
            self.ax_mode_shape.set_title('一阶屈曲模态')
            self.ax_mode_shape.axis('off')
            self.canvas_mode_shape.draw_idle()

        except Exception as e:
            self.log(f'[ERROR] 更新预测结果时失败: {e}')

    def _on_predict_error(self, tb_text: str):
        self.predict_running = False
        self.log(f'[ERROR] 性能预测线程异常：\n{tb_text}')

    def _finish_predict(self):
        # 检查是否是用户停止的
        if self.predict_running:
            self.log('[INFO] 性能预测已终止。')

        self.predict_running = False
        self._predict_worker = None
        self.load_from_config()

    def _start_optimization(self):
        if self.predict_running:  # 使用共享标志表示任何正在运行的任务
            return

        inputs = self._get_optimize_inputs()
        if inputs is None:
            return

        # 切换到结果选项卡并清除先前结果
        self.tabs_results.setCurrentIndex(1)
        self._clear_optimize_results()

        mode = self.cfg.get_mode(self.level_name)  # '预训练' 或 '微调'

        # --- 定义 model_path 和 scaler_path ---
        model_path = ""
        scaler_path = ""

        try:
            if mode == '微调':
                # --- 从 config 加载两个路径 ---
                model_path = self.cfg.get_ft(self.level_name, 'optimize_model')
                scaler_path = self.cfg.get_ft(self.level_name, 'optimize_scaler')

                if not model_path or not scaler_path or not os.path.exists(model_path) or not os.path.exists(
                        scaler_path):
                    self.log('[ERROR] 微调模式下未找到优化设计模型或Scaler，或文件不存在。')
                    msg_warning(self, "模型错误", "请先导入优化设计的微调模型和Scaler。")
                    return

                self.log(f'[INFO] 使用优化设计微调模型: {model_path}')

            else:  # 预训练
                selected_pt_mode = self.cfg.get_pt(self.level_name, 'default')
                if '优化设计' not in selected_pt_mode:
                    self.log('[ERROR] 预训练模式下未选择优化设计。')
                    msg_warning(self, "模式错误", "请在首页选择 '优化设计' 预训练模型模式。")
                    return
                # 构建指向 '优化设计' 子目录的路径
                model_base_dir = os.path.join(MODEL_DIR, self.level_name, "优化设计")

                # --- 设置预训练的完整路径 ---
                model_path = os.path.join(model_base_dir, 'model_shear&buckle_reverse_design.joblib')
                scaler_path = os.path.join(model_base_dir, 'model_shear&buckle_scaler_reverse_design.joblib')
                self.log(f'[INFO] 使用优化设计预训练模型: {model_path}')

                if not os.path.isdir(model_base_dir) or not os.path.exists(model_path) or not os.path.exists(
                        scaler_path):
                    self.log(f'[ERROR] 预训练模型目录或文件不存在: {model_base_dir}')
                    msg_critical(self, "目录错误", f"预训练模型文件缺失:\n{model_base_dir}")
                    return

            self.predict_running = True  # 设置运行标志
            self.load_from_config()  # 更新按钮状态
            self.log('[INFO] 启动加筋结构优化设计...')

            if self._optimize_worker:
                self._optimize_worker.deleteLater()  # 清理之前的 worker (如有)

            # --- 将路径传递给 Worker ---
            self._optimize_worker = StiffenedPanelOptimizeWorker(model_path, scaler_path, inputs, self)

            self._optimize_worker.update_progress.connect(self._on_optimize_update)
            self._optimize_worker.finished.connect(self._on_optimize_finished)  # 连接到记录完成的槽
            self._optimize_worker.error.connect(self._on_optimize_error)
            self._optimize_worker.log_signal.connect(self.log)
            # 使用 finished 信号进行清理，包括正常完成和错误情况
            self._optimize_worker.finished.connect(self._finish_optimize)
            self._optimize_worker.start()

        except Exception as e:
            self.log(f'[CRITICAL] 无法启动优化线程：{e}\n{traceback.format_exc()}')
            msg_critical(self, "线程错误", f"无法启动优化线程:\n{e}")
            self._finish_optimize()  # 确保清理

    def _on_optimize_update(self, generation, best_perf, best_individual,
                            gens, w1s, h1s, ds, perfs):
        """优化过程中（每代）更新图表和结果。"""
        # 缓存数据以便保存
        self.opt_plot_data = {'gens': gens, 'w1s': w1s, 'h1s': h1s, 'ds': ds, 'perfs': perfs}

        # 更新4个图表
        datas = {
            'W1': (w1s, 'W1'),
            'H1': (h1s, 'H1'),
            'd': (ds, 'd'),
            'perf': (perfs, '性能指标')
        }
        for key, (data, title) in datas.items():
            fig, ax, canvas = self.opt_canvases[key]
            ax.clear()
            ax.plot(gens, data, 'r-')
            ax.set_title(f'{title} 变化曲线')
            ax.set_xlabel('迭代次数')
            ax.grid(True)
            canvas.draw_idle()

        # 更新结果表 (实时)
        W1, H1, d = best_individual
        self.table_opt_results.setItem(0, 1, QTableWidgetItem(f"{W1:.2f}"))
        self.table_opt_results.setItem(1, 1, QTableWidgetItem(f"{H1:.2f}"))
        self.table_opt_results.setItem(2, 1, QTableWidgetItem(f"{d:.2f}"))

        # 计算对应的 Eigenvalue1
        eigenvalue_calc = 0.0
        if best_perf != float('-inf'):
            denominator = W1 + 2 * H1
            if denominator > 1e-9:
                eigenvalue_calc = best_perf * denominator

        # 根据 best_perf 是否有效来显示计算值或提示信息
        if best_perf != float('-inf'):
            self.table_opt_results.setItem(3, 1, QTableWidgetItem(f"{eigenvalue_calc:.2f}"))  # 显示计算值
            self.table_opt_results.setItem(4, 1, QTableWidgetItem(f"{best_perf:.4f}"))
        else:
            # 如果当前最优解还不满足约束 (perf = -inf)，则显示占位符
            self.table_opt_results.setItem(3, 1, QTableWidgetItem("(< 约束)"))  # 或 N/A
            self.table_opt_results.setItem(4, 1, QTableWidgetItem("N/A"))

    def _on_optimize_finished(self, gens, w1s, h1s, ds, perfs):
        self.predict_running = False
        self.log('[INFO] 优化完成。')
        # 最终更新一次图表和表格
        if gens:
            best_W1 = w1s[-1]
            best_H1 = h1s[-1]
            best_d = ds[-1]
            best_perf = perfs[-1]

            self.table_opt_results.setItem(0, 1, QTableWidgetItem(f"{best_W1:.2f}"))
            self.table_opt_results.setItem(1, 1, QTableWidgetItem(f"{best_H1:.2f}"))
            self.table_opt_results.setItem(2, 1, QTableWidgetItem(f"{best_d:.2f}"))

            R_constraint = self.table_optimize_inputs.item(0, 1).text()
            eigenvalue_calc = best_perf * (best_W1 + 2 * best_H1)

            if best_perf != float('-inf'):
                self.table_opt_results.setItem(3, 1, QTableWidgetItem(f"{eigenvalue_calc:.2f}"))
                self.table_opt_results.setItem(4, 1, QTableWidgetItem(f"{best_perf:.4f}"))
            else:
                self.table_opt_results.setItem(3, 1, QTableWidgetItem("未找到解"))
                self.table_opt_results.setItem(4, 1, QTableWidgetItem("N/A"))

    def _on_optimize_error(self, tb_text: str):
        self.predict_running = False
        self.log(f'[ERROR] 优化线程异常：\n{tb_text}')

    def _finish_optimize(self):
        # 检查是否是用户停止的
        if self.predict_running:
            self.log('[INFO] 优化设计已终止。')

        self.predict_running = False
        self._optimize_worker = None
        self.load_from_config()

    def _stop_task(self):
        """停止当前正在运行的预测或优化任务。"""
        # 立即禁用按钮，防止重复点击
        self.btn_stop.setEnabled(False)

        if not self.predict_running:
            self.log('[INFO] 没有正在运行的任务需要停止。')
            return

        stopped_something = False
        # 尝试停止预测 worker
        if self._predict_worker and self._predict_worker.isRunning():
            try:
                # self.log('[INFO] 正在请求停止性能预测...') # Worker会记录
                self._predict_worker.request_stop()
                stopped_something = True
            except Exception as e:
                self.log(f'[ERROR] 停止预测线程时出错: {e}')

        # 尝试停止优化 worker
        if self._optimize_worker and self._optimize_worker.isRunning():
            try:
                # self.log('[INFO] 正在请求停止优化设计...') # Worker会记录
                self._optimize_worker.stop() # 调用优化 worker 的停止方法
                stopped_something = True
            except Exception as e:
                self.log(f'[ERROR] 请求停止优化线程时出错: {e}')

        if not stopped_something:
            self.log('[INFO] 没有检测到正在运行的任务以供停止。')
            # 即使没有检测到，也最好重置状态以防万一
            self.predict_running = False
            self.load_from_config()

    def export_state(self):
        state = {
            'predict_inputs': [],
            'optimize_inputs': [],
            'predict_loads': self.current_buckling_loads,
            'predict_mode_shape': self.current_mode_shape_image.tolist() if self.current_mode_shape_image is not None else None,
            'optimize_results': [],
            'opt_plot_data': self.opt_plot_data
        }
        try:
            for row in range(self.table_predict_inputs.rowCount()):
                state['predict_inputs'].append(self.table_predict_inputs.item(row, 1).text())
            for row in range(self.table_optimize_inputs.rowCount()):
                state['optimize_inputs'].append(self.table_optimize_inputs.item(row, 1).text())
            for row in range(self.table_opt_results.rowCount()):
                item = self.table_opt_results.item(row, 1)
                state['optimize_results'].append(item.text() if item else "")
        except Exception as e:
            self.log(f'[WARN] 导出加筋结构状态时出错: {e}')
        return state

    def import_state(self, state: dict):
        try:
            pred_inputs = state.get('predict_inputs', [])
            if len(pred_inputs) == self.table_predict_inputs.rowCount():
                for row, val in enumerate(pred_inputs):
                    self.table_predict_inputs.setItem(row, 1, QTableWidgetItem(val))

            opt_inputs = state.get('optimize_inputs', [])
            if len(opt_inputs) == self.table_optimize_inputs.rowCount():
                for row, val in enumerate(opt_inputs):
                    self.table_optimize_inputs.setItem(row, 1, QTableWidgetItem(val))

            # 恢复屈曲载荷到曲线图
            pred_loads = state.get('predict_loads', [])
            self.current_buckling_loads = pred_loads  # 恢复缓存 (N)

            # 尝试根据列表数据重绘曲线
            try:
                # 确保数据有效
                loads = [float(v) for v in pred_loads if v is not None]
                if len(loads) == 5:
                    # 加载状态时转换为 kN
                    loads_kN = [load / 1000.0 for load in loads]
                    self.ax_load_curve.clear()
                    self.ax_load_curve.plot(range(1, 6), loads_kN, 'bo-')  # 使用 kN
                    self.ax_load_curve.set_xlabel('阶数')
                    self.ax_load_curve.set_ylabel('屈曲载荷 (kN)')  # Y轴单位
                    self.ax_load_curve.set_title('屈曲载荷')
                    self.ax_load_curve.grid(True)
                    self.canvas_load_curve.draw_idle()
            except Exception as e:
                self.log(f'[WARN] 恢复加筋结构载荷曲线失败: {e}')
            opt_results = state.get('optimize_results', [])
            if len(opt_results) == self.table_opt_results.rowCount():
                for row, val in enumerate(opt_results):
                    self.table_opt_results.setItem(row, 1, QTableWidgetItem(val))
            opt_plots = state.get('opt_plot_data', {})
            self.opt_plot_data = opt_plots
            if opt_plots and 'gens' in opt_plots:
                gens = opt_plots['gens']
                datas = {
                    'W1': (opt_plots.get('w1s', []), 'W1'),
                    'H1': (opt_plots.get('h1s', []), 'H1'),
                    'd': (opt_plots.get('ds', []), 'd'),
                    'perf': (opt_plots.get('perfs', []), '性能指标')
                }
                for key, (data, title) in datas.items():
                    if data and len(data) == len(gens):
                        fig, ax, canvas = self.opt_canvases[key]
                        ax.clear()
                        ax.plot(gens, data, 'r-')
                        ax.set_title(f'{title} 变化曲线')
                        ax.set_xlabel('迭代次数')
                        ax.grid(True)
                        canvas.draw_idle()
            # --- 恢复屈曲模态图 ---
            mode_shape_list = state.get('predict_mode_shape')
            if mode_shape_list:
                try:
                    img_data = np.array(mode_shape_list, dtype=np.float32)
                    # 检查数据是否有效 (例如，是否为3D数组 HxWxC)
                    if img_data.ndim == 3:
                        self.current_mode_shape_image = img_data
                        self.ax_mode_shape.clear()
                        self.ax_mode_shape.imshow(img_data)
                        self.ax_mode_shape.set_title('一阶屈曲模态')
                        self.ax_mode_shape.axis('off')
                        self.canvas_mode_shape.draw_idle()
                        self.log('[INFO] 已恢复屈曲模态图。')
                    else:
                        self.log('[WARN] 恢复模态图失败：数据维度不正确。')
                except Exception as e:
                    self.log(f'[WARN] 恢复屈曲模态图失败: {e}')
            self.log('[INFO] 已恢复加筋结构页状态。')
        except Exception as e:
            self.log(f'[WARN] 导入加筋结构页状态时发生错误：{e}')

# =============================
# 层级页：机身段
# =============================
class FuselagePage(_BaseLevelPage):
    """
    （当前为占位实现）
    负责根据结构和载荷参数，预测机身段性能（目前返回模拟数据）。
    """

    def __init__(self, config: ProjectConfig, write_log, on_cfg_changed):
        super().__init__(config, write_log, '机身段', on_cfg_changed)
        layout = QHBoxLayout(self)

        # --- 使用 QVBoxLayout 代替 QFormLayout ---
        left = QGroupBox('输入参数（机身段）')
        left_vbox = QVBoxLayout(left)
        left_vbox.setSpacing(10)

        # --- 创建一个 QFormLayout 专门用于参数输入 ---
        form = QFormLayout()

        self.edt_radius = QLineEdit('2.1')
        self.edt_length = QLineEdit('5.0')
        self.edt_skin_layup = QLineEdit('[45/-45/0/90]s')
        self.edt_skin_t = QLineEdit('2.2e-3')
        self.edt_frame_pitch = QLineEdit('0.5')
        self.edt_stringer_pitch = QLineEdit('0.15')
        self.edt_pressure = QLineEdit('0.074e6')
        self.edt_Nx = QLineEdit('3.0e6')
        self.edt_Ny = QLineEdit('1.0e6')
        for k, w in [("筒体半径 (m)", self.edt_radius), ("筒体长度 (m)", self.edt_length),
                     ("蒙皮铺层", self.edt_skin_layup), ("蒙皮厚度 (m)", self.edt_skin_t),
                     ("环框间距 (m)", self.edt_frame_pitch), ("纵梁间距 (m)", self.edt_stringer_pitch),
                     ("舱内压力 (Pa)", self.edt_pressure), ("轴向载荷 Nx (N/m)", self.edt_Nx),
                     ("环向载荷 Ny (N/m)", self.edt_Ny)]:
            form.addRow(k, w)

        left_vbox.addLayout(form)

        self.lbl_mode = QLabel()
        self.btn_import_ft = QPushButton('导入微调模型')
        self.btn_predict = QPushButton('开始预测')
        self.btn_stop = QPushButton('停止预测')
        self.btn_import_ft.clicked.connect(self._import_finetuned_model)
        self.btn_predict.clicked.connect(self._predict)
        self.btn_stop.clicked.connect(self._stop_predict)

        ops = QWidget()
        ops_l = QHBoxLayout(ops)
        ops_l.setContentsMargins(0, 0, 0, 0)
        ops_l.addWidget(self.btn_import_ft)
        ops_l.addWidget(self.btn_predict)
        ops_l.addWidget(self.btn_stop)

        controls_container = QWidget()
        controls_vbox = QVBoxLayout(controls_container)
        controls_vbox.setContentsMargins(0, 0, 0, 0)
        controls_vbox.setSpacing(4)
        controls_vbox.addWidget(self.lbl_mode)
        controls_vbox.addWidget(ops)

        left_vbox.addWidget(controls_container)
        left_vbox.addStretch(1)

        # 必填字段
        self._required_fields = [self.edt_radius, self.edt_length, self.edt_skin_layup, self.edt_skin_t,
                                 self.edt_frame_pitch, self.edt_stringer_pitch, self.edt_pressure, self.edt_Nx,
                                 self.edt_Ny]

        def __refresh_buttons_required__():
            self._update_predict_buttons([self.btn_import_ft], self.btn_predict, self.btn_stop, self.lbl_mode)

        for __w in self._required_fields:
            __target = __w[1] if isinstance(__w, (tuple, list)) and len(__w) >= 2 else __w
            __target.textChanged.connect(lambda *_: __refresh_buttons_required__())
        __refresh_buttons_required__()

        right = QGroupBox('结果与可视化')
        rv = QVBoxLayout(right)
        # 添加 Matplotlib 画布
        self.fig_res = Figure(figsize=(5, 4), tight_layout=True)
        self.ax_res = self.fig_res.add_subplot(111)
        self.canvas_res = FigureCanvas(self.fig_res)
        protect_canvas(self.canvas_res)
        self.toolbar_res = AutoNavToolbar(self.canvas_res, self)
        rv.addWidget(self.toolbar_res)
        rv.addWidget(self.canvas_res)
        self.res_summary = QLabel('尚无结果')
        rv.addWidget(self.res_summary)
        rv.addStretch(1)

        # 调整左右布局比例为 2:3
        layout.addWidget(left, 2)
        layout.addWidget(right, 3)

        self.current_plot_data = None  # 缓存绘图数据
        self.load_from_config()
        self._redraw_results()  # 初始化绘图区

    def _redraw_results(self, plot_data=None):
        """绘制机身段结果图（当前为模拟数据）。"""
        try:
            self.ax_res.clear()
            if plot_data and 'x' in plot_data and 'y' in plot_data:
                x = plot_data['x']
                y = plot_data['y']
                self.ax_res.bar(x, y, color='skyblue', label='模拟屈曲载荷')
                self.ax_res.set_title('模拟机身段性能')
                self.ax_res.set_xlabel('模式')
                self.ax_res.set_ylabel('模拟载荷因子')
                self.ax_res.legend()
            else:
                self.ax_res.set_title('机身段性能 (尚无结果)')
                self.ax_res.set_xlabel('模式')
                self.ax_res.set_ylabel('模拟载荷因子')
            self.ax_res.grid(True, alpha=0.3)
            self.canvas_res.draw_idle()
        except Exception as e:
            self.log(f'[WARN] 绘制机身段结果失败: {e}')

    def _clear_results(self):
        self.current_plot_data = None
        self._redraw_results(None)
        if hasattr(self, 'res_summary'):
            self.res_summary.setText('尚无结果')

    def load_from_config(self):
        """更新机身段页面的按钮状态"""
        self._update_predict_buttons(
            [self.btn_import_ft],
            self.btn_predict,
            self.btn_stop,
            self.lbl_mode
        )

    def _import_finetuned_model(self):
        # 使用自定义居中文件打开对话框
        f, _ = get_open_file(self, '导入微调模型', 'Model (*.h5 *.pt *.pth)')
        if f:
            self.cfg.set_ft(self.level_name, f, 'default')
            self.log(f'[INFO] 机身段页：已导入微调模型 -> {f}')
            self._update_predict_buttons([self.btn_import_ft], self.btn_predict, self.btn_stop, self.lbl_mode,
                                         extra_disable_when_ft_missing=[b for b in [getattr(self, 'btn_predict_band', None), getattr(self, 'btn_predict_envelope', None)] if b])
            self.on_cfg_changed()

    def _predict(self):
        if not self._can_predict():
            self._update_predict_buttons([self.btn_import_ft], self.btn_predict, self.btn_stop, self.lbl_mode,
                                         extra_disable_when_ft_missing=[b for b in
                                                                        [getattr(self, 'btn_predict_band', None),
                                                                         getattr(self, 'btn_predict_envelope', None)] if
                                                                        b])
            return
        # 原有参数校验逻辑保持
        missing = self._collect_missing([w for w in getattr(self, '_required_fields', [])]) if hasattr(self,
                                                                                                       '_required_fields') else []
        if missing:
            self._warn_missing(missing)
            return
        self.predict_running = True
        self._update_predict_buttons([self.btn_import_ft], self.btn_predict, self.btn_stop, self.lbl_mode,
                                     extra_disable_when_ft_missing=[b for b in [getattr(self, 'btn_predict_band', None),
                                                                                getattr(self, 'btn_predict_envelope',
                                                                                        None)] if b])
        self.log('[INFO] 启动FuselagePage预测…')
        self._clear_results()  # 清空上次结果

        # 线程：仅做占位计算，未来可替换为真实求解
        try:
            if hasattr(self, '_level_worker') and self._level_worker is not None:
                self._level_worker.terminate()

            def _compute():
                # 模拟计算（占位），未来可替换为真实求解
                time.sleep(1.5)  # 模拟耗时
                if self.predict_running:  # 检查是否在中途被停止
                    sim_x = list(range(1, 6))
                    sim_y = [random.uniform(0.8, 1.2) * (10 - i) for i in sim_x]
                    return {'x': sim_x, 'y': sim_y, 'summary': f'模拟预测完成。最大载荷因子: {max(sim_y):.3f}'}
                return None  # 被停止

            self._level_worker = GenericLevelPredictWorker(_compute, parent=self)
            self._level_worker.done.connect(self._on_level_done)
            self._level_worker.error.connect(self._on_level_error)
            self._level_worker.log_signal.connect(self.log)
            self._level_worker.finished.connect(lambda: self._finish_level_predict())
            self._level_worker.start()
        except Exception as e:
            self.log(f'[ERROR] 无法启动预测线程：{e}')
            self.predict_running = False
            self._update_predict_buttons([self.btn_import_ft], self.btn_predict, self.btn_stop, self.lbl_mode,
                                         extra_disable_when_ft_missing=[b for b in
                                                                        [getattr(self, 'btn_predict_band', None),
                                                                         getattr(self, 'btn_predict_envelope', None)] if
                                                                        b])

    def _on_level_done(self, payload):
        """机身段预测完成的回调函数。"""
        self.predict_running = False
        if payload and isinstance(payload, dict):
            self.current_plot_data = payload
            self._redraw_results(payload)
            summary_text = payload.get('summary', '预测完成。')
            self.res_summary.setText(summary_text)
            self.log(f'[INFO] {summary_text}')
        else:
            self.res_summary.setText('预测被取消或无结果。')
            self.log('[INFO] 预测被取消或无结果。')

        # 确保在 done 之后调用清理
        self._finish_level_predict()

    def _on_level_error(self, tb_text: str):
        self.predict_running = False
        self._clear_results()
        super()._on_level_error(tb_text)
        # 确保在 error 之后调用清理
        self._finish_level_predict()

    def _finish_level_predict(self):
        # 检查是否是用户停止的 (predict_running 标志在 done/error 之前是 True)
        if self.predict_running:
            self.log('[INFO] 机身段预测已终止。')

        # 调用基类的方法来重置按钮
        super()._finish_level_predict()

    def _stop_predict(self):
        # 立即禁用按钮
        self.btn_stop.setEnabled(False)

        if hasattr(self, '_level_worker') and self._level_worker.isRunning():
            self.log('[INFO] 正在停止机身段预测...')
            self._level_worker.terminate()  # 强制停止
        else:
            self.log('[INFO] 机身段预测任务未在运行。')

        # 强制停止(terminate)可能不会触发 'finished' 信号,
        # 所以我们手动调用清理函数来重置按钮状态
        self._finish_level_predict()

    def export_state(self):
        state = {
            'radius': self.edt_radius.text(),
            'length': self.edt_length.text(),
            'skin_layup': self.edt_skin_layup.text(),
            'skin_t': self.edt_skin_t.text(),
            'frame_pitch': self.edt_frame_pitch.text(),
            'stringer_pitch': self.edt_stringer_pitch.text(),
            'pressure': self.edt_pressure.text(),
            'Nx': self.edt_Nx.text(),
            'Ny': self.edt_Ny.text(),
            'res_summary': self.res_summary.text() if hasattr(self, 'res_summary') else '',
            'plot_data': self.current_plot_data
        }
        return state

    def import_state(self, state: dict):
        try:
            self.edt_radius.setText(str(state.get('radius', '')))
            self.edt_length.setText(str(state.get('length', '')))
            self.edt_skin_layup.setText(str(state.get('skin_layup', '')))
            self.edt_skin_t.setText(str(state.get('skin_t', '')))
            self.edt_frame_pitch.setText(str(state.get('frame_pitch', '')))
            self.edt_stringer_pitch.setText(str(state.get('stringer_pitch', '')))
            self.edt_pressure.setText(str(state.get('pressure', '')))
            self.edt_Nx.setText(str(state.get('Nx', '')))
            self.edt_Ny.setText(str(state.get('Ny', '')))
            if hasattr(self, 'res_summary'):
                self.res_summary.setText(str(state.get('res_summary', '')))

            # 恢复绘图
            plot_data = state.get('plot_data')
            if plot_data:
                self.current_plot_data = plot_data
                self._redraw_results(plot_data)
            else:
                self._clear_results()
        except Exception as e:
            self.log(f'[WARN] 导入机身段页项目状态时发生错误：{e}')

class MainWindow(QMainWindow):
    """
    应用程序的主窗口。
    管理顶部工具栏、左侧导航栏、中心页面堆栈(QStackedWidget)
    和右侧检查器(RightInspector)。
    负责处理页面路由、布局策略（三栏/两栏切换）、
    项目状态的保存/加载以及打开微调窗口。
    """
    def center_on_primary_screen(self):
        """仅用于程序首次显示：把主窗口居中到系统主屏的可用工作区中心。"""
        try:
            pr = QGuiApplication.primaryScreen().availableGeometry()
            frame = self.frameGeometry()
            frame.moveCenter(pr.center())
            self.move(frame.topLeft())
        except Exception:
            # 兜底：不移动
            pass

    def showEvent(self, event):
        try:
            if not getattr(self, '_did_initial_center', False):
                self._did_initial_center = True
                # 第一次初始运行按系统主屏居中
                self.center_on_primary_screen()
                # 用 0ms 再对齐一次，确保窗口框架计算完（避免原生窗口装饰影响）
                QTimer.singleShot(0, self.center_on_primary_screen)
            else:
                # 后续显示（例如从最小化恢复）不要强制改回主屏
                pass
        except Exception:
            pass
        return super().showEvent(event)

    def center_on_screen(self):
        """将主窗口居中到**当前屏幕**的可用区域中心。"""
        try:
            s = self.screen() or QGuiApplication.screenAt(self.frameGeometry().center()) or QGuiApplication.primaryScreen()
            geo = s.availableGeometry()
            frame = self.frameGeometry()
            frame.moveCenter(geo.center())
            self.move(frame.topLeft())
        except Exception:
            pass

    def __init__(self):
        super().__init__()
        self._did_initial_center = False
        self.setWindowTitle(APP_NAME)
        # 初始尺寸，真正的窗口尺寸将由压缩后的内容决定。
        self.resize(1280, 800)
        self.config = ProjectConfig()
        self.ft_win = None
        # 防止侧边栏调整重入的内部标志
        self._in_apply_sidebar = False
        # 顶部工具条
        self._setup_toolbar()
        # 左侧导航（隐藏：用户可拖动分割条展开）
        self.nav = QListWidget()
        for text, icon in [
            ("Dashboard", get_std_icon("go-home", self)),
            ("微结构", get_asset_icon("微结构", "image-x-generic")),
            ("层合板", get_asset_icon("层合板", "view-list-details")),
            ("加筋结构", get_asset_icon("加筋结构", "view-list-tree")),
            ("机身段", get_asset_icon("机身段", "applications-engineering")),
        ]:
            self.nav.addItem(QListWidgetItem(icon, text))
        self.nav.setCurrentRow(0)
        self.nav.currentRowChanged.connect(self._on_nav_changed)
        # 稍微放宽左侧栏，保证“Dashboard”等文本不被截断；仍保持最小宽度为0以支持隐藏
        self.nav.setIconSize(QSize(18, 18))  # 统一工具栏图标大小，保证按钮完整显示
        self.nav.setStyleSheet('QListWidget{padding-left:6px;}')  # 通过样式表隐藏工具栏分隔线/边框，避免占位影响
        # 页面区
        self.inspector = RightInspector(self.config)
        # 不为右侧栏设定固定宽度，依赖 QSplitter 的 stretch factor 动态调整其大小
        on_cfg_changed = self.inspector.update_from_config  # 统一回调
        self.pages = QStackedWidget()
        self.dashboard = DashboardPage(self.config, on_go_predict=self._go_predict_from_dashboard, on_go_finetune=self._go_finetune_from_dashboard,
                                       on_log=self.inspector.write_log, open_ft_window=self._open_finetune_window, on_cfg_changed=on_cfg_changed)
        self.page_micro = MicrostructurePage(self.config, self.inspector.write_log, on_cfg_changed)
        self.page_lami = LaminatePage(self.config, self.inspector.write_log, on_cfg_changed)
        self.page_stiff = StiffenedPanelPage(self.config, self.inspector.write_log, on_cfg_changed)
        self.page_fuse = FuselagePage(self.config, self.inspector.write_log, on_cfg_changed)
        self.pages.addWidget(self.dashboard)  # 0
        self.pages.addWidget(self.page_micro)  # 1
        self.pages.addWidget(self.page_lami)  # 2
        self.pages.addWidget(self.page_stiff)  # 3
        self.pages.addWidget(self.page_fuse)  # 4
        self.pages.setCurrentIndex(0)
        # ---- 分割布局：主页固定三栏 / 非主页两栏（可调） ----
        # 为满足“主页三栏不可移动、不可折叠，非主页两栏可自适应移动且无左侧栏”的需求：
        # 这里构建两个分割布局：
        #   1) self.home_splitter:  左(导航) / 中(页面) / 右(概览)，句柄宽度=0（不可拖动），三栏均不可折叠；
        #   2) self.work_splitter:  中(页面) / 右(概览)，初始比例8:2，可拖动调整；
        # 并通过 self._root_stack 在两种布局之间切换，彻底“删除”非主页界面的左侧栏（UI中完全不占位）。
        self.home_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.home_splitter.addWidget(self.nav)
        # 预置占位：稍后在 _apply_sidebar_policy 内动态将 self.pages / self.inspector 放入对应位置
        self._home_center_placeholder = QWidget()
        self._home_right_placeholder = QWidget()
        self.home_splitter.addWidget(self._home_center_placeholder)
        self.home_splitter.addWidget(self._home_right_placeholder)
        self.home_splitter.setHandleWidth(8)
        self.home_splitter.setCollapsible(0, False)
        self.home_splitter.setCollapsible(1, False)
        self.home_splitter.setCollapsible(2, False)
        self.home_splitter.setChildrenCollapsible(False)
        self.home_splitter.setStretchFactor(0, 5)
        self.home_splitter.setStretchFactor(1, 25)
        self.home_splitter.setStretchFactor(2, 10)
        self.work_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._work_center_placeholder = QWidget()
        self._work_right_placeholder = QWidget()
        self.work_splitter.addWidget(self._work_center_placeholder)
        self.work_splitter.addWidget(self._work_right_placeholder)
        self.work_splitter.setHandleWidth(8)
        self.work_splitter.setCollapsible(0, False)
        self.work_splitter.setCollapsible(1, False)
        self.work_splitter.setChildrenCollapsible(False)
        self.work_splitter.setStretchFactor(0, 8)
        self.work_splitter.setStretchFactor(1, 2)
        self.work_splitter.splitterMoved.connect(self._on_splitter_moved)
        # 作为主容器，在“主页三栏”和“非主页两栏”两种布局之间切换
        self._root_stack = QStackedWidget()
        self._root_stack.addWidget(self.home_splitter)  # 0: 主页三栏
        self._root_stack.addWidget(self.work_splitter)  # 1: 非主页两栏
        self.setCentralWidget(self._root_stack)
        # 宽度比例：主页 1:5:2；非主页 8:2（页面:概览）
        self._home_nav_ratio = 1
        self._home_page_ratio = 5
        self._home_insp_ratio = 2
        self._other_page_ratio = (8.0, 2.0)
        # 初次应用布局策略，并在页面切换时重用
        self._apply_sidebar_policy(self.pages.currentIndex())
        self.pages.currentChanged.connect(self._apply_sidebar_policy)
        QTimer.singleShot(0, lambda: self._apply_sidebar_policy(self.pages.currentIndex()))
        # 状态栏
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.gpu_label = QLabel('GPU: 未检测')
        self.task_label = QLabel('就绪')
        self.progress = QProgressBar()
        # 不限制进度条宽度；让状态栏自动布局
        self.progress.setVisible(False)
        self.status.addWidget(self.task_label, 1)
        self.status.addPermanentWidget(self.progress)
        self.status.addPermanentWidget(self.gpu_label)
        self._apply_qss()
        # 初始刷新概览
        self.inspector.update_from_config()

    # -- Toolbar --
    def _setup_toolbar(self):
        """设置主窗口顶部的工具栏。"""
        tb = QToolBar('主工具条')
        tb.setIconSize(QSize(18, 18))
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(tb)
        act_new = QAction(get_std_icon('document-new', self), '新建项目', self)
        act_open = QAction(get_std_icon('document-open', self), '打开项目', self)
        act_save = QAction(get_std_icon('document-save', self), '保存项目', self)
        act_help = QAction(get_std_icon('help-contents', self), '帮助文档', self)
        act_home = QAction(get_std_icon('go-home', self), '返回首页', self)
        act_home.triggered.connect(lambda: self._route_to_level('Dashboard'))
        # 连接动作
        act_new.triggered.connect(self.__new_project_action)
        act_open.triggered.connect(self.__open_project_action)
        act_save.triggered.connect(self.__save_project_action)
        act_help.triggered.connect(self.__show_help_action)
        for a in (act_new, act_open, act_save):
            tb.addAction(a)
        tb.addAction(act_home)
        tb.addSeparator()
        tb.addAction(act_help)
        # 教程与示例资源下拉（紧挨“帮助”）
        btn_examples = QToolButton()
        btn_examples.setText('教程与示例')
        btn_examples.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu_examples = QMenu(btn_examples)
        # 子菜单：入门示例
        submenu_intro = QMenu('入门示例', menu_examples)
        for lv in DashboardPage.LEVELS:
            act = submenu_intro.addAction(lv)
            act.triggered.connect(lambda _, s=lv: self.pages.widget(0)._open_intro(s))
        menu_examples.addMenu(submenu_intro)
        # 子菜单：微调数据示例下载
        submenu_dl = QMenu('微调数据示例下载', menu_examples)
        for lv in DashboardPage.LEVELS:
            act = submenu_dl.addAction(lv)
            act.triggered.connect(lambda _, s=lv: self.pages.widget(0)._download_samples(s))
        menu_examples.addMenu(submenu_dl)
        btn_examples.setMenu(menu_examples)
        tb.addWidget(btn_examples)
        lang = QComboBox()
        lang.addItems(['中文', 'English'])
        tb.addSeparator()
        tb.addWidget(QLabel('语言'))
        tb.addWidget(lang)

    def _on_nav_changed(self, row: int):
        self.pages.setCurrentIndex(min(row, self.pages.count() - 1))

    def _route_to_level(self, level: str):
        """
        按名称导航到指定层级页面。
        封装了选择QStackedWidget索引、同步左侧QListWidget导航列表和刷新
        右侧Inspector概览的逻辑。
        在设置 `nav.currentRow` 时会阻塞信号，以避免递归触发。
        """
        mapping = {'Dashboard': 0, '微结构': 1, '层合板': 2, '加筋结构': 3, '机身段': 4}
        idx = mapping.get(level, 0)
        # 更改中心页面。这将发出连接到_apply_sidebar_policy的currentChanged信号。
        self.pages.setCurrentIndex(idx)
        if hasattr(self, 'nav'):
            current_row = self.nav.currentRow()
            # 保持导航列表同步。阻塞信号以防止在设置当前行时触发_on_nav_changed（它将递归回_route_to_level）。
            if current_row != idx:
                self.nav.blockSignals(True)
                self.nav.setCurrentRow(idx)
                self.nav.blockSignals(False)
        # 如果新页面公开load_from_config钩子，则调用它
        page = self.pages.widget(idx)
        if hasattr(page, 'load_from_config'):
            page.load_from_config()
        # 无论选择哪个层级，刷新检查器
        self.inspector.update_from_config()
        '''
        当 'routing to a level' 时，总是重新应用侧边栏策略。即使页面索引没有改变（例如，当已经在仪表板上时，单击“返回首页”），
        我们仍然需要在用户手动拖动手柄后重新计算分割器大小以恢复原始比例。'_apply_sidebar_policy'方法在内部防止重入。
        '''
        self._apply_sidebar_policy(idx)

    def _go_predict_from_dashboard(self, level: str):
        self._route_to_level(level)

    def _go_finetune_from_dashboard(self, level: str, model_type: str=None):
        self._open_finetune_window(level, model_type)

    def _open_finetune_window(self, level: str, model_type: str=None):
        # 创建居中于主窗口所在屏幕的微调训练窗口。相对于主窗口独立变化。
        self.ft_win = FineTuneWindow(self.config, level, self.inspector.write_log, self.inspector.update_from_config, model_type)
        try:
            _move_window_to_screen_center(self.ft_win, self)
        except Exception:
            pass
        # 独立、置前显示
        self.ft_win.setWindowModality(Qt.WindowModality.NonModal)
        self.ft_win.raise_()
        self.ft_win.activateWindow()
        self.ft_win.show()

    # 布局策略：根据页面索引控制左侧导航栏的显示与隐藏及三栏比例
    def _apply_sidebar_policy(self, index: int):
        """
        根据页面索引切换根布局：
        - 首页(index==0)：显示“主页三栏”布局，三栏固定、不可拖动/折叠；
        - 非首页(index>0)：显示“两栏(页面/概览)”布局，初始 8:2，可拖动；
        并在切换时确保 self.pages / self.inspector 被放入当前激活的 splitter 中。
        """
        if getattr(self, '_in_apply_sidebar', False):
            return
        self._in_apply_sidebar = True
        try:
            # 确保 pages/inspector 挂载到当前应使用的 splitter
            self._ensure_splitter_hosting(index)
            if index == 0:
                # 主页：固定三栏，设置为指定比例
                self.home_splitter.setHandleWidth(8)
                self.home_splitter.setCollapsible(0, False)
                self.home_splitter.setCollapsible(1, False)
                self.home_splitter.setCollapsible(2, False)
                self.home_splitter.setChildrenCollapsible(False)
                try:
                    total_w = self.home_splitter.size().width()
                    handle_w = self.home_splitter.handleWidth()
                    avail_w = max(0, total_w - 2 * handle_w)  # 两个句柄，但句柄宽度为0时该项为0
                except Exception:
                    avail_w = 0
                nav_r, page_r, insp_r = (self._home_nav_ratio, self._home_page_ratio, self._home_insp_ratio)
                total_r = nav_r + page_r + insp_r or 1.0
                if avail_w > 0:
                    w_nav = int(avail_w * nav_r / total_r)
                    w_page = int(avail_w * page_r / total_r)
                    w_insp = avail_w - w_nav - w_page
                else:
                    # 宽度未知时给出一个稳定的分配（不会影响不可拖动特性）
                    w_nav, w_page, w_insp = (200, 800, 320)
                self.home_splitter.setSizes([max(80, w_nav), max(160, w_page), max(80, w_insp)])
            else:
                # 非主页：仅两栏，8:2，可拖动
                self.work_splitter.setHandleWidth(8)
                self.work_splitter.setCollapsible(0, False)
                self.work_splitter.setCollapsible(1, False)
                try:
                    total_w = self.work_splitter.size().width()
                    handle_w = self.work_splitter.handleWidth()
                    avail_w = max(0, total_w - handle_w)
                except Exception:
                    avail_w = 0
                page_r, insp_r = self._other_page_ratio
                total_r = page_r + insp_r or 1.0
                if avail_w > 0:
                    w_page = int(avail_w * page_r / total_r)
                    w_insp = avail_w - w_page
                else:
                    w_page, w_insp = (800, 200)
                self.work_splitter.setSizes([w_page, w_insp])
        finally:
            self._in_apply_sidebar = False

    def _ensure_splitter_hosting(self, index: int):
        """
        确保 self.pages / self.inspector 分别挂载到：
          - 首页：self.home_splitter 的索引 1/2；（索引0为导航）
          - 非首页：self.work_splitter 的索引 0/1。
        并切换根容器 self._root_stack 的当前页面；同时将 self.splitter 指向当前激活的 splitter，
        以兼容其它可能引用 self.splitter 的代码。
        """
        try:
            if index == 0:
                # 切换到主页三栏布局
                self._root_stack.setCurrentIndex(0)
                # 先彻底解绑，防止残余父子关系导致 replaceWidget 失败
                self.nav.setParent(None)
                self.pages.setParent(None)
                self.inspector.setParent(None)
                # 清空 home_splitter 的三个槽位（若有占位或旧部件）
                for slot in (0, 1, 2):
                    w = self.home_splitter.widget(slot)
                    if w is not None:
                        w.setParent(None)
                # 重新按 0/1/2 插入 nav/pages/inspector
                self.home_splitter.insertWidget(0, self.nav)
                self.home_splitter.insertWidget(1, self.pages)
                self.home_splitter.insertWidget(2, self.inspector)
                # 显式显示
                self.nav.show()
                self.pages.show()
                self.inspector.show()
                # 设置尺寸与约束
                self._set_home_sizes_exact(4)
                self.splitter = self.home_splitter
            else:
                # 切换到非主页两栏布局
                self._root_stack.setCurrentIndex(1)
                # 彻底解绑 pages/inspector，避免父子关系残留
                self.pages.setParent(None)
                self.inspector.setParent(None)
                # 清空 work_splitter 的两个槽位
                for slot in (0, 1):
                    w = self.work_splitter.widget(slot)
                    if w is not None:
                        w.setParent(None)
                # 重新插入 pages / inspector
                self.work_splitter.insertWidget(0, self.pages)
                self.work_splitter.insertWidget(1, self.inspector)
                self.pages.show()
                self.inspector.show()
                self.work_splitter.setHandleWidth(8)
                self.work_splitter.setCollapsible(0, False)
                self.work_splitter.setCollapsible(1, False)
                self.work_splitter.setChildrenCollapsible(False)
                total_w = max(1, self.work_splitter.size().width())
                handle_w = self.work_splitter.handleWidth()
                avail_w = max(1, total_w - handle_w)
                r0, r1 = (8.0, 2.0)
                s0 = max(240, int(avail_w * r0 / (r0 + r1)))
                s1 = max(120, avail_w - s0)
                self.work_splitter.setSizes([s0, s1])
                self.splitter = self.work_splitter
        except Exception:
            self.splitter = self.home_splitter if index == 0 else self.work_splitter

    def _on_splitter_moved(self, pos: int, index: int):
        """非主页两栏布局下，限制页面区不要被拖到过小，避免嵌入式 Matplotlib 画布在极端尺寸下出错。"""
        # 仅在非主页布局（work_splitter）下生效
        if getattr(self, '_root_stack', None) is None or self._root_stack.currentIndex() != 1:
            return
        sizes = self.work_splitter.sizes()
        if sizes and sizes[0] < 80:  #保护中心区域的最小宽度
            sizes[0] = 80
            self.work_splitter.setSizes(sizes)

    def __new_project_action(self):
        dlg = QDialog(self)
        dlg.setWindowTitle('新建项目')
        try:
            dlg.installEventFilter(_CenterOnMainScreenFilter(self))
        except Exception:
            pass
        layout = QVBoxLayout(dlg)
        lbl = QLabel('请输入项目名称：')
        edt = QLineEdit()
        edt.setPlaceholderText('请输入项目名称')
        layout.addWidget(lbl)
        layout.addWidget(edt)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=dlg)
        layout.addWidget(btns)
        ok_btn = btns.button(QDialogButtonBox.StandardButton.Ok)
        # 名称为空时禁用确认按钮
        if ok_btn is not None:
            ok_btn.setEnabled(False)

        def _on_text_changed(t):
            if ok_btn is not None:
                ok_btn.setEnabled(bool((t or '').strip()))
        edt.textChanged.connect(_on_text_changed)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        try:
            s = (self.windowHandle().screen() if self.windowHandle() else None) or QGuiApplication.screenAt(self.frameGeometry().center()) or QGuiApplication.primaryScreen()
            pr = s.availableGeometry()
            sz = dlg.sizeHint()
            w, h = (max(1, sz.width()), max(1, sz.height()))
            x = pr.center().x() - w // 2
            y = pr.center().y() - h // 2
            dlg.move(int(x), int(y))
        except Exception:
            pass
        if dlg.exec():
            name = edt.text().strip()
            if not name:
                return
            self.config.name = name
            self.dashboard.edt_proj.setText(self.config.name)
            self.dashboard.edt_proj.setStyleSheet('')  # 移除（可能存在的）错误高亮样式
            self.inspector.update_from_config()
            if hasattr(self, 'inspector') and hasattr(self.inspector, 'write_log'):
                self.inspector.write_log(f'[INFO] 新建项目：{self.config.name}')

    def __project_filters(self):
        return 'Project (*.json)'

    def __save_project_action(self):
        # 使用自定义居中文件保存对话框
        path, _ = get_save_file(self, '保存项目文件', self.__project_filters())
        if not path:
            return
        try:
            gather = getattr(self, '_gather_project_state', None)
            if callable(gather):
                state = gather()
            else:
                # 回退：收集最小项目状态
                state = {
                    'app': APP_NAME,
                    'version': 1,
                    'project': {
                        'name': getattr(self, "config", None).name if hasattr(self, "config") else "Project",
                        'level': getattr(self, "config", None).level if hasattr(self, "config") else "微结构",
                        'units': getattr(self, "config", None).units if hasattr(self, "config") else "SI",
                        'model_mode': getattr(self, "config", None).model_mode if hasattr(self, "config") else "预训练",
                        'pretrained_models': getattr(self, "config", None).pretrained_models if hasattr(self,
                                                                                                        "config") else {},
                        'finetuned_models': getattr(self, "config", None).finetuned_models if hasattr(self,
                                                                                                      "config") else {},
                        'finetune_image_paths': getattr(self, "config", None).finetune_image_paths if hasattr(self,
                                                                                                              "config") else [],
                        'finetune_param_file': getattr(self, "config", None).finetune_param_file if hasattr(self,
                                                                                                            "config") else "",
                    },
                    'pages': {}
                }
                if hasattr(self, 'page_micro') and hasattr(self.page_micro, 'export_state'):
                    state['pages']['microstructure'] = self.page_micro.export_state()
                if hasattr(self, 'page_lami') and hasattr(self.page_lami, 'export_state'):
                    state['pages']['laminate'] = self.page_lami.export_state()
                if hasattr(self, 'page_stiff') and hasattr(self.page_stiff, 'export_state'):
                    state['pages']['stiffened'] = self.page_stiff.export_state()
                if hasattr(self, 'page_fuse') and hasattr(self.page_fuse, 'export_state'):
                    state['pages']['fuselage'] = self.page_fuse.export_state()
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            if hasattr(self, 'inspector') and hasattr(self.inspector, 'write_log'):
                self.inspector.write_log(f'[INFO] 项目已保存：{path}')
            if hasattr(self, 'status'):
                self.status.showMessage(f'已保存项目：{os.path.basename(path)}', 5000)
        except Exception as e:
            msg_warning(self, '保存失败', f'保存项目文件失败：{e}')

    def __open_project_action(self):
        # 使用自定义居中文件打开对话框
        path, _ = get_open_file(self, '打开项目文件', self.__project_filters())
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            # 应用项目状态（包含 UI 更新）
            self._apply_project_state(state)
            # 记录日志与状态栏
            if hasattr(self, 'inspector') and hasattr(self.inspector, 'write_log'):
                self.inspector.write_log(f'[INFO] 项目已打开：{path}')
            if hasattr(self, 'status'):
                self.status.showMessage(f'已打开项目：{os.path.basename(path)}', 5000)
        except Exception as e:
            msg_warning(self, '打开失败', f'打开项目文件失败：{e}')

    def __show_help_action(self):
        """菜单栏“帮助”动作：仅打开 Manuals and Tutorials/帮助文档.pdf"""
        return open_help_pdf(self)

    def _apply_qss(self):
        self.setStyleSheet("""
        QMainWindow { background: #f7f9fc; }
        QLabel#h1 { font-size: 24px; font-weight: 800; color: #1f2937; }
        QLabel#sub { color: #4b5563; }
        QListWidget { border: 0; background: #eef2f7; }
        QListWidget::item { padding: 10px; }
        QListWidget::item:selected { background: #dbeafe; }
        QToolBar { background: #ffffff; padding: 4px; }
        QStatusBar { background: #ffffff; }
        /* 按钮微凸起风格 */
        QPushButton, QToolButton {
            padding: 6px 12px;
            border-radius: 8px;
            border: 1px solid #c7d2fe;
            background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #ffffff, stop:1 #eef2ff);
            border-bottom: 3px solid #93c5fd;
        }
        QPushButton:hover, QToolButton:hover {
            background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #f5f7ff, stop:1 #e6ecff);
        }
        QPushButton:pressed, QToolButton:pressed {
            border-bottom: 1px solid #93c5fd;
            padding-top: 7px;
            padding-bottom: 5px;
        }
        /*
         * QSS重置：
         * 全局的 QToolButton 样式会影响 QFileDialog (非原生) 中的标准图标按钮。
         * 此处为 QFileDialog 内的 QToolButton 恢复透明背景和无边框样式，
         * 以确保文件对话框的 "返回"、"新建文件夹" 等图标能正常显示。
         */
        QFileDialog QToolButton {
            background: transparent;
            border: none;
            padding: 4px;
        }
        QFileDialog QToolButton:hover {
            background: #f3f6fb;
        }
        QFileDialog QToolButton:pressed {
            background: #e8eef7;
        }
        """)

    def _gather_project_state(self):
        state = {
            'app': APP_NAME,
            'version': 1,
            'project': {
                'name': self.config.name,
                'level': self.config.level,
                'units': self.config.units,
                'model_mode': self.config.model_mode,
                'pretrained_models': dict(self.config.pretrained_models),
                'finetuned_models': dict(self.config.finetuned_models),
                'finetune_image_paths': list(self.config.finetune_image_paths),
                'finetune_param_file': self.config.finetune_param_file,
                'finetune_output_file': getattr(self.config, 'finetune_output_file', ''),
            },
            'pages': {}
        }
        try:
            if hasattr(self, 'page_micro') and hasattr(self.page_micro, 'export_state'):
                state['pages']['microstructure'] = self.page_micro.export_state()
        except Exception as e:
            self.inspector.write_log(f'[WARN] 导出微结构页状态失败：{e}')
        try:
            if hasattr(self, 'page_lami') and hasattr(self.page_lami, 'export_state'):
                state['pages']['laminate'] = self.page_lami.export_state()
        except Exception as e:
            self.inspector.write_log(f'[WARN] 导出层合板页状态失败：{e}')
        try:
            if hasattr(self, 'page_stiff') and hasattr(self.page_stiff, 'export_state'):
                state['pages']['stiffened'] = self.page_stiff.export_state()
        except Exception as e:
            self.inspector.write_log(f'[WARN] 导出加筋结构页状态失败：{e}')
        try:
            if hasattr(self, 'page_fuse') and hasattr(self.page_fuse, 'export_state'):
                state['pages']['fuselage'] = self.page_fuse.export_state()
        except Exception as e:
            self.inspector.write_log(f'[WARN] 导出机身段页状态失败：{e}')
        return state

    def _apply_project_state(self, state: dict):
        if not isinstance(state, dict):
            raise ValueError('项目文件格式不正确')
        pj = state.get('project', {})
        # 恢复基本配置
        self.config.name = pj.get('name', self.config.name)
        self.config.level = pj.get('level', self.config.level)
        self.config.units = pj.get('units', self.config.units)
        self.config.model_mode = pj.get('model_mode', self.config.model_mode)
        self.config.pretrained_models = dict(pj.get('pretrained_models', self.config.pretrained_models))
        self.config.finetuned_models = dict(pj.get('finetuned_models', self.config.finetuned_models))
        self.config.finetune_image_paths = list(pj.get('finetune_image_paths', self.config.finetune_image_paths))
        self.config.finetune_param_file = pj.get('finetune_param_file', self.config.finetune_param_file)
        setattr(self.config, 'finetune_output_file', pj.get('finetune_output_file', getattr(self.config, 'finetune_output_file', '')))
        # 同步到 Dashboard 基本控件
        self.dashboard.edt_proj.setText(self.config.name)
        # 选中层级
        idx = self.dashboard.cmb_level.findText(self.config.level)
        if idx >= 0:
            self.dashboard.cmb_level.setCurrentIndex(idx)
        # 单位
        idx = self.dashboard.cmb_units.findText(self.config.units)
        if idx >= 0:
            self.dashboard.cmb_units.setCurrentIndex(idx)
        # 模式
        idx = self.dashboard.cmb_mode.findText(self.config.model_mode)
        if idx >= 0:
            self.dashboard.cmb_mode.setCurrentIndex(idx)
        # 预训练模型（微结构/其他）
        if self.config.level == '微结构':
            if self.dashboard.cmb_pretrained_A:
                i = self.dashboard.cmb_pretrained_A.findText(self.config.get_pt('微结构', 'CNN'))
                if i >= 0:
                    self.dashboard.cmb_pretrained_A.setCurrentIndex(i)
            if self.dashboard.cmb_pretrained_B:
                i = self.dashboard.cmb_pretrained_B.findText(self.config.get_pt('微结构', 'ICNN'))
                if i >= 0:
                    self.dashboard.cmb_pretrained_B.setCurrentIndex(i)
        elif self.dashboard.cmb_pretrained_single:
            i = self.dashboard.cmb_pretrained_single.findText(self.config.get_pt(self.config.level, 'default'))
            if i >= 0:
                self.dashboard.cmb_pretrained_single.setCurrentIndex(i)
        # 微调数据选择回显，长路径截断显示并设置完整路径提示
        if self.config.finetune_image_paths:
            try:
                img_path = self.config.finetune_image_paths[0]
                self.dashboard.lbl_img_cnt.set_full_text(img_path)
            except Exception:
                self.dashboard.lbl_img_cnt.set_full_text(self.config.finetune_image_paths[0])
        if self.config.finetune_param_file:
            try:
                self.dashboard.lbl_tbl_path.set_full_text(self.config.finetune_param_file)
            except Exception:
                self.dashboard.lbl_tbl_path.set_full_text(self.config.finetune_param_file)
        outf = getattr(self.config, 'finetune_output_file', '')
        if outf and hasattr(self.dashboard, 'lbl_out_path'):
            try:
                self.dashboard.lbl_out_path.set_full_text(outf)
            except Exception:
                self.dashboard.lbl_out_path.set_full_text(outf)
        # 刷新按钮状态
        self.dashboard._update_ft_go_button()
        # 分页状态恢复
        pages = state.get('pages', {})
        try:
            if 'microstructure' in pages and hasattr(self.page_micro, 'import_state'):
                self.page_micro.import_state(pages.get('microstructure', {}))
        except Exception as e:
            self.inspector.write_log(f'[WARN] 恢复微结构页状态失败：{e}')
        try:
            if 'laminate' in pages and hasattr(self.page_lami, 'import_state'):
                self.page_lami.import_state(pages.get('laminate', {}))
        except Exception as e:
            self.inspector.write_log(f'[WARN] 恢复层合板页状态失败：{e}')
        try:
            if 'stiffened' in pages and hasattr(self.page_stiff, 'import_state'):
                self.page_stiff.import_state(pages.get('stiffened', {}))
        except Exception as e:
            self.inspector.write_log(f'[WARN] 恢复加筋结构页状态失败：{e}')
        try:
            if 'fuselage' in pages and hasattr(self.page_fuse, 'import_state'):
                self.page_fuse.import_state(pages.get('fuselage', {}))
        except Exception as e:
            self.inspector.write_log(f'[WARN] 恢复机身段页状态失败：{e}')
        # 切换到保存时的层级
        self._route_to_level(self.config.level)
        # 刷新右侧概览
        self.inspector.update_from_config()
        # 状态栏提示
        self.status.showMessage('项目已加载', 5000)

    def _set_home_sizes_exact(self, retries: int = 4):
        """
        严格设置主页三栏布局的比例 (1:5:2)。
        如果窗口宽度尚未就绪 (为0)，则使用 QTimer 延迟重试。
        """
        self.home_splitter.setHandleWidth(8)
        for i in (0, 1, 2):
            self.home_splitter.setCollapsible(i, False)
        self.home_splitter.setChildrenCollapsible(False)
        self.home_splitter.setStretchFactor(0, 5)
        self.home_splitter.setStretchFactor(1, 25)
        self.home_splitter.setStretchFactor(2, 10)
        total_w = self.home_splitter.size().width()
        handle_w = self.home_splitter.handleWidth()
        avail = max(0, total_w - 2 * handle_w)
        if (avail <= 0 or total_w <= 0) and retries > 0:
            QTimer.singleShot(0, lambda: self._set_home_sizes_exact(retries - 1))
            return
        r0, r1, r2 = (1, 5, 2)
        tr = r0 + r1 + r2
        s0 = max(120, int(avail * r0 / tr)) if avail > 0 else 200
        s1 = max(240, int(avail * r1 / tr)) if avail > 0 else 800
        s2 = max(160, avail - s0 - s1) if avail > 0 else 320
        self.home_splitter.setSizes([s0, s1, s2])
        if retries > 0:
            QTimer.singleShot(0, lambda: self._set_home_sizes_exact(retries - 1))

def main():
    """应用程序主入口点。"""
    app = QApplication(sys.argv)
    icon = QIcon(os.path.join(ASSETS_DIR, "logo.ico"))
    app.setWindowIcon(icon)
    # --- GPU 初始化 ---
    try:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if physical_devices:
            print(f'Found {len(physical_devices)} GPU(s).')
            # 设置显存按需增长，避免一次性占满
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            gpu_status = f'GPU: {physical_devices[0].name}'
        else:
            print('No GPU hardware devices available, using CPU.')
            gpu_status = 'GPU: Not Found (Using CPU)'
    except Exception as e:
        print(f'Error during GPU initialization: {e}')
        gpu_status = 'GPU: Error'
    # 如需兼容老模型，可关闭 eager（默认保持开启）
    # tf.compat.v1.disable_eager_execution()
    tf.random.set_seed(1)
    # --- GPU 初始化结束 ---
    w = MainWindow()
    w.setWindowIcon(icon)
    w.gpu_label.setText(gpu_status)  # 更新状态栏 GPU 信息
    w.show()
    w.center_on_screen()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()