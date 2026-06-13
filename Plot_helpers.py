"""Grafiket hapen nje nga nje — mbyll me X, pastaj vjen tjetri."""

from __future__ import annotations

import gc
import importlib.util
from pathlib import Path

_CONFIGURED = False


def _qt_available() -> bool:
    for module in ("PyQt5", "PySide2", "PySide6"):
        if importlib.util.find_spec(module) is not None:
            return True
    return False


def configure_matplotlib() -> None:
    """Thirret para seaborn/matplotlib.pyplot."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    import matplotlib
    import matplotlib.pyplot as plt

    backends: list[str] = []
    if _qt_available():
        backends.extend(["Qt5Agg", "QtAgg"])
    backends.append("TkAgg")

    for backend in backends:
        try:
            matplotlib.use(backend, force=True)
            test_fig = plt.figure()
            plt.close(test_fig)
            break
        except Exception:
            continue

    plt.ioff()
    _CONFIGURED = True


def save_plot(path: Path, dpi: int = 150) -> None:
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.gcf()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.show(block=True)
    plt.close(fig)
    gc.collect()


def cleanup_matplotlib() -> None:
    import matplotlib.pyplot as plt

    plt.close("all")
    gc.collect()


save_show_close = save_plot

