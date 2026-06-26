"""Terminal progress for multi-episode evaluation."""

from __future__ import annotations

import sys
import time


class ProgressBar:
    """Minimal progress bar; uses ``tqdm`` when installed."""

    def __init__(
        self,
        desc: str,
        *,
        total: int | None = None,
        disable: bool = False,
        unit: str = "run",
    ) -> None:
        self.desc = desc
        self.total = total
        self.disable = disable or not sys.stderr.isatty()
        self.unit = unit
        self.n = 0
        self._postfix: dict[str, object] = {}
        self._tqdm = None
        self._last_line_len = 0
        self._started = time.monotonic()

        if self.disable:
            return

        try:
            from tqdm import tqdm

            self._tqdm = tqdm(
                total=total,
                desc=desc,
                unit=unit,
                dynamic_ncols=True,
                file=sys.stderr,
                leave=True,
            )
        except ImportError:
            self._render()

    def set_postfix(self, **kwargs: object) -> None:
        self._postfix = kwargs
        if self._tqdm is not None:
            self._tqdm.set_postfix(**kwargs, refresh=True)
        elif not self.disable:
            self._render()

    def update(self, n: int = 1) -> None:
        self.n += n
        if self.disable:
            return
        if self._tqdm is not None:
            self._tqdm.update(n)
            if self._postfix:
                self._tqdm.set_postfix(**self._postfix, refresh=False)
        else:
            self._render()

    def _render(self) -> None:
        elapsed = max(time.monotonic() - self._started, 1e-6)
        rate = self.n / elapsed
        extra = ""
        if self._postfix:
            extra = "  " + "  ".join(f"{k}={v}" for k, v in self._postfix.items())

        if self.total and self.total > 0:
            frac = min(self.n / self.total, 1.0)
            width = 28
            filled = int(width * frac)
            bar = "=" * filled + ">" + " " * max(width - filled - 1, 0)
            line = (
                f"\r{self.desc} [{bar}] {self.n}/{self.total} {self.unit} "
                f"({rate:.1f}/s){extra}"
            )
        else:
            line = f"\r{self.desc} {self.n} {self.unit} ({rate:.1f}/s){extra}"

        pad = max(self._last_line_len - len(line), 0)
        sys.stderr.write(line + " " * pad)
        sys.stderr.flush()
        self._last_line_len = len(line)

    def close(self) -> None:
        if self.disable:
            return
        if self._tqdm is not None:
            self._tqdm.close()
        else:
            sys.stderr.write("\n")
            sys.stderr.flush()
