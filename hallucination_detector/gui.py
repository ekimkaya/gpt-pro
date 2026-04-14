"""Desktop GUI for the hallucination detector.

Stdlib-only (Tkinter) so the wheel ships without extra dependencies and
can be packaged cleanly with PyInstaller into Windows/Mac/Linux installers.

Run with:
    hd-gui
or:
    python -m hallucination_detector.gui

The GUI is intentionally minimal — three panels (draft, controls,
findings) — because the audience is attorneys, not engineers.
"""

from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from .backends.courtlistener import CourtListenerClient
from .backends.multi import MultiJurisdictionValidator
from .backends.restatement import RestatementCorpus
from .detector import HallucinationDetector
from .models import Severity
from .network_policy import format_report, report
from .policy import FirmPolicy
from .providers import StubProvider


_SEV_COLORS = {
    Severity.CRITICAL: "#cc0000",
    Severity.HIGH: "#cc6600",
    Severity.MEDIUM: "#996600",
    Severity.LOW: "#666666",
    Severity.INFO: "#444444",
}


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Hallucination Detector")
        root.geometry("1100x720")

        self._build_menu()
        self._build_layout()

        self._validator = MultiJurisdictionValidator(
            backends=[CourtListenerClient(), RestatementCorpus()]
        )
        self._policy = FirmPolicy()

    # ----- layout -------------------------------------------------------- #

    def _build_menu(self) -> None:
        menu = tk.Menu(self.root)
        self.root.config(menu=menu)
        f = tk.Menu(menu, tearoff=0)
        f.add_command(label="Open draft…", command=self._open_file)
        f.add_command(label="Save report (JSON)…", command=self._save_report)
        f.add_separator()
        f.add_command(label="Quit", command=self.root.destroy)
        menu.add_cascade(label="File", menu=f)

        h = tk.Menu(menu, tearoff=0)
        h.add_command(label="Show network policy", command=self._show_network)
        h.add_command(label="About", command=self._about)
        menu.add_cascade(label="Help", menu=h)

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        # Header
        header = ttk.Frame(outer)
        header.pack(fill=tk.X)
        ttk.Label(
            header,
            text="Paste or open the attorney's draft. Click Validate.",
            font=("TkDefaultFont", 11, "bold"),
        ).pack(side=tk.LEFT)

        # Controls
        controls = ttk.Frame(outer)
        controls.pack(fill=tk.X, pady=(8, 8))
        self.offline_var = tk.BooleanVar(value=False)
        self.quotes_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            controls, text="Fully offline (no API calls)",
            variable=self.offline_var,
        ).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Checkbutton(
            controls, text="Verify quoted language",
            variable=self.quotes_var,
        ).pack(side=tk.LEFT, padx=(0, 12))
        self.run_btn = ttk.Button(controls, text="Validate draft", command=self._run)
        self.run_btn.pack(side=tk.RIGHT)

        # Two-pane body
        body = ttk.PanedWindow(outer, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True)

        left = ttk.LabelFrame(body, text="Draft", padding=4)
        self.draft = scrolledtext.ScrolledText(left, wrap=tk.WORD, font=("TkFixedFont", 10))
        self.draft.pack(fill=tk.BOTH, expand=True)
        body.add(left, weight=2)

        right = ttk.LabelFrame(body, text="Findings", padding=4)
        self.findings = scrolledtext.ScrolledText(
            right, wrap=tk.WORD, font=("TkDefaultFont", 10), state=tk.DISABLED,
        )
        self.findings.pack(fill=tk.BOTH, expand=True)
        body.add(right, weight=2)

        # Status bar
        self.status = tk.StringVar(value="Ready.")
        ttk.Label(outer, textvariable=self.status, anchor=tk.W).pack(fill=tk.X, pady=(8, 0))

        # Severity color tags
        for sev, color in _SEV_COLORS.items():
            self.findings.tag_configure(sev.value, foreground=color)
        self.findings.tag_configure("header", font=("TkDefaultFont", 11, "bold"))
        self.findings.tag_configure("blocked", foreground="white", background="#cc0000")
        self.findings.tag_configure("clean", foreground="white", background="#2a7a2a")

    # ----- actions ------------------------------------------------------- #

    def _open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Open draft", filetypes=[("Text", "*.txt *.md"), ("All", "*.*")],
        )
        if not path:
            return
        with open(path) as f:
            self.draft.delete("1.0", tk.END)
            self.draft.insert("1.0", f.read())
        self.status.set(f"Loaded {os.path.basename(path)}")

    def _save_report(self) -> None:
        if not getattr(self, "_last_report", None):
            messagebox.showinfo("Save report", "Run a validation first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        import json
        with open(path, "w") as f:
            json.dump(self._last_report.as_dict(), f, indent=2)
        self.status.set(f"Saved report to {path}")

    def _show_network(self) -> None:
        opinion_fetcher = None
        if not self.offline_var.get() and self.quotes_var.get():
            for b in self._validator.backends:
                if hasattr(b, "fetch_opinion_text"):
                    opinion_fetcher = b.fetch_opinion_text
                    break
        d = HallucinationDetector(
            creator=StubProvider(lambda p, s: "", model="unused"),
            validator=self._validator,
            opinion_fetcher=opinion_fetcher,
            score_uncertainty=False,
        )
        text = format_report(report(d))
        messagebox.showinfo("Network policy", text)

    def _about(self) -> None:
        messagebox.showinfo(
            "About",
            "Hallucination Detector for legal AI.\n"
            "Runs entirely on this computer.\n"
            "No telemetry, no cloud, no phone-home.",
        )

    def _run(self) -> None:
        text = self.draft.get("1.0", tk.END).strip()
        if not text:
            self.status.set("Paste a draft first.")
            return
        self.run_btn.config(state=tk.DISABLED)
        self.status.set("Validating…")
        # Run in a worker thread so the GUI stays responsive.
        threading.Thread(target=self._run_worker, args=(text,), daemon=True).start()

    def _run_worker(self, text: str) -> None:
        try:
            opinion_fetcher = None
            if not self.offline_var.get() and self.quotes_var.get():
                for b in self._validator.backends:
                    if hasattr(b, "fetch_opinion_text"):
                        opinion_fetcher = b.fetch_opinion_text
                        break
            detector = HallucinationDetector(
                creator=StubProvider(lambda p, s: "", model="unused"),
                validator=self._validator,
                opinion_fetcher=opinion_fetcher,
                policy=self._policy,
                score_uncertainty=False,
            )
            report_obj = detector.check_draft(
                text,
                use_api=not self.offline_var.get(),
                use_quotes=self.quotes_var.get(),
            )
            self.root.after(0, self._render, report_obj)
        except Exception as e:  # noqa: BLE001
            self.root.after(0, self._error, str(e))

    def _render(self, report_obj) -> None:
        self._last_report = report_obj
        self.findings.config(state=tk.NORMAL)
        self.findings.delete("1.0", tk.END)
        if report_obj.blocked:
            self.findings.insert(
                tk.END,
                f"  BLOCKED — {report_obj.block_reason or 'severity threshold exceeded'}  \n\n",
                "blocked",
            )
        elif not report_obj.findings:
            self.findings.insert(tk.END, "  No issues detected.  \n\n", "clean")
        self.findings.insert(
            tk.END, f"Max severity: {report_obj.max_severity.value}\n\n", "header"
        )
        self.findings.insert(tk.END, "Findings:\n", "header")
        for f in report_obj.findings:
            self.findings.insert(
                tk.END, f"  [{f.severity.value:>8}] {f.layer}: {f.message}\n",
                f.severity.value,
            )
        self.findings.insert(tk.END, "\nCitations:\n", "header")
        for c in report_obj.citations:
            self.findings.insert(
                tk.END,
                f"  [{c.status.value:>16}] {c.kind.value:<13} {c.normalized}\n",
            )
        self.findings.config(state=tk.DISABLED)
        self.run_btn.config(state=tk.NORMAL)
        self.status.set(
            f"Done. {len(report_obj.findings)} finding(s); "
            f"{len(report_obj.citations)} citation(s)."
        )

    def _error(self, msg: str) -> None:
        self.run_btn.config(state=tk.NORMAL)
        self.status.set("Error.")
        messagebox.showerror("Validation failed", msg)


def main() -> int:
    try:
        root = tk.Tk()
    except tk.TclError as e:
        print(f"Cannot open display: {e}", file=__import__("sys").stderr)
        return 1
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
