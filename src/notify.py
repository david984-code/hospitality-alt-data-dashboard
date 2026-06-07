"""Weekly regime-change watcher: email a summary when the signal state flips.

Run:
  uv run python -m src.notify              # email only if the regime/alerts changed
  uv run python -m src.notify --always     # always email (useful for a first test)
  uv run python -m src.notify --dry-run    # print the email, never send or persist

SMTP credentials come from the environment (never hardcoded):
  SMTP_HOST, SMTP_PORT (default 587), SMTP_USER, SMTP_PASSWORD, NOTIFY_TO
For Gmail, SMTP_USER is your address and SMTP_PASSWORD is an App Password.
"""

from __future__ import annotations

import argparse
import json
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

import config
from src import pipeline

_STATE_PATH = config.DATA_DIR / "notify_state.json"


def current_state(res: pipeline.PipelineResult) -> dict:
    """Distill the pipeline result down to the regime state we watch for changes."""
    latest = res.signals.monthly.iloc[-1]
    alerts = []
    if "alert" in res.anomalies:
        alerts = sorted(res.anomalies.loc[res.anomalies["alert"] == "YES", "signal"].tolist())
    return {
        "as_of": str(res.signals.monthly.index[-1]),
        "gate": "ON" if latest["gate"] > 0 else "OFF",
        "tsa_yoy": round(float(latest["tsa_yoy"]), 1),
        "tsa_accel": round(float(latest["tsa_accel"]), 2),
        "alerts": alerts,
    }


def load_last_state(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2))


def has_changed(prev: dict | None, cur: dict) -> bool:
    """A change = the gate flipped ON/OFF, or the set of active anomaly alerts changed."""
    if prev is None:
        return True
    return prev.get("gate") != cur["gate"] or prev.get("alerts") != cur["alerts"]


def compose(prev: dict | None, cur: dict) -> tuple[str, str]:
    if prev is not None and prev["gate"] != cur["gate"]:
        tag = f"{prev['gate']} -> {cur['gate']}"
    else:
        tag = cur["gate"]
    subject = f"[Hospitality Alt-Data] Gate {tag} (as of {cur['as_of']})"
    body = "\n".join(
        [
            f"Signal gate: {cur['gate']}",
            f"TSA YoY: {cur['tsa_yoy']}%   acceleration (gate driver): {cur['tsa_accel']}",
            f"Active anomaly alerts: {', '.join(cur['alerts']) or 'none'}",
            "",
            "Dashboard: run `uv run streamlit run app.py`.",
            "Research / monitoring tool — not investment advice.",
        ]
    )
    return subject, body


def send_email(subject: str, body: str) -> None:
    user = os.environ["SMTP_USER"]
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = os.getenv("NOTIFY_TO", user)
    msg.set_content(body)
    with smtplib.SMTP(os.environ["SMTP_HOST"], int(os.getenv("SMTP_PORT", "587"))) as server:
        server.starttls()
        server.login(user, os.environ["SMTP_PASSWORD"])
        server.send_message(msg)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Email a regime-change alert if the state changed")
    ap.add_argument("--always", action="store_true", help="email even if nothing changed")
    ap.add_argument("--dry-run", action="store_true", help="print only; never send or persist")
    ap.add_argument("--skip-trends", action="store_true", help="stub Google Trends (CI)")
    args = ap.parse_args(argv)

    res = pipeline.run(skip_trends=args.skip_trends)
    cur = current_state(res)
    prev = load_last_state(_STATE_PATH)
    subject, body = compose(prev, cur)
    print(subject, body, sep="\n")

    if args.dry_run:
        return 0
    if has_changed(prev, cur) or args.always:
        send_email(subject, body)
        print("[notify] email sent")
    else:
        print("[notify] no regime change; no email sent")
    save_state(_STATE_PATH, cur)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
