"""Save a bounded PNG preview received over the authenticated Agent channel."""

import base64
import hashlib
import html
import json
from pathlib import Path
import tempfile


def save_preview(result, home):
    preview = result.get("preview")
    if preview is None:
        return None
    if (not isinstance(preview, dict) or preview.get("name") != "iq_report.png"
            or not isinstance(preview.get("base64"), str) or len(preview["base64"]) > 1400000):
        raise ValueError("Invalid report preview")
    data = base64.b64decode(preview["base64"], validate=True)
    if (len(data) > 1024 * 1024 or not data.startswith(b"\x89PNG\r\n\x1a\n")
            or hashlib.sha256(data).hexdigest() != preview.get("sha256")):
        raise ValueError("Report image failed integrity check")
    root = Path(home) / "reports"
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="iq-", dir=root))
    (directory / "iq_report.png").write_bytes(data)
    status = result.get("status", {})
    details = html.escape(json.dumps({"controller_status": status.get("status"),
                                     "experiments": status.get("experiment_count"),
                                     "iq_passes": status.get("consecutive_iq_passes"),
                                     "server_report": result.get("directory")}, ensure_ascii=False, indent=2))
    (directory / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Recorded IQ report</title>'
        '<style>body{max-width:1100px;margin:24px auto;font:16px system-ui;background:#f5f6fa;color:#182033}'
        'img{display:block;width:100%;height:auto}pre{white-space:pre-wrap}</style>'
        '<h1>IQ measurement report · SIMULATED</h1><p>Rendered from recorded shots, not invented by a language model. Policy and acceptance status are shown in the figure.</p>'
        '<pre>' + details + '</pre><img src="iq_report.png" alt="Six-panel IQ discrimination analysis">',
        encoding="utf-8")
    return directory / "index.html"
