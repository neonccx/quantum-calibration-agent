"""Render the real CLI components with labelled demo data; no server connection."""

import argparse
from pathlib import Path
from rich.console import Console
from qmagent.terminal_design import TerminalDesign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path)
    args = parser.parse_args()
    console = Console(width=86, record=True)
    ui = TerminalDesign(console)
    ui.banner({"version": "0.4", "session_id": "design-preview · illustrative data",
               "directory": "<server session directory>",
               "settings": {"policy": "hf", "model": "Nanbeige4.2-3B", "adapter": None}}, remote=True)
    console.print(ui.text(ui.prompt() + "Show calibration progress"))
    ui.status({"status": "active", "experiment_count": 4, "max_steps": 30,
               "consecutive_iq_passes": 0, "completed_stages": ["S21", "Spectroscopy"],
               "final_state": {"readout_frequency_GHz": 6.51, "qubit_frequency_GHz": 5.03},
               "final_iq_gate": None, "reason": None})
    console.print(ui.text("Illustrative interface data, not a measured result."))
    console.print(ui.text(ui.prompt() + "Explain the next step"))
    ui.section("Nanbeige · explanation, not an experiment record")
    console.print(ui.text("Calibrate the pi-pulse amplitude, then check detuning.\nUse /plan to preview. No experiment runs before confirmation."))
    if args.html:
        if args.html.exists():
            raise ValueError("Choose a new preview file")
        console.save_html(str(args.html), inline_styles=True)


if __name__ == "__main__":
    main()
