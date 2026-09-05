"""End-to-end streaming evidence; use the saved SSH profile or a server-local service."""
import argparse
import json
from pathlib import Path
import time

from qmagent.storage import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.local:
        from qmagent.service import AgentService
        from qmagent.session import CalibrationSession
        from qmagent.settings import load_settings
        service = AgentService(args.home, CalibrationSession.create(args.home, load_settings(args.home)))
    else:
        from qmagent.remote import RemoteService, load_remote
        service = RemoteService(load_remote(args.home), on_wait=lambda text: print(text, flush=True))
    cases = [
        ("long", "Write an educational explanation in English of the single-qubit calibration workflow. "
         "Use eight numbered paragraphs of approximately 60 words each (about 480 words total). "
         "Cover resonator S21, spectroscopy, pi amplitude, Ramsey, T1, echo, IQ discrimination, and "
         "independent validation. Explain only; do not propose or execute any operation."),
        ("chinese", "请用中文两句话说明 IQ 分割是什么，只解释概念。"),
    ]
    evidence = {"info": service.info(), "cases": []}
    try:
        for name, prompt in cases:
            service.clear_chat()
            chunks, times = [], []
            started = time.monotonic()
            def receive(text):
                times.append(time.monotonic() - started)
                chunks.append(text)
                print(text, end="", flush=True)
            result = service.ask(text=prompt, on_text=receive)
            print(flush=True)
            case = {"name": name, "prompt": prompt, "result": result,
                    "wall_seconds": time.monotonic() - started,
                    "chunk_times_seconds": times, "chunks": chunks,
                    "stream_matches_final": "".join(chunks) == result["message"]}
            evidence["cases"].append(case)
            atomic_json(args.output / "results.json", evidence, replace=True)
            assert case["stream_matches_final"] and len(chunks) > 1
            assert times[0] < times[-1] < case["wall_seconds"]
            assert result["status"]["experiment_count"] == 0 and result["proposal"] is None
            assert not result["generation"]["hit_output_limit"]
            if name == "long":
                assert result["generation"]["generated_tokens"] > 256
            print(json.dumps({"case": name, "chunks": len(chunks), "first_text": times[0],
                              "wall": case["wall_seconds"], "generation": result["generation"]}), flush=True)
        evidence["report"] = service.report()
        atomic_json(args.output / "results.json", evidence, replace=True)
    finally:
        service.close()


if __name__ == "__main__":
    main()
