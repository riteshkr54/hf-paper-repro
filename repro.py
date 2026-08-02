#!/usr/bin/env python3
"""Reproduction of ICML 2026 paper #4580 — Possibilistic Predictive Uncertainty for Deep Learning
(DAPPr; arXiv 2605.00600, https://openreview.net/forum?id=NhL0mBNFk4).

Fixed run contract: `python repro.py` reads config.json at the branch tip and dispatches:
  task=claim1_audit | cifar | fedl | fgvc | fig5 | obqa
Official code audited/adapted:
  https://github.com/MaxwellYaoNi/DAPPr (dappr.py, edl.py, utils/, splits)
  https://github.com/TaeseongYoon/F-EDL (fedl/)
"""
import json, sys, importlib


def main():
    with open("config.json") as f:
        config = json.load(f)
    task = config["task"]
    print(f"=== DAPPr repro | task={task} | config={json.dumps(config)} ===", flush=True)
    if task == "claim1_audit":
        import modules.audit_claim1 as m
    elif task == "cifar":
        import modules.cifar_train as m
    elif task == "fedl":
        import modules.fedl_run as m
    elif task == "fgvc":
        import modules.fgvc_train as m
    elif task == "fig5":
        import modules.fig5 as m
    elif task == "obqa":
        import modules.obqa as m
    else:
        raise ValueError(f"unknown task {task}")
    result = m.run(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
