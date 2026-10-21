"""Run run_glue_prune.py unchanged, but print every Trainer.log() dict as one JSON line.

CoFiTrainer.train() never fires on_train_begin, so HF's ProgressCallback never prints the training loss;
the loss only lands in trainer.state.log_history. This wrapper is measurement-only harness code: it does not
change any computation. It also dumps state.log_history to <output_dir>/log_history.json at the end.
Usage: python3 log_wrapper.py <run_glue_prune.py args...>   (run from the code directory)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
import trainer.trainer as cofi_trainer  # noqa: E402

_orig_log = cofi_trainer.CoFiTrainer.log
_instances = []


def _log(self, logs):
    if self not in _instances:
        _instances.append(self)
    rec = {"global_step": getattr(self, "global_step", None), "time": time.time()}
    for k, v in logs.items():
        try:
            rec[k] = float(v) if not isinstance(v, (list, dict, str)) else v
        except Exception:
            rec[k] = str(v)
    print("LOGJSON " + json.dumps(rec), flush=True)
    return _orig_log(self, logs)


cofi_trainer.CoFiTrainer.log = _log

import run_glue_prune  # noqa: E402

if __name__ == "__main__":
    os.environ["WANDB_DISABLED"] = "true"
    t_start = time.time()
    try:
        run_glue_prune.main()
    finally:
        for tr in _instances:
            try:
                with open(os.path.join(tr.args.output_dir, "log_history.json"), "w") as f:
                    json.dump(tr.state.log_history, f, default=str)
            except Exception as e:
                print("LOGJSON_DUMP_FAILED", e)
        print(f"WRAPPER_TOTAL_SECONDS {time.time() - t_start:.2f}", flush=True)
