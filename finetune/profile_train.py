"""Profile a few QLoRA training steps with torch.profiler and log the results to wandb.

Usage (from the DFS repo root, on the GPU box, with your environment activated):
    WANDB_PROJECT=dfs-retro-qlora python finetune/profile_train.py finetune/qwen_qlora_uspto50k.yaml
    PROFILE_STACK=1 ...   # also record Python stacks (much slower, much bigger trace)

Runs 8 optimizer steps and stops: 3 unprofiled (steps 2-3 give the clean step time),
2 profiler warmup, 3 recorded. Logged to the wandb run "<run_name>-profile":
    profile/step_time_clean_s     mean step time with the profiler off (your real speed)
    profile/step_time_profiled_s  mean step time while recording (inflated by the profiler)
    profile/peak_gpu_mem_gb       peak allocated GPU memory
    profile/top_kernels           table of the 50 slowest ops (GPU/CPU time, calls, GPU memory)
    torch_trace_<run id>          artifact with the gzipped Chrome trace, for https://ui.perfetto.dev
The uncompressed trace also stays in saves/profile_run/trace/ for HTA (Holistic Trace Analysis).
"""
import gzip
import os
import shutil
import sys
import time

import torch
import wandb
from transformers import TrainerCallback

from llamafactory.hparams import read_args
from llamafactory.train.tuner import run_exp

WAIT, WARMUP, ACTIVE = 3, 2, 3  # optimizer steps: unprofiled, profiler warmup, recorded
TRACE_DIR = "saves/profile_run/trace"
WITH_STACK = os.environ.get("PROFILE_STACK") == "1"


def _attr(evt, *names):
    # torch renamed cuda_* to device_* in 2.4; read whichever this version has.
    for n in names:
        if hasattr(evt, n):
            return getattr(evt, n)
    return 0


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


class WandbProfilerCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        os.makedirs(TRACE_DIR, exist_ok=True)
        # HTA only picks up files ending in .pt.trace.json
        self.trace_path = os.path.join(TRACE_DIR, "rank0.pt.trace.json")
        self.clean_times, self.profiled_times = [], []
        self.prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=WAIT, warmup=WARMUP, active=ACTIVE, repeat=1),
            on_trace_ready=lambda p: p.export_chrome_trace(self.trace_path),
            profile_memory=True,
            record_shapes=True,
            with_stack=WITH_STACK,
        )
        self.prof.start()

    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        self.t0 = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        dt = time.perf_counter() - self.t0
        step = state.global_step  # already incremented: 1 for the first step
        if 2 <= step <= WAIT:  # step 1 carries allocator / kernel warm-up, skip it
            self.clean_times.append(dt)
        elif step > WAIT + WARMUP:
            self.profiled_times.append(dt)
        self.prof.step()
        if step >= WAIT + WARMUP + ACTIVE:
            self.prof.stop()
            if state.is_world_process_zero:
                self._log_to_wandb(args)
            control.should_training_stop = True

    def _log_to_wandb(self, args):
        run = wandb.run or wandb.init(project=os.environ.get("WANDB_PROJECT"), name=args.run_name)

        events = self.prof.key_averages()
        table = wandb.Table(columns=["op", "calls", "gpu_total_ms", "gpu_self_ms",
                                     "cpu_total_ms", "gpu_mem_self_mb"])
        rows = sorted(events, key=lambda e: _attr(e, "device_time_total", "cuda_time_total"), reverse=True)
        for e in rows[:50]:
            table.add_data(
                e.key,
                e.count,
                _attr(e, "device_time_total", "cuda_time_total") / 1e3,
                _attr(e, "self_device_time_total", "self_cuda_time_total") / 1e3,
                e.cpu_time_total / 1e3,
                _attr(e, "self_device_memory_usage", "self_cuda_memory_usage") / 2**20,
            )
        summary = {
            "profile/step_time_clean_s": _mean(self.clean_times),
            "profile/step_time_profiled_s": _mean(self.profiled_times),
            "profile/peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
        }
        run.log({"profile/top_kernels": table, **summary})
        run.summary.update(summary)

        gz_path = self.trace_path + ".gz"
        with open(self.trace_path, "rb") as src, gzip.open(gz_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        artifact = wandb.Artifact(f"torch_trace_{run.id}", type="profile")
        artifact.add_file(gz_path)
        run.log_artifact(artifact)

        print(events.table(sort_by="self_cuda_time_total", row_limit=25))
        for k, v in summary.items():
            print(f"{k}: {v:.3f}")
        print(f"trace: {self.trace_path}")


args = read_args()  # parses sys.argv: yaml in argv[1] plus key=value overrides -> dict
if not isinstance(args, dict):
    sys.exit("usage: python finetune/profile_train.py <config.yaml> [key=value ...]")
args.update(
    report_to="wandb",
    run_name=f"{args.get('run_name', 'qlora')}-profile",
    output_dir="saves/profile_run",
    save_strategy="no",
    eval_strategy="no",
)
run_exp(args, callbacks=[WandbProfilerCallback()])
