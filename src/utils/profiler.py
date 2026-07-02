import cProfile
import os
import sys
import threading
from datetime import datetime

from utils.utils import BASE


profiler = cProfile.Profile()
profile_context = {
    "task": "unknown",
    "clients": "all",
    "duration": "unknown",
}


def update_profile_context(task, clients):
    profile_context["task"] = task
    profile_context["clients"] = clients


def build_profile_path(prefix, suffix="prof"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task = str(profile_context["task"]).replace(" ", "_")
    clients = (
        "_".join(profile_context["clients"])
        if isinstance(profile_context["clients"], list)
        else str(profile_context["clients"])
    )
    return os.path.join(BASE, f"profiler_data/{prefix}_{task}_{clients}_{timestamp}.{suffix}")


def enable_profiling():
    profiler.enable()


def save_profile():
    profiler.disable()
    output_path = build_profile_path("optc_profile")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    profiler.dump_stats(output_path)
    profiler.create_stats()
    profiler.enable()
    print(f"Profile saved to {output_path}", flush=True)


def save_and_exit(sig, frame):
    print("saving profile...", flush=True)
    save_profile()
    sys.exit(0)


def periodic_save(interval_seconds=600):
    while True:
        threading.Event().wait(interval_seconds)
        save_profile()
        print("Checkpoint saved.", flush=True)


def start_periodic_saver(interval_seconds=600):
    thread = threading.Thread(target=periodic_save, args=(interval_seconds,), daemon=True)
    thread.start()
    return thread
