"""Pre-download every model the web UI's language presets need.

Run as `python -m livecaption.prefetch` (the app installer does this). Without it, the
first Start in the web UI silently downloads several GB while the page shows nothing but
"Starting…" -- fine in a terminal where Hugging Face progress bars are visible, hopeless
in the no-terminal menu-bar app. Downloading here, where the user is already watching an
installer run, turns the first Start into a warm-cache load of a few seconds.

Only fills the Hugging Face cache (snapshot_download); no model is loaded into memory,
so this works on any network-connected run and is idempotent/resumable.
"""

from __future__ import annotations

from huggingface_hub import snapshot_download

from . import config
from .webapp import LANG_PRESETS


def run() -> None:
    repos: set[str] = {config.VAD_MODEL, config.DIAR_MODEL, config.DEFAULT_MT_MODEL}
    for preset in LANG_PRESETS.values():
        repos.add(config.ASR_MODELS[preset["asr_model"]][1])
    todo = sorted(repos)
    print(f"Downloading {len(todo)} models (~6GB total on a fresh machine; resumable):")
    for i, repo in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {repo}")
        snapshot_download(repo)
    print("All models ready — the first recording will start instantly.")


if __name__ == "__main__":
    run()
