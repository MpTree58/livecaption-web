"""Command-line entry point.

Parse args, wire up audio source → ASR → translation → output, wait for Ctrl-C.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import typer
from rich.console import Console

from . import config
from .asr import AsrWorker, build_recognizer
from .audio import AudioSource, FileSource, MicSource, SystemAudioSource
from .languages import normalize_target_language
from .models import resolve_audiotee
from .render import FileWriter, Renderer
from .translate import Translator, preview_due

# Turn off huggingface_hub download progress bars (they spam the screen even when the model is
# already cached); download failures still raise. hub is lazily imported when each model loads,
# so setting this after the import block is still in time.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


class SourceKind(StrEnum):
    mic = "mic"
    system = "system"
    both = "both"
    file = "file"


class ColorTheme(StrEnum):
    auto = "auto"
    light = "light"
    dark = "dark"


class AsrModel(StrEnum):
    nemotron_0_6b = "nemotron-0.6b"
    nemotron_0_6b_8bit = "nemotron-0.6b-8bit"
    qwen3_1_7b = "qwen3-1.7b"
    qwen3_1_7b_8bit = "qwen3-1.7b-8bit"
    qwen3_0_6b = "qwen3-0.6b"
    qwen3_0_6b_8bit = "qwen3-0.6b-8bit"


class MtModel(StrEnum):
    hy_mt2_1_8b_8bit = "hy-mt2-1.8b-8bit"
    hy_mt2_1_8b_4bit = "hy-mt2-1.8b-4bit"
    hy_mt2_7b_4bit = "hy-mt2-7b-4bit"
    hy_mt2_7b_8bit = "hy-mt2-7b-8bit"


# CLI choice values must stay in lockstep with the config registries (single source of truth).
assert {m.value for m in AsrModel} == set(config.ASR_MODELS)
assert {m.value for m in MtModel} == set(config.MT_MODELS)

# Resolve defaults once at module level (typer arg defaults must not call functions -- B008).
_DEFAULT_ASR_MODEL = AsrModel(config.DEFAULT_ASR_MODEL_ALIAS)
_DEFAULT_MT_MODEL = MtModel(config.DEFAULT_MT_MODEL_ALIAS)


def main(
    # --- Audio & input ---
    source: SourceKind = typer.Option(
        SourceKind.mic, rich_help_panel="Audio & input",
        help="mic / system (meeting output) / both (dual-track) / file",
    ),
    audio_file: Path | None = typer.Option(
        None, "--file", "-f", rich_help_panel="Audio & input",
        help="Audio file to transcribe (with --source file); wav/mp3/m4a, any sample rate",
    ),
    mic_device: str | None = typer.Option(
        None, rich_help_panel="Audio & input", help="Microphone device name or index",
    ),
    include_pid: int | None = typer.Option(
        None, rich_help_panel="Audio & input",
        help="Capture only this process PID's audio (e.g. Zoom's PID)",
    ),
    audiotee: str | None = typer.Option(
        None, rich_help_panel="Audio & input", help="Path to the audiotee binary (system source)",
    ),
    list_devices: bool = typer.Option(
        False, rich_help_panel="Audio & input", help="List audio input devices and exit",
    ),
    podcast: bool = typer.Option(
        False, "--podcast", rich_help_panel="Audio & input",
        help="Podcast-file preset: keep audio timestamps, use less intrusive qwen speaker "
        "labels, and warn if the transcript appears incomplete",
    ),
    # --- ASR ---
    asr_model: AsrModel = typer.Option(
        _DEFAULT_ASR_MODEL, "--asr-model", metavar="MODEL", show_choices=False,
        rich_help_panel="ASR",
        help="Engine auto-derived from the choice. English: nemotron-0.6b, nemotron-0.6b-8bit. "
        "Chinese (needs the \\[qwen] extra): qwen3-1.7b, qwen3-1.7b-8bit, qwen3-0.6b, "
        "qwen3-0.6b-8bit",
    ),
    asr_lang: str = typer.Option(
        config.ASR_LANGUAGE, "--asr-lang", rich_help_panel="ASR",
        help="Spoken language, e.g. en-us / English, ja-jp / Japanese ('auto' = model-detected; "
        "pass an invalid value to list all supported locales)",
    ),
    diarize: bool = typer.Option(
        True, "--diarize/--no-diarize", rich_help_panel="ASR",
        help="Speaker diarization (Sortformer, up to 4 speakers): sentences are "
        "split per speaker and labeled S1/S2/…",
    ),
    qwen_word_diarize: bool | None = typer.Option(
        None, "--qwen-word-diarize/--no-qwen-word-diarize", rich_help_panel="ASR",
        help="qwen3 models only: mid-sentence speaker split via the Qwen3-ForcedAligner model "
        "(extra ~0.6B model + a per-utterance alignment pass). On by default with a qwen3-* "
        "--asr-model; use --no-qwen-word-diarize for utterance-level. Needs --diarize",
    ),
    # --- Translation ---
    translate: bool = typer.Option(
        True, "--translate/--no-translate", rich_help_panel="Translation",
        help="Enable translation",
    ),
    target_lang: str = typer.Option(
        config.DEFAULT_TARGET_LANG, rich_help_panel="Translation",
        help="Translation target language, e.g. zh-cn / Chinese, ja-jp / Japanese",
    ),
    mt_model: MtModel = typer.Option(
        _DEFAULT_MT_MODEL, "--mt-model", metavar="MODEL", show_choices=False,
        rich_help_panel="Translation",
        help="hy-mt2-1.8b-8bit (default, ~2GB), hy-mt2-1.8b-4bit, "
        "hy-mt2-7b-4bit (~4.2GB, more accurate), hy-mt2-7b-8bit",
    ),
    context: int = typer.Option(
        config.MT_CONTEXT_SENTENCES, rich_help_panel="Translation",
        help="Prior sentences passed as translation context (0=off; try 2-3)",
    ),
    # --- Output & display ---
    out: Path | None = typer.Option(
        None, "--out", "-o", rich_help_panel="Output & display",
        help="Transcript file path (default: timestamped file under ~/.cache/livecaption)",
    ),
    theme: ColorTheme = typer.Option(
        ColorTheme.auto, "--theme", rich_help_panel="Output & display",
        help="Color theme: auto (detect background via COLORFGBG, else high-contrast default) "
        "/ light / dark",
    ),
    diff: bool = typer.Option(
        True, "--diff/--no-diff", rich_help_panel="Output & display",
        help="Render final-pass corrections inline: corrected-away words struck "
        "through in grey, new words in green",
    ),
    mem: bool = typer.Option(
        False, "--mem", rich_help_panel="Output & display",
        help="Show MLX unified-memory usage (active/cache/peak) in the bottom status line",
    ),
) -> None:
    console = Console()

    if list_devices:
        import sounddevice as sd

        console.print(sd.query_devices())
        raise typer.Exit()

    target_language = None
    if translate:
        try:
            target_language = normalize_target_language(target_lang)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from None

    # Resolve the chosen aliases to (engine, model id); the engine drives the ASR branch below.
    backend_name, asr_model_id = config.ASR_MODELS[asr_model.value]
    mt_model_id = config.MT_MODELS[mt_model.value]

    if podcast and source is not SourceKind.file:
        console.print("[red]--podcast only applies to --source file[/red]")
        raise typer.Exit(1)

    # --qwen-word-diarize is qwen3-only (it drives the Qwen3-ForcedAligner); reject it for the
    # nemotron engine rather than silently ignoring it. None = not passed.
    if backend_name != "qwen3" and qwen_word_diarize is not None:
        console.print(
            "[red]--qwen-word-diarize / --no-qwen-word-diarize only applies to a qwen3-* "
            "--asr-model[/red]"
        )
        raise typer.Exit(1)

    # --- build audio sources ---
    need_system = source in (SourceKind.system, SourceKind.both)
    need_mic = source in (SourceKind.mic, SourceKind.both)

    sources: list[AudioSource] = []
    if source is SourceKind.file:
        if audio_file is None or not audio_file.exists():
            console.print("[red]--source file requires an existing --file path[/red]")
            raise typer.Exit(1)
        sources.append(FileSource(str(audio_file), label="file"))
    if need_mic:
        dev: object = mic_device
        if mic_device is not None and mic_device.isdigit():
            dev = int(mic_device)
        sources.append(MicSource(label="me", device=dev))
    if need_system:
        audiotee_bin = resolve_audiotee(audiotee)
        pids = [include_pid] if include_pid is not None else None
        sources.append(SystemAudioSource(audiotee_bin, label="them", include_pids=pids))

    # --- load and warm up the ASR + VAD (+ diarization) models (first run auto-downloads
    # from HF) ---
    try:
        if backend_name == "qwen3":
            from .asr_qwen import build_qwen_recognizer

            # on by default for qwen3 (config.QWEN_WORD_DIARIZE); None = flag not passed
            word_diarize = (
                qwen_word_diarize
                if qwen_word_diarize is not None
                else (False if podcast else config.QWEN_WORD_DIARIZE)
            )
            if podcast and diarize and qwen_word_diarize is None:
                console.print(
                    "[dim]Podcast mode: qwen word-level diarization is disabled for cleaner "
                    "speaker labels. Pass --qwen-word-diarize to force it.[/dim]"
                )
            if diarize and not word_diarize:
                console.print(
                    "[dim]Note: qwen3 diarization is utterance-level (one speaker per "
                    "sentence). Drop --no-qwen-word-diarize for mid-sentence speaker splits "
                    "(loads the Qwen3-ForcedAligner model).[/dim]"
                )
            recognizer = build_qwen_recognizer(
                asr_model_id, asr_lang, diarize, diarize and word_diarize,
                log=lambda m: console.print(f"[dim]{m}[/dim]"),
            )
        else:
            recognizer = build_recognizer(
                asr_model_id, asr_lang, diarize,
                log=lambda m: console.print(f"[dim]{m}[/dim]"),
            )
    except (ValueError, ImportError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        console.print("\nInterrupted.")
        raise typer.Exit(130) from None

    if out is None:
        out = (
            Path.home()
            / ".cache"
            / "livecaption"
            / f"transcript-{datetime.now():%Y%m%d-%H%M%S}.txt"
        )
    # meeting transcripts may contain sensitive content; tighten the directory permissions to
    # owner-only
    out.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    # only label lines with their source ([me]/[them]) when there's more than one source to
    # disambiguate (both mode); a single source's label is constant noise
    writer = FileWriter(out, translate=translate, show_label=len(sources) > 1)

    stop_event = threading.Event()

    def _on_sigint(*_) -> None:  # noqa: ANN002
        stop_event.set()
        # restore the default handler: a second Ctrl-C raises KeyboardInterrupt to
        # force-interrupt a stuck cleanup
        signal.signal(signal.SIGINT, signal.default_int_handler)

    signal.signal(signal.SIGINT, _on_sigint)

    with Renderer(
        console=console, theme=theme.value, show_mem=mem, translate=translate
    ) as renderer:
        # --- event callbacks (translator is built after handle_translation, so no placeholder
        # callback) ---
        def handle_translation(label: str, zh_segments: list, started_at: datetime) -> None:
            renderer.translation(label, zh_segments, started_at)
            if writer:
                writer.translation(label, zh_segments, started_at)

        def handle_preview(label: str, zh_segments: list, started_at: datetime) -> None:
            # live provisional translation -> terminal only (never written to the file)
            renderer.preview(label, zh_segments, started_at)

        def handle_translation_disabled() -> None:
            # the model failed to load: release any finals buffered for EN/ZH pairing so they
            # aren't stuck off-screen / unwritten, and fall back to printing EN immediately
            renderer.flush_pending()
            if writer:
                writer.flush_pending()

        translator: Translator | None = None
        if translate:
            translator = Translator(
                mt_model_id,
                target_language.prompt_name if target_language else target_lang,
                on_translation=handle_translation,
                on_ready=lambda: console.print("[dim]Translation model ready.[/dim]"),
                on_failed=handle_translation_disabled,
                on_preview=handle_preview,
                context_size=context,
            )
            console.print(f"[dim]Loading translation model {mt_model_id} …[/dim]")
            translator.start()

        # words already covered by the last live-preview translation, per source (debounce state)
        preview_words: dict[str, int] = {}
        last_final_at: dict[str, datetime] = {}
        source_samples_seen: dict[str, int] = {}

        def handle_partial(
            label: str, text: str, started_at: datetime, speaker: int | None = None
        ) -> None:
            renderer.partial(label, text, started_at, speaker)
            # P2 live preview: retranslate the in-progress utterance once it has grown by
            # MT_PREVIEW_MIN_WORDS and contains a sentence-ender (debounced; latest-wins)
            if translator:
                nwords = preview_due(text, preview_words.get(label, 0))
                if nwords is not None:
                    preview_words[label] = nwords
                    translator.submit_preview(label, [(speaker, text, None)], started_at)

        def handle_final(label: str, segments: list, started_at: datetime) -> None:
            last_final_at[label] = started_at
            preview_words.pop(label, None)  # reset debounce for the next utterance
            # segments = [(speaker, text, diff), ...]; --no-diff drops the two-pass inline
            # correction spans so the terminal shows plain corrected text
            shown = (
                segments
                if diff
                else [(spk, text, None) for spk, text, _d in segments]
            )
            renderer.final(label, shown, started_at)
            if writer:
                # the file always gets the clean, corrected text (diff is ignored there)
                writer.final(label, segments, started_at)
            if translator:
                translator.submit(label, segments, started_at)

        def handle_source_done(label: str, samples_seen: int) -> None:
            source_samples_seen[label] = samples_seen

        def warn_if_file_transcript_looks_incomplete() -> None:
            """File/podcast guardrail: the run consumed all audio, but the last emitted final is
            far from the media end. That can be a long outro/music bed, but for podcast text
            workflows it often means the run was interrupted or useful speech was missed."""
            if source is not SourceKind.file:
                return
            for label, samples_seen in source_samples_seen.items():
                last = last_final_at.get(label)
                if last is None:
                    console.print(
                        f"[yellow]Warning: '{label}' consumed "
                        f"{samples_seen / config.SAMPLE_RATE:.1f}s of audio but emitted no "
                        "final transcript.[/yellow]"
                    )
                    continue
                media_sec = samples_seen / config.SAMPLE_RATE
                # File-mode timestamps are anchored at 2000-01-01 00:00:00.
                last_sec = (
                    last - datetime(last.year, last.month, last.day)
                ).total_seconds()
                gap = media_sec - last_sec
                if gap >= config.FILE_TRAILING_GAP_WARN_SEC:
                    console.print(
                        f"[yellow]Warning: last transcript for '{label}' is "
                        f"{gap / 60:.1f} min before the end of the audio. This may be outro "
                        "music/silence, or the transcript may be incomplete.[/yellow]"
                    )

        workers: list[AsrWorker] = []
        for src in sources:
            # the weights carry no decode state and can be shared; each worker create_stream's
            # its own independent state.
            # worker crash → stop_event: let the whole pipeline shut down gracefully instead of
            # zombie-living
            worker = AsrWorker(
                recognizer, src.queue, src.label, handle_partial, handle_final,
                on_error=stop_event.set,
                on_done=handle_source_done,
                use_audio_timestamps=source is SourceKind.file,
            )
            workers.append(worker)
            worker.start()
            src.start()

        labels = " + ".join(s.label for s in sources)
        console.print(
            f"[bold green]● Listening[/bold green]: {labels}  (Ctrl-C to stop)"
        )
        console.print(f"[dim]Transcript → {out}[/dim]")
        if need_system:
            console.print(
                "[dim]Tip: if system audio stays empty, the terminal app likely lacks "
                "Screen & System Audio Recording permission — see README.[/dim]"
            )

        try:
            if source is SourceKind.file:
                # after the file is fully fed (SENTINEL), the worker exits on its own; then
                # drain the translation queue and finish
                for w in workers:
                    while w.is_alive() and not stop_event.is_set():
                        w.join(timeout=0.2)
                warn_if_file_transcript_looks_incomplete()
                if translator is not None and not stop_event.is_set():
                    # None goes at the tail, so the backlog of translations gets done first
                    translator.stop()
                    translator.join()
            else:
                stop_event.wait()
        finally:
            try:
                for src in sources:
                    src.stop()
                for worker in workers:
                    # the finalization path includes a full-sentence two-pass re-decode + a
                    # full-sentence diar feed, so give it plenty of time; if you can't wait,
                    # press Ctrl-C again to force quit
                    worker.join(timeout=15)
                    if worker.is_alive():
                        print(
                            f"\n[warn] ASR worker '{worker.label}' did not finish in "
                            "time; the last sentence may be lost.",
                            file=sys.stderr,
                        )
                if translator is not None:
                    translator.stop()
                    translator.join(timeout=10)
                    if translator.is_alive():
                        print(
                            "\n[warn] pending translations dropped on exit.",
                            file=sys.stderr,
                        )
            except KeyboardInterrupt:
                console.print("\n[red]Force quit; pending output dropped.[/red]")
            finally:
                if writer:
                    writer.close()

    console.print("\n[dim]Stopped.[/dim]")


def run() -> None:
    typer.run(main)


if __name__ == "__main__":
    run()
