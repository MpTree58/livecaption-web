#!/usr/bin/env bash
# One-shot installer: turn livecaption into a double-clickable macOS menu-bar app.
#
# What it does (idempotent; safe to re-run to update):
#   1. installs uv if missing
#   2. clones (or updates) the repo into ~/livecaption   [skipped when run from a checkout]
#   3. installs Python dependencies (uv sync --extra app)
#   4. builds audiotee (system-audio capture) if possible
#   5. pre-downloads the speech/translation models (~6GB, with progress bars)
#   6. creates /Applications/livecaption.app  -> menu-bar 🎙 app, no terminal ever again
#   7. (default) installs a login item so the app starts automatically at login
#
# Usage:
#   curl -LsSf https://raw.githubusercontent.com/MpTree58/livecaption-web/main/scripts/install_app.sh | bash
# Options (append after `bash -s --` in the curl form, or as normal args):
#   --no-login    do not auto-start at login
#   --uninstall   remove the app + login item (keeps the repo, models and recordings)
set -euo pipefail

REPO_URL="https://github.com/MpTree58/livecaption-web.git"
APP_NAME="livecaption"
BUNDLE_ID="com.livecaption.menubar"
LAUNCH_AGENT="$HOME/Library/LaunchAgents/$BUNDLE_ID.plist"

say()  { printf '\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!!  %s\033[0m\n' "$*"; }

INSTALL_LOGIN=1
UNINSTALL=0
for arg in "$@"; do
    case "$arg" in
        --no-login)  INSTALL_LOGIN=0 ;;
        --uninstall) UNINSTALL=1 ;;
        *) echo "unknown option: $arg (use --no-login / --uninstall)"; exit 2 ;;
    esac
done

# /Applications is admin-group writable on a normal Mac; fall back to ~/Applications.
APP_PARENT="/Applications"
if [ ! -w "$APP_PARENT" ]; then
    APP_PARENT="$HOME/Applications"
    mkdir -p "$APP_PARENT"
fi
APP_DIR="$APP_PARENT/$APP_NAME.app"

if [ "$UNINSTALL" = 1 ]; then
    launchctl bootout "gui/$(id -u)" "$LAUNCH_AGENT" 2>/dev/null || true
    rm -f "$LAUNCH_AGENT"
    rm -rf "/Applications/$APP_NAME.app" "$HOME/Applications/$APP_NAME.app"
    say "Removed the app and login item. The repo, models and recordings are untouched."
    exit 0
fi

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
    echo "livecaption needs an Apple Silicon Mac (M1 or newer)." >&2
    exit 1
fi

# git/swift both come from the Command Line Tools; without them nothing below works.
if ! xcode-select -p >/dev/null 2>&1; then
    warn "Apple's Command Line Tools are missing. A dialog should pop up now —"
    warn "click Install, wait for it to finish, then run this installer again."
    xcode-select --install 2>/dev/null || true
    exit 1
fi

# --- 1. uv ------------------------------------------------------------------
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    say "Installing uv (Python package manager)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# --- 2. the repo ------------------------------------------------------------
# When run from inside a checkout (bash scripts/install_app.sh), use that checkout
# as-is; when piped from curl, clone/update ~/livecaption.
LC_DIR=""
SCRIPT_SRC="${BASH_SOURCE[0]:-}"
if [ -n "$SCRIPT_SRC" ] && [ -f "$SCRIPT_SRC" ]; then
    CANDIDATE="$(cd "$(dirname "$SCRIPT_SRC")/.." 2>/dev/null && pwd || true)"
    if [ -n "$CANDIDATE" ] && grep -qs '^name = "livecaption"' "$CANDIDATE/pyproject.toml"; then
        LC_DIR="$CANDIDATE"
    fi
fi
if [ -z "$LC_DIR" ]; then
    LC_DIR="${LIVECAPTION_DIR:-$HOME/livecaption}"
    if [ -d "$LC_DIR/.git" ]; then
        say "Updating existing checkout at $LC_DIR"
        git -C "$LC_DIR" pull --ff-only || warn "git pull failed; continuing with the current checkout"
    else
        say "Cloning livecaption into $LC_DIR"
        git clone "$REPO_URL" "$LC_DIR"
    fi
fi

# --- 3. Python deps ---------------------------------------------------------
# qwen = the Chinese/mixed-language ASR engine the web UI's presets rely on
say "Installing Python dependencies (first run downloads ~2GB of packages)"
(cd "$LC_DIR" && uv sync --extra app --extra qwen)

# --- 4. audiotee (system-audio capture; mic-only works without it) ----------
if [ -x "$LC_DIR/bin/audiotee" ]; then
    say "audiotee already built"
else
    say "Building audiotee (captures the other side of your meetings)"
    bash "$LC_DIR/scripts/build_audiotee.sh" \
        || warn "audiotee build failed — the mic still works; re-run scripts/build_audiotee.sh later for system audio"
fi

# --- 5. models --------------------------------------------------------------
# Download the speech/translation models NOW, with visible progress bars, instead of
# leaving the first in-app Start stuck on a progress-less "Starting…" for ~6GB.
say "Downloading speech models (one-time, ~6GB; resumable if interrupted)"
(cd "$LC_DIR" && uv run --frozen --extra app --extra qwen python -m livecaption.prefetch) \
    || warn "model download incomplete — the app still works; the first Start will fetch the rest"

# --- 6. the .app bundle -----------------------------------------------------
say "Creating $APP_DIR"
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS"

cat > "$APP_DIR/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>livecaption</string>
    <key>CFBundleDisplayName</key><string>livecaption</string>
    <key>CFBundleIdentifier</key><string>$BUNDLE_ID</string>
    <key>CFBundleVersion</key><string>1.0</string>
    <key>CFBundleShortVersionString</key><string>1.0</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleExecutable</key><string>$APP_NAME</string>
    <key>LSMinimumSystemVersion</key><string>14.0</string>
    <key>LSUIElement</key><true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>livecaption transcribes meetings from your microphone, entirely on this Mac.</string>
    <key>NSAudioCaptureUsageDescription</key>
    <string>livecaption captures system audio to transcribe the other side of your meetings.</string>
</dict>
</plist>
PLIST

# The launcher embeds the checkout path chosen above; \$-escaped vars resolve at launch time.
cat > "$APP_DIR/Contents/MacOS/$APP_NAME" <<LAUNCH
#!/bin/bash
# Launcher for the livecaption menu-bar app (generated by scripts/install_app.sh).
export PATH="\$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:\$PATH"
# Local-only app: no accounts, no authentication, nothing ever leaves this Mac.
# Force-clear any central-sync variables so recordings can never be uploaded.
unset LC_SYNC_URL LC_SYNC_TOKEN
cd "$LC_DIR" || exit 1
exec uv run --frozen --extra app --extra qwen livecaption-app
LAUNCH
chmod +x "$APP_DIR/Contents/MacOS/$APP_NAME"

# --- 7. login item ----------------------------------------------------------
if [ "$INSTALL_LOGIN" = 1 ]; then
    say "Installing login item (the app starts automatically when you log in)"
    mkdir -p "$HOME/Library/LaunchAgents"
    cat > "$LAUNCH_AGENT" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$BUNDLE_ID</string>
    <key>ProgramArguments</key>
    <array><string>$APP_DIR/Contents/MacOS/$APP_NAME</string></array>
    <key>RunAtLoad</key><true/>
</dict>
</plist>
PLIST
    launchctl bootout "gui/$(id -u)" "$LAUNCH_AGENT" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$LAUNCH_AGENT" 2>/dev/null \
        || launchctl load "$LAUNCH_AGENT" 2>/dev/null || true
fi

# --- done -------------------------------------------------------------------
echo
say "Done! Look for the 🎙 icon in the menu bar (top-right of the screen)."
echo "    Click it -> 'Open livecaption' to record and browse your meetings."
echo "    (If it's not there yet, open it once from Launchpad / Applications.)"
echo
echo "    Models are already downloaded — the first recording starts right away."
echo
echo "    To also caption the OTHER side of a meeting (system audio), grant permission once:"
echo "      System Settings > Privacy & Security > Screen & System Audio Recording"
echo "      > 'System Audio Recording Only' (the LOWER sub-section on macOS 15+)"
echo "      > add 'livecaption' from $APP_PARENT, then quit and reopen the app."
echo
echo "    Uninstall the app any time with:  bash scripts/install_app.sh --uninstall"
