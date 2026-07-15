"""Password hashing and token generation (stdlib only -- no external crypto deps).

Passwords are stored as ``pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>``. PBKDF2-HMAC-
SHA256 is a standard, slow-by-design KDF included in Python's hashlib; the per-password
random salt means two users with the same password get different stored hashes, and the
iteration count makes brute-forcing a leaked hash expensive.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

_ITERATIONS = 240_000  # slow enough to deter offline brute force; fine for interactive login


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ITERATIONS)
    return f"pbkdf2_sha256${_ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_hex, hash_hex = stored.split("$")
    except ValueError:
        return False
    if algo != "pbkdf2_sha256":
        return False
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iters)
    )
    # constant-time compare so timing doesn't leak how much of the hash matched
    return hmac.compare_digest(dk.hex(), hash_hex)


def new_api_token() -> str:
    """Opaque bearer token a Mac capture agent uses to push meetings for its user."""
    return "lc_" + secrets.token_urlsafe(32)


def new_secret_key() -> str:
    """Session-cookie signing key (persisted so sessions survive a server restart)."""
    return secrets.token_urlsafe(48)
