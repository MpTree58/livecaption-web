"""Admin CLI for the central server: create users and manage their agent tokens.

There is no open sign-up (this is an internal tool) -- an admin creates each account here,
then hands the user their login and their agent API token.

    livecaption-server-admin adduser alice
    livecaption-server-admin listusers
    livecaption-server-admin token alice --regen
    livecaption-server-admin passwd alice
"""

from __future__ import annotations

import argparse
import getpass
import sys

from . import db


def _prompt_password(username: str) -> str:
    p1 = getpass.getpass(f"Password for {username}: ")
    p2 = getpass.getpass("Repeat: ")
    if p1 != p2:
        print("Passwords do not match.", file=sys.stderr)
        raise SystemExit(1)
    if not p1:
        print("Empty password.", file=sys.stderr)
        raise SystemExit(1)
    return p1


def cmd_adduser(args: argparse.Namespace) -> None:
    password = args.password or _prompt_password(args.username)
    try:
        u = db.create_user(args.username, password, is_admin=args.admin)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(1) from None
    print(f"Created user {u['username']}" + (" (admin)" if u["is_admin"] else ""))
    print(f"  agent API token: {u['api_token']}")
    print("  → put this in the user's Mac agent config (server URL + this token).")


def cmd_listusers(_args: argparse.Namespace) -> None:
    users = db.list_users()
    if not users:
        print("(no users yet — create one with: adduser <name>)")
        return
    for u in users:
        flag = " [admin]" if u["is_admin"] else ""
        print(f"{u['username']}{flag}\n  token: {u['api_token']}")


def cmd_token(args: argparse.Namespace) -> None:
    u = db.get_user_by_username(args.username)
    if not u:
        print(f"no such user: {args.username}", file=sys.stderr)
        raise SystemExit(1)
    if args.regen:
        token = db.regen_token(u["id"])
        print(f"regenerated token for {args.username}:\n  {token}")
        print("  (the user must update their Mac agent with the new token)")
    else:
        print(u["api_token"])


def cmd_passwd(args: argparse.Namespace) -> None:
    u = db.get_user_by_username(args.username)
    if not u:
        print(f"no such user: {args.username}", file=sys.stderr)
        raise SystemExit(1)
    db.set_password(u["id"], _prompt_password(args.username))
    print(f"password updated for {args.username}")


def cmd_deluser(args: argparse.Namespace) -> None:
    u = db.get_user_by_username(args.username)
    if not u:
        print(f"no such user: {args.username}", file=sys.stderr)
        raise SystemExit(1)
    owned = db.count_owned_meetings(u["id"])
    if owned and not args.purge and not args.transfer:
        print(
            f"{args.username} owns {owned} meeting(s). Re-run with one of:\n"
            f"  --transfer <user>   reassign their meetings to another user, then delete\n"
            f"  --purge             delete their meetings too",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if args.transfer:
        target = db.get_user_by_username(args.transfer)
        if not target:
            print(f"no such transfer target: {args.transfer}", file=sys.stderr)
            raise SystemExit(1)
        n = db.transfer_meetings(u["id"], target["id"])
        print(f"transferred {n} meeting(s) to {args.transfer}")
    if not args.yes:
        what = " and their meetings" if args.purge else ""
        if input(f"Delete user {args.username}{what}? [y/N] ").strip().lower() != "y":
            print("aborted.")
            return
    db.delete_user(u["id"])
    print(f"deleted user {args.username}")


def run() -> None:
    db.init_db()
    p = argparse.ArgumentParser(prog="livecaption-server-admin")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("adduser", help="create a user")
    a.add_argument("username")
    a.add_argument("--password", help="set non-interactively (else prompt)")
    a.add_argument("--admin", action="store_true")
    a.set_defaults(func=cmd_adduser)

    lu = sub.add_parser("listusers", help="list users and their agent tokens")
    lu.set_defaults(func=cmd_listusers)

    t = sub.add_parser("token", help="show or regenerate a user's agent token")
    t.add_argument("username")
    t.add_argument("--regen", action="store_true")
    t.set_defaults(func=cmd_token)

    pw = sub.add_parser("passwd", help="change a user's password")
    pw.add_argument("username")
    pw.set_defaults(func=cmd_passwd)

    du = sub.add_parser("deluser", help="delete a user (whitelist removal, e.g. when they leave)")
    du.add_argument("username")
    du.add_argument("--transfer", metavar="USER", help="reassign their meetings to this user first")
    du.add_argument("--purge", action="store_true", help="delete their meetings too")
    du.add_argument("--yes", "-y", action="store_true", help="skip the confirmation prompt")
    du.set_defaults(func=cmd_deluser)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    run()
