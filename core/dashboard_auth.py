"""Hash generator for the dashboard password.

The Streamlit dashboard (``app.py``) never stores the plaintext password
anywhere; it validates against a SHA-256 hash held in the
``AITB__DASHBOARD__PASSWORD_HASH`` environment variable. This script
exists solely to generate that hash.

Usage::

    python -m core.dashboard_auth
    (you'll be prompted for a password, not echoed, the hash is printed to stdout)

Assign the generated hash to the environment variable::

    export AITB__DASHBOARD__PASSWORD_HASH=<generated_hash>   # Linux/Mac
    $env:AITB__DASHBOARD__PASSWORD_HASH = "<generated_hash>"  # PowerShell
"""

from __future__ import annotations

import getpass
import hashlib


def hash_password(password: str) -> str:
    """Returns the SHA-256 hex digest of the password."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def main() -> None:
    password = getpass.getpass("Dashboard password: ")
    confirm = getpass.getpass("Password (again): ")
    if not password:
        raise SystemExit("An empty password is not accepted.")
    if password != confirm:
        raise SystemExit("Passwords did not match.")
    print(hash_password(password))


if __name__ == "__main__":
    main()
