"""Dashboard şifresi için hash üretici.

Streamlit dashboard'u (``app.py``) düz metin şifreyi hiçbir yerde saklamaz;
``AITB__DASHBOARD__PASSWORD_HASH`` ortam değişkeninde tutulan bir SHA-256
hash'ine karşı doğrulama yapar. Bu script yalnızca o hash'i üretmek içindir.

Kullanım::

    python -m core.dashboard_auth
    (şifre sorulur, ekrana yazılmaz, hash stdout'a basılır)

Üretilen hash'i ortam değişkenine ata::

    export AITB__DASHBOARD__PASSWORD_HASH=<uretilen_hash>   # Linux/Mac
    $env:AITB__DASHBOARD__PASSWORD_HASH = "<uretilen_hash>"  # PowerShell
"""

from __future__ import annotations

import getpass
import hashlib


def hash_password(password: str) -> str:
    """Şifrenin SHA-256 hex özetini döndürür."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def main() -> None:
    password = getpass.getpass("Dashboard sifresi: ")
    confirm = getpass.getpass("Sifre (tekrar): ")
    if not password:
        raise SystemExit("Bos sifre kabul edilmez.")
    if password != confirm:
        raise SystemExit("Sifreler eslesmedi.")
    print(hash_password(password))


if __name__ == "__main__":
    main()
