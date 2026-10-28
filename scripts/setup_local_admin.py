"""Local-only setup. Never prints credentials or replaces nonempty settings."""
import argparse
import getpass
import os
import secrets
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.env_loader import _dotenv_values
from backend.admin_auth import hash_password


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true', help='Prepare settings without asking for a password')
    args = parser.parse_args()
    path = ROOT / '.env'
    original = path.read_bytes() if path.exists() else b''
    values = _dotenv_values(path)
    updates = {}
    defaults = {
        'AUTH_COOKIE_SECURE': 'false',
        'AUTH_TRUST_PROXY': 'false',
        'APP_ALLOWED_ORIGINS': 'http://127.0.0.1:8080,http://localhost:8080',
        'ADMIN_SESSION_DB': 'artifacts/admin_sessions.sqlite3',
        'ADMIN_SESSION_TTL_SECONDS': '3600',
        'ADMIN_LOGIN_MAX_ATTEMPTS': '5',
        'ADMIN_LOGIN_WINDOW_SECONDS': '900',
    }
    def effective(key):
        return os.environ.get(key, '').strip() or values.get(key, '').strip()
    for key, value in defaults.items():
        if not effective(key):
            updates[key] = value
    if not effective('ADMIN_SESSION_SECRET'):
        updates['ADMIN_SESSION_SECRET'] = secrets.token_urlsafe(48)
    elif len(effective('ADMIN_SESSION_SECRET')) < 32:
        raise SystemExit('Existing session secret is too short; preserved. Review it locally before continuing.')
    if not effective('ADMIN_PASSWORD_HASH') and not args.prepare:
        if not sys.stdin.isatty():
            raise SystemExit('Run from your own interactive terminal. Piped passwords are refused.')
        with warnings.catch_warnings():
            warnings.simplefilter('error', getpass.GetPassWarning)
            password = getpass.getpass('Administrator password (12+ characters, hidden): ')
            confirm = getpass.getpass('Confirm password (hidden): ')
        if len(password) < 12 or password != confirm:
            raise SystemExit('Passwords must match and contain at least 12 characters. Nothing written.')
        updates['ADMIN_PASSWORD_HASH'] = hash_password(password)
        del password, confirm
    if updates:
        # Append only missing/empty settings; retain all existing bytes and credentials.
        if (path.read_bytes() if path.exists() else b'') != original:
            raise SystemExit('.env changed during setup; nothing written. Run again.')
        newline = b'\r\n' if b'\r\n' in original else b'\n'
        addition = newline.join(f'{k}={v}'.encode('utf-8') for k, v in updates.items()) + newline
        prefix = newline if original and not original.endswith(b'\n') else b''
        with path.open('ab') as stream:
            stream.write(prefix + addition)
    print('Local settings prepared; existing nonempty settings preserved. No credentials displayed.')
    print('Password hash: ' + ('configured' if effective('ADMIN_PASSWORD_HASH') or 'ADMIN_PASSWORD_HASH' in updates else 'PENDING: run again without --prepare in your terminal'))
    if effective('AUTH_COOKIE_SECURE').lower() not in {'', 'false', '0'}:
        print('Existing Secure cookie setting preserved: HTTP login needs review; use HTTPS or explicitly adjust locally.')
    print('Restart the project services after configuration. Use http://127.0.0.1:8080 consistently.')


if __name__ == '__main__':
    main()
