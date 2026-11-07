"""sd.cpp-webui - extensions - the network access an extension gets

An extension reaches the network through the app only (api.fetch_json and
api.download), and only to the hosts it declared with HOSTS. The app
follows redirects itself, checking every hop against those hosts, sends
the extension's headers (an API key) to the first host only, and refuses
addresses that aren't public. Downloads land in a folder the app picks,
never one the extension names, and each one is written to a log the
extensions can't edit through the api (shown in the Extensions tab).

This protects an honest extension from hostile data (a redirect, a url or
a file name from a server). It is not a sandbox: an extension is Python
code and can open its own connections, and HOSTS is only what the
extension declares it will contact, not something the app can enforce.
"""

import contextlib
import hashlib
import http.client
import ipaddress
import json
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone

# Where each kind of model goes: the config key of its folder
MODEL_KINDS = {
    'checkpoint': 'ckpt_dir',
    'diffusion': 'unet_dir',
    'vae': 'vae_dir',
    'text_encoder': 'txt_enc_dir',
    'embedding': 'emb_dir',
    'lora': 'lora_dir',
    'upscaler': 'upscl_dir',
    'controlnet': 'cnnet_dir',
    'adetailer': 'ad_dir',
    'ipadapter': 'ipadapter_dir',
    'pulid': 'pulid_dir',
    'animatediff': 'animatediff_dir',
    'photomaker': 'phtmkr_dir',
    'taesd': 'taesd_dir',
}
MODEL_FILES = ('.safetensors', '.sft', '.gguf', '.ckpt', '.pt', '.pth',
               '.bin', '.onnx')

MAX_REDIRECTS = 5
MAX_JSON_BYTES = 5 * 1024 * 1024
TIMEOUT = 30
CHUNK = 1024 * 1024

LOG_NAME = 'downloads.jsonl'

_HOST = re.compile(r'(\*\.)?[a-z0-9](?:[a-z0-9-]*[a-z0-9])?'
                   r'(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)+')
_FILENAME = re.compile(r"[\w.()+,@=\- ]+")


class BlockedError(Exception):
    """The network access was refused by the app (host, address, size)."""


# ------------------------------------------------------------------ hosts

def valid_host_pattern(pattern) -> bool:
    """'example.com' or '*.example.com': a name, never an address."""
    if not isinstance(pattern, str) or not _HOST.fullmatch(pattern):
        return False
    try:
        ipaddress.ip_address(pattern.removeprefix('*.'))
        return False
    except ValueError:
        return True


def host_allowed(host: str, patterns) -> bool:
    for pattern in patterns:
        if pattern.startswith('*.'):
            if host.endswith(pattern[1:]):
                return True
        elif host == pattern:
            return True
    return False


def _resolve(host: str) -> list[str]:
    return [info[4][0] for info in socket.getaddrinfo(host, 443)]


def public_address(host: str) -> str:
    """
    An address of host, resolved now. BlockedError unless every address
    it has is public.
    """
    try:
        addresses = _resolve(host)
    except OSError as e:
        raise BlockedError(f"cannot resolve {host}: {e}") from e
    if not addresses:
        raise BlockedError(f"cannot resolve {host}")
    for address in addresses:
        if not ipaddress.ip_address(address.split('%')[0]).is_global:
            raise BlockedError(f"{host} is not a public address")
    return addresses[0]


def check_url(url: str, patterns) -> urllib.parse.SplitResult:
    """The parsed url if the extension may reach it, else BlockedError."""
    parts = urllib.parse.urlsplit(str(url))
    host = (parts.hostname or '').lower()
    if parts.scheme != 'https' or not host:
        raise BlockedError(f"only https urls are allowed, not {url!r}")
    if parts.username or parts.password:
        raise BlockedError("urls with a user or password are not allowed")
    try:
        port = parts.port
    except ValueError:
        raise BlockedError("the url has an invalid port") from None
    if port not in (None, 443):
        raise BlockedError("only the default https port is allowed")
    if not host_allowed(host, patterns):
        raise BlockedError(
            f"{host} is not one of the extension\'s HOSTS "
            f'({", ".join(patterns) or "none declared"})'
        )
    public_address(host)  # an early, clear error; connecting checks again
    return parts


# ---------------------------------------------------------------- requests

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """
    Connects to the address it checked, not to one resolved again: the
    check and the connection can't see different answers (DNS rebinding).
    The certificate is still checked against the host name.
    """

    def connect(self):
        if self._tunnel_host:  # through a proxy, which resolves the target
            return super().connect()
        address = public_address(self.host)
        sock = socket.create_connection(
            (address, self.port), self.timeout, self.source_address
        )
        self.sock = self._context.wrap_socket(
            sock, server_hostname=self.host
        )


class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(_PinnedHTTPSConnection, req)


_OPENER = urllib.request.build_opener(_NoRedirect, _PinnedHTTPSHandler)


def _urlopen(request):
    return _OPENER.open(request, timeout=TIMEOUT)


def _clean_headers(headers) -> dict:
    result = {}
    for name, value in dict(headers or {}).items():
        pair = str(name), str(value)
        if any(char in text for text in pair for char in '\r\n'):
            raise ValueError("a header cannot hold a line break")
        result[pair[0]] = pair[1]
    return result


def open_url(url: str, patterns, headers=None):
    """
    GET url: (response, final url). Redirects are followed here, each one
    checked; the headers go to the first host only.
    """
    headers = _clean_headers(headers)
    first = None
    for _ in range(MAX_REDIRECTS + 1):
        parts = check_url(url, patterns)
        host = parts.hostname.lower()
        first = first or host
        request = urllib.request.Request(
            url, headers={'User-Agent': 'sd.cpp-webui'}
            | (headers if host == first else {})
        )
        try:
            return _urlopen(request), url
        except urllib.error.HTTPError as e:
            location = e.headers.get('Location') if e.headers else None
            if e.code in (301, 302, 303, 307, 308) and location:
                e.close()
                url = urllib.parse.urljoin(url, location)
                continue
            raise
    raise BlockedError(f"more than {MAX_REDIRECTS} redirects")


def fetch_json(url: str, patterns, headers=None):
    """GET a url and parse the JSON (at most MAX_JSON_BYTES)."""
    response, _ = open_url(url, patterns, headers)
    with response:
        data = response.read(MAX_JSON_BYTES + 1)
    if len(data) > MAX_JSON_BYTES:
        raise BlockedError("the answer is too big")
    return json.loads(data)


# --------------------------------------------------------------- downloads

def safe_filename(name, model: bool) -> str:
    """The file name itself: no folders, no tricks. ValueError if not."""
    if (not isinstance(name, str) or not name or len(name) > 200
            or name.startswith('.') or not _FILENAME.fullmatch(name)):
        raise ValueError(f"{name!r} is not a plain file name")
    if model and not name.lower().endswith(MODEL_FILES):
        raise ValueError(
            f'a model file ends with one of {", ".join(MODEL_FILES)}'
        )
    return name


def download(
    url: str, patterns, folder: str, filename: str, headers=None,
    sha256: str | None = None, progress: Callable | None = None
) -> dict:
    """
    Download url into folder/filename, without replacing a file. Returns
    {'name', 'size', 'sha256', 'host'}. The file appears complete or not at
    all. Raises BlockedError, OSError, FileExistsError or ValueError (also
    when the file isn't the expected one).
    """
    os.makedirs(folder, exist_ok=True)
    final = os.path.join(folder, filename)
    if os.path.lexists(final):
        raise FileExistsError(f"{filename} already exists")
    part = final + '.part'

    response, last = open_url(url, patterns, headers)
    digest = hashlib.sha256()
    size = 0
    try:
        with response, open(part, 'wb') as out:
            total = int(response.headers.get('Content-Length') or 0)
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                digest.update(chunk)
                size += len(chunk)
                if progress:
                    progress(size, total or None)
        found = digest.hexdigest()
        if sha256 and found != sha256.lower():
            raise ValueError("the file does not match the expected sha256")
        os.replace(part, final)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(part)
        raise
    return {'name': filename, 'size': size, 'sha256': found,
            'host': urllib.parse.urlsplit(last).hostname}


# --------------------------------------------------------------------- log

def log_path(config_path: str) -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(config_path)), 'extensions', LOG_NAME
    )


def _plain_url(url: str) -> str:
    """The url without its query (where tokens live) or fragment."""
    parts = urllib.parse.urlsplit(str(url))
    host = parts.hostname or ''
    return urllib.parse.urlunsplit(
        (parts.scheme, host, parts.path, '', '')
    )[:300]


def log_download(path: str, **record):
    """Append one record to the permanent log."""
    record['url'] = _plain_url(record.get('url', ''))
    record = {'time': datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"), **record}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def read_downloads(path: str, limit: int = 500) -> list[dict]:
    """The log, newest first."""
    try:
        with open(path, encoding='utf-8') as f:
            lines = f.readlines()
    except OSError:
        return []
    records = []
    for line in reversed(lines):
        try:
            records.append(json.loads(line))
        except ValueError:
            continue
        if len(records) >= limit:
            break
    return records
