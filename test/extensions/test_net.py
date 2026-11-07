"""The network an extension gets: declared hosts, downloads, the log."""

import hashlib
import io
import json
import urllib.error
from email.message import Message

import pytest

from modules.extensions import net
from modules.extensions.api import api_for
from modules.extensions.host import downloads_log, extensions
from modules.extensions.loader import load_extensions

PUBLIC = '93.184.216.34'


class Response(io.BytesIO):
    def __init__(self, data=b'', headers=None):
        super().__init__(data)
        self.headers = headers or {'Content-Length': str(len(data))}


def redirect(location, code=307):
    headers = Message()
    headers['Location'] = location
    return urllib.error.HTTPError('u', code, 'redirect', headers, None)


@pytest.fixture(autouse=True)
def own_folder(config, monkeypatch, tmp_path):
    """The data folders and the log live in this test's tmp folder."""
    monkeypatch.setattr(config, 'config_path', str(tmp_path / 'cfg.json'))


@pytest.fixture()
def network(monkeypatch):
    """A fake internet: pages by url, every host resolving to PUBLIC."""
    pages = {}
    seen = []

    def urlopen(request):
        seen.append((request.full_url, dict(request.header_items())))
        page = pages[request.full_url]
        if isinstance(page, Exception):
            raise page
        return Response(page) if isinstance(page, bytes) else page

    monkeypatch.setattr(net, '_urlopen', urlopen)
    monkeypatch.setattr(net, '_resolve', lambda host: [PUBLIC])
    return pages, seen


# ------------------------------------------------------------------ hosts

@pytest.mark.parametrize('pattern, ok', [
    ('example.com', True),
    ('localhost', False),
    ('example.com/path', False),
    ('*.com.', False),
    ('exa mple.com', False),
])
def test_host_patterns(pattern, ok):
    assert net.valid_host_pattern(pattern) is ok


@pytest.mark.parametrize('url', [
    'http://example.com/x',
    'https://other.com/x',
    'ftp://example.com/x',
    '',
])
def test_urls_that_are_refused(url, network):
    with pytest.raises(net.BlockedError):
        net.check_url(url, ['example.com'])


def test_private_addresses_are_refused(monkeypatch):
    for address in ('127.0.0.1', '10.0.0.5', '192.168.1.2', '169.254.1.1',
                    '::1', 'fe80::1%eth0'):
        monkeypatch.setattr(net, '_resolve', lambda host, a=address: [a])
        with pytest.raises(net.BlockedError, match='not a public address'):
            net.check_url('https://example.com/x', ['example.com'])
    monkeypatch.setattr(net, '_resolve', lambda host: [PUBLIC, '10.0.0.1'])
    with pytest.raises(net.BlockedError):
        net.check_url('https://example.com/x', ['example.com'])


# --------------------------------------------------------------- requests

def test_redirects_are_checked_and_headers_stay_on_the_first_host(network):
    pages, seen = network
    pages['https://a.example.com/dl'] = redirect(
        'https://files.cdn.example.net/blob?sig=1')
    pages['https://files.cdn.example.net/blob?sig=1'] = b'{"ok": true}'
    patterns = ['a.example.com', '*.cdn.example.net']
    response, final = net.open_url(
        'https://a.example.com/dl', patterns, {'Authorization': 'Bearer k'})
    assert final == 'https://files.cdn.example.net/blob?sig=1'
    first, second = seen
    assert first[1].get('Authorization') == 'Bearer k'
    assert 'Authorization' not in second[1]


def test_a_redirect_to_an_undeclared_host_is_refused(network):
    pages, seen = network
    pages['https://a.example.com/dl'] = redirect('https://evil.org/steal')
    with pytest.raises(net.BlockedError, match='evil.org'):
        net.open_url('https://a.example.com/dl', ['a.example.com'])
    assert len(seen) == 1                 # evil.org was never asked


def test_redirect_to_http_or_a_loop_is_refused(network):
    pages, _ = network
    pages['https://a.example.com/x'] = redirect('http://a.example.com/y')
    with pytest.raises(net.BlockedError, match='https'):
        net.open_url('https://a.example.com/x', ['a.example.com'])
    pages['https://a.example.com/loop'] = redirect('https://a.example.com/loop')
    with pytest.raises(net.BlockedError, match='redirects'):
        net.open_url('https://a.example.com/loop', ['a.example.com'])


def test_headers_cannot_hold_line_breaks(network):
    with pytest.raises(ValueError):
        net.open_url('https://a.example.com/x', ['a.example.com'],
                     {'X': 'a\r\nHost: evil'})


def test_json_size_is_capped(network, monkeypatch):
    pages, _ = network
    monkeypatch.setattr(net, 'MAX_JSON_BYTES', 10)
    pages['https://a.example.com/big'] = b'[' + b'1,' * 20 + b'1]'
    with pytest.raises(net.BlockedError, match='too big'):
        net.fetch_json('https://a.example.com/big', ['a.example.com'])


# ------------------------------------------------------------- file names

@pytest.mark.parametrize('name', [
    '../x.safetensors',
    'a\\b.safetensors',
    'x\x00.safetensors',
    5,
])
def test_file_names_are_plain(name):
    with pytest.raises(ValueError):
        net.safe_filename(name, model=True)


# --------------------------------------------------- the api, end to end

@pytest.fixture()
def dl_api(make_ext, config, network, monkeypatch, tmp_path):
    """The api of an extension that declared HOSTS, folders in tmp."""
    cfg = make_ext({'dl.py': "HOSTS = ['example.com', '*.cdn.example.net']\n"
                             "def register(api): pass\n"})
    models = tmp_path / 'models' / 'loras'
    monkeypatch.setitem(config.data, 'lora_dir', str(models))
    monkeypatch.setitem(config.data, 'ckpt_dir', '')
    assert load_extensions(cfg) == ['dl']
    return api_for(extensions()[0]), models


def test_a_model_lands_in_the_apps_folder_and_is_logged(dl_api, network):
    api, models = dl_api
    pages, seen = network
    data = b'model-bytes' * 1000
    pages['https://example.com/get'] = redirect(
        'https://x.cdn.example.net/f?X-Amz-Signature=SECRET')
    pages['https://x.cdn.example.net/f?X-Amz-Signature=SECRET'] = data
    steps = []
    result = api.download(
        'https://example.com/get', 'my lora.safetensors', kind='lora',
        headers={'Authorization': 'Bearer k'},
        sha256=hashlib.sha256(data).hexdigest().upper(),
        progress=lambda done, total: steps.append((done, total)))
    assert (models / 'my lora.safetensors').read_bytes() == data
    assert result == {'name': 'my lora.safetensors', 'size': len(data),
                      'sha256': hashlib.sha256(data).hexdigest(),
                      'host': 'x.cdn.example.net'}
    assert steps[-1] == (len(data), len(data))
    assert list(models.glob('*.part')) == []

    (record,) = downloads_log()
    assert record['status'] == 'ok' and record['extension'] == 'dl'
    assert record['kind'] == 'lora' and record['file'] == 'my lora.safetensors'
    assert record['host'] == 'x.cdn.example.net'
    assert record['bytes'] == len(data)
    assert record['url'] == 'https://example.com/get'     # no query
    assert 'SECRET' not in json.dumps(record)
    assert 'Bearer' not in json.dumps(record)


def test_the_path_is_never_given_to_the_extension(dl_api, network):
    api, models = dl_api
    network[0]['https://example.com/a'] = b'x'
    result = api.download('https://example.com/a', 'a.safetensors', 'lora')
    assert str(models) not in json.dumps(result)


@pytest.mark.parametrize('kwargs, error', [
    ({'filename': '../evil.safetensors', 'kind': 'lora'}, ValueError),
    ({'filename': 'a.safetensors', 'kind': 'checkpoint'}, ValueError),
])
def test_bad_requests_never_reach_the_network(dl_api, network, kwargs, error):
    api, models = dl_api
    with pytest.raises(error):
        api.download('https://example.com/a', **kwargs)
    assert network[1] == []
    assert downloads_log() == []


def test_an_existing_file_is_never_replaced(dl_api, network):
    api, models = dl_api
    models.mkdir(parents=True)
    (models / 'a.safetensors').write_bytes(b'mine')
    network[0]['https://example.com/a'] = b'theirs'
    with pytest.raises(FileExistsError):
        api.download('https://example.com/a', 'a.safetensors', 'lora')
    assert (models / 'a.safetensors').read_bytes() == b'mine'
    assert downloads_log()[0]['status'] == 'failed'


def test_failed_and_refused_downloads_are_logged_and_leave_nothing(
    dl_api, network
):
    api, models = dl_api
    pages, _ = network
    pages['https://example.com/bad'] = b'data'
    with pytest.raises(ValueError, match='sha256'):
        api.download('https://example.com/bad', 'b.safetensors', 'lora',
                     sha256='0' * 64)
    with pytest.raises(net.BlockedError, match='HOSTS'):
        api.download('https://evil.org/x', 'c.safetensors', 'lora')

    class Broken(Response):
        def read(self, n=-1):
            raise OSError('connection reset')
    pages['https://example.com/cut'] = Broken(b'x')
    with pytest.raises(OSError):
        api.download('https://example.com/cut', 'd.safetensors', 'lora')

    assert list(models.iterdir()) == []                  # no .part left
    log = downloads_log()
    assert [r['status'] for r in log] == ['failed'] * 3
    assert 'evil.org' in log[1]['error']


def test_without_hosts_nothing_can_be_reached(make_ext, config, network):
    cfg = make_ext({'nh.py': "def register(api): pass\n"})
    load_extensions(cfg)
    api = api_for(extensions()[0])
    with pytest.raises(net.BlockedError, match='none declared'):
        api.fetch_json('https://example.com/x')
    with pytest.raises(net.BlockedError):
        api.download('https://example.com/x', 'a.txt')


def test_fetch_json_goes_through_the_declared_hosts(dl_api, network):
    api, _ = dl_api
    network[0]['https://example.com/api?q=1'] = b'{"items": [1, 2]}'
    assert api.fetch_json('https://example.com/api?q=1') == {'items': [1, 2]}
    with pytest.raises(net.BlockedError):
        api.fetch_json('https://other.org/api')


def test_the_log_is_permanent_and_newest_first(dl_api, network):
    api, _ = dl_api
    for n in range(3):
        network[0][f'https://example.com/{n}'] = b'x'
        api.download(f'https://example.com/{n}', f'{n}.safetensors', 'lora')
    assert [r['file'] for r in downloads_log()] == [
        '2.safetensors', '1.safetensors', '0.safetensors']
    assert net.read_downloads(net.log_path('/nowhere/config.json')) == []


# ------------------------------------------------------------ declaration

def test_hosts_are_read_from_the_constant_and_checked(make_ext, capsys):
    cfg = make_ext({
        'good.py': "HOSTS = ['Example.com', '*.cdn.net']\ndef register(api): pass\n",
        'bad.py': "HOSTS = ['127.0.0.1', 'ok.com']\ndef register(api): pass\n",
        'str.py': "HOSTS = 'one.org'\ndef register(api): pass\n",
    })
    load_extensions(cfg)
    hosts = {e.id: e.hosts for e in extensions()}
    assert hosts == {'good': ['example.com', '*.cdn.net'], 'bad': [],
                     'str': ['one.org']}
    out = capsys.readouterr().out
    assert 'HOSTS has invalid entries' in out


def test_the_connection_goes_to_the_address_that_was_checked(monkeypatch):
    """A second lookup can't give a private address (DNS rebinding)."""
    answers = iter([[PUBLIC], ['10.0.0.1']])
    monkeypatch.setattr(net, '_resolve', lambda host: next(answers))
    monkeypatch.setattr(
        net.socket, 'create_connection',
        lambda *a, **k: pytest.fail('connected to an unchecked address'))
    net.check_url('https://example.com/x', ['example.com'])   # public
    conn = net._PinnedHTTPSConnection('example.com')
    with pytest.raises(net.BlockedError, match='not a public address'):
        conn.connect()                                         # private now
