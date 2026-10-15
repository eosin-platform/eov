"""A client for eov's scripting interface, using only the Python standard library.

eov listens on a local endpoint only, never on the network: a Unix-domain
socket in a folder only you can open (Linux, macOS), or a port on 127.0.0.1
(Windows). A client proves it knows the secret the running eov keeps in its
endpoint's ``.info`` file, which only you can read, and eov proves it knows it
too; then the client sends JSON-RPC calls, one length-prefixed frame each.

    from eov_client import Client

    with Client() as eov:
        print(eov.version()["eov"])
        for slide in eov.slides():
            print(slide["file_id"], slide["filename"], slide["width"], slide["height"])
        tile = eov.read_region(x=1000, y=2000, width=512, height=512)
        with open("tile.png", "wb") as file:
            file.write(tile["data"])

Every method of the interface is ``Client.call(method, params)``; the other
methods are thin wrappers that leave out parameters you do not give. Errors
are raised as :class:`EovError`, whose ``kind`` says what went wrong
(``not_found``, ``invalid_params``, ``refused``, ...). See "Scripting" in
eov's README for every method and its parameters.

Finding the running eov: ``Client(info=...)`` takes the endpoint's ``.info``
file; otherwise the client looks where eov registers (``$EOV_RUNTIME_DIR``,
``$XDG_RUNTIME_DIR/eov`` on Linux, ``$TMPDIR/eov`` on macOS,
``%LOCALAPPDATA%\\eov\\run`` on Windows) for the endpoint eov's default setup
uses, named from a hash of eov's home, configuration file and plugin folder
(pass ``config=`` and ``plugin_dir=`` if eov runs with ``--config`` or
``--plugin-dir``), then for the only endpoint registered there, and finally
asks ``eov script-endpoint --json``.

Run as a program, it calls one method: ``python3 eov_client.py slides`` or
``python3 eov_client.py view.set '{"magnification": 20}'``.
"""

import base64
import glob
import hashlib
import hmac
import json
import os
import select
import shutil
import socket
import struct
import subprocess
import sys

__all__ = ["Client", "Endpoint", "EovError", "endpoint_name", "find_endpoint", "runtime_dirs"]

PROTOCOL = b"eov-rpc1"
NONCE_BYTES = 32
PROOF_BYTES = 32
SECRET_BYTES = 32
MAX_CALL_BYTES = 1 << 20
MAX_ANSWER_BYTES = 64 << 20
FLATPAK_ID = "io.eosin.eov"


class EovError(Exception):
    """An error eov answered a call with, or one met while reaching eov.

    ``kind`` is eov's error kind (``parse_error``, ``invalid_request``,
    ``method_not_found``, ``invalid_params``, ``internal``, ``not_found``,
    ``refused``, ``too_large``, ``busy``, ``failed``, ``unavailable``,
    ``cancelled``, ``disabled``), or ``not_running`` and ``connection`` for
    problems on the way; ``code`` is the JSON-RPC error code.
    """

    def __init__(self, message, kind="connection", code=None):
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.code = code

    def __str__(self):
        return "%s (%s)" % (self.message, self.kind)


# -- finding the running eov -------------------------------------------------


def _env_path(name):
    value = os.environ.get(name)
    return value if value else None


def _user_home():
    return os.path.expanduser("~")


def eov_home():
    """eov's home folder: ``$EOV_HOME``, else ``~/.eov``."""
    return _env_path("EOV_HOME") or os.path.join(_user_home(), ".eov")


def _data_dir():
    """eov's data folder (``$EOV_DATA_DIR``, else the platform's local data folder)."""
    explicit = _env_path("EOV_DATA_DIR")
    if explicit:
        return explicit
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA")
        if local:
            return os.path.join(local, "eov")
        return os.path.join(eov_home(), "data")
    if sys.platform == "darwin":
        return os.path.join(_user_home(), "Library", "Application Support", "eov")
    base = os.environ.get("XDG_DATA_HOME")
    if not (base and os.path.isabs(base)):
        base = os.path.join(_user_home(), ".local", "share")
    return os.path.join(base, "eov")


def runtime_dirs():
    """The folders where a running eov may register, most likely first."""
    explicit = os.environ.get("EOV_RUNTIME_DIR")
    if explicit and os.path.isabs(explicit):
        return [explicit]
    dirs = []
    if os.name == "posix" and sys.platform != "darwin":
        runtime = os.environ.get("XDG_RUNTIME_DIR")
        if runtime and os.path.isabs(runtime):
            dirs.append(os.path.join(runtime, "eov"))
            dirs.append(os.path.join(runtime, "app", FLATPAK_ID))
    if sys.platform == "darwin":
        temp = os.environ.get("TMPDIR")
        if temp and os.path.isabs(temp):
            dirs.append(os.path.join(temp, "eov"))
    dirs.append(os.path.join(_data_dir(), "run"))
    return dirs


def endpoint_name(home=None, config=None, plugin_dir=None):
    """The name of the endpoint of the eov with this home, configuration file
    and plugin folder (defaults: eov's), computed as eov does: ``eov-`` and
    the hex of the first 8 bytes of SHA-256 of the three paths, each followed
    by a NUL byte."""
    home = os.path.expanduser(home) if home is not None else eov_home()
    if config is None:
        config = _env_path("EOV_CONFIG") or os.path.join(home, "config.toml")
    if plugin_dir is None:
        plugin_dir = _env_path("EOV_PLUGIN_DIR") or os.path.join(home, "plugins")
    digest = hashlib.sha256()
    for part in (home, os.path.expanduser(config), os.path.expanduser(plugin_dir)):
        digest.update(os.fsencode(part))
        digest.update(b"\0")
    return "eov-" + digest.digest()[:8].hex()


class Endpoint:
    """Where a running eov listens, as its ``.info`` file says: a port (0 for
    the Unix-domain socket next to the file), the process id and the secret."""

    def __init__(self, info_path):
        self.info_path = os.fspath(info_path)
        try:
            with open(self.info_path, "r", encoding="ascii") as file:
                fields = file.read().split()
        except OSError as error:
            raise EovError("cannot read %s: %s" % (self.info_path, error), "not_running")
        try:
            self.port, self.pid = int(fields[0]), int(fields[1])
            self.secret = bytes.fromhex(fields[2])
        except (IndexError, ValueError):
            raise EovError("%s is not an eov endpoint" % self.info_path, "not_running")
        if len(self.secret) != SECRET_BYTES or len(fields) != 3:
            raise EovError("%s is not an eov endpoint" % self.info_path, "not_running")
        base = self.info_path[: -len(".info")] if self.info_path.endswith(".info") else self.info_path
        self.socket_path = base + ".sock" if self.port == 0 else None

    def __repr__(self):
        where = self.socket_path or "127.0.0.1:%d" % self.port
        return "Endpoint(%r, pid=%d)" % (where, self.pid)

    def connect(self, timeout=10.0):
        """A socket connected to the endpoint (not yet authenticated)."""
        try:
            if self.socket_path is not None:
                if not hasattr(socket, "AF_UNIX"):
                    raise EovError("this Python cannot open Unix-domain sockets")
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    sock.settimeout(timeout)
                    sock.connect(self.socket_path)
                except OSError:
                    sock.close()
                    raise
                return sock
            sock = socket.create_connection(("127.0.0.1", self.port), timeout=timeout)
            # Calls are small frames: send them at once.
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return sock
        except OSError as error:
            raise EovError("no eov answers at %r: %s" % (self, error), "not_running")

    def answers(self, timeout=2.0):
        """Whether something listens there (a crashed eov leaves its file)."""
        try:
            self.connect(timeout).close()
            return True
        except EovError:
            return False


def _eov_endpoint(eov, config, plugin_dir):
    """The ``.info`` file ``eov script-endpoint --json`` names, if it can run."""
    executable = eov or os.environ.get("EOV") or shutil.which("eov")
    if not executable:
        return None
    command = [executable]
    if config is not None:
        command += ["--config", os.fspath(config)]
    if plugin_dir is not None:
        command += ["--plugin-dir", os.fspath(plugin_dir)]
    command += ["script-endpoint", "--json"]
    try:
        output = subprocess.run(command, capture_output=True, check=True, timeout=30).stdout
        return json.loads(output)["info"]
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
        return None


def find_endpoint(config=None, plugin_dir=None, eov=None):
    """The endpoint of the running eov (see the module documentation)."""
    name = endpoint_name(config=config, plugin_dir=plugin_dir)
    candidates = [os.path.join(folder, name + ".info") for folder in runtime_dirs()]
    for path in candidates:
        if os.path.isfile(path):
            endpoint = Endpoint(path)
            if endpoint.answers():
                return endpoint
    if config is None and plugin_dir is None:
        # eov runs with another setup: the only endpoint registered.
        found = []
        for folder in runtime_dirs():
            for path in sorted(glob.glob(os.path.join(glob.escape(folder), "eov-*.info"))):
                try:
                    endpoint = Endpoint(path)
                except EovError:
                    continue
                if endpoint.answers():
                    found.append(endpoint)
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            raise EovError(
                "several eovs are running (%s); pass info=, or config= and plugin_dir="
                % ", ".join(endpoint.info_path for endpoint in found),
                "not_running",
            )
    info = _eov_endpoint(eov, config, plugin_dir)
    if info and os.path.isfile(info):
        return Endpoint(info)
    raise EovError(
        "no eov is running (looked for %s in %s)" % (name, ", ".join(runtime_dirs())),
        "not_running",
    )


# -- the protocol ------------------------------------------------------------------


def _recv_exact(sock, count):
    chunks = []
    while count > 0:
        chunk = sock.recv(min(count, 1 << 20))
        if not chunk:
            raise EovError("eov hung up", "connection")
        chunks.append(chunk)
        count -= len(chunk)
    return b"".join(chunks)


def _handshake(sock, secret):
    """Proves the secret to eov after eov proved it: nothing is sent to a
    program that does not know it."""
    client_nonce = os.urandom(NONCE_BYTES)
    sock.sendall(PROTOCOL + client_nonce)
    answer = _recv_exact(sock, PROOF_BYTES + NONCE_BYTES)
    server_proof, server_nonce = answer[:PROOF_BYTES], answer[PROOF_BYTES:]
    expected = hmac.new(secret, b"server" + client_nonce, hashlib.sha256).digest()
    if not hmac.compare_digest(server_proof, expected):
        raise EovError("the program at the endpoint is not the eov that registered there")
    sock.sendall(hmac.new(secret, b"client" + server_nonce, hashlib.sha256).digest())


def _read_frame(sock):
    (length,) = struct.unpack(">I", _recv_exact(sock, 4))
    if length > MAX_ANSWER_BYTES:
        raise EovError("an answer of %d bytes is larger than the protocol allows" % length)
    return _recv_exact(sock, length)


def _path_params(path):
    """``path`` as eov takes it: absolute, as text, or as ``os_path`` (hex
    of the platform's encoding) when it is not valid Unicode."""
    path = os.path.abspath(os.fspath(path))
    if isinstance(path, bytes):
        if os.name == "nt":
            return {"path": os.fsdecode(path)}
        try:
            return {"path": path.decode("utf-8")}
        except UnicodeDecodeError:
            return {"os_path": path.hex()}
    try:
        path.encode("utf-8")
        return {"path": path}
    except UnicodeEncodeError:
        if os.name == "nt":
            units = path.encode("utf-16-le", "surrogatepass")
            return {"os_path": "".join("%04x" % unit for (unit,) in struct.iter_unpack("<H", units))}
        return {"os_path": os.fsencode(path).hex()}


def _params(**values):
    """The parameters that were given (``None`` leaves one out)."""
    return {key: value for key, value in values.items() if value is not None}


class Client:
    """A connection to the running eov.

    ``timeout`` bounds each call (exports and large reads can take a while).
    The connection is opened on the first call and opened again when eov
    closed it after being idle; use the client as a context manager, or call
    :meth:`close`.
    """

    def __init__(self, endpoint=None, *, info=None, config=None, plugin_dir=None, eov=None, timeout=600.0):
        if endpoint is None:
            endpoint = Endpoint(info) if info is not None else find_endpoint(config, plugin_dir, eov)
        self.endpoint = endpoint
        self.timeout = timeout
        self._sock = None
        self._next_id = 1

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def _hung_up(self):
        """Whether eov closed the connection (it does after two idle minutes)."""
        try:
            readable, _, _ = select.select([self._sock], [], [], 0)
            return bool(readable) and self._sock.recv(1, socket.MSG_PEEK) == b""
        except (OSError, ValueError):
            return True

    def _connection(self):
        if self._sock is not None and self._hung_up():
            self.close()
        if self._sock is None:
            sock = self.endpoint.connect(min(self.timeout, 10.0))
            try:
                sock.settimeout(min(self.timeout, 10.0))
                _handshake(sock, self.endpoint.secret)
            except (OSError, EovError) as error:
                sock.close()
                if isinstance(error, EovError):
                    raise
                raise EovError("cannot reach eov: %s" % error)
            self._sock = sock
        return self._sock

    def call(self, method, params=None):
        """Calls ``method`` with ``params`` (a dict) and returns its result."""
        request = {"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params or {}}
        self._next_id += 1
        body = json.dumps(request, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(body) > MAX_CALL_BYTES:
            raise EovError("the call is larger than eov accepts", "too_large")
        sock = self._connection()
        try:
            sock.settimeout(self.timeout)
            sock.sendall(struct.pack(">I", len(body)) + body)
            reply = json.loads(_read_frame(sock).decode("utf-8"))
        except (OSError, ValueError, EovError) as error:
            self.close()
            raise EovError("the call %s broke off: %s" % (method, error))
        error = reply.get("error")
        if error is not None:
            kind = (error.get("data") or {}).get("kind", "error")
            raise EovError(error.get("message", "the call failed"), kind, error.get("code"))
        return reply.get("result")

    # -- the viewer ------------------------------------------------------------

    def version(self):
        """eov's version, the protocol version, the methods and the limits."""
        return self.call("version")

    def slides(self):
        """The open slides (``file_id``, ``filename``, ``path``, size, levels, pixel size, ``identity``, ...)."""
        return self.call("slides")

    def panes(self):
        """The panes, left to right, with the slide each shows and its view."""
        return self.call("panes")

    def open_slide(self, path, pane=None, new_pane=False, view=None):
        """Opens a slide (or shows it, if it is open) and returns its ``file_id``,
        ``pane`` and ``view`` once it is shown. ``view`` is a dict with ``x``,
        ``y`` (level-0 pixels, or text such as ``"300um"``) and ``zoom`` (screen
        pixels per level-0 pixel, ``"20x"`` or ``"fit"``) or ``magnification``."""
        return self.call("open", dict(_path_params(path), **_params(pane=pane, new_pane=new_pane or None, view=view)))

    def close_slide(self, slide):
        """Closes every tab of the slide."""
        return self.call("close", {"slide": slide})

    def view(self, pane=None):
        """A pane's view (default: the focused pane)."""
        return self.call("view.get", _params(pane=pane))

    def set_view(self, pane=None, x=None, y=None, zoom=None, magnification=None, rotation=None, flipped=None, animate=None):
        """Moves a pane's view; returns the view it shows now."""
        return self.call(
            "view.set",
            _params(pane=pane, x=x, y=y, zoom=zoom, magnification=magnification, rotation=rotation, flipped=flipped, animate=animate),
        )

    # -- pixels ------------------------------------------------------------------

    def read_region(self, x, y, width, height, slide=None, path=None, level=None, downsample=None, mpp=None, format="png", alpha=False):
        """Reads a region: ``x``, ``y`` are its top-left corner in level-0
        pixels, ``width`` and ``height`` the size of the result, at ``level``,
        ``downsample`` (level-0 pixels per result pixel) or ``mpp`` (µm per
        pixel; default: level 0). From an open ``slide`` (default: the focused
        one) or any slide file at ``path``, which need not be open.

        Returns the answer with ``data`` as bytes: a PNG file for ``"png"``,
        rows of RGBA or RGB pixels for ``"rgba"`` and ``"rgb"`` (with numpy:
        ``numpy.frombuffer(r["data"], numpy.uint8).reshape(r["height"], r["width"], 4)``).
        """
        params = _params(slide=slide, x=x, y=y, width=width, height=height, level=level, downsample=downsample, mpp=mpp, format=format, alpha=alpha or None)
        if path is not None:
            params.update(_path_params(path))
        result = self.call("region.read", params)
        result["data"] = base64.b64decode(result["data"])
        return result

    # -- annotations ---------------------------------------------------------------

    def roi(self, slide=None):
        """The slide's region of interest (``x``, ``y``, ``width``, ``height``), or ``None``."""
        return self.call("roi.get", _params(slide=slide))["roi"]

    def set_roi(self, x=None, y=None, width=None, height=None, slide=None, clear=False):
        """Sets the region of interest (or clears it with ``clear=True``); undoable in eov."""
        if clear:
            params = dict(_params(slide=slide), roi=None)
        else:
            params = _params(slide=slide, x=x, y=y, width=width, height=height)
        return self.call("roi.set", params)["roi"]

    def measurements(self, slide=None):
        """The slide's measurements, with their values in µm and the label eov shows."""
        return self.call("measurements.list", _params(slide=slide))

    def add_measurement(self, mode, points=None, shape=None, x=None, y=None, width=None, height=None, slide=None):
        """Adds a measurement: ``mode`` is line, polyline, polygon, angle (with
        ``points``) or rectangle, ellipse (with ``x``, ``y``, ``width``,
        ``height``); or give the plugin API's ``shape``."""
        return self.call(
            "measurements.add",
            _params(slide=slide, mode=mode, points=points, shape=shape, x=x, y=y, width=width, height=height),
        )

    def remove_measurements(self, ids=None, all=False, slide=None):
        """Deletes measurements by ``measurement_id``, or all of them."""
        return self.call("measurements.remove", _params(slide=slide, ids=ids, all=all or None))

    def bookmarks(self, slide=None):
        """The slide's bookmarks."""
        return self.call("bookmarks.list", _params(slide=slide))

    def add_bookmark(self, name=None, slide=None, x=None, y=None, zoom=None, magnification=None, rotation=None, flipped=None):
        """Bookmarks a view of the slide (by default the one it shows)."""
        return self.call(
            "bookmarks.add",
            _params(slide=slide, name=name, x=x, y=y, zoom=zoom, magnification=magnification, rotation=rotation, flipped=flipped),
        )

    # -- analysis --------------------------------------------------------------------

    def quantify(self, slide=None, model=None, stain=None, thresholds=None, region=None):
        """Positive-pixel analysis and H-score of one stain in the region of
        interest (or ``region``), as eov's Quantify Stain panel measures it."""
        return self.call("quantify", _params(slide=slide, model=model, stain=stain, thresholds=thresholds, region=region))

    def export_view(self, path, pane=None, format=None, dpi=None, jpeg_quality=None, scale_bar=None, roi=None, measurements=None, processing_note=None, overwrite=False):
        """Exports a pane's view as eov's export dialog does (PNG, JPEG or TIFF)."""
        params = _params(
            pane=pane,
            format=format,
            dpi=dpi,
            jpeg_quality=jpeg_quality,
            scale_bar=scale_bar,
            roi=roi,
            measurements=measurements,
            processing_note=processing_note,
            overwrite=overwrite or None,
        )
        params.update(_path_params(path))
        return self.call("export.view", params)


def main(argv=None):
    """``python3 eov_client.py METHOD [PARAMS]``: calls one method, prints the result."""
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: eov_client.py METHOD [PARAMS-AS-JSON]", file=sys.stderr)
        return 2
    try:
        params = json.loads(argv[1]) if len(argv) > 1 else {}
    except ValueError as error:
        print("eov_client: the parameters are not JSON: %s" % error, file=sys.stderr)
        return 2
    if isinstance(params, dict) and isinstance(params.get("path"), str):
        params.update(_path_params(params.pop("path")))
    try:
        with Client() as eov:
            print(json.dumps(eov.call(argv[0], params), indent=2))
    except EovError as error:
        print("eov_client: %s" % error, file=sys.stderr)
        return 3 if error.kind == "not_running" else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
