"""Discovery of the avatars installed on the robot plus the JSON payload the
operator app (TeleportVR) consumes.

This module is deliberately free of ROS dependencies so it can be unit tested on
a workstation.  The ROS plumbing lives in ``ros2_avatar_list.py``.

Payload schema (version 1)::

    {
      "version": 1,
      "revision": "c41f9a",
      "active": "raf",
      "avatars": [
        {"id": "raf", "display_name": "Rafael", "thumbnail_jpeg_base64": "..."}
      ]
    }

* ``id`` is exactly the string the node accepts on ``/operator/name``.
* ``revision`` is a hash over ids, display names and thumbnails -- *not* over
  ``active`` -- so the app only re-decodes thumbnails when it really changed.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

PAYLOAD_VERSION = 1

DEFAULT_AVATAR_ROOT = "/workspace/avatars"

THUMBNAIL_SIZE = 192          # square, px
THUMBNAIL_QUALITY = 85        # initial JPEG quality
THUMBNAIL_MIN_QUALITY = 40    # lowest quality we fall back to for the size cap
THUMBNAIL_MAX_BYTES = 20 * 1024

THUMBNAIL_FILENAMES = ("thumbnail.jpg", "thumbnail.jpeg", "thumbnail.png")
METADATA_FILENAME = "avatar.json"

#: Folder name used for the non-splat "AI face" that the node renders itself.
AI_FACE_TARGET = "ai"

#: Spherical harmonics degree the viewer loads avatars with (``--sh-degree`` in
#: run.sh). ``GaussianModel.load_ply`` asserts the matching f_rest_* count.
VIEWER_SH_DEGREE = 0
#: Properties ``GaussianModel.load_ply`` reads from the first PLY element.
PLY_REQUIRED_PROPERTIES = ("x", "y", "z", "opacity", "f_dc_0", "f_dc_1", "f_dc_2")
#: Property families it reads by prefix; ``binding`` is optional there, but the
#: load crashes on ``self.binding[mask]`` without it.
PLY_REQUIRED_PREFIXES = ("scale_", "rot", "binding")
#: Arrays ``FlameGaussianModel.load_ply`` takes from the avatar's .npz, which the
#: viewer passes as both the fine-tuned FLAME parameters and the motion.
FLAME_PARAM_KEYS = ("shape", "static_offset", "translation", "rotation", "neck_pose",
                    "jaw_pose", "eyes_pose", "expr", "dynamic_offset")
#: Upper bound for a PLY header; a real one is a few hundred bytes.
PLY_HEADER_MAX_BYTES = 1 << 20


def read_ply_header(path: Path) -> List[Tuple[str, List[str]]]:
    """``[(element name, [property names])]`` from a PLY header, data unread."""
    elements: List[Tuple[str, List[str]]] = []
    with open(path, "rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError("not a PLY file")
        read = 0
        while True:
            line = handle.readline()
            read += len(line)
            if not line or read > PLY_HEADER_MAX_BYTES:
                raise ValueError("PLY header has no end_header")
            words = line.decode("ascii", "replace").split()
            if not words:
                continue
            if words[0] == "end_header":
                return elements
            if words[0] == "element" and len(words) >= 2:
                elements.append((words[1], []))
            elif words[0] == "property" and elements:
                elements[-1][1].append(words[-1])


def check_avatar_folder(folder: Path, sh_degree: int = VIEWER_SH_DEGREE) -> Optional[str]:
    """Why the viewer could not load this folder as an avatar, or None if it can.

    Mirrors what ``LocalViewer.load_avatar`` and ``load_ply`` need, from the
    PLY header and the .npz's table of contents only, so it is cheap enough
    to run on every scan.  The viewer takes whichever .ply and .npz it finds
    first, so every one in the folder has to pass.
    """
    import zipfile

    ply_files = sorted(folder.glob("*.ply"))
    if not ply_files:
        return "no .ply"
    n_rest = 3 * (sh_degree + 1) ** 2 - 3
    for ply in ply_files:
        try:
            elements = read_ply_header(ply)
        except (OSError, ValueError) as exc:
            return f"{ply.name}: {exc}"
        if not elements or elements[0][0] != "vertex":
            first = elements[0][0] if elements else "none"
            return f"{ply.name}: first element is '{first}', not 'vertex'"
        props = elements[0][1]
        missing = [p for p in PLY_REQUIRED_PROPERTIES if p not in props]
        missing += [p + "*" for p in PLY_REQUIRED_PREFIXES if not any(q.startswith(p) for q in props)]
        if missing:
            return f"{ply.name}: missing {', '.join(missing)}"
        rest = sum(1 for p in props if p.startswith("f_rest_"))
        if rest != n_rest:
            return f"{ply.name}: {rest} f_rest_* properties, the viewer expects {n_rest}"

    npz_files = sorted(folder.glob("*.npz"))
    if not npz_files:
        return "no flame_param.npz"
    for npz in npz_files:
        try:
            with zipfile.ZipFile(npz) as archive:
                keys = {name[:-4] for name in archive.namelist() if name.endswith(".npy")}
        except (OSError, zipfile.BadZipFile) as exc:
            return f"{npz.name}: {exc}"
        missing = [k for k in FLAME_PARAM_KEYS if k not in keys]
        if missing:
            return f"{npz.name}: missing {', '.join(missing)}"
    return None

#: Legacy ids the operator app has been sending since before this list existed.
#: ``NAME_MAP`` in ``ros2_blendshape_node.py`` is the single source of truth; it
#: is injected into the registry so the two can never drift apart.
#: Ordering matters: it is the order the app showed the faces in, and we keep it.


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Avatar:
    """One entry of the list the app shows."""

    id: str                      # what the app sends on /operator/name
    display_name: str            # UTF-8 label shown in the headset
    target: str                  # folder name under the avatar root, or "ai"
    folder: Optional[Path]       # None for virtual avatars (the AI face)
    thumbnail_jpeg: bytes        # 192x192 JPEG, <= THUMBNAIL_MAX_BYTES
    aliases: Tuple[str, ...] = ()  # further strings accepted on /operator/name

    @property
    def is_ai_face(self) -> bool:
        return self.target == AI_FACE_TARGET

    def to_json_dict(self) -> dict:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "thumbnail_jpeg_base64": base64.b64encode(self.thumbnail_jpeg).decode("ascii"),
        }


# --------------------------------------------------------------------------- #
# Thumbnails
# --------------------------------------------------------------------------- #

def encode_thumbnail(image: Image.Image,
                     size: int = THUMBNAIL_SIZE,
                     quality: int = THUMBNAIL_QUALITY,
                     max_bytes: int = THUMBNAIL_MAX_BYTES) -> bytes:
    """Centre-crop ``image`` to a square, resize to ``size`` and encode as JPEG.

    Quality is stepped down until the result fits into ``max_bytes`` so the
    payload stays small even for noisy renders.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    w, h = image.size
    if w != h:
        edge = min(w, h)
        left = (w - edge) // 2
        top = (h - edge) // 2
        image = image.crop((left, top, left + edge, top + edge))

    if image.size != (size, size):
        image = image.resize((size, size), Image.LANCZOS)

    data = b""
    for q in range(quality, THUMBNAIL_MIN_QUALITY - 1, -5):
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=q, optimize=True)
        data = buf.getvalue()
        if len(data) <= max_bytes:
            return data
    return data  # best effort; caller logs if this is still too large


def _to_pil(obj) -> Optional[Image.Image]:
    """Accept what a renderer hook may hand us: PIL image, ndarray or bytes."""
    if obj is None:
        return None
    if isinstance(obj, Image.Image):
        return obj
    if isinstance(obj, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(obj)))
    if isinstance(obj, np.ndarray):
        arr = obj
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L")
        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            return Image.fromarray(arr[:, :, :3], mode="RGB")
    raise TypeError(f"unsupported thumbnail type: {type(obj)!r}")


def placeholder_thumbnail(label: str, size: int = THUMBNAIL_SIZE) -> bytes:
    """Deterministic stand-in so an avatar without artwork still shows up.

    Deterministic matters: a random image would change ``revision`` on every
    scan and make the app re-decode every thumbnail.
    """
    # Hue from the label so faces stay visually distinguishable in the picker.
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    base = (40 + digest[0] % 60, 40 + digest[1] % 60, 60 + digest[2] % 60)

    image = Image.new("RGB", (size, size), base)
    draw = ImageDraw.Draw(image)

    initials = "".join(part[0] for part in label.replace("_", " ").replace("-", " ").split()[:2])
    initials = (initials or label[:2]).upper()

    # Default bitmap font; no font files to ship, no fontconfig on the robot.
    try:
        bbox = draw.textbbox((0, 0), initials)
        scale = max(1, int(size * 0.45 / max(1, bbox[3] - bbox[1])))
        tile = Image.new("RGB", (bbox[2] - bbox[0] + 4, bbox[3] - bbox[1] + 4), base)
        ImageDraw.Draw(tile).text((2 - bbox[0], 2 - bbox[1]), initials, fill=(235, 235, 235))
        tile = tile.resize((tile.width * scale, tile.height * scale), Image.NEAREST)
        image.paste(tile, ((size - tile.width) // 2, (size - tile.height) // 2))
    except Exception:
        draw.text((size // 2 - 8, size // 2 - 4), initials, fill=(235, 235, 235))

    return encode_thumbnail(image)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

class _NullLogger:
    def info(self, *_a, **_kw):
        pass

    debug = warning = error = info


class AvatarRegistry:
    """Scans the avatar directory and caches the encoded payload.

    The payload is built once per scan (and once per ``active`` change, which is
    cheap and rare), never per service call or publish.
    """

    def __init__(self,
                 root: os.PathLike | str = DEFAULT_AVATAR_ROOT,
                 legacy_name_map: Optional[Dict[str, str]] = None,
                 thumbnail_renderer: Optional[Callable[[str, Optional[Path]], object]] = None,
                 thumbnail_cache_dir: Optional[os.PathLike | str] = None,
                 logger=None):
        self.root = Path(root)
        self.logger = logger or _NullLogger()
        self.thumbnail_renderer = thumbnail_renderer

        # id -> folder name, in the order the app used to show them.
        self.legacy_name_map: Dict[str, str] = dict(legacy_name_map or {})
        self._legacy_order = list(self.legacy_name_map.keys())

        # Rendered thumbnails are expensive, so they survive restarts on disk.
        self.thumbnail_cache_dir = Path(thumbnail_cache_dir) if thumbnail_cache_dir \
            else self.root / ".thumbnail_cache"

        self._avatars: List[Avatar] = []
        self._by_lookup: Dict[str, Avatar] = {}
        self._revision = ""
        self._active = ""
        self._payload = ""
        self._file_cache: Dict[Tuple[str, int, int], bytes] = {}
        # folder name -> why it is left out, so each reason is logged once
        # instead of on every rescan.
        self._rejected: Dict[str, str] = {}
        self.last_scan_time = 0.0

    # -- public state ------------------------------------------------------ #

    @property
    def avatars(self) -> Sequence[Avatar]:
        return tuple(self._avatars)

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def active(self) -> str:
        return self._active

    @property
    def payload(self) -> str:
        """Cached JSON string; empty before the first successful scan."""
        return self._payload

    def resolve(self, name: Optional[str]) -> Optional[Avatar]:
        """Map anything the app may send on /operator/name to an avatar.

        Accepts the canonical id, any legacy alias and the plain folder name,
        case-insensitively.
        """
        if not name:
            return None
        return self._by_lookup.get(name.strip().lower())

    def id_for_target(self, target: str) -> str:
        for avatar in self._avatars:
            if avatar.target == target:
                return avatar.id
        return ""

    def set_active(self, avatar_id: Optional[str]) -> bool:
        """Record what is being rendered. Returns True when it changed."""
        new_active = avatar_id or ""
        if new_active and new_active not in {a.id for a in self._avatars}:
            resolved = self.resolve(new_active)
            new_active = resolved.id if resolved else ""
        if new_active == self._active:
            return False
        self._active = new_active
        self._rebuild_payload()
        return True

    # -- scanning ---------------------------------------------------------- #

    def directory_signature(self) -> Tuple:
        """Cheap fingerprint used to decide whether a full rescan is needed."""
        entries = []
        try:
            for child in sorted(self.root.iterdir(), key=lambda p: p.name):
                if child.name.startswith("."):
                    continue
                try:
                    st = child.stat()
                except OSError:
                    continue
                entries.append((child.name, int(st.st_mtime_ns)))
                if child.is_dir():
                    for extra in (METADATA_FILENAME,) + THUMBNAIL_FILENAMES:
                        try:
                            est = (child / extra).stat()
                        except OSError:
                            continue
                        entries.append((f"{child.name}/{extra}", int(est.st_mtime_ns), est.st_size))
        except OSError as exc:
            self.logger.warning(f"[avatar-list] cannot read avatar root {self.root}: {exc}")
            return ("unreadable",)
        return tuple(entries)

    def scan(self) -> bool:
        """Rebuild the list. Returns True when the payload changed."""
        avatars: List[Avatar] = []
        seen_targets: set = set()

        rejected: Dict[str, str] = {}
        for folder in self._candidate_folders():
            try:
                avatar = self._build_avatar(folder)
            except Exception as exc:  # a broken avatar must not kill the list
                rejected[folder.name] = str(exc)
                if self._rejected.get(folder.name) != str(exc):
                    self.logger.warning(f"[avatar-list] skipping '{folder.name}': {exc}")
                continue
            if avatar is None:
                continue
            avatars.append(avatar)
            seen_targets.add(avatar.target)
        self._rejected = rejected

        # The AI face is rendered by the node itself, so it has no folder of its
        # own on most robots -- keep offering it as long as a legacy id maps to
        # it, otherwise the app would lose a face it has today.
        if AI_FACE_TARGET not in seen_targets and AI_FACE_TARGET in self.legacy_name_map.values():
            try:
                avatars.append(self._build_virtual_ai_avatar())
            except Exception as exc:
                self.logger.warning(f"[avatar-list] skipping AI face: {exc}")

        avatars.sort(key=self._sort_key)

        revision = self._compute_revision(avatars)
        changed = (revision != self._revision) or not self._payload

        self._avatars = avatars
        self._revision = revision
        self._by_lookup = self._build_lookup(avatars)
        self.last_scan_time = time.time()

        # An avatar that disappeared must not stay 'active'.
        if self._active and self._active not in {a.id for a in avatars}:
            self._active = ""
            changed = True

        if changed:
            self._rebuild_payload()
        return changed

    # -- internals --------------------------------------------------------- #

    def _candidate_folders(self) -> Iterable[Path]:
        try:
            children = sorted(self.root.iterdir(), key=lambda p: p.name.lower())
        except OSError as exc:
            self.logger.warning(f"[avatar-list] cannot list {self.root}: {exc}")
            return []
        return [c for c in children if c.is_dir() and not c.name.startswith(".")]

    def _build_avatar(self, folder: Path) -> Optional[Avatar]:
        # An avatar folder is one the viewer can actually load: it holds a
        # gaussian splat. Everything else in the avatar root (videos, images,
        # caches) is not an avatar.
        if not any(folder.glob("*.ply")):
            self.logger.debug(f"[avatar-list] '{folder.name}' has no .ply, not an avatar")
            return None
        # A .ply alone is not enough: offering a folder the viewer then fails
        # to load would leave the operator with a broken face.
        reason = check_avatar_folder(folder)
        if reason:
            raise ValueError(f"not loadable by the viewer: {reason}")

        target = folder.name
        ids = self._ids_for_target(target)
        avatar_id = ids[0] if ids else target
        aliases = tuple(ids[1:]) if ids else ()

        display_name = self._read_display_name(folder) or avatar_id
        thumbnail = self._thumbnail_for(avatar_id, folder)

        return Avatar(id=avatar_id, display_name=display_name, target=target,
                      folder=folder, thumbnail_jpeg=thumbnail, aliases=aliases)

    def _build_virtual_ai_avatar(self) -> Avatar:
        ids = self._ids_for_target(AI_FACE_TARGET)
        avatar_id = ids[0] if ids else AI_FACE_TARGET
        folder = self.root / AI_FACE_TARGET
        display_name = (self._read_display_name(folder) if folder.is_dir() else None) or avatar_id
        thumbnail = self._thumbnail_for(avatar_id, folder if folder.is_dir() else None)
        return Avatar(id=avatar_id, display_name=display_name, target=AI_FACE_TARGET,
                      folder=None, thumbnail_jpeg=thumbnail, aliases=tuple(ids[1:]))

    def _ids_for_target(self, target: str) -> List[str]:
        """Legacy ids pointing at this folder, in the app's historical order."""
        return [name for name in self._legacy_order if self.legacy_name_map[name] == target]

    def _read_display_name(self, folder: Path) -> Optional[str]:
        meta_path = folder / METADATA_FILENAME
        if not meta_path.is_file():
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
        except (OSError, ValueError) as exc:
            self.logger.warning(f"[avatar-list] bad {METADATA_FILENAME} in '{folder.name}': {exc}")
            return None
        name = meta.get("display_name")
        if isinstance(name, str) and name.strip():
            return name.strip()
        return None

    def _thumbnail_for(self, avatar_id: str, folder: Optional[Path]) -> bytes:
        if folder is not None:
            for filename in THUMBNAIL_FILENAMES:
                path = folder / filename
                if not path.is_file():
                    continue
                try:
                    return self._encode_file(path)
                except Exception as exc:
                    self.logger.warning(f"[avatar-list] unusable {filename} for '{avatar_id}': {exc}")

        rendered = self._rendered_thumbnail(avatar_id, folder)
        if rendered is not None:
            return rendered

        return placeholder_thumbnail(avatar_id)

    def _encode_file(self, path: Path) -> bytes:
        st = path.stat()
        key = (str(path), int(st.st_mtime_ns), st.st_size)
        cached = self._file_cache.get(key)
        if cached is not None:
            return cached
        with Image.open(path) as image:
            image.load()
            data = encode_thumbnail(image)
        if len(data) > THUMBNAIL_MAX_BYTES:
            self.logger.warning(
                f"[avatar-list] {path.name} still {len(data)} B after re-encoding")
        if len(self._file_cache) > 128:   # bounded: keys carry mtime/size
            self._file_cache.clear()
        self._file_cache[key] = data
        return data

    def _rendered_thumbnail(self, avatar_id: str, folder: Optional[Path]) -> Optional[bytes]:
        """Neutral frontal frame rendered by the node, cached on disk.

        The render itself is supplied by the viewer through ``thumbnail_renderer``
        because only the viewer owns the GPU context and the FLAME model.
        """
        cache_path = self.thumbnail_cache_dir / f"{avatar_id}.jpg"
        if cache_path.is_file():
            try:
                return self._encode_file(cache_path)
            except Exception as exc:
                self.logger.warning(f"[avatar-list] bad cached thumbnail for '{avatar_id}': {exc}")

        if self.thumbnail_renderer is None:
            return None

        try:
            image = _to_pil(self.thumbnail_renderer(avatar_id, folder))
        except Exception as exc:
            self.logger.warning(f"[avatar-list] rendering thumbnail for '{avatar_id}' failed: {exc}")
            return None
        if image is None:
            return None

        data = encode_thumbnail(image)
        try:
            self.thumbnail_cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(".jpg.tmp")
            tmp.write_bytes(data)
            tmp.replace(cache_path)
        except OSError as exc:
            self.logger.warning(f"[avatar-list] cannot cache thumbnail for '{avatar_id}': {exc}")
        return data

    def _sort_key(self, avatar: Avatar):
        """Legacy faces keep their historical order, new ones follow by name."""
        try:
            return (0, self._legacy_order.index(avatar.id), "")
        except ValueError:
            return (1, 0, avatar.display_name.casefold())

    def _build_lookup(self, avatars: Sequence[Avatar]) -> Dict[str, Avatar]:
        lookup: Dict[str, Avatar] = {}
        for avatar in avatars:
            for key in (avatar.id, avatar.target, *avatar.aliases):
                lookup.setdefault(key.strip().lower(), avatar)
        return lookup

    @staticmethod
    def _compute_revision(avatars: Sequence[Avatar]) -> str:
        digest = hashlib.sha256()
        for avatar in avatars:
            digest.update(avatar.id.encode("utf-8"))
            digest.update(b"\0")
            digest.update(avatar.display_name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(_sha256(avatar.thumbnail_jpeg).encode("ascii"))
            digest.update(b"\n")
        return digest.hexdigest()[:6]

    def _rebuild_payload(self) -> None:
        payload = {
            "version": PAYLOAD_VERSION,
            "revision": self._revision,
            "active": self._active,
            "avatars": [a.to_json_dict() for a in self._avatars],
        }
        self._payload = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


# --------------------------------------------------------------------------- #
# Offline inspection: python3 avatar_registry.py [--root DIR] [--full]
# --------------------------------------------------------------------------- #

def _main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Print the avatar list payload (no ROS needed).")
    parser.add_argument("--root", default=DEFAULT_AVATAR_ROOT, help="avatar directory")
    parser.add_argument("--full", action="store_true", help="print the payload including base64 thumbnails")
    args = parser.parse_args(argv)

    class _PrintLogger:
        def info(self, m):
            print(m)
        debug = lambda self, m: None
        warning = info
        error = info

    # The legacy ids live next to the /operator/name handler; importing that
    # module needs ROS, which a workstation may not have.
    try:
        from ros2_blendshape_node import NAME_MAP as legacy_name_map
    except Exception as exc:
        print(f"(no legacy id map: {exc})")
        legacy_name_map = None

    registry = AvatarRegistry(root=args.root, legacy_name_map=legacy_name_map,
                              logger=_PrintLogger())
    registry.scan()

    if args.full:
        print(registry.payload)
        return 0

    print(f"root:     {registry.root}")
    print(f"revision: {registry.revision}")
    print(f"payload:  {len(registry.payload)} bytes")
    for avatar in registry.avatars:
        print(f"  {avatar.id:<16} {avatar.display_name:<20} "
              f"thumb {len(avatar.thumbnail_jpeg):>6} B  target={avatar.target}"
              + (f"  aliases={','.join(avatar.aliases)}" if avatar.aliases else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
