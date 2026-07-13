# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
#
# SPDX-License-Identifier: MIT
# SPDX-PackageName: translocate

"""Serialize a reconciled inventory into a scheme-independent wheel.

The inventory already encodes the wheel's logical model: each entry
carries a category label and a category-relative path. Emitting maps
those onto archive paths, drops install-site provenance, leaves out
entry-point wrappers (the new wheel's installer regenerates them),
and writes a fresh RECORD last.

The resulting wheel is suitable for reïnstall and redistribution but
is not bit-identical to the upstream artifact: any file the source
install rewrote in place (most commonly script shebangs) will differ,
and RECORD differs because it's regenerated.
"""

from __future__ import annotations

import csv
import io
import logging
import re
import shutil
import zipfile
from concurrent.futures import Executor
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from translocate._hashing import (
    format_record_hash,
    hash_bytes_sha256,
)
from translocate._version import __version__
from translocate.errors import IncompleteMetadata, TranslocateRefused
from translocate.inventory import (
    Category,
    ContentKind,
    Entry,
    Inventory,
)
from translocate.scripts import neutralized_script

logger = logging.getLogger(__name__)

_DROPPED_DIST_INFO_LEAVES = {
    "INSTALLER",
    "REQUESTED",
    "direct_url.json",
    "RECORD.jws",
    "RECORD.p7s",
    "RECORD",
}

# A PEP 425 wheel tag is a ``<python>-<abi>-<platform>`` triple of
# tokens drawn from letters, digits, and ``._+-``. Tags matching this
# shape get inherited into the regenerated WHEEL file and folded into
# the wheel's filename; tags outside it are skipped at parse time.
_VALID_TAG_RE = re.compile(
    r"^[A-Za-z0-9_.+-]+-[A-Za-z0-9_.+-]+-[A-Za-z0-9_.+-]+$"
)

# Copy streamed members into the archive in 1 MiB blocks so deflate
# works over large chunks.
_COPY_BUFFER = 1 << 20


@dataclass
class EmitResult:
    wheel_path: Path
    members_emitted: int
    unrecorded_residue: List[Path] = field(default_factory=list)
    outside_known_scheme: List[Path] = field(default_factory=list)
    excluded_entrypoint_scripts: List[str] = field(default_factory=list)
    hash_mismatches: List[Path] = field(default_factory=list)


def _should_emit(entry: Entry) -> bool:
    """Whether ``entry`` produces a wheel archive member.

    The single source of truth for the emit filter. Entry-point
    wrappers, residue, outside-scheme paths, bytecode, signatures,
    provenance, the RECORD self-row, missing files, and the dropped
    ``.dist-info`` leaves produce nothing; everything else is an
    on-disk member.
    """
    flags = entry.flags
    if (
        flags.is_entrypoint_script
        or flags.is_unrecorded_residue
        or flags.outside_known_scheme
        or flags.is_pyc
        or flags.is_signature
        or flags.is_provenance
        or flags.is_record
        or flags.missing_on_disk
    ):
        return False
    if entry.content_ref.kind != ContentKind.ON_DISK:
        return False
    if entry.content_ref.on_disk_path is None:
        return False
    if (
        entry.category == Category.DIST_INFO
        and entry.rel_path.as_posix() in _DROPPED_DIST_INFO_LEAVES
    ):
        return False
    return True


def _archive_path(inventory: Inventory, entry: Entry) -> str:
    """Return the archive path for an emit-able entry."""
    name = inventory.dist_name.replace("-", "_")
    version = inventory.dist_version
    data_root = f"{name}-{version}.data"
    dist_info_root = f"{name}-{version}.dist-info"
    rel = entry.rel_path.as_posix()

    if entry.category == Category.ROOT_PURELIB:
        if inventory.root_is_purelib:
            return rel
        return f"{data_root}/purelib/{rel}"
    if entry.category == Category.ROOT_PLATLIB:
        if not inventory.root_is_purelib:
            return rel
        return f"{data_root}/platlib/{rel}"
    if entry.category == Category.SCRIPTS:
        return f"{data_root}/scripts/{rel}"
    if entry.category == Category.HEADERS:
        return f"{data_root}/headers/{rel}"
    if entry.category == Category.DATA:
        return f"{data_root}/data/{rel}"
    if entry.category == Category.DIST_INFO:
        return f"{dist_info_root}/{rel}"
    raise RuntimeError(f"no archive path for category {entry.category}")


def _source_tags(dist_name: str, existing_wheel: Optional[bytes]) -> List[str]:
    """Parse the PEP 425 tags from a source ``WHEEL`` body.

    Skips malformed ``Tag`` lines with a warning. When no usable tags
    remain, or there is no ``WHEEL`` at all, returns
    ``["py3-none-any"]``.
    """
    tags: List[str] = []
    if existing_wheel:
        for raw_line in existing_wheel.decode("utf-8").splitlines():
            key, _, value = raw_line.partition(":")
            if key.strip() != "Tag":
                continue
            value = value.strip()
            if _VALID_TAG_RE.match(value):
                tags.append(value)
            else:
                logger.warning(
                    "ignoring malformed Tag %r in %s WHEEL",
                    value,
                    dist_name,
                )
    return tags or ["py3-none-any"]


def _build_wheel_metadata(
    dist_name: str,
    root_is_purelib: bool,
    existing_wheel: Optional[bytes],
) -> Tuple[bytes, List[str]]:
    """Build a fresh ``WHEEL`` file body matching the emitted layout.

    Returns ``(bytes, tag list)``. Tags and any ``Build`` number are
    inherited from the source distribution's ``WHEEL`` when present;
    ``Root-Is-Purelib`` is rewritten to match whichever root the
    emitter actually placed at the archive root.
    """
    fields = {
        "Wheel-Version": "1.0",
        "Generator": f"translocate {__version__}",
        "Root-Is-Purelib": "true" if root_is_purelib else "false",
    }
    if existing_wheel:
        for raw_line in existing_wheel.decode("utf-8").splitlines():
            key, _, value = raw_line.partition(":")
            if key.strip() == "Build":
                fields["Build"] = value.strip()
    tags = _source_tags(dist_name, existing_wheel)
    out = io.StringIO()
    for k, v in fields.items():
        out.write(f"{k}: {v}\n")
    for t in tags:
        out.write(f"Tag: {t}\n")
    return out.getvalue().encode("utf-8"), tags


def _compress_tags(tags: List[str]) -> str:
    """Collapse PEP 425 ``py-abi-plat`` triples into a wheel-filename component.

    When the supplied triples cover a Cartesian product of their
    unique python, abi, and platform values, PEP 427 allows them to
    appear as a single ``{pys}-{abis}-{plats}`` component with
    ``.``-joined alternatives inside each axis. When they don't (e.g.
    two triples differ in more than one axis without filling the
    product), there is no valid compression, so we fall back to
    joining the raw tags with ``.``.
    """
    triples: List[Tuple[str, str, str]] = []
    for tag in tags:
        parts = tag.split("-")
        if len(parts) != 3:
            return ".".join(tags)
        triples.append((parts[0], parts[1], parts[2]))
    pys: List[str] = []
    abis: List[str] = []
    plats: List[str] = []
    seen_pys: set = set()
    seen_abis: set = set()
    seen_plats: set = set()
    for py, abi, plat in triples:
        if py not in seen_pys:
            seen_pys.add(py)
            pys.append(py)
        if abi not in seen_abis:
            seen_abis.add(abi)
            abis.append(abi)
        if plat not in seen_plats:
            seen_plats.add(plat)
            plats.append(plat)
    if len(triples) != len(pys) * len(abis) * len(plats):
        return ".".join(tags)
    return f"{'.'.join(pys)}-{'.'.join(abis)}-{'.'.join(plats)}"


def _filename(dist_name: str, version: str, tags: List[str]) -> str:
    name = dist_name.replace("-", "_")
    return f"{name}-{version}-{_compress_tags(tags)}.whl"


def wheel_filename(dist_name: str, version: str, dist_info_dir: Path) -> str:
    """Compute the PEP 427 filename for this distribution's wheel."""
    try:
        existing: Optional[bytes] = (dist_info_dir / "WHEEL").read_bytes()
    except OSError:
        existing = None
    return _filename(dist_name, version, _source_tags(dist_name, existing))


def _zip_external_attr(mode: Optional[int], is_executable: bool) -> int:
    if mode is not None:
        return (mode & 0xFFFF) << 16
    base = 0o755 if is_executable else 0o644
    return (base & 0xFFFF) << 16


@dataclass(slots=True)
class _Member:
    archive_path: str
    digest: str = ""
    size: int = 0
    # A streamed member carries ``on_disk`` and its reconcile-time
    # digest; an in-memory member carries ``data`` and gets hashed.
    on_disk: Optional[Path] = None
    data: Optional[bytes] = None
    mode: Optional[int] = None
    is_executable: bool = False


def _is_executable(entry: Entry) -> bool:
    return entry.category == Category.SCRIPTS or (
        entry.src_mode is not None and bool(entry.src_mode & 0o111)
    )


def _member_data(entry: Entry) -> bytes:
    data = entry.content_ref.on_disk_path.read_bytes()
    if entry.category == Category.SCRIPTS:
        data = neutralized_script(data)
    return data


def _plan_member(
    inventory: Inventory, entry: Entry, archive_path: str
) -> _Member:
    """Plan one member: stream it if its content is unchanged and
    already hashed, otherwise read it into memory to be hashed.
    """
    reusable = (
        entry.category != Category.SCRIPTS
        and entry.disk_hash is not None
        and entry.disk_size is not None
    )
    if reusable:
        return _Member(
            archive_path=archive_path,
            digest=entry.disk_hash[1],
            size=entry.disk_size,
            on_disk=entry.content_ref.on_disk_path,
        )
    return _Member(
        archive_path=archive_path,
        data=_member_data(entry),
        mode=entry.src_mode,
        is_executable=_is_executable(entry),
    )


def _compression(level: int) -> Tuple[int, Optional[int]]:
    """Map a 0-9 level to a ``(compress_type, compresslevel)`` pair.

    Level 0 stores members uncompressed; 1-9 deflate at that level.
    """
    if level <= 0:
        return zipfile.ZIP_STORED, None
    return zipfile.ZIP_DEFLATED, level


def _stream_into_zip(
    zf: zipfile.ZipFile,
    archive_path: str,
    on_disk: Path,
    compress_type: int,
    compresslevel: Optional[int],
) -> None:
    """Copy ``on_disk`` into ``zf`` without holding its bytes in memory."""
    info = zipfile.ZipInfo.from_file(on_disk, archive_path)
    info.compress_type = compress_type
    info._compresslevel = compresslevel
    with open(on_disk, "rb") as src, zf.open(info, "w") as dest:
        shutil.copyfileobj(src, dest, _COPY_BUFFER)


def _build_record(rows: List[Tuple[str, str, int]], record_path: str) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    for archive_path, digest, size in rows:
        writer.writerow(
            [archive_path, format_record_hash("sha256", digest), str(size)]
        )
    writer.writerow([record_path, "", ""])
    return buf.getvalue().encode("utf-8")


def emit_wheel(
    inventory: Inventory,
    out_dir: Path,
    pool: Optional[Executor] = None,
    compression_level: int = 6,
) -> EmitResult:
    """Write a wheel for ``inventory`` into ``out_dir``.

    The output filename follows PEP 427: ``{name}-{version}-{tag}.whl``
    with multi-tag wheels compressed when their python/abi/platform
    values form a Cartesian product. ``compression_level`` is a 0-9
    deflate level; 0 stores members uncompressed. The ``EmitResult``
    lists the written path plus anything reported but not emitted:
    residue, outside-scheme paths, excluded entry-point scripts, and
    any carried hash mismatches.
    """
    compress_type, compresslevel = _compression(compression_level)
    result = EmitResult(wheel_path=out_dir / "pending.whl", members_emitted=0)

    existing_wheel: Optional[bytes] = None
    emit_entries: List[Entry] = []
    for entry in inventory.entries:
        on_disk = entry.content_ref.on_disk_path
        if entry.flags.is_entrypoint_script:
            result.excluded_entrypoint_scripts.append(
                entry.content_ref.entry_point_name or entry.rel_path.name
            )
        elif entry.flags.is_unrecorded_residue and on_disk is not None:
            result.unrecorded_residue.append(on_disk)
        elif entry.flags.outside_known_scheme and on_disk is not None:
            result.outside_known_scheme.append(on_disk)
        if entry.flags.hash_mismatch and on_disk is not None:
            result.hash_mismatches.append(on_disk)
        if (
            entry.category == Category.DIST_INFO
            and entry.rel_path.as_posix() == "WHEEL"
            and on_disk is not None
        ):
            existing_wheel = on_disk.read_bytes()
        if _should_emit(entry):
            emit_entries.append(entry)

    wheel_bytes, tags = _build_wheel_metadata(
        inventory.dist_name, inventory.root_is_purelib, existing_wheel
    )
    name_underscore = inventory.dist_name.replace("-", "_")
    dist_info_root = f"{name_underscore}-{inventory.dist_version}.dist-info"
    wheel_archive_path = f"{dist_info_root}/WHEEL"
    record_archive_path = f"{dist_info_root}/RECORD"

    # The regenerated WHEEL replaces any inherited copy so its
    # ``Root-Is-Purelib`` and ``Tag`` lines describe this emit.
    members = [
        _plan_member(inventory, entry, path)
        for entry in emit_entries
        if (path := _archive_path(inventory, entry)) != wheel_archive_path
    ]
    members.append(_Member(archive_path=wheel_archive_path, data=wheel_bytes))

    # Hash the in-memory members in the pool while the archive is
    # written; their digests are only needed later for RECORD.
    to_hash = [m for m in members if m.data is not None]
    hash_futures = (
        [pool.submit(hash_bytes_sha256, m.data) for m in to_hash]
        if pool is not None
        else None
    )

    members.sort(key=lambda m: m.archive_path)

    wheel_path = out_dir / _filename(
        inventory.dist_name, inventory.dist_version, tags
    )
    result.wheel_path = wheel_path
    # ``zipfile.ZipFile`` is not thread-safe for concurrent writes, so
    # the append runs serially here while the pool hashes in parallel.
    with zipfile.ZipFile(
        wheel_path, "w", compress_type, compresslevel=compresslevel
    ) as zf:
        for member in members:
            if member.data is not None:
                info = zipfile.ZipInfo(member.archive_path)
                info.external_attr = _zip_external_attr(
                    member.mode, member.is_executable
                )
                zf.writestr(
                    info,
                    member.data,
                    compress_type=compress_type,
                    compresslevel=compresslevel,
                )
            else:
                _stream_into_zip(
                    zf,
                    member.archive_path,
                    member.on_disk,
                    compress_type,
                    compresslevel,
                )
            result.members_emitted += 1

        if hash_futures is not None:
            for member, future in zip(to_hash, hash_futures):
                member.digest, member.size = future.result()
        else:
            for member in to_hash:
                member.digest, member.size = hash_bytes_sha256(member.data)

        record_rows = [(m.archive_path, m.digest, m.size) for m in members]
        record_bytes = _build_record(record_rows, record_archive_path)
        record_info = zipfile.ZipInfo(record_archive_path)
        record_info.external_attr = _zip_external_attr(None, False)
        zf.writestr(
            record_info,
            record_bytes,
            compress_type=compress_type,
            compresslevel=compresslevel,
        )
        result.members_emitted += 1
    return result


def emit_stub_wheel(
    dist_name: str,
    version: str,
    dist_info_dir: Path,
    out_dir: Path,
    root_is_purelib: bool,
    compression_level: int = 6,
) -> EmitResult:
    """Write a metadata-only stub wheel for one distribution.

    The stub archive holds three members: the source ``METADATA``
    verbatim, a regenerated ``WHEEL``, and a regenerated ``RECORD``.
    That is enough for a resolver to place the distribution in a
    dependency graph. Installing the stub yields an empty package.

    Raises ``TranslocateRefused`` wrapping ``IncompleteMetadata`` when
    the distribution has no readable ``METADATA``.
    """
    try:
        metadata_bytes = (dist_info_dir / "METADATA").read_bytes()
    except OSError as e:
        raise TranslocateRefused(
            IncompleteMetadata(
                dist_info=dist_info_dir,
                detail=f"cannot read METADATA: {e}",
            )
        )

    existing_wheel: Optional[bytes] = None
    try:
        existing_wheel = (dist_info_dir / "WHEEL").read_bytes()
    except OSError:
        pass

    wheel_bytes, tags = _build_wheel_metadata(
        dist_name, root_is_purelib, existing_wheel
    )
    name_underscore = dist_name.replace("-", "_")
    dist_info_root = f"{name_underscore}-{version}.dist-info"
    record_archive_path = f"{dist_info_root}/RECORD"
    members = [
        (f"{dist_info_root}/METADATA", metadata_bytes),
        (f"{dist_info_root}/WHEEL", wheel_bytes),
    ]

    compress_type, compresslevel = _compression(compression_level)
    wheel_path = out_dir / _filename(dist_name, version, tags)
    result = EmitResult(wheel_path=wheel_path, members_emitted=0)
    record_rows: List[Tuple[str, str, int]] = []
    with zipfile.ZipFile(
        wheel_path, "w", compress_type, compresslevel=compresslevel
    ) as zf:
        for archive_path, data in members:
            info = zipfile.ZipInfo(archive_path)
            info.external_attr = _zip_external_attr(None, False)
            zf.writestr(
                info,
                data,
                compress_type=compress_type,
                compresslevel=compresslevel,
            )
            digest, size = hash_bytes_sha256(data)
            record_rows.append((archive_path, digest, size))
            result.members_emitted += 1
        record_bytes = _build_record(record_rows, record_archive_path)
        record_info = zipfile.ZipInfo(record_archive_path)
        record_info.external_attr = _zip_external_attr(None, False)
        zf.writestr(
            record_info,
            record_bytes,
            compress_type=compress_type,
            compresslevel=compresslevel,
        )
        result.members_emitted += 1
    return result
