# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
#
# SPDX-License-Identifier: MIT
# SPDX-PackageName: translocate

"""Command-line entrypoint for ``translocate clone`` and ``translocate repackage``.

Both subcommands accept one or more distribution names; their dep
closures are unioned and deduplicated by PEP 503 canonical name.
Failures surface as typed refusals from ``errors`` paired with
specific exit codes, so callers can react without parsing prose.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from concurrent.futures import Executor
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from translocate import _logging, _target
from translocate._dag import traverse
from translocate._logging import DETAIL
from translocate._pool import cpu_pool, io_pool
from translocate.closure import (
    Closure,
    ClosureMember,
    LocatedMember,
    RefusedMember,
    closure_for,
    deps_file_lines,
    filter_already_at_target,
)
from translocate.emit.wheel import (
    EmitResult,
    emit_stub_wheel,
    emit_wheel,
    wheel_filename,
)
from translocate.errors import (
    ABIMismatch,
    AlreadyPresentAtTarget,
    CloneToSelf,
    DistributionNotFound,
    EditableInstall,
    EggInfoOnly,
    HashMismatch,
    IncompleteMetadata,
    InvalidDistributionName,
    Refusal,
    SourceTargetSharedRoot,
    TargetNotAPython,
    TranslocateRefused,
)
from translocate.inventory import InstallScheme, Inventory
from translocate.locate import (
    LocatedDistribution,
    canonicalize,
    locate,
    validate_name,
)
from translocate.materialize import (
    MaterializeResult,
    materialize,
)
from translocate.materialize.linking import FallbackPolicy, LinkMode
from translocate.materialize.verify import (
    VerifyResult,
    top_level_importable_names,
    verify_clone,
)
from translocate.reconcile import reconcile
from translocate.record import parse_record
from translocate.scheme import (
    distribution_root_is_purelib,
    resolve_source_scheme,
    scheme_from_target_query,
)
from translocate.uvconfig import render_uv_config

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_GENERIC = 1
EXIT_INVALID_INVOCATION = 2
EXIT_DIST_NOT_FOUND = 10
EXIT_UNTRANSLOCATABLE = 11
EXIT_TARGET_PROBLEM = 12
EXIT_ABI_MISMATCH = 13
EXIT_VERIFICATION_FAILED = 20


def _refusal_exit_code(refusal: Refusal) -> int:
    if isinstance(refusal, InvalidDistributionName):
        return EXIT_INVALID_INVOCATION
    if isinstance(refusal, DistributionNotFound):
        return EXIT_DIST_NOT_FOUND
    if isinstance(refusal, (EditableInstall, EggInfoOnly, IncompleteMetadata)):
        return EXIT_UNTRANSLOCATABLE
    if isinstance(
        refusal, (CloneToSelf, SourceTargetSharedRoot, TargetNotAPython)
    ):
        return EXIT_TARGET_PROBLEM
    if isinstance(refusal, ABIMismatch):
        return EXIT_ABI_MISMATCH
    if isinstance(refusal, HashMismatch):
        return EXIT_VERIFICATION_FAILED
    if isinstance(refusal, AlreadyPresentAtTarget):
        return EXIT_OK
    return EXIT_GENERIC


def _check_abi(
    source_scheme: InstallScheme, target_scheme: InstallScheme
) -> Optional[Refusal]:
    if source_scheme.soabi and target_scheme.soabi:
        if source_scheme.soabi != target_scheme.soabi:
            return ABIMismatch(
                source_soabi=source_scheme.soabi,
                target_soabi=target_scheme.soabi,
                source_ext_suffix=source_scheme.ext_suffix,
                target_ext_suffix=target_scheme.ext_suffix,
            )
    if source_scheme.ext_suffix and target_scheme.ext_suffix:
        if source_scheme.ext_suffix != target_scheme.ext_suffix:
            return ABIMismatch(
                source_soabi=source_scheme.soabi,
                target_soabi=target_scheme.soabi,
                source_ext_suffix=source_scheme.ext_suffix,
                target_ext_suffix=target_scheme.ext_suffix,
            )
    return None


def _is_platform_specific(dist_info_dir: Path) -> bool:
    """Whether the source distribution ships platform-specific code.

    Reads PEP 427 ``Tag:`` lines from the source ``WHEEL`` file. Tags
    whose platform component is ``any`` (e.g. ``py3-none-any``)
    describe pure-Python wheels; any other platform value
    (``linux_x86_64``, ``manylinux_2_17_x86_64``, etc.) marks the
    distribution as platform-specific. When ``WHEEL`` is absent or
    unparseable, the conservative answer is "platform-specific" so
    the ABI check still applies.
    """
    wheel_path = dist_info_dir / "WHEEL"
    if not wheel_path.exists():
        return True
    try:
        text = wheel_path.read_text(encoding="utf-8")
    except OSError:
        return True
    tag_seen = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.lower().startswith("tag:"):
            continue
        tag = line.split(":", 1)[1].strip()
        parts = tag.split("-")
        if len(parts) != 3:
            continue
        tag_seen = True
        if parts[-1] != "any":
            return True
    return not tag_seen


def _real(p: Path) -> Path:
    return Path(os.path.realpath(p))


def _check_target_distinct(
    source_scheme: InstallScheme, target_scheme: InstallScheme
) -> Optional[Refusal]:
    if _real(source_scheme.prefix) == _real(target_scheme.prefix):
        return CloneToSelf(
            source_prefix=source_scheme.prefix,
            target_prefix=target_scheme.prefix,
        )
    for label, src, tgt in (
        ("purelib", source_scheme.purelib, target_scheme.purelib),
        ("platlib", source_scheme.platlib, target_scheme.platlib),
        ("scripts", source_scheme.scripts, target_scheme.scripts),
    ):
        if _real(src) == _real(tgt):
            return SourceTargetSharedRoot(category=label, path=_real(src))
    return None


@dataclass
class _CloneContext:
    target_python: Path
    target_scheme: InstallScheme
    link_mode: LinkMode
    fallback: FallbackPolicy
    smoke_test_imports: bool
    verify_hashes: bool
    on_hash_mismatch: str


def _prepare_clone_context(args: argparse.Namespace) -> _CloneContext:
    target_python = _target.find_target_interpreter(Path(args.target))
    query = _target.query_target_scheme(target_python)
    target_scheme = scheme_from_target_query(query)
    return _CloneContext(
        target_python=target_python,
        target_scheme=target_scheme,
        link_mode=LinkMode(args.link_mode),
        fallback=FallbackPolicy.COPY,
        smoke_test_imports=args.smoke_test_imports,
        verify_hashes=args.verify_hashes,
        on_hash_mismatch=args.on_hash_mismatch,
    )


def _build_inventory(located: LocatedDistribution, pool: Executor) -> Inventory:
    source_scheme = resolve_source_scheme()
    inventory = parse_record(located, source_scheme)
    reconcile(inventory, pool=pool)
    return inventory


def _hash_mismatch_refusal(
    inventory: Inventory, on_hash_mismatch: str
) -> Optional[Refusal]:
    """Return a refusal when ``fail`` mode meets a hash mismatch.

    Under ``warn`` the mismatched files are carried and reported by the
    backend instead.
    """
    if on_hash_mismatch != "fail":
        return None
    paths = [
        entry.content_ref.on_disk_path
        for entry in inventory.entries
        if entry.flags.hash_mismatch
        and entry.content_ref.on_disk_path is not None
    ]
    if not paths:
        return None
    return HashMismatch(dist_name=inventory.dist_name, paths=paths)


@dataclass
class _ClonedDistribution:
    inventory: Inventory
    target_dist_info_dir: Path


def _clone_one(
    located: LocatedDistribution,
    ctx: _CloneContext,
    pool: Executor,
) -> Tuple[Optional[Refusal], Optional[_ClonedDistribution]]:
    inventory = _build_inventory(located, pool)
    mismatch = _hash_mismatch_refusal(inventory, ctx.on_hash_mismatch)
    if mismatch is not None:
        return mismatch, None
    abi = _check_abi(inventory.source_scheme, ctx.target_scheme)
    if abi is not None and _is_platform_specific(inventory.dist_info_dir):
        return abi, None
    distinct = _check_target_distinct(
        inventory.source_scheme, ctx.target_scheme
    )
    if distinct is not None:
        return distinct, None
    target_version = _target.query_target_version(
        ctx.target_python, inventory.dist_name
    )
    if target_version is not None:
        return (
            AlreadyPresentAtTarget(
                name=inventory.dist_name,
                source_version=inventory.dist_version,
                target_version=target_version,
            ),
            None,
        )
    result, target_scheme = materialize(
        inventory=inventory,
        target_python=ctx.target_python,
        link_mode=ctx.link_mode,
        fallback=ctx.fallback,
        pool=pool,
    )
    verification = verify_clone(
        inventory=inventory,
        target_dist_info_dir=result.target_dist_info_dir,
        target_python=ctx.target_python,
        smoke_test_import=False,
        verify_hashes=ctx.verify_hashes,
        pool=pool,
    )
    _report_clone(inventory, result, verification)
    if not verification.ok:
        raise _VerificationFailed()
    return None, _ClonedDistribution(
        inventory=inventory,
        target_dist_info_dir=result.target_dist_info_dir,
    )


class _VerificationFailed(Exception):
    pass


@dataclass(slots=True)
class _CloneOutcome:
    """The result of attempting one member's clone.

    ``succeeded`` is whether the clone completed; ``code`` is its
    exit-code contribution (``EXIT_OK`` when the failure doesn't
    count); ``cloned`` carries the materialized distribution on
    success.
    """

    canonical: str
    succeeded: bool
    code: int
    cloned: Optional[_ClonedDistribution]


def _clone_node(
    member: LocatedMember,
    ctx: _CloneContext,
    pool: Executor,
    is_root: bool,
    strict: bool,
) -> _CloneOutcome:
    """Clone one member, reporting the outcome instead of raising.

    A failure contributes its exit code only when the member is a root
    (roots are always strict) or ``strict`` is set; otherwise the
    contribution is ``EXIT_OK``.
    """
    canonical = member.canonical_name

    def gated(code: int) -> int:
        return code if (is_root or strict) else EXIT_OK

    try:
        refusal, cloned = _clone_one(member.located, ctx, pool)
    except _VerificationFailed:
        logger.error("verification failed for %s; skipping it", member.name)
        return _CloneOutcome(
            canonical, False, gated(EXIT_VERIFICATION_FAILED), None
        )
    except TranslocateRefused as e:
        _print_refusal(e.refusal)
        return _CloneOutcome(
            canonical, False, gated(_refusal_exit_code(e.refusal)), None
        )
    except Exception:
        logger.exception(
            "unexpected error cloning %s; skipping it", member.name
        )
        return _CloneOutcome(canonical, False, gated(EXIT_GENERIC), None)

    if refusal is not None:
        _print_refusal(refusal)
        return _CloneOutcome(
            canonical, False, gated(_refusal_exit_code(refusal)), None
        )
    return _CloneOutcome(canonical, True, EXIT_OK, cloned)


def _report_clone(
    inventory: Inventory,
    materialize_result: MaterializeResult,
    verification: VerifyResult,
) -> None:
    name = inventory.dist_name
    logger.info(
        f"cloned {name} {inventory.dist_version}: "
        f"{materialize_result.members_materialized} files into "
        f"{materialize_result.target_prefix}"
    )
    if materialize_result.fallback_count:
        logger.warning(
            f"hardlinks across filesystems are unavailable, so "
            f"{materialize_result.fallback_count} file(s) of {name} "
            f"were copied instead"
        )
    if materialize_result.members_skipped_residue:
        residue = materialize_result.members_skipped_residue
        logger.warning(
            f"skipped {len(residue)} file(s) under {name}'s package "
            f"directory that the source RECORD does not list"
        )
        for path in residue:
            logger.log(DETAIL, f"  residue: {path}")
    if materialize_result.members_skipped_outside_scheme:
        outside = materialize_result.members_skipped_outside_scheme
        logger.warning(
            f"skipped {len(outside)} file(s) of {name} whose RECORD path "
            f"falls outside any known install scheme"
        )
        for path in outside:
            logger.log(DETAIL, f"  outside-scheme: {path}")
    if materialize_result.hash_mismatches:
        mismatches = materialize_result.hash_mismatches
        logger.warning(
            f"carried {len(mismatches)} file(s) of {name} whose on-disk "
            f"content differs from the source RECORD"
        )
        for path in mismatches:
            logger.log(DETAIL, f"  hash mismatch: {path}")
    for check in verification.checks_passed:
        logger.debug(f"verified {name}: {check}")
    for failure in verification.failures:
        logger.info(f"verification failed for {name}: {failure}")


def _report_emit(inventory: Inventory, emit_result: EmitResult) -> None:
    name = inventory.dist_name
    logger.info(
        f"repackaged {name} {inventory.dist_version} into "
        f"{emit_result.wheel_path} with {emit_result.members_emitted} members"
    )
    if emit_result.excluded_entrypoint_scripts:
        excluded = emit_result.excluded_entrypoint_scripts
        logger.log(
            DETAIL,
            f"excluded {len(excluded)} entry-point script(s) of {name} "
            f"so the target installer can regenerate them",
        )
        for ep_name in excluded:
            logger.debug(f"  entry-point: {ep_name}")
    if emit_result.unrecorded_residue:
        residue = emit_result.unrecorded_residue
        logger.warning(
            f"omitted {len(residue)} file(s) under {name}'s package "
            f"directory that the source RECORD does not list"
        )
        for path in residue:
            logger.log(DETAIL, f"  residue: {path}")
    if emit_result.outside_known_scheme:
        outside = emit_result.outside_known_scheme
        logger.warning(
            f"omitted {len(outside)} file(s) of {name} whose RECORD path "
            f"falls outside any known install scheme"
        )
        for path in outside:
            logger.log(DETAIL, f"  outside-scheme: {path}")


def _resolve_roots(
    names: List[str],
) -> Union[List[Tuple[str, LocatedDistribution]], Refusal]:
    """Validate and locate every user-supplied name.

    Returns either the deduplicated list of ``(original-name,
    located)`` pairs, or the first refusal encountered. User-supplied
    names are hard requirements: an invalid or unfindable one fails
    the whole call rather than being skipped.
    """
    for name in names:
        check = validate_name(name)
        if check is not None:
            return check
    resolved: List[Tuple[str, LocatedDistribution]] = []
    seen: set = set()
    for name in names:
        canonical = canonicalize(name)
        if canonical in seen:
            continue
        seen.add(canonical)
        located = locate(name)
        if isinstance(located, Refusal):
            return located
        resolved.append((name, located))
    return resolved


def _plan_for(
    roots: List[Tuple[str, LocatedDistribution]], include_deps: bool
) -> Closure:
    return closure_for(roots, depth=None if include_deps else 0)


def _run_clone(args: argparse.Namespace) -> int:
    resolved = _resolve_roots(args.names)
    if isinstance(resolved, Refusal):
        _print_refusal(resolved)
        return _refusal_exit_code(resolved)
    plan = _plan_for(resolved, args.include_deps)

    ctx = _prepare_clone_context(args)
    target_installed = _target.query_target_installed_names(ctx.target_python)
    to_clone, already_present = filter_already_at_target(plan, target_installed)

    if args.dry_run:
        return _dry_run_clone(args, ctx, plan, to_clone, already_present)

    member_by_canonical = {m.canonical_name: m for m in to_clone}
    root_canonicals = {r.canonical_name for r in plan.roots}
    # Restrict the closure's edges to the runnable set; an edge to an
    # already-present dep is a dead end (the target was consistent
    # before this run, so its subtree is present too).
    edges = {
        canonical: tuple(
            dep
            for dep in plan.edges.get(canonical, ())
            if dep in member_by_canonical
        )
        for canonical in member_by_canonical
    }
    runnable_roots = [c for c in root_canonicals if c in member_by_canonical]

    with io_pool() as pool:

        def run_node(canonical: str) -> _CloneOutcome:
            return _clone_node(
                member_by_canonical[canonical],
                ctx,
                pool,
                canonical in root_canonicals,
                args.strict_deps,
            )

        results: Dict[str, _CloneOutcome] = asyncio.run(
            traverse(
                set(member_by_canonical),
                edges,
                runnable_roots,
                run_node,
                lambda outcome: outcome.succeeded,
            )
        )

    exit_code = EXIT_OK
    cloned: List[_ClonedDistribution] = []
    for outcome in results.values():
        exit_code = max(exit_code, outcome.code)
        if outcome.cloned is not None:
            cloned.append(outcome.cloned)

    for member in already_present:
        logger.info(
            f"skipped {member.name} {_version_str(member)}: "
            f"already present at target"
        )
    for canonical, member in member_by_canonical.items():
        if canonical not in results:
            logger.debug(
                "skipped %s: nothing that requires it was cloned",
                member.name,
            )
    for member in plan.skipped:
        _handle_untranslocatable(member, args.strict_deps)
        if args.strict_deps:
            exit_code = max(exit_code, _refusal_exit_code(member.refusal))

    if ctx.smoke_test_imports:
        _run_post_clone_smoke_imports(cloned, ctx.target_python)
    if args.deps_file:
        _emit_deps_file(plan, args.deps_file, args.include_deps)

    return exit_code


def _dry_run_clone(
    args: argparse.Namespace,
    ctx: _CloneContext,
    plan: Closure,
    to_clone: List[LocatedMember],
    already_present: List[LocatedMember],
) -> int:
    """Report the clone plan without materializing anything.

    Runs the checks that need no per-file work: source/target
    distinctness, and the ABI comparison for platform-specific
    distributions. Skips reconciliation, materialization,
    verification, and smoke imports.
    """
    source_scheme = resolve_source_scheme()
    distinct = _check_target_distinct(source_scheme, ctx.target_scheme)
    if distinct is not None:
        _print_refusal(distinct)
        return _refusal_exit_code(distinct)
    abi = _check_abi(source_scheme, ctx.target_scheme)

    exit_code = EXIT_OK
    root_canonicals = {r.canonical_name for r in plan.roots}
    for member in to_clone:
        located = member.located
        if abi is not None and _is_platform_specific(located.dist_info_dir):
            _print_refusal(abi)
            if member.canonical_name in root_canonicals or args.strict_deps:
                exit_code = max(exit_code, _refusal_exit_code(abi))
            continue
        logger.info(
            f"would clone {located.canonical_name} {located.version} "
            f"into {ctx.target_scheme.prefix}"
        )
    for member in already_present:
        logger.info(
            f"skipped {member.name} {_version_str(member)}: "
            f"already present at target"
        )
    for member in plan.skipped:
        _handle_untranslocatable(member, args.strict_deps)
        if args.strict_deps:
            exit_code = max(exit_code, _refusal_exit_code(member.refusal))
    if args.deps_file:
        _emit_deps_file(plan, args.deps_file, args.include_deps)
    return exit_code


def _run_post_clone_smoke_imports(
    cloned: List[_ClonedDistribution], target_python: Path
) -> None:
    """Smoke-import every cloned distribution against the assembled target.

    The sweep runs after every member is on disk so a dependent that
    landed before its dependencies still imports cleanly.
    """
    successes = 0
    for member in cloned:
        dist_name = member.inventory.dist_name
        candidates = top_level_importable_names(member.inventory)
        if not candidates:
            logger.info(
                f"smoke import for {dist_name} found no importable "
                f"top-level module"
            )
            continue
        first_error = ""
        ok_name: Optional[str] = None
        for name in candidates:
            ok, err = _target.smoke_import(target_python, name)
            if ok:
                ok_name = name
                break
            if not first_error:
                first_error = f"{name}: {err}"
        if ok_name is not None:
            successes += 1
            suffix = f" (as {ok_name})" if ok_name != dist_name else ""
            logger.debug(f"smoke import passed for {dist_name}{suffix}")
        else:
            logger.info(f"smoke import failed for {dist_name}: {first_error}")
    if successes > 1:
        logger.log(DETAIL, f"{successes} smoke imports passed")
    elif successes == 1:
        logger.log(DETAIL, "smoke import passed")


def _run_repackage(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    if out_dir.exists() and not out_dir.is_dir():
        logger.error(
            f"--out path {out_dir} exists but is not a directory. "
            f"Wheel filenames are auto-generated, so --out must be "
            f"a directory."
        )
        return EXIT_INVALID_INVOCATION

    resolved = _resolve_roots(args.names)
    if isinstance(resolved, Refusal):
        _print_refusal(resolved)
        return _refusal_exit_code(resolved)
    plan = _plan_for(resolved, args.include_deps)

    if args.mkdir and not args.dry_run:
        out_dir.mkdir(parents=False, exist_ok=True)

    exit_code = EXIT_OK
    produced: List[Path] = []
    emitted_names: List[str] = []
    seen_canonical: set = set()
    candidates = list(plan.roots) + [
        m
        for m in plan.members
        if m.canonical_name not in {r.canonical_name for r in plan.roots}
    ]
    stub_scheme = resolve_source_scheme() if args.metadata_only else None
    # Only the full emit hashes per-file payloads, so the stub and
    # dry-run paths skip the pools.
    needs_pools = not (args.dry_run or args.metadata_only)
    with ExitStack() as stack:
        io = stack.enter_context(io_pool()) if needs_pools else None
        cpu = stack.enter_context(cpu_pool()) if needs_pools else None
        for member in candidates:
            if member.canonical_name in seen_canonical:
                continue
            seen_canonical.add(member.canonical_name)
            located = member.located
            if args.dry_run:
                fname = wheel_filename(
                    located.canonical_name,
                    located.version,
                    located.dist_info_dir,
                )
                kind = "stub" if args.metadata_only else "wheel"
                logger.info(
                    f"would write {kind} {fname} for "
                    f"{located.canonical_name} {located.version}"
                )
                produced.append(out_dir / fname)
                emitted_names.append(member.canonical_name)
                continue
            if args.metadata_only:
                try:
                    emit_result = emit_stub_wheel(
                        located.canonical_name,
                        located.version,
                        located.dist_info_dir,
                        out_dir,
                        distribution_root_is_purelib(
                            located.dist_info_dir, stub_scheme
                        ),
                        compression_level=args.compression_level,
                    )
                except TranslocateRefused as e:
                    _print_refusal(e.refusal)
                    exit_code = max(exit_code, _refusal_exit_code(e.refusal))
                    continue
                produced.append(emit_result.wheel_path)
                emitted_names.append(member.canonical_name)
                logger.info(
                    f"wrote metadata-only stub for {located.canonical_name} "
                    f"{located.version} into {emit_result.wheel_path}"
                )
                continue
            inventory = _build_inventory(located, io)
            mismatch = _hash_mismatch_refusal(inventory, args.on_hash_mismatch)
            if mismatch is not None:
                _print_refusal(mismatch)
                exit_code = _refusal_exit_code(mismatch)
                continue
            emit_result = emit_wheel(
                inventory,
                out_dir,
                pool=cpu,
                compression_level=args.compression_level,
            )
            produced.append(emit_result.wheel_path)
            emitted_names.append(member.canonical_name)
            _report_emit(inventory, emit_result)
        for member in plan.skipped:
            if member.canonical_name in seen_canonical:
                continue
            seen_canonical.add(member.canonical_name)
            _handle_untranslocatable(member, args.strict_deps)
            if args.strict_deps:
                exit_code = _refusal_exit_code(member.refusal)

    if args.dry_run:
        if len(produced) == 1:
            logger.info(f"dry run: 1 wheel would be produced in {out_dir}")
        elif produced:
            logger.info(
                f"dry run: {len(produced)} wheels would be produced "
                f"in {out_dir}"
            )
    elif len(produced) == 1:
        logger.info(f"produced 1 wheel in {out_dir}")
    elif produced:
        logger.info(f"produced {len(produced)} wheels in {out_dir}")
        for path in produced:
            logger.log(DETAIL, f"  {path.name}")

    if args.uv_config:
        _emit_uv_config(args.uv_config, out_dir, emitted_names)
    if args.deps_file:
        _emit_deps_file(plan, args.deps_file, args.include_deps)

    return exit_code


def _emit_uv_config(
    path_str: str, out_dir: Path, package_names: List[str]
) -> None:
    text = render_uv_config(f"file://{out_dir.resolve()}", package_names)
    if path_str == "-":
        print(text, end="")
        return
    Path(path_str).write_text(text, encoding="utf-8")
    logger.info(
        f"wrote uv index config for {len(package_names)} package(s) "
        f"to {path_str}"
    )


def _handle_untranslocatable(member: RefusedMember, strict: bool) -> None:
    message = f"untranslocatable dep {member.name}: {member.refusal}"
    if strict:
        logger.error(message)
    else:
        logger.warning(message)


def _version_str(member: ClosureMember) -> str:
    if isinstance(member, LocatedMember):
        return member.located.version
    return "?"


def _deps_file_lines(plan: Closure, include_deps: bool) -> List[str]:
    """Compute the pinned ``name==version`` lines for a deps file.

    With ``include_deps``, lists every member of the walked closure.
    Without it, lists each root plus its direct ``Requires-Dist``
    resolved against the source environment's installed set.
    """
    if include_deps:
        return deps_file_lines(plan)
    roots = [(r.name, r.located) for r in plan.roots]
    return deps_file_lines(closure_for(roots, depth=1))


def _emit_deps_file(plan: Closure, path_str: str, include_deps: bool) -> None:
    lines = _deps_file_lines(plan, include_deps)
    if path_str == "-":
        for line in lines:
            print(line)
        return
    out = Path(path_str)
    with out.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def _print_refusal(refusal: Refusal) -> None:
    logger.error(str(refusal))


def _add_verbosity_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity. Pass -vv for debug-level output.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="count",
        default=0,
        help="Reduce verbosity. Pass -qq to show only errors.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="translocate",
        description=(
            "Translocate an installed Python distribution into a "
            "wheel or another interpreter's site-packages."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    clone = subparsers.add_parser(
        "clone",
        help="Project installed distributions into a target interpreter's scheme.",
    )
    clone.add_argument(
        "names",
        nargs="+",
        metavar="NAME",
        help=(
            "Distribution name(s) to clone. Multiple names "
            "deduplicate against each other and against the dep "
            "closure if --include-deps is set."
        ),
    )
    clone.add_argument(
        "--target",
        required=True,
        help="Target prefix or interpreter path.",
    )
    clone.add_argument(
        "--link-mode",
        choices=[m.value for m in LinkMode],
        default=LinkMode.HARDLINK.value,
        help="Link mode for projected files (default: hardlink).",
    )
    clone.add_argument(
        "--include-deps",
        action="store_true",
        help="Walk the source environment's dep closure.",
    )
    clone.add_argument(
        "--strict-deps",
        action="store_true",
        help="Treat untranslocatable deps as hard errors.",
    )
    clone.add_argument(
        "--deps-file",
        help="Write resolved closure as name==version lines; - for stdout.",
    )
    clone.add_argument(
        "--smoke-test-imports",
        action="store_true",
        help="Run import <name> in the target after structural checks.",
    )
    clone.add_argument(
        "--verify-hashes",
        action="store_true",
        help="Also re-hash every cloned file during verification.",
    )
    clone.add_argument(
        "--on-hash-mismatch",
        choices=["warn", "fail"],
        default="warn",
        help="When a source file differs from its recorded hash: "
        "carry it with a warning, or refuse.",
    )
    clone.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve packages, probe the target, and report what "
        "would be cloned, without materializing anything.",
    )
    _add_verbosity_flags(clone)
    clone.set_defaults(func=_run_clone)

    repackage = subparsers.add_parser(
        "repackage",
        help="Serialize installed distributions back into wheels.",
    )
    repackage.add_argument(
        "names",
        nargs="+",
        metavar="NAME",
        help=(
            "Distribution name(s) to repackage. Multiple names "
            "deduplicate against each other and against the dep "
            "closure if --include-deps is set."
        ),
    )
    repackage.add_argument(
        "--out",
        required=True,
        help=(
            "Output directory for wheel files. Wheel filenames are "
            "auto-generated per PEP 427 from the source metadata."
        ),
    )
    repackage.add_argument(
        "--mkdir",
        action="store_true",
        help="Create the --out directory; its parent must already exist.",
    )
    repackage.add_argument(
        "--include-deps",
        action="store_true",
        help="Walk and repackage the source dep closure.",
    )
    repackage.add_argument(
        "--strict-deps",
        action="store_true",
        help="Treat untranslocatable deps as hard errors.",
    )
    repackage.add_argument(
        "--deps-file",
        help="Write resolved closure as name==version lines; - for stdout.",
    )
    repackage.add_argument(
        "--compression-level",
        type=int,
        choices=range(10),
        default=6,
        metavar="{0-9}",
        help="Deflate level for wheel members; 0 stores uncompressed.",
    )
    repackage.add_argument(
        "--on-hash-mismatch",
        choices=["warn", "fail"],
        default="warn",
        help="When a source file differs from its recorded hash: "
        "carry it with a warning, or refuse.",
    )
    repackage.add_argument(
        "--metadata-only",
        action="store_true",
        help="Emit stub wheels holding just dist-info for use "
        "as a uv lock-time index. Not for installation. "
        "Installing one yields an empty package.",
    )
    repackage.add_argument(
        "--uv-config",
        metavar="PATH",
        help="Write a pyproject.toml snippet configuring --out as a "
        "uv index for the emitted packages. Pass - for stdout. "
        "A file path is written to even under --dry-run.",
    )
    repackage.add_argument(
        "--dry-run",
        action="store_true",
        help="Locate packages, walk dependency closures, "
        "and report what would be written, but don't write wheels.",
    )
    _add_verbosity_flags(repackage)
    repackage.set_defaults(func=_run_repackage)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _logging.configure(args.verbose, args.quiet)
    try:
        return args.func(args)
    except TranslocateRefused as e:
        _print_refusal(e.refusal)
        return _refusal_exit_code(e.refusal)
    except OSError as e:
        logger.error("%s", e)
        return EXIT_GENERIC
    except KeyboardInterrupt:
        return EXIT_GENERIC


if __name__ == "__main__":
    raise SystemExit(main())
