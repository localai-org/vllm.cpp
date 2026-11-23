#!/usr/bin/env python3
"""Hold a pull request body to the commit contract BEFORE the squash lands it.

This repository sets ``squash_merge_commit_message = PR_BODY``, so the body IS
the message of the commit that reaches ``main``. One guard already reads it, in
``.github/workflows/ci.yml`` (#848), and that guard is correct. It is also not a
precondition of merging: on #1257 it was still ``pending`` when the merge went
ahead, the body carried a malformed attribution value, and the result is a
message on ``main`` that can never be repaired, because ``main`` is never
force-pushed (#1262). This command is the operator's own local read of the same
bytes, in their shell instead of in a queue (#1263).

THE RULE IS NOT HERE. ``scripts/check-commit-trailers.py`` owns it, and this
tool passes it the body on standard input with the same ``--message-file -
--filled`` the CI step uses. One implementation, two callers, and nothing to
drift.

WHY THIS IS NOT A ONE-LINE PIPE. Measured before it was written::

    gh pr view 999999 --json body --jq .body | check-commit-trailers.py --message-file - --filled
    -> commit trailer check FAILED: the message is empty.

The forge had answered "could not resolve to a pull request". A pipe throws away
``gh``'s exit status, so an unreachable forge rendered as a verdict about a body
nobody had read. Here ``gh`` runs with its status captured and an unread body is
reported as ``REMOTE_UNVERIFIED``, which is the protocol's answer to unknown. A
skip is not.

THIS IS NOT A CHECKER. It reaches the network, so no gate may depend on it and
``scripts/agent-preflight.sh`` must never run it. Its OFFLINE half is
``--body-file``, which validates a message already on disk and never invokes
``gh``; that is the half the suite drives.

    scripts/agent-pr-body.py --pr 1263          # read the live body, origin then upstream
    scripts/agent-pr-body.py --body-file draft  # the same contract, offline

Exit statuses: 0 the body will land clean, 1 the body was read and fails the
contract, 3 the body was NOT read. 2 belongs to ``argparse`` and is a usage
error. Refusing to conflate 1 and 3 is the whole point: this row exists because
"nobody looked" was allowed to pass for "looked and fine".
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts/check-commit-trailers.py"
REPOSITORY = re.compile(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+\Z")

EXIT_OK = 0
EXIT_CONTRACT = 1
EXIT_UNVERIFIED = 3


class Unverified(ValueError):
    """No verdict could be rendered. Carries its own reported prefix."""


def _load_ready():
    """`scripts/agent-ready.py` as a module, for ONE reading of `origin`.

    The identity of this repository is already derived there and is already
    tested there. Copying the expression would give the two commands two answers
    to the same question, which is the defect this whole tool exists to avoid at
    a different scale.
    """

    spec = importlib.util.spec_from_file_location(
        "agent_ready_for_pr_body", ROOT / "scripts/agent-ready.py"
    )
    if spec is None or spec.loader is None:
        raise Unverified("UNVERIFIED: cannot load scripts/agent-ready.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


REMOTES = ("origin", "upstream")


def repositories() -> list[str]:
    """Candidate ``owner/name`` for ``--pr``, in the order they are asked.

    ``origin`` first, then ``upstream``. A pull request number is unique inside
    one repository and means nothing outside it: on a fork checkout ``origin``
    is the fork, which carries none of the numbers that land upstream, so the
    fork's answer is ``Could not resolve`` for a pull request that exists one
    remote away.

    ``origin`` leads because an invocation that works today must keep meaning
    what it meant: a maintainer whose ``origin`` is canonical never spends a
    call on a remote that may be somebody else's fork, and a number present in
    both repositories is still read from the one this clone was made from. A
    checkout with no ``origin`` at all stays the refusal it always was, rather
    than being silently re-pointed at whatever other remote happens to exist.

    The identity is derived by ``agent-ready.py``, so the two commands cannot
    disagree about what a remote URL names.
    """

    ready = _load_ready()
    candidates: list[str] = []
    for remote in REMOTES:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "remote", "get-url", remote],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if result.returncode:
            if remote == REMOTES[0]:
                raise Unverified(
                    f"REMOTE_UNVERIFIED: {result.stderr.strip() or 'no origin remote'}"
                )
            continue
        try:
            identity = ready.repository_identity(result.stdout.strip())
        except ready.RemoteUnverified as exc:
            if remote == REMOTES[0]:
                raise Unverified(f"REMOTE_UNVERIFIED: {exc}") from exc
            continue
        if identity not in candidates:
            candidates.append(identity)
    return candidates


def fetch_body(repo: str, number: int) -> str:
    """The live body, or ``Unverified``. Never an empty string standing in.

    ``--json body`` rather than ``--jq .body``: the JSON is parsed here, so an
    absent or null field is distinguishable from a body that is genuinely the
    empty string, and the answer does not depend on how one build of ``gh``
    renders a null.
    """

    try:
        result = subprocess.run(
            ["gh", "pr", "view", str(number), "--repo", repo, "--json", "body"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise Unverified(f"REMOTE_UNVERIFIED: cannot run gh: {exc}") from exc
    if result.returncode:
        detail = result.stderr.strip() or f"gh exited {result.returncode}"
        raise Unverified(f"REMOTE_UNVERIFIED: {detail}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise Unverified(f"REMOTE_UNVERIFIED: gh returned malformed JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("body"), str):
        raise Unverified("REMOTE_UNVERIFIED: gh returned no body field for this pull request")
    return payload["body"]


def fetch_live_body(number: int) -> str:
    """The body from the first candidate repository that carries ``number``.

    The first repository that answers is the answer, and a fallback is named on
    stderr: a body is judged as the bytes of one repository's pull request, so
    the operator has to be able to see which one produced them.

    No candidate answering is ``REMOTE_UNVERIFIED`` and never "this pull
    request does not exist". Only a ``Could not resolve`` reply proves absence,
    and any other failure may be a forge that did not answer, so every failure
    is carried with the repository that produced it.
    """

    candidates = repositories()
    failures: list[str] = []
    for repo in candidates:
        try:
            body = fetch_body(repo, number)
        except Unverified as exc:
            failures.append(f"{repo}: {exc}")
            continue
        if repo != candidates[0]:
            print(
                f"NOTE: #{number} is not in {candidates[0]}; reading it from {repo}",
                file=sys.stderr,
            )
        return body
    raise Unverified(
        "REMOTE_UNVERIFIED: no candidate repository returned this pull request.\n  "
        + "\n  ".join(failures)
    )


def read_body_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise Unverified(f"UNVERIFIED: cannot read {path}: {exc}") from exc


def validate(body: str) -> int:
    """Delegate to the checker. Its verdict is the verdict, and its report the report."""

    if not CHECKER.is_file():
        raise Unverified(f"UNVERIFIED: {CHECKER} is missing")
    result = subprocess.run(
        [sys.executable, str(CHECKER), "--message-file", "-", "--filled"],
        input=body, text=True, cwd=ROOT,
    )
    if result.returncode not in {0, 1}:
        raise Unverified(
            "UNVERIFIED: the trailer checker rendered no verdict "
            f"(exit {result.returncode})"
        )
    verdict = EXIT_CONTRACT if result.returncode else EXIT_OK
    problems, warnings = closing_keyword_records(body)
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    for problem in problems:
        print(f"ERROR: {problem}", file=sys.stderr)
    return EXIT_CONTRACT if problems else verdict


CLOSING = re.compile(
    r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#([1-9][0-9]*)\b", re.IGNORECASE
)


def closing_keyword_records(body: str) -> list[str]:
    """Every `Closes #N` in the body whose local issue record does not read CLOSED.

    The body IS the squash message, so a closing keyword here closes the GitHub
    mirror the moment it lands. Local files are the issue authority, so if the
    branch does not also carry `ISSUE-GH-N.md` reading `State: CLOSED`, the two
    authorities disagree from the instant of the merge and the local record --
    the authoritative one -- is the half left saying OPEN.

    Read from the working tree, so run this from the branch being merged; that
    is the same tree whose records would land.
    """

    problems: list[str] = []
    warnings: list[str] = []
    for number in sorted({m.group(1) for m in CLOSING.finditer(body)}, key=int):
        matches = sorted((ROOT / ".agents/issues").rglob(f"ISSUE-GH-{number}.md"))
        if not matches:
            # WARNING, not a refusal. An authority that does not exist cannot be
            # contradicted, and issues predating the local-first migration have no
            # record at all: measured across the open queue this arm fires on 22 of
            # 24 failures and would have blocked three pull requests that legitimately
            # landed on 2026-09-11 (#3141, #3138, #3128). A gate that fires on
            # ordinary work is the defect, not the discipline.
            warnings.append(
                f"body closes #{number} and no .agents/issues/**/ISSUE-GH-{number}.md "
                "exists; import it (scripts/agent-issue.py import-github) so the local "
                "record carries the resolution evidence"
            )
            continue
        text = matches[0].read_text(encoding="utf-8", errors="replace")
        state = next(
            (l.split(":", 1)[1].strip() for l in text.splitlines() if l.startswith("State:")),
            "",
        )
        if state != "CLOSED":
            problems.append(
                f"body closes #{number} but {matches[0].relative_to(ROOT)} reads "
                f"State: {state or '(absent)'}; close it locally first "
                "(scripts/agent-issue.py close) or drop the keyword"
            )
    return problems, warnings


def pull_request_number(value: str) -> int:
    if re.fullmatch(r"[1-9][0-9]*", value) is None:
        raise argparse.ArgumentTypeError("a pull request number is a positive integer")
    return int(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pr", type=pull_request_number, help="pull request number")
    parser.add_argument(
        "--body-file", type=Path, help="validate a message already on disk, offline"
    )
    parser.add_argument(
        "--repo",
        help="owner/name; when omitted, origin then upstream are asked in that order",
    )
    args = parser.parse_args()

    if (args.pr is None) == (args.body_file is None):
        parser.error("pass exactly one of --pr or --body-file")
    if args.repo is not None and REPOSITORY.fullmatch(args.repo) is None:
        parser.error("--repo must be owner/name")
    if args.repo is not None and args.pr is None:
        parser.error("--repo is meaningless without --pr")

    try:
        if args.body_file is not None:
            body = read_body_file(args.body_file)
        elif args.repo is not None:
            body = fetch_body(args.repo, args.pr)
        else:
            body = fetch_live_body(args.pr)
        return validate(body)
    except Unverified as exc:
        print(exc, file=sys.stderr)
        print(
            "  Unknown is not absence and not success. Nothing was read, so "
            "nothing is known about the message this pull request would land.",
            file=sys.stderr,
        )
        return EXIT_UNVERIFIED


if __name__ == "__main__":
    raise SystemExit(main())
