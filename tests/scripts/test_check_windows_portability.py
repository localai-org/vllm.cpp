#!/usr/bin/env python3
"""Behavior tests for the fail-closed Windows portability checker."""

from __future__ import annotations

import json
import importlib.util
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock


REPO = Path(__file__).resolve().parents[2]
CHECKER = REPO / "scripts" / "check-windows-portability.py"
CHECKER_SPEC = importlib.util.spec_from_file_location("windows_portability", CHECKER)
assert CHECKER_SPEC is not None and CHECKER_SPEC.loader is not None
checker = importlib.util.module_from_spec(CHECKER_SPEC)
CHECKER_SPEC.loader.exec_module(checker)
UNSUPPORTED_TIER_FILTER = (
    "--test-case=elementwise CPU GEMM: the forced tier is the tier that actually ran"
)
NOMINMAX_REAL_CLOSURE = (
    "src/vllm/model_executor/models/minimax_h3_sharded.cpp",
    "src/vllm/v1/kv_offload/fs_io.cpp",
    "src/vt/cpu/cpu_threadpool.cpp",
    "src/vllm/platform/console_shutdown.cpp",
    "src/vllm/platform/process.cpp",
)


# The single MSVC warning arm of the safe fixture, replaced wholesale by the
# #774 warning-policy tests. Anchored by count, never by "it is in there".
MSVC_WARNING_ARM = "add_compile_options(/fp:strict /W4 /WX)"


SAFE_FILES = {
    "CMakeLists.txt": """
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
        if(MSVC)
          add_compile_definitions(NOMINMAX _CRT_SECURE_NO_WARNINGS)
          add_compile_options(/fp:strict /W4 /WX)
        else()
          add_compile_options(-ffp-contract=off)
        endif()
        set_source_files_properties(src/vt/cpu/cpu_matmul_elem_avx2.cpp
          PROPERTIES COMPILE_OPTIONS "$<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>")
        set_source_files_properties(src/vt/cpu/cpu_matmul_elem_f16c.cpp
          PROPERTIES COMPILE_OPTIONS
            "$<$<CXX_COMPILER_ID:MSVC>:/arch:AVX>;$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-mf16c>")
    """,
    "src/vllm/entrypoints/openai/server_main.cpp": """
        int server_main_contract;
    """,
    "src/vllm/platform/process.cpp": """
        #include "vllm/platform/process.h"
        #ifdef _WIN32
        CreateProcessW(nullptr, command_line.data(), nullptr, nullptr, FALSE,
                       0, nullptr, nullptr, &startup, &process);
        #else
        #include <sys/wait.h>
        fork(); execvp(argv[0], argv); waitpid(pid, &status, 0);
        #endif
    """,
    "src/vllm/platform/console_shutdown.cpp": """
        #ifdef _WIN32
        struct WindowsHandlerRegistry {
          std::atomic<WindowsHandlerState*> published;
          std::atomic<unsigned> entrants;
        };
        WindowsHandlerState::~WindowsHandlerState() {
          if (quit_event) CloseHandle(quit_event);
          if (stop_event) CloseHandle(stop_event);
        }
        WindowsHandlerRegistry& HandlerRegistry() {
          static auto* registry = new WindowsHandlerRegistry();
          return *registry;
        }
        bool DrainEntrantsWithTimeout(std::atomic<unsigned>& entrants) {
          const ULONGLONG start = GetTickCount64();
          while (entrants.load(std::memory_order_seq_cst) != 0) {
            if (GetTickCount64() - start >= 5000) return false;
            SwitchToThread();
          }
          return true;
        }
        bool DrainInFlightWithTimeout(std::atomic<unsigned>& in_flight) {
          const ULONGLONG start = GetTickCount64();
          while (in_flight.load(std::memory_order_seq_cst) != 0) {
            if (GetTickCount64() - start >= 5000) return false;
            SwitchToThread();
          }
          return true;
        }
        std::atomic<unsigned> in_flight;
        bool DispatchControlEvent(DWORD event) {
          auto& registry = HandlerRegistry();
          registry.entrants.fetch_add(1, std::memory_order_seq_cst);
          WindowsHandlerState* state =
              registry.published.load(std::memory_order_seq_cst);
          if (state != nullptr) {
            state->in_flight.fetch_add(1, std::memory_order_seq_cst);
          }
          registry.entrants.fetch_sub(1, std::memory_order_seq_cst);
          if (state == nullptr) return false;
          const bool resumed = true;
          SetEvent(state->stop_event);
          state->in_flight.fetch_sub(1, std::memory_order_seq_cst);
          return resumed;
        }
        BOOL WINAPI ConsoleControlHandler(DWORD event) {
          return DispatchControlEvent(event);
        }
        SetConsoleCtrlHandler(ConsoleControlHandler, TRUE);
        ConsoleShutdown::~ConsoleShutdown() {
        registry.published.store(nullptr, std::memory_order_seq_cst);
        DrainEntrantsWithTimeout(registry.entrants);
        DrainInFlightWithTimeout(state->in_flight);
        if (!safe_to_close) RetireHandlerState(std::move(state));
        else state.reset();
        }
        #else
        #include <unistd.h>
        pipe(fds); read(fds[0], data, 1); write(fds[1], data, 1); close(fds[0]);
        #endif
    """,
    "src/vllm/v1/kv_offload/lmcache/remote_client.cpp": """
        #ifdef _WIN32
        WSAStartup(MAKEWORD(2, 2), &data);
        WSAGetLastError(); closesocket(socket);
        if (recv_result == 0) { Close(); throw peer_closed; }
        #else
        #include <sys/socket.h>
        close(socket);
        #endif
    """,
    "src/vllm/v1/kv_offload/fs_io.cpp": """
        #ifdef _WIN32
        #include <windows.h>
        CreateFileW(path.c_str(), GENERIC_READ, 0, nullptr, OPEN_EXISTING,
                    FILE_ATTRIBUTE_NORMAL, nullptr); CREATE_NEW;
        FlushFileBuffers(file);
        MoveFileExW(source.c_str(), destination.c_str(),
                    MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
        #else
        #include <unistd.h>
        open(path, O_RDONLY); read(fd, data, size); close(fd);
        #endif
    """,
    "scripts/build-windows-release.ps1": """
        function Invoke-UnsupportedTierProbe {
          param([Parameter(Mandatory)][string]$TierTest,
                [scriptblock]$Runner)
          $arguments = @(
            '--test-case=elementwise CPU GEMM: the forced tier is the tier that actually ran'
          )
          if ($null -eq $Runner) {
            $probeOutput = @(& $TierTest @arguments 2>&1)
            $probeExitCode = $LASTEXITCODE
          } else {
            $probeResult = & $Runner $TierTest $arguments
            $probeOutput = @($probeResult.Output)
            $probeExitCode = [int]$probeResult.ExitCode
          }
          if ($probeExitCode -ne 1) { throw "unexpected exit" }
          $diagnostic = $probeOutput -join "`n"
          if ($diagnostic -notmatch
              [regex]::Escape("unknown x86 ISA tier 'amx'")) {
            throw "missing diagnostic"
          }
        }
        function Invoke-UnsupportedTierContractTests {
          $calls = [System.Collections.Generic.List[object]]::new()
          $good = {
            param([string]$Program, [string[]]$Arguments)
            $calls.Add([pscustomobject]@{ Arguments = @($Arguments) })
            [pscustomobject]@{
              ExitCode = 1
              Output = @("unknown x86 ISA tier 'amx'")
            }
          }.GetNewClosure()
          Invoke-UnsupportedTierProbe -TierTest "fake-tier-test.exe" -Runner $good
          if ($calls.Count -ne 1) { throw "wrong fake call count" }
          if ($calls[0].Arguments.Count -ne 1) { throw "wrong fake argument count" }
          if ($calls[0].Arguments[0] -ne '--test-case=elementwise CPU GEMM: the forced tier is the tier that actually ran') {
            throw "wrong fake argument"
          }
          $badResults = @(
            [pscustomobject]@{ ExitCode = 0 },
            [pscustomobject]@{ ExitCode = 134 },
            [pscustomobject]@{ ExitCode = -1073741819 },
            [pscustomobject]@{ ExitCode = 3 },
            [pscustomobject]@{ ExitCode = 2 }
          )
        }
        if ($ContractTest) {
          Invoke-CrtContractTests
          Invoke-UnsupportedTierContractTests
        }
        New-Item .cmake/api/v1/query/codemodel-v2
        cmake -S . -B $BuildDir -G "Visual Studio 17 2022" -A x64 `
          -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded
        python scripts/check-windows-portability.py --build-dir $BuildDir
        cmake --build $BuildDir --config Release --target server tests
        & "$BuildDir/tests/test_openai_api_server.exe"
        cmake --install $BuildDir --config Release
        dumpbin /directives library.lib
        dumpbin /imports bin/vllm-server.exe
        Invoke-CrtAudit
        Invoke-Checked $server @("--help")
        $savedTier = $env:VT_CPU_MATMUL_TIER
        try {
          $env:VT_CPU_MATMUL_TIER = "portable"
          Invoke-Checked $tierTest @()
          $env:VT_CPU_MATMUL_TIER = "avx2"
          Invoke-Checked $tierTest @()
          $env:VT_CPU_MATMUL_TIER = "amx"
          Invoke-UnsupportedTierProbe -TierTest $tierTest
        } finally {
          $env:VT_CPU_MATMUL_TIER = $savedTier
        }
        Invoke-WebRequest "$BaseUrl/health"
        Invoke-WebRequest "$BaseUrl/version"
        CTRL_BREAK_EVENT
        Invoke-Checked python @($smokeHarness, $server)
        VCRUNTIME MSVCP CONCRT UCRTBASE api-ms-win-crt- MSVCR LIBCMT
    """,
    "src/vt/cpu/cpu_matmul_elem.cpp": """
        void PortableDispatcher();
    """,
    "src/vllm/model_executor/models/minimax_h3_sharded.cpp": """
        #include <filesystem>
        bool ok = std::filesystem::is_directory(path) ||
                  std::filesystem::is_regular_file(path);
    """,
    "include/vllm/model_executor/models/device_pool.h": """
        #include <bit>
        auto width = std::bit_width(bytes);
    """,
}


class WindowsPortabilityCheckerTest(unittest.TestCase):
    def make_tree(self, mutations: dict[str, str] | None = None) -> Path:
        root = Path(self.tempdir.name)
        files = dict(SAFE_FILES)
        files.update(mutations or {})
        for relative, content in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(textwrap.dedent(content), encoding="utf-8")
        self.source_manifest = root / ".windows-portability-sources.json"
        self.source_manifest.write_text(
            json.dumps({"sources": sorted(files)}), encoding="utf-8"
        )
        return root

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def run_checker(self, root: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(CHECKER), "--root", str(root),
             "--test-source-manifest", str(self.source_manifest)],
            text=True,
            capture_output=True,
            check=False,
        )

    def assert_rejected(self, relative: str, content: str, reason: str) -> None:
        result = self.run_checker(self.make_tree({relative: content}))
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(reason, result.stdout + result.stderr)

    @staticmethod
    def nominmax_real_sources() -> dict[str, str]:
        return {
            relative: (REPO / relative).read_text(encoding="utf-8")
            for relative in NOMINMAX_REAL_CLOSURE
        }

    @staticmethod
    def with_nominmax_fallback(source: str, *, before: bool,
                               guarded: bool = True) -> str:
        header = "#include <windows.h>"
        if source.count(header) != 1:
            raise AssertionError("expected exactly one windows.h include")
        definition = "#define NOMINMAX\n"
        if guarded:
            definition = "#ifndef NOMINMAX\n#define NOMINMAX\n#endif\n"
        return source.replace(
            header,
            definition + header if before else header + "\n" + definition,
        )

    def test_accepts_complete_guarded_contract(self) -> None:
        result = self.run_checker(self.make_tree())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_real_tree_declares_strict_msvc_platform_contract(self) -> None:
        cmake = (REPO / "CMakeLists.txt").read_text(encoding="utf-8")
        warnings = (REPO / "cmake/CompilerWarnings.cmake").read_text(
            encoding="utf-8"
        )
        contract = cmake + "\n" + warnings
        for token in (
            "NOMINMAX",
            "_CRT_SECURE_NO_WARNINGS",
            "/utf-8",
            "/W4",
            "/WX",
        ):
            with self.subTest(token=token):
                self.assertIn(token, contract)

    def msvc_warning_arm(self, replacement: str) -> str:
        """Return the safe CMakeLists with its MSVC warning arm replaced."""
        cmake = textwrap.dedent(SAFE_FILES["CMakeLists.txt"])
        if cmake.count(MSVC_WARNING_ARM) != 1:
            raise AssertionError("expected exactly one MSVC warning arm")
        return cmake.replace(MSVC_WARNING_ARM, replacement)

    def test_real_tree_msvc_warning_policy_reaches_the_cxx_compile(self) -> None:
        """#774: the tokens must survive the reader the checker actually uses."""
        cmake = (REPO / "CMakeLists.txt").read_text(encoding="utf-8")
        warnings = (REPO / "cmake/CompilerWarnings.cmake").read_text(
            encoding="utf-8"
        )
        # #1649: scoped exactly as check() scopes it -- the vendored BoringSSL
        # `/w` is PRIVATE on fetched targets and does not answer for the policy.
        reaching = checker.msvc_cxx_flag_text(
            checker.without_foreign_target_compile_options(cmake) + "\n" + warnings
        )
        for flag in ("/W4", "/WX"):
            with self.subTest(required=flag):
                self.assertTrue(checker.has_msvc_flag(reaching, flag))
        for flag in ("/WX-", "/W0", "/w"):
            with self.subTest(negation=flag):
                self.assertFalse(checker.has_msvc_flag(reaching, flag))
        # The prose in CMakeLists.txt is NOT what answers for the policy.
        self.assertIn("/W4 /WX remains unchanged", cmake)
        self.assertFalse(
            checker.has_msvc_flag(checker.msvc_cxx_flag_text(cmake), "/WX")
        )

    def test_rejects_warnings_as_errors_disabled_by_wx_minus(self) -> None:
        """`"/WX" in "/WX-"` is True, and /WX- is the INVERSE policy (#774)."""
        disabled = self.msvc_warning_arm(
            "add_compile_options(/fp:strict /W4 /WX-)"
        )
        self.assert_rejected("CMakeLists.txt", disabled, "negated on the C/C++")
        self.assert_rejected("CMakeLists.txt", disabled, "missing /WX")

    def test_rejects_policy_satisfied_only_on_objcxx(self) -> None:
        """The measured PR #640 shape at commit 74ba3823f (#774).

        Objective-C++ is the Metal backend and never compiles under MSVC, so a
        bare /WX confined to it answers for nothing on Windows.
        """
        objcxx_only = self.msvc_warning_arm(
            "set_property(TARGET vllm PROPERTY COMPILE_WARNING_AS_ERROR OFF)\n"
            "add_compile_options(/fp:strict\n"
            "  $<$<COMPILE_LANGUAGE:CXX>:/W4>\n"
            "  $<$<COMPILE_LANGUAGE:CXX>:/WX->\n"
            "  $<$<COMPILE_LANGUAGE:OBJCXX>:/W4>\n"
            "  $<$<COMPILE_LANGUAGE:OBJCXX>:/WX>)"
        )
        self.assert_rejected("CMakeLists.txt", objcxx_only, "missing /WX")
        self.assert_rejected("CMakeLists.txt", objcxx_only, "negated on the C/C++")

    def test_rejects_prefix_lookalike_warning_flags(self) -> None:
        """/W44996 sets ONE warning to level 4; /WXsomething is not /WX."""
        lookalikes = self.msvc_warning_arm(
            "add_compile_options(/fp:strict /W44996 /WXsomething)"
        )
        self.assert_rejected("CMakeLists.txt", lookalikes, "missing /W4 /WX")

    def test_rejects_warning_level_disabled_alongside_the_policy(self) -> None:
        """/w is the disable spelling of /W4 and wins as the later flag."""
        self.assert_rejected(
            "CMakeLists.txt",
            self.msvc_warning_arm("add_compile_options(/fp:strict /W4 /WX /w)"),
            "negated on the C/C++ compile by /w",
        )

    def test_a_vendored_target_may_silence_its_own_warnings(self) -> None:
        """#1649: `/w` PRIVATE on a FETCHED target is not a project-wide negation.

        `target_compile_options(<vendored> PRIVATE /w)` reaches only that
        target's translation units. Reading it as a negation of the project
        policy is what made `windows-msvc-cpu` red on main and on every pull
        request, for a `/w` the vendored BoringSSL targets legitimately want and
        that CMakeLists.txt has a comment explaining.
        """
        vendored = self.msvc_warning_arm(
            "add_compile_options(/fp:strict /W4 /WX)\n"
            "foreach(_boringssl_target ssl crypto)\n"
            "  target_compile_options(${_boringssl_target} PRIVATE\n"
            "    $<IF:$<CXX_COMPILER_ID:MSVC>,/w,-w>)\n"
            "endforeach()"
        )
        result = self.run_checker(self.make_tree({"CMakeLists.txt": vendored}))
        self.assertNotIn("negated on the C/C++", result.stdout + result.stderr)

    def test_a_project_target_may_not_silence_the_policy(self) -> None:
        """The property #1649 must not break: the narrowing is by TARGET.

        `vllm` is declared in this tree by `add_library`, so silencing it IS a
        negation of the policy on this project's own code, and stays refused.
        Without this the vendored allowance would be a hole big enough to drive
        the whole library through.
        """
        own_target = self.msvc_warning_arm(
            "add_compile_options(/fp:strict /W4 /WX)\n"
            "add_library(vllm STATIC src/vt/cpu/cpu_matmul_elem.cpp)\n"
            "target_compile_options(vllm PRIVATE /w)"
        )
        self.assert_rejected(
            "CMakeLists.txt", own_target, "negated on the C/C++ compile by /w"
        )

    def test_a_foreach_binding_naming_a_project_target_is_not_vendored(self) -> None:
        """The loop variable is RESOLVED, not trusted because it is a variable.

        `foreach(t vllm)` + `target_compile_options(${t} PRIVATE /w)` is the
        same negation as naming `vllm` outright, so hiding it behind the
        vendored construct must not launder it.
        """
        laundered = self.msvc_warning_arm(
            "add_compile_options(/fp:strict /W4 /WX)\n"
            "add_library(vllm STATIC src/vt/cpu/cpu_matmul_elem.cpp)\n"
            "foreach(_t vllm)\n"
            "  target_compile_options(${_t} PRIVATE /w)\n"
            "endforeach()"
        )
        self.assert_rejected(
            "CMakeLists.txt", laundered, "negated on the C/C++ compile by /w"
        )

    def test_an_unresolvable_target_stays_in_scope(self) -> None:
        """Fail-safe direction: what the checker cannot resolve, it still reads.

        An unbound `${...}` could name anything, including a project target, so
        the narrowing declines to launder it.
        """
        opaque = self.msvc_warning_arm(
            "add_compile_options(/fp:strict /W4 /WX)\n"
            "target_compile_options(${_unbound} PRIVATE /w)"
        )
        self.assert_rejected(
            "CMakeLists.txt", opaque, "negated on the C/C++ compile by /w"
        )

    def test_rejects_policy_declared_only_in_a_comment(self) -> None:
        self.assert_rejected(
            "CMakeLists.txt",
            self.msvc_warning_arm("# the MSVC policy is /W4 /WX"),
            "missing /W4 /WX",
        )

    def test_accepts_the_policy_on_every_shape_that_reaches_cxx(self) -> None:
        """The inverse pin: repaired does not mean stricter about everything."""
        for label, arm in (
            ("cxx genex", "add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4 /WX>)"),
            ("c and cxx", "add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/W4 /WX>)"),
            ("config genex",
             "add_compile_options($<$<CONFIG:Debug>:/W4> $<$<CONFIG:Debug>:/WX>)"),
            # Targeted suppressions are a DELIBERATE non-goal of #774: they
            # narrow what /W4 reports, they do not invert /W4 or /WX.
            ("targeted suppression",
             "add_compile_options(/fp:strict /W4 /WX /wd4324)"),
        ):
            with self.subTest(shape=label):
                result = self.run_checker(
                    self.make_tree({"CMakeLists.txt": self.msvc_warning_arm(arm)})
                )
                self.assertEqual(
                    result.returncode, 0, result.stdout + result.stderr
                )

    def test_real_tree_has_no_unguarded_local_nominmax_redefinitions(self) -> None:
        cmake = (REPO / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertRegex(
            cmake,
            r"add_compile_definitions\([^)]*\bNOMINMAX\b[^)]*\)",
        )

        def is_absence_guard(directive: str) -> bool:
            return bool(
                re.fullmatch(r"ifndef\s+NOMINMAX", directive)
                or re.fullmatch(
                    r"if\s+!\s*defined\s*(?:\(\s*NOMINMAX\s*\)|NOMINMAX)",
                    directive,
                )
            )

        offenders = []
        for root in (REPO / "src", REPO / "tests"):
            for path in root.rglob("*"):
                if path.suffix not in {".cc", ".cpp", ".cxx", ".h", ".hpp"}:
                    continue
                absence_guards = []
                for line in path.read_text(encoding="utf-8").splitlines():
                    match = re.match(r"\s*#\s*(\w+)(.*)$", line)
                    if match is None:
                        continue
                    kind, tail = match.group(1), match.group(2).strip()
                    if kind in {"if", "ifdef", "ifndef"}:
                        absence_guards.append(is_absence_guard(f"{kind} {tail}"))
                    elif kind in {"else", "elif"}:
                        self.assertTrue(absence_guards, path)
                        absence_guards[-1] = (
                            is_absence_guard(f"if {tail}")
                            if kind == "elif"
                            else False
                        )
                    elif kind == "endif":
                        self.assertTrue(absence_guards, path)
                        absence_guards.pop()
                    elif (
                        kind == "define"
                        and re.match(r"NOMINMAX\b", tail)
                        and not any(absence_guards)
                    ):
                        offenders.append(path.relative_to(REPO).as_posix())
                self.assertEqual(absence_guards, [], path)

        self.assertEqual(sorted(offenders), [])

    def test_real_tree_uses_portable_windows_allocation_and_math(self) -> None:
        backend = (REPO / "src/vt/cpu/cpu_backend.cpp").read_text(
            encoding="utf-8"
        )
        for token in ("_aligned_malloc", "_aligned_free", "std::aligned_alloc"):
            with self.subTest(token=token):
                self.assertIn(token, backend)

        offenders = []
        for root in (REPO / "src", REPO / "tests"):
            for path in root.rglob("*"):
                if path.suffix in {".cc", ".cpp", ".h", ".hpp"}:
                    if re.search(r"\bM_PI\b", path.read_text(encoding="utf-8")):
                        offenders.append(path.relative_to(REPO).as_posix())
        self.assertEqual(offenders, [])

    def test_real_tree_closes_observed_msvc_source_warnings(self) -> None:
        forbidden = {
            "src/vt/cpu/cpu_paged_attn.cpp": "constexpr KvKind kKV",
            "src/vllm/v1/worker/gpu/runner.cpp": (
                "std::vector<int32_t> positions(step.positions.begin()"
            ),
            "src/vllm/model_executor/models/gemma4_audio.cpp": (
                "std::max<double>(nts - 1, 1)"
            ),
            "src/vllm/v1/kv_offload/lmcache/lmcache_connector.cpp": (
                "conn_cfg.num_layers = ctx.identity->num_hidden_layers"
            ),
            "src/vllm/v1/core/kv_cache_utils.cpp": "MemAvailable: %ld kB",
            "src/vllm/model_executor/models/minimax_h3_video_vae.cpp": (
                "float* v = target->data()"
            ),
        }
        for relative, token in forbidden.items():
            with self.subTest(relative=relative):
                source = (REPO / relative).read_text(encoding="utf-8")
                self.assertNotIn(token, source)

        threadpool = (REPO / "src/vt/cpu/cpu_threadpool.h").read_text(
            encoding="utf-8"
        )
        for token in (
            "#pragma warning(push)",
            "#pragma warning(disable : 4324)",
            "#pragma warning(pop)",
        ):
            with self.subTest(token=token):
                self.assertIn(token, threadpool)

        manager_source = (
            REPO / "src/vllm/v1/core/single_type_kv_cache_manager.cpp"
        ).read_text(encoding="utf-8")
        active = checker._cpp_structural_view(manager_source)
        shadow_contracts = {
            "FullAttentionManager::find_longest_cache_hit": (
                r"\bBlockPool\s*&\s*block_pool\b",
                r"\bKVCacheSpec\s*&\s*kv_cache_spec\b",
                r"\b(?:const\s+)?int\s+block_size\b",
            ),
            "SlidingWindowManager::find_longest_cache_hit": (
                r"\bBlockPool\s*&\s*block_pool\b",
                r"\bKVCacheSpec\s*&\s*kv_cache_spec\b",
                r"\b(?:const\s+)?int\s+block_size\b",
            ),
            "SlidingWindowManager::reachable_block_mask": (
                r"\b(?:const\s+)?int\s+block_size\b",
            ),
            "ChunkedLocalAttentionManager::find_longest_cache_hit": (
                r"\bBlockPool\s*&\s*block_pool\b",
                r"\bKVCacheSpec\s*&\s*kv_cache_spec\b",
            ),
            "MambaManager::find_longest_cache_hit": (
                r"\bBlockPool\s*&\s*block_pool\b",
                r"\bKVCacheSpec\s*&\s*kv_cache_spec\b",
                r"\b(?:const\s+)?int\s+block_size\b",
            ),
        }
        for method, declarations in shadow_contracts.items():
            signature = rf"\b{re.escape(method)}\s*\("
            span = checker._cpp_function_body_span(manager_source, signature)
            self.assertIsNotNone(span, method)
            assert span is not None
            match = re.search(signature, active)
            self.assertIsNotNone(match, method)
            assert match is not None
            method_scope = active[match.start():span[2] + 1]
            for declaration in declarations:
                with self.subTest(method=method, declaration=declaration):
                    self.assertNotRegex(method_scope, declaration)

        deepseek = (
            REPO / "src/vllm/model_executor/models/deepseek_v4.cpp"
        ).read_text(encoding="utf-8")
        active_deepseek = checker._cpp_structural_view(deepseek)
        probe_contracts = (
            r"detail::DeepseekV4ExpertProbeInput\s*\(\s*H\s*,\s*0\.017f\s*\)",
            r"detail::DeepseekV4ExpertProbeInput\s*\(\s*mi\s*,\s*0\.013f\s*\)",
            r"0\.5f\s*\*\s*std::sin\s*\(\s*frequency\s*\*\s*"
            r"static_cast<float>\s*\(\s*i\s*\+\s*1\s*\)\s*\)",
        )
        for contract in probe_contracts:
            with self.subTest(contract=contract):
                self.assertRegex(active_deepseek, contract)

    def test_logprobs_slice_parameter_does_not_shadow_member(self) -> None:
        header = (REPO / "include/vllm/v1/outputs.h").read_text(
            encoding="utf-8"
        )
        active_header = checker._cpp_structural_view(header)
        header_signature = r"\bLogprobsTensors\s+slice_request\s*\("
        header_matches = list(re.finditer(header_signature, active_header))
        self.assertEqual(
            len(header_matches),
            1,
            "LogprobsTensors::slice_request declaration",
        )
        header_match = header_matches[0]
        header_end = re.match(
            r"(?:[^()]|\([^()]*\))*\)\s*const\s*;",
            active_header[header_match.end():],
        )
        self.assertIsNotNone(
            header_end,
            "LogprobsTensors::slice_request declaration",
        )
        assert header_end is not None
        header_declaration = active_header[
            header_match.start():header_match.end() + header_end.end()
        ]

        source = (REPO / "src/vllm/v1/outputs.cpp").read_text(
            encoding="utf-8"
        )
        active_source = checker._cpp_structural_view(source)
        source_signature = r"\bLogprobsTensors::slice_request\s*\("
        span = checker._cpp_function_body_span(source, source_signature)
        self.assertIsNotNone(span, "LogprobsTensors::slice_request")
        assert span is not None
        source_matches = list(re.finditer(source_signature, active_source))
        self.assertEqual(
            len(source_matches),
            1,
            "LogprobsTensors::slice_request definition",
        )
        source_declaration = active_source[source_matches[0].start():span[1]]

        for location, declaration in (
            ("header", header_declaration),
            ("source", source_declaration),
        ):
            with self.subTest(location=location):
                self.assertNotRegex(declaration, r"\bint\s+num_positions\b")

    def test_windows_command_line_expectation_is_not_stringized(self) -> None:
        source = (
            REPO / "tests/vllm/entrypoints/openai/test_api_server.cpp"
        ).read_text(encoding="utf-8")
        case_start = source.index(
            'TEST_CASE("platform process: Windows command line preserves '
            'every argv byte")'
        )
        case_end = source.index("\nTEST_CASE(", case_start + 1)
        case = source[case_start:case_end]

        binding = re.search(
            r'const\s+std::wstring\s+expected\s*=\s*'
            r'LR"(?P<delimiter>[^\s()\\]{1,16})\('
            r'(?P<payload>.*?)\)(?P=delimiter)"\s*;',
            case,
            re.DOTALL,
        )
        self.assertIsNotNone(binding, "named custom-delimited expectation")
        assert binding is not None
        self.assertEqual(
            binding.group("payload"),
            r'"ffmpeg" "two words" "C:\path\\" "a\"b" ""',
        )
        self.assertRegex(
            case,
            r"CHECK\s*\(\s*vllm::platform::BuildWindowsCommandLine\s*"
            r"\(\s*argv\s*\)\s*==\s*expected\s*\)\s*;",
        )

    def test_cpu_isa_test_owns_ostream_for_string_view_diagnostics(self) -> None:
        source = (REPO / "tests/vt/test_cpu_isa_x86.cpp").read_text(
            encoding="utf-8"
        )
        directives = checker._cpp_directive_view(source)
        includes = list(
            re.finditer(r"(?m)^\s*#\s*include\s*<ostream>\s*$", directives)
        )
        self.assertEqual(len(includes), 1)
        self.assertLess(
            includes[0].start(),
            directives.index('#include "doctest/doctest.h"'),
        )
        self.assertIn("std::string_view::npos", source)

    def test_posix_cache_source_requires_exact_not_win32_cmake_guard(self) -> None:
        source = "src/vt/cuda/nvfp4_persistent_cache.cpp"
        for condition, expected in (("NOT WIN32", {source}), ("WIN32", set()), ("NOT APPLE", set())):
            root = self.make_tree({
                "CMakeLists.txt": textwrap.dedent(SAFE_FILES["CMakeLists.txt"]) +
                f"\nif({condition})\n  target_sources(vllm PRIVATE {source})\nendif()\n",
                source: "#include <unistd.h>\nvoid f() { close(1); }\n",
            })
            with self.subTest(condition=condition):
                self.assertEqual(checker.windows_excluded_sources(root), expected)

    def test_rejects_unguarded_posix_surface(self) -> None:
        self.assert_rejected(
            "src/vllm/entrypoints/openai/server_main.cpp",
            "#include <sys/wait.h>\nfork();\n",
            "unguarded POSIX",
        )

    def test_rejects_glibc_fopen_e_mode(self) -> None:
        self.assert_rejected(
            "src/vllm/v1/core/kv_cache_utils.cpp",
            'std::FILE* f = std::fopen("/proc/meminfo", "re");\n',
            "glibc-only fopen",
        )
        self.assert_rejected(
            "src/vllm/v1/core/kv_cache_utils.cpp",
            'freopen("/proc/meminfo", "we", stream);\n',
            "glibc-only fopen",
        )

    def test_accepts_standard_fopen_mode(self) -> None:
        result = self.run_checker(self.make_tree({
            "src/vllm/v1/core/kv_cache_utils.cpp":
                'std::FILE* f = std::fopen("/proc/meminfo", "r");\n',
        }))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_rejects_global_avx2_and_missing_static_runtime(self) -> None:
        self.assert_rejected(
            "CMakeLists.txt",
            "add_compile_options(/arch:AVX2)\n",
            "static MSVC runtime",
        )
        self.assert_rejected(
            "CMakeLists.txt",
            'set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")\n'
            "add_compile_options(/arch:AVX2)\n",
            "global /arch:AVX2",
        )

    def test_rejects_lossy_windows_paths(self) -> None:
        self.assert_rejected(
            "src/vllm/v1/kv_offload/fs_io.cpp",
            "#ifdef _WIN32\nCreateFileA(path.string().c_str(), 0, 0, nullptr, 0, 0, nullptr);\n#endif\n",
            "lossy Windows path",
        )

    def test_rejects_shell_process_launch(self) -> None:
        for source in (
            "#ifdef _WIN32\nsystem(command.c_str());\n#endif\n",
            '#ifdef _WIN32\n'
            'CreateProcessW(L"cmd.exe", nullptr, nullptr);\n'
            '#endif\n',
            '#ifdef _WIN32\n'
            'CreateProcessW(L"C:\\\\Windows\\\\System32\\\\cmd.exe", '
            'nullptr, nullptr);\n'
            '#endif\n',
        ):
            with self.subTest(source=source):
                self.assert_rejected(
                    "src/vllm/entrypoints/openai/server_main.cpp",
                    source,
                    "shell invocation",
                )

    def test_accepts_cpu_and_vulkan_command_buffer_identifiers(self) -> None:
        closures = {
            "cpu": "void RecordCpuCommandBuffer(int command_buffer) {}\n",
            "vulkan": """
                void RecordVulkanCommands(VkCommandBuffer cmd) {
                  // A cmd.exe shell is not launched by a command-buffer variable.
                  const char* note = "the cmd command-buffer variable is inert prose";
                  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
                }
            """,
        }
        for backend, source in closures.items():
            with self.subTest(backend=backend):
                result = self.run_checker(self.make_tree({
                    f"src/vt/{backend}/portability_audit_fixture.cpp": source,
                }))
                self.assertEqual(
                    result.returncode, 0, result.stdout + result.stderr
                )

    def test_powershell_ast_path_is_opaque_for_cpu_and_vulkan(self) -> None:
        ast_commands = [{
            "text": "\n".join((
                "Invoke-CrtAudit",
                'Invoke-Checked $server @("--help")',
                "Invoke-UnsupportedTierProbe -TierTest $tierTest",
                "Invoke-Checked python @($smokeHarness)",
                "Invoke-UnsupportedTierContractTests",
            )),
            "offset": 0,
        }]
        path_env = "VLLM_CPP_POWERSHELL_AUDIT_PATH"

        for backend, windows_root in (
            ("cpu", r"C:\actions\vllm.cpp CPU checkout"),
            ("vulkan", r"D:\a\vllm.cpp Vulkan checkout"),
        ):
            with self.subTest(backend=backend):
                root = Path(self.tempdir.name) / windows_root
                script = root / "scripts" / "build-windows-release.ps1"
                script.parent.mkdir(parents=True, exist_ok=True)
                script.write_text("", encoding="utf-8")
                calls: list[tuple[list[str], dict[str, object]]] = []

                def fake_run(
                    command: list[str], **kwargs: object
                ) -> subprocess.CompletedProcess[str]:
                    calls.append((command, kwargs))
                    command_index = (
                        command.index("-Command") if "-Command" in command
                        else -1
                    )
                    if command_index >= 0 and command_index != len(command) - 2:
                        return subprocess.CompletedProcess(
                            command,
                            1,
                            "",
                            "The file could not be read: Cannot process argument "
                            "because the value of argument path is not valid.",
                        )
                    if command_index >= 0:
                        return subprocess.CompletedProcess(
                            command, 0, json.dumps(ast_commands), ""
                        )
                    return subprocess.CompletedProcess(command, 0, "", "")

                errors: list[str] = []
                with mock.patch.object(
                    checker.shutil,
                    "which",
                    return_value=r"C:\Program Files\PowerShell\7\pwsh.exe",
                ), mock.patch.object(
                    checker.subprocess, "run", side_effect=fake_run
                ), mock.patch.object(
                    checker, "_validate_powershell_ast_order"
                ):
                    checker._validate_powershell_ast(script, errors)

                self.assertEqual(errors, [])
                self.assertEqual(len(calls), 2)
                ast_argv, ast_kwargs = calls[0]
                self.assertEqual(ast_argv.index("-Command"), len(ast_argv) - 2)
                self.assertNotIn("-Path", ast_argv)
                self.assertEqual(ast_kwargs["env"][path_env], str(script))
                self.assertFalse(ast_kwargs.get("shell", False))
                contract_argv, contract_kwargs = calls[1]
                self.assertEqual(
                    contract_argv[contract_argv.index("-File") + 1],
                    str(script),
                )
                self.assertEqual(
                    contract_argv[contract_argv.index("-SourceDir") + 1],
                    str(root),
                )
                self.assertFalse(contract_kwargs.get("shell", False))

    def test_powershell_ast_audit_is_scoped_to_real_roots(self) -> None:
        # CI: test_fetch_content_deps_sources_are_out_of_scan_scope ran the
        # checker on a synthetic root without the fixture manifest, so the
        # native AST audit parsed the synthetic stub release script and
        # errored "missing active configure in PowerShell AST" on every
        # pwsh lane. The stub exercises the source contract; only a real
        # run -- one without --test-source-manifest -- owns the AST and
        # fake-tool execution contract.
        root = self.make_tree()

        with mock.patch.object(checker.shutil, "which", return_value="pwsh"), \
                mock.patch.object(checker, "subprocess") as fake_subprocess, \
                mock.patch.object(checker, "_validate_powershell_ast") as audit:
            fake_subprocess.run.return_value = subprocess.CompletedProcess(
                [], 0, "", ""
            )
            checker.check(root, source_manifest=root / "m")

        self.assertFalse(audit.called)

        with mock.patch.object(checker.shutil, "which", return_value="pwsh"), \
                mock.patch.object(checker, "subprocess") as fake_subprocess, \
                mock.patch.object(checker, "_validate_powershell_ast") as audit:
            fake_subprocess.run.return_value = subprocess.CompletedProcess(
                [], 0, "", ""
            )
            checker.check(root)

        self.assertTrue(audit.called)
        self.assertEqual(
            audit.call_args[0][0],
            root / "scripts/build-windows-release.ps1",
        )

    def test_rejects_missing_winsock_support(self) -> None:
        self.assert_rejected(
            "src/vllm/v1/kv_offload/lmcache/remote_client.cpp",
            "#ifdef _WIN32\nthrow unsupported;\n#else\n#include <sys/socket.h>\n#endif\n",
            "LMCache Winsock support",
        )

    def test_rejects_missing_windows_kv_filesystem_support(self) -> None:
        self.assert_rejected(
            "src/vllm/v1/kv_offload/fs_io.cpp",
            "#ifdef _WIN32\nthrow unsupported;\n#else\n#include <unistd.h>\n#endif\n",
            "Windows KV filesystem support",
        )

    def test_rejects_function_target_attribute_in_portable_cpu_tu(self) -> None:
        self.assert_rejected(
            "src/vt/cpu/cpu_matmul_elem.cpp",
            '__attribute__((target("f16c"))) void leaked_baseline();\n',
            "F16C must be isolated",
        )

    def test_rejects_missing_compiler_specific_f16c_flags(self) -> None:
        self.assert_rejected(
            "CMakeLists.txt",
            textwrap.dedent(SAFE_FILES["CMakeLists.txt"]).replace(
                'set_source_files_properties(src/vt/cpu/cpu_matmul_elem_f16c.cpp\n'
                '  PROPERTIES COMPILE_OPTIONS\n'
                '    "$<$<CXX_COMPILER_ID:MSVC>:/arch:AVX>;$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-mf16c>")',
                "",
            ),
            "dedicated F16C translation unit",
        )

    def test_f16c_tu_accepts_avx_but_rejects_avx2(self) -> None:
        avx = textwrap.dedent(SAFE_FILES["CMakeLists.txt"])
        result = self.run_checker(self.make_tree({"CMakeLists.txt": avx}))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assert_rejected(
            "CMakeLists.txt",
            avx.replace("/arch:AVX>", "/arch:AVX2>"),
            "F16C translation unit must not require AVX2",
        )

    def test_rejects_dynamic_msvc_runtime(self) -> None:
        self.assert_rejected(
            "CMakeLists.txt",
            textwrap.dedent(SAFE_FILES["CMakeLists.txt"]).replace(
                '"MultiThreaded$<$<CONFIG:Debug>:Debug>"',
                '"MultiThreadedDLL"',
            ),
            "exact static MSVC runtime",
        )

    def test_scans_all_shipped_server_sources(self) -> None:
        self.assert_rejected(
            "src/vllm/model_executor/models/minimax_h3_sharded.cpp",
            "#include <sys/stat.h>\nstruct stat st; stat(path, &st);\n",
            "unguarded POSIX",
        )
        self.assert_rejected(
            "include/vllm/model_executor/models/device_pool.h",
            "auto n = __builtin_clzll(bytes);\n",
            "non-portable compiler builtin",
        )

    def test_pins_windows_file_publish_and_header_contracts(self) -> None:
        fs = textwrap.dedent(SAFE_FILES["src/vllm/v1/kv_offload/fs_io.cpp"])
        cmake = textwrap.dedent(SAFE_FILES["CMakeLists.txt"]).replace(
            "add_compile_definitions(NOMINMAX _CRT_SECURE_NO_WARNINGS)", ""
        )
        result = self.run_checker(self.make_tree({"CMakeLists.txt": cmake}))
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("NOMINMAX", result.stdout + result.stderr)
        self.assert_rejected(
            "src/vllm/v1/kv_offload/fs_io.cpp",
            fs.replace("CREATE_NEW", "CREATE_ALWAYS"),
            "CREATE_NEW",
        )
        self.assert_rejected(
            "src/vllm/v1/kv_offload/fs_io.cpp",
            fs.replace("MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH",
                       "MOVEFILE_REPLACE_EXISTING"),
            "MOVEFILE_WRITE_THROUGH",
        )

    def test_nominmax_contract_follows_central_compile_definition(self) -> None:
        cmake = textwrap.dedent(SAFE_FILES["CMakeLists.txt"])
        fs = textwrap.dedent(SAFE_FILES["src/vllm/v1/kv_offload/fs_io.cpp"])
        central = "add_compile_definitions(NOMINMAX _CRT_SECURE_NO_WARNINGS)"

        result = self.run_checker(self.make_tree())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

        without_central = cmake.replace(central, "")
        result = self.run_checker(self.make_tree({"CMakeLists.txt": without_central}))
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("NOMINMAX", result.stdout + result.stderr)

        guarded = fs.replace(
            "#include <windows.h>",
            "#ifndef NOMINMAX\n#define NOMINMAX\n#endif\n#include <windows.h>",
        )
        result = self.run_checker(self.make_tree({
            "CMakeLists.txt": without_central,
            "src/vllm/v1/kv_offload/fs_io.cpp": guarded,
        }))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

        late_fallback = fs.replace(
            "#include <windows.h>",
            "#include <windows.h>\n#ifndef NOMINMAX\n#define NOMINMAX\n#endif",
        )
        result = self.run_checker(self.make_tree({
            "CMakeLists.txt": without_central,
            "src/vllm/v1/kv_offload/fs_io.cpp": late_fallback,
        }))
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("before windows.h", result.stdout + result.stderr)

        unguarded = fs.replace(
            "#include <windows.h>", "#define NOMINMAX\n#include <windows.h>"
        )
        self.assert_rejected(
            "src/vllm/v1/kv_offload/fs_io.cpp",
            unguarded,
            "unguarded source-local NOMINMAX",
        )

    def test_nominmax_real_closure_accepts_guarded_fallbacks_before_headers(
            self) -> None:
        cmake = textwrap.dedent(SAFE_FILES["CMakeLists.txt"])
        without_central = cmake.replace(
            "add_compile_definitions(NOMINMAX _CRT_SECURE_NO_WARNINGS)", ""
        )
        safe_fs = textwrap.dedent(
            SAFE_FILES["src/vllm/v1/kv_offload/fs_io.cpp"]
        )
        real_sources = self.nominmax_real_sources()
        for relative, source in real_sources.items():
            with self.subTest(relative=relative):
                result = self.run_checker(self.make_tree({
                    "CMakeLists.txt": without_central,
                    "src/vllm/v1/kv_offload/fs_io.cpp":
                        self.with_nominmax_fallback(safe_fs, before=True),
                    relative: self.with_nominmax_fallback(
                        source, before=True
                    ),
                }))
                self.assertEqual(
                    result.returncode, 0, result.stdout + result.stderr
                )

    def test_nominmax_real_closure_rejects_late_and_unguarded_fallbacks(
            self) -> None:
        cmake = textwrap.dedent(SAFE_FILES["CMakeLists.txt"])
        without_central = cmake.replace(
            "add_compile_definitions(NOMINMAX _CRT_SECURE_NO_WARNINGS)", ""
        )
        real_sources = self.nominmax_real_sources()
        all_guarded = {
            relative: self.with_nominmax_fallback(source, before=True)
            for relative, source in real_sources.items()
        }
        for relative, source in real_sources.items():
            with self.subTest(relative=relative, shape="late"):
                mutations = dict(all_guarded)
                mutations[relative] = self.with_nominmax_fallback(
                    source, before=False
                )
                result = self.run_checker(self.make_tree({
                    "CMakeLists.txt": without_central,
                    **mutations,
                }))
                self.assertNotEqual(
                    result.returncode, 0, result.stdout + result.stderr
                )
                self.assertIn(
                    f"{relative}: NOMINMAX must be defined centrally",
                    result.stdout + result.stderr,
                )

            with self.subTest(relative=relative, shape="unguarded"):
                mutations = dict(real_sources)
                mutations[relative] = self.with_nominmax_fallback(
                    source, before=True, guarded=False
                )
                result = self.run_checker(self.make_tree(mutations))
                self.assertNotEqual(
                    result.returncode, 0, result.stdout + result.stderr
                )
                self.assertRegex(
                    result.stdout + result.stderr,
                    rf"{re.escape(relative)}:\d+: unguarded source-local "
                    r"NOMINMAX is forbidden",
                )

    def test_nominmax_real_closure_ignores_comment_and_literal_decoys(
            self) -> None:
        real_sources = self.nominmax_real_sources()
        target = "src/vllm/platform/process.cpp"
        source = real_sources[target]
        decoys = {
            "raw string": 'constexpr auto kRaw = R"TAG(\n#define NOMINMAX\n)TAG";\n',
            "ordinary string": 'constexpr auto kString = "\\\n#define NOMINMAX";\n',
            "line comment": "// #define NOMINMAX\n",
            "block comment": "/*\n#define NOMINMAX\n*/\n",
        }
        for shape, decoy in decoys.items():
            with self.subTest(shape=shape):
                mutations = dict(real_sources)
                mutations[target] = source.replace(
                    "#include <windows.h>",
                    decoy + "#include <windows.h>",
                )
                result = self.run_checker(self.make_tree(mutations))
                self.assertEqual(
                    result.returncode, 0, result.stdout + result.stderr
                )

    def test_nominmax_real_closure_without_contract_rejects_all_five_sources(
            self) -> None:
        cmake = textwrap.dedent(SAFE_FILES["CMakeLists.txt"])
        without_central = cmake.replace(
            "add_compile_definitions(NOMINMAX _CRT_SECURE_NO_WARNINGS)", ""
        )
        result = self.run_checker(self.make_tree({
            "CMakeLists.txt": without_central,
            **self.nominmax_real_sources(),
        }))
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        output = result.stdout + result.stderr
        for relative in NOMINMAX_REAL_CLOSURE:
            with self.subTest(relative=relative):
                self.assertIn(
                    f"{relative}: NOMINMAX must be defined centrally", output
                )

    def test_pins_peer_close_invalidation(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/v1/kv_offload/lmcache/remote_client.cpp"]
        )
        result = self.run_checker(self.make_tree({
            "src/vllm/v1/kv_offload/lmcache/remote_client.cpp": source,
        }))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assert_rejected(
            "src/vllm/v1/kv_offload/lmcache/remote_client.cpp",
            source.replace("Close(); throw peer_closed", "throw peer_closed"),
            "peer-close must invalidate",
        )

    def test_console_handler_uses_stable_event_and_drains_inflight(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        for mutation in (
            "registry.entrants.fetch_add(1, std::memory_order_seq_cst);",
            "registry.published.load(std::memory_order_seq_cst);",
            "state->in_flight.fetch_add(1, std::memory_order_seq_cst);",
            "SetEvent(state->stop_event);",
            "state->in_flight.fetch_sub(1, std::memory_order_seq_cst);",
            "registry.published.store(nullptr, std::memory_order_seq_cst);",
            "DrainEntrantsWithTimeout(registry.entrants);",
            "DrainInFlightWithTimeout(state->in_flight);",
            "RetireHandlerState(std::move(state));",
        ):
            self.assert_rejected(
                "src/vllm/platform/console_shutdown.cpp",
                source.replace(mutation, ""),
                "stable event/in-flight",
            )

    def test_console_function_body_skips_prototypes_and_decoys(self) -> None:
        source = (REPO / "src/vllm/platform/console_shutdown.cpp").read_text(
            encoding="utf-8"
        )
        declarations = {
            "bool DrainEntrantsWithTimeout(std::atomic<unsigned>& entrants) {":
                "bool DrainEntrantsWithTimeout(std::atomic<unsigned>& entrants);\n"
                "bool EntrantsDeclarationDecoy() { return false; }\n",
            "bool DrainInFlightWithTimeout(std::atomic<unsigned>& in_flight) {":
                "bool DrainInFlightWithTimeout(std::atomic<unsigned>& in_flight);\n"
                "bool InFlightDeclarationDecoy() { return false; }\n",
            "bool DispatchControlEvent(DWORD event, HANDLE acquired_event = nullptr,":
                "bool DispatchControlEvent(DWORD event, HANDLE acquired_event,\n"
                "                          HANDLE resume_event);\n"
                "bool DispatchDeclarationDecoy() { return false; }\n",
        }
        for definition, prefix in declarations.items():
            with self.subTest(definition=definition):
                self.assertEqual(source.count(definition), 1)
                mutated = source.replace(definition, prefix + definition, 1)
                result = self.run_checker(self.make_tree({
                    "src/vllm/platform/console_shutdown.cpp": mutated,
                }))
                self.assertEqual(result.returncode, 0,
                                 result.stdout + result.stderr)

    def test_console_final_tail_requires_bound_local_resumed(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        self.assert_rejected(
            "src/vllm/platform/console_shutdown.cpp",
            source.replace("return resumed;", "return true;", 1),
            "final in-flight decrement",
        )
        self.assert_rejected(
            "src/vllm/platform/console_shutdown.cpp",
            source.replace(
                "bool DispatchControlEvent(DWORD event) {",
                "bool resumed = true;\n"
                "bool DispatchControlEvent(DWORD event) {",
                1,
            ).replace("const bool resumed = true;\n", "", 1),
            "final in-flight decrement",
        )

    def test_console_final_tail_rejects_object_and_function_macros(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        decrement = "state->in_flight.fetch_sub(1, std::memory_order_seq_cst);"
        mutations = (
            (
                "#define POST_DECREMENT_RESULT "
                "(SetEvent(state->stop_event), resumed)\n  " + decrement,
                "return POST_DECREMENT_RESULT;",
                "final in-flight decrement",
            ),
            (
                "#define resumed (SetEvent(state->stop_event), true)\n  " +
                decrement,
                "return resumed;",
                "trusted final-tail token",
            ),
            (
                "#define POST_DECREMENT_RESULT() "
                "(SetEvent(state->stop_event), resumed)\n  " + decrement,
                "return POST_DECREMENT_RESULT();",
                "final in-flight decrement",
            ),
        )
        for macro_and_decrement, returned, reason in mutations:
            with self.subTest(returned=returned):
                mutated = source.replace(decrement, macro_and_decrement, 1)
                mutated = mutated.replace("return resumed;", returned, 1)
                self.assert_rejected(
                    "src/vllm/platform/console_shutdown.cpp",
                    mutated,
                    reason,
                )

    def test_console_final_tail_rejects_macros_for_trusted_tokens(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        decrement = "state->in_flight.fetch_sub(1, std::memory_order_seq_cst);"
        mutations = (
            "#define return SetEvent(state->stop_event); return\n  ",
            "#define return(value) SetEvent(state->stop_event); return value\n  ",
            "#define resumed \\\n  (SetEvent(state->stop_event), true)\n  ",
            "#define fetch_sub(value, order) \\\n  (SetEvent(state->stop_event), 0)\n  ",
        )
        for directive in mutations:
            with self.subTest(directive=directive):
                self.assert_rejected(
                    "src/vllm/platform/console_shutdown.cpp",
                    source.replace(decrement, directive + decrement, 1) +
                    "\n#undef return\n#undef resumed\n#undef fetch_sub\n",
                    "trusted final-tail token",
                )

    def test_console_final_tail_rejects_split_keyword_macro(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        decrement = "state->in_flight.fetch_sub(1, std::memory_order_seq_cst);"
        split_keyword = (
            "#define \\\n"
            "return SetEvent(state->stop_event); return\n  "
        )
        self.assert_rejected(
            "src/vllm/platform/console_shutdown.cpp",
            source.replace(decrement, split_keyword + decrement, 1),
            "trusted final-tail token",
        )

    def test_console_final_tail_macro_state_is_checked_at_each_token(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        decrement = "state->in_flight.fetch_sub(1, std::memory_order_seq_cst);"
        mutations = (
            "#define return SetEvent(state->stop_event); return",
            "#define return(value) SetEvent(state->stop_event); return value",
        )
        for directive in mutations:
            with self.subTest(directive=directive):
                mutated = source.replace(
                    decrement,
                    decrement + "\n  " + directive,
                    1,
                )
                self.assert_rejected(
                    "src/vllm/platform/console_shutdown.cpp",
                    mutated,
                    "trusted final-tail token",
                )

    def test_console_trusted_tails_restore_pushed_macros(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        timeout = "if (GetTickCount64() - start >= 5000) return false;"
        decrement = "state->in_flight.fetch_sub(1, std::memory_order_seq_cst);"
        cases = (
            (
                timeout,
                '#define false (SwitchToThread(), false)\n'
                '#pragma push_macro("false")\n'
                '#undef false\n'
                '#pragma pop_macro("false")\n' + timeout,
                "DrainEntrantsWithTimeout trusted return-tail token",
                0,
            ),
            (
                timeout,
                '#define GetTickCount64() (SwitchToThread(), GetTickCount64())\n'
                '#if UNKNOWN_WINDOWS_BUILD_FLAG\n'
                '#pragma push_macro( \\\n'
                '                    "GetTickCount64" )\n'
                '#undef GetTickCount64\n'
                '#pragma pop_macro("GetTickCount64")\n'
                '#endif\n' + timeout,
                "DrainInFlightWithTimeout trusted return-tail token",
                1,
            ),
            (
                decrement,
                '#define fetch_sub(value, order) '
                '(SetEvent(state->stop_event), 0)\n'
                '#pragma push_macro("fetch_sub")\n'
                '#undef fetch_sub\n'
                '#pragma pop_macro("fetch_sub")\n  ' + decrement,
                "trusted final-tail token",
                0,
            ),
            (
                decrement,
                decrement + '\n  '
                '#define return SetEvent(state->stop_event); return\n'
                '#pragma push_macro("return")\n'
                '#undef return\n'
                '#pragma pop_macro("return")',
                "trusted final-tail token",
                0,
            ),
        )
        for old, replacement, reason, occurrence in cases:
            with self.subTest(reason=reason, occurrence=occurrence):
                position = -1
                for _ in range(occurrence + 1):
                    position = source.find(old, position + 1)
                self.assertGreaterEqual(position, 0)
                mutated = source[:position] + replacement + source[position + len(old):]
                self.assert_rejected(
                    "src/vllm/platform/console_shutdown.cpp", mutated, reason
                )

    def test_console_trusted_tails_accept_push_pop_restoring_undefined(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        timeout = "if (GetTickCount64() - start >= 5000) return false;"
        safe = (
            '#pragma push_macro("false")\n'
            '#define false unsafe_false\n'
            '#pragma pop_macro("false")\n' + timeout
        )
        for occurrence in (0, 1):
            with self.subTest(occurrence=occurrence):
                position = -1
                for _ in range(occurrence + 1):
                    position = source.find(timeout, position + 1)
                self.assertGreaterEqual(position, 0)
                mutated = source[:position] + safe + source[position + len(timeout):]
                result = self.run_checker(self.make_tree({
                    "src/vllm/platform/console_shutdown.cpp": mutated,
                }))
                self.assertEqual(result.returncode, 0,
                                 result.stdout + result.stderr)

    def test_console_trusted_tails_honor_phase_two_comment_splicing(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        timeout = "if (GetTickCount64() - start >= 5000) return false;"
        decrement = "state->in_flight.fetch_sub(1, std::memory_order_seq_cst);"
        cases = (
            (
                timeout,
                "#define false (SwitchToThread(), false)\n"
                "// the physical splice swallows the undef \\\n"
                "#undef false\n" + timeout,
                "DrainEntrantsWithTimeout trusted return-tail token",
                0,
            ),
            (
                timeout,
                "#define return SwitchToThread(); return\n"
                "// the physical splice swallows the undef \\\n"
                "#undef return\n" + timeout,
                "DrainInFlightWithTimeout trusted return-tail token",
                1,
            ),
            (
                decrement,
                "#define return SetEvent(state->stop_event); return\n"
                "// the physical splice swallows the undef \\\n"
                "#undef return\n  " + decrement,
                "trusted final-tail token",
                0,
            ),
        )
        for old, replacement, reason, occurrence in cases:
            with self.subTest(reason=reason, occurrence=occurrence):
                position = -1
                for _ in range(occurrence + 1):
                    position = source.find(old, position + 1)
                self.assertGreaterEqual(position, 0)
                mutated = source[:position] + replacement + source[position + len(old):]
                self.assert_rejected(
                    "src/vllm/platform/console_shutdown.cpp", mutated, reason
                )

    def test_console_final_tail_accepts_only_macros_active_at_the_tail(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        decrement = "state->in_flight.fetch_sub(1, std::memory_order_seq_cst);"
        safe_directives = (
            "#define return unsafe_return\n#undef return\n  ",
            "#if 0\n#define return unsafe_return\n#endif\n  ",
            "#if 0\n#define return unsafe_return\n"
            "#else\n#define harmless_tail_macro 1\n#endif\n  ",
        )
        for directives in safe_directives:
            with self.subTest(directives=directives):
                result = self.run_checker(self.make_tree({
                    "src/vllm/platform/console_shutdown.cpp":
                        source.replace(decrement, directives + decrement, 1),
                }))
                self.assertEqual(
                    result.returncode, 0, result.stdout + result.stderr
                )

    def test_console_final_tail_accepts_resumed_macros_inactive_at_return(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        decrement = "state->in_flight.fetch_sub(1, std::memory_order_seq_cst);"
        safe_directives = (
            "#define resumed unsafe_resumed\n#undef resumed\n  ",
            "#if 0\n#define resumed unsafe_resumed\n#endif\n  ",
            '#pragma push_macro("resumed")\n'
            "#define resumed unsafe_resumed\n"
            '#pragma pop_macro("resumed")\n  ',
        )
        for directives in safe_directives:
            with self.subTest(directives=directives):
                result = self.run_checker(self.make_tree({
                    "src/vllm/platform/console_shutdown.cpp":
                        source.replace(decrement, directives + decrement, 1),
                }))
                self.assertEqual(
                    result.returncode, 0, result.stdout + result.stderr
                )

        self.assert_rejected(
            "src/vllm/platform/console_shutdown.cpp",
            source.replace(
                decrement,
                "#define resumed unsafe_resumed\n  " + decrement,
                1,
            ),
            "trusted final-tail token",
        )

        result = self.run_checker(self.make_tree({
            "src/vllm/platform/console_shutdown.cpp":
                source + "\n#define return unsafe_after_dispatch\n",
        }))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_console_final_tail_ignores_comment_and_string_decoys(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        decrement = "state->in_flight.fetch_sub(1, std::memory_order_seq_cst);"
        decoys = (
            'const char* macro_decoy = "#define resumed false";\n'
            "  (void)macro_decoy;\n"
            "  // #define POST_DECREMENT_RESULT false\n  "
        )
        result = self.run_checker(self.make_tree({
            "src/vllm/platform/console_shutdown.cpp":
                source.replace(decrement, decoys + decrement, 1),
        }))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_console_drain_tails_reject_active_keyword_macros_independently(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        boundaries = {
            "DrainEntrantsWithTimeout": "bool DrainInFlightWithTimeout",
            "DrainInFlightWithTimeout": "std::atomic<unsigned> in_flight",
        }
        for function, boundary in boundaries.items():
            with self.subTest(function=function):
                definition = f"bool {function}"
                mutated = source.replace(
                    definition,
                    "#define return SwitchToThread(); return\n" + definition,
                    1,
                ).replace(boundary, "#undef return\n" + boundary, 1)
                self.assert_rejected(
                    "src/vllm/platform/console_shutdown.cpp",
                    mutated,
                    f"{function} trusted return-tail token",
                )

    def test_console_drain_tails_reject_one_spliced_macro_across_both(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        mutated = source.replace(
            "bool DrainEntrantsWithTimeout",
            "#define \\\nreturn SwitchToThread(); return\n"
            "bool DrainEntrantsWithTimeout",
            1,
        ).replace(
            "std::atomic<unsigned> in_flight",
            "#undef return\nstd::atomic<unsigned> in_flight",
            1,
        )
        for function in ("DrainEntrantsWithTimeout", "DrainInFlightWithTimeout"):
            with self.subTest(function=function):
                self.assert_rejected(
                    "src/vllm/platform/console_shutdown.cpp",
                    mutated,
                    f"{function} trusted return-tail token",
                )

    def test_console_drain_tails_reject_aliases_of_every_trusted_token(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        boundaries = {
            "DrainEntrantsWithTimeout": "bool DrainInFlightWithTimeout",
            "DrainInFlightWithTimeout": "std::atomic<unsigned> in_flight",
        }
        directives = {
            "if": "#define if(condition) if ((SwitchToThread(), condition))",
            "GetTickCount64": "#define GetTickCount64() (SwitchToThread(), GetTickCount64())",
            "start": "#define start (SwitchToThread(), start)",
            "return": "#define return SwitchToThread(); return",
            "false": "#define false (SwitchToThread(), false)",
            "true": "#define true (SwitchToThread(), true)",
        }
        for function, boundary in boundaries.items():
            for token, directive in directives.items():
                with self.subTest(function=function, token=token):
                    definition = f"bool {function}"
                    mutated = source.replace(
                        definition, f"{directive}\n{definition}", 1
                    ).replace(boundary, f"#undef {token}\n{boundary}", 1)
                    self.assert_rejected(
                        "src/vllm/platform/console_shutdown.cpp",
                        mutated,
                        f"{function} trusted return-tail token",
                    )

    def test_console_drain_tail_macro_state_is_bound_to_each_return(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        safe_mutations = (
            source.replace(
                "bool DrainEntrantsWithTimeout",
                "#define return unsafe_return\n#undef return\n"
                "bool DrainEntrantsWithTimeout",
                1,
            ),
            source.replace(
                "bool DrainInFlightWithTimeout",
                "#if 0\n#define true unsafe_true\n#endif\n"
                "bool DrainInFlightWithTimeout",
                1,
            ),
            source.replace(
                "std::atomic<unsigned> in_flight",
                "#define false unsafe_after_drain\n#undef false\n"
                "std::atomic<unsigned> in_flight",
                1,
            ),
        )
        for index, mutated in enumerate(safe_mutations):
            with self.subTest(index=index):
                result = self.run_checker(self.make_tree({
                    "src/vllm/platform/console_shutdown.cpp": mutated,
                }))
                self.assertEqual(
                    result.returncode, 0, result.stdout + result.stderr
                )

    def test_console_drain_structural_view_blanks_safe_directives(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        safe_replacements = (
            (
                "if (GetTickCount64() - start >= 5000) return false;",
                "#define false unsafe_false\n"
                "#undef false\n"
                "if (GetTickCount64() - start >= 5000) return false;",
            ),
            (
                "if (GetTickCount64() - start >= 5000) return false;",
                "#define \\\n"
                "false unsafe_false\n"
                "#undef false\n"
                "if (GetTickCount64() - start >= 5000) return false;",
            ),
            (
                "return true;",
                "#define true unsafe_true\n"
                "#undef true\n"
                "return true;",
            ),
        )
        for old, new in safe_replacements:
            for occurrence in (0, 1):
                with self.subTest(old=old, occurrence=occurrence):
                    position = -1
                    for _ in range(occurrence + 1):
                        position = source.find(old, position + 1)
                    self.assertGreaterEqual(position, 0)
                    mutated = source[:position] + new + source[position + len(old):]
                    result = self.run_checker(self.make_tree({
                        "src/vllm/platform/console_shutdown.cpp": mutated,
                    }))
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )

    def test_console_drain_structural_view_keeps_macro_state_fail_closed(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        timeout = "if (GetTickCount64() - start >= 5000) return false;"
        uncertain = (
            "#if UNKNOWN_WINDOWS_BUILD_FLAG\n"
            "#define false (SwitchToThread(), false)\n"
            "#endif\n" + timeout
        )
        self.assert_rejected(
            "src/vllm/platform/console_shutdown.cpp",
            source.replace(timeout, uncertain, 1),
            "trusted return-tail token",
        )
        spliced = (
            "#define \\\n"
            "false (SwitchToThread(), false)\n" + timeout
        )
        self.assert_rejected(
            "src/vllm/platform/console_shutdown.cpp",
            source.replace(timeout, spliced, 1),
            "trusted return-tail token",
        )

    def test_console_function_body_skips_constrained_template_prototypes(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        prototypes = {
            "DispatchControlEvent": textwrap.dedent("""
                template <typename T>
                bool DispatchControlEvent(T item)
                    requires requires(T value) { value(); };
                template <typename T>
                bool DispatchControlEvent(T item) noexcept
                    requires (sizeof(T) > 0);
                template <typename T>
                bool DispatchControlEvent(T item) [[deprecated]];
            """),
            "DrainEntrantsWithTimeout": textwrap.dedent("""
                template <typename T>
                bool DrainEntrantsWithTimeout(T item)
                    requires requires(T value) { value(); };
                template <typename T>
                bool DrainEntrantsWithTimeout(T item) noexcept
                    requires (sizeof(T) > 0);
                template <typename T>
                bool DrainEntrantsWithTimeout(T item) [[deprecated]];
            """),
            "DrainInFlightWithTimeout": textwrap.dedent("""
                template <typename T>
                bool DrainInFlightWithTimeout(T item)
                    requires requires(T value) { value(); };
                template <typename T>
                bool DrainInFlightWithTimeout(T item) noexcept
                    requires (sizeof(T) > 0);
                template <typename T>
                bool DrainInFlightWithTimeout(T item) [[deprecated]];
            """),
        }
        all_prototypes = "\n".join(prototypes.values())
        valid = source.replace(
            "bool DrainEntrantsWithTimeout", all_prototypes +
            "\nbool DrainEntrantsWithTimeout", 1
        )
        result = self.run_checker(self.make_tree({
            "src/vllm/platform/console_shutdown.cpp": valid,
        }))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

        decrement = "state->in_flight.fetch_sub(1, std::memory_order_seq_cst);"
        self.assert_rejected(
            "src/vllm/platform/console_shutdown.cpp",
            valid.replace(decrement, "", 1),
            "stable event/in-flight",
        )
        timeout = "if (GetTickCount64() - start >= 5000) return false;"
        self.assertEqual(valid.count(timeout), 2)
        for occurrence in (0, 1):
            position = -1
            for _ in range(occurrence + 1):
                position = valid.find(timeout, position + 1)
            mutated = valid[:position] + valid[position + len(timeout):]
            self.assert_rejected(
                "src/vllm/platform/console_shutdown.cpp",
                mutated,
                "finite timeout",
            )

    def test_unsigned_elapsed_subtraction_survives_tick_wrap(self) -> None:
        mask = (1 << 64) - 1
        start = mask - 2
        self.assertEqual((1 - start) & mask, 4)
        self.assertLess((1 - start) & mask, 5)
        self.assertEqual((2 - start) & mask, 5)

    def test_console_partial_event_creation_cleanup_is_structural(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        for mutation in (
            "CloseHandle(quit_event);",
            "CloseHandle(stop_event);",
        ):
            self.assert_rejected(
                "src/vllm/platform/console_shutdown.cpp",
                source.replace(mutation, ""),
                "partial event creation cleanup",
            )

    def test_console_handler_rejects_object_access_callback_and_blocking(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        for statement in (
            "impl_->RequestStop();",
            "stop_();",
            "std::mutex lock;",
            "std::condition_variable ready;",
        ):
            self.assert_rejected(
                "src/vllm/platform/console_shutdown.cpp",
                source.replace("SetEvent(state->stop_event);",
                               f"SetEvent(state->stop_event); {statement}"),
                "OS handler may use only stable atomics and Win32 events",
            )

    def test_console_teardown_order_is_structural(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        self.assert_rejected(
            "src/vllm/platform/console_shutdown.cpp",
            source.replace(
                "registry.published.store(nullptr, std::memory_order_seq_cst);\n"
                "DrainEntrantsWithTimeout(registry.entrants);",
                "DrainEntrantsWithTimeout(registry.entrants);\n"
                "registry.published.store(nullptr, std::memory_order_seq_cst);",
            ),
            "stable event/in-flight",
        )
        self.assert_rejected(
            "src/vllm/platform/console_shutdown.cpp",
            source.replace(
                "DrainInFlightWithTimeout(state->in_flight);\n"
                "if (!safe_to_close) RetireHandlerState(std::move(state));\n"
                "else state.reset();",
                "else state.reset();\n"
                "if (!safe_to_close) RetireHandlerState(std::move(state));\n"
                "DrainInFlightWithTimeout(state->in_flight);",
            ),
            "stable event/in-flight",
        )

    def test_build_script_requires_live_gate_order(self) -> None:
        script = textwrap.dedent(SAFE_FILES["scripts/build-windows-release.ps1"])
        self.assert_rejected(
            "scripts/build-windows-release.ps1",
            script.replace(
                'Invoke-CrtAudit\nInvoke-Checked $server @("--help")',
                'Invoke-Checked $server @("--help")\nInvoke-CrtAudit',
            ),
            "native gate order",
        )
        self.assert_rejected(
            "scripts/build-windows-release.ps1",
            script.replace(
                'Invoke-Checked $server @("--help")',
                'if ($false) { Invoke-Checked $server @("--help") }',
            ),
            "live --help smoke",
        )

    def test_build_script_requires_isolated_unsupported_tier_probe(self) -> None:
        script = textwrap.dedent(SAFE_FILES["scripts/build-windows-release.ps1"])
        result = self.run_checker(self.make_tree({
            "scripts/build-windows-release.ps1": script,
        }))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for old, new, reason in (
            (
                UNSUPPORTED_TIER_FILTER,
                "",
                "isolated unsupported-tier filter",
            ),
            ("$probeExitCode -ne 1", "$probeExitCode -eq 0",
             "exact unsupported-tier exit status"),
            ("unknown x86 ISA tier 'amx'", "unrelated diagnostic",
             "unsupported-tier diagnostic"),
            ("ExitCode = 134", "ExitCode = 1",
             "unsupported-tier crash contract"),
            ("ExitCode = -1073741819", "ExitCode = 1",
             "unsupported-tier crash contract"),
            ("ExitCode = 3", "ExitCode = 1",
             "unsupported-tier crash contract"),
            ("ExitCode = 2", "ExitCode = 1",
             "unsupported-tier crash contract"),
            ("@(& $TierTest @arguments 2>&1)",
             "@(& $TierTest @arguments)",
             "merged unsupported-tier stdout/stderr capture"),
            ("$env:VT_CPU_MATMUL_TIER = $savedTier", "",
             "unsupported-tier environment restoration"),
            ("  Invoke-UnsupportedTierContractTests\n", "",
             "unsupported-tier fake-tool contract"),
        ):
            with self.subTest(old=old):
                self.assertIn(old, script)
                self.assert_rejected(
                    "scripts/build-windows-release.ps1",
                    script.replace(old, new, 1),
                    reason,
                )
        self.assert_rejected(
            "scripts/build-windows-release.ps1",
            script.replace(
                "Invoke-UnsupportedTierProbe -TierTest $tierTest",
                "& $tierTest",
                1,
            ),
            "AMX refusal must use only the isolated",
        )
        self.assert_rejected(
            "scripts/build-windows-release.ps1",
            script.replace(
                "Invoke-UnsupportedTierProbe -TierTest $tierTest",
                "Invoke-Checked $tierTest @()\n"
                "  Invoke-UnsupportedTierProbe -TierTest $tierTest",
                1,
            ),
            "AMX refusal must use only the isolated",
        )
        quoted_filter = f"'{UNSUPPORTED_TIER_FILTER}'"
        self.assert_rejected(
            "scripts/build-windows-release.ps1",
            script.replace(
                quoted_filter,
                f"{quoted_filter},\n    {quoted_filter}",
                1,
            ),
            "one exact unsupported-tier filter argument",
        )
        self.assert_rejected(
            "scripts/build-windows-release.ps1",
            f"$filterDecoy = {quoted_filter}\n" + script.replace(
                quoted_filter, "'--test-case=unrelated test'", 1
            ),
            "isolated unsupported-tier filter",
        )

    def test_real_unsupported_tier_helper_is_structurally_scoped(self) -> None:
        script = (REPO / "scripts/build-windows-release.ps1").read_text(
            encoding="utf-8"
        )
        capture = "$probeOutput = @(& $TierTest @arguments 2>&1)"
        self.assertEqual(script.count(capture), 1)
        mutations = [(
            script.replace(
                capture, "& $TierTest\n        " + capture, 1
            ),
            "exactly one filtered process invocation",
        )]

        runner_call = "$probeResult = & $Runner $TierTest $arguments"
        self.assertEqual(script.count(runner_call), 1)
        mutations.extend((
            (
                script.replace(
                    runner_call,
                    "$null = & $Runner $TierTest @()\n        " + runner_call,
                    1,
                ),
                "exactly one fake-runner invocation",
            ),
            (
                script.replace(
                    runner_call,
                    "$probeResult = & $Runner $TierTest @()",
                    1,
                ),
                "exact unsupported-tier filter arguments",
            ),
        ))

        helper_start = script.index("function Invoke-UnsupportedTierProbe")
        helper_end = script.index(
            "function Invoke-UnsupportedTierContractTests", helper_start
        )
        helper = script[helper_start:helper_end]
        diagnostic = "unknown x86 ISA tier 'amx'"
        self.assertEqual(helper.count(diagnostic), 1)
        mutated_helper = helper.replace(diagnostic, "wrong helper diagnostic", 1)
        mutated = script[:helper_start] + mutated_helper + script[helper_end:]
        self.assertIn(diagnostic, mutated)
        mutations.append((mutated, "unsupported-tier diagnostic"))
        canonical_diagnostic = (
            '    $diagnostic = $probeOutput -join "`n"\n'
            '    if ($diagnostic -notmatch '
            '[regex]::Escape("unknown x86 ISA tier \'amx\'")) {'
        )
        fixture_only_diagnostic = (
            '    if (($probeOutput -join "`n") -notmatch '
            '[regex]::Escape("unknown x86 ISA tier \'amx\'")) {'
        )
        self.assertIn(canonical_diagnostic, helper)
        mutations.append((
            script.replace(
                canonical_diagnostic, fixture_only_diagnostic, 1
            ),
            "exact unsupported-tier probe body",
        ))
        # The checker reads `$calls.Add(` inside the `$good` scriptblock of
        # Invoke-UnsupportedTierContractTests (`good_runner`), and NOWHERE
        # else. `script.replace(..., 1)` mutates the first occurrence in the
        # file, which since #583 added Invoke-CheckedContractTests (#512) is a
        # DIFFERENT function the checker never looks at -- so the checker
        # stayed green and this case has been failing on main ever since
        # (#680). Anchor the mutation to the governed occurrence, and assert
        # that occurrence is unique rather than assuming it.
        governed_at = script.index("function Invoke-UnsupportedTierContractTests")
        self.assertEqual(script[governed_at:].count("$calls.Add("), 1)
        mutations.extend((
            (
                script[:governed_at] + script[governed_at:].replace(
                    "$calls.Add(", "$calls.Append(", 1
                ),
                "fake runner call recording",
            ),
            (
                script.replace("$calls.Count -ne 1", "$calls.Count -lt 0", 1),
                "exactly one recorded fake call",
            ),
            (
                script.replace(
                    "$calls[0].Arguments.Count -ne 1",
                    "$calls[0].Arguments.Count -lt 0",
                    1,
                ),
                "exactly one recorded fake argument",
            ),
            (
                script.replace(
                    'Invoke-UnsupportedTierProbe -TierTest "fake-tier-test.exe" '
                    "-Runner $good",
                    'Invoke-UnsupportedTierProbe -TierTest "fake-tier-test.exe" '
                    "-Runner $good\n    "
                    'Invoke-UnsupportedTierProbe -TierTest "fake-tier-test.exe" '
                    "-Runner $good",
                    1,
                ),
                "exactly one unsupported-tier probe invocation",
            ),
        ))
        contract_start = script.index("function Invoke-UnsupportedTierContractTests")
        contract = script[contract_start:]
        self.assertIn(UNSUPPORTED_TIER_FILTER, contract)
        mutated_contract = contract.replace(
            UNSUPPORTED_TIER_FILTER, "--test-case=wrong fake filter", 1
        )
        mutations.append((
            script[:contract_start] + mutated_contract,
            "exact recorded fake filter argument",
        ))
        for mutation, reason in mutations:
            with self.subTest(reason=reason):
                self.assert_rejected(
                    "scripts/build-windows-release.ps1", mutation, reason
                )

    def test_unsupported_tier_helper_requires_mandatory_tier_parameter(self) -> None:
        canonical = "[Parameter(Mandatory)][string]$TierTest"
        optional = "[string]$TierTest"
        fixture = textwrap.dedent(SAFE_FILES["scripts/build-windows-release.ps1"])
        real = (REPO / "scripts/build-windows-release.ps1").read_text(
            encoding="utf-8"
        )
        for label, script in (("fixture", fixture), ("real", real)):
            with self.subTest(label=label):
                self.assertEqual(script.count(canonical), 1)
                self.assert_rejected(
                    "scripts/build-windows-release.ps1",
                    script.replace(canonical, optional, 1),
                    "exact unsupported-tier probe body",
                )

    def test_unsupported_tier_rejects_equivalent_unfiltered_target_forms(self) -> None:
        script = textwrap.dedent(SAFE_FILES["scripts/build-windows-release.ps1"])
        capture = "$probeOutput = @(& $TierTest @arguments 2>&1)"
        equivalent_targets = (
            "${TierTest}",
            "${TiErTeSt}",
            "($TierTest)",
            "$($TierTest)",
            "$local:TierTest",
            "${local:TierTest}",
            "$script:TiErTeSt",
            "${private:TierTest}",
        )
        for target in equivalent_targets:
            with self.subTest(target=target):
                mutated = script.replace(
                    capture, f"$null = & {target}\n            {capture}", 1
                )
                self.assert_rejected(
                    "scripts/build-windows-release.ps1",
                    mutated,
                    "exactly one filtered process invocation",
                )

    def test_unsupported_tier_helper_rejects_every_extra_executable_form(self) -> None:
        script = textwrap.dedent(SAFE_FILES["scripts/build-windows-release.ps1"])
        capture = "$probeOutput = @(& $TierTest @arguments 2>&1)"
        before_forms = (
            "$alias = $TierTest\n            & $alias",
            "$env:TierAlias = $TierTest\n"
            "            Invoke-Expression $env:TierAlias",
            "$alias = Get-Variable TierTest -ValueOnly\n            . $alias",
            "Start-Process -FilePath $TierTest -Wait",
            "[System.Diagnostics.Process]::Start($TierTest)",
        )
        for executable in before_forms:
            for placement in ("before", "after"):
                with self.subTest(executable=executable, placement=placement):
                    replacement = (
                        executable + "\n            " + capture
                        if placement == "before" else
                        capture + "\n            " + executable
                    )
                    self.assert_rejected(
                        "scripts/build-windows-release.ps1",
                        script.replace(capture, replacement, 1),
                        "exact unsupported-tier probe body",
                    )

    def test_unsupported_tier_amx_scope_rejects_indirect_execution(self) -> None:
        script = textwrap.dedent(SAFE_FILES["scripts/build-windows-release.ps1"])
        assignment = '$env:VT_CPU_MATMUL_TIER = "amx"'
        probe = "Invoke-UnsupportedTierProbe -TierTest $tierTest"
        indirect_forms = (
            "$alias = $tierTest; & $alias",
            "$env:TierAlias = $tierTest; Invoke-Expression $env:TierAlias",
            "$alias = Get-Variable tierTest -ValueOnly; . $alias",
            "Start-Process -FilePath $tierTest -Wait",
            "[System.Diagnostics.Process]::Start($tierTest)",
        )
        for executable in indirect_forms:
            for placement in ("before", "after"):
                with self.subTest(executable=executable, placement=placement):
                    mutation = (
                        script.replace(
                            assignment,
                            assignment + "\n          " + executable,
                            1,
                        ) if placement == "before" else
                        script.replace(
                            probe,
                            probe + "\n          " + executable,
                            1,
                        )
                    )
                    self.assert_rejected(
                        "scripts/build-windows-release.ps1",
                        mutation,
                        "exact AMX refusal block",
                    )

    def test_unsupported_tier_structural_view_ignores_literal_and_comment_decoys(self) -> None:
        script = textwrap.dedent(SAFE_FILES["scripts/build-windows-release.ps1"])
        capture = "$probeOutput = @(& $TierTest @arguments 2>&1)"
        decoy = "@(& $TierTest @arguments 2>&1)"
        accepted = (
            script.replace(
                'throw "unexpected exit"',
                f"throw 'diagnostic text {decoy} {UNSUPPORTED_TIER_FILTER} "
                "unknown x86 ISA tier ''amx'''",
                1,
            ),
            script.replace(
                capture,
                capture + f" # inline decoy {decoy}",
                1,
            ),
            script.replace(
                capture,
                f"# full-line decoy {decoy}\n            " + capture,
                1,
            ),
        )
        for mutation in accepted:
            result = self.run_checker(self.make_tree({
                "scripts/build-windows-release.ps1": mutation,
            }))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_unsupported_tier_filter_is_one_exact_process_argument(self) -> None:
        script = textwrap.dedent(SAFE_FILES["scripts/build-windows-release.ps1"])
        arguments = re.findall(
            r"(?m)^\s*['\"](--test-case=.*)['\"]\s*$", script
        )
        self.assertEqual(arguments, [UNSUPPORTED_TIER_FILTER])

    def test_unsupported_tier_exact_literals_reject_runtime_backtick_escapes(self) -> None:
        fixture = textwrap.dedent(SAFE_FILES["scripts/build-windows-release.ps1"])
        real = (REPO / "scripts/build-windows-release.ps1").read_text(
            encoding="utf-8"
        )
        for label, script in (("fixture", fixture), ("real", real)):
            with self.subTest(label=label, literal="filter"):
                quoted_filter = f"'{UNSUPPORTED_TIER_FILTER}'"
                self.assertIn(quoted_filter, script)
                escaped_filter = '"--`test-case=elementwise CPU GEMM: the forced tier is the tier that actually ran"'
                self.assert_rejected(
                    "scripts/build-windows-release.ps1",
                    script.replace(quoted_filter, escaped_filter, 1),
                    "exact unsupported-tier probe body",
                )
            with self.subTest(label=label, literal="diagnostic"):
                helper_start = script.index("function Invoke-UnsupportedTierProbe")
                helper_end = script.index(
                    "function Invoke-UnsupportedTierContractTests", helper_start
                )
                helper = script[helper_start:helper_end]
                diagnostic = '"unknown x86 ISA tier \'amx\'"'
                self.assertIn(diagnostic, helper)
                mutated_helper = helper.replace(
                    diagnostic, '"u`nknown x86 ISA tier \'amx\'"', 1
                )
                mutated = script[:helper_start] + mutated_helper + script[helper_end:]
                self.assert_rejected(
                    "scripts/build-windows-release.ps1",
                    mutated,
                    "exact unsupported-tier probe body",
                )

    def test_unsupported_tier_exact_filter_accepts_plain_double_quotes(self) -> None:
        script = textwrap.dedent(SAFE_FILES["scripts/build-windows-release.ps1"])
        quoted_filter = f"'{UNSUPPORTED_TIER_FILTER}'"
        result = self.run_checker(self.make_tree({
            "scripts/build-windows-release.ps1": script.replace(
                quoted_filter, f'"{UNSUPPORTED_TIER_FILTER}"', 1
            ),
        }))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_real_codemodel_requires_reachable_sources_and_internal_headers(self) -> None:
        root = self.make_tree()
        process_header = root / "src/vllm/platform/process.h"
        process_header.parent.mkdir(parents=True, exist_ok=True)
        process_header.write_text("#include <unistd.h>\n", encoding="utf-8")
        cmake_contract = textwrap.dedent(SAFE_FILES["CMakeLists.txt"])
        (root / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.20)\n"
            "project(portability LANGUAGES CXX)\n" + cmake_contract +
            "\nadd_library(runtime STATIC\n"
            "  src/vllm/platform/process.cpp\n"
            "  src/vllm/platform/console_shutdown.cpp\n"
            "  src/vllm/v1/kv_offload/lmcache/remote_client.cpp\n"
            "  src/vllm/v1/kv_offload/fs_io.cpp)\n"
            "target_include_directories(runtime PRIVATE src include)\n"
            "add_executable(server src/vllm/entrypoints/openai/server_main.cpp)\n"
            "target_link_libraries(server PRIVATE runtime)\n",
            encoding="utf-8",
        )
        build = root / "build"
        query = build / ".cmake/api/v1/query"
        query.mkdir(parents=True)
        (query / "codemodel-v2").touch()
        configured = subprocess.run(
            ["cmake", "-S", str(root), "-B", str(build), "-G", "Ninja"],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(configured.returncode, 0,
                         configured.stdout + configured.stderr)
        result = subprocess.run(
            [sys.executable, str(CHECKER), "--root", str(root),
             "--build-dir", str(build)],
            text=True, capture_output=True, check=False,
        )
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("src/vllm/platform/process.h", result.stdout + result.stderr)

        cmake = (root / "CMakeLists.txt").read_text(encoding="utf-8")
        (root / "CMakeLists.txt").write_text(
            cmake.replace("  src/vllm/platform/process.cpp\n", ""),
            encoding="utf-8",
        )
        subprocess.run(
            ["cmake", "-S", str(root), "-B", str(build), "-G", "Ninja"],
            text=True, capture_output=True, check=True,
        )
        result = subprocess.run(
            [sys.executable, str(CHECKER), "--root", str(root),
             "--build-dir", str(build)],
            text=True, capture_output=True, check=False,
        )
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("required implementation is not reachable",
                      result.stdout + result.stderr)

    def test_fetch_content_deps_sources_are_out_of_scan_scope(self) -> None:
        # CI run 37034253522: the Windows release build configures into
        # build-pr-windows-vulkan/ INSIDE the source tree, so the codemodel
        # carries the vendored ggml of the FetchContent parakeet_cpp
        # dependency (build-pr-windows-vulkan/_deps/parakeet_cpp-src/...),
        # and the POSIX rule flagged amx.cpp:12, mmq.cpp:18, ggml-cpu.c:2542
        # and ggml.c:85-88. That code is third-party; the checker's contract
        # is THIS project's sources reaching Windows-incompatible POSIX.
        # The Linux self-configure uses a build dir OUTSIDE the root, so the
        # same sources silently dropped out there and hid the asymmetry.
        root = self.make_tree()
        vendored = (root /
                    "build-pr-windows-vulkan/_deps/parakeet_cpp-src/"
                    "third_party/ggml/src/ggml-cpu/amx/amx.cpp")
        vendored.parent.mkdir(parents=True, exist_ok=True)
        vendored.write_text(
            "#include <unistd.h>\nvoid leaked_posix() { fork(); }\n",
            encoding="utf-8",
        )
        cmake_contract = textwrap.dedent(SAFE_FILES["CMakeLists.txt"])
        (root / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.20)\n"
            "project(portability LANGUAGES CXX)\n" + cmake_contract +
            "\nadd_library(runtime STATIC\n"
            "  src/vllm/platform/process.cpp\n"
            "  src/vllm/platform/console_shutdown.cpp\n"
            "  src/vllm/v1/kv_offload/lmcache/remote_client.cpp\n"
            "  src/vllm/v1/kv_offload/fs_io.cpp)\n"
            "target_include_directories(runtime PRIVATE src include)\n"
            "add_executable(server src/vllm/entrypoints/openai/server_main.cpp)\n"
            "target_link_libraries(server PRIVATE runtime)\n"
            "add_library(parakeet STATIC\n"
            "  build-pr-windows-vulkan/_deps/parakeet_cpp-src/third_party/"
            "ggml/src/ggml-cpu/amx/amx.cpp)\n"
            "target_link_libraries(server PRIVATE parakeet)\n",
            encoding="utf-8",
        )
        build = root / "build-pr-windows-vulkan"
        query = build / ".cmake/api/v1/query"
        query.mkdir(parents=True)
        (query / "codemodel-v2").touch()
        configured = subprocess.run(
            ["cmake", "-S", str(root), "-B", str(build), "-G", "Ninja"],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(configured.returncode, 0,
                         configured.stdout + configured.stderr)
        result = subprocess.run(
            [sys.executable, str(CHECKER), "--root", str(root),
             "--build-dir", str(build),
             "--test-source-manifest", str(self.source_manifest)],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("unguarded POSIX", result.stdout + result.stderr)

    def test_representative_real_source_mutations_fail(self) -> None:
        console = (REPO / "src/vllm/platform/console_shutdown.cpp").read_text(
            encoding="utf-8"
        )
        self.assert_rejected(
            "src/vllm/platform/console_shutdown.cpp",
            console.replace(
                "registry.entrants.fetch_add(1, std::memory_order_seq_cst);",
                "",
                1,
            ),
            "stable event/in-flight",
        )

        process = (REPO / "src/vllm/platform/process.cpp").read_text(
            encoding="utf-8"
        )
        process_header = (REPO / "src/vllm/platform/process.h").read_text(
            encoding="utf-8"
        )
        self.assert_rejected(
            "src/vllm/platform/process.cpp",
            process.replace(
                '#include "vllm/platform/process.h"',
                '#include "vllm/platform/process.h"\n'
                '#include <unistd.h>\n'
                'void leaked_posix() { fork(); }',
                1,
            ),
            "unguarded POSIX",
        )
        root = self.make_tree({
            "src/vllm/platform/process.cpp": process,
            "src/vllm/platform/process.h": process_header +
            "\n#include <unistd.h>\nvoid leaked_posix() { fork(); }\n",
        })
        result = self.run_checker(root)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("src/vllm/platform/process.h", result.stdout + result.stderr)

    def test_real_console_rejects_state_access_after_final_decrement(self) -> None:
        console = (REPO / "src/vllm/platform/console_shutdown.cpp").read_text(
            encoding="utf-8"
        )
        decrement = (
            "state->in_flight.fetch_sub(1, std::memory_order_seq_cst);"
        )
        self.assertEqual(console.count(decrement), 1)
        self.assert_rejected(
            "src/vllm/platform/console_shutdown.cpp",
            console.replace(
                decrement,
                decrement + "\n  SetEvent(state->stop_event);",
                1,
            ),
            "final in-flight decrement",
        )

    def test_real_console_rejects_macro_post_decrement_result(self) -> None:
        console = (REPO / "src/vllm/platform/console_shutdown.cpp").read_text(
            encoding="utf-8"
        )
        decrement = "state->in_flight.fetch_sub(1, std::memory_order_seq_cst);"
        self.assertEqual(console.count(decrement), 1)
        self.assertEqual(console.count("return resumed;"), 1)
        mutated = console.replace(
            decrement,
            "#define POST_DECREMENT_RESULT "
            "(SetEvent(state->stop_event), resumed)\n  " + decrement,
            1,
        ).replace(
            "return resumed;", "return POST_DECREMENT_RESULT;", 1
        )
        self.assert_rejected(
            "src/vllm/platform/console_shutdown.cpp",
            mutated,
            "final in-flight decrement",
        )

    def test_real_console_drains_require_reachable_timeout_branches(self) -> None:
        console = (REPO / "src/vllm/platform/console_shutdown.cpp").read_text(
            encoding="utf-8"
        )
        timeout = (
            "if (GetTickCount64() - start >= kHandlerDrainTimeoutMs) return false;"
        )
        self.assertEqual(console.count(timeout), 2)
        decoy = (
            'const char* timeout_decoy = "if (GetTickCount64() - start >= '
            'kHandlerDrainTimeoutMs) return false;";\n'
            "    // if (GetTickCount64() - start >= kHandlerDrainTimeoutMs) "
            "return false;"
        )
        for occurrence in (0, 1):
            position = -1
            for _ in range(occurrence + 1):
                position = console.find(timeout, position + 1)
            self.assertGreaterEqual(position, 0)
            mutated = console[:position] + decoy + console[position + len(timeout):]
            self.assert_rejected(
                "src/vllm/platform/console_shutdown.cpp",
                mutated,
                "finite timeout",
            )

    def test_real_console_rejects_absolute_deadline_in_each_drain(self) -> None:
        console = (REPO / "src/vllm/platform/console_shutdown.cpp").read_text(
            encoding="utf-8"
        )
        start = "const ULONGLONG start = GetTickCount64();"
        elapsed = (
            "if (GetTickCount64() - start >= kHandlerDrainTimeoutMs) return false;"
        )
        self.assertEqual(console.count(start), 2)
        self.assertEqual(console.count(elapsed), 2)
        for function in ("DrainEntrantsWithTimeout", "DrainInFlightWithTimeout"):
            with self.subTest(function=function):
                function_at = console.index(f"bool {function}")
                next_function = console.find("\nbool ", function_at + 1)
                function_end = len(console) if next_function < 0 else next_function
                body = console[function_at:function_end]
                self.assertIn(start, body)
                self.assertIn(elapsed, body)
                body = body.replace(
                    start,
                    "const ULONGLONG deadline = "
                    "GetTickCount64() + kHandlerDrainTimeoutMs;",
                    1,
                ).replace(
                    elapsed,
                    "if (GetTickCount64() >= deadline) return false;",
                    1,
                )
                mutated = console[:function_at] + body + console[function_end:]
                self.assert_rejected(
                    "src/vllm/platform/console_shutdown.cpp",
                    mutated,
                    "finite timeout",
                )

    def test_rejects_disabled_smoke_and_missing_crt_audit(self) -> None:
        script = textwrap.dedent(SAFE_FILES["scripts/build-windows-release.ps1"])
        result = self.run_checker(self.make_tree({
            "scripts/build-windows-release.ps1": script,
        }))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assert_rejected(
            "scripts/build-windows-release.ps1",
            script.replace('Invoke-Checked $server @("--help")',
                           '# Invoke-Checked $server @("--help")'),
            "live --help smoke",
        )
        self.assert_rejected(
            "scripts/build-windows-release.ps1",
            script.replace("dumpbin /imports bin/vllm-server.exe", ""),
            "PE CRT import audit",
        )
        self.assert_rejected(
            "scripts/build-windows-release.ps1",
            script.replace("UCRTBASE", "UNRELATED_SYSTEM_DLL"),
            "dynamic CRT rejection",
        )


# The tree carried 53 files containing the token and 57 `[[noreturn]]` sites when
# this rule landed. The floor is deliberately well below that and deliberately
# well above zero: it is the non-vacuity guard for the real-tree case below, and
# a floor at or below the count that case can reach by reading NOTHING is a mute
# switch rather than a gate.
NORETURN_POPULATION_FLOOR = 40


class NoReturnNonVoidTests(unittest.TestCase):
    """MSVC C4646: `[[noreturn]]` on a function whose return type is not `void`.

    `/W4 /WX` turns it into C2220 and the whole project fails to compile, while
    GCC and Clang accept it in silence -- so `main` carried it (#1829) with every
    POSIX lane green. These cases are the Linux-side instrument for that class.
    """

    # The fixture-tree helpers are BORROWED, not inherited: subclassing
    # `WindowsPortabilityCheckerTest` would re-run its ~80 unrelated cases under
    # a second class name and double this suite's runtime.
    make_tree = WindowsPortabilityCheckerTest.make_tree
    run_checker = WindowsPortabilityCheckerTest.run_checker

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)

    def sites(self, source: str) -> list[str]:
        return [kind for _line, kind, _spelling in
                checker.noreturn_nonvoid_sites(source)]

    def spellings(self, source: str) -> list[str]:
        return [spelling for _line, _kind, spelling in
                checker.noreturn_nonvoid_sites(source)]

    # --- the real tree -----------------------------------------------------

    def test_real_tree_has_no_noreturn_on_a_non_void_return(self) -> None:
        """The case that would have caught #1829 before Windows ever saw it."""
        errors = checker.noreturn_nonvoid_errors(REPO)
        self.assertEqual(errors, [], "\n".join(errors))

    def test_real_tree_scan_reaches_the_noreturn_population(self) -> None:
        """Non-vacuity: the zero above is a zero because files were READ."""
        paths = checker.noreturn_scan_paths(REPO)
        carrying = [
            path for path in paths
            if "noreturn" in path.read_text(encoding="utf-8", errors="replace")
        ]
        self.assertGreaterEqual(len(carrying), NORETURN_POPULATION_FLOOR,
                                f"only {len(carrying)} of {len(paths)} scanned "
                                "files carry the token")
        relatives = {path.relative_to(REPO).as_posix() for path in paths}
        # The widening past `shipped_server_sources` is the point of the rule:
        # `windows-msvc-cpu` builds the tests and the examples too.
        self.assertIn("src/vllm/model_executor/models/dots3_note.h", relatives)
        self.assertTrue(any(name.startswith("tests/") for name in relatives))
        self.assertTrue(any(name.startswith("examples/") for name in relatives))
        self.assertTrue(any(name.endswith(".cu") for name in relatives))
        # Vendored code is not ours and compiles with /w.
        self.assertFalse([name for name in relatives
                          if name.startswith("third_party/")])

    def test_real_tree_declaration_still_carries_no_attribute(self) -> None:
        """#1829's exact site, asserted by uniqueness rather than by line."""
        header = (REPO / "src/vllm/model_executor/models/dots3_note.h").read_text(
            encoding="utf-8")
        declarations = re.findall(
            r"[^\n]*\bForwardLogits\s+ForwardDevice\s*\(", header)
        self.assertEqual(len(declarations), 1, declarations)
        self.assertNotIn("noreturn", declarations[0])

    # --- accepted shapes ---------------------------------------------------

    def test_void_returns_are_accepted(self) -> None:
        for source in (
            "[[noreturn]] void Fail(const std::string& why);",
            "[[noreturn]] static void RaiseForUnsupported(int code);",
            "extern \"C\" [[noreturn]] void vt_bad_args(int n, int nrc) { abort(); }",
            "  [[noreturn]] void Fail(std::string_view r) const { throw 1; }",
            "[[noreturn]] inline constexpr void Never();",
            "[[noreturn]] void Ns::Cls::Fail(int a) { throw 1; }",
            "[[noreturn]] auto Die() -> void;",
            "[[noreturn]] ~Guard();",
        ):
            with self.subTest(source=source):
                self.assertEqual(self.sites(source), [])

    def test_the_token_in_a_comment_or_a_literal_is_not_a_site(self) -> None:
        # `dots3_note_registry.cpp` really does discuss the attribute in prose.
        self.assertEqual(
            self.sites("// [[noreturn]] until W3-W10 land.\nint g();"), [])
        self.assertEqual(
            self.sites("/* [[noreturn]] int x( */\nvoid g();"), [])
        self.assertEqual(
            self.sites('const char* s = "[[noreturn]] int x(";'), [])

    # --- refused shapes ----------------------------------------------------

    def test_non_void_returns_are_refused(self) -> None:
        for source, expected in (
            ("[[noreturn]] int Die();", "int"),
            ("[[noreturn]] void* Die();", "void *"),
            ("[[noreturn]] std::pair<int, int> Die();", "std::pair<int, int>"),
            ("[[noreturn]] auto Die() -> int;", "auto"),
            ("[[noreturn]] static ForwardLogits ForwardDevice(\n"
             "    const std::vector<int32_t>& token_ids);", "ForwardLogits"),
        ):
            with self.subTest(source=source):
                self.assertEqual(self.sites(source), [expected])

    def test_the_reported_line_is_the_source_line(self) -> None:
        source = "\n".join([
            "// filler",
            "/* two",
            "   lines */",
            "void ok();",
            "[[noreturn]] int Die();",
        ])
        self.assertEqual(checker.noreturn_nonvoid_sites(source),
                         [(5, "int", "[[noreturn]]")])

    def test_an_unparsable_declaration_is_reported_not_dropped(self) -> None:
        """Fail closed: a shape the parser cannot read is a shape to look at."""
        self.assertEqual(self.sites("[[noreturn]];"),
                         ["<unparsed declaration>"])

    # --- every spelling of the same defect ----------------------------------

    def test_every_noreturn_spelling_on_a_non_void_return_is_refused(self) -> None:
        """The defect is the SHAPE. C4646's own text names `__declspec`.

        Before this widened, three of these five returned `[]` while the rule
        claimed to gate the class -- and `_Noreturn` sat in `_DECL_SPECIFIERS`
        as a token to strip, which made the hole read as coverage.
        """
        for source, spelling, return_type in (
            ("[[noreturn]] ForwardLogits f();",
             "[[noreturn]]", "ForwardLogits"),
            ("[[gnu::noreturn]] ForwardLogits f();",
             "[[gnu::noreturn]]", "ForwardLogits"),
            ("__declspec(noreturn) ForwardLogits f();",
             "__declspec(noreturn)", "ForwardLogits"),
            ("__attribute__((noreturn)) ForwardLogits f();",
             "__attribute__((noreturn))", "ForwardLogits"),
            # `_Noreturn` is C11, so the C return type goes with it.
            ("_Noreturn int f();", "_Noreturn", "int"),
        ):
            with self.subTest(source=source):
                self.assertEqual(self.sites(source), [return_type])
                self.assertEqual(self.spellings(source), [spelling])

    def test_every_noreturn_spelling_on_void_is_accepted(self) -> None:
        for source in (
            "[[gnu::noreturn]] void Fail(int code);",
            "__declspec(noreturn) void Fail(int code);",
            "__attribute__((noreturn)) void Fail(int code);",
            "_Noreturn void Fail(int code);",
            "[[noreturn]] _Noreturn void Fail(int code);",
            "[[noreturn]] [[nodiscard]] void Fail(int code);",
        ):
            with self.subTest(source=source):
                self.assertEqual(self.sites(source), [])

    def test_the_gnu_spelling_is_read_in_its_trailing_position(self) -> None:
        """`void f() __attribute__((noreturn));` is where GNU puts it.

        Read forwards it finds `;` before any `(`, so without the backward read
        a correct `void` declaration would report `<unparsed declaration>` -- a
        false red introduced by the widening itself.
        """
        for source in (
            "void Fail(int code) __attribute__((noreturn));",
            "void Fail(int code) __attribute__((noreturn)) { abort(); }",
            "auto Die() __attribute__((noreturn)) -> void;",
            "template <typename T>\nvoid Fail(T v) __attribute__((noreturn));",
            "class X {\n public:\n  void F(int c) __attribute__((noreturn));\n};",
        ):
            with self.subTest(source=source):
                self.assertEqual(self.sites(source), [])
        self.assertEqual(
            self.sites("ForwardLogits Fail(int c) __attribute__((noreturn));"),
            ["ForwardLogits"])
        self.assertEqual(
            self.sites("template <typename T>\nT Fail(T v) __attribute__((noreturn));"),
            ["T"])

    def test_an_export_macro_is_not_mistaken_for_a_declarator(self) -> None:
        """`VLLM_API` expands to `__declspec(dllexport)` on Windows.

        The checker reads the UNEXPANDED token, so this stays the fail-closed
        `'VLLM_API void'` report it already was; what must not happen is the
        widened parse taking `__declspec(`'s own paren for the declarator's.
        """
        self.assertEqual(self.sites("[[noreturn]] VLLM_API void f();"),
                         ["VLLM_API void"])
        # The real expansion, and the POSIX one beside it, are not sites at all.
        self.assertEqual(self.sites(
            "#define VLLM_API __declspec(dllexport)\n"
            "#define VLLM_API __attribute__((visibility(\"default\")))\n"), [])
        self.assertEqual(self.sites(
            "[[noreturn]] __declspec(dllexport) void f();"), [])

    def test_the_c11_spelling_survives_the_read_fast_path(self) -> None:
        """`_Noreturn` has no lowercase `noreturn` in it.

        `noreturn_nonvoid_errors` skips a file whose text does not carry the
        token, and a case-sensitive fast path would skip the C11 spelling
        entirely -- a hole that the detector-level cases cannot see, because
        they never go through the file read.
        """
        root = self.make_tree({
            "src/vt/cpu/refusing.c": "_Noreturn int Die(void);\n",
        })
        self.assertEqual(
            checker.noreturn_nonvoid_errors(root),
            ["src/vt/cpu/refusing.c:1: _Noreturn on a non-void return type "
             "'int' is MSVC C4646, and /W4 /WX makes it C2220"])

    def test_the_real_tree_carries_only_the_cpp11_spelling(self) -> None:
        """Non-vacuity for the widening: the zero above is measured, not assumed.

        Widening a detector over a tree that already satisfies it proves nothing
        unless the census is recorded. 56 active `[[noreturn]]` and nothing else
        on 2026-08-24.
        """
        census: dict[str, int] = {}
        for path in checker.noreturn_scan_paths(REPO):
            source = path.read_text(encoding="utf-8", errors="replace")
            if "noreturn" not in source and "Noreturn" not in source:
                continue
            active = checker.without_cpp_comments_and_literals(source)
            for match in checker._NORETURN_ATTRIBUTE.finditer(active):
                spelling = " ".join(match.group(0).split())
                census[spelling] = census.get(spelling, 0) + 1
        self.assertEqual(set(census), {"[[noreturn]]"}, census)
        # The same floor the file-count case uses, so this rule has ONE knob to
        # raise rather than a second magic number to drift away from it.
        self.assertGreaterEqual(census["[[noreturn]]"],
                                NORETURN_POPULATION_FLOOR, census)

    # --- reached through the checker's own entry point ---------------------

    def test_check_reports_a_noreturn_non_void_site(self) -> None:
        clean = self.make_tree()
        self.assertEqual(checker.check(clean, source_manifest=self.source_manifest),
                         [])
        root = self.make_tree({
            "src/vllm/model_executor/models/refusing.h": """
                [[noreturn]] static ForwardLogits ForwardDevice(int t);
            """,
        })
        errors = checker.check(root, source_manifest=self.source_manifest)
        self.assertEqual(
            [error for error in errors if "noreturn" in error],
            ["src/vllm/model_executor/models/refusing.h:2: [[noreturn]] on a "
             "non-void return type 'ForwardLogits' is MSVC C4646, and /W4 /WX "
             "makes it C2220"],
            errors,
        )

    def test_the_cli_refuses_a_noreturn_non_void_site(self) -> None:
        result = self.run_checker(self.make_tree({
            "src/vt/cpu/refusing.cpp": "[[noreturn]] int Die() { throw 1; }\n",
        }))
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("[[noreturn]] on a non-void return type",
                      result.stdout + result.stderr)

    def test_the_message_names_the_spelling_that_was_written(self) -> None:
        """`[[noreturn]]` in an error about `__declspec(noreturn)` misdirects."""
        errors = checker.check(self.make_tree({
            "src/vt/cpu/refusing.cpp": "__declspec(noreturn) int Die();\n",
        }), source_manifest=self.source_manifest)
        self.assertEqual(
            [error for error in errors if "noreturn" in error],
            ["src/vt/cpu/refusing.cpp:1: __declspec(noreturn) on a non-void "
             "return type 'int' is MSVC C4646, and /W4 /WX makes it C2220"],
            errors,
        )


class WindowsPortabilityWiringTests(unittest.TestCase):
    """A checker no lane runs catches nothing -- which is exactly how #1829 landed.

    This checker was invoked by no workflow, no hook and no preflight (#646,
    #680), so the wiring is asserted here rather than trusted to stay.
    """

    CHECKER_COMMAND = "python3 scripts/check-windows-portability.py"
    SUITE_COMMAND = "python3 -m unittest tests.scripts.test_check_windows_portability"

    def test_ci_runs_the_checker_and_its_suite_unconditionally(self) -> None:
        import yaml

        workflow = yaml.safe_load(
            (REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
        job = workflow["jobs"]["agent-record"]
        # `check-test-registration.py` counts a step as registered only when the
        # job owning it carries no `if:` -- a registration behind a condition is
        # not one (#873).
        self.assertNotIn("if", job)
        bodies = [step.get("run", "") for step in job["steps"]]
        self.assertTrue(
            any(self.CHECKER_COMMAND in body for body in bodies), bodies)
        self.assertTrue(
            any(self.SUITE_COMMAND in body for body in bodies), bodies)
        # The generator the checker forces is not on the runner image by default.
        owning = next(body for body in bodies if self.CHECKER_COMMAND in body)
        self.assertIn("ninja-build", owning)
        self.assertLess(owning.index("ninja-build"),
                        owning.index(self.CHECKER_COMMAND))

    def test_preflight_runs_the_suite_or_skips_with_a_reason(self) -> None:
        preflight = (REPO / "scripts/agent-preflight.sh").read_text(encoding="utf-8")
        # Asserted by count, and the failure message names the missing fragment
        # rather than echoing the whole script back at the reader.
        for fragment in (
            "tests.scripts.test_check_windows_portability",
            # A missing tool must be a SKIP, never a silent `ok`.
            'run "test_check_windows_portability"',
            'skip "test_check_windows_portability"',
        ):
            with self.subTest(fragment=fragment):
                self.assertEqual(
                    preflight.count(fragment), 1,
                    f"scripts/agent-preflight.sh must carry {fragment!r} once")


if __name__ == "__main__":
    unittest.main()
