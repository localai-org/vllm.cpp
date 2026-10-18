"""Live CPU regressions for the patched ROCm attachment controller (#3076).

The upstream files are pinned to rocm-systems 97f5574fe2fdc7bef44fb01545347912ee9f1779.
Only logging, build configuration, and SDK declarations are adapted for CPU CI.
The loader, attachment entry, and ptrace implementation remain the real sources.
"""
import json
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import copy
import signal
import time
import unittest
import contextlib
import struct

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = Path(__file__).parent / "fixtures/rocprof_attach_upstream"
PATCH = ROOT / "tools/bench/strix_four_engine/patches/rocprof-controller-preflight.patch"
CLI = ROOT / "tools/bench/strix_four_engine/rocprof_attach_preflight.py"

# CI-context exemption, recorded: the pinned rocm-systems controller
# (97f5574fe2fdc7bef44fb01545347912ee9f1779) is x86-only ptrace code --
# PTRACE_GETREGS/PTRACE_SETREGS and struct user_regs_struct with rax/rsp/rip
# do not exist on aarch64 (the kernel uses user_pt_regs via PTRACE_GETREGSET,
# with reg names x0..x30/sp/pc). The upstream pin has no aarch64 attachment
# implementation to mirror, so on a non-x86_64 host the setUpClass compile
# cannot succeed and there is nothing this suite could measure. The suite
# SKIPS there with this reason instead of erroring the whole class; it still
# runs for real on the x86_64 lanes.
import platform

CONTROLLER_IS_X86_64_ONLY = platform.machine() in ("x86_64", "amd64")


@unittest.skipUnless(
    CONTROLLER_IS_X86_64_ONLY,
    "the pinned rocm-systems attachment controller is x86-only ptrace code; "
    "no aarch64 implementation exists upstream to mirror, so the fixture "
    "cannot compile or measure anything on this architecture",
)
class ControllerPreflight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="rocprof-preflight-test-")
        cls.base = Path(cls.temp.name)
        cls.source = cls.base / "projects/rocprofiler-sdk/source/lib/rocprofv3-attach"
        shutil.copytree(UPSTREAM, cls.source)
        if PATCH.exists():
            subprocess.run(["git", "init", "-q", str(cls.base)], check=True)
            spec = importlib.util.spec_from_file_location("profiler_preflight", CLI)
            cls.builder = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.builder)
            cls.builder.apply_controller_patch(cls.base, lambda command, **kw:
                subprocess.run(command, check=True, capture_output=True, **kw))
        stubs = {
            "details/filesystem.hpp": '#pragma once\n#include <filesystem>\nnamespace fs = std::filesystem;\n',
            "lib/common/logging.hpp": '''#pragma once
#include <iostream>
#include <optional>
#include <memory>
#include <cstring>
#include <climits>
#include <algorithm>
#include <string_view>
#define ROCP_TRACE std::cerr
#define ROCP_ERROR std::cerr
#define ROCP_INFO std::cerr
#define LOG(x) std::cerr
#define LOG_IF(x,y) if(y) std::cerr
inline bool FLAGS_colorlogtostderr;
namespace rocprofiler { namespace common {
struct logging_config { bool install_failure_handler; };
inline void init_logging(const char*, logging_config) {}
}}
''',
            "lib/common/static_object.hpp": '#pragma once\n',
            "lib/common/environment.hpp": '''#pragma once
#include <string>
#include <cstdlib>
namespace rocprofiler { namespace common {
inline std::string get_env(const char* k, const char* d) {
auto v = std::getenv(k); return v ? v : d;
}}}
''',
            "rocprofiler-sdk/defines.h": '''#pragma once
#define ROCPROFILER_EXTERN_C_INIT extern "C" {
#define ROCPROFILER_EXTERN_C_FINI }
#define ROCPROFILER_EXPORT __attribute__((visibility("default")))
''',
            "rocprofiler-sdk/rocprofiler.h": '''#pragma once
#include "defines.h"
enum rocprofiler_status_t { ROCPROFILER_STATUS_SUCCESS = 0,
ROCPROFILER_STATUS_ERROR = 1, ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT = 2 };
''',
        }
        for name, contents in stubs.items():
            p = cls.base / "include" / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(contents)
        cls.lib = cls.base / "librocprofv3-attach.so.1"
        subprocess.run(["g++", "-std=c++17", "-shared", "-fPIC", "-pthread",
                        "-I", str(cls.base / "include"),
                        str(cls.source / "ptrace_session.cpp"),
                        str(cls.source / "rocprofv3_attach.cpp"), "-ldl", "-lcrypto",
                        "-Wl,-soname,librocprofv3-attach.so.1", "-o", str(cls.lib)],
                       check=True, capture_output=True)
        cls.driver = cls.base / "driver.py"
        cls.driver.write_text('''import ctypes, sys
lib = ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_GLOBAL)
fn = getattr(lib, sys.argv[3] if len(sys.argv) > 3 else "preflight", None)
if fn is None:
    print("controller preflight entry is absent", file=sys.stderr)
    sys.exit(90)
if len(sys.argv) > 3 and sys.argv[3] == "attach":
    fn.argtypes = [ctypes.c_uint32]
    result = fn(int(sys.argv[2]))
    if result == 0:
        result = lib.detach()
    sys.exit(result)
fn.argtypes = [ctypes.c_uint32, ctypes.c_int]
sys.exit(fn(int(sys.argv[2]), int(sys.argv[4]) if len(sys.argv) > 4 else 1))
''')
        cls.fixture = cls.base / "librocprofiler-register.so"
        c = cls.base / "register.c"
        c.write_text('''#include <stdio.h>
#include <stdlib.h>
static void called(void) {
const char* p=getenv("FORBIDDEN_LOG");
if(p) { FILE* f=fopen(p,"a"); if(f) { fputs("registration call\\n",f); fclose(f); } }
}
int rocprofiler_register_attach(void* a, void* b) { called(); return 0; }
int rocprofiler_register_detach(void) { called(); return 0; }
''')
        subprocess.run(["gcc", "-shared", "-fPIC", "-Wl,--build-id", str(c),
                        "-o", str(cls.fixture)], check=True)
        shutil.copy2(cls.fixture, cls.base / "librocprofiler-sdk.so")
        spy = cls.base / "spy.c"
        spy.write_text('''#define _GNU_SOURCE
#include <sys/ptrace.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <stdarg.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/user.h>
#include <elf.h>
ssize_t pread(int fd, void* buf, size_t size, off_t offset) {
ssize_t (*real)(int,void*,size_t,off_t)=dlsym(RTLD_NEXT,"pread");
ssize_t n=real(fd,buf,size,offset);
const char* fault=getenv("METADATA_FAULT");
if(fault && (!strcmp(fault,"allocation-overflow") || !strcmp(fault,"allocation-too-small")) &&
   offset==0 && n>sizeof(Elf64_Ehdr)) {
Elf64_Ehdr* eh=buf;
if(!memcmp(eh->e_ident,ELFMAG,SELFMAG) && eh->e_phoff+eh->e_phnum*sizeof(Elf64_Phdr)<(size_t)n) {
Elf64_Phdr* ph=(void*)((char*)buf+eh->e_phoff);
for(int i=0;i<eh->e_phnum;++i) if(ph[i].p_type==PT_LOAD) {
ph[i].p_memsz=!strcmp(fault,"allocation-overflow") ? UINT64_MAX : 0; break; }
}
}
return n;
}
long sysconf(int name) {
long (*real)(int)=dlsym(RTLD_NEXT,"sysconf");
const char* fault=getenv("METADATA_FAULT");
if(fault && !strcmp(fault,"invalid-page") && name==_SC_PAGESIZE) return 0;
return real(name);
}
static int seized;
void* dlsym(void* handle, const char* name) {
void* (*real)(void*,const char*)=dlvsym(RTLD_NEXT,"dlsym","GLIBC_2.2.5");
void* value=real(handle,name);
const char* fault=getenv("METADATA_FAULT");
if(value && fault && !strcmp(fault,"exported-offset") &&
   !strcmp(name,"rocprofiler_register_attach")) return (char*)value+1;
return value;
}
static void forbidden(const char* op) {
const char* name = getenv("FORBIDDEN_LOG");
if(name) { FILE* f = fopen(name, "a"); if(f) { fprintf(f, "%s\\n", op); fclose(f); } }
errno = EPERM;
}
long ptrace(enum __ptrace_request op, ...) {
const char* fault=getenv("METADATA_FAULT");
if(fault && !strcmp(fault,"simulate-injection")) {
va_list ap; va_start(ap,op); (void)va_arg(ap,int); (void)va_arg(ap,void*);
void* data=va_arg(ap,void*); va_end(ap);
if(op==PTRACE_SEIZE) seized=1;
if(op==PTRACE_GETREGS) { struct user_regs_struct* r=data; memset(r,0,sizeof(*r)); r->rax=0x100000; r->rsp=0x800000; }
if(op==PTRACE_SETREGS) { struct user_regs_struct* r=data;
FILE* f=fopen(getenv("SEIZURE_LOG"),"a"); fprintf(f,"%llu\\n",r->rax); fclose(f); }
return 0;
}
if(fault && !strcmp(fault,"post-seizure")) {
const char* name = getenv("SEIZURE_LOG"); FILE* f=fopen(name,"a");
fprintf(f,"%d\\n",op); fclose(f);
if(op==PTRACE_SEIZE) { seized=1; return 0; }
if(op==PTRACE_INTERRUPT || op==PTRACE_DETACH) return 0;
}
forbidden("ptrace"); return -1;
}
pid_t waitpid(pid_t pid,int* status,int options) {
if(seized) { if(status) *status=0; return pid; }
pid_t (*real)(pid_t,int*,int)=dlsym(RTLD_NEXT,"waitpid"); return real(pid,status,options);
}
int kill(pid_t pid, int sig) { forbidden("kill"); return -1; }
ssize_t process_vm_writev(pid_t p, const struct iovec* l, unsigned long lc,
const struct iovec* r, unsigned long rc, unsigned long flags) {
forbidden("process_vm_writev"); return -1;
}
int openat(int dir, const char* name, int flags, ...) {
static int count;
int (*real)(int,const char*,int,...) = dlsym(RTLD_NEXT, "openat");
const char* fault = getenv("METADATA_FAULT");
if(fault && !strcmp(name, "maps")) {
if(!strcmp(fault,"unreadable")) { errno=EACCES; return -1; }
if(!strcmp(fault,"malformed")) { int fd=memfd_create("bad-maps",0); write(fd,"not a map",9); lseek(fd,0,0); return fd; }
static int target_reads;
char procpath[64], targetpath[128]; snprintf(procpath,sizeof(procpath),"/proc/self/fd/%d",dir);
ssize_t z=readlink(procpath,targetpath,sizeof(targetpath)-1); if(z<0) return -1; targetpath[z]=0;
char own[64]; snprintf(own,sizeof(own),"/proc/%d",getpid());
int target=strcmp(targetpath,own)!=0;
if(target && !strcmp(fault,"reservation-maps"))
    return real(AT_FDCWD,getenv("RESERVATION_MAPS"),O_RDONLY);
if(target) ++target_reads;
if(target && (!strcmp(fault,"mapped-inode") || !strcmp(fault,"mapped-device") || !strcmp(fault,"symbol-noexec") ||
             !strcmp(fault,"unaccounted-load") ||
             (target_reads>=3 && (!strcmp(fault,"target-recheck") ||
                                  !strcmp(fault,"sdk-recheck") || !strcmp(fault,"permission-recheck"))))) {
int original=real(dir,name,flags); if(original<0) return original;
FILE* input=fdopen(original,"r"); int fd=memfd_create("fault-maps",0);
char* line=NULL; size_t capacity=0; int added=0;
while(getline(&line,&capacity,input)>0) {
char* wanted=strstr(line,!strcmp(fault,"sdk-recheck") ? "librocprofiler-sdk.so" : "librocprofiler-register.so");
if(wanted) {
unsigned long long begin,end,offset,inode; unsigned maj,min; char perms[5]; int consumed;
if(sscanf(line,"%llx-%llx %4s %llx %x:%x %llu %n",&begin,&end,perms,&offset,&maj,&min,&inode,&consumed)!=7) abort();
if(!strcmp(fault,"unaccounted-load")) {
if(!added++) dprintf(fd,"%llx-%llx %s %llx %x:%x %llu %s",begin+0x100000000ULL,begin+0x100000000ULL+4096,perms,offset,maj,min,inode,line+consumed);
} else {
if(!strcmp(fault,"mapped-device")) ++min;
else if(!strcmp(fault,"symbol-noexec")) perms[2]='-';
else if(!strcmp(fault,"permission-recheck")) perms[1]=perms[1]=='w' ? '-' : 'w';
else ++inode;
dprintf(fd,"%llx-%llx %s %llx %x:%x %llu %s",begin,end,perms,offset,maj,min,inode,line+consumed);
continue;
}
}
write(fd,line,strlen(line));
}
free(line); fclose(input); lseek(fd,0,0); return fd;
}
}
if(fault && ((!strcmp(fault,"identity-change") && ++count > 1) ||
(!strcmp(fault,"post-seizure") && seized)) && !strcmp(name,"stat")) {
int old=real(dir,name,flags); if(old<0) return old;
char buf[4096]; ssize_t n=read(old,buf,sizeof(buf)-1); close(old);
if(n<=0) return -1; buf[n]=0;
char* value=strrchr(buf,')')+2; int field=3;
while(field<22 && value) { value=strchr(value,' '); if(value) { ++value; ++field; } }
if(value) value[0] = value[0]=='1' ? '2' : '1';
int fd=memfd_create("changed-stat",0); write(fd,buf,n); lseek(fd,0,0); return fd;
}
return real(dir,name,flags);
}
''')
        cls.spy = cls.base / "spy.so"
        subprocess.run(["gcc", "-shared", "-fPIC", str(spy), "-ldl", "-o", str(cls.spy)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_live(self, *, folder=None, mutate=None, mode="preflight", fault=None, extra="", report_fd=1,
                 maps_transform=None):
        folder = folder or self.base
        fixture = folder / "librocprofiler-register.so"
        log = folder / "forbidden.log"
        log.unlink(missing_ok=True)
        child = subprocess.Popen(["python3", "-c", '''import ctypes, sys
ctypes.CDLL(sys.argv[1]); ctypes.CDLL(sys.argv[2]); print("ready", flush=True)
exec(sys.argv[3])
sys.stdin.read()
''', str(fixture), str(folder / "librocprofiler-sdk.so"), extra],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                 env=dict(os.environ, FORBIDDEN_LOG=str(log)))
        try:
            self.assertEqual(child.stdout.readline(), "ready\n")
            if extra:
                self.assertEqual(child.stdout.readline(), "extra-ready\n")
            if mutate:
                mutate(child, fixture)
            env = dict(os.environ, LD_LIBRARY_PATH=str(self.base), LD_PRELOAD=str(self.spy),
                       FORBIDDEN_LOG=str(log), SEIZURE_LOG=str(folder / "seizure.log"))
            if fault:
                env["METADATA_FAULT"] = fault
            if maps_transform:
                replacement = folder / "reservation-maps"
                replacement.write_text(maps_transform(Path(f"/proc/{child.pid}/maps").read_text(), fixture))
                env.update(METADATA_FAULT="reservation-maps", RESERVATION_MAPS=str(replacement))
            result = subprocess.run(["python3", str(self.driver), str(self.lib), str(child.pid), mode, str(report_fd)],
                                    env=env, capture_output=True, text=True, timeout=10)
            self.assertFalse(log.exists(), log.read_text() if log.exists() else "")
            return result, child.pid
        finally:
            child.communicate(timeout=5)

    def copied(self):
        directory = Path(tempfile.mkdtemp(dir=self.base))
        shutil.copy2(self.fixture, directory / self.fixture.name)
        shutil.copy2(self.base / "librocprofiler-sdk.so", directory / "librocprofiler-sdk.so")
        return directory

    @contextlib.contextmanager
    def reservation_fixture(self, *, virtual_base=0, allocation_tail=False,
                            object_name="librocprofiler-register.so", file_prefix=0):
        """glibc-2.39 elf/dl-map-segments.h:75-111: real dlopen reservations."""
        fixture = self.base / object_name
        original = fixture.read_bytes()
        try:
            command = ["gcc", "-shared", "-fPIC", "-Wl,--build-id,-z,max-page-size=0x200000",
                       f"-Wl,-Ttext-segment={virtual_base:#x}",
                       str(self.base / "register.c"), "-o", str(fixture)]
            subprocess.run(command, check=True)
            if allocation_tail:
                data = bytearray(fixture.read_bytes())
                phoff = struct.unpack_from("<Q", data, 32)[0]
                count = struct.unpack_from("<H", data, 56)[0]
                first = next(phoff + i * 56 for i in range(count)
                             if struct.unpack_from("<I", data, phoff + i * 56)[0] == 1)
                # A partial second BSS page in the first load, before the next load.
                struct.pack_into("<Q", data, first + 40, 0x1800)
                fixture.write_bytes(data)
            if file_prefix:
                # Keep segment alignment and all virtual addresses. The loader reads
                # the header at zero, then maps the original image at a nonzero offset.
                image = fixture.read_bytes()
                data = bytearray(file_prefix) + bytearray(image)
                data[:64] = image[:64]
                phoff = struct.unpack_from("<Q", image, 32)[0]
                shoff = struct.unpack_from("<Q", image, 40)[0]
                phnum = struct.unpack_from("<H", image, 56)[0]
                shnum = struct.unpack_from("<H", image, 60)[0]
                struct.pack_into("<Q", data, 32, phoff + file_prefix)
                struct.pack_into("<Q", data, 40, shoff + file_prefix)
                for index in range(phnum):
                    field = file_prefix + phoff + index * 56 + 8
                    struct.pack_into("<Q", data, field, struct.unpack_from("<Q", data, field)[0] + file_prefix)
                for index in range(1, shnum):
                    field = file_prefix + shoff + index * 64 + 24
                    struct.pack_into("<Q", data, field, struct.unpack_from("<Q", data, field)[0] + file_prefix)
                fixture.write_bytes(data)
            yield command
        finally:
            fixture.write_bytes(original)

    def test_real_loader_reservations_pass_public_preflight(self):
        for virtual_base, object_name, file_prefix in ((0, "librocprofiler-register.so", 0),
                                                       (0x200000, "librocprofiler-register.so", 0),
                                                       (0, "librocprofiler-register.so", 0x200000),
                                                       (0, "librocprofiler-sdk.so", 0)):
            with self.subTest(virtual_base=virtual_base, object_name=object_name, file_prefix=file_prefix), \
                    self.reservation_fixture(virtual_base=virtual_base, object_name=object_name,
                                             file_prefix=file_prefix) as command:
                observed = {}
                fixture = self.base / object_name
                def observe(child, path):
                    observed["maps"] = [line for line in Path(f"/proc/{child.pid}/maps").read_text().splitlines()
                                        if str(fixture) in line]
                result, _ = self.run_live(mutate=observe)
                headers = subprocess.check_output(["readelf", "-lW", str(fixture)], text=True)
                print(json.dumps({"reservation_compile": command, "headers": headers, **observed,
                                  "exit": result.returncode, "stderr": result.stderr}), flush=True)
                gaps = [line for line in observed["maps"] if line.split()[1] == "---p"]
                self.assertGreaterEqual(len(gaps), 1, "fixture did not produce a loader reservation")
                self.assertEqual(int(observed["maps"][0].split()[2], 16), file_prefix)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(result.stdout)
                self.assertEqual(report["status"], "PASS")
                self.assertIs(report["target_mutated"], False)
                key = "sdk" if object_name == "librocprofiler-sdk.so" else "target"
                self.assertEqual(report[key]["sha256"], hashlib.sha256(fixture.read_bytes()).hexdigest())

    def test_reservation_mapping_refusals(self):
        # Each alteration preserves the other mappings and the loaded bytes.
        for fault in ("read", "write", "execute", "shared", "offset", "unaligned-start",
                      "unaligned-end", "before-object", "after-object", "cross-load", "inode", "device"):
            with self.subTest(fault=fault), self.reservation_fixture():
                rejected = {}
                def transform(text, path):
                    lines = text.splitlines()
                    entries = [(i, line.split(None, 5)) for i, line in enumerate(lines) if str(path) in line]
                    index, fields = next((i, list(f)) for i, f in entries if f[1] == "---p")
                    begin, end = (int(n, 16) for n in fields[0].split("-"))
                    offset = int(fields[2], 16)
                    base = int(entries[0][1][0].split("-")[0], 16)
                    if fault in ("read", "write", "execute", "shared"):
                        fields[1] = {"read": "r--p", "write": "-w-p", "execute": "--xp", "shared": "---s"}[fault]
                    elif fault == "offset": offset += 4096
                    elif fault == "unaligned-start": begin += 1; offset += 1
                    elif fault == "unaligned-end": end -= 1
                    elif fault == "before-object":
                        begin, end, offset = base - 4096, base, (1 << 64) - 4096
                    elif fault == "after-object":
                        begin = int(entries[-1][1][0].split("-")[1], 16)
                        end, offset = begin + 4096, begin - base
                    elif fault == "cross-load": end += 4096
                    elif fault == "inode": fields[4] = str(int(fields[4]) + 1)
                    elif fault == "device": fields[3] = "00:00"
                    fields[0], fields[2] = f"{begin:x}-{end:x}", f"{offset:x}"
                    rejected.update(path=str(path), interval=fields[0])
                    lines[index] = " ".join(fields)
                    return "\n".join(lines) + "\n"
                result, _ = self.run_live(maps_transform=transform)
                self.assertNotEqual(result.returncode, 0, fault)
                self.assertIn(rejected["path"], result.stderr)
                self.assertIn(rejected["interval"], result.stderr)

    def test_reservation_cannot_cover_allocation_tail(self):
        with self.reservation_fixture(allocation_tail=True):
            rejected = {}
            def transform(text, path):
                lines = text.splitlines()
                entries = [(i, line.split(None, 5)) for i, line in enumerate(lines) if str(path) in line]
                index, fields = next((i, list(f)) for i, f in entries if f[1] == "---p")
                base = int(entries[0][1][0].split("-")[0], 16)
                self.assertEqual(int(fields[0].split("-")[0], 16), base + 0x2000)
                # File bytes stop in page one. This page is a p_memsz allocation tail,
                # even though its synthetic map has every reservation identity/offset.
                fields[0], fields[2] = f"{base + 0x1000:x}-{base + 0x2000:x}", "1000"
                rejected["interval"] = fields[0]
                lines.append(" ".join(fields))
                return "\n".join(lines) + "\n"
            result, _ = self.run_live(maps_transform=transform)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unaccounted mapped load instance", result.stderr)
            self.assertIn(str(self.fixture), result.stderr)
            self.assertIn(rejected["interval"], result.stderr)

    def test_reservation_allocation_arithmetic_overflow_is_refused(self):
        with self.reservation_fixture():
            result, _ = self.run_live(fault="allocation-overflow")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("ELF allocation arithmetic overflow", result.stderr)

    def test_reservation_allocation_metadata_is_refused(self):
        for fault, reason in (("invalid-page", "invalid system page size"),
                              ("allocation-too-small", "ELF file bytes exceed allocation")):
            with self.subTest(fault=fault), self.reservation_fixture():
                result, _ = self.run_live(fault=fault)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(reason, result.stderr)

    def test_identical_live_objects_pass_without_attachment(self):
        """Removing public preflight or reporting a different identity fails."""
        result, pid = self.run_live()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["pid"], pid)
        self.assertEqual(report["target_mutated"], False)
        self.assertEqual(report["controller"]["sha256"], hashlib.sha256(self.fixture.read_bytes()).hexdigest())
        notes = subprocess.check_output(["readelf", "-n", str(self.fixture)], text=True)
        build_id = next(line.split("Build ID:", 1)[1].strip() for line in notes.splitlines() if "Build ID:" in line)
        self.assertEqual(report["controller"]["build_id"], build_id)
        self.assertEqual(report["target"]["build_id"], build_id)
        symbols = subprocess.check_output(["nm", "-D", str(self.fixture)], text=True)
        for key in ("attach", "detach"):
            expected = int(next(line.split()[0] for line in symbols.splitlines()
                                if line.endswith("rocprofiler_register_" + key)), 16)
            self.assertEqual(report[key + "_offset"], expected)

    def test_same_name_different_bytes_rejected_before_ptrace(self):
        """Deleting byte comparison must fail even with equal symbol offsets."""
        for mode in ("preflight", "attach"):
            folder = self.copied()
            with (folder / self.fixture.name).open("ab") as f:
                f.write(b"different immutable fixture bytes")
            result, _ = self.run_live(folder=folder, mode=mode)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("registration bytes differ", result.stderr)

    def test_deleted_and_replaced_files_rejected(self):
        """Removing backing file identity validation accepts stale paths."""
        for replace in (False, True):
            def mutation(child, path):
                path.unlink()
                if replace:
                    shutil.copy2(self.fixture, path)
            result, _ = self.run_live(folder=self.copied(), mutate=mutation)
            self.assertNotEqual(result.returncode, 0)

    def test_duplicate_load_instance_rejected(self):
        """Grouping only by inode would accept two dlmopen load instances."""
        extra = '''dl = ctypes.CDLL(None); dl.dlmopen.restype = ctypes.c_void_p
dl.dlmopen.argtypes = [ctypes.c_long, ctypes.c_char_p, ctypes.c_int]
assert dl.dlmopen(-1, sys.argv[1].encode(), 2)
print("extra-ready", flush=True)'''
        result, _ = self.run_live(extra=extra)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ambiguous ELF load instances", result.stderr)

    def test_invalid_process_rejected(self):
        """Dropping process lifetime validation accepts an exited target."""
        result, _ = self.run_live(mutate=lambda child, path: child.communicate(timeout=5))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("process identity", result.stderr)

    def test_metadata_failures_rejected(self):
        """Faults at procfs reads must not become success or ptrace calls."""
        for fault in ("unreadable", "malformed", "identity-change"):
            for mode in ("preflight", "attach"):
                result, _ = self.run_live(fault=fault, mode=mode)
                self.assertNotEqual(result.returncode, 0, (fault, result.stderr))

    def test_independent_identity_boundaries_reject_faults(self):
        """Each object check sees an otherwise-valid fault at its own boundary."""
        for fault, reason in (("mapped-inode", "mapped backing file"),
                              ("mapped-device", "mapped backing file"),
                              ("unaccounted-load", "unaccounted mapped load"),
                              ("exported-offset", "symbol offsets differ"),
                              ("symbol-noexec", "not in an executable mapping"),
                              ("target-recheck", "mapped backing file"),
                              ("sdk-recheck", "mapped backing file"),
                              ("permission-recheck", "mapped segments changed")):
            for mode in ("preflight", "attach"):
                with self.subTest(fault=fault, mode=mode):
                    result, _ = self.run_live(fault=fault, mode=mode)
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertIn(reason, result.stderr)

    def test_registration_symbol_page_must_remain_executable_and_mapped(self):
        """A mapped segment start does not prove the registration code is mapped."""
        original = self.fixture.read_bytes()
        source = self.base / "large-register.c"
        source.write_text('__asm__(".pushsection .text\\n.space 16384,0x90\\n.popsection\\n");\n'
                          'int rocprofiler_register_attach(void*a,void*b){return 0;}\n'
                          'int rocprofiler_register_detach(void){return 0;}\n')
        try:
            subprocess.run(["gcc", "-shared", "-fPIC", "-Wl,--build-id", str(source),
                            "-o", str(self.fixture)], check=True)
            for action in ("unmap", "unmap-interior", "noexec"):
                extra = '''import os
lib=ctypes.CDLL(sys.argv[1]); address=ctypes.cast(lib.rocprofiler_register_attach,ctypes.c_void_p).value
size=os.sysconf("SC_PAGESIZE"); page=address-address%size
libc=ctypes.CDLL(None)
libc.munmap.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
libc.mprotect.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int]
''' + ("assert libc.munmap(page-size,size)==0\n" if action == "unmap-interior" else
       "assert libc.munmap(page,size)==0\n" if action == "unmap" else
       "assert libc.mprotect(page,size,1)==0\n") + 'print("extra-ready",flush=True)'
                with self.subTest(action=action):
                    result, _ = self.run_live(extra=extra)
                    self.assertNotEqual(result.returncode, 0, result.stdout)
        finally:
            self.fixture.write_bytes(original)

    def test_report_write_failure_nonzero(self):
        """Ignoring write errors would report success without a complete report."""
        result, _ = self.run_live(report_fd=-1)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("report write failed", result.stderr)

    def test_nonzero_elf_load_address(self):
        """Treating the first map as load bias loses nonzero ELF virtual addresses."""
        folder = self.copied()
        subprocess.run(["gcc", "-shared", "-fPIC", "-Wl,--build-id,-Ttext-segment=0x10000",
                        str(self.base / "register.c"), "-o", str(folder / self.fixture.name)], check=True)
        # The actual controller loader chooses this fixture, not a test resolver.
        original = self.fixture.read_bytes()
        try:
            shutil.copyfile(folder / self.fixture.name, self.fixture)
            result, _ = self.run_live(folder=folder)
            self.assertEqual(result.returncode, 0, result.stderr)
        finally:
            self.fixture.write_bytes(original)

    def test_missing_symbols_rejected(self):
        """Dropping either symbol validation accepts an unusable registration object."""
        original = self.fixture.read_bytes()
        try:
            for omitted in ("attach", "detach"):
                source = self.base / "missing.c"
                source.write_text("int rocprofiler_register_" +
                                  ("detach" if omitted == "attach" else "attach") + "() {return 0;}\n")
                subprocess.run(["gcc", "-shared", "-fPIC", "-Wl,--build-id", str(source),
                                "-o", str(self.fixture)], check=True)
                result, _ = self.run_live()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("registration symbol", result.stderr)
        finally:
            self.fixture.write_bytes(original)

    def test_symbol_from_dependency_rejected(self):
        """Accepting dlsym results from another object produces invalid offsets."""
        original = self.fixture.read_bytes()
        dependency = self.base / "libdependency.so"
        shutil.copy2(self.fixture, dependency)
        source = self.base / "dependent.c"
        source.write_text("extern int rocprofiler_register_attach(void*,void*);\n"
                          "int dependency_anchor(void) {return rocprofiler_register_attach(0,0);}\n")
        try:
            subprocess.run(["gcc", "-shared", "-fPIC", "-Wl,--build-id", str(source),
                            "-L" + str(self.base), "-ldependency", "-Wl,-rpath,$ORIGIN",
                            "-o", str(self.fixture)], check=True)
            result, _ = self.run_live()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("different object", result.stderr)
        finally:
            self.fixture.write_bytes(original)

    def test_preparation_rejects_unpinned_source_without_writes(self):
        """Skipping the source pin check can patch an unrelated revision."""
        destination = self.base / "must-not-exist"
        result = subprocess.run(["python3", str(CLI), "prepare", "--source", str(ROOT),
                                 "--output", str(destination)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("source revision", result.stderr)
        self.assertFalse(destination.exists())

    def test_preparation_failure_stops_owned_descendants(self):
        """Timeout and command failure must terminate forked workers too."""
        for failure in ("timeout", "error"):
            with self.subTest(failure=failure):
                folder = Path(tempfile.mkdtemp(dir=self.base))
                shim = folder / "git"
                shim.write_text('''#!/usr/bin/python3
import os, pathlib, time, sys
pid = os.fork()
if pid == 0:
    for fd in (0, 1, 2): os.close(fd)
    time.sleep(60)
else:
    pathlib.Path(os.environ["CHILD_PID"]).write_text(str(pid))
    pathlib.Path(os.environ["CHILD_PID"] + ".parent").write_text(str(os.getpid()))
    if os.environ["FAILURE"] == "error": sys.exit(1)
    time.sleep(60)
''')
                shim.chmod(0o755)
                child_file = folder / "child.pid"
                try:
                    result = subprocess.run(["python3", str(CLI), "prepare", "--source", str(folder),
                                             "--output", str(folder / "out"), "--timeout", "1"],
                                            env=dict(os.environ, PATH=str(folder) + ":" + os.environ["PATH"],
                                                     CHILD_PID=str(child_file), FAILURE=failure),
                                            capture_output=True, text=True, timeout=8)
                except subprocess.TimeoutExpired:
                    for path in (child_file, Path(str(child_file) + ".parent")):
                        if path.exists():
                            try: os.kill(int(path.read_text()), signal.SIGKILL)
                            except ProcessLookupError: pass
                    self.fail("preparation cleanup exceeded its bounded wait")
                self.assertNotEqual(result.returncode, 0)
                pid = int(child_file.read_text())
                def running():
                    try:
                        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
                    except FileNotFoundError:
                        return False
                try:
                    deadline = time.monotonic() + 2
                    while running() and time.monotonic() < deadline:
                        time.sleep(0.02)
                    self.assertFalse(running(), "preparation left a live descendant")
                    self.assertFalse((folder / "out/build.json").exists())
                finally:
                    if running(): os.kill(pid, signal.SIGKILL)

    def test_preparation_cli_builds_patched_controller(self):
        """The public prepare path must patch the sources it actually compiles.

        Git metadata and CMake configuration are offline harness adaptations.
        Patch application and compilation use real git and g++ on pinned sources.
        This is CPU entry-point coverage, not a production SDK build claim.
        """
        folder = Path(tempfile.mkdtemp(dir=self.base))
        shim = folder / "bin"
        shim.mkdir()
        config = dict(git=shutil.which("git"), upstream=str(UPSTREAM),
                      include=str(self.base / "include"), pin=self.builder.PIN)
        (folder / "config.json").write_text(json.dumps(config))
        script = '''#!/usr/bin/python3
import json, os, pathlib, shutil, subprocess, sys
c = json.loads(pathlib.Path(os.environ["PREP_FIXTURE"]).read_text())
a = sys.argv[1:]
if pathlib.Path(sys.argv[0]).name == "git":
    if a[0] == "apply":
        sys.exit(subprocess.call([c["git"], *a]))
    if "rev-parse" in a: print(c["pin"])
    elif a[0] == "clone":
        clone = pathlib.Path(a[-1])
        shutil.copytree(c["upstream"], clone / "projects/rocprofiler-sdk/source/lib/rocprofv3-attach")
        subprocess.run([c["git"], "init", "-q", str(clone)], check=True)
    elif "submodule" in a and "status" in a: print(" " + c["pin"] + " external/fixture")
    elif "status" not in a and "checkout" not in a and "submodule" not in a:
        raise RuntimeError(a)
else:
    if "-S" in a:
        build = pathlib.Path(a[a.index("-B") + 1]); build.mkdir()
        (build / "source").write_text(a[a.index("-S") + 1])
    else:
        build = pathlib.Path(a[1])
        src = pathlib.Path((build / "source").read_text()) / "source/lib/rocprofv3-attach"
        subprocess.run(["g++", "-std=c++17", "-shared", "-fPIC", "-pthread",
            "-I", c["include"], str(src / "ptrace_session.cpp"),
            str(src / "rocprofv3_attach.cpp"), "-ldl", "-lcrypto",
            "-Wl,-soname,librocprofv3-attach.so.1", "-o",
            str(build / "librocprofv3-attach.so.1")], check=True)
'''
        for name in ("git", "cmake"):
            path = shim / name
            path.write_text(script)
            path.chmod(0o755)
        output = folder / "prepared"
        result = subprocess.run(["python3", str(CLI), "prepare", "--source", str(folder),
                                 "--output", str(output)], capture_output=True, text=True,
                                env=dict(os.environ, PATH=str(shim) + ":" + os.environ["PATH"],
                                         PREP_FIXTURE=str(folder / "config.json")), timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads((output / "build.json").read_text())
        self.assertEqual(record["status"], "BUILT_NOT_HARDWARE_VALIDATED")
        library = Path(record["controller"])
        self.assertEqual(record["controller_sha256"], hashlib.sha256(library.read_bytes()).hexdigest())
        # Resolve the real compiled export in a child. Never attach to a process.
        entry = subprocess.run(["python3", "-c",
                                "import ctypes,sys; assert ctypes.CDLL(sys.argv[1]).preflight",
                                str(library)], capture_output=True, text=True)
        self.assertEqual(entry.returncode, 0, entry.stderr)

    def test_wrapper_publishes_actual_controller_snapshot(self):
        """The public CLI must invoke the real controller and preserve its evidence."""
        child = subprocess.Popen(["python3", "-c", '''import ctypes,sys
ctypes.CDLL(sys.argv[1]); ctypes.CDLL(sys.argv[2]); print("ready",flush=True); sys.stdin.read()
''', str(self.fixture), str(self.base / "librocprofiler-sdk.so")],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(child.stdout.readline(), "ready\n")
            report = self.base / "wrapper-report.json"
            result = subprocess.run(["python3", str(CLI), "preflight", "--controller", str(self.lib),
                                     "--pid", str(child.pid), "--report", str(report)],
                                    env=dict(os.environ, LD_LIBRARY_PATH=str(self.base)),
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads(report.read_text())
            self.assertEqual(record["pid"], child.pid)
            self.assertEqual(record["controller"]["sha256"], hashlib.sha256(self.fixture.read_bytes()).hexdigest())
            self.assertIn(str(self.lib), [d["path"] for d in record["controller_process_dependencies"]])
            self.assertEqual(record["disposition"], "SNAPSHOT_ONLY_NOT_ATTACHMENT_AUTHORITY")
            original = report.read_bytes()
            again = subprocess.run(["python3", str(CLI), "preflight", "--controller", str(self.lib),
                                    "--pid", str(child.pid), "--report", str(report)], capture_output=True)
            self.assertNotEqual(again.returncode, 0)
            self.assertEqual(report.read_bytes(), original)
        finally:
            child.communicate(timeout=5)

    def test_wrapper_rejects_incomplete_or_malformed_controller_reports(self):
        """A zero exit and PASS label cannot replace complete identity evidence."""
        source = self.base / "report-fixture.c"
        source.write_text('#include <stdlib.h>\n#include <unistd.h>\n#include <string.h>\n'
                          'int preflight(unsigned p,int fd) { const char* s=getenv("REPORT_FIXTURE"); '
                          'return write(fd,s,strlen(s)) < 0; }\n')
        library = self.base / "report-fixture.so"
        subprocess.run(["gcc", "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
        obj = {"path": "/fixture.so", "sha256": "a" * 64, "build_id": "1234", "load_bias": 4096}
        valid = {"status": "PASS", "pid": os.getpid(), "target_mutated": False,
                 "start_time": "123", "controller": obj, "target": copy.deepcopy(obj),
                 "sdk": copy.deepcopy(obj), "attach_offset": 256, "detach_offset": 512}
        faults = [{"status": "PASS", "pid": os.getpid(), "target_mutated": False}]
        for key in valid:
            bad = copy.deepcopy(valid); del bad[key]; faults.append(bad)
        for key in ("controller", "target", "sdk"):
            for field in obj:
                bad = copy.deepcopy(valid); del bad[key][field]; faults.append(bad)
            for field, value in (("path", "relative"), ("sha256", "x" * 64),
                                 ("build_id", ""), ("load_bias", True)):
                bad = copy.deepcopy(valid); bad[key][field] = value; faults.append(bad)
        for key, value in (("pid", True), ("start_time", 123), ("start_time", "bad"),
                           ("attach_offset", True), ("detach_offset", -1),
                           ("attach_offset", 2**64), ("target_mutated", 0)):
            bad = copy.deepcopy(valid); bad[key] = value; faults.append(bad)
        for field, value in (("sha256", "b" * 64), ("build_id", "5678")):
            bad = copy.deepcopy(valid); bad["target"][field] = value; faults.append(bad)
        faults.extend([[], None])
        # Each payload changes one predicate while preserving all other evidence.
        for key, value in (("pid", float(os.getpid())), ("pid", os.getpid() + 1),
                           ("start_time", "0"), ("start_time", str(2**64))):
            bad = copy.deepcopy(valid); bad[key] = value; faults.append(bad)
        for owner in ("controller", "target", "sdk"):
            for field, value in (("path", "/fixture\0.so"), ("load_bias", -1),
                                 ("load_bias", 2**64)):
                bad = copy.deepcopy(valid); bad[owner][field] = value; faults.append(bad)
        for field, value in (("sha256", "a" * 63), ("sha256", "a" * 65),
                             ("build_id", "abc")):
            bad = copy.deepcopy(valid); bad["sdk"][field] = value; faults.append(bad)
        for symbol in ("attach_offset", "detach_offset"):
            bad = copy.deepcopy(valid); bad[symbol] = 0; faults.append(bad)
            for owner in ("controller", "target"):
                bad = copy.deepcopy(valid)
                bad["detach_offset" if symbol == "attach_offset" else "attach_offset"] = 1
                bad[owner]["load_bias"] = 2**64 - bad[symbol]
                faults.append(bad)
        controls = [copy.deepcopy(valid)]
        for key, value in (("start_time", "1"), ("start_time", str(2**64 - 1)),
                           ("attach_offset", 1), ("detach_offset", 1)):
            good = copy.deepcopy(valid); good[key] = value; controls.append(good)
        for owner in ("controller", "target", "sdk"):
            good = copy.deepcopy(valid); good[owner]["load_bias"] = 0; controls.append(good)
            good = copy.deepcopy(valid)
            good[owner]["load_bias"] = 2**64 - 1 - (512 if owner != "sdk" else 0)
            controls.append(good)
        for symbol in ("attach_offset", "detach_offset"):
            good = copy.deepcopy(valid); good[symbol] = 2**64 - 1
            good["controller"]["load_bias"] = good["target"]["load_bias"] = 0
            controls.append(good)
        good = copy.deepcopy(valid); good["sdk"]["build_id"] = "ab"; controls.append(good)
        for number, payload in enumerate(controls):
            with self.subTest(valid_boundary=number):
                report = self.base / f"valid-boundary-{number}.json"
                result = subprocess.run(["python3", str(CLI), "preflight", "--controller", str(library),
                                         "--pid", str(os.getpid()), "--report", str(report)],
                                        env=dict(os.environ, REPORT_FIXTURE=json.dumps(payload)),
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(report.read_text())["status"], "PASS")
        for number, payload in enumerate(faults):
            with self.subTest(number=number, payload=payload):
                report = self.base / f"invalid-report-{number}.json"
                result = subprocess.run(["python3", str(CLI), "preflight", "--controller", str(library),
                                         "--pid", str(os.getpid()), "--report", str(report)],
                                        env=dict(os.environ, REPORT_FIXTURE=json.dumps(payload)),
                                        capture_output=True, text=True, timeout=10)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertFalse(report.exists())

    def test_full_gate_dispatches_focused_suite(self):
        """Deleting the preflight registration must remove an observed dispatch."""
        scratch = self.base / "dispatch"
        (scratch / "scripts").mkdir(parents=True)
        (scratch / "bin").mkdir()
        shutil.copy2(ROOT / "scripts/agent-preflight.sh", scratch / "scripts/agent-preflight.sh")
        log = scratch / "python-dispatches"
        shim = scratch / "bin/python3"
        # Observe the actual shell dispatch boundary without recursively running
        # this full suite and every unrelated repository gate inside itself.
        shim.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$DISPATCH_LOG"\nexit 0\n')
        shim.chmod(0o755)
        subprocess.run(["bash", str(scratch / "scripts/agent-preflight.sh"), "--quiet"],
                       env=dict(os.environ, PATH=str(scratch / "bin") + ":" + os.environ["PATH"],
                                DISPATCH_LOG=str(log)), capture_output=True, timeout=30)
        self.assertIn("tests/scripts/test_rocprof_attach_preflight.py", log.read_text().splitlines())

    def test_post_seizure_recheck_releases_without_writes(self):
        """Deleting post-seizure revalidation reaches forbidden memory operations."""
        folder = self.copied()
        result, _ = self.run_live(folder=folder, mode="attach", fault="post-seizure")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("seized=1", result.stderr)
        # Linux ptrace ABI: SEIZE, INTERRUPT, DETACH. No injected call or write.
        self.assertEqual((folder / "seizure.log").read_text().splitlines(), ["16902", "16903", "17"])

    def test_checked_addresses_reach_attach_and_detach_registers(self):
        """Wrong address propagation or deleted registration calls lose the witness."""
        folder = self.copied()
        address_file = folder / "addresses.json"
        extra = '''import json
lib=ctypes.CDLL(sys.argv[1])
addresses=[ctypes.cast(getattr(lib,n),ctypes.c_void_p).value for n in
("rocprofiler_register_attach","rocprofiler_register_detach")]
open(''' + repr(str(address_file)) + ''',"w").write(json.dumps(addresses))
print("extra-ready",flush=True)'''
        result, _ = self.run_live(folder=folder, extra=extra, mode="attach", fault="simulate-injection")
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = json.loads(address_file.read_text())
        observed = [int(line) for line in (folder / "seizure.log").read_text().splitlines()
                    if int(line) > 0x100000]
        self.assertEqual(observed, expected)

    def test_malformed_sdk_elf_rejected(self):
        """Skipping ELF validation accepts a mapped file with an invalid ELF header."""
        folder = self.copied()
        def mutation(child, path):
            with (folder / "librocprofiler-sdk.so").open("r+b") as f:
                f.write(b"BAD!")
        result, _ = self.run_live(folder=folder, mutate=mutation)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ELF", result.stderr)


if __name__ == "__main__":
    unittest.main()
