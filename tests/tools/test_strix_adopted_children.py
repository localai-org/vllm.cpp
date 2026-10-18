"""Real process ownership at the qualification Adapter boundary (#3108)."""
import json
import ctypes
import platform
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]

TORCH = '''from types import SimpleNamespace
version=SimpleNamespace(hip="fixture",cuda=None)
cuda=SimpleNamespace(is_available=lambda:True,get_device_properties=lambda _:SimpleNamespace(gcnArchName="gfx1151"))
'''
TOKENIZER = '''class AutoTokenizer:
    @classmethod
    def from_pretrained(cls,*a,**kw): return cls()
    def encode(self,*a,**kw): return [10]
'''
ENGINE = '''import asyncio, os, time
from pathlib import Path
class Engine:
    def __init__(self,**kwargs):
        self.kwargs=kwargs
        self.loop=asyncio.new_event_loop()
        read,write=os.pipe()
        child=os.fork()
        if child==0:
            os.close(read)
            import multiprocessing
            semaphore=multiprocessing.get_context("spawn").Semaphore()
            tracker=Path(f"/proc/self/task/{os.getpid()}/children").read_text().strip()
            os.write(write,tracker.encode());os.close(write)
            os._exit(0)
        os.close(write)
        self.tracker=int(os.read(read,100));os.close(read)
        assert os.waitpid(child,0)[0]==child
        # The fixture supervisor owns any orphan the unpatched controller misses.
        # Wait before configure returns, never add a grace period to Adapter.close.
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            status=Path(f"/proc/{self.tracker}/stat").read_text().rsplit(")",1)[1].split()[0]
            if status=="Z":break
            time.sleep(.001)
        else:raise RuntimeError("tracker did not reach deterministic exited state")
    def get_server_info(self):
        return dict(self.kwargs,max_total_num_tokens=8192,tracker_pid=self.tracker)
    def shutdown(self):pass
'''

BASIC = '''import json, os, signal, sys, time
from pathlib import Path
mode=os.environ.get("CHILD_CASE", "none")
for line in sys.stdin:
    command=json.loads(line)
    result=dict(schema=1,id=command["id"],status="ok")
    if command["command"]=="configure":
        child=0
        if mode in ("zero","zero-held","nonzero","signal","live","escaped"):
            child=os.fork()
            if child==0:
                if mode=="escaped":os.setsid()
                for fd in (0,1,2):os.close(fd)
                if mode=="live":
                    while True:signal.pause()
                if mode=="signal":os.kill(os.getpid(),signal.SIGTERM)
                os._exit(7 if mode=="nonzero" else 0)
            if mode!="live":
                deadline=time.monotonic()+5
                while Path(f"/proc/{child}/stat").read_text().rsplit(")",1)[1].split()[0]!="Z":
                    if time.monotonic()>deadline:raise RuntimeError("fixture child failed to exit")
                    time.sleep(.001)
        result["child"]=child
        print(json.dumps(result),flush=True)
        if mode=="early-exit":sys.exit(0)
    else:
        if mode=="timeout":time.sleep(60)
        if mode.startswith("shutdown-error"):result.update(status="error",error="injected shutdown failure")
        print(json.dumps(result),flush=True)
        if mode in ("shutdown-error-held","zero-held"):
            while not Path(os.environ["CHILD_GATE"]).exists():time.sleep(.001)
        sys.exit(9 if mode=="parent-error" else 0)
'''

DRIVER_IMPORTS = '''import ctypes, json, os, signal, subprocess, sys, threading, time
from pathlib import Path
from unittest import mock
from tools.bench.strix_four_engine import child_lifecycle as lifecycle
from tools.bench.strix_four_engine.qualify import Adapter
folder=Path(sys.argv[1])
def make(mode="none",timeout=1):
    output=folder / ("out-"+str(len(list(folder.glob("out-*")))))
    output.mkdir()
    record=dict(command=[sys.executable,str(folder/"basic.py")],
                environment=dict(variables=dict(os.environ,CHILD_CASE=mode,CHILD_GATE=str(folder/"release"))))
    return Adapter(record,output,100000,timeout)
def absent(pid):
    try:os.killpg(pid,0)
    except ProcessLookupError:return True
    return False
'''

ISOLATE = '''import ctypes,os,signal,sys,time
from pathlib import Path
libc=ctypes.CDLL(None,use_errno=True);libc.prctl.argtypes=[ctypes.c_int];libc.prctl.restype=ctypes.c_int
assert libc.prctl(36,ctypes.c_ulong(1),ctypes.c_ulong(0),ctypes.c_ulong(0),ctypes.c_ulong(0))==0
child=os.fork()
if child==0:os.execv(sys.executable,[sys.executable,sys.argv[1],sys.argv[2]])
deadline=time.monotonic()+15
while True:
    pid,status=os.waitpid(child,os.WNOHANG)
    if pid:break
    if time.monotonic()>=deadline:
        os.kill(child,signal.SIGKILL);pid,status=os.waitpid(child,0);break
    time.sleep(.005)
# All remaining children belong to this isolated fixture. Kill only its exact
# adopted PIDs, then reap; never signal the test runner's inherited process group.
deadline=time.monotonic()+3
while True:
    children=Path(f"/proc/self/task/{os.getpid()}/children").read_text().split()
    for value in children:
        try:os.kill(int(value),signal.SIGKILL)
        except ProcessLookupError:pass
    try:pid,ignored=os.waitpid(-1,os.WNOHANG)
    except ChildProcessError:break
    if time.monotonic()>=deadline:raise RuntimeError("fixture cleanup exhausted")
    if not pid:time.sleep(.005)
sys.exit(os.waitstatus_to_exitcode(status) if os.WIFEXITED(status) else 99)
'''

# Only this fixture supervisor uses an unrestricted wait: it owns exactly the
# forked controller and its adopted fixture children. Production must use -pgid.
SUPERVISOR = '''import contextlib, ctypes, json, os, sys
from pathlib import Path
libc=ctypes.CDLL(None,use_errno=True)
libc.prctl.argtypes=[ctypes.c_int];libc.prctl.restype=ctypes.c_int
assert libc.prctl(36,ctypes.c_ulong(1),ctypes.c_ulong(0),ctypes.c_ulong(0),ctypes.c_ulong(0))==0
read,write=os.pipe()
child=os.fork()
if child==0:
    os.close(read)
    from tools.bench.strix_four_engine.qualify import Adapter,PROMPTS
    try:
        from tools.bench.strix_four_engine.child_lifecycle import controller_lifecycle
    except ModuleNotFoundError:
        controller_lifecycle=contextlib.nullcontext
    report={}
    try:
        with controller_lifecycle():
            record=dict(command=[sys.executable,"-m","tools.bench.strix_four_engine.python_adapter"],
                        environment=dict(variables=dict(os.environ)))
            adapter=Adapter(record,Path(sys.argv[1]),100000,5)
            result=adapter.exchange(dict(command="configure",engine="patched SGLang",model="fixture",gguf="fixture",prompts=PROMPTS,prompt_ids=None))
            tracker=result["runtime_info"]["tracker_pid"]
            adapter.close()
            report=dict(ok=True,tracker_absent=not Path(f"/proc/{tracker}").exists(),
                        evidence=getattr(adapter,"lifecycle_evidence",None))
    except BaseException as error:report=dict(ok=False,error=str(error))
    os.write(write,json.dumps(report).encode());os.close(write);os._exit(0)
os.close(write);report=os.read(read,100000);os.close(read)
assert os.waitpid(child,0)[0]==child
while True:
    try:pid,status=os.waitpid(-1,os.WNOHANG)
    except ChildProcessError:break
    if pid==0:raise RuntimeError("fixture leaked live child")
print(report.decode())
'''


# CI-context exemption, recorded: the qualification runtime is bound to the
# measured Strix host. tools/bench/strix_four_engine/child_lifecycle.py
# (Native.__init__) refuses anything but Linux x86-64 glibc 2.39, and real
# qualification additionally requires the Strix lease. Off that host these
# tests cannot exercise the runtime they own, so the class SKIPS with this
# reason instead of failing a hundred cases deep; it still runs for real on
# the Strix x86-64 lane.
import platform

STRIX_RUNTIME = (sys.platform == "linux" and platform.machine() == "x86_64"
                 and platform.libc_ver()[1] == "2.39")


@unittest.skipUnless(
    STRIX_RUNTIME,
    "the adopted-children runtime refuses non-Linux-x86_64/glibc-2.39 hosts "
    "(tools/bench/strix_four_engine/child_lifecycle.py); no aarch64 port "
    "exists to exercise",
)
class AdoptedChildrenTests(unittest.TestCase):
    def driver(self, body):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            (folder / "basic.py").write_text(BASIC)
            (folder / "driver.py").write_text(DRIVER_IMPORTS+body)
            result=subprocess.run([sys.executable,"-c",ISOLATE,str(folder/"driver.py"),str(folder)],
                                  cwd=ROOT, env=dict(os.environ,PYTHONPATH=str(ROOT)),
                                  text=True,capture_output=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_real_python_adapter_tracker_is_reaped_before_group_absence(self):
        """Deleting Adapter adoption/reaping leaves a real tracker zombie."""
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name, source in (("torch.py", TORCH), ("transformers.py", TOKENIZER),
                                 ("sglang.py", ENGINE)):
                (folder / name).write_text(source)
            output = folder / "output"
            output.mkdir()
            result = subprocess.run([sys.executable, "-c", SUPERVISOR, str(output)],
                                    cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(folder) + os.pathsep + str(ROOT)),
                                    text=True, capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertTrue(report["ok"], report)
            self.assertTrue(report["tracker_absent"], report)

    def test_normal_zero_children_and_repeated_close_restore_actual_state(self):
        self.driver('''
with lifecycle.controller_lifecycle():
    for mode in ("none","zero"):
        a=make(mode); configured=a.exchange(dict(command="configure"))
        assert lifecycle.Native().state()==1
        a.close()
        assert a.process.returncode==0 and absent(a.process.pid)
        assert lifecycle.Native().state()==0 and a.lifecycle_evidence["restored"]
        if mode=="zero":
            assert a.lifecycle_evidence["adopted"]==[dict(pid=configured["child"],status=0,cleanup=False)]
        with mock.patch.object(os,"killpg",side_effect=AssertionError("repeated close touched group")):
            a.close()
''')

    def test_live_abnormal_parent_and_protocol_failures_remain_failures(self):
        self.driver('''
for mode,reason in (("live","live adopted"),("nonzero","abnormal adopted"),
                    ("signal","abnormal adopted"),("parent-error","teardown failed"),
                    ("shutdown-error","injected shutdown failure"),("timeout","command timeout")):
    with lifecycle.controller_lifecycle():
        a=make(mode,timeout=.1); a.exchange(dict(command="configure"))
        try:a.close()
        except Exception as error:assert reason in str(error),(mode,error)
        else:raise AssertionError("accepted "+mode)
        assert a.process.returncode is not None and absent(a.process.pid)
        assert lifecycle.Native().state()==0 and a.lifecycle_evidence["restored"]
''')

    def test_already_exited_parent_cannot_skip_shutdown_protocol(self):
        self.driver('''
with lifecycle.controller_lifecycle():
    a=make("early-exit");a.exchange(dict(command="configure"));a.process.wait(timeout=2)
    try:a.close()
    except Exception as error:assert "shutdown" in str(error),error
    else:raise AssertionError("missing shutdown acknowledgment accepted")
    assert lifecycle.Native().state()==0
''')

    def test_foreign_group_status_is_not_consumed_by_owned_drain(self):
        self.driver('''
with lifecycle.controller_lifecycle():
    a=make("zero");a.exchange(dict(command="configure"))
    foreign=os.fork()
    if foreign==0:os.setsid();os._exit(13)
    try:
        deadline=time.monotonic()+2
        while Path(f"/proc/{foreign}/stat").read_text().rsplit(")",1)[1].split()[0]!="Z":
            assert time.monotonic()<deadline;time.sleep(.001)
        try:a.close()
        except RuntimeError as error:assert "unresolved controller children" in str(error),error
        else:raise AssertionError("foreign child ownership accepted")
        assert lifecycle._controller.poisoned
        assert a.lifecycle_evidence["unresolved_children"][0]["pid"]==foreign
        assert os.waitpid(foreign,0)==(foreign,13<<8)
    finally:
        try:os.waitpid(foreign,os.WNOHANG)
        except ChildProcessError:pass
''')

    def test_escaped_adopted_child_is_refused_without_foreign_group_operations(self):
        self.driver('''
with lifecycle.controller_lifecycle():
    a=make("escaped");child=a.exchange(dict(command="configure"))["child"]
    real_kill=os.killpg;real_wait=os.waitpid
    def kill(group,sig):
        assert group==a.process.pid,"foreign group signaled"
        return real_kill(group,sig)
    def wait(pid,flags):
        assert pid in (a.process.pid,-a.process.pid),"foreign child reaped"
        return real_wait(pid,flags)
    with mock.patch.object(os,"killpg",side_effect=kill) as kills,mock.patch.object(os,"waitpid",side_effect=wait) as waits:
        try:a.close()
        except RuntimeError as error:assert "unresolved controller children" in str(error),error
        else:raise AssertionError("escaped child accepted")
    # Cleanup catches exceptions from syscall doubles. Inspect attempts afterward.
    assert kills.call_args_list and waits.call_args_list
    assert all(call.args[0]==a.process.pid for call in kills.call_args_list),kills.call_args_list
    assert all(call.args[0] in (a.process.pid,-a.process.pid) for call in waits.call_args_list),waits.call_args_list
    assert lifecycle._controller.poisoned
    assert a.lifecycle_evidence["unresolved_children"][0]["pid"]==child
    assert a.lifecycle_evidence["unresolved_children"][0]["pgid"]==child
    assert real_wait(child,0)==(child,0)
''')

    def test_echild_does_not_replace_group_absence(self):
        self.driver('''
with lifecycle.controller_lifecycle():
    a=make("zero");a.exchange(dict(command="configure"))
    real_wait=os.waitpid
    def wait(pid,flags):
        if pid<0:raise ChildProcessError()
        return real_wait(pid,flags)
    try:
        with mock.patch.object(os,"waitpid",side_effect=wait):
            a.close()
    except Exception as error:assert "descendants survived" in str(error),error
    else:raise AssertionError("ECHILD falsely proved group absence")
    # This injected syscall fault hid a real owned zombie; fixture restores wait.
    while True:
        try:pid,status=real_wait(-a.process.pid,os.WNOHANG)
        except ChildProcessError:break
        assert pid
    assert absent(a.process.pid)
''')

    def test_remaining_inventory_failures_are_not_absence(self):
        self.driver('''
import io
for fault in ("missing","unreadable","bound","count","malformed","drift","pid","ppid"):
    with lifecycle.controller_lifecycle():
        a=make("escaped" if fault in ("drift","pid","ppid") else "none")
        child=a.exchange(dict(command="configure"))["child"]
        original=Path.open;seen=0
        def opened(path,*args,**kwargs):
            global seen
            if str(path)==f"/proc/self/task/{os.getpid()}/children":
                if fault=="missing":raise FileNotFoundError("inventory missing")
                if fault=="unreadable":raise PermissionError("inventory unreadable")
                if fault=="bound":return io.StringIO(" "*65537)
                if fault=="count":return io.StringIO("1 "*4097)
                if fault=="malformed":return io.StringIO("-1")
            if fault in ("drift","pid","ppid") and str(path)==f"/proc/{child}/stat":
                seen+=1
                with original(path,*args,**kwargs) as source:value=source.read()
                if fault=="drift" and seen%2==0:
                    prefix,fields=value.rsplit(")",1);fields=fields.split()
                    fields[19]=str(int(fields[19])+1);value=prefix+") "+" ".join(fields)
                if fault=="pid":value=str(child+1)+" ("+value.split(" (",1)[1]
                if fault=="ppid":
                    prefix,fields=value.rsplit(")",1);fields=fields.split()
                    fields[1]="1";value=prefix+") "+" ".join(fields)
                return io.StringIO(value)
            return original(path,*args,**kwargs)
        with mock.patch.object(Path,"open",opened):
            try:a.close()
            except (OSError,RuntimeError) as error:
                expected={"bound":"bound exhausted","count":"invalid controller child inventory"}.get(fault,"inventory")
                assert expected in str(error),error
            else:raise AssertionError("inventory failure accepted: "+fault)
        assert lifecycle._controller.poisoned
        assert a.lifecycle_evidence["cleanup_errors"]
        assert lifecycle.Native().state()==0
        if child:assert os.waitpid(child,0)==(child,0)
''')

    def test_sigchld_changes_during_adapter_ownership_refuse_close(self):
        self.driver('''
with lifecycle.controller_lifecycle():
    a=make("none");a.exchange(dict(command="configure"))
    native=a.lifecycle.owner.native;original=native.signal_state
    with mock.patch.object(native,"signal_state",return_value=(None,1,0,None)):
        try:a.close()
        except RuntimeError as error:assert "SIGCHLD arrangement changed" in str(error),error
        else:raise AssertionError("changed SIGCHLD arrangement accepted")
    assert absent(a.process.pid)
    assert native.state()==0
''')

    def test_remaining_child_list_change_is_not_an_empty_inventory(self):
        self.driver('''
import io
with lifecycle.controller_lifecycle():
    a=make();a.exchange(dict(command="configure"));original=Path.open;calls=[0]
    def opened(path,*args,**kwargs):
        if str(path)==f"/proc/self/task/{os.getpid()}/children":
            calls[0]+=1
            return io.StringIO("" if calls[0]==1 else "999999")
        return original(path,*args,**kwargs)
    with mock.patch.object(Path,"open",opened):
        try:a.close()
        except RuntimeError as error:assert "inventory identity drift" in str(error),error
        else:raise AssertionError("changed child list accepted")
    assert calls[0]>=2
    assert lifecycle._controller.poisoned and lifecycle.Native().state()==0
''')

    def test_sigaction_query_failure_refuses_controller_before_launch(self):
        self.driver('''
lib=ctypes.CDLL(None,use_errno=True)
with mock.patch.object(lib,"sigaction",return_value=-1),mock.patch.object(lifecycle.ctypes,"CDLL",return_value=lib):
    with mock.patch.object(subprocess,"Popen",side_effect=AssertionError("launch after failed sigaction")) as launch:
        try:
            with lifecycle.controller_lifecycle():make()
        except OSError as error:assert "sigaction query failed" in str(error),error
        else:raise AssertionError("SIGCHLD query failure accepted")
    launch.assert_not_called()
assert lifecycle.Native().state()==0
''')

    def test_sigchld_change_after_controller_acquisition_refuses_launch(self):
        self.driver('''
with lifecycle.controller_lifecycle():
    native=lifecycle._controller.native
    with mock.patch.object(native,"signal_state",return_value=(None,1,0,None)):
        with mock.patch.object(subprocess,"Popen",side_effect=AssertionError("launch after signal arrangement drift")) as launch:
            try:make()
            except RuntimeError as error:assert "SIGCHLD arrangement changed" in str(error),error
            else:raise AssertionError("changed launch signal arrangement accepted")
        launch.assert_not_called()
    assert native.state()==0
''')

    def test_foreign_child_after_controller_acquisition_refuses_launch(self):
        self.driver('''
with lifecycle.controller_lifecycle():
    read,write=os.pipe();child=os.fork()
    if child==0:os.close(write);os.read(read,1);os._exit(0)
    os.close(read)
    try:
        with mock.patch.object(subprocess,"Popen",side_effect=AssertionError("launch with foreign child")) as launch:
            try:make()
            except RuntimeError as error:assert "pre-existing children" in str(error),error
            else:raise AssertionError("foreign child launch accepted")
        launch.assert_not_called()
        assert lifecycle.Native().state()==0
    finally:os.close(write);assert os.waitpid(child,0)==(child,0)
''')

    def test_constructor_failure_restores_and_preserves_original_error(self):
        self.driver('''
with lifecycle.controller_lifecycle():
    original=subprocess.Popen
    with mock.patch.object(subprocess,"Popen",side_effect=OSError("launch denied")):
        try:make()
        except OSError as error:assert "launch denied" in str(error)
        else:raise AssertionError("launch error lost")
    assert lifecycle.Native().state()==0
    a=make();a.exchange(dict(command="configure"));a.close()
''')

    def test_imported_nested_concurrent_and_threaded_owners_refuse(self):
        self.driver('''
try:make()
except RuntimeError as error:assert "explicit controller_lifecycle" in str(error)
else:raise AssertionError("implicit imported ownership accepted")
with lifecycle.controller_lifecycle():
    try:
        with lifecycle.controller_lifecycle():pass
    except RuntimeError:pass
    else:raise AssertionError("nested ownership accepted")
    a=make();a.exchange(dict(command="configure"))
    try:make()
    except RuntimeError:pass
    else:raise AssertionError("concurrent Adapter accepted")
    a.close()
errors=[]
def childthread():
    try:
        # Independently exercise main-thread identity, not the OS-thread count.
        with mock.patch.object(Path,"iterdir",return_value=iter([Path("/proc/self/task/fixture")])):
            with lifecycle.controller_lifecycle():pass
    except RuntimeError as error:errors.append(str(error))
t=threading.Thread(target=childthread);t.start();t.join()
assert errors and "main OS thread" in errors[0]
stop=threading.Event();ready=threading.Event()
def foreignthread():ready.set();stop.wait()
t=threading.Thread(target=foreignthread);t.start();ready.wait()
try:
    with lifecycle.controller_lifecycle():pass
except RuntimeError as error:assert "OS thread" in str(error)
else:raise AssertionError("competing OS thread accepted")
finally:stop.set();t.join()
with lifecycle.controller_lifecycle():
    child=os.fork()
    if child==0:
        try:a=make()
        except RuntimeError:os._exit(0)
        a.close();os._exit(5)
    assert os.waitpid(child,0)==(child,0),"forked controller reused inherited ownership"
try:make()
except RuntimeError as error:assert "explicit controller_lifecycle" in str(error)
else:raise AssertionError("released controller ownership remained reusable")
''')

    def test_platform_signal_autoreap_and_preexisting_children_refuse(self):
        self.driver('''
for target,value in (("platform","darwin"),("machine","aarch64")):
    patch=mock.patch.object(sys,"platform",value) if target=="platform" else mock.patch.object(lifecycle.platform,"machine",return_value=value)
    with patch:
        try:
            with lifecycle.controller_lifecycle():pass
        except RuntimeError:pass
        else:raise AssertionError("unsupported platform accepted")
lib=ctypes.CDLL(None,use_errno=True)
with mock.patch.object(lib,"gnu_get_libc_version",return_value=b"2.38"),mock.patch.object(lifecycle.ctypes,"CDLL",return_value=lib):
    try:
        with lifecycle.controller_lifecycle():pass
    except RuntimeError as error:assert "sigaction ABI" in str(error)
    else:raise AssertionError("unverified libc ABI accepted")
for handler in (signal.SIG_IGN,lambda *args:None):
    prior=signal.signal(signal.SIGCHLD,handler)
    try:
        with lifecycle.controller_lifecycle():pass
    except RuntimeError as error:assert "SIGCHLD" in str(error)
    else:raise AssertionError("competing handler accepted")
    finally:signal.signal(signal.SIGCHLD,prior)
# Use a native SA_NOCLDWAIT with SIG_DFL, which getsignal cannot detect.
native=lifecycle.Native();old=lifecycle.Sigaction();action=lifecycle.Sigaction();action.flags=2
native.lib.sigaction.argtypes=[ctypes.c_int,ctypes.POINTER(lifecycle.Sigaction),ctypes.POINTER(lifecycle.Sigaction)]
assert native.lib.sigaction(signal.SIGCHLD,ctypes.byref(action),ctypes.byref(old))==0
try:
    with lifecycle.controller_lifecycle():pass
except RuntimeError as error:assert "SIGCHLD" in str(error)
else:raise AssertionError("SA_NOCLDWAIT accepted")
finally:assert native.lib.sigaction(signal.SIGCHLD,ctypes.byref(old),None)==0
callback=ctypes.CFUNCTYPE(None,ctypes.c_int)(lambda _:None)
action=lifecycle.Sigaction();action.handler=ctypes.cast(callback,ctypes.c_void_p).value
assert native.lib.sigaction(signal.SIGCHLD,ctypes.byref(action),ctypes.byref(old))==0
try:
    with lifecycle.controller_lifecycle():pass
except RuntimeError as error:assert "SIGCHLD" in str(error)
else:raise AssertionError("native SIGCHLD handler accepted")
finally:assert native.lib.sigaction(signal.SIGCHLD,ctypes.byref(old),None)==0
native.set_state(1)
try:
    with lifecycle.controller_lifecycle():pass
except RuntimeError as error:assert "unowned" in str(error)
else:raise AssertionError("unowned subreaper accepted")
finally:native.set_state(0)
read,write=os.pipe();foreign=os.fork()
if foreign==0:os.close(write);os.read(read,1);os._exit(0)
os.close(read)
try:
    with lifecycle.controller_lifecycle():pass
except RuntimeError as error:assert "pre-existing" in str(error)
else:raise AssertionError("foreign children accepted at acquisition")
finally:os.close(write);os.waitpid(foreign,0)
''')

    def test_wait_group_flags_and_parent_status_ownership(self):
        self.driver('''
with lifecycle.controller_lifecycle():
    a=make("zero-held");a.exchange(dict(command="configure"));real=os.waitpid
    original_wait=a.process.wait
    def parent_wait(*args,**kwargs):
        (folder/"release").touch()
        return original_wait(*args,**kwargs)
    def checked(pid,flags):
        if pid<0:
            assert pid==-a.process.pid and flags==os.WNOHANG
            assert a.process.returncode==0,"drain stole Popen status"
        return real(pid,flags)
    with mock.patch.object(os,"waitpid",side_effect=checked),mock.patch.object(a.process,"wait",side_effect=parent_wait):a.close()
    assert a.process.returncode==0 and a.lifecycle_evidence["adopted"]
''')

    def test_native_sigaction_layout_matches_compiled_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            source=folder/"abi.c"
            source.write_text('#include <signal.h>\n#include <stddef.h>\n#include <stdio.h>\n'
                              'int main(void){printf("%zu %zu %zu %zu %zu %d\\n",sizeof(struct sigaction),'
                              'offsetof(struct sigaction,sa_handler),offsetof(struct sigaction,sa_mask),'
                              'offsetof(struct sigaction,sa_flags),offsetof(struct sigaction,sa_restorer),SA_NOCLDWAIT);}\n')
            subprocess.run(["gcc",str(source),"-o",str(folder/"abi")],check=True)
            actual=subprocess.check_output([str(folder/"abi")],text=True).strip()
            self.assertEqual(actual,"152 0 8 136 144 2")
            from tools.bench.strix_four_engine.child_lifecycle import Sigaction
            self.assertEqual([ctypes.sizeof(Sigaction), Sigaction.handler.offset,
                              Sigaction.mask.offset, Sigaction.flags.offset, Sigaction.restorer.offset],
                             [152, 0, 8, 136, 144])

    def test_acquisition_syscall_and_verification_fail_before_launch(self):
        self.driver('''
for fail_option in (36,37):
    with lifecycle.controller_lifecycle():
        native=lifecycle._controller.native;original=native.lib.prctl
        def failed(option,*args):
            if option==fail_option and (option==37 or args[0].value==1):return -1
            return original(option,*args)
        with mock.patch.object(native.lib,"prctl",side_effect=failed), mock.patch.object(subprocess,"Popen",side_effect=AssertionError("launched after failed acquisition")):
            try:make()
            except OSError as error:assert "prctl" in str(error)
            else:raise AssertionError("failed prctl accepted")
        assert native.state()==0
with lifecycle.controller_lifecycle():
    native=lifecycle._controller.native
    with mock.patch.object(native,"state",side_effect=[0,0,0]),mock.patch.object(subprocess,"Popen",side_effect=AssertionError("launched without verification")):
        try:make()
        except RuntimeError as error:assert "acquisition verification" in str(error)
        else:raise AssertionError("unverified acquisition accepted")
    assert native.state()==0
''')

    def test_restoration_failure_is_retained_and_poisoned_owner_cannot_relaunch(self):
        self.driver('''
with lifecycle.controller_lifecycle():
    a=make();a.exchange(dict(command="configure"))
    native=lifecycle._controller.native;original=native.lib.prctl
    def failed(option,*args):
        if option==36 and args[0].value==0:return -1
        return original(option,*args)
    try:
        with mock.patch.object(native.lib,"prctl",side_effect=failed):
            try:a.close()
            except RuntimeError as error:assert "restoration failed" in str(error)
            else:raise AssertionError("restoration failure accepted")
        assert not a.lifecycle_evidence["restored"] and a.lifecycle_evidence["cleanup_errors"]
        native.set_state(0)
        try:make()
        except RuntimeError as error:assert "unresolved" in str(error)
        else:raise AssertionError("poisoned owner reused")
    finally:native.set_state(0)
with lifecycle.controller_lifecycle():pass
''')

    def test_restoration_verification_and_original_error_are_preserved(self):
        self.driver('''
with lifecycle.controller_lifecycle():
    a=make("shutdown-error");a.exchange(dict(command="configure"))
    native=lifecycle._controller.native
    with mock.patch.object(native,"state",return_value=1):
        try:a.close()
        except Exception as error:
            assert "injected shutdown failure" in str(error)
            assert any("restoration verification" in item for item in error.lifecycle_evidence["cleanup_errors"])
        else:raise AssertionError("original failure lost")
    assert native.state()==0 and not a.lifecycle_evidence["restored"]
''')

    def test_cleanup_signal_error_does_not_skip_direct_parent_wait(self):
        self.driver('''
with lifecycle.controller_lifecycle():
    a=make("shutdown-error-held");a.exchange(dict(command="configure"))
    real=os.killpg
    def denied(pid,sig):
        if sig==signal.SIGKILL:
            (folder/"release").touch()
            raise PermissionError("fixture kill denied")
        return real(pid,sig)
    with mock.patch.object(os,"killpg",side_effect=denied):
        try:a.close()
        except Exception as error:
            assert "injected shutdown failure" in str(error)
            assert any("kill denied" in item for item in error.lifecycle_evidence["cleanup_errors"])
        else:raise AssertionError("original failure lost")
    assert a.process.returncode==0,"cleanup skipped Popen wait"
    assert lifecycle.Native().state()==0
''')

    def test_transport_cleanup_error_does_not_replace_original_error(self):
        self.driver('''
with lifecycle.controller_lifecycle():
    a=make("shutdown-error");a.exchange(dict(command="configure"))
    actual=a.stderr
    class FailedClose:
        def close(self):actual.close();raise OSError("diagnostic close failed")
        def write(self,*args):return actual.write(*args)
        def flush(self):return actual.flush()
    a.stderr=FailedClose()
    try:a.close()
    except Exception as error:
        assert "injected shutdown failure" in str(error),error
        assert any("diagnostic close failed" in x for x in error.lifecycle_evidence["cleanup_errors"])
    else:raise AssertionError("original failure lost")
    assert lifecycle.Native().state()==0
''')

    def test_context_cleanup_preserves_original_exception(self):
        self.driver('''
try:
    with lifecycle.controller_lifecycle():
        a=make();a.exchange(dict(command="configure"))
        raise ValueError("original controller error")
except ValueError as error:assert "original controller error" in str(error)
else:raise AssertionError("original context error lost")
assert absent(a.process.pid) and lifecycle.Native().state()==0
assert a.closed and a.stderr.closed,"context did not retire transport ownership"
with mock.patch.object(os,"killpg",side_effect=AssertionError("abandoned close touched group")):
    a.close()
''')

    def test_drain_child_count_and_deadline_are_bounded(self):
        self.driver('''
for kind in ("count","time"):
    with lifecycle.controller_lifecycle():
        a=make();a.exchange(dict(command="configure"));real=os.waitpid
        clock=[time.monotonic()];calls=[0]
        def wait(pid,flags):
            if pid>=0:return real(pid,flags)
            calls[0]+=1
            if calls[0]>9000:raise AssertionError("past child count budget")
            if kind=="time":clock[0]+=2
            return 999999,0
        with mock.patch.object(os,"waitpid",side_effect=wait),mock.patch.object(lifecycle.time,"monotonic",side_effect=lambda:clock[0]):
            try:a.close()
            except RuntimeError as error:assert "drain bound exhausted" in str(error)
            else:raise AssertionError("unbounded child drain accepted")
        assert calls[0]<=8192 and len(a.lifecycle_evidence["adopted"])<=8192
        if kind=="time":assert calls[0]==4,"one-second drain/five-second cleanup limits changed"
        assert lifecycle.Native().state()==0
''')

    def test_cleanup_deadline_cannot_turn_a_live_child_into_success(self):
        self.driver('''
with lifecycle.controller_lifecycle():
    a=make("shutdown-error");a.exchange(dict(command="configure"));real=os.waitpid
    clock=[time.monotonic()];calls=[0]
    def wait(pid,flags):
        if pid>=0:return real(pid,flags)
        calls[0]+=1;clock[0]+=6
        return 0,0
    with mock.patch.object(os,"waitpid",side_effect=wait),mock.patch.object(lifecycle.time,"monotonic",side_effect=lambda:clock[0]):
        try:a.close()
        except Exception as error:
            assert "injected shutdown failure" in str(error)
            assert any("bound exhausted" in item for item in error.lifecycle_evidence["cleanup_errors"])
        else:raise AssertionError("cleanup changed original failure")
    assert calls[0]==1 and lifecycle.Native().state()==0
''')


if __name__ == "__main__":
    unittest.main()
