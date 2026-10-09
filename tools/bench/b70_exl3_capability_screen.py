#!/usr/bin/env python3
"""Freeze R11 held-out inputs and check native answers; no inference fallback."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

def manifest():
    tasks=[
      {"id":"code-merge","kind":"code","function":"merge_intervals",
       "prompt":"Write Python 3 function merge_intervals(intervals). Input is a list of integer [start,end] pairs with start<=end. Return sorted disjoint merged intervals as lists. Merge overlaps and touching endpoints. Empty input returns []. Do not mutate inputs. Use no imports. Return only Python code, without Markdown.",
       "cases":[{"args":[[]],"expected":[]},{"args":[[[7,9],[1,3],[3,5],[2,2],[-4,-1],[8,12]]],"expected":[[-4,-1],[1,5],[7,12]]},{"args":[[[5,5],[5,5],[0,10],[12,14]]],"expected":[[0,10],[12,14]]},{"args":[[[10,12],[2,4],[4,10]]],"expected":[[2,12]]}]},
      {"id":"code-brackets","kind":"code","function":"balanced_brackets",
       "prompt":"Write Python 3 function balanced_brackets(text) returning a bool. Check whether all (), [] and {} brackets are balanced and properly nested. Ignore other characters. Empty input is balanced. Use no imports. Return only Python code, without Markdown.",
       "cases":[{"args":[s],"expected":b} for s,b in [("",True),("a{b[(c)]}d",True),("([)]",False),("x]",False),("{{[",False),("α(β[γ]{δ})",True),("{}[]()",True)]]},
      {"id":"code-group","kind":"code","function":"stable_group_by",
       "prompt":"Write Python 3 function stable_group_by(records, key). Each record is a dict containing key, whose values are strings. Return a list of [value, list_of_records] groups in order of each value's first appearance. Preserve record order within groups and do not mutate inputs. Empty records returns []. Use no imports. Return only Python code, without Markdown.",
       "cases":[{"args":[[],"project"],"expected":[]},{"args":[[{"p":"b","v":2},{"p":"a","v":9},{"p":"b","v":1},{"p":"c","v":0}],"p"],"expected":[["b",[{"p":"b","v":2},{"p":"b","v":1}]],["a",[{"p":"a","v":9}]],["c",[{"p":"c","v":0}]]]},{"args":[[{"zone":"","id":3},{"zone":"west","id":4},{"zone":"","id":5}],"zone"],"expected":[["",[{"zone":"","id":3},{"zone":"","id":5}]],["west",[{"zone":"west","id":4}]]]}]},
      {"id":"code-window","kind":"code","function":"max_window_sum",
       "prompt":"Write Python 3 function max_window_sum(values, width). Return the largest sum among contiguous windows of exactly width elements. Values are integers, possibly negative. Raise ValueError if width<=0 or width>len(values). Do not mutate input. Use a sliding window in O(n) time, no imports. Return only Python code, without Markdown.",
       "cases":[{"args":[[-7,-2,-9,-3],2],"expected":-9},{"args":[[4,-1,2,8,-6,5],3],"expected":9},{"args":[[3,-4,2],1],"expected":3},{"args":[[3,-4,2],3],"expected":1},{"args":[[],1],"exception":"ValueError"},{"args":[[1,2],0],"exception":"ValueError"}]}]
    for target in [173,947]:
        lines=[]
        for i in range(1024):
            code=hashlib.sha256(f"heldout-r11-record-{i}-b703".encode()).hexdigest()[:12].upper()
            lines.append(f"Record {i:04d}: asset atlas-{i:04d}; access_code {code}; owner team-{i%37:02d}; zone sector-{i%19:02d}. Notes: archived inventory entry, verified by the warehouse clerk during the autumn stock review.")
        expected=hashlib.sha256(f"heldout-r11-record-{target}-b703".encode()).hexdigest()[:12].upper()
        tasks.append({"id":f"retrieval-{target}","kind":"retrieval","prompt":"Find the requested access_code in this archive. Ignore other records.\n\n"+"\n".join(lines)+f"\n\nQuestion: What is the access_code for Record {target:04d}? Return only its exact 12-character code.","min_prompt_tokens":32768,"max_prompt_tokens":65536,"max_tokens":64,"expected":expected})
    schema={"type":"object","properties":{"priority":{"type":"string","enum":["low","medium","high"]},"ticket":{"type":"integer"},"requires_callback":{"type":"boolean"}},"required":["priority","ticket","requires_callback"],"additionalProperties":False}
    tasks.append({"id":"structured-ticket","kind":"structured","max_tokens":160,"prompt":"Return a JSON object with priority, ticket and requires_callback. Ticket number is 731, priority high. The customer explicitly requested a callback. Return only the object.","json_schema":schema,"expected":{"priority":"high","ticket":731,"requires_callback":True}})
    tasks.append({"id":"tool-reservation","kind":"structured","max_tokens":160,"prompt":'Return only a JSON tool invocation for reserve_room. Use keys "name" and "arguments". Arguments must contain room "Birch", date "2027-02-16", start "14:30", duration_minutes 45 and attendees 6.',"json_schema":{"type":"object","properties":{"name":{"type":"string","enum":["reserve_room"]},"arguments":{"type":"object","properties":{"room":{"type":"string"},"date":{"type":"string"},"start":{"type":"string"},"duration_minutes":{"type":"integer"},"attendees":{"type":"integer"}},"required":["room","date","start","duration_minutes","attendees"],"additionalProperties":False}},"required":["name","arguments"],"additionalProperties":False},"expected":{"name":"reserve_room","arguments":{"room":"Birch","date":"2027-02-16","start":"14:30","duration_minutes":45,"attendees":6}}})
    for task in tasks:
        task.setdefault("max_tokens",512)
        task.setdefault("max_prompt_tokens",1024)
    return {"schema":"b70-exl3-r11-capability-v1","system":"Follow the user's task precisely. Give only the requested answer. Do not include reasoning or commentary.","tasks":tasks}

CODE_WORKER=r"""
import ast,copy,json,resource,sys
resource.setrlimit(resource.RLIMIT_CPU,(2,2))
resource.setrlimit(resource.RLIMIT_AS,(256*1024*1024,256*1024*1024))
resource.setrlimit(resource.RLIMIT_FSIZE,(0,0))
job=json.load(sys.stdin);tree=ast.parse(job["code"])
for n in tree.body:
 if not isinstance(n,ast.FunctionDef) and not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)):
  raise ValueError("only function definitions/docstrings are permitted")
for n in ast.walk(tree):
 if isinstance(n,(ast.Import,ast.ImportFrom,ast.ClassDef,ast.Global,ast.Nonlocal)):
  raise ValueError("unsupported construct")
 if isinstance(n,ast.Attribute) and n.attr.startswith("_"):
  raise ValueError("private attributes not permitted")
 if isinstance(n,ast.Name) and n.id.startswith("__"):
  raise ValueError("private names not permitted")
allowed={k:getattr(__import__("builtins"),k) for k in ("len","range","enumerate","zip","min","max","sum","sorted","reversed","abs","all","any","list","dict","tuple","set","str","int","float","bool","isinstance","ValueError","IndexError","TypeError")}
scope={"__builtins__":allowed};exec(compile(tree,"<model-output>","exec"),scope)
fn=scope[job["function"]];out=[]
for case in job["cases"]:
 args=copy.deepcopy(case["args"]);before=copy.deepcopy(args)
 try: value=fn(*args);entry={"value":value}
 except Exception as exc: entry={"exception":type(exc).__name__}
 entry["input_unchanged"]=args==before;out.append(entry)
print(json.dumps(out))
"""

def clean(text):
    # Only complete wrappers may be stripped; truncated code remains a failure.
    text=re.sub(r"^\s*<think>.*?</think>\s*","",text,flags=re.S)
    fence=chr(96)*3
    match=re.fullmatch(r"\s*"+fence+r"(?:python|py)?\s*\n(.*?)\n"+fence+r"\s*",text,flags=re.S)
    return match.group(1) if match else text.strip()

def same(a,b):
    if type(a) is not type(b): return False
    if isinstance(a,list): return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a,dict): return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    return a==b

def check(task,answer):
    if answer.get("finish_reason")=="length": return {"passed":False,"reason":"output_limit"}
    if answer.get("finish_reason")!="stop":
        return {"passed":False,"reason":"generation_"+str(answer.get("finish_reason"))}
    text=clean(answer["text"])
    if task["kind"]=="code":
        job={"code":text,"function":task["function"],"cases":task["cases"]}
        try: child=subprocess.run([sys.executable,"-I","-c",CODE_WORKER],input=json.dumps(job),text=True,capture_output=True,timeout=4)
        except subprocess.TimeoutExpired: return {"passed":False,"reason":"code_timeout"}
        if child.returncode: return {"passed":False,"reason":"code_execution","detail":child.stderr[-2000:]}
        try: actual=json.loads(child.stdout)
        except json.JSONDecodeError: return {"passed":False,"reason":"invalid_code_worker_output"}
        checks=[]
        for case,got in zip(task["cases"],actual):
            ok=got["input_unchanged"]
            ok &= (got.get("exception")==case["exception"]) if "exception" in case else ("value" in got and same(got["value"],case["expected"]))
            checks.append(bool(ok))
        return {"passed":len(checks)==len(task["cases"]) and all(checks),"case_checks":checks,"actual":actual}
    if task["kind"]=="retrieval": return {"passed":text==task["expected"],"actual":text,"expected":task["expected"]}
    try: actual=json.loads(text)
    except json.JSONDecodeError: return {"passed":False,"reason":"invalid_json","actual":text}
    return {"passed":same(actual,task["expected"]),"actual":actual,"expected":task["expected"]}

def self_check():
    task=next(t for t in manifest()["tasks"] if t["id"]=="code-window")
    correct="def max_window_sum(values, width):\n if width<=0 or width>len(values): raise ValueError()\n s=sum(values[:width]); best=s\n for i in range(width,len(values)):\n  s+=values[i]-values[i-width]; best=max(best,s)\n return best\n"
    assert check(task,{"text":correct,"finish_reason":"stop"})["passed"]
    assert not check(task,{"text":"def max_window_sum(values,width):\n return 0","finish_reason":"stop"})["passed"]
    assert not check(task,{"text":correct,"finish_reason":"length"})["passed"]
    j=next(t for t in manifest()["tasks"] if t["id"]=="structured-ticket")
    assert check(j,{"text":json.dumps(j["expected"]),"finish_reason":"stop"})["passed"]
    assert not check(j,{"text":json.dumps(dict(j["expected"],requires_callback=1)),"finish_reason":"stop"})["passed"]
    assert not check(j,{"text":"{","finish_reason":"stop"})["passed"]
    assert not check(j,{"text":json.dumps(j["expected"]),"finish_reason":"error"})["passed"]
    print("7/7 focused grader controls passed")

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("prepare");p.add_argument("--out",type=Path,required=True)
    v=sub.add_parser("validate");v.add_argument("--tasks",type=Path,required=True);v.add_argument("--outputs",type=Path,required=True);v.add_argument("--out",type=Path,required=True)
    sub.add_parser("self-check");args=parser.parse_args()
    if args.command=="self-check": self_check();return 0
    if args.command=="prepare":
        args.out.mkdir(parents=True,exist_ok=True);p=args.out/"tasks-v1.json"
        if p.exists(): raise FileExistsError(p)
        p.write_text(json.dumps(manifest(),indent=2)+"\n")
        print(json.dumps({"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"tasks":8}));return 0
    tasks=json.loads(args.tasks.read_text());outputs=json.loads(args.outputs.read_text());byid={t["id"]:t for t in tasks["tasks"]};records=[]
    for answer in outputs["tasks"]:
        records.append({"id":answer["id"],"kind":byid[answer["id"]]["kind"],**check(byid[answer["id"]],answer)})
    result={"schema":"b70-exl3-r11-capability-check-v1","tasks_sha256":hashlib.sha256(args.tasks.read_bytes()).hexdigest(),"outputs_sha256":hashlib.sha256(args.outputs.read_bytes()).hexdigest(),"completed":len(records),"required":len(byid),"passed":sum(r["passed"] for r in records),"results":records}
    result["suite_complete"]=len(records)==len(byid) and {r["id"] for r in records}==set(byid)
    if args.out.exists(): raise FileExistsError(args.out)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ["completed","required","passed"]}))
    return 0 if result["suite_complete"] and all(r["passed"] for r in records) else 1
if __name__=="__main__": raise SystemExit(main())
