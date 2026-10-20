"""Offline independent Jinja2 oracle; no C++ output is read."""
import copy
import hashlib
import json
from pathlib import Path
import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

root = Path(__file__).resolve().parent
source = root.parent / 'gemma4_e4b_google/chat_template.jinja'
assert hashlib.sha256(source.read_bytes()).hexdigest() == '0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5'
env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=False, autoescape=False)
def fail(message):
    raise ValueError(message)
env.globals['raise_exception'] = fail
template = env.from_string(source.read_text())
rows = []
def add(name, messages, tools=None, **kwargs):
    for thinking in [None, False, True]:
        for generation in [False, True]:
            kw = dict(kwargs)
            if thinking is not None:
                kw['enable_thinking'] = thinking
            request = dict(model='google/gemma-4-E4B-it', messages=messages, chat_template_kwargs=kw)
            if tools:
                request['tools'] = tools
            native = copy.deepcopy(messages)
            for m in native:
                if isinstance(m.get('content'), list):
                    m['content'] = [dict(type='text', text=p) if isinstance(p,str) else p for p in m['content']]
                for tc in m.get('tool_calls', []):
                    tc['function']['arguments'] = json.loads(tc['function']['arguments'] or '{}') or {}
            expected = template.render(messages=native, tools=tools or [], bos_token='<bos>', eos_token='<eos>', add_generation_prompt=generation, **kw)
            rows.append(dict(name=f'{name}/{thinking}/{generation}',request=request,generation=generation,expected=expected))
def msg(role,content,**extra):
    return dict(role=role,content=content,**extra)
def call(id,name,args):
    return dict(id=id,type='function',function=dict(name=name,arguments=json.dumps(args,ensure_ascii=False)))
text=lambda s:dict(type='text',text=s)
media=[dict(type='image',image='opaque'),dict(type='image_url',image_url={'url':'opaque','detail':'high'}),dict(type='audio',audio={'opaque':[1,2]}),dict(type='input_audio',input_audio={'data':'AA==','format':'wav'}),dict(type='video',video=[{'timestamp':1,'frame':'opaque'}])]
add('strings',[msg('user','  Héllo\n'),msg('assistant','Hi'),msg('user','next')])
for role in ['system','developer']:
    add(role,[msg(role,'  rules  '),msg('user','hello')])
    add(role+'-parts',[msg(role,[text(' first '),text(' second ')]),msg('user','hello')])
add('assistant-continuation',[msg('user','hi'),msg('assistant','first'),msg('assistant','second')])
add('empty',[msg('user',[])])
add('bare-parts',[msg('user',[' a ',' b '])])
for p in media:
    add(p['type'],[msg('user',[text('before'),p,text('after')])])
add('mixed',[msg('user',[text('one'),*media,text('two'),media[0],text('three')])])
tools=[dict(type='function',function=dict(name='lookup',description='Find "é"\n',parameters=dict(type='object',properties={'q':{'type':'string'},'options':{'type':'object','properties':{'flag':{'type':'boolean'}}}},required=['q']))),dict(type='function',function=dict(name='other',description='Other',parameters={'type':'object','properties':{}}))]
calls=[call('a','lookup',{'q':'quote" slash\\ newline\n é','options':{'z':[True,False,None,2,{'nested':'x'}],'a':3}}),call('b','other',{})]
add('tools-pending',[msg('user','go'),msg('assistant',None,tool_calls=calls)],tools)
conversation=[msg('user','go'),msg('assistant',None,tool_calls=calls,reasoning='latest thought'),msg('tool','answer',tool_call_id='a'),msg('tool','other answer',tool_call_id='b'),msg('assistant','done')]
add('tools-continuation',conversation,tools)
add('tool-media',[msg('user','go'),msg('assistant',None,tool_calls=calls),msg('tool',[text('a'),*media,text('b')],tool_call_id='a'),msg('tool','second',tool_call_id='b')],tools)
add('history',conversation+[msg('user','again'),msg('assistant','<|channel>thought\nhidden<channel|> visible',reasoning_content='new thought')],tools)
add('preserve',conversation+[msg('user','again')],tools,preserve_thinking=True)
add('reasoning-precedence',[msg('user','x'),msg('assistant','y',reasoning='primary',reasoning_content='secondary')])
# F1: native schema mappings, including recursive calls and mapping-only branches.
empty_array = dict(type='array', items={})
schemas = {
    'empty-items': empty_array,
    'nested-empty-items': dict(type='object', properties={'child': empty_array}),
    'array-object-empty-items': dict(type='array', items=dict(type='object', properties={'child': empty_array})),
    'object-empty-properties': dict(type='object', properties={}),
    'array-empty-properties': dict(type='array', items=dict(type='object', properties={})),
    'nonempty-items': dict(type='array', items=dict(type='string')),
    'empty-enum': dict(type='string', enum=[]),
    'empty-required': dict(type='object', properties={}, required=[]),
}
for name, schema in schemas.items():
    add('schema-' + name, [msg('user', 'hello')], [dict(type='function', function=dict(
        name='f', parameters=dict(type='object', properties={'x': schema})))])
for name, parameters in [('empty-parameters', {}), ('empty-properties', dict(type='object', properties={}))]:
    add('schema-' + name, [msg('user', 'hello')], [dict(type='function', function=dict(name='f', parameters=parameters))])
(root/'oracle.json').write_text(json.dumps(dict(jinja2=jinja2.__version__,checkpoint='ee0ef6023621cff504d758262d4e04895a5af4a2',cases=rows),ensure_ascii=False,indent=2)+'\n')
print(jinja2.__version__,len(rows),'cases')
