"""Strict completion accounting for the pinned bounded Qwen vision recipe.

The recipe uses no custom stop strings/ids. Its generation_config.json EOS ids
are 248046 and 248044. A diagnostic MTP packet may contain up to three ids past
the public output boundary; those must not count towards public usage.
"""


def completion_length(response, prompt_tokens, max_tokens=64):
    usage = response['usage']
    if set(usage) != {'prompt_tokens', 'completion_tokens', 'total_tokens'} or any(
            type(v) is not int for v in usage.values()):
        raise RuntimeError('invalid integer usage')
    count = usage['completion_tokens']
    if usage['prompt_tokens'] != prompt_tokens or not 0 < count <= max_tokens or \
            usage['total_tokens'] != prompt_tokens + count:
        raise RuntimeError('wrong completion accounting')
    choices = response['choices']
    if len(choices) != 1 or choices[0]['index'] != 0:
        raise RuntimeError('wrong completion choice')
    finish = choices[0]['finish_reason']
    if finish not in ('length', 'stop') or (finish == 'length' and count != max_tokens):
        raise RuntimeError('wrong completion finish')
    return count


def committed_ids(response, prompt_tokens, sampled_ids, max_tokens=64,
                  eos_ids=(248046, 248044)):
    count = completion_length(response, prompt_tokens, max_tokens)
    if not count <= len(sampled_ids) <= count + 3 or any(type(i) is not int for i in sampled_ids):
        raise RuntimeError('wrong sampled token count/type')
    ids = sampled_ids[:count]
    if any(i in eos_ids for i in ids[:-1]):
        raise RuntimeError('public completion passed EOS')
    if response['choices'][0]['finish_reason'] == 'stop' and ids[-1] not in eos_ids:
        raise RuntimeError('stop completion lacks its EOS')
    return ids
