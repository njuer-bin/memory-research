import re, sys
tex = open('paper/main.tex', encoding='utf-8').read()
envs = re.findall(r'\\begin\{(\w+\*?)\}|\\end\{(\w+\*?)\}', tex)
stack = []
ok = True
for b, e in envs:
    if b:
        stack.append(b)
    elif e:
        if not stack or stack[-1] != e:
            print('MISMATCH at', e, 'stack tail:', stack[-3:])
            ok = False
            break
        stack.pop()
print('env balanced:', ok and not stack, '| unclosed:', stack)
labels = set(re.findall(r'\\label\{([^}]+)\}', tex))
refs = set(re.findall(r'\\ref\{([^}]+)\}', tex))
print('labels:', len(labels), 'refs:', len(refs))
print('dangling refs:', sorted(refs - labels))
print('unused labels:', sorted(labels - refs))
print('brace balance:', tex.count('{') - tex.count('}'))
