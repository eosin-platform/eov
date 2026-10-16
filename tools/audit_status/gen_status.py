"""Regenerates the "Status after the 1.0 improvement pass" section of AUDIT.md
from status_data.py (per finding) and features_data.py (§14): python3 tools/audit_status/gen_status.py
"""
import re, sys, subprocess, collections
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from status_data import S
from features_data import FEATURES

ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
findings = []
for line in open(os.path.join(HERE, 'findings.txt')):
    parts = line.rstrip('\n').split(' · ', 2)
    if len(parts) < 3: continue
    fid, sev, title = parts
    findings.append((fid, sev, title))
assert len(findings) == 169, len(findings)
missing = [f for f, _, _ in findings if f not in S]
assert not missing, missing

LABEL = {'F': 'Fixed', 'P': 'Partly fixed', 'W': 'In progress', 'N': 'Not fixed'}
FLABEL = {'F': 'Done', 'P': 'Partly', 'W': 'In progress', 'N': 'Not done'}
AREAS = [('PLG', 'Plugin API and host runtime'), ('WSI', 'Core WSI, tiles, cache'), ('RND', 'Rendering'),
         (('CLR','EXP','DS'), 'Colour, export, dataset'), ('UI', 'App state and interaction'),
         ('SH', 'Shell, CLI, config, docs'), ('UPD', 'Install, update, packaging'), ('ANN', 'Annotations plugin'),
         ('VAE', 'eovae plugin'), ('CEL', 'Cellaris plugin'), ('GPD', 'Gamepad plugin')]

def area_of(fid):
    p = fid.split('-')[0]
    for key, name in AREAS:
        if (isinstance(key, tuple) and p in key) or p == key:
            return name
    raise KeyError(fid)

head = subprocess.check_output(['git', '-C', ROOT, 'rev-parse', '--short', 'HEAD'], text=True).strip()
out = []
w = out.append
w('## Status after the 1.0 improvement pass')
w('')
w(f'*Updated 2026-10-15, at `eov` master @ `{head}`, version 1.0.0 (every component, plugins included, moved to 1.0.0).*')
w('')
w('The findings in §§3–13 describe the audited 0.4.9 revision and are left as written. This section records what the 1.0 pass changed. "Where" lists the commits in this repository that made the change. For the annotations, cellaris and gamepad submodules it lists the superproject commits that recorded the submodule; the fixes themselves are commits in those repositories. eovae lives in this repository.')
w('')
# overview
counts = collections.OrderedDict()
for fid, sev, title in findings:
    a = area_of(fid)
    c = counts.setdefault(a, collections.Counter())
    c[S[fid][0]] += 1
    c['n'] += 1
w('| Area | Findings | Fixed | Partly fixed | In progress | Not fixed |')
w('|---|---|---|---|---|---|')
tot = collections.Counter()
for a, c in counts.items():
    w(f"| {a} | {c['n']} | {c['F']} | {c['P'] or '–'} | {c['W'] or '–'} | {c['N'] or '–'} |")
    tot.update(c)
w(f"| **Total** | **{tot['n']}** | **{tot['F']}** | **{tot['P']}** | **{tot['W']}** | **{tot['N']}** |")
w('')
w('### What is still open')
w('')
for fid, sev, title in findings:
    st, where, note = S[fid]
    if st != 'F':
        w(f'- **{fid}** ({LABEL[st].lower()}): {note}')
for sec, rows in FEATURES:
    for name, st, where, note in rows:
        if st in ('P', 'N', 'W'):
            n = f': {note}' if note else ''
            w(f'- §{sec.split()[0]} **{name}** ({FLABEL[st].lower()}){n}')
w('')
w('### Per-finding status')
w('')
w('| Id | Sev. | Finding | Status | Where | Notes |')
w('|---|---|---|---|---|---|')
for fid, sev, title in findings:
    st, where, note = S[fid]
    title = title.replace('|', '\\|')
    if len(title) > 90:
        title = title[:87].rstrip() + '…'
    status = LABEL[st] if st == 'F' else f'**{LABEL[st]}**'
    where_md = ' '.join(f'`{x}`' if re.fullmatch(r'[0-9a-f]{7}', x) else x for x in where.split())
    w(f'| {fid} | {sev.title()} | {title} | {status} | {where_md} | {note} |')
w('')
w('### Features from §14')
w('')
for sec, rows in FEATURES:
    w(f'**§{sec}**')
    w('')
    w('| Feature | Status | Where | Notes |')
    w('|---|---|---|---|')
    for name, st, where, note in rows:
        where_md = ' '.join(f'`{x}`' if re.fullmatch(r'[0-9a-f]{7}', x) else x for x in where.split())
        status = FLABEL[st] if st == 'F' else f'**{FLABEL[st]}**'
        w(f'| {name} | {status} | {where_md} | {note} |')
    w('')
w('---')
w('')
section = '\n'.join(out)

p = ROOT + '/AUDIT.md'
s = open(p).read()
start = s.find('## Status after the 1.0 improvement pass')
if start >= 0:
    end = s.find('## 1. Summary', start)
    s = s[:start] + section + s[end:]
else:
    anchor = '---\n\n## 1. Summary'
    assert s.count(anchor) == 1
    s = s.replace(anchor, '---\n\n' + section + '## 1. Summary')
    s = s.replace('## Contents\n1. [Summary]', '## Contents\n0. [Status after the 1.0 improvement pass](#status-after-the-10-improvement-pass)\n1. [Summary]')
open(p, 'w').write(s)
print(tot)
