"""Seal and check a completed experiment's delivery artifacts; does not run training."""
from pathlib import Path
import hashlib,json,os,re
from datetime import datetime,timezone

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'docs/ai-selfplay-loop.md').exists())
RUN=HERE/'run'
COMP=HERE.parent/'comparisons/v003-vs-v002-20261029'
read=lambda p:json.loads(p.read_text())
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def save(p,data):p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
manifest=read(HERE/'manifest.json');results=read(HERE/'results.json');comparison=read(COMP/'result.json');verification=read(RUN/'verification.json')
assert manifest['status']==comparison['status']=='complete'
assert manifest['completedIterations']==10 and verification['passed'] and comparison['verification']['passed']
assert comparison['configuration']['games']==400 and comparison['configuration']['pairedSeeds']==200
assert comparison['configuration']['candidate']['sha256']==manifest['best']['jsonSha256']
assert comparison['configuration']['opponent']['sha256']=='f4128e2cf621490b100afd6df08f7fbf79f9c9e16777a10850ce54206790acf3'
assert results['independentComparison']['overall']==comparison['overall']
for name,expected in read(RUN/'source-snapshot.json')['files'].items():assert digest(RUN/'source'/name)==expected
for kind in ('best','latest'):
    m=manifest[kind]
    for ext in ('pt','json'):assert digest(HERE/m[ext])==m[ext+'Sha256']
for path,h in comparison['source']['sourceSha256'].items():assert digest(ROOT/path)==h
for name,h in comparison['artifacts'].items():assert digest(COMP/'run'/name)==h
for name,v in verification['reports'].items():assert digest(RUN/name)==v['sha256']
assert read(RUN/'ui-verification.json')['dashboardSha256']==digest(RUN/'dashboard.html')
# The external node_modules symlink is intentionally excluded. Locked package manifests are in source/.
symlinks=[]
for folder,dirs,names in os.walk(RUN,followlinks=False):
    for name in dirs[:]:
        p=Path(folder)/name
        if p.is_symlink():symlinks.append({'path':str(p.relative_to(RUN)),'target':os.readlink(p)});dirs.remove(name)
assert [s['path'] for s in symlinks]==['source/node_modules']
# Verify documentation links. The only not-yet-written target is the manifest produced below.
checked=[]
for doc in [HERE/'README.md',COMP/'README.md',HERE.parent/'README.md',ROOT/'docs/ai-selfplay-loop.md',ROOT/'docs/ai-rl-scaling.md']:
    text=doc.read_text()
    for match in re.finditer(r'\[[^\]]*\]\(([^)]+)\)',text):
        link=match.group(1).split('#')[0]
        if not link or '://' in link:continue
        path=(doc.parent/link).resolve()
        assert path.exists() or path==HERE/'artifacts.sha256',(doc,link)
        checked.append({'document':str(doc.relative_to(ROOT)),'target':link})
    if doc in [HERE/'README.md',COMP/'README.md']:assert '正在运行' not in text and '进行中' not in text
save(RUN/'delivery-verification.json',{'passed':True,'checkedAt':datetime.now(timezone.utc).isoformat(),'documentLinksChecked':len(checked),'links':checked,'modelIdentitiesChecked':True,'frozenSourceAndComparisonSourceUnchanged':True,'reportArtifactHashesChecked':True,'excludedExternalSymlinks':symlinks,'browserInteractionVerified':False,'browserLimitation':verification['browserLimitation']})
status=read(RUN/'finalization-status.json')
status.update(phase='ready-to-report',comparisonState=read(COMP/'run/state.json'),comparisonVerified=True,documentsFinalized=True,remaining=['Reply with full results and links; pause harmonies-10 without archiving or committing.'])
save(RUN/'finalization-status.json',status)
files=[]
for folder,dirs,names in os.walk(RUN,followlinks=False):
    dirs[:]=[d for d in dirs if not (Path(folder)/d).is_symlink()]
    files.extend(Path(folder)/n for n in names if not (Path(folder)/n).is_symlink())
lines=[]
for p in sorted(files):lines.append(f'{digest(p)}  {p.relative_to(RUN)}')
(HERE/'artifacts.sha256').write_text('\n'.join(lines)+'\n')
print(json.dumps({'passed':True,'artifactFiles':len(files),'artifactBytes':sum(p.stat().st_size for p in files),'documentLinks':len(checked),'excludedExternalSymlinks':symlinks,'checksumManifestSha256':digest(HERE/'artifacts.sha256')},ensure_ascii=False))
