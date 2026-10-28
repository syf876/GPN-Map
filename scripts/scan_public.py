"""Scan local project files. Prints locations only, never secret values."""
import pathlib,re,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
PATTERNS={
 'provider_credential':r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]+|AKIA[A-Z0-9]{16})\b",
 'password_hash':r"scrypt\$[0-9]+\$[0-9]+\$[0-9]+\$[a-f0-9]{16,}\$[a-f0-9]{32,}",
 'private_key':r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
 'absolute_path':r"(?<![A-Za-z])[A-Za-z]:[\\/]|/(?:Users|home)/[^\s]+",
 'email':r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
}
def scan():
 files=[]
 for p in ROOT.rglob('*'):
  if any(x in p.parts for x in ('.git','.venv','__pycache__','artifacts')) or not p.is_file():continue
  files.append(p)
 findings=[]
 for p in files:
  rel=p.relative_to(ROOT).as_posix()
  if p.stat().st_size>10*1024*1024:findings.append((rel,0,'large_file'))
  if p.name=='.env' or (p.name.startswith('.env.') and p.name!='.env.example'):findings.append((rel,0,'private_config'))
  binary=p.suffix.lower() in {'.png','.jpg','.jpeg','.mp4'}
  data=p.read_bytes()
  text='\n'.join(x.decode('ascii') for x in re.findall(rb'[\x20-\x7e]{12,}',data)) if binary else data.decode('utf-8')
  if p.suffix=='.svg':text=re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+','embedded-image',text)
  for name,pattern in PATTERNS.items():
   if binary and name in {"absolute_path","email"}:continue
   # Third-party license contacts and vendored library URLs are public attribution.
   if name=='email' and ('licenses/' in rel or '/vendor/' in rel):continue
   for m in re.finditer(pattern,text):findings.append((rel,text[:m.start()].count('\n')+1,name))
 for rel,line,kind in findings:print(f'{rel}:{line}: {kind}')
 print(f'Scanned {len(files)} files; {len(findings)} findings')
 return bool(findings)
if __name__=='__main__':sys.exit(scan())
