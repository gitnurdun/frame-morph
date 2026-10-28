#!/usr/bin/env python3
"""Bundle private dependencies; keep the host's glibc and NVIDIA driver."""
import os,pathlib,re,shutil,subprocess,sys
build,root,sysroot=map(lambda p:pathlib.Path(p).resolve(),sys.argv[1:4])
(root/'bin').mkdir(exist_ok=True);(root/'lib').mkdir(exist_ok=True)
env=dict(os.environ);env['LD_LIBRARY_PATH']=str(sysroot/'usr/lib/x86_64-linux-gnu')+':'+env.get('LD_LIBRARY_PATH','')
def replace_file(source,destination,strip=False):
    temporary=destination.with_name(destination.name+'.updating')
    shutil.copy2(source,temporary,follow_symlinks=True)
    if strip:subprocess.run(['strip',str(temporary)],check=True)
    os.replace(temporary,destination)
for name in ('framemorph','framemorph_tests'):
    exe=build/name;replace_file(exe,root/'bin'/name,strip=True)
    result=subprocess.run(['ldd',str(exe)],env=env,capture_output=True,text=True,check=True)
    if 'not found' in result.stdout:raise RuntimeError(result.stdout)
    for library,path in re.findall(r'^\s*(\S+)\s+=>\s+(/\S+)',result.stdout,re.M):
        source=pathlib.Path(path)
        if str(source).startswith(str(sysroot)+'/'):
            replace_file(source,root/'lib'/library)
print('Packaged private libraries:',len(list((root/'lib').glob('*'))))
