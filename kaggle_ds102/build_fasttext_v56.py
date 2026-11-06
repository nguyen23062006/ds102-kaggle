import os,platform,json,hashlib,urllib.request,zipfile,subprocess,time
from pathlib import Path
assert platform.system()=='Linux' and os.environ.get('COLAB_RELEASE_TAG')
r=Path('/content/ds102_v56');r.mkdir(exist_ok=True);start=time.time()
archive=r/'fastText-v0.9.2.zip'
url='https://codeload.github.com/facebookresearch/fastText/zip/refs/tags/v0.9.2'
if not archive.exists():urllib.request.urlretrieve(url,archive)
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='93cc8a21633e6c6ecf4b5164925ad68a5eda332373d4316d2f56d05f4bfbe823','Unexpected upstream source archive'
base=r/'upstream';base.mkdir(exist_ok=True)
with zipfile.ZipFile(archive) as z:
    for n in z.namelist():assert (base/n).resolve().is_relative_to(base.resolve())
    z.extractall(base)
src=base/'fastText-0.9.2';assert (src/'LICENSE').read_text().startswith('MIT License')
subprocess.run(['make','-j2','CXX=c++ -include cstdint'],cwd=src,check=True)
binary=src/'fasttext';assert binary.is_file()
record=dict(version='0.9.2',source_url=url,source_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),compiler=subprocess.check_output(['c++','--version'],text=True).splitlines()[0],build_flags='make -j2 CXX=c++ -include cstdint',license='MIT',downloaded_assets='official source code only; no vectors/models/corpora',elapsed_seconds=time.time()-start)
(r/'fasttext_build_manifest.json').write_text(json.dumps(record,indent=2));print(json.dumps(record));(r/'build_exit.txt').write_text('0')

