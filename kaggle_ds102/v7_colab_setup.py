import os, platform, zipfile, hashlib, json
from pathlib import Path
assert platform.system() == 'Linux' and os.environ.get('COLAB_RELEASE_TAG'), 'Hosted Colab only'
bundle = Path('/content/ds102_v7_bundle.zip')
if not bundle.exists():
    from google.colab import files
    uploaded = files.upload()
    assert bundle.exists(), 'Upload ds102_v7_bundle.zip'
    del uploaded
root = Path('/content/ds102_v7_bundle')
root.mkdir(exist_ok=True)
with zipfile.ZipFile(bundle) as z:
    for name in z.namelist():
        assert (root/name).resolve().is_relative_to(root.resolve()), 'Unsafe ZIP path'
    z.extractall(root)
manifest = json.loads((root/'bundle_manifest.json').read_text())
for name, expected in manifest.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
print('Bundle checked:', len(manifest), 'files. Hosted Colab CPU:', os.cpu_count())
