"""Inserta el modelo entrenado y las imágenes de MNIST en src/app.html.

Uso:  python build.py [ruta_extra_sin_doctype.html]

Genera index.html (documento completo, funciona sin internet salvo las fuentes; es lo que sirve GitHub Pages).
Si se da una ruta extra, escribe también una versión sin <!doctype> para publicar como Artifact.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).parent
DATA = ROOT / 'datos'

template = (ROOT / 'src' / 'app.html').read_text(encoding='utf-8')
model = json.loads((DATA / 'model.json').read_text(encoding='utf-8'))
test_sprite = (DATA / 'test_sprite.txt').read_text(encoding='utf-8').strip()
train_sprite = (DATA / 'train_sprite.txt').read_text(encoding='utf-8').strip()

for marker in ('/*__MODEL__*/null', "'__TEST_SPRITE__'", "'__TRAIN_SPRITE__'"):
    assert template.count(marker) == 1, f'marcador no encontrado o repetido: {marker}'

body = (template
        .replace('/*__MODEL__*/null', json.dumps(model, separators=(',', ':')))
        .replace("'__TEST_SPRITE__'", json.dumps(test_sprite))
        .replace("'__TRAIN_SPRITE__'", json.dumps(train_sprite)))

full = ('<!doctype html>\n<html lang="es">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        + body + '\n</html>\n')
out = ROOT / 'index.html'
out.write_text(full, encoding='utf-8')
print(f'{out.name}: {out.stat().st_size / 1e6:.2f} MB')

if len(sys.argv) > 1:
    extra = pathlib.Path(sys.argv[1])
    extra.write_text(body, encoding='utf-8')
    print(f'{extra}: {extra.stat().st_size / 1e6:.2f} MB')
