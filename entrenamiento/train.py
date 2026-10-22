"""Entrena una MLP 784-16-16-10 (ReLU + softmax) sobre MNIST con numpy puro
y exporta pesos + subconjuntos de imágenes para la animación web."""
import gzip, json, base64, sys, io, os
import numpy as np
from PIL import Image

RAW = sys.argv[1]
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)


def load_images(name):
    with gzip.open(os.path.join(RAW, name), 'rb') as f:
        data = f.read()
    n = int.from_bytes(data[4:8], 'big')
    return np.frombuffer(data, np.uint8, offset=16).reshape(n, 28, 28)


def load_labels(name):
    with gzip.open(os.path.join(RAW, name), 'rb') as f:
        data = f.read()
    return np.frombuffer(data, np.uint8, offset=8)


Xtr = load_images('train-images-idx3-ubyte.gz')
Ytr = load_labels('train-labels-idx1-ubyte.gz')
Xte = load_images('t10k-images-idx3-ubyte.gz')
Yte = load_labels('t10k-labels-idx1-ubyte.gz')
print('train', Xtr.shape, 'test', Xte.shape)


def augment(batch):
    """Afinidad aleatoria (rotación, escala, cizalla, traslación) con muestreo bilineal."""
    B = batch.shape[0]
    ang = rng.uniform(-12, 12, B) * np.pi / 180
    sc = rng.uniform(0.85, 1.12, B)
    sh = rng.uniform(-0.15, 0.15, B)
    tx = rng.uniform(-2.5, 2.5, B)
    ty = rng.uniform(-2.5, 2.5, B)
    c, s = np.cos(ang), np.sin(ang)
    # matriz inversa (salida -> entrada)
    a = c / sc; b = (-s + sh) / sc; d = s / sc; e = c / sc
    yy, xx = np.mgrid[0:28, 0:28].astype(np.float32)
    xx = xx - 13.5; yy = yy - 13.5
    sx = a[:, None, None] * xx + b[:, None, None] * yy + 13.5 - tx[:, None, None]
    sy = d[:, None, None] * xx + e[:, None, None] * yy + 13.5 - ty[:, None, None]
    x0 = np.floor(sx).astype(int); y0 = np.floor(sy).astype(int)
    fx = sx - x0; fy = sy - y0
    pad = np.pad(batch, ((0, 0), (1, 1), (1, 1)))
    def g(yi, xi):
        yi = np.clip(yi + 1, 0, 29); xi = np.clip(xi + 1, 0, 29)
        return pad[np.arange(B)[:, None, None], yi, xi]
    out = (g(y0, x0) * (1 - fx) * (1 - fy) + g(y0, x0 + 1) * fx * (1 - fy)
           + g(y0 + 1, x0) * (1 - fx) * fy + g(y0 + 1, x0 + 1) * fx * fy)
    # grosor de trazo aleatorio (dilatación/erosión suave)
    thick = rng.uniform(0, 1, B)
    dil = np.maximum.reduce([out, np.roll(out, 1, 1), np.roll(out, 1, 2), np.roll(out, -1, 1), np.roll(out, -1, 2)])
    out = np.where((thick > 0.75)[:, None, None], 0.5 * out + 0.5 * dil, out)
    return np.clip(out, 0, 1)


sizes = [784, 16, 16, 10]
W = [rng.normal(0, np.sqrt(2 / sizes[i]), (sizes[i + 1], sizes[i])).astype(np.float32) for i in range(3)]
b = [np.zeros(sizes[i + 1], np.float32) for i in range(3)]


def forward(x):
    a1 = np.maximum(0, x @ W[0].T + b[0])
    a2 = np.maximum(0, a1 @ W[1].T + b[1])
    z3 = a2 @ W[2].T + b[2]
    z3 = z3 - z3.max(1, keepdims=True)
    p = np.exp(z3); p /= p.sum(1, keepdims=True)
    return a1, a2, p


def accuracy(X, Y):
    _, _, p = forward(X.reshape(len(X), -1).astype(np.float32) / 255)
    return (p.argmax(1) == Y).mean()


params = W + b
m = [np.zeros_like(p) for p in params]
v = [np.zeros_like(p) for p in params]
lr0, wd, bs, epochs = 2e-3, 1e-4, 128, 45
step = 0
Xtr_f = Xtr.astype(np.float32) / 255
for ep in range(epochs):
    lr = lr0 * 0.5 * (1 + np.cos(np.pi * ep / epochs)) + 5e-5
    idx = rng.permutation(len(Xtr))
    tot = 0
    for k in range(0, len(idx), bs):
        bi = idx[k:k + bs]
        xb = Xtr_f[bi]
        if ep < epochs - 3:
            mask = rng.uniform(0, 1, len(bi)) < 0.7
            if mask.any():
                xb = xb.copy(); xb[mask] = augment(xb[mask])
        x = xb.reshape(len(bi), -1)
        y = Ytr[bi]
        a1, a2, p = forward(x)
        tot += -np.log(p[np.arange(len(bi)), y] + 1e-9).sum()
        d3 = p.copy(); d3[np.arange(len(bi)), y] -= 1; d3 /= len(bi)
        gW2 = d3.T @ a2; gb2 = d3.sum(0)
        d2 = (d3 @ W[2]) * (a2 > 0)
        gW1 = d2.T @ a1; gb1 = d2.sum(0)
        d1 = (d2 @ W[1]) * (a1 > 0)
        gW0 = d1.T @ x; gb0 = d1.sum(0)
        grads = [gW0 + wd * W[0], gW1 + wd * W[1], gW2 + wd * W[2], gb0, gb1, gb2]
        step += 1
        for i, (pp, gg) in enumerate(zip(params, grads)):
            m[i] = 0.9 * m[i] + 0.1 * gg
            v[i] = 0.999 * v[i] + 0.001 * gg * gg
            mh = m[i] / (1 - 0.9 ** step); vh = v[i] / (1 - 0.999 ** step)
            pp -= (lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)
    print(f'epoch {ep+1:2d} loss {tot/len(idx):.4f} test acc {accuracy(Xte, Yte)*100:.2f}%', flush=True)

test_acc = accuracy(Xte, Yte)
print('FINAL test acc', test_acc)

# escalas de activación para visualizar (percentil 98 de activaciones positivas)
a1, a2, p = forward(Xte.reshape(len(Xte), -1).astype(np.float32) / 255)
pred = p.argmax(1)
s1 = float(np.percentile(a1[a1 > 0], 98)); s2 = float(np.percentile(a2[a2 > 0], 98))
print('scales', s1, s2)


def f32b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr, np.float32).tobytes()).decode()


def sprite(imgs, per_row=100):
    n = len(imgs); rows = (n + per_row - 1) // per_row
    canvas = np.zeros((rows * 28, per_row * 28), np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, per_row)
        canvas[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = im
    buf = io.BytesIO()
    Image.fromarray(canvas, 'L').save(buf, 'PNG', optimize=True)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode(), len(buf.getvalue())


# subconjunto de prueba: 1000 imágenes balanceadas (100 por dígito), orden mezclado
te_idx = np.concatenate([rng.permutation(np.where(Yte == d)[0])[:100] for d in range(10)])
te_idx = rng.permutation(te_idx)
# subconjunto de entrenamiento para el entrenamiento en vivo: 6000 (600 por dígito)
tr_idx = np.concatenate([rng.permutation(np.where(Ytr == d)[0])[:600] for d in range(10)])
tr_idx = rng.permutation(tr_idx)

te_uri, te_bytes = sprite(Xte[te_idx])
tr_uri, tr_bytes = sprite(Xtr[tr_idx])
print('sprite bytes test', te_bytes, 'train', tr_bytes)

sub_acc = float((pred[te_idx] == Yte[te_idx]).mean())
out = {
    'arch': sizes,
    'testAcc': float(test_acc),
    'subsetAcc': sub_acc,
    'scales': [1.0, s1, s2, 1.0],
    'W': [f32b64(w) for w in W],
    'b': [f32b64(x) for x in b],
    'testLabels': ''.join(str(int(y)) for y in Yte[te_idx]),
    'trainLabels': ''.join(str(int(y)) for y in Ytr[tr_idx]),
}
with open(os.path.join(OUT, 'model.json'), 'w') as f:
    json.dump(out, f)
with open(os.path.join(OUT, 'test_sprite.txt'), 'w') as f:
    f.write(te_uri)
with open(os.path.join(OUT, 'train_sprite.txt'), 'w') as f:
    f.write(tr_uri)
# predicciones de referencia para verificar la implementación JS
np.save(os.path.join(OUT, 'ref_probs.npy'), p[te_idx[:20]])
print('subset acc', sub_acc)
print('ref first 5 preds', pred[te_idx[:5]], 'labels', Yte[te_idx[:5]])
print('ref first probs', np.round(p[te_idx[0]], 4).tolist())
