# MNIST paso a paso

Animación interactiva, para clases de inteligencia artificial, que muestra cómo una red neuronal reconoce dígitos escritos a mano.

**Ver en línea:** https://jorgecif.github.io/red-neuronal-mnist/

La red no es una simulación. Es un perceptrón multicapa 784 → 16 → 16 → 10 (ReLU + softmax) entrenado con las 60 000 imágenes de MNIST, que acierta el 95,6 % de las 10 000 imágenes de prueba. La página hace en JavaScript el mismo cálculo que el entrenamiento en numpy, con esos mismos pesos.

## Capítulos

1. **La imagen:** una cuadrícula de 28 × 28 píxeles se convierte en un vector de 784 números.
2. **Una neurona:** pesos, suma ponderada, sesgo y función de activación ReLU.
3. **La red completa:** propagación hacia adelante, capa por capa.
4. **La decisión:** softmax convierte los puntajes en probabilidades.
5. **El aprendizaje:** pérdida, retropropagación y descenso del gradiente. Una red nueva se entrena de verdad en el navegador con 6 000 imágenes.
6. **Tu turno:** dibuja un dígito y la red lo clasifica en tiempo real.

Teclado: ← → para avanzar, espacio para reproducir, 1 a 6 para saltar de capítulo. Para abrir un capítulo directamente, agrega `#capitulo3` (o el número que quieras) al final de la dirección.

## Estructura

| Ruta | Contenido |
| --- | --- |
| `index.html` | La animación lista para usar (generada; no se edita a mano). |
| `src/app.html` | Código fuente de la animación. |
| `datos/` | Pesos entrenados (`model.json`) e imágenes de MNIST empaquetadas como PNG. |
| `entrenamiento/train.py` | Script de entrenamiento con numpy. |
| `build.py` | Inserta los datos en `src/app.html` y genera `index.html`. |

## Modificar y regenerar

Después de editar `src/app.html`:

```bash
python build.py
```

Para volver a entrenar la red, descarga los cuatro archivos `.gz` de MNIST en una carpeta y ejecuta (requiere numpy y Pillow):

```bash
python entrenamiento/train.py carpeta_mnist carpeta_salida
```

Luego copia `model.json`, `test_sprite.txt` y `train_sprite.txt` de `carpeta_salida` a `datos/` y ejecuta `python build.py`.

## Créditos

Base de datos MNIST: Yann LeCun, Corinna Cortes y Christopher J. C. Burges, distribuida bajo licencia CC BY-SA 3.0. Este repositorio incluye 7 000 de sus imágenes.
