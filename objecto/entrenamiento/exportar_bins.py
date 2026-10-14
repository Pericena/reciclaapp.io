"""Entrena botella contra no botella y escribe los .bin de TensorFlow.js."""
from pathlib import Path
import numpy as np
import cv2

RAIZ = Path(__file__).resolve().parent.parent
KAGGLE = Path.home() / ".cache" / "kagglehub" / "datasets" / "diptosen" / "bottle-detection-dataset" / "versions" / "1"
SALIDA = RAIZ / "web" / "modelo_botellas"
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp"}
TAM = 100

POSITIVAS = [
    KAGGLE / "datasets" / "bottles_closeups",
    KAGGLE / "datasets" / "bottles_occluded",
    RAIZ / "img" / "p",
    RAIZ / "dato" / "p",
    RAIZ / "dato" / "img" / "p2",
]
NEGATIVAS = [
    RAIZ / "img" / "n",
    RAIZ / "dato" / "n",
    RAIZ / "dato" / "img" / "n2",
]


def listar(carpetas):
    fotos = []
    for carpeta in carpetas:
        if not carpeta.is_dir():
            print("No está:", carpeta)
            continue
        encontradas = [ruta for ruta in carpeta.iterdir() if ruta.suffix.lower() in EXTENSIONES]
        print(f"{carpeta}: {len(encontradas)}")
        fotos.extend(encontradas)
    return fotos


def cargar(rutas, etiqueta):
    imagenes = []
    etiquetas = []
    for ruta in rutas:
        imagen = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
        if imagen is None:
            continue
        imagen = cv2.resize(imagen, (TAM, TAM), interpolation=cv2.INTER_AREA)
        imagenes.append(imagen.astype(np.float32) / 255.0)
        etiquetas.append(etiqueta)
    return imagenes, etiquetas


def main():
    import tensorflow as tf

    positivas = listar(POSITIVAS)
    negativas = listar(NEGATIVAS)
    if len(positivas) < 10 or len(negativas) < 10:
        raise SystemExit("Faltan fotos para entrenar.")

    azar = np.random.default_rng(7)
    if len(negativas) > len(positivas):
        negativas = [negativas[i] for i in azar.choice(len(negativas), size=len(positivas), replace=False)]

    imagenes = []
    etiquetas = []
    for grupo, etiqueta in ((positivas, 1.0), (negativas, 0.0)):
        lote_x, lote_y = cargar(grupo, etiqueta)
        imagenes.extend(lote_x)
        etiquetas.extend(lote_y)

    entradas = np.array(imagenes, dtype=np.float32)[..., np.newaxis]
    salidas = np.array(etiquetas, dtype=np.float32)
    orden = azar.permutation(len(salidas))
    entradas, salidas = entradas[orden], salidas[orden]
    print("Muestras", len(salidas), "botella", int(salidas.sum()), "no botella", int((salidas == 0).sum()))

    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(TAM, TAM, 1), name="image_input"),
        tf.keras.layers.Conv2D(32, 3, activation="relu", name="conv2d"),
        tf.keras.layers.MaxPooling2D(2, name="max_pooling2d"),
        tf.keras.layers.Conv2D(64, 3, activation="relu", name="conv2d_1"),
        tf.keras.layers.MaxPooling2D(2, name="max_pooling2d_1"),
        tf.keras.layers.Conv2D(128, 3, activation="relu", name="conv2d_2"),
        tf.keras.layers.MaxPooling2D(2, name="max_pooling2d_2"),
        tf.keras.layers.Flatten(name="flatten"),
        tf.keras.layers.Dense(100, activation="relu", name="dense"),
        tf.keras.layers.Dense(1, activation="sigmoid", name="dense_1"),
    ], name="botellas_classifier")
    modelo.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    modelo.fit(entradas, salidas, epochs=12, batch_size=16, validation_split=0.15, verbose=2)

    bloques = []
    for capa in modelo.layers:
        for arreglo in capa.get_weights():
            bloques.append(np.ascontiguousarray(arreglo, dtype=np.float32))
    blob = b"".join(bloque.tobytes() for bloque in bloques)
    trozo = 4 * 1024 * 1024
    partes = [blob[i:i + trozo] for i in range(0, len(blob), trozo)]
    SALIDA.mkdir(parents=True, exist_ok=True)
    for indice, parte in enumerate(partes, start=1):
        destino = SALIDA / f"group1-shard{indice}of{len(partes)}.bin"
        destino.write_bytes(parte)
        print("Escrito", destino.name, len(parte))
    if len(partes) != 2:
        raise SystemExit(f"Salieron {len(partes)} shards; model.json espera 2.")
    print("Listo en", SALIDA)


if __name__ == "__main__":
    main()
