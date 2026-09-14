import time
import csv
import numpy as np
from datetime import datetime, timezone
import os

DOT = 0.054     # segundos
DASH = 0.162
INTRA_ELEMENT = 0.054
INTRA_CHAR = 0.162

MORSE_DICT = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E",
    "..-.": "F", "--.": "G", "....": "H", "..": "I", ".---": "J",
    "-.-": "K", ".-..": "L", "--": "M", "-.": "N", "---": "O",
    ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T",
    "..-": "U", "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y",
    "--..": "Z",
    "-----": "0", ".----": "1", "..---": "2", "...--": "3",
    "....-": "4", ".....": "5", "-....": "6", "--...": "7",
    "---..": "8", "----.": "9"
}

    #   Notas:
    #   Cada transmisión dura 10 s y contiene el indicativo a 22 WPM, seguido de cuatro rayas de 1 s. 
    #   Más concretamente, el controlador usa un tiempo elemental de 54 ms: punto = 54 ms, raya Morse = 162 ms, 
    #   separación dentro de una letra = 54 ms, separación entre letras = 162 ms y separación de palabra = 378 ms. 
    #   Después vienen las cuatro rayas de 1 s separadas por 162 ms. 
    #   
    #   A fs = 1000 Hz: aproximadamente 54 muestras por punto y 162 por raya.

def _aplanar_muestras(muestras):
    """
    Acepta:
        - un np.ndarray 1D
        - una lista de bloques np.ndarray

    Devuelve un único array 1D.
    """
    if isinstance(muestras, (list, tuple)):
        if len(muestras) == 0:
            return np.array([], dtype=float)

        muestras = np.concatenate(
            [np.asarray(bloque).ravel() for bloque in muestras]
        )

    return np.asarray(muestras, dtype=float).ravel()


def _histeresis_db(x_db, threshold_off, threshold_on):

    salida = np.zeros(len(x_db), dtype=np.uint8)

    estado = False

    for i, valor in enumerate(x_db):

        if estado:
            if valor < threshold_off:
                estado = False

        else:
            if valor > threshold_on:
                estado = True

        salida[i] = int(estado)

    return salida


def _rle(binaria):
    """
    Run Length Encoding.

    000111100011
        ↓
    [(0,3), (1,4), (0,3), (1,2)]
    """

    if len(binaria) == 0:
        return []

    cambios = (
        np.flatnonzero(
            np.diff(binaria.astype(np.int8)) != 0
        )
        + 1
    )

    inicios = np.r_[0, cambios]
    finales = np.r_[cambios, len(binaria)]

    return [
        (
            int(binaria[inicio]),
            int(fin - inicio)
        )
        for inicio, fin
        in zip(inicios, finales)
    ]


def _eliminar_glitches(binaria, min_samples):
    """
    Elimina tramos imposiblemente cortos.

    A 22 WPM el elemento mínimo real dura unos 54 ms.
    Por tanto un cambio ON/OFF de, por ejemplo, 5-20 ms
    no puede ser Morse real.
    """

    resultado = binaria.copy()

    # Varias pasadas porque al corregir un glitch
    # pueden fusionarse tres tramos.
    for _ in range(10):

        runs = _rle(resultado)

        if len(runs) < 3:
            break

        posiciones = []

        pos = 0

        for estado, longitud in runs:
            posiciones.append(pos)
            pos += longitud

        hubo_cambio = False

        # No modificamos primero ni último
        for i in range(1, len(runs) - 1):

            estado, longitud = runs[i]

            estado_anterior = runs[i - 1][0]
            estado_siguiente = runs[i + 1][0]

            # Un tramo muy corto entre dos estados iguales
            # es casi con seguridad un glitch.
            if (
                longitud < min_samples
                and estado_anterior == estado_siguiente
            ):

                inicio = posiciones[i]
                fin = inicio + longitud

                resultado[inicio:fin] = estado_anterior

                hubo_cambio = True

        if not hubo_cambio:
            break

    return resultado


def extraer_identificador(
    muestras,
    fs=1000,
    unidad_s=0.054,
    debug=False
):

    potencia = np.asarray(
        muestras,
        dtype=float
    ).ravel()

    if len(potencia) < fs:
        return {
            "identificador": "",
            "morse": "",
            "error": "Número de muestras insuficiente"
        }

    # -------------------------------------------------------
    # 1. Suavizado muy corto
    # -------------------------------------------------------

    # 7 ms:
    # suficientemente pequeño frente a los 54 ms del punto.
    n = int(round(0.007 * fs))

    if n % 2 == 0:
        n += 1

    kernel = np.ones(n) / n

    potencia_suave = np.convolve(
        potencia,
        kernel,
        mode="same"
    )

    # -------------------------------------------------------
    # 2. Pasamos a dB
    # -------------------------------------------------------

    eps = np.finfo(float).tiny

    potencia_db = 10 * np.log10(
        potencia_suave + eps
    )

    # -------------------------------------------------------
    # 3. Estimamos RUIDO
    # -------------------------------------------------------

    # Usamos los últimos 0.8 segundos.
    # En la transmisión IBP ya deberían haber terminado
    # callsign + rayas largas.

    n_ruido = min(
        len(potencia_db),
        int(0.8 * fs)
    )

    zona_ruido = potencia_db[-n_ruido:]

    ruido_db = np.median(zona_ruido)

    # -------------------------------------------------------
    # 4. Estimamos nivel de SEÑAL
    # -------------------------------------------------------

    # Sólo nos interesa el comienzo del slot, donde están
    # el callsign y la primera raya.
    inicio = potencia_db[
        :min(len(potencia_db), int(4.5 * fs))
    ]

    # Un percentil alto representa razonablemente el CW ON.
    senal_db = np.percentile(inicio, 95)

    separacion_db = senal_db - ruido_db

    if separacion_db < 3:
        return {
            "identificador": "",
            "morse": "",
            "error": "No hay separación suficiente señal/ruido",
            "ruido_db": ruido_db,
            "senal_db": senal_db
        }

    # -------------------------------------------------------
    # 5. Umbral
    # -------------------------------------------------------

    # Punto medio EN dB entre ruido y señal.
    threshold = (
        ruido_db +
        0.5 * separacion_db
    )

    # Histéresis de +-1 dB
    threshold_on = threshold + 1.0
    threshold_off = threshold - 1.0

    binaria = _histeresis_db(
        potencia_db,
        threshold_off,
        threshold_on
    )

    # -------------------------------------------------------
    # 6. Eliminación de glitches
    # -------------------------------------------------------

    unidad = unidad_s * fs

    # Ningún elemento Morse real debería durar menos
    # de aproximadamente la mitad de una unidad.
    min_glitch = int(
        round(0.55 * unidad)
    )

    binaria = _eliminar_glitches(
        binaria,
        min_glitch
    )

    # -------------------------------------------------------
    # 7. Medimos duraciones
    # -------------------------------------------------------

    tramos = _rle(binaria)

    morse_chars = []

    actual = ""

    duraciones_on = []
    duraciones_off = []

    iniciado = False

    # A partir de 6 unidades ya no puede ser una raya Morse.
    # La raya normal son 3 unidades.
    limite_raya_larga = 6 * unidad

    for estado, duracion in tramos:

        if estado == 1:

            duraciones_on.append(duracion)

            # -----------------------------------------------
            # Raya IBP de ~1 segundo -> fin del identificador
            # -----------------------------------------------

            if duracion >= limite_raya_larga:

                if actual:
                    morse_chars.append(actual)

                break

            # Un pulso demasiado corto sigue siendo basura
            if duracion < 0.55 * unidad:
                continue

            iniciado = True

            # -----------------------------------------------
            # Punto/raya
            # -----------------------------------------------

            if duracion < 2 * unidad:
                actual += "."

            else:
                actual += "-"

        else:

            duraciones_off.append(duracion)

            if not iniciado or not actual:
                continue

            # -----------------------------------------------
            # Separación interna
            # -----------------------------------------------

            if duracion < 2 * unidad:
                continue

            # -----------------------------------------------
            # Separación entre caracteres
            # -----------------------------------------------

            if duracion < 5 * unidad:

                morse_chars.append(actual)
                actual = ""

            # -----------------------------------------------
            # Separación todavía mayor
            # -----------------------------------------------

            else:

                morse_chars.append(actual)
                actual = ""

    if actual:
        morse_chars.append(actual)

    identificador = "".join(
        MORSE_DICT.get(x, "?")
        for x in morse_chars
    )

    morse = " ".join(morse_chars)

    if debug:

        print()
        print("-------- DEBUG MORSE --------")

        print(
            f"Muestras recibidas: {len(potencia)} "
            f"({len(potencia)/fs:.3f} s)"
        )

        print(
            f"Ruido: {ruido_db:.2f} dB"
        )

        print(
            f"Señal estimada: {senal_db:.2f} dB"
        )

        print(
            f"Separación: {separacion_db:.2f} dB"
        )

        print(
            f"Threshold OFF: {threshold_off:.2f} dB"
        )

        print(
            f"Threshold ON: {threshold_on:.2f} dB"
        )

        print(
            "ON [muestras]:",
            duraciones_on[:50]
        )

        print(
            "OFF [muestras]:",
            duraciones_off[:50]
        )

        print(
            "ON [ms]:",
            [
                round(1000 * x / fs, 1)
                for x in duraciones_on[:50]
            ]
        )

        print(
            "OFF [ms]:",
            [
                round(1000 * x / fs, 1)
                for x in duraciones_off[:50]
            ]
        )

        print(
            "Morse:",
            morse
        )

        print(
            "Identificador:",
            identificador
        )

        print("-----------------------------")
        print()

    return {
        "identificador": identificador,
        "morse": morse,
        "duraciones_on": duraciones_on,
        "duraciones_off": duraciones_off,
        "ruido_db": ruido_db,
        "senal_db": senal_db,
        "threshold_on_db": threshold_on,
        "threshold_off_db": threshold_off
    }




########################################################################################################
#  VALIDACIÓN DE LA DETECCIÓN


def log_long_dash_result(
    beacon_index,
    callsign,
    country,
    detected,
    lmax,
    noise_floor,
    threshold,
    filename="detecciones.csv"
):
    
    # Fecha y hora UTC en el instante de registrar el resultado
    timestamp = datetime.now(timezone.utc).isoformat()

    # Relación señal/ruido usada para analizar posteriormente el umbral
    ratio = lmax / noise_floor if noise_floor > 0 else 0

    # Comprobamos si hay que escribir la cabecera
    write_header = (
        not os.path.exists(filename)
        or os.path.getsize(filename) == 0
    )

    with open(
        filename,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        fieldnames = [
            "timestamp_utc",
            "beacon_index",
            "callsign",
            "country",
            "detected",
            "lmax",
            "noise_floor",
            "threshold",
            "ratio"
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter=";"
        )

        if write_header:
            writer.writeheader()

        writer.writerow({
            "timestamp_utc": timestamp,
            "beacon_index": beacon_index,
            "callsign": callsign,
            "country": country,
            "detected": detected,
            "lmax": lmax,
            "noise_floor": noise_floor,
            "threshold": threshold,
            "ratio": ratio
        })