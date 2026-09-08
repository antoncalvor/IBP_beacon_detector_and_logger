import time
import csv
import numpy as np
from datetime import datetime, timezone
import os

def identify_morse_elements(
    on_off,
    fs=1000,
    dot_range_ms=(35, 65),
    dash_range_ms=(95, 200),
    letter_space_range_ms=(100, 260),
    word_space_min_ms=300):

    on_off = np.asarray(on_off, dtype=bool) # Aseguramos que on_off es array booleano de numpy
    elementos = []

    # A continuación se dejan lineas de código para obtener el número de muestras
    # que ocuparían los puntos, rayas y espacios en función de su duración y la
    # frecuencia a la que se haza dejado diezmada la señal.

    # En nuestro caso de fs=1000 Hz, podrían omitirse estas líneas y usar directamente
    # los valores en muestras, pero así queda más claro y adaptable a otras fs.

    dot_min = int(dot_range_ms[0] * fs / 1000)
    dot_max = int(dot_range_ms[1] * fs / 1000)

    dash_min = int(dash_range_ms[0] * fs / 1000)
    dash_max = int(dash_range_ms[1] * fs / 1000)

    letter_min = int(letter_space_range_ms[0] * fs / 1000)
    letter_max = int(letter_space_range_ms[1] * fs / 1000)

    word_min = int(word_space_min_ms * fs / 1000)

    i = 0
    N = len(on_off)

    while i < N:
        value = on_off[i]
        count = 0

        # A continuación, un contador que incrementa 'count' solo cuando hay elementos en
        # "on_off" iguales al actual valor on_off[i]
        # El contados se detiene por si solo cuando aparezca un valor diferente,
        # gracias a la condición del while, osea cuando se termine un elemento
        while i < N and on_off[i] == value:
            count += 1
            i += 1

        # Aquí detectamos si se trata de punto o raya en base a duración en muestras
        if value:  # == if value=True, osea, tramo ON
            if dot_min <= count <= dot_max:
                elementos.append(".")
            elif dash_min <= count <= dash_max:
                elementos.append("-")
            else:
                elementos.append("?")  # ON no clasificable

        # Aquí detectamos si se trata de espacio entre letras o palabras
        else:  # == if value=False, osea, tramo OFF
            if letter_min <= count <= letter_max:
                elementos.append(" ")
            elif count >= word_min:
                elementos.append("/")
            # silencios cortos se ignoran: separación interna entre punto/raya

    return elementos

def decode_morse(symbols):
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

    mensaje = ""
    palabra = ""

    for s in symbols:
        if s == "." or s == "-":
            palabra += s
        elif s == " ":
            if palabra:
                mensaje += MORSE_DICT.get(palabra, "?")
                palabra = ""
        elif s == "/":
            if palabra:
                mensaje += MORSE_DICT.get(palabra, "?")
                palabra = ""
            mensaje += " "

    if palabra:
        mensaje += MORSE_DICT.get(palabra, "?")

    return mensaje

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