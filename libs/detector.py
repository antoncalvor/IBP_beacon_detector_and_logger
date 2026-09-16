from scipy import signal
import numpy as np
import libs.IBP as IBP

BLOCK_LEN = 8000                                # Tamaño del bloque de datos a leer de la SDR. Debe ser múltiplo de 4 para poder hacer el desplazamiento en frecuencia conservando la fase
                                                # Antonio usa 8064. Escogemos 8000 para que con la f_s que tenemos, resulte en capturas de 32 ms
SAMPLE_RATE_HZ = 250_000
DEC_FAC = [5, 5, 5, 2]
F_DECIMATED = SAMPLE_RATE_HZ / np.prod(DEC_FAC)     # Frecuencia de muestreo final tras decimación. En este caso 1000 Hz


class LDMatchedFir(object):
    def __init__(self,F_DECIMATED):
        self.h=np.ones(int(F_DECIMATED)) #construccion de la respuesta impulsional FIR de longitud 'F_DECIMATED' (entero). Corresponde a una duración temporal de 1 segundo.
        self.zi=signal.lfilter_zi(self.h,1)
    def Filtra(self,entrada):
        salida, zf=signal.lfilter(self.h,1,entrada,zi=self.zi)
        self.zi=zf
        return(salida)

def trim_history(history, minimum_sample):

    """
    Elimina bloques completamente anteriores a minimum_sample.

    history contiene:
        (indice_absoluto_primera_muestra, array_potencia)
    """

    while history:

        block_start, block = history[0]
        block_end = block_start + len(block)

        if block_end <= minimum_sample:
            history.popleft()
        else:
            break


def extract_history(history, start_sample, end_sample):

    """
    Extrae del historial continuo las muestras comprendidas
    entre start_sample y end_sample.

    Los índices son índices absolutos dentro del stream
    diezmado a 1 kHz.
    """

    partes = []

    for block_start, block in history:

        block_end = block_start + len(block)

        # Bloque completamente anterior
        if block_end <= start_sample:
            continue

        # Ya hemos pasado la región que queremos
        if block_start >= end_sample:
            break

        local_start = max(
            start_sample,
            block_start
        ) - block_start

        local_end = min(
            end_sample,
            block_end
        ) - block_start

        partes.append(
            block[
                int(local_start):
                int(local_end)
            ]
        )

    if not partes:
        return np.array([], dtype=float)

    return np.concatenate(partes)