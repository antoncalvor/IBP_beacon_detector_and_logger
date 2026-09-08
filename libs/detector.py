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