import numpy as np
from scipy import signal
from rtlsdr import RtlSdr
import time

SAMPLE_RATE_HZ = 250_000
CENTER_FREQ_HZ = 139_037_500.0  # (14100000 + 125000000 - fs/4) teniendo en cuenta el efecto del Upconverter y para evitar que la señal se pierda en el DC
PPM_CORRECTION_PPM = 0          # Parts Per Million, corrije el error de frecuencia del oscilador interno de la SDR. Ajustar si es necesario.
GAIN_DB = 7                     # Ganancia en dB (ajustable, o 'auto' para máxima)

DEC_FAC = [5, 5, 5, 2]                          # Factores de decimación para obtener una fs final de 1000 Hz. Es importante que la frecuencia diezmada
                                                # final sea múltiplo de 4 para poder hacer el desplazamiento en frecuencia conservando la fase


def configure_sdr():
    sdr = RtlSdr()

    sdr.sample_rate = SAMPLE_RATE_HZ
    sdr.center_freq = CENTER_FREQ_HZ

    if PPM_CORRECTION_PPM != 0:
        try:
            sdr.freq_correction = int(PPM_CORRECTION_PPM)
        except Exception as e:
            print(f"[WARN] No se pudo aplicar freq_correction={PPM_CORRECTION_PPM} ppm: {e}")

    sdr.gain = GAIN_DB
    print("[INFO] Ganancia de la SDR establecida en:", sdr.gain)

    print(f"[INFO] SDR sintonizado inicialmente en {CENTER_FREQ_HZ/1e6:.6f} MHz, Fs={SAMPLE_RATE_HZ} Hz")
    time.sleep(0.1)
    return sdr

def get_data_block(sdr, block_size):

    samples = sdr.read_samples(block_size)
    return samples

def bb_shift(complex_data):
    pattern = np.array([1, -1j, -1, 1j], dtype=np.complex64)
    ##En el caso de que la señal estuviera en -fs/4, no en fs/4, el patrón sería:
    #pattern = np.array([1, 1j, -1, -1j], dtype=np.complex64) #sentido contrario
    exp_ = np.tile(pattern, len(complex_data)//4 + 1)[:len(complex_data)] # asegurando que tiene la misma longitud que 'iq'. Esto sería "asegurar la sincronización de fase"
    complex_centered = complex_data * exp_
    return complex_centered

def decimate_and_filter(complex_data):
    
    complex_dec = complex_data # Si no se ha desplazado a banda base
    for i, M in enumerate(DEC_FAC):
        complex_dec = signal.decimate(complex_dec, q=M, ftype='fir', zero_phase=True)    
    return complex_dec

def update_buffer(new_block, dec_block_len): # actualiza el buffer circular con un nuevo bloque de muestras
    
    global buffer

    if len(new_block) != dec_block_len:
        raise ValueError(f"Esperaba {dec_block_len}, recibidas {len(new_block)}")
    buffer[:dec_block_len] = buffer[dec_block_len:]     # movemos la segunda mitad del buffer a la primera mitad
    buffer[dec_block_len:] = new_block                  # metemos el nuevo bloque en la segunda mitad del buffer