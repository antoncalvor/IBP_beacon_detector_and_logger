import libs.IBP as IBP
import libs.sdr_functions as sdr_functions
import libs.detector as detector
import libs.decode_log as decode_log

import numpy as np
import time
import scipy
import matplotlib
import queue
import threading
from threading import Thread
from threading import Event
from rtlsdr import RtlSdr
from scipy.io import wavfile
from scipy.signal import spectrogram
from datetime import datetime, timezone

##############################################################################################################
# CONFIGURACION SDR


BLOCK_LEN = 8000                                # Tamaño del bloque de datos a leer de la SDR. Debe ser múltiplo de 4 para poder hacer el desplazamiento en frecuencia conservando la fase
                                                # Antonio usa 8064. Escogemos 8000 para que con la f_s que tenemos, resulte en capturas de 32 ms

SAMPLE_RATE_HZ = 250_000
DEC_FAC = [5, 5, 5, 2]
F_DECIMATED = SAMPLE_RATE_HZ / np.prod(DEC_FAC)     # Frecuencia de muestreo final tras decimación. En este caso 1000 Hz


###############################################################################################################


def thread_capture(q12, stop_event, sdr):
    global buffer

    last_detection_time = 0
    COOLDOWN_S = 8

    try:
        while not stop_event.is_set():
            raw_iq_data = sdr_functions.get_data_block(sdr, BLOCK_LEN)      # Adquisición de muestras de la SDR
            complex_centered = sdr_functions.bb_shift(raw_iq_data)      # Desplazamiento de la señal a DC
            complex_signal = sdr_functions.decimate_and_filter(complex_centered)        # Decimación y filtrado de la señal a 1000 Hz

            try:
                q12.put(complex_signal.copy(), timeout=0.5)
            except queue.Full:
                print("[ERROR] q12 llena: el detector no procesa suficientemente rápido")
                stop_event.set()
                break

    except Exception as e:
        print(f"[ERROR] Captura: {e}")

    finally:
        try:
            sdr.close()
        except Exception:
            pass


def thread_detector(q12, stop_event, ibp_tracker):

    
    MatchedFilter = detector.LDMatchedFir(F_DECIMATED)     #Filtro adaptado para detectar la raya larga de 1 segundo de duración
        
    try:
        for k in range(int(5*SAMPLE_RATE_HZ/BLOCK_LEN)):      # 5 por el número de tramas en
            buffer = q12.get(timeout=1)
            aux= np.square(np.abs(buffer))
            MFOut=MatchedFilter.Filtra(aux)     # MFOut es la energía acumulada en 1 segundo de la señal recibida, tras pasar por el filtro adaptado
    except Exception as e:
        print(f"[ERROR] Detector: {e}")
        return

    print("Arrancando detector de raya larga...")

    print("Inicialización de umbral de detección...")

    # Despreciamos los primeros 5 segundos de captura para estabilizar el filtro:
    

    NoiseFloor=np.mean(MFOut)       # Media de únicamente el último segundo de señal real capturada
    print('NoiseFloor',NoiseFloor)

    Lmax=-1000      # Inicializamos el acumulador del máximo pico del filtro adaptado
    
    PastBaliza=ibp_tracker.bindex       # Inicialización de indice de baliza pasada
    NewBaliza=ibp_tracker.bindex        # Inicialización de indice de baliza actual
    
    CumMean=0       # Inicializamos la media acumulada de la salida del filtro adaptado

    #Lmax representa el máximo observado, mientras que CumMean 
    # representa la energía media reciente. Juntos permiten distinguir entre ruido y 
    # señal real: un pico aislado puede no ser suficiente, pero si la media también 
    # está elevada durante un periodo, entonces la señal es más sospechosa de ser una 
    # detección real.

    NF=2
    AdTh=NF*NoiseFloor      #Umbral adaptativo para detección de raya larga

    while not stop_event.is_set():
        try:
            buffer = q12.get(timeout=1)
            NewBaliza=ibp_tracker.bindex        # Actualización de indice de baliza
            aux= np.square(np.abs(buffer))        # Cálculo de la magnitud al cuadrado de la señal recibida. Potencia
            MFOut=MatchedFilter.Filtra(aux)     #Actualización de salida del filtro
            #Máximo del filtro adaptado
            Mymax=np.max(MFOut)     # Valor pico del último segundo filtrado

            #Máquina de estados
            
            if PastBaliza==NewBaliza:       #Seguimos en la misma baliza
                CumMean=0.5*(CumMean+np.mean(MFOut))        #Media acumulada de la salida del filtro adaptado
                if Mymax > Lmax:
                    Lmax=Mymax        #Actualizo el máximo del filtro adaptado en esta baliza

            else:       # Está transmitiendo la siguiente baliza, y podemos decidir si durante los
                        # 10 segundos de la baliza anterior hubo una detección de raya larga

                detected = Lmax > AdTh  # Determinamos si hubo detección de raya larga en la baliza anterior
                callsign, country = IBP.ibp_beacons[PastBaliza]
                ratio = Lmax / NoiseFloor

                if detected: # Se superó el umbral, hay detección
                    txt = "!!! Positivo {} Max= {:.4f}"                    

                    print(
                        f"!!! DETECCIÓN [{PastBaliza:02d}] "
                        f"{callsign} ({country}) "
                        f"Max={Lmax:.6f} "
                        f"Umbral={AdTh:.6f}"
                        f"ratio={ratio:.2f}"
                    )

                    # Aquí podrías agregar código para registrar la detección, enviar una señal,
                    # o cualquier otra acción que desees realizar cuando se detecte una baliza.

                else: #No supero el umbral no hay detección
                    # Actualizamos el umbral adaptativo para la siguiente baliza,
                    # usando una media ponderada del ruido observado y el umbral anterior:
                    NoiseFloor=0.9*NoiseFloor+0.1*CumMean
                    AdTh=NF*NoiseFloor

                    print(
                        f"Baliza [{PastBaliza:02d}] "
                        f"{callsign} ({country}) "
                        f"NoiseFloor={NoiseFloor:.6f} "
                        f"Umbral={AdTh:.6f} "
                        f"Lmax={Lmax:.6f}"
                    )

                decode_log.log_long_dash_result(
                    beacon_index=PastBaliza,
                    callsign=callsign,
                    country=country,
                    detected=detected,
                    lmax=Lmax,
                    noise_floor=NoiseFloor,
                    threshold=AdTh
                )

                PastBaliza=NewBaliza
                Lmax=Mymax
                CumMean=0

        except queue.Empty:
            print("LongDashTask: Timeout esperando datos")
            break

        except Exception:
            import traceback
            traceback.print_exc()
            break




    # # # q23.put({
    # # #     "identificador": identificador if detection["detected_raw"] else "",
    # # #     "timestamp": detection["timestamp"],
    # # #     "detected_raw": detection["detected_raw"],
    # # #     "score": detection["score"],
    # # #     "noise": detection["noise"],
    # # #     "ratio": detection["ratio"],
    # # #     "pos": detection["pos"]
    # # # })
    # q12.task_done()

    
def thread_logger(q23, stop_event):

    pass

###############################################################################################################


def main():

    sdr = sdr_functions.configure_sdr()

    #Inicialización taks IBP
    ibp_tracker=IBP.IBP()
    ibp_tracker.InitIBP()
    time.sleep(2)

    # 1. Crear colas
    q12 = queue.Queue(maxsize=5)

    # 2. Crear señal de parada
    stop_event = threading.Event()

    # 3. Crear threads
    t1 = threading.Thread(target=thread_capture, args=(q12, stop_event, sdr))
    t2 = threading.Thread(target=thread_detector, args=(q12, stop_event, ibp_tracker))

    # 4. Arrancarlos
    t1.start()
    t2.start()

    # 5. Mantener el programa vivo
    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Parando...")

    # 6. Señal de parada
    stop_event.set()

    # 7. Esperar a que terminen
    t1.join()
    t2.join()


if __name__ == "__main__":
    main()