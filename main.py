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

    decimator = sdr_functions.filter_decimator(DEC_FAC)     # Creamos el objeto decimador y filtrador de la señal

    block_duration = BLOCK_LEN / SAMPLE_RATE_HZ


    try:
        while not stop_event.is_set():
            raw_iq_data = sdr_functions.get_data_block(sdr, BLOCK_LEN)      # Adquisición de muestras de la SDR

            # Instante aproximado correspondiente al centro
            # temporal del bloque que acabamos de adquirir.
            capture_time = (time.time() - block_duration / 2)


            complex_centered = sdr_functions.bb_shift(raw_iq_data)      # Desplazamiento de la señal a DC
            complex_signal = decimator.process(complex_centered)        # Decimación y filtrado de la señal a 1000 Hz

            try:
                q12.put((capture_time, complex_signal.copy()), timeout=0.5)     # Metemos 32 ms de la señal diezmada en la cola. La cola tiene como máximo 5 + 32 ms = 160 ms de señal.
                if q12.qsize() >= 10:

                    print(
                        f"[WARN] q12 acumulando bloques: "
                        f"{q12.qsize()}/50"
                    )
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


def thread_detector(q12, q23, stop_event, ibp_tracker):

    # Este hilo está siempre corriendo, y prosigue la ejecución a medida que el primer hilo deposita muestras en la cola.
    
    MatchedFilter = detector.LDMatchedFir(F_DECIMATED)     #Filtro adaptado para detectar la raya larga de 1 segundo de duración
            
    try:
        for k in range(int(5*SAMPLE_RATE_HZ/BLOCK_LEN)):      # 5 por el número de bloques que hay en 1 segundo de captura (muestras/segundo / muestras/bloque)
            capture_time, buffer = q12.get(timeout=1)
            last_capture_time = capture_time
            aux= np.square(np.abs(buffer))
            MFOut=MatchedFilter.Filtra(aux)     # MFOut es un array de 32 valores, cada uno una suma móvil de 1000 muestras
    except Exception as e:
        print(f"[ERROR] Detector: {e}")
        return

    print("Arrancando detector de raya larga...")

    print("Inicialización de umbral de detección...")

    # Despreciamos los primeros 5 segundos de captura para estabilizar el filtro:
    
    NoiseFloor=np.mean(MFOut)       # Media de únicamente el último segundo de señal real capturada
    print('Umbral de detección "NoiseFloor":',NoiseFloor)

    Lmax=-1000      # Inicializamos el acumulador del máximo pico del filtro adaptado
    Lmax_sample = None
    
    PastBaliza=ibp_tracker.bindex       # Inicialización de indice de baliza pasada
    NewBaliza=ibp_tracker.bindex        # Inicialización de indice de baliza actual
    
    CumMean=0       # Inicializamos la media acumulada de la salida del filtro adaptado

    #Lmax representa el máximo observado, mientras que CumMean 
    # representa la energía media reciente. Juntos permiten distinguir entre ruido y 
    # señal real: un pico aislado puede no ser suficiente, pero si la media también 
    # está elevada durante un periodo, entonces la señal es más sospechosa de ser una 
    # detección real.

    NF=2
    AdTh=NF*NoiseFloor      #   "Adaptive Theshold": Umbral adaptativo para detección de raya larga

    current_beacon = ibp_tracker.get_beacon_index(
        last_capture_time
    )

    slot_blocks = []
    slot_sample_count = 0
    Lmax = -np.inf
    Lmax_slot_sample = None
    CumMean = 0

    # --------------------------------------------------------
    # PRE-ROLL PARA EL DECODIFICADOR MORSE
    # --------------------------------------------------------
    #
    # Conservamos los últimos 2 segundos del slot anterior.
    # Esto evita cortar el comienzo del indicativo si existe
    # un pequeño desfase entre reloj software y muestras RF.

    PRE_ROLL_SECONDS = 2.0

    PRE_ROLL_SAMPLES = int(PRE_ROLL_SECONDS * F_DECIMATED)

    # Muestras anteriores al comienzo nominal del slot actual.
    slot_pre_roll = np.array([], dtype=float)

    # El primer slot será parcial porque hemos arrancado
    # después de los 5 segundos de estabilización.
    # Lo descartaremos.
    slot_complete = False

    while not stop_event.is_set():

        try:

            capture_time, buffer = q12.get(timeout=1)
            block_beacon = (ibp_tracker.get_beacon_index(capture_time))

            # ====================================================
            # CAMBIO DE SLOT
            # ====================================================

            if block_beacon != current_beacon:

                # ============================================
                # SLOT QUE ACABA DE FINALIZAR
                # ============================================

                if slot_blocks:
                    finished_slot = np.concatenate(slot_blocks)

                else:
                    finished_slot = np.array([], dtype=float)

                # Los últimos 2 segundos de este slot serán
                # el pre-roll del siguiente.
                
                if len(finished_slot) > 0:
                    next_pre_roll = finished_slot[-PRE_ROLL_SAMPLES:].copy()

                else:
                    next_pre_roll = np.array([], dtype=float)

                # ============================================
                # PROCESAR SLOT COMPLETO
                # ============================================

                if (slot_complete and len(finished_slot) > 0):

                    callsign, country = (IBP.ibp_beacons[current_beacon])
                    detected = (Lmax > AdTh)
                    ratio = (Lmax / NoiseFloor if NoiseFloor > 0 else 0)

                    if detected:

                        print(f"!!! DETECCIÓN [{current_beacon:02d}] {callsign} ({country}) Max={Lmax:.6f} Umbral={AdTh:.6f} ratio={ratio:.2f}")

                        # ====================================
                        # BUFFER PARA DECODIFICADOR MORSE
                        # ====================================
                        #
                        # pre-roll del slot anterior
                        # +
                        # slot actual completo

                        if len(slot_pre_roll) > 0:
                            muestras_decode = (np.concatenate((slot_pre_roll, finished_slot)))

                        else:
                            muestras_decode = (finished_slot.copy())

                        # Posición REAL de Lmax dentro del
                        # array que recibe el decoder.
                        if Lmax_slot_sample is not None:
                            lmax_decode_offset = (len(slot_pre_roll) + Lmax_slot_sample)

                        else:
                            lmax_decode_offset = None

                        # Posición donde empieza nominalmente
                        # el slot actual dentro del buffer.
                        slot_boundary_offset = (len(slot_pre_roll))

                        log_timestamp = datetime.now(timezone.utc).isoformat()

                        q23.put({
                            "muestras": muestras_decode,
                            "beacon_index": current_beacon,
                            "callsign_esperado": callsign,
                            "country": country,
                            "lmax_offset": lmax_decode_offset,
                            "slot_boundary_offset": slot_boundary_offset,
                            "lmax": Lmax,

                            # Datos necesarios para escribir el CSV
                            # después de la decodificación.
                            "noise_floor": NoiseFloor,
                            "threshold": AdTh,
                            "timestamp_utc": log_timestamp
                        })

                    else:

                        # Primero registramos la detección negativa usando
                        # EXACTAMENTE los valores con los que se tomó la decisión.
                        decode_log.log_long_dash_result(
                            beacon_index=current_beacon,
                            callsign=callsign,
                            country=country,
                            detected=False,
                            lmax=Lmax,
                            noise_floor=NoiseFloor,
                            threshold=AdTh,
                            decoded_callsign=""
                        )

                        NoiseFloor = (0.9 * NoiseFloor + 0.1 * CumMean)
                        AdTh = (NF * NoiseFloor)

                        print(
                            f"Baliza "
                            f"[{current_beacon:02d}] "
                            f"{callsign} ({country}) "
                            f"NoiseFloor="
                            f"{NoiseFloor:.6f} "
                            f"Umbral={AdTh:.6f} "
                            f"Lmax={Lmax:.6f}"
                        )

                # ============================================
                # COMENZAR EL NUEVO SLOT
                # ============================================

                current_beacon = block_beacon

                new_callsign, new_country = (IBP.ibp_beacons[current_beacon])

                print(
                    f"[{current_beacon:02d}] "
                    f"Recibiendo: "
                    f"{new_callsign} ({new_country})"
                )

                # El final del anterior pasa a ser el
                # pre-roll del nuevo.

                slot_pre_roll = next_pre_roll
                slot_blocks = []
                slot_sample_count = 0
                Lmax = -np.inf
                Lmax_slot_sample = None
                CumMean = 0

                # Tras haber observado una frontera,
                # podemos considerar completo el siguiente.
                slot_complete = True

            # ====================================================
            # ESTE BLOQUE YA SABEMOS A QUÉ SLOT PERTENECE
            # ====================================================

            aux = np.square(np.abs(buffer))

            block_start_in_slot = slot_sample_count

            slot_blocks.append(aux.copy())

            slot_sample_count += len(aux)

            # ====================================================
            # DETECTOR DE RAYA LARGA
            # ====================================================

            MFOut = MatchedFilter.Filtra(aux)

            local_max_index = int(np.argmax(MFOut))

            Mymax = float(MFOut[local_max_index])

            CumMean = 0.5 * (CumMean + np.mean(MFOut))

            if Mymax > Lmax:
                Lmax = Mymax
                Lmax_slot_sample = (block_start_in_slot + local_max_index)

        except queue.Empty:
            print("LongDashTask: Timeout esperando datos")
            break

        except Exception:
            import traceback
            traceback.print_exc()
            break

    
def thread_logger(q23, stop_event):

    while not stop_event.is_set() or not q23.empty():

        try:
            item = q23.get(timeout=0.5)

        except queue.Empty:
            continue

        nombre_captura = (f"captura_{item['callsign_esperado']}_{time.time_ns()}.npy")

        np.save(nombre_captura, item["muestras"])

        print(f"[MORSE] Buffer decoder: {len(item['muestras'])} muestras = {len(item['muestras']) / F_DECIMATED:.3f} s")

        print(f"[MORSE] Frontera nominal del slot en {item['slot_boundary_offset']} muestras = {item['slot_boundary_offset'] / F_DECIMATED:.3f} s")

        if item["lmax_offset"] is not None:

            print(f"[MORSE] Lmax real dentro del buffer en {item['lmax_offset']} muestras = {item['lmax_offset'] / F_DECIMATED:.3f} s")

        resultado = decode_log.extraer_identificador(item["muestras"], fs=1000, debug=True)

        recibido = resultado["identificador"]
        esperado = item["callsign_esperado"]

        correcto = recibido == esperado

        decode_log.log_long_dash_result(
            beacon_index=item["beacon_index"],
            callsign=item["callsign_esperado"],
            country=item["country"],
            detected=True,
            lmax=item["lmax"],
            noise_floor=item["noise_floor"],
            threshold=item["threshold"],
            decoded_callsign=recibido,
            timestamp_utc=item["timestamp_utc"]
        )

        print(f"Esperado: {esperado} | Decodificado: {recibido} | Morse: {resultado['morse']} | OK: {correcto}")
        q23.task_done()

###############################################################################################################


def main():

    sdr = sdr_functions.configure_sdr()

    #Inicialización taks IBP
    ibp_tracker=IBP.IBP()
    # # # # ibp_tracker.InitIBP()
    # # # # time.sleep(2)

    # 1. Crear colas
    q12 = queue.Queue(maxsize=50)
    q23 = queue.Queue(maxsize=5)

    # 2. Crear señal de parada
    stop_event = threading.Event()

    # 3. Crear threads
    t1 = threading.Thread(target=thread_capture, args=(q12, stop_event, sdr))
    t2 = threading.Thread(target=thread_detector, args=(q12, q23, stop_event, ibp_tracker))
    t3 = threading.Thread(target=thread_logger, args=(q23, stop_event))

    # 4. Arrancarlos
    t1.start()
    t2.start()
    t3.start()
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
    t3.join()

if __name__ == "__main__":
    main()