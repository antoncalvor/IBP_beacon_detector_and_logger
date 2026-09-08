import time
from threading import Thread
# Lista oficial de balizas IBP (orden IMPORTANTE)
# Cada baliza transmite durante 10 segundos en secuencia
ibp_beacons = [
    ("4U1UN", "United Nations"),
    ("VE8AT", "Canada"),
    ("W6WX", "United States"),
    ("KH6WO", "Hawaii"),
    ("ZL6B", "New Zealand"),
    ("VK6RBP", "Australia"),
    ("JA2IGY", "Japan"),
    ("RR9O", "Russia"),
    ("VR2B", "Hong Kong"),
    ("4S7B", "Sri Lanka"),
    ("ZS6DN", "South Africa"),
    ("5Z4B", "Kenya"),
    ("4X6TU", "Israel"),
    ("OH2B", "Finland"),
    ("CS3B", "Madeira"),
    ("LU4AA", "Argentina"),
    ("OA4B", "Peru"),
    ("YV5B", "Venezuela"),
]

class IBP(object):
    def __init__(self):
        self.SLOT_DURATION = 10  # segundos por baliza
        self.NUM_BEACONS = len(ibp_beacons)
        self.CYCLE_DURATION = self.SLOT_DURATION * self.NUM_BEACONS  # 180 segundos
        self.bindex=0
        print(self.SLOT_DURATION,self.NUM_BEACONS,self.CYCLE_DURATION)

    #   Le das un instante de tiempo y te devuelve qué baliza IBP corresponde a ese instante.
    def get_beacon_index(self,timestamp=None):
        t = int(timestamp) % self.CYCLE_DURATION
        index = t // self.SLOT_DURATION
        return index

    #   Mostrar por pantalla la baliza que corresponde al instante actual.
    def print_current_beacon(self):
        index, (callsign, country) = self.get_current_beacon()
        print(f"[{index:02d}] Recibiendo: {callsign} ({country})")

    #   Crear un hilo separado que ejecute continuamente IBPTask().
    def InitIBP(self):
        self.IBP=Thread(target=self.IBPTask)
        self.IBP.daemon=True
        self.IBP.start()

    #   Sincronizarse con los slots de 10 segundos de IBP y mantener actualizado self.bindex.
    def IBPTask(self):
        print("IBP: syncing")
        timestamp = time.time()
        while int(timestamp*10) % 100 != 0:
                timestamp = time.time()
        t = int(timestamp) % self.CYCLE_DURATION
        self.bindex = t // self.SLOT_DURATION
        print("IBP: sync done")
        while True:
            (callsign, country)=ibp_beacons[self.bindex]
            print(f"[{self.bindex:02d}] Recibiendo: {callsign} ({country})")
            time.sleep(9.9)
            timestamp = time.time()
            while int(timestamp*10) % 100 != 0:
                timestamp = time.time()
            self.bindex=(self.bindex+1) % self.NUM_BEACONS