import numpy as np
import pyvisa
import matplotlib.pyplot as plt
import time

class VNA:

    def __init__(self, ip):
        rm1 = pyvisa.ResourceManager()
        self.__VNA = rm1.open_resource(f"tcpip0::{ip}::inst0::INSTR")  # Connessione tramite IP

        self.__VNA.write("*CLS")  # Clear status
        VNA = self.__VNA.query("INST:SEL 'NA'; *OPC?")  # Seleziona la modalità Network Analyzer
        if VNA[0] != '1': 
            raise Exception("Failed to select NA mode")  # Controlla che l'operazione sia andata a buon fine

        self.__VNA.write("AVER:MODE POINT; *OPC")  # Imposta la media a livello di punti
        if not self.wait_for_opc():
            raise Exception("Failed to select averaging mode")  # Controlla che la media sia stata impostata correttamente

        self.__VNA.write("DISP:WIND:TRAC1:Y:AUTO; *OPC")  # Autoscaling sull'asse Y
        if not self.wait_for_opc():
            raise Exception("Failed to select autoscaling")  # Controlla che l'autoscaling sia stato eseguito

        self.__VNA.write("CALC:SMO 0; *OPC")  # Disattiva lo smoothing
        if not self.wait_for_opc():
            raise Exception("Failed to turn off smoothing")  # Controlla che lo smoothing sia stato disattivato


    def off(self):
        try:
            self.__VNA.clear()  # Pulisce il buffer
        finally:
            self.__VNA.close()  # Chiude la connessione


    def wait_for_opc(self, timeout=300):
        """
        Metodo generale per attendere il completamento del comando inviato tramite *OPC.
        
        :param timeout: Tempo massimo (in secondi) da attendere prima di sollevare un TimeoutError.
        """
        start_time = time.time()
        while True:
            status = self.__VNA.query("*OPC?")  # Interroga lo stato del comando *OPC
            if status.strip() == '1':  # Se la risposta è '1', l'operazione è completata
                return True
            if time.time() - start_time > timeout:
                raise TimeoutError("Timeout: l'operazione non è stata completata.")
            time.sleep(0.5)  # Pausa prima di verificare di nuovo


    def wait(self, wait_time):
        time.sleep(wait_time)  # Attende il tempo specificato

    
    def set_freq_limits(self, set_min, set_max):
        set_min_G = set_min * 1e9  # Converti in Hz
        set_max_G = set_max * 1e9  # Converti in Hz
    
        self.__VNA.write(f'FREQ:STAR {set_min_G}')
        self.__VNA.write(f'FREQ:STOP {set_max_G}')
        self.__VNA.write('*OPC')  # Segna che i comandi sono stati inviati

        if self.wait_for_opc():
            print("Frequenza impostata correttamente.")


    def set_freq_span(self, set_center, set_span):
        set_center_G = set_center * 1e9  # Frequenza centrale in Hz
        set_span_G = set_span * 1e9  # Span di frequenza in Hz
        
        self.__VNA.write(f'FREQ:CENT {set_center_G}')
        self.__VNA.write(f'FREQ:SPAN {set_span_G}')
        self.__VNA.write('*OPC')

        if self.wait_for_opc():
            print("Frequenza centrale e span impostati correttamente.")


    def set_power(self, set_power):
        self.__VNA.write(f'SOUR:POW {set_power}')
        self.__VNA.write('*OPC')

        if self.wait_for_opc():
            print(f"Potenza impostata correttamente a {set_power} dBm.")


    def set_ifband(self, ifband):
        self.__VNA.write(f'BWID {ifband}')
        self.__VNA.write('*OPC')

        if self.wait_for_opc():
            print(f"Larghezza di banda IF impostata correttamente a {ifband} Hz.")


    def set_sweep_time(self, sweep_time):
        self.__VNA.write(f'SWE:TIME {sweep_time}')
        self.__VNA.write('*OPC')

        if self.wait_for_opc():
            print(f"Tempo di sweep impostato correttamente a {sweep_time} secondi.")


    def set_sweep_points(self, sweep_points):
        self.__VNA.write(f'SWE:POIN {sweep_points}')
        self.__VNA.write('*OPC')

        if self.wait_for_opc():
            print(f"Numero di punti di sweep impostato correttamente a {sweep_points}.")


    def set_n_means(self, n_means):
        self.__VNA.write(f"SENS:AVER:COUN {n_means}")
        self.__VNA.write('*OPC')

        if self.wait_for_opc():
            print(f"Numero di medie impostato correttamente a {n_means}.")
