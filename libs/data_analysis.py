# Función para visualizar la señal IQ en el dominio del tiempo
def visualizar_espectrograma_iq(complex_data, fs, titulo="Espectrograma", t_ventana=30e-3):
    
    if complex_data is None or len(complex_data) == 0:
        print("No hay datos para representar.")
        return

    L = max(8, int(t_ventana * fs))  # evitar ventanas demasiado pequeñas
    if L > len(complex_data):
        L = len(complex_data)

    if L < 2:
        print("La señal es demasiado corta para calcular el espectrograma.")
        return

    f, t, Sxx = spectrogram(
        complex_data,
        fs=fs,
        nperseg=L,
        noverlap=L // 2,
        return_onesided=False,
        mode="magnitude"
    )

    f = np.fft.fftshift(f)
    Sxx = np.fft.fftshift(Sxx, axes=0)

    Sxx_db = 20 * np.log10(Sxx + 1e-12)

    plt.figure(figsize=(12, 6))
    #plt.pcolormesh(t, f, Sxx_db, shading="gouraud") # Suaviza el espectrograma, pero tarda literalmente 10 minutos en procesar
    plt.pcolormesh(t, f, Sxx_db, shading="auto")
    plt.title(titulo)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Frecuencia [Hz]")
    plt.colorbar(label="Magnitud [dB]")
    plt.tight_layout()
    plt.show()



# Funcion para visualizar la señal real envolvente/potencia y los resultados de los filtros adaptados
def visualizar_senal_real(x, fs, titulo="Señal real", ylabel="Amplitud"):
    
    if x is None or len(x) == 0:
        print("No hay datos para representar.")
        return

    t = np.arange(len(x)) / fs

    plt.figure(figsize=(12, 4))
    plt.plot(t, x)
    plt.title(titulo)
    plt.xlabel("Tiempo [s]")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()