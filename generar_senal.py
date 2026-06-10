import math
import random
import csv


n_puntos = 10000
x_min = -200
x_max = 200
paso = (x_max - x_min) / (n_puntos - 1)


picos = [
    {"A": 0.2, "mu": -5, "sigma": 4},
    {"A": 0.3, "mu": -80, "sigma": 2},
    {"A": 0.1, "mu": 100, "sigma": 3}
]


nivel_ruido = 0.02
random.seed(42)  
nombre_archivo = "senal.csv"

with open(nombre_archivo, mode='w', newline='') as archivo_csv:
    escritor = csv.writer(archivo_csv)
    
    escritor.writerow(["X", "Senal_Limpia", "Senal_Con_Ruido"])
    
    for i in range(n_puntos):
        x = x_min + i * paso
        
        
        senal_limpia = 0.0
        for pico in picos:
            exponente = -((x - pico["mu"]) ** 2) / (2 * (pico["sigma"] ** 2))
            senal_limpia += pico["A"] * math.exp(exponente)
            
        
        ruido = random.gauss(0, nivel_ruido)
        senal_con_ruido = senal_limpia + ruido

        escritor.writerow([round(x, 4), round(senal_limpia, 4), round(senal_con_ruido, 4)])

print(f"¡Listo! Archivo '{nombre_archivo}' generado con éxito.")
