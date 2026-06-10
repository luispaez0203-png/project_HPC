## **Inferencia Bayesiana Paralela con MCMC** 

Ajuste de mezclas gaussianas en se˜nales experimentales 

_- High Performance Computing HPCF_ 

**Autor:** Luis Eduardo Cardoso Paez **Profesor:** Edwin Fl´orez G´omez **Universidad de Puerto Rico** 

10 de junio de 2026 

1 

## **Resumen** 

En este proyecto hacemos uso de la Inferencia Bayesiana para la estimaci´on de par´ametros en se˜nales astron´omicas. Construimos un modelo de suma de picos gaussianos, que tenga como par´ametro de entrada el n´umero posibles picos, _k_ , que pueda tener la se˜nal. Para hacer esto, implementamos un algoritmo de Markov Chain Monte Carlo (MCMC) con el m´etodo de Metropoli-Hastings para estimar lo par´ametros de todos los _k_ picos gaussianos superpuestos con el ruido. Desde el enfoque de HPC usamos el paralelismo a nivel de tareas ejecutando 4 cadenas MCMC independientes simult´aneamente en diferentes n´ucleos de la computadora o de cl´uster, esto permite una exploraci´on eficiente del espacio de par´ametros, mostrando si un pico detectado aparece en todas las cadena. Finalmente, los resultados de todas las cadenas se combinan mediante un promedio para obtener una estimaci´on general de los par´ametros de cada pico. 

## **1. Descripci´on del Proyecto** 

El proyecto estima, a partir de un archivo CSV con una se˜nal experimental, los par´ametros de cada _k_ picos gaussianos. Estos par´ametros son: 

- 

- _A_ Amplitud de cada pico el cual esta relacionado con la intensidad del gas 

- _µ_ — Centro (posici´on) de cada pico que relacionada con la velocidad a la que se mueve 

- _σ_ — Ancho (desviaci´on est´andar) de cada pico que relacionado con la temperatura del gas 

El modelo asume la forma: 

**==> picture [380 x 36] intentionally omitted <==**

Donde _k_ , es la cantidad de picos que se cree que tiene la se˜nal astron´omica. 

## **1.1. Fundamento Bayesiano** 

La inferencia bayesiana se basa en el teorema de Bayes: 

**==> picture [318 x 29] intentionally omitted <==**

Aplicando logaritmo: 

**==> picture [375 x 13] intentionally omitted <==**

La ecuaci´on (3), es la que usaremos para hacer la inferencia basados en el modelo que se muestra a continuaci´on. 

2 

## **1.1.1. Verosimilitud (Likelihood) -** _P_ ( **Datos** _|θ_ ) 

La verosimilitud mide la capacidad del modelo para explicar los datos observados: 

**==> picture [325 x 31] intentionally omitted <==**

ˆ Donde, _yi_ son los datos observados, _yi_ es el valor predicho por el modelo, _σ_[2] es la varianza del ruido. 

## **1.1.2. Distribuci´on a Priori -** _P_ ( _θ_ ) 

La distribuci´on a priori incorpora restricciones f´ısicas y conocimiento previo del sistema: 

**==> picture [378 x 48] intentionally omitted <==**

## **1.1.3. Distribuci´on a Posteriori -** _P_ ( _θ|_ **Datos** ) 

La distribuci´on a posteriori representa la probabilidad actualizada de los par´ametros despu´es de observar los datos: 

**==> picture [294 x 13] intentionally omitted <==**

Es decir: 

**==> picture [314 x 13] intentionally omitted <==**

Esta distribuci´on combina la informaci´on de los datos observados (verosimilitud) con el conocimiento previo (prior) para obtener una estimaci´on m´as robusta de los par´ametros. 

## **1.2. Tipo de Paralelismo** 

Para este proyecto usaremos **Task Parallelism** (Paralelismo a nivel de tareas): 

- Se ejecutan 4 cadenas MCMC independientes simult´aneamente, cada una en un n´ucleo distinto del cl´uster o de la maquina local. 

Cada cadena usa una semilla aleatoria diferente (seed = 1000 + chain ~~i~~ d), garantizando exploraci´on diversa del espacio de par´ametros. 

Al finalizar, los resultados de cada cadena se combinan promediando. 

3 

|**Core 1: Cadena 1 (seed=1001)**|_→_result<br>~~c~~hain<br>1.txt|
|---|---|
|**Core 2: Cadena 2 (seed=1002)**|_→_result<br>~~c~~hain<br>2.txt|
|**Core 3: Cadena 3 (seed=1003)**|_→_result<br>~~c~~hain<br>3.txt|
|**Core 4: Cadena 4 (seed=1004)**|_→_result<br>~~c~~hain<br>4.txt|



Figura 1: Esquema de paralelizaci´on de cadenas MCMC 

## **Ventajas sobre el modo lineal** 

El tiempo de ejecuci´on es similar al de 1 sola cadena 

Mejor exploraci´on del espacio de par´ametros 

Verificaci´on de posibles picos para evitar falsos positivos. 

## **2. Archivos del Repositorio** 

Aunque se tienen se˜nales astron´omicas reales para poder realizar el experimento NO TENGO PERMITIDO COMPARTIR estas se˜nales hasta que las investigaciones que se est´an haciendo con esas hayan concluido. Por tal motivo, podemos crear se˜nales sint´eticas para hacer el experimento. En el repositorio vamos a encontrar los siguientes archivo: 

Listing 1: Estructura del repositorio 

|1<br>2<br>3<br>4|`generar_senal .py`<br>`# Genera la senal`<br>`sintetica`<br>`con K picos`<br>`gaussianos`<br>`bayes_parallel .py`<br>`# Algoritmo`<br>`MCMC`<br>`paralelo (una cadena por`<br>`ejecucion)`<br>`analisis_cadenas .py`<br>`# Combina`<br>`resultados y genera`<br>`graficas`<br>`README.md`<br>`# Este`<br>`archivo`|
|---|---|



Donde la rutina generar ~~s~~ enal.py una senal sint´etica con _k_ picos guassianos. En este caso, hay que ajustar los par´ametros para cada picos en el programa, cuando lo ejecutamos nos genera un archivo llamado senal.csv que va ser el par´ametros de entrada de bayes ~~p~~ arallel.py. En bayes ~~p~~ arallel.py se tiene todo el modelo Bayesiano que se encarga de hacer la estimaci´on de los par´ametros, tiene como par´ametros de entrada la se˜nal, el n´umero de picos _k_ y la cantidad de iteraciones del m´etodo MCMC. En analisis ~~c~~ adenas.py combinamos los resultados y generamos las gr´aficas que se muestran en pantalla y previamente se guardan en la carpeta. EL procedimiento para ejecutarlo se muestra a continuaci´on. 

## **3. C´omo Ejecutarlo** 

Para poder ejecutar este programa, tenemos que tener instalado pandas. Si no est´a instalado, lo podemos hacer como: 

1 `pip install numpy pandas matplotlib scipy` 

4 

## **3.1. Generar la se˜nal sint´etica (opcional)** 

- 1 `python generar_senal .py` 

Esto crea `senal.csv` con 3 picos gaussianos y ruido gaussiano ( _σ_ = 0 _,_ 02). 

## **3.2. Ejecutar las cadenas MCMC en paralelo** 

## **En el cl´uster (Linux):** 

1 `for id in 1 2 3 4; do` 2 `python3 bayes_parallel .py --file senal.csv --K 3 --chain_id $id --n_iter 15000 &` 3 `done` 4 `wait` 5 `echo "todas las cadenas finalizadas"` 

## **En Windows (PowerShell / VS Code):** 

Los parametros de entrada son la senal.csv, el n´umero de picos _k_ = 3 y el n´umero de interaciones _n_ ~~_i_~~ _ter_ = 20000. 

1 

```
for%iin(1234)dostartpythonbayes_parallel .py--filesenal.
csv--K3--chain_id%i--n_iter20000
```

Esto genera los archivos: 

```
resultchain1.txt
resultchain2.txt
resultchain3.txt
resultchain4.txt
```

## **3.3. Analizar resultados y generar gr´aficas** 

1 `python analisis_cadenas .py` 

Produce: 

- Tabla en consola con los par´ametros estimados vs valores reales y % de error 

- 

- `analisis cadenas.png` Comparaci´on de las 4 cadenas por par´ametro 

- `senal` ~~`r`~~ `econstruida.png` — Se˜nal reconstruida vs se˜nal original 

5 

## **3.4. Descargar resultados del cl´uster (PSCP)** 

- 1 `pscp lcardoso@boqueron .hpcf.upr.edu:/ home/lcardoso /*. png C:\ Users\ profesor\Desktop\` 

- 2 `pscp lcardoso@boqueron .hpcf.upr.edu:/ home/lcardoso /*. txt C:\ Users\ profesor\Desktop\` 

## **4. Par´ametros del Algoritmo** 

|**Par´ametro**|**Valor**|**Descripci´on**|
|---|---|---|
|–fle<br>–K<br>–chain<br>~~i~~d<br>–n<br>iter<br>sigma<br>noise<br>burn<br>~~i~~n|senal.csv<br>3<br>1-4<br>20000<br>0.02<br>30 %|Archivo CSV con columnas x, y<br>N´umero de picos gaussianos<br>ID de la cadena (defne la semilla)<br>N´umero de iteraciones MCMC<br>Nivel de ruido de la se˜nal<br>Fracci´on descartada al inicio|



Cuadro 1: Par´ametros de configuraci´on del algoritmo 

## **5. Resultados Obtenidos** 

|**Par´ametro**|**Valor Real**|**Estimado**|**Error**|
|---|---|---|---|
|_A_1<br>_A_2<br>_A_3|0.3<br>0.2<br>0.1|_∼_0.2639<br>_∼_0.1950<br>_∼_0.0872|_∼_12 %<br>_∼_2.5 %<br>_∼_12 %|
|_µ_1<br>_µ_2<br>_µ_3|-80<br>-5<br>100|_∼_-80.0006<br>_∼_-5.1402<br>_∼_99.8047|_∼_0.00075 %<br>_∼_2.8 %<br>_∼_0.19 %|
|_σ_1<br>_σ_2<br>_σ_3|2<br>4<br>3|_∼_2.6271<br>_∼_4.2280<br>_∼_3.8433|_∼_31 %<br>_∼_5 %<br>_∼_25 %|



Cuadro 2: Resultados de estimaci´on de par´ametros 

## **6. C´odigo Fuente** 

## **6.1. generar** ~~**s**~~ **enal.py** 

6 

Listing 2: C´odigo para generar se˜nal sint´etica 

1 `import math` 2 `import random` 3 `import csv` 4 5 6 `n_puntos = 10000` 7 `x_min = -200` 8 `x_max = 200` 9 `paso = (x_max - x_min) / (n_puntos - 1)` 10 11 12 `picos = [` 13 `{ "A" : 0.2, "mu" : -5, "sigma" : 4},` 14 `{ "A" : 0.3, "mu" : -80, "sigma" : 2},` 15 `{ "A" : 0.1, "mu" : 100, "sigma" : 3}` 16 `]` 17 18 19 `nivel_ruido = 0.02` 20 `random.seed (42)` 21 `nombre_archivo = "senal.csv"` 22 23 `with open (nombre_archivo , mode= ’w’ , newline= ’’) as archivo_csv:` 24 `escritor = csv.writer(archivo_csv)` 25 26 `escritor.writerow ([ "X" , "Senal_Limpia" , " Senal_Con_Ruido " ])` 27 28 `for i in range (n_puntos):` 29 `x = x_min + i * paso` 30 31 32 `senal_limpia = 0.0` 33 `for pico in picos:` 34 `exponente = -((x - pico[ "mu" ]) ** 2) / (2 * (pico[ "sigma " ] ** 2))` 35 `senal_limpia += pico[ "A" ] * math.exp(exponente)` 36 37 38 `ruido = random.gauss(0, nivel_ruido)` 39 `senal_con_ruido = senal_limpia + ruido` 40 41 `escritor.writerow ([ round (x, 4), round (senal_limpia , 4), round (senal_con_ruido , 4)])` 42 43 `print (f " Listo ! Archivo ’{nombre_archivo }’ generado con xito ." )` 

7 

44 `print ( "Contiene 1000 puntos con picos en X=[2.5 , 5.0, 7.5] y ruido moderado." )` 

## **6.2. bayes** ~~**p**~~ **arallel.py** 

Listing 3: Algoritmo MCMC paralelo 

1 `import numpy as np` 2 `import matplotlib.pyplot as plt` 3 `from scipy.signal import find_peaks` 4 `import argparse` 5 `import pandas as pd` 6 `import sys` 7 8 `parser = argparse. ArgumentParser ()` 9 `parser.add_argument ( "--file" , type = str , required=True)` 10 `parser.add_argument ( "--K" , type = int , required=True)` 11 `parser.add_argument ( "--chain_id" , type = int , default =0)` 12 `parser.add_argument ( "--n_iter" , type = int , default =15000)` 13 14 `args = parser.parse_args ()` 15 `# Creamos una semilla diferente para cada cadena. porque daria el mismo resultado.` 16 `np.random.seed (1000 + args.chain_id)` 17 18 `K = args.K` 19 `data = pd.read_csv(args. file )` 20 21 22 `try :` 23 `col_x = [c for c in data.columns if c.lower () in [ ’x’ , ’ tiempo_o_x ’ , ’time ’ , ’fre’ ]][0]` 24 `col_y = [c for c in data.columns if c.lower () in [ ’y’ , ’ senal_con_ruido ’ , ’temperature ’ , ’amplitude ’ ]][0]` 25 26 `x = data[col_x ]. values` 27 `y = data[col_y ]. values` 28 `print (f "[Cadena {args.chain_id }] Columnas detectadas con xito -> X: ’{col_x}’, Y: ’{col_y}’" )` 29 `except IndexError:` 30 `print (f " ERROR: No se pudieron mapear las columnas en ’{args. file}’." )` 31 `print (f "Columnas disponibles en tu archivo: {list(data.columns)} " )` 32 `sys.exit (1)` 33 34 `# =========================` 

8 

35 `# Modelos de suma de guassianas` 36 `# =========================` 37 `def gaussian_mixture (x, A, mu , sigma):` 38 `y_out = np.zeros_like(x)` 39 `for Ai , mi , si in zip (A, mu , sigma):` 40 `y_out += Ai * np.exp(-(x - mi)**2 / (2 * si **2))` 41 `return y_out` 42 `# LIKELIHOOD` 43 `def log_likelihood (theta , x, y, K, sigma_noise):` 44 `A = theta [0:K]` 45 `mu = theta[K:2*K]` 46 `sigma = theta [2*K:3*K]` 47 `y_hat = gaussian_mixture (x, A, mu , sigma)` 48 `return -0.5 * np. sum ((y - y_hat)**2 / sigma_noise **2)` 49 50 `# PRIOR` 51 `def log_prior(theta , K, xmin , xmax):` 52 `A = theta [0:K]` 53 `mu = theta[K:2*K]` 54 `sigma = theta [2*K:3*K]` 55 56 `if np. any (sigma < 0.01):` 57 `return -np.inf` 58 `if np. any (mu < xmin) or np. any (mu > xmax):` 59 `return -np.inf` 60 `if np. any (A < 0):` 61 `return -np.inf` 62 `lp = 0` 63 `lp -= np. sum ((A/20) **2)` 64 `lp -= np. sum (( sigma /10) **2)` 65 `return lp` 66 67 `# POSTERIOR` 68 `def log_posterior (theta , x, y, K, sigma_noise):` 69 `xmin , xmax = np. min (x), np. max (x)` 70 `lp = log_prior(theta , K, xmin , xmax)` 71 `if not np.isfinite(lp):` 72 `return -np.inf` 73 `return lp + log_likelihood (theta , x, y, K, sigma_noise)` 

74 75 

76 `# ORDENAMIENTO` 

77 78 

79 `def sort_parameters (theta , K):` 80 `A = theta [0:K]` 81 `mu = theta[K:2*K]` 

9 

82 `sigma = theta [2*K:3*K]` 83 `idx = np.argsort(mu)` 84 `return np.concatenate ([A[idx], mu[idx], sigma[idx ]])` 85 86 87 `def initialize(x, y, K):` 88 89 `#peaks , props = find_peaks(y, distance =50, height =0.05)` 90 `peaks , props = find_peaks(y, distance= max (1, len (x)//(20*K)), height=np. max (y)*0.02)` 91 92 `if len (peaks) >= K:` 93 94 `top_idx = np.argsort(props[ "peak_heights" ]) [:: -1][:K]` 95 `peaks = peaks[top_idx]` 96 `else :` 97 98 `peaks = np.linspace (0, len (x) -1, K).astype( int )` 99 

100 `A = y[peaks]` 101 `mu = x[peaks]` 102 103 `sigma = np.ones(K) * (x. max () - x. min ()) / 100` 104 105 `return np.concatenate ([A, mu , sigma ])` 106 107 108 109 110 `# =========================` 111 `# MCMC` 112 `# =========================` 113 114 `def run_mcmc(x, y, K, n_iter =20000):` 115 `sigma_noise = 0.02 #DEJAMOS EL RUIDO CONSNTANTE PARA ESTE EXPERIMENTO` 116 `theta_init = initialize(x, y, K)` 117 `perturbacion = np.random.normal (0, 0.05, size= len (theta_init))` 118 `theta = theta_init + perturbacion` 119 `theta = sort_parameters (theta , K)` 120 121 `current_lp = log_posterior(theta , x, y, K, sigma_noise)` 122 123 `samples = []` 124 `accept = 0` 125 126 

10 

127 `rango = x. max () - x. min ()` 128 `step_A = 0.05` 129 `step_mu = rango * 0.001` 130 `step_sigma = 0.05` 131 132 `steps = np.concatenate ([` 133 `np.ones(K) * step_A ,` 134 `np.ones(K) * step_mu ,` 135 `np.ones(K) * step_sigma` 136 `])` 137 138 `for i in range (n_iter):` 139 140 `proposal = theta + np.random.normal (0, 1, size= len (theta)) * steps` 141 `proposal = sort_parameters (proposal , K)` 142 `prop_lp = log_posterior(proposal , x, y, K, sigma_noise)` 143 144 `if np.log(np.random.rand ()) < (prop_lp - current_lp):` 145 `theta = proposal` 146 `current_lp = prop_lp` 147 `accept += 1` 148 149 `samples.append(theta.copy ())` 150 151 `if i % 1000 == 0 and i > 0:` 152 `print (f "[Cadena {args.chain_id }] Iter {i}/{ n_iter}, Tasa de a c e p t a c i n parcial: {accept /(i+1) :.3f}" )` 153 154 `print (f " [Cadena {args.chain_id }] Finalizada. Tasa de a c e p t a c i n final: {accept / n_iter :.3f}" )` 155 `return np.array(samples)` 156 157 158 159 `# RESULTADOS` 160 161 162 `def summarize(samples , K):` 163 `burn_in = int ( len (samples) * 0.3)` 164 `mean = np.mean(samples[burn_in :], axis =0)` 165 `return mean [0:K], mean[K:2*K], mean [2*K:3*K]` 166 167 168 

169 `if __name__ == "__main__" :` 170 `samples = run_mcmc(x, y, K, n_iter=args.n_iter)` 

11 

171 `A_est , mu_est , sigma_est = summarize(samples , K)` 172 173 `print (f "\n--- RESUMEN ESTIMACIONES CADENA {args.chain_id} ---" )` 174 `print ( "Amplitudes (A): " , np. round (A_est , 4))` 175 `print ( "Medias (mu): " , np. round (mu_est , 4))` 176 `print ( "Sigmas (sigma): " , np. round (sigma_est , 4))` 177 178 `out = np.concatenate ([A_est , mu_est , sigma_est ])` 179 `np.savetxt(f " result_chain_{args.chain_id }.txt" , out)` 180 `print (f " Resultados consolidados en: result_chain_{args.chain_id }.txt\n" )` 

## **6.3. analisis cadenas.py** 

Listing 4: An´alisis de resultados y generaci´on de gr´aficas 

1 `import numpy as np` 2 `import pandas as pd` 3 `import matplotlib.pyplot as plt` 4 `from scipy.signal import find_peaks` 5 6 `def leer_resultados_cadena (archivo , K):` 7 `with open (archivo , ’r’) as f:` 8 `lines = f.readlines ()` 9 `params = []` 10 `for line in lines [:K]:` 11 `A, mu , sigma = map ( float , line.strip ().split ())` 12 `params.append ([A, mu , sigma ])` 13 `return np.array(params)` 14 15 `def modelo_gaussiano (x, params):` 16 `y = np.zeros_like(x)` 17 `for A, mu , sigma in params:` 18 `y += A * np.exp(-(x - mu)**2 / (2 * sigma **2))` 19 `return y` 20 21 `def main ():` 22 `K = 3` 23 `resultados_cadenas = []` 24 25 `for chain_id in range (1, 5):` 26 `params = leer_resultados_cadena (f ’result_chain_{chain_id }. txt’ , K)` 27 `resultados_cadenas .append(params)` 28 29 `# Valores reales` 30 `valores_reales = np.array ([` 

12 

31 `[10.0 , 20.0 , 2.0] ,` 32 `[15.0 , 50.0 , 3.0] ,` 33 `[12.0 , 80.0 , 1.5]` 34 `])` 35 36 `# Promedio final` 37 `promedio_final = np.mean(resultados_cadenas , axis =0)` 38 39 `# Calcular errores` 40 `print ( "\n" + "=" *70)` 41 `print ( "RESULTADOS FINALES - C O M P A R A C I N CON VALORES REALES" )` 42 `print ( "=" *70)` 43 `print (f "{’ P a r m e t r o ’:<12} {’Real ’:<12} {’Estimado ’:<12} ( %) ’:<12}" )` 44 `print ( "-" *70)` 45 46 `for i in range (K):` 47 `(f "A{i+1} {valores_reales [i ,0]: <12.4f} {` 

```
print(f"{’ P a rm e t r o’:<12}{’Real ’:<12}{’Estimado ’:<12}{’Error
( %) ’:<12}")
print("-"*70)
```

```
foriinrange(K):
print(f"A{i+1}{valores_reales [i ,0]: <12.4f}{
promedio_final [i ,0]: <12.4f}{abs(( promedio_final [i,0]-
valores_reales [i ,0])/valores_reales [i ,0]*100) : <12.2f}")
print(f"mu{i+1}{valores_reales [i ,1]: <12.4f}{
promedio_final [i ,1]: <12.4f}{abs(( promedio_final [i,1]-
valores_reales [i ,1])/valores_reales [i ,1]*100) : <12.2f}")
print(f"sigma{i+1}{valores_reales [i ,2]: <12.4f}{
promedio_final [i ,2]: <12.4f}{abs(( promedio_final [i,2]-
valores_reales [i ,2])/valores_reales [i ,2]*100) : <12.2f}")
print("-"*70)
```

48 

49 50 51 52 53 54 55 56 57 58 59 

```
#Graficarc o m p a r a c indecadenas
fig ,axes=plt.subplots (3,3,figsize =(15,12))
parametros=[’Amplitud(A)’,’Centro()’,’Ancho()’]
```

```
foriinrange(K):
```

```
forj,param_idxinenumerate ([0,1,2]):
```

```
ax=axes[i,j]
```

```
valores_cadenas=[ resultados_cadenas [chain ][i,
```

```
param_idx]forchaininrange (4)]
```

60 

61 

62 

63 

64 

```
ax.bar(range (1,5),valores_cadenas ,alpha =0.7,label=’
Cadenas ’)
```

```
ax.axhline(y= valores_reales [i,param_idx],color=’r’,
’--’
linestyle=,
```

```
label=f’Real:{valores_reales [i,param_idx ]:.2
f}’)
```

```
ax.axhline(y= promedio_final [i,param_idx],color=’g’,
’-’
linestyle=,
```

```
label=f’Promedio:{promedio_final [i,param_idx
]:.2f}’)
```

13 

65 `ax.set_xlabel( ’Cadena ’)` 66 `ax.set_ylabel(parametros[param_idx ])` 67 `ax.set_title(f ’Pico {i+1} - {parametros[param_idx ]}’)` 68 `ax.legend ()` 69 `ax.grid(True , alpha =0.3)` 70 71 `plt.suptitle( ’ C o m p a r a c i n de Cadenas MCMC por P a r m e t r o ’ , fontsize =16)` 72 `plt.tight_layout ()` 73 `plt.savefig( ’analisis_cadenas .png’ , dpi =150)` 74 `print ( "\ n G r f i c a guardada: analisis_cadenas .png" )` 75 76 `# Reconstruir s e a l` 77 `df = pd.read_csv( ’senal.csv’)` 78 `x = df[ ’x’ ]. values` 79 `y_original = df[ ’y’ ]. values` 80 `y_reconstruida = modelo_gaussiano (x, promedio_final )` 81 82 `plt.figure(figsize =(12 , 6))` 83 `plt.plot(x, y_original , ’b-’ , alpha =0.7, label= ’ S e a l Original (con ruido)’ , linewidth =1)` 84 `plt.plot(x, y_reconstruida , ’r-’ , label= ’ S e a l Reconstruida ’ , linewidth =2)` 85 86 `# Marcar picos` 87 `for i in range (K):` 88 `plt.axvline(x= promedio_final [i, 1], color= ’gray ’ , linestyle= ’:’ , alpha =0.5)` 89 `plt.text( promedio_final [i, 1], promedio_final [i, 0],` 90 `f ’Pico {i+1}\ n ={ promedio_final [i ,1]:.1f}’ ,` 91 `ha= ’center ’ , va= ’bottom ’ , fontsize =9)` 92 93 `plt.xlabel( ’x’)` 94 `plt.ylabel( ’y’)` 95 `plt.title( ’ S e a l Reconstruida vs S e a l Original ’)` 96 `plt.legend ()` 97 `plt.grid(True , alpha =0.3)` 98 `plt.savefig( ’senal_reconstruida .png’ , dpi =150)` 99 `print ( " G r f i c a guardada: senal_reconstruida .png" )` 100 101 `plt.show ()` 102 103 `if __name__ == "__main__" :` 104 `main ()` 

14 

## **7. Conclusiones** 

En este se muestra exitosamente la implementaci´on de un algoritmo MCMC paralelo para la inferencia bayesiana de par´ametros con un modelo de suma gaussianas. Las principales observaciones son: 

- El paralelismo a nivel de tareas permite una exploraci´on m´as eficiente del espacio de par´ametros sin aumentar significativamente el tiempo de c´omputo. 

- Aunque los errores de estimacion en algunos par´ametros fue alto el m´etodo ha demostrado que es una alternativa viable para este tipo de problemas. Una de las posibles causas de este error es el alto ruido de la se˜nal o una configuraci´on m´as estricta en la prior. 

- El uso de m´ultiples cadenas con diferentes semillas aleatorias permite verificar la convergencia de cada pico gaussiano, al mostrar todas las cadenas la detecci´on del mismo pico. 

- La implementaci´on es escalable y puede adaptarse f´acilmente para un n´umero diferente de picos gaussianos. 

15 

