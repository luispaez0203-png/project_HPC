# Inferencia Bayesiana Paralela con MCMC

### Ajuste de mezclas gaussianas en señales experimentales

![Gráfica de señal](G359p5P0p0_corre.png)

*High Performance Computing - HPCF*

**Autor:** Luis Eduardo Cardoso Paez  
**Profesor:** Edwin Flórez Gómez  
**Universidad de Puerto Rico**

---

## Resumen

En este proyecto hacemos uso de la Inferencia Bayesiana para la estimación de parámetros en señales astronómicas. Construimos un modelo de suma de picos gaussianos, que tenga como parámetro de entrada el número posibles picos, $k$, que pueda tener la señal. Para hacer esto, implementamos un algoritmo de Markov Chain Monte Carlo (MCMC) con el método de Metropoli-Hastings para estimar lo parámetros de todos los $k$ picos gaussianos superpuestos con el ruido. Desde el enfoque de HPC usamos el paralelismo a nivel de tareas ejecutando 4 cadenas MCMC independientes simultáneamente en diferentes núcleos de la computadora o de clúster, esto permite una exploración eficiente del espacio de parámetros, mostrando si un pico detectado aparece en todas las cadena. Finalmente, los resultados de todas las cadenas se combinan mediante un promedio para obtener una estimación general de los parámetros de cada pico.

---

## Descripción del Proyecto

El proyecto estima, a partir de un archivo CSV con una señal experimental, los parámetros de cada $k$ picos gaussianos. Estos parámetros son:

- **$A$** — Amplitud de cada pico el cual esta relacionado con la intensidad del gas
- **$\mu$** — Centro (posición) de cada pico que relacionada con la velocidad a la que se mueve
- **$\sigma$** — Ancho (desviación estándar) de cada pico que relacionado con la temperatura del gas

El modelo asume la forma:

$$ y(x) = \sum_{i=1}^{k} A_i \cdot \exp\left( -\frac{(x - \mu_i)^2}{2\sigma_i^2} \right) + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma_{\text{ruido}}^2) $$

Donde $k$, es la cantidad de picos que se cree que tiene la señal astronómica.

### Fundamento Bayesiano

La inferencia bayesiana se basa en el teorema de Bayes:

$$ P(\theta | \text{Datos}) = \frac{P(\text{Datos} | \theta) \cdot P(\theta)}{P(\text{Datos})} $$

Aplicando logaritmo:

$$ \ln P(\theta | \text{Datos}) = \ln P(\text{Datos} | \theta) + \ln P(\theta) - \ln P(\text{Datos}) $$

La ecuación (3), es la que usaremos para hacer la inferencia basados en el modelo que se muestra a continuación.

#### Verosimilitud (Likelihood) - $P(\text{Datos} | \theta)$

La verosimilitud mide la capacidad del modelo para explicar los datos observados:

$$ \ln P(\text{Datos} | \theta) = -\frac{1}{2} \sum_{i} (y_i - \hat{y}_i)^2 \sigma^2 $$

Donde, $y_i$ son los datos observados, $\hat{y}_i$ es el valor predicho por el modelo, $\sigma^2$ es la varianza del ruido.

#### Distribución a Priori - $P(\theta)$

La distribución a priori incorpora restricciones físicas y conocimiento previo del sistema:

$$ \begin{aligned} A &> 0 \quad \text{(amplitud positiva)} \\ \sigma &> 0 \quad \text{(ancho positivo)} \\ \mu &\in [x_{\text{min}}, x_{\text{max}}] \quad \text{(centro dentro del rango de medición)} \end{aligned} $$

#### Distribución a Posteriori - $P(\theta | \text{Datos})$

La distribución a posteriori representa la probabilidad actualizada de los parámetros después de observar los datos:

$$ P(\theta | \text{Datos}) \propto \mathcal{L} \cdot \text{Prior} $$

Es decir:

$$ \text{Posterior} \propto \text{Likelihood} \times \text{Prior} $$

Esta distribución combina la información de los datos observados (verosimilitud) con el conocimiento previo (prior) para obtener una estimación más robusta de los parámetros.

### Tipo de Paralelismo

Para este proyecto usaremos **Task Parallelism** (Paralelismo a nivel de tareas):

- Se ejecutan 4 cadenas MCMC independientes simultáneamente, cada una en un núcleo distinto del clúster o de la maquina local.
- Cada cadena usa una semilla aleatoria diferente (seed = 1000 + chain_id), garantizando exploración diversa del espacio de parámetros.
- Al finalizar, los resultados de cada cadena se combinan promediando.

**Esquema de paralelización:**

| Core | Cadena | Salida |
|------|--------|--------|
| Core 1 | Cadena 1 (seed=1001) | → result_chain_1.txt |
| Core 2 | Cadena 2 (seed=1002) | → result_chain_2.txt |
| Core 3 | Cadena 3 (seed=1003) | → result_chain_3.txt |
| Core 4 | Cadena 4 (seed=1004) | → result_chain_4.txt |

#### Ventajas sobre el modo lineal

- El tiempo de ejecución es similar al de 1 sola cadena
- Mejor exploración del espacio de parámetros
- Verificación de posibles picos para evitar falsos positivos

---

## Archivos del Repositorio

Aunque se tienen señales astronómicas reales para poder realizar el experimento NO TENGO PERMITIDO COMPARTIR estas señales hasta que las investigaciones que se están haciendo con esas hayan concluido. Por tal motivo, podemos crear señales sintéticas para hacer el experimento. En el repositorio vamos a encontrar los siguientes archivo:

```bash
generar_senal.py        # Genera la senal sintetica con K picos gaussianos
bayes_parallel.py       # Algoritmo MCMC paralelo (una cadena por ejecucion)
analisis_cadenas.py     # Combina resultados y genera graficas
README.md               # Este archivo
