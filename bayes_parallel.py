import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import argparse
import pandas as pd
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, required=True)
parser.add_argument("--K", type=int, required=True)
parser.add_argument("--chain_id", type=int, default=0)
parser.add_argument("--n_iter", type=int, default=15000)

args = parser.parse_args()
# Creamos una semilla diferente para cada cadena. porque daria el mismo resultado. 
np.random.seed(1000 + args.chain_id)

K = args.K
data = pd.read_csv(args.file)


try:
    col_x = [c for c in data.columns if c.lower() in ['x', 'tiempo_o_x', 'time', 'fre']][0]
    col_y = [c for c in data.columns if c.lower() in ['y', 'senal_con_ruido', 'temperature', 'amplitude']][0]

    x = data[col_x].values
    y = data[col_y].values
    print(f"[Cadena {args.chain_id}] Columnas detectadas con éxito -> X: '{col_x}', Y: '{col_y}'")
except IndexError:
    print(f" ERROR: No se pudieron mapear las columnas en '{args.file}'.")
    print(f"Columnas disponibles en tu archivo: {list(data.columns)}")
    sys.exit(1)

# =========================
# Modelos de suma de guassianas 
# =========================
def gaussian_mixture(x, A, mu, sigma):
    y_out = np.zeros_like(x)
    for Ai, mi, si in zip(A, mu, sigma):
        y_out += Ai * np.exp(-(x - mi)**2 / (2 * si**2))
    return y_out
# LIKELIHOOD
def log_likelihood(theta, x, y, K, sigma_noise):
    A     = theta[0:K]
    mu    = theta[K:2*K]
    sigma = theta[2*K:3*K]
    y_hat = gaussian_mixture(x, A, mu, sigma)
    return -0.5 * np.sum((y - y_hat)**2 / sigma_noise**2)

# PRIOR
def log_prior(theta, K, xmin, xmax):
    A     = theta[0:K]
    mu    = theta[K:2*K]
    sigma = theta[2*K:3*K]

    if np.any(sigma < 0.01):
        return -np.inf
    if np.any(mu < xmin) or np.any(mu > xmax):
        return -np.inf
    if np.any(A < 0):
        return -np.inf
    lp = 0
    lp -= np.sum((A/20)**2)
    lp -= np.sum((sigma/10)**2)
    return lp

# POSTERIOR
def log_posterior(theta, x, y, K, sigma_noise):
    xmin, xmax = np.min(x), np.max(x)
    lp = log_prior(theta, K, xmin, xmax)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, K, sigma_noise)


# ORDENAMIENTO


def sort_parameters(theta, K):
    A     = theta[0:K]
    mu    = theta[K:2*K]
    sigma = theta[2*K:3*K]
    idx   = np.argsort(mu)
    return np.concatenate([A[idx], mu[idx], sigma[idx]])


def initialize(x, y, K):
   
    #peaks, props = find_peaks(y, distance=50, height=0.05)
    peaks, props = find_peaks(y, distance=max(1, len(x)//(20*K)), height=np.max(y)*0.02)

    if len(peaks) >= K:
        
        top_idx = np.argsort(props["peak_heights"])[::-1][:K]
        peaks   = peaks[top_idx]
    else:
       
        peaks = np.linspace(0, len(x)-1, K).astype(int)

    A     = y[peaks]
    mu    = x[peaks]
    
    sigma = np.ones(K) * (x.max() - x.min()) / 100

    return np.concatenate([A, mu, sigma])




# =========================
# MCMC 
# =========================

def run_mcmc(x, y, K, n_iter=20000):
    sigma_noise = 0.02  #DEJAMOS EL RUIDO CONSNTANTE PARA ESTE EXPERIMENTO 
    theta_init = initialize(x, y, K)
    perturbacion = np.random.normal(0, 0.05, size=len(theta_init))
    theta = theta_init + perturbacion
    theta = sort_parameters(theta, K)

    current_lp  = log_posterior(theta, x, y, K, sigma_noise)

    samples = []
    accept  = 0

   
    rango      = x.max() - x.min()
    step_A     = 0.05          
    step_mu    = rango * 0.001  
    step_sigma = 0.05           

    steps = np.concatenate([
        np.ones(K) * step_A,
        np.ones(K) * step_mu,
        np.ones(K) * step_sigma
    ])

    for i in range(n_iter):
      
        proposal   = theta + np.random.normal(0, 1, size=len(theta)) * steps
        proposal   = sort_parameters(proposal, K)
        prop_lp    = log_posterior(proposal, x, y, K, sigma_noise)

        if np.log(np.random.rand()) < (prop_lp - current_lp):
            theta      = proposal
            current_lp = prop_lp
            accept    += 1

        samples.append(theta.copy())

        if i % 1000 == 0 and i > 0:
            print(f"[Cadena {args.chain_id}] Iter {i}/{n_iter}, Tasa de aceptación parcial: {accept/(i+1):.3f}")

    print(f"🎉 [Cadena {args.chain_id}] Finalizada. Tasa de aceptación final: {accept / n_iter:.3f}")
    return np.array(samples)



# RESULTADOS


def summarize(samples, K):
    burn_in = int(len(samples) * 0.3)
    mean    = np.mean(samples[burn_in:], axis=0)
    return mean[0:K], mean[K:2*K], mean[2*K:3*K]



if __name__ == "__main__":
    samples                  = run_mcmc(x, y, K, n_iter=args.n_iter)
    A_est, mu_est, sigma_est = summarize(samples, K)

    print(f"\n--- RESUMEN ESTIMACIONES CADENA {args.chain_id} ---")
    print("Amplitudes (A):   ", np.round(A_est, 4))
    print("Medias (mu):      ", np.round(mu_est, 4))
    print("Sigmas (sigma):   ", np.round(sigma_est, 4))

    out = np.concatenate([A_est, mu_est, sigma_est])
    np.savetxt(f"result_chain_{args.chain_id}.txt", out)
    print(f" Resultados consolidados en: result_chain_{args.chain_id}.txt\n")