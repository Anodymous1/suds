import numpy as np

# =====================================================================
# 1. Individual Parameter Prior Bounds
# =====================================================================

# --- Linear Space (Uniform Priors) ---
alpha_min, alpha_max = 0.5, 3.0
beta_min, beta_max = 2.0, 10.0
gamma_min, gamma_max = -1.0, 2.0
beta0_min, beta0_max = -0.5, 1.0

# --- Base-10 Log Space (Log-Uniform Priors) ---
log_rho_s_min, log_rho_s_max = 3.0, 10.0                          # log10(10^3) to log10(10^10)
log_r_s_min, log_r_s_max = -2.0, 2.0                             # log10(10^-2) to log10(10^2)
log_r_star_over_r_s_min, log_r_star_over_r_s_max = -3.0, 0.0    # log10(10^-3) to log10(1)
log_r_a_over_r_star_min, log_r_a_over_r_star_max = -1.0, 3.0      # log10(10^-1) to log10(10^3)
log_uncertainty_min, log_uncertainty_max = np.log10(0.01), np.log10(20.0) # -2.0 to ~1.301


# =====================================================================
# 2. Aggregated Lists of Bounds (Log-Uniforms in Log-Space)
# =====================================================================

# ---- WITHOUT UNCERTAINTY (8 Parameters) ----
mins_without_uncertainty = [
    alpha_min, 
    beta_min, 
    gamma_min, 
    log_rho_s_min, 
    log_r_s_min, 
    log_r_star_over_r_s_min, 
    log_r_a_over_r_star_min, 
    beta0_min
]

maxs_without_uncertainty = [
    alpha_max, 
    beta_max, 
    gamma_max, 
    log_rho_s_max, 
    log_r_s_max, 
    log_r_star_over_r_s_max, 
    log_r_a_over_r_star_max, 
    beta0_max
]


# ---- WITH UNCERTAINTY (9 Parameters) ----
mins_with_uncertainty = [
    alpha_min, 
    beta_min, 
    gamma_min, 
    log_rho_s_min, 
    log_r_s_min, 
    log_r_star_over_r_s_min, 
    log_r_a_over_r_star_min, 
    beta0_min, 
    log_uncertainty_min
]

maxs_with_uncertainty = [
    alpha_max, 
    beta_max, 
    gamma_max, 
    log_rho_s_max, 
    log_r_s_max, 
    log_r_star_over_r_s_max, 
    log_r_a_over_r_star_max, 
    beta0_max, 
    log_uncertainty_max
]