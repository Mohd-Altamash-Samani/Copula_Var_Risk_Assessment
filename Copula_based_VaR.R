# Load libraries
library(PerformanceAnalytics)
library(rugarch)
library(copula)
library(MASS)
library(tseries)
library(fGarch)
library(quantmod)


# -------------------- Get Stock Data --------------------
# US Stocks
getSymbols(c("AAPL", "MSFT", "AMZN", "NVDA", "GOOGL"), from = "2020-01-01", to = "2024-01-01")
us_prices <- na.omit(merge(Cl(AAPL), Cl(MSFT), Cl(AMZN), Cl(NVDA), Cl(GOOGL)))
us_returns <- na.omit(Return.calculate(us_prices, method = "log"))

# Indian Stocks
getSymbols(c("TCS.NS", "HDFCBANK.NS", "RELIANCE.NS", "SUNPHARMA.NS", "TATAMOTORS.NS"), from = "2020-01-01", to = "2024-01-01")
ind_prices <- na.omit(merge(Cl(TCS.NS), Cl(HDFCBANK.NS), Cl(RELIANCE.NS), Cl(SUNPHARMA.NS), Cl(TATAMOTORS.NS)))
ind_returns <- na.omit(Return.calculate(ind_prices, method = "log"))

# Portfolio returns
us_port <- rowMeans(us_returns)
ind_port <- rowMeans(ind_returns)

# Confidence level
confidence <- 0.99


# -------------------- Traditional VaR --------------------
# US
us_hist_var       <- quantile(us_port, probs = 1 - confidence)
us_param_normal   <- -qnorm(confidence, mean(us_port), sd(us_port))
us_param_t        <- -qt(confidence, df = 5) * sd(us_port) + mean(us_port)
us_mc_sim         <- rnorm(10000, mean(us_port), sd(us_port))
us_mc_var         <- quantile(us_mc_sim, probs = 1 - confidence)

# India
ind_hist_var      <- quantile(ind_port, probs = 1 - confidence)
ind_param_normal  <- -qnorm(confidence, mean(ind_port), sd(ind_port))
ind_param_t       <- -qt(confidence, df = 5) * sd(ind_port) + mean(ind_port)
ind_mc_sim        <- rnorm(10000, mean(ind_port), sd(ind_port))
ind_mc_var        <- quantile(ind_mc_sim, probs = 1 - confidence)


# -------------------- Copula-Based VaR --------------------
# Define copula families
copulas <- list(
  "Clayton"  = claytonCopula(dim = ncol(us_returns)),
  "Gumbel"   = gumbelCopula(dim = ncol(us_returns)),
  "Frank"    = frankCopula(dim = ncol(us_returns)),
  "Gaussian" = normalCopula(dim = ncol(us_returns)),
  "t"        = tCopula(dim = ncol(us_returns))
)

set.seed(123)

# Function to compute copula VaR
compute_copula_var <- function(returns, copula_model, df = 5, n_sim = 10000, conf = 0.99) {
  # Step 1: Fit marginal t-distributions
  marginals <- lapply(as.data.frame(returns), function(x) {
    fitdistr(x, densfun = "t")
  })
  
  # Step 2: Convert returns to pseudo-observations
  u <- pobs(as.matrix(returns))
  
  # Step 3: Fit copula to pseudo-observations
  fit <- fitCopula(copula_model, u, method = "ml")
  
  # Step 4: Simulate from fitted copula
  sim <- rCopula(n_sim, fit@copula)
  
  # Step 5: Transform using inverse CDF of fitted marginal t-distribution
  q_sim <- sapply(1:ncol(sim), function(i) {
    u_col <- sim[, i]
    est  <- marginals[[i]]$estimate
    qt(u_col, df = est["df"]) * est["s"] + est["m"]
  })
  
  # Step 6: Compute portfolio return and VaR
  port_returns <- rowMeans(q_sim)
  return(quantile(port_returns, probs = 1 - conf))
}


# Calculate VaR for each copula
us_copula_vars  <- sapply(copulas, function(cop) compute_copula_var(us_returns, cop))
ind_copula_vars <- sapply(copulas, function(cop) compute_copula_var(ind_returns, cop))


# Load required library
library(ggplot2)
library(tidyr)
library(dplyr)

# Create your VaR results data frame
var_results <- data.frame(
  Method = c("Historical", "Parametric Normal", "Parametric t", "Monte Carlo (Normal)",
             "Clayton", "Gumbel", "Frank", "Gaussian", "t"),
  India_VaR = c(-0.04010017, -0.03344223, -0.04640530, -0.03273671,
                -0.04423917, -0.03100893, -0.02920093, -0.03633708, -0.03668228),
  US_VaR = c(-0.05313744, -0.04958316, -0.06924235, -0.04731585,
             -0.05694304, -0.04504377, -0.04202898, -0.05451687, -0.05604806)
)

# Convert data to long format for ggplot
var_long <- var_results %>%
  pivot_longer(cols = c(India_VaR, US_VaR), names_to = "Market", values_to = "VaR")

# Plot
ggplot(var_long, aes(x = reorder(Method, VaR), y = VaR, fill = Market)) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_flip() +  # horizontal bars for readability
  labs(
    title = "Comparison of Value at Risk (VaR) at 99% Confidence Level",
    x = "Method",
    y = "VaR",
    fill = "Market"
  ) +
  theme_minimal() +
  scale_fill_manual(values = c("India_VaR" = "#1f78b4", "US_VaR" = "#e31a1c")) +
  theme(text = element_text(size = 12))


# -------------------- Final VaR Table --------------------
var_results <- data.frame(
  Method    = c("Historical", "Parametric Normal", "Parametric t", "Monte Carlo (Normal)", names(copulas)),
  India_VaR = c(ind_hist_var, ind_param_normal, ind_param_t, ind_mc_var, ind_copula_vars),
  US_VaR    = c(us_hist_var, us_param_normal, us_param_t, us_mc_var, us_copula_vars)
)

print(var_results)
