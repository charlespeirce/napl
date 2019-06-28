################################################################################
## Load packages
# install.packages("rstan")
# install.packages("rstanarm")
# install.packages(c("coda","mvtnorm","devtools","loo"))
# library(devtools)
# devtools::install_github("rmcelreath/rethinking")
# https://mc-stan.org/users/documentation/case-studies/nngp.html
library(rethinking)
library("rstan")
library("rstanarm")
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
source("houlsbyfun.R")
################################################################################

corGP <- matrix(NA, nrow = 40, ncol = 10)
for(j in 1:10){
################################################################################
## Sample size
N <- 5
## Generate attributes
# Alternative 1
x1 <- runif(N, 0, 1)
a1 <- runif(N, 0, 1)
b1 <- runif(N, 0, 1)
c1 <- runif(N, 0, 1)
X1 <- cbind(x1, a1, b1, c1)
# Alternative 2
x2 <- runif(N, 0, 1)
a2 <- runif(N, 0, 1)
b2 <- runif(N, 0, 1)
c2 <- runif(N, 0, 1)
X2 <- cbind(x2, a2, b2, c2)
# Generate probs
eta <- 1*(x1 - x2) - 2*(a1 - a2) + 3*(b1 - b2) - 4*(c1 - c2)
p <- exp(eta)/(1 + exp(eta))
# Generate choices
y <- rbinom(N, 1, p)
################################################################################

for(r in 1:35){
################################################################################
# Fit initial model
initf1 <- function() {
  list(alpha = 1, rho1 = 1, rho2 = 1, rho3 = 1, rho4 = 1)
}
iter = 300
chains = 1
fit1 <- stan(file = "houlsby.stan", 
             data=list(x1 = x1, x2 = x2, a1 = a1, a2 = a2, b1 = b1, b2 = b2, c1 = c1, c2 = c2, N1 = length(y)), 
             iter = iter, chains = chains, verbose = TRUE, control = list(adapt_delta=0.8), init = initf1)
sims1 <- extract(fit1)
uhat <- colMeans(sims1$f)
alpha <- median(sims1$alpha)
rho1 <- median(sims1$rho1)
rho2 <- median(sims1$rho2)
rho3 <- median(sims1$rho3)
rho4 <- median(sims1$rho4)

# Correlation on test data
if((r + 5) %in% c(10, 20, 30, 40)){
  ## Test data
  x1test <- runif(1000, 0, 1)
  a1test <- runif(1000, 0, 1)
  b1test <- runif(1000, 0, 1)
  c1test <- runif(1000, 0, 1)
  X1test <- cbind(x1test, a1test, b1test, c1test)

  x2test <- runif(1000, 0, 1)
  a2test <- runif(1000, 0, 1)
  b2test <- runif(1000, 0, 1)
  c2test <- runif(1000, 0, 1)
  X2test <- cbind(x2test, a2test, b2test, c2test)

  etatest <- 1*(x1test - x2test) - 2*(a1test - a2test) + 3*(b1test - b2test) - 4*(c1test - c2test)
  ptest <- exp(etatest)/(1 + exp(etatest))
  ytest <- rbinom(length(ptest), 1, ptest)

  N1 <- dim(X1)[1]
  N2 <- dim(X1test)[1]
  N <- N1 + N2
  Kall <- houlsbyK(rbind(X1, X1test), rbind(X2, X2test), params = list(alpha = alpha, rho1 = rho1, rho2 = rho2, rho3 = rho3, rho4 = rho4))
  Kstar <- Kall[1:N1, (N1 + 1):N]
  K <- Kall[1:N1, 1:N1]
  fhat <- t(Kstar) %*% solve(K) %*% uhat
  cat("Test correlation: ", cor(fhat, etatest), "\n")
  corGP[r, j] <- cor(fhat, etatest)
}
################################################################################

################################################################################
## Maximum information gain
## Newdata
x1new <- runif(1000, 0, 1)
a1new <- runif(1000, 0, 1)
b1new <- runif(1000, 0, 1)
c1new <- runif(1000, 0, 1)
X1new <- cbind(x1new, a1new, b1new, c1new)

x2new <- runif(1000, 0, 1)
a2new <- runif(1000, 0, 1)
b2new <- runif(1000, 0, 1)
c2new <- runif(1000, 0, 1)
X2new <- cbind(x2new, a2new, b2new, c2new)

# Correlation on test data
alpha <- median(sims1$alpha)
rho1 <- median(sims1$rho1)
rho2 <- median(sims1$rho2)
rho3 <- median(sims1$rho3)
rho4 <- median(sims1$rho4)

N1 <- dim(X1)[1]
N2 <- 1
N <- N1 + N2
IGs <- c()
post1 <- postpred(X1, X1new, X2, X2new, params = list(alpha = alpha, rho1 = rho1, rho2 = rho2, rho3 = rho3, rho4 = rho4), uhat)
for(i in 1:1000){
  postmu <- post1$mu[i]
  postsigmasq <- post1$sigmasq[i, i]
  IGs[i] <- IG(postmu, postsigmasq)
}
IGopt <- which(IGs == max(IGs))[1]

x1 <- c(x1, x1new[IGopt])
a1 <- c(a1, a1new[IGopt])
b1 <- c(b1, b1new[IGopt])
c1 <- c(c1, c1new[IGopt])
X1 <- rbind(X1, X1new[IGopt, ])

x2 <- c(x2, x2new[IGopt])
a2 <- c(a2, a2new[IGopt])
b2 <- c(b2, b2new[IGopt])
c2 <- c(c2, c2new[IGopt])
X2 <- rbind(X2, X2new[IGopt, ])

etanew <- 1*(x1new[IGopt] - x2new[IGopt]) - 2*(a1new[IGopt] - a2new[IGopt]) + 3*(b1new[IGopt] - b2new[IGopt]) - 4*(c1new[IGopt] - c2new[IGopt])
pnew <- exp(etanew)/(1 + exp(etanew))
# Generate choices
ynew <- rbinom(1, 1, pnew)
y <- c(y, ynew)
}
}
################################################################################

################################################################################
write.csv(corGP, "results/linearGP.csv")
################################################################################
