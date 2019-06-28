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
################################################################################

################################################################################
houlsbyK <- function(A, B, params = list(alpha = 1, rho1 = 1, rho2 =1, rho3 = 1, rho4 = 1), delta = 1e-9){
  alpha = params$alpha
  rho1 = params$rho1
  rho2 = params$rho2
  rho3 = params$rho3
  rho4 = params$rho4
  d <- dim(A)[2]
  N <- dim(A)[1]
  K <- matrix(NA, N, N)
  K_ik <- matrix(NA, N, N)
  K_jl <- matrix(NA, N, N)
  K_il <- matrix(NA, N, N)
  K_jk <- matrix(NA, N, N)
  for(i in 1:N){
    for(j in i:N){
    K_ik[i,j] = alpha*exp(-.5*rho1*(A[i, 1] - A[j, 1])^2 -.5*rho2*(A[i, 2] - A[j, 2])^2 -.5*rho3*(A[i, 3] - A[j, 3])^2 -.5*rho4*(A[i, 4] - A[j, 4])^2)
    K_ik[j,i] = K_ik[i,j]
    K_jl[i,j] = alpha*exp(-.5*rho1*(B[i, 1] - B[j, 1])^2 -.5*rho2*(B[i, 2] - B[j, 2])^2 -.5*rho3*(B[i, 3] - B[j, 3])^2 -.5*rho4*(B[i, 4] - B[j, 4])^2) 
    K_jl[j,i] = K_jl[i,j]
    K_il[i,j] = alpha*exp(-.5*rho1*(A[i, 1] - B[j, 1])^2 -.5*rho2*(A[i, 2] - B[j, 2])^2 -.5*rho3*(A[i, 3] - B[j, 3])^2 -.5*rho4*(A[i, 4] - B[j, 4])^2)
    K_il[j,i] = K_il[i,j]
    K_jk[i,j] = alpha*exp(-.5*rho1*(B[i, 1] - A[j, 1])^2 -.5*rho2*(B[i, 2] - A[j, 2])^2 -.5*rho3*(B[i, 3] - A[j, 3])^2 -.5*rho4*(B[i, 4] - A[j, 4])^2)
    K_jk[j,i] = K_jk[i,j]
    }
  }

  K <- K_ik + K_jl - K_il - K_jk
  diag(K) <- diag(K) + delta #sigma
  return(K)
}
################################################################################

################################################################################
## Information gain
IG <- function(mu, sigmasq){
  C <- sqrt((pi * log(2))/2)
  h <- function(z){
    -z*log(z) - (1-z)*log(1-z)
  }
  first <- h(pnorm(mu*sqrt(sigmasq + 1)))
  second <- C*sqrt(sigmasq + C^2)*exp(-(mu^2)*(2*(sigmasq + C^2))^(-1))
  return(first - second)
}
################################################################################

################################################################################
##
postpred <- function(X1, X1test, X2, X2test, params = list(alpha = 1, rho1 = 1, rho2 = 1, rho3 = 1, rho4 = 1), uhat){
  N1 <- dim(X1)[1]
  N2 <- dim(X1test)[1]
  N <- N1 + N2
  Kall <- houlsbyK(rbind(X1, X1test), rbind(X2, X2test), params = params)
  Kstar <- Kall[1:N1, (N1 + 1):N]
  Kstarstar <- Kall[(N1 + 1):N, (N1 + 1):N]
  K <- Kall[1:N1, 1:N1]
  fhat <- t(Kstar) %*% solve(K) %*% uhat
  etahat <- c(uhat)
  phat <- exp(etahat)/(1 + exp(etahat))
  W <- diag(as.vector(phat*(1-phat)))
  sigmasq <- Kstarstar - t(Kstar) %*% solve(K + solve(W)) %*% Kstar
  return(list(mu = fhat, sigmasq = sigmasq))
}
################################################################################

