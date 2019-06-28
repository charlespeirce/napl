################################################################################
##
corOracle <- c()
corAdd <- c()
corGP <- c()
################################################################################

for(j in 1:10){
################################################################################
## Training data
options(scipen = 99, digits = 2)
Nstart <- 10
x1 <- runif(Nstart, 0, 1)
a1 <- runif(Nstart, 0, 1)
b1 <- runif(Nstart, 0, 1)
c1 <- runif(Nstart, 0, 1)
X1 <- cbind(x1, a1, b1, c1)

x2 <- runif(Nstart, 0, 1)
a2 <- runif(Nstart, 0, 1)
b2 <- runif(Nstart, 0, 1)
c2 <- runif(Nstart, 0, 1)
X2 <- cbind(x2, a2, b2, c2)

eta <- 1*((x1 - .5)^2 - (x2 - .5)^2) - 2*((a1 - .4)^2 - (a2 - .4)^2) + 3*((b1 - .6)^2 - (b2 - .6)^2) - 4*((c1 - .5)^2 - (c2 - .5)^2)
p <- exp(eta)/(1 + exp(eta))
y <- rbinom(length(p), 1, p)

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

etatest <- 1*((x1test - .5)^2 - (x2test - .5)^2) - 2*((a1test - .4)^2 - (a2test - .4)^2) + 3*((b1test - .6)^2 - (b2test - .6)^2) - 4*((c1test - .5)^2 - (c2test - .5)^2)
ptest <- exp(etatest)/(1 + exp(etatest))
ytest <- rbinom(length(ptest), 1, ptest)
################################################################################

################################################################################
# Oracle glm
xdiff <- log(x1) - log(x2)
adiff <- log(a1) - log(a2)
bdiff <- log(b1) - log(b2)
cdiff <- log(c1) - log(c2)
glm1 <- glm(y ~ xdiff + adiff + bdiff + cdiff, family = binomial(link = "logit"))
predglm1 <- predict(glm1, newdata = data.frame(xdiff = x1test - x2test, adiff = a1test - a2test, bdiff = b1test - b2test, cdiff = c1test - c2test))
cat("Oracle corr: ", cor(predglm1, etatest), "\n")
corOracle[j] <- cor(predglm1, etatest)
################################################################################

################################################################################
## Additive
source("additivefuns.R")
splinereg <- NRspline(X1, X2, y)
betas <- splinereg$betas
Utest <- make.U(X1test, X2test)
predtest <- Utest %*% betas
cor(predtest, etatest)
cat("Additive corr: ", cor(predtest, etatest), "\n")
corAdd[j] <- cor(predtest, etatest)
################################################################################

################################################################################
## GP Houlsby
# Fit initial model
source("houlsbyfun.R")
initf1 <- function() {
  list(alpha = 1, rho1 = 1, rho2 = 1, rho3 = 1, rho4 = 1)
}
iter = 300
chains = 1
fit1 <- stan(file = "houlsby.stan", 
             data=list(x1 = x1, x2 = x2, a1 = a1, a2 = a2, b1 = b1, b2 = b2, c1 = c1, c2 = c2, N1 = length(y)), 
             iter = iter, chains = chains, verbose = TRUE, control = list(adapt_delta=0.8), init = initf1)

# Correlation on test data
sims1 <- extract(fit1)
uhat <- colMeans(sims1$f)
alpha <- median(sims1$alpha)
rho1 <- median(sims1$rho1)
rho2 <- median(sims1$rho2)
rho3 <- median(sims1$rho3)
rho4 <- median(sims1$rho4)
N1 <- dim(X1)[1]
N2 <- dim(X1test)[1]
N <- N1 + N2
Kall <- houlsbyK(rbind(X1, X1test), rbind(X2, X2test), params = list(alpha = alpha, rho1 = rho1, rho2 = rho2, rho3 = rho3, rho4 = rho4))
Kstar <- Kall[1:N1, (N1 + 1):N]
K <- Kall[1:N1, 1:N1]
fhat <- t(Kstar) %*% solve(K) %*% uhat
cat("GP corr: ", cor(fhat, etatest), "\n")
corGP[j] <- cor(fhat, etatest)
################################################################################
}

################################################################################
write.csv(data.frame(oracle = corOracle, additive = corAdd, gp = corGP), "results/ideal10.csv")
################################################################################
