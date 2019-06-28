################################################################################
source("additivefuns.R")
corAdd <- matrix(NA, nrow = 40, ncol = 10)
################################################################################

for(q in 1:10){
################################################################################
## Training data
options(scipen = 99, digits = 2)
Nstart <- 5
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

eta <- 1*(x1 - x2) - 2*(a1 - a2) + 3*(b1 - b2) - 4*(c1 - c2)
p <- exp(eta)/(1 + exp(eta))
y <- rbinom(length(p), 1, p)
################################################################################

for(r in 5:40){
################################################################################
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
################################################################################

################################################################################
# Oracle glm
xdiff <- x1 - x2
adiff <- a1 - a2
bdiff <- b1 - b2
cdiff <- c1 - c2
glm1 <- glm(y ~ xdiff + adiff + bdiff + cdiff, family = binomial(link = "logit"))
predglm1 <- predict(glm1, newdata = data.frame(xdiff = x1test - x2test, adiff = a1test - a2test, bdiff = b1test - b2test, cdiff = c1test - c2test))
#plot(predglm1, etatest)
cat("Oracle corr: ", cor(predglm1, etatest), "\n")

## Homebrew
# Check on test data
splinereg <- NRspline(X1, X2, y)
betas <- splinereg$betas
Utest <- make.U(X1test, X2test)
predtest <- Utest %*% betas
#plot(predtest, etatest)
cor(predtest, etatest)
cat("Sample size: ", length(y), "\n")
cat("Homebrew corr: ", cor(predtest, etatest), "\n")
corAdd[r, q] <- cor(predtest, etatest)
################################################################################

################################################################################
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

dopt <- c()
for(j in 1:1000){
  dopt[j] <- det(Fcalc(X1, X2, X1new[j, ], X2new[j, ], splinereg$S, splinereg$betas))
}
optdesign <- which(dopt == max(dopt))[1]
cat("D criterion: ", dopt[optdesign], "\n")
cat("Z-score of D optimal: ", (dopt[optdesign] - mean(dopt))/sd(dopt), "\n\n")

x1 <- c(x1, x1new[optdesign])
a1 <- c(a1, a1new[optdesign])
b1 <- c(b1, b1new[optdesign])
c1 <- c(c1, c1new[optdesign])
X1 <- rbind(X1, X1new[optdesign, ])

X2 <- rbind(X2, X2new[optdesign, ])
x2 <- c(x2, x2new[optdesign])
a2 <- c(a2, a2new[optdesign])
b2 <- c(b2, b2new[optdesign])
c2 <- c(c2, c2new[optdesign])

etanew <- 1*(x1new - x2new) - 2*(a1new - a2new) + 3*(b1new - b2new) - 4*(c1new - c2new)
pnew <- exp(etanew)/(1 + exp(etanew))
ynew <- rbinom(1, 1, pnew)
y <- c(y, ynew)
}
################################################################################
}

################################################################################
write.csv(corAdd, "results/linearAdd.csv")
################################################################################

