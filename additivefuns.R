################################################################################
rk <- function(x, z){
  ((z - 0.5)^2 - 1/12)*((x - 0.5)^2 - 1/12)/4 - ((abs(x - z) - 0.5)^4 - 0.5*(abs(x - z) - 0.5)^2 + 7/240)/24
}

spl.X <- function(x, z){
  # number of columns
  q <- length(z) + 1
  # number of data
  n <- length(x)
  # matrix of 1s
  X <- matrix(1, nrow = n, ncol = q)
  # set second column to x
  X[ , 1] <- x
  X[ , 2:q] <- outer(x, z, FUN = rk)
  X
}

spl.S <- function(xk){
  # set up the penalized regression spline penalty matrix,
  # given knot sequence xk
  q <- length(xk) + 1
  S <- matrix(0, q, q) # initialize matrix to 0
  S[2:q, 2:q] <- outer(xk, xk, FUN = rk) # fill in non-zero part
  S
}

make.knots <- function(xleft, xright){
  knots1 <- unique(quantile(c(xleft, xright), probs = seq(.01, .99, by = .05)))
  return(knots1)
}
################################################################################

################################################################################
make.U <- function(Xleft, Xright){
  library(Matrix)
  N <- dim(Xleft)[1]
  K <- dim(Xleft)[2]
  knotlist <- list()
  Ulist <- list() 
  for(k in 1:K){
    z <- knotlist[[k]] <- make.knots(Xleft[, k], Xright[, k])
    Ulist[[k]] <- spl.X(Xleft[, k], z) - spl.X(Xright[, k], z)
  }
  U <- do.call(cbind, Ulist)
  return(U)
}

make.basis <- function(Xleft, Xright, lambda){
  library(Matrix)
  N <- dim(Xleft)[1]
  K <- dim(Xleft)[2]
  knotlist <- list()
  Ulist <- list() 
  Slist <- list() 
  for(k in 1:K){
    z <- knotlist[[k]] <- make.knots(Xleft[, k], Xright[, k])
    Ulist[[k]] <- spl.X(Xleft[, k], z) - spl.X(Xright[, k], z)
    Slist[[k]] <- lambda[k]*spl.S(z)
  }
  U <- do.call(cbind, Ulist)
  S <- as.matrix(bdiag(Slist))
  return(list(U = U, S = S))
}

NRinner <- function(U, S, H, y){
  epsilonold <- 0.01
  epsilonnew <- 0.001
  its <- 1
  betaold <- rep(0, dim(U)[2])
  for(i in 1:20){
    epsilonold <- epsilonnew
    etahat <- U %*% betaold
    phat <- exp(etahat)/(1 + exp(etahat))
    W <- diag(as.numeric(phat*(1-phat)))
    betanew <- betaold + solve(t(U) %*% W %*% U + H + S) %*% t(U) %*% (y - phat)  
    epsilonnew <- as.vector(dist(rbind(t(betanew), t(betaold))))
    betaold <- betanew
    its <- its + 1
    if(its > 100){
      cat("Warning! Max iterations reached\n")
      break
    } 
  }
  return(betanew)
}

NRouter <- function(X1, X2, y, iters = 200){
  n <- length(y)
  K <- dim(X1)[2]
  lambdas <- matrix(NA, nrow = iters, ncol = K)
  for(i in 1:iters){
    lambdas[i, ] <- runif(K, .1, 10)
  }
  GCV <- c()
  for(j in 1:dim(lambdas)[1]){
    B <- make.basis(X1, X2, lambdas[j, ])
    U <- B$U
    S <- B$S
    kappa <- dim(S)[1]
    H <- diag(kappa)*0.00000000001
    betaj <- NRinner(U, S, H, y)
    muhat <- exp(U %*% betaj)/(1 + exp(U %*% betaj))
    W <- diag(as.vector(muhat*(1-muhat)))
    A <- U %*% (solve(t(U) %*% W %*% U + H + S)) %*% t(U)
    GCV[j] <- n*sum((y - muhat)^2)/(n - sum(diag(A)))^2
#    cat("GCV: ", GCV[j], "for lambda = ", lambdas[j, ],  "\n")      
  }
  lambdamax <- lambdas[which(GCV == min(GCV)), ]
  return(lambdamax)
}

NRspline <- function(X1, X2, y){
  maxlambda <- NRouter(X1, X2, y)
  cat("Max lambda: ", maxlambda, "\n")
  B <- make.basis(X1, X2, maxlambda)
  U <- B$U
  S <- B$S
  kappa <- dim(S)[1]
  H <- diag(kappa)*0.00000000001
  betanew <- NRinner(U, S, H, y)
  return(list(betas = betanew, U = U, S = S))
}

Fcalc <- function(Xleft, Xright, xlnew, xrnew, S, beta){
  kappa <- dim(S)[1]
  H <- diag(kappa)*0.00000000001
  U <- make.U(rbind(Xleft, xlnew), rbind(Xright, xrnew))  
  eta <- U %*% beta
  p <- exp(eta)/(1 + exp(eta))
  W <- diag(as.numeric(p*(1-p)))
  info <- t(U) %*% W %*% U + H + S
  return(info)
}
################################################################################

