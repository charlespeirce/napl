// Predict from Gaussian Process
// All data parameters must be passed as a list to the Stan call
// Based on original file from https://code.google.com/p/stan/source/browse/src/models/misc/gaussian-process/

data {
  //Observed data
//  real delta;
  int<lower=1> N1;
  real x1[N1];
  real x2[N1];
  real a1[N1];
  real a2[N1];
  real b1[N1];
  real b2[N1];
  real c1[N1];
  real c2[N1];
  int<lower = 0, upper = 1> y[N1];
}

transformed data {
  real delta = 1e-9;
}

parameters {
  real<lower=0> rho1;
  real<lower=0> rho2;
  real<lower=0> rho3;
  real<lower=0> rho4;  
  real<lower=0> alpha;
  vector[N1] eta;
}

transformed parameters {
  vector[N1] f;
  {
    matrix[N1, N1] L_K;
    matrix[N1, N1] K_ik;
    matrix[N1, N1] K_jl;
    matrix[N1, N1] K_il;
    matrix[N1, N1] K_jk;
    matrix[N1, N1] K;

  //
  for(i in 1:N1){
    for(j in 1:N1){
    K_ik[i,j] = alpha*exp(-.5*rho1*pow(x1[i] - x1[j],2) -.5*rho2*pow(a1[i] - a1[j],2) -.5*rho3*pow(b1[i] - b1[j],2) -.5*rho4*pow(c1[i] - c1[j],2)); 
    K_jl[i,j] = alpha*exp(-.5*rho1*pow(x2[i] - x2[j],2) -.5*rho2*pow(a2[i] - a2[j],2) -.5*rho3*pow(b2[i] - b2[j],2) -.5*rho4*pow(c2[i] - c2[j],2)); 
    K_il[i,j] = alpha*exp(-.5*rho1*pow(x1[i] - x2[j],2) -.5*rho2*pow(a1[i] - a2[j],2) -.5*rho3*pow(b1[i] - b2[j],2) -.5*rho4*pow(c1[i] - c2[j],2)); 
    K_jk[i,j] = alpha*exp(-.5*rho1*pow(x2[i] - x1[j],2) -.5*rho2*pow(a2[i] - a1[j],2) -.5*rho3*pow(b2[i] - b1[j],2) -.5*rho4*pow(c2[i] - c1[j],2)); 
    K[i,j] = K_ik[i, j] + K_jl[i, j] - K_il[i, j] - K_jk[i, j];
    }
  }

  for(n in 1:N1){
    K[n, n] = K[n, n] + delta; //  + sigma;// sigma + delta;
  }

  L_K = cholesky_decompose(K);
  f = L_K * eta;
  }

}

model {
  rho1 ~ inv_gamma(5, 5);
  rho2 ~ inv_gamma(5, 5);
  rho3 ~ inv_gamma(5, 5);
  rho4 ~ inv_gamma(5, 5);
  alpha ~ std_normal();
  eta ~ std_normal();
  y ~ bernoulli_logit(f[1:N1]);
}

// http://natelemoine.com/fast-gaussian-process-models-in-stan/
