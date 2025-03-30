// Log-logistic AFT model
data {
  int<lower=0> N;             // Number of observations
  int<lower=0> P;             // Number of predictors
  matrix[N, P] X;             // Predictor matrix X[, 1] is intercept
  vector<lower=0>[N] y;       // Observed survival times
  vector<lower=0, upper=1>[N] event;  // Event indicator (1=event, 0=censored)
  
  int N_pred;
  vector[N_pred] t_surv;    // time to predict survival at
  
}

parameters {
  vector[P] gamma;             // Regression coefficients for scale
  real<lower=0> shape;        // Shape parameter (b in the formula)
}

transformed parameters {
  // Location parameter (log-scale)
  vector[N] mu;      
  
  mu = X * gamma;
}

model {
  // Priors - arbitrary at the moment
  target += normal_lpdf(gamma | 0, 2);
  target += gamma_lpdf(shape | 1, 0.1);
  
  // Likelihood
  for (i in 1:N) {
    if (event[i] == 1) {
      // For observed events, use the log-logistic density
      target += log(shape) - mu[i] + (shape - 1) * (log(y[i]) - mu[i]) - 
                2 * log1p(pow(y[i] / exp(mu[i]), shape));
    } else {
      // For censored observations, use the log survival function
      target += -log1p(pow(y[i] / exp(mu[i]), shape));
    }
  }
}

generated quantities {
  vector[N_pred] surv0;
  vector[N_pred] surv1;
  
  real med_surv_time0;
  real med_surv_time1;
  
  // obviously raising 1 to anything is 1 so only need the scale part
  med_surv_time0 = exp(gamma[1]);
  med_surv_time1 = exp(gamma[1] + gamma[2]);
  
  for(i in 1:N_pred){
    surv0[i] =  1 / (1 + pow(t_surv[i]/exp(gamma[1]),  shape));
    surv1[i] =  1 / (1 + pow(t_surv[i]/exp(gamma[1] + gamma[2]),  shape));
  }
  
  
}


