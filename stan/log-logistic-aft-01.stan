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
  vector<lower=0>[N] scale;   // Scale parameter for each observation (a in the formula)
  vector<lower=0>[N] mu;      // Location parameter (log-scale)
  
  mu = X * gamma;
  for (i in 1:N) {
    scale[i] = exp(mu[i]);
  }
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
                2 * ( 1 + exp(  shape * (log(y[i]) - mu[i]) ) );
    } else {
      // For censored observations, use the survival function
      target += log1p( exp( shape * (log(y[i])   - mu[i])  )  ) ;
    }
  }
}

generated quantities {
  vector[N_pred] surv0;
  vector[N_pred] surv1;
  
  
  for(i in 1:N_pred){
    surv0[i] =  1 / (1 + pow(t_surv[i]/exp(gamma[1]),  shape));
    surv1[i] =  1 / (1 + pow(t_surv[i]/exp(gamma[1] + gamma[2]),  shape));
  }
  
  
}


