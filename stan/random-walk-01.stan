data {    
  int N; 
  // the way the model is set up it does not matter if some of the n's are
  // zero because the likelihood uses y_sub, which is obtained by reference
  // to the missing indicator y_mis, which explicitly says that there were
  // no observations at the given value of x.
  array[N] int y;    
  array[N] int n;    
  vector[N] x;    
  array[N] int y_mis; 
  
  int prior_only;    
  
  // priors
  real r_nu;
  
}    
transformed data {
  // x_diff gives us the variable spacing in x and allows us to scale
  // the variance appropriately
  vector[N-1] x_diff;
  // the number of observations we truly had once missingness is accounted for
  int N_sub = N - sum(y_mis);
  // our truly observed responses (successes) and trials
  array[N_sub] int y_sub;
  array[N_sub] int n_sub;
  // 
  for(i in 1:(N-1)){x_diff[i] = x[i+1] - x[i];}
  // go through the data that was passed in and build the data on which 
  // we will fit the model
  int j = 1;
  for(i in 1:N){
    if(y_mis[i] == 0){
      y_sub[j] = y[i];
      n_sub[j] = n[i];
      j += 1;
    }
  }  
}
parameters{  
  // the first response
  real b0;    
  // offsets
  vector[N-1] delta;    
  // how variable the response is
  real<lower=0> nu;   
}    
transformed parameters{    
  // the complete modelled mean response
  vector[N] e; 
  // this is the variance scaled for the distance between each x
  // note this is truly a variance and not an sd
  vector[N-1] tau;    
  // 
  vector[N_sub] eta_sub;    
  // adjust the variance for the distance b/w doses    
  // note that nu is squared to turn it into variance
  for(i in 2:N){tau[i-1] = x_diff[i-1]*pow(nu, 2);}    
  // resp is random walk with missingness filled in due to the 
  // dependency in the prior
  e[1] = b0;    
  // each subsequent observation has a mean equal to the previous one
  // plus some normal deviation with mean zero and variance calibrated for
  // the distance between subsequent observations.
  for(i in 2:N){e[i] = e[i-1] + delta[i-1] * sqrt(tau[i-1]);}    
  // eta_sub is what gets passed to the likelihood
  { 
    int k = 1;
    for(i in 1:N){
      if(y_mis[i] == 0){
        eta_sub[k] = e[i];
        k += 1;
      }
    }
  }
}    
model{    
  // prior on initial response
  target += logistic_lpdf(b0 | 0, 1);
  // prior on sd
  target += exponential_lpdf(nu | r_nu);
  // standard normal prior on the offsets
  target += normal_lpdf(delta | 0, 1);    
  if(!prior_only){target += binomial_logit_lpmf(y_sub | n_sub, eta_sub);}    
}    
generated quantities{    
  // predicted values at each value of x
  vector[N] p;    
  vector[N-1] e_diff;    
  vector[N-1] e_grad;    
  // compute diffs
  for(i in 1:(N-1)){e_diff[i] = e[i+1] - e[i];}
  e_grad = e_diff ./ x_diff;
  p = inv_logit(e);
}    

