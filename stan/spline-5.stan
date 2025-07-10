functions {
  real spike_slab_lpdf(real delta, real pi, real sigma1, real sigma2) {
    real log_p1 = log1m(pi) + normal_lpdf(delta | 0, sigma1);
    real log_p2 = log(pi) + normal_lpdf(delta | 0, sigma2);
    return log_sum_exp(log_p1, log_p2);
  }
}

data {
  int<lower=0> N;
  vector[N] y;
  // basis (excludes intercept)
  int Q;
  matrix[N, Q] B;
  
  // for prediction over domain of x
  int N_ref;
  matrix[N_ref, Q] B_ref;
  
  int prior_only;
}
parameters {
  real b0;
  real<lower=0> s_e;
  // offsets
  real z0;
  real z1;
  vector[Q-2] delta2;   
  real<lower=0, upper=1> pi; // mixture weight
  real<lower=0> s_spike;
  real<lower=0> s_slab;
  
}
transformed parameters{
  
  vector[Q] g;
  vector[Q-1] delta1;
  
  g[1] = z0;
  delta1[1] = z1;
  
  for(i in 2:(Q-1)){
    // delta2 has already been scaled via the horseshoe above
    delta1[i] = delta1[i-1] + delta2[i-1];
  }
  
  for(i in 2:Q){
    g[i] = g[i-1] + delta1[i-1];
  }
  
}
model {
  
  target += normal_lpdf(b0 | 0, 10);
  
  // standard normal prior on the offsets
  target += normal_lpdf(z0 | 0, 1); 
  target += normal_lpdf(z1 | 0, 1); 
  
  // mixing prob
  target += beta_lpdf(pi | 1, 1);
  target += exponential_lpdf(s_spike | 1); 
  target += exponential_lpdf(s_slab | 1); 

  for (i in 1:(Q - 2)) {
    target += spike_slab_lpdf(delta2[i] | pi, s_spike, s_slab);
  }
  
  target += exponential_lpdf(s_e | 1);
  if(!prior_only){
    target += normal_lpdf(y | b0 + B * g, s_e);  
  }
  
}
generated quantities{
  
  vector[N_ref] mu = b0 + B_ref * g;
  
  
}
