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
  real<lower=0> s_d;
}
transformed parameters{
  vector[Q] g;
  vector[Q-1] delta1;
  
  g[1] = z0;
  delta1[1] = z1;
  
  for(i in 2:(Q-1)){
    delta1[i] = delta1[i-1] + delta2[i-1] * s_d;
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
  target += normal_lpdf(delta2 | 0, 1);  
  // prior on sd for rw
  target += exponential_lpdf(s_d | 1);  
  
  
  target += exponential_lpdf(s_e | 1);
  if(!prior_only){
    target += normal_lpdf(y | b0 + B * g, s_e);  
  }
  
}
generated quantities{
  
  vector[N_ref] mu = b0 + B_ref * g;
  
  
}
