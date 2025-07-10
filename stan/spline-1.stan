data {
  int<lower=0> N;
  vector[N] y;
  // basis (excludes intercept)
  int Q;
  matrix[N, Q] B;
  
  // for prediction over domain of x
  int N_ref;
  matrix[N_ref, Q] B_ref;
  
  
  vector[2] prior_g;
  int prior_only;
}
parameters {
  real b0;
  real<lower=0> s_e;
  vector[Q] g;
}
transformed parameters{
  
}
model {
  
  target += normal_lpdf(b0 | 0, 2);
  target += normal_lpdf(g | prior_g[1], prior_g[2]);
  target += exponential_lpdf(s_e | 1);
  if(!prior_only){
    target += normal_lpdf(y | b0 + B * g, s_e);  
  }
  
}
generated quantities{
  
  vector[N_ref] mu = b0 + B_ref * g;
  
  
}
