

data {
  // num expt
  int<lower=0> K;
  array[K] int X;
  int<lower=0> P;
  array[P] int n;
}
transformed data{
}
parameters {
  real<lower=0, upper=1> theta;
}
transformed parameters{
  // unnormalised density
  vector[P] lq;
  for(i in 1:P){
    // record the unnormalised density for every possible value 
    // of n
    
    // log pmf for the array of values in X conditional on a given n[i]
    // and the parameter theta PLUS the prior on n 
    lq[i] = binomial_lpmf(X | n[i], theta) + log(1./P);
  }
}
model {
  target += uniform_lpdf(theta | 0, 1);
  
  // marginalise out the troublesome n
  target += log_sum_exp(lq);
  
}
generated quantities{
  // probability of n given X, i.e. the distribution of n | x
  vector[P] p_n_X;
  real mu_n;
  p_n_X = exp(lq - log_sum_exp(lq));
  
  {
    vector[P] tmp;
    for(i in 1:P){
      tmp[i] =  p_n_X[i] * n[i];
    }
    mu_n = sum(tmp);
  }
  
  
}

