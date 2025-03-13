

data {
  // num expt
  int<lower=0> K;
  array[K] int X;
}
transformed data{
  array[4] int n;
  // these are the permissible values of n, 
  // i.e. 5, 6, 7, 8
  for(i in 1:4){
    n[i] = 4 + i;
  }
}
parameters {
  real<lower=0, upper=1> theta;
}
transformed parameters{
  // unnormalised density
  vector[4] lq;
  for(i in 1:4){
    // record the unnormalised density for every possible value 
    // of n
    
    // log pmf for the array of values in X conditional on a given n[i]
    // and the parameter theta PLUS the prior on the given n, which is 
    // a discrete uniform, i.e. for each n, the prior is 0.25
    lq[i] = binomial_lpmf(X | n[i], theta) + log(0.25);
  }
}
model {
  target += uniform_lpdf(theta | 0, 1);
  
  // marginalise out the troublesome n
  target += log_sum_exp(lq);
  
}
generated quantities{
  // probability of n given X, i.e. the distribution of n | x
  vector[4] p_n_X;
  p_n_X = exp(lq - log_sum_exp(lq));
  
}

