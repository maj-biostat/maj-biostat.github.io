data {
  // Each unit has an outcome corresponding to one of K categories
  int K;
  // Number of units
  int N;
  // Number of terms in linear predictor (all lp have the 
  // same terms here). Includes intercept.
  int D;
  // Outcome variable (one of the categories)
  array[N] int y;
  // design matrix
  matrix[N, D] x;
}
parameters {
  //            k1  k2  k3 etc
  // intercept  -   -   -
  //         x  -   -   -
  matrix[D, K-1] b_raw;
}
transformed parameters {
  //            k1  k2  k3 etc
  // intercept  -   -   -
  //         x  -   -   -
  matrix[D, K] b;
  
  b[, 1:(K-1)] = b_raw;
  b[, K] = rep_vector(0.0, D);
}
model {
  matrix[N, K] x_beta = x * b;

  to_vector(b_raw) ~ normal(0, 5);

  for (n in 1:N) {
    y[n] ~ categorical_logit(x_beta[n]');
  }
}
generated quantities{
  
  matrix[K, 2] p;
  
  vector[K] l0 = to_vector(b[1, ]);
  vector[K] l1 = to_vector(b[1, ] + b[2, ]);
  
  p[, 1] = softmax(l0);
  p[, 2] = softmax(l1);
  
}
