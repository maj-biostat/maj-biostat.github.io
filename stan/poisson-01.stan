data {
  // Each unit has an outcome corresponding to one of K categories
  int K;
  // Number of units
  int N;
  // Outcome variable (one of the categories)
  // y1= walk/bike, y2=public transport, y3=car
  array[N] int y1;
  array[N] int y2;
  array[N] int y3;
  // design matrix
  vector[N] x;
}
parameters {
  vector[K] a;
  vector[K] b;
}
model {

  to_vector(a) ~ normal(0, 5);
  to_vector(b) ~ normal(0, 5);
  
  target += poisson_log_lpmf(y1 | a[1] + x * b[1]);
  target += poisson_log_lpmf(y2 | a[2] + x * b[2]);
  target += poisson_log_lpmf(y3 | a[3] + x * b[3]);
  
}
generated quantities{
  
  matrix[K, 2] p;
  
  p[1, 1] = exp(a[1]) * inv( exp(a[1]) + exp(a[2]) + exp(a[3]));
  p[2, 1] = exp(a[2]) * inv( exp(a[1]) + exp(a[2]) + exp(a[3]));
  p[3, 1] = exp(a[3]) * inv( exp(a[1]) + exp(a[2]) + exp(a[3]));
  
  p[1, 2] = exp(a[1] + b[1]) * inv( exp(a[1] + b[1]) + exp(a[2] + b[2]) + exp(a[3] + b[3]));
  p[2, 2] = exp(a[2] + b[2]) * inv( exp(a[1] + b[1]) + exp(a[2] + b[2]) + exp(a[3] + b[3]));
  p[3, 2] = exp(a[3] + b[3]) * inv( exp(a[1] + b[1]) + exp(a[2] + b[2]) + exp(a[3] + b[3]));

}
