functions {
  real custom_lpdf(vector x, real alpha) {
    
    int n_x = num_elements(x);
    vector[n_x] lpdf;
    for(i in 1:n_x){
      
      lpdf[i] = log1m(alpha) - alpha * log(x[i]);
    }  
    return sum(lpdf);
  }
}
data {
  int N;
  vector[N] y;
}

parameters {
  real<lower=0, upper = 1> a;
}
model {
  target += exponential_lpdf(a | 1);
  target += custom_lpdf(y | a);   
}


