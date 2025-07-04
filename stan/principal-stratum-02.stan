data {
    int<lower=0> N; // number of observations
    int<lower=0> PS; // number of predictors for principal stratum model
    int<lower=0> PG; // number of predictors for outcome model
    array[N] int<lower=0, upper=1> Z; // treatment arm
    array[N] int<lower=0, upper=1> D; // post randomization confounding variable
    array[N] int<lower=0, upper=1> Y; // binary outcome
    matrix[N, PS] XS; // model matrix for principal stratum model
    matrix[N, PG] XG; // model matrix for outcome model
}
 
transformed data {
   array[4] int S;
   S[1] = 1;
   S[2] = 2;
   S[3] = 2;
   S[4] = 3;
}
 
parameters {
    matrix[2, PS] beta_S; // coefficients for principal stratum model
    matrix[4, PG] beta_G; // coefficients for outcome model
}
 
transformed parameters {
}
 
model {
    // random effect
    // prior
    
    // use informative prior for intercepts
    beta_S[:, 1] ~ normal(0, 2);
    beta_G[:, 1] ~ normal(0, 2);
    
    if (PS >= 2)
        to_vector(beta_S[:, 2:PS]) ~ normal(0, 1);
    if (PG >= 2)
        to_vector(beta_G[:, 2:PG]) ~ normal(0, 1);
    // model
    for (n in 1:N) {
        int length;
        array[3] real log_prob;
        log_prob[1] = 0;
        for (s in 2:3) {
            log_prob[s] = XS[n] * beta_S[s-1]';
        }
        if (Z[n] == 0 && D[n] == 0)
            length = 2;
        else if (Z[n] == 1 && D[n] == 0)
            length = 1;
        else if (Z[n] == 1 && D[n] == 1)
            length = 2;
        else if (Z[n] == 0 && D[n] == 1)
            length = 1;
        {
            array[length] real log_l;
            if (Z[n] == 0 && D[n] == 0) {
                // Z:0 D:0 S:0/1 never takers or compliers
                log_l[1] = log_prob[1] + bernoulli_lpmf(Y[n] | inv_logit(XG[n] * beta_G[1]'));
                log_l[2] = log_prob[2] + bernoulli_lpmf(Y[n] | inv_logit(XG[n] * beta_G[2]'));
            }
            else if (Z[n] == 1 && D[n] == 0) {
                // Z:1 D:0 S:0 never takers (defiers don't exist)
                log_l[1] = log_prob[1] + bernoulli_lpmf(Y[n] | inv_logit(XG[n] * beta_G[1]'));
            }
            else if (Z[n] == 1 && D[n] == 1) {
                // Z:1 D:1 S:1/2 compliers or always takers
                log_l[1] = log_prob[2] + bernoulli_lpmf(Y[n] | inv_logit(XG[n] * beta_G[3]'));
                log_l[2] = log_prob[3] + bernoulli_lpmf(Y[n] | inv_logit(XG[n] * beta_G[4]'));
            }
            else if (Z[n] == 0 && D[n] == 1) {
                // Z:0 D:1 S:2 always takers
                log_l[1] = log_prob[3] + bernoulli_lpmf(Y[n] | inv_logit(XG[n] * beta_G[4]'));
            }
            target += log_sum_exp(log_l) - log_sum_exp(log_prob);
        }
    }
}
 
generated quantities {
    vector[3] strata_prob; // the probability of being in each stratum
    vector[4] mean_effect; // mean response
    {
        matrix[N, 3] log_prob;
        vector[4] numer;
        matrix[N, 4] expected_mean;
        for (i in 1:N)
            for (j in 1:4)
                expected_mean[i, j] = inv_logit(XG[i] * beta_G[j]');
        log_prob[:, 1] = rep_vector(0, N);
        log_prob[:, 2:3] = XS * beta_S';
        for (n in 1:N) {
            log_prob[n] -= log_sum_exp(log_prob[n]);
        }
        for (s in 1:3) strata_prob[s] = mean(exp(log_prob[:, s]));
        for (g in 1:4) {
            numer[g] = mean(expected_mean[:, g] .* exp(log_prob[:, S[g]]));
            mean_effect[g] = numer[g] / strata_prob[S[g]];
        }
    }
}
