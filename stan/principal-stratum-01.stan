data {
    int<lower=0> N; // number of observations
    int<lower=0> PS; // number of predictors for principal stratum model
    int<lower=0> PG; // number of predictors for outcome model
    int<lower=0, upper=1> Z[N]; // treatment arm
    int<lower=0, upper=1> D[N]; // post randomization confounding variable
    int<lower=0, upper=1> Y[N]; // binary outcome
    matrix[N, PS] XS; // model matrix for principal stratum model
    matrix[N, PG] XG; // model matrix for outcome model
}
transformed data {
    int S[8];
   S[1] = 1;
   S[2] = 1;
   S[3] = 2;
   S[4] = 2;
   S[5] = 3;
   S[6] = 3;
   S[7] = 4;
   S[8] = 4;
}
 
parameters {
    matrix[3, PS] beta_S; // coefficients for principal stratum model
    matrix[8, PG] beta_G; // coefficients for outcome model
}
transformed parameters {
}
model {
    // random effect
    // prior
    if (PS >= 2)
        to_vector(beta_S[:, 2:PS]) ~ normal(0, 1);
    if (PG >= 2)
        to_vector(beta_G[:, 2:PG]) ~ normal(0, 1);
    // model
    for (n in 1:N) {
        int length;
        real log_prob[4];
        log_prob[1] = 0;
        for (s in 2:4) {
            log_prob[s] = XS[n] * beta_S[s-1]';
        }
        if (Z[n] == 0 && D[n] == 0)
            length = 2;
        else if (Z[n] == 1 && D[n] == 0)
            length = 2;
        else if (Z[n] == 1 && D[n] == 1)
            length = 2;
        else if (Z[n] == 0 && D[n] == 1)
            length = 2;
        {
            real log_l[length];
            if (Z[n] == 0 && D[n] == 0) {
                // Z:0 D:0 S:0/1
                log_l[1] = log_prob[1] + bernoulli_lpmf(Y[n] | inv_logit(XG[n] * beta_G[1]'));
                log_l[2] = log_prob[2] + bernoulli_lpmf(Y[n] | inv_logit(XG[n] * beta_G[3]'));
            }
            else if (Z[n] == 1 && D[n] == 0) {
                // Z:1 D:0 S:0/2
                log_l[1] = log_prob[1] + bernoulli_lpmf(Y[n] | inv_logit(XG[n] * beta_G[2]'));
                log_l[2] = log_prob[3] + bernoulli_lpmf(Y[n] | inv_logit(XG[n] * beta_G[6]'));
            }
            else if (Z[n] == 1 && D[n] == 1) {
                // Z:1 D:1 S:1/3
                log_l[1] = log_prob[2] + bernoulli_lpmf(Y[n] | inv_logit(XG[n] * beta_G[4]'));
                log_l[2] = log_prob[4] + bernoulli_lpmf(Y[n] | inv_logit(XG[n] * beta_G[8]'));
            }
            else if (Z[n] == 0 && D[n] == 1) {
                // Z:0 D:1 S:2/3
                log_l[1] = log_prob[3] + bernoulli_lpmf(Y[n] | inv_logit(XG[n] * beta_G[5]'));
                log_l[2] = log_prob[4] + bernoulli_lpmf(Y[n] | inv_logit(XG[n] * beta_G[7]'));
            }
            target += log_sum_exp(log_l) - log_sum_exp(log_prob);
        }
    }
}
generated quantities {
    vector[4] strata_prob; // the probability of being in each stratum
    vector[8] mean_effect; // mean response
    {
        matrix[N, 4] log_prob;
        vector[8] numer;
        matrix[N, 8] expected_mean;
        for (i in 1:N)
            for (j in 1:8)
                expected_mean[i, j] = inv_logit(XG[i] * beta_G[j]');
        log_prob[:, 1] = rep_vector(0, N);
        log_prob[:, 2:4] = XS * beta_S';
        for (n in 1:N) {
            log_prob[n] -= log_sum_exp(log_prob[n]);
        }
        for (s in 1:4) strata_prob[s] = mean(exp(log_prob[:, s]));
        for (g in 1:8) {
            numer[g] = mean(expected_mean[:, g] .* exp(log_prob[:, S[g]]));
            mean_effect[g] = numer[g] / strata_prob[S[g]];
        }
    }
}
