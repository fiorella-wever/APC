
# function to create the inputs based on the output
simulate_data <- function(ts_id, L_ts, sigma = 0.2){
  # this function creates 2 input time-series, x and b, 
  # and a binary output y.
  # x is a continuous variable and 
  # b is a binary variable.
  # x is modeled as a noisy sine curve
  # b is modeled as Bernoulli variable
  # the output depends on the frequency, and offset of x
  # and on the probability of b.
  #sigma = 0.5
  
  b_prob <- runif(1)
  x_add_freq<- runif(1)
  x_offset <- runif(1)
  
  y <- 
    sign(b_prob-0.5)  * 
    sign(0.5-x_add_freq) # * 
    # sign((b_prob-0.5) * (0.5-x_offset)) 
  
  y <- (y > 0)*1
  
  b <- rbinom(L_ts, size = 1, prob = b_prob)
  t <- 1:L_ts
  
  # min_period <- 2 # the number of time-points between 2 max
  min_period <- 5
  # max_period <- 10 # 
  max_period <- 20
  period <- min_period + x_add_freq * (max_period - min_period)
  
  x <- 
    x_offset + 1 + 
    cos(t/period*2*pi) + 
    rnorm(L_ts, sd = sigma)
  
  tibble(
    ts_id,
    b_prob,
    x_add_freq,
    x_offset,
    y = y %>% factor(., levels = c(0,1)),
    expand_grid(input_var = c("b","x"), t = 1:L_ts),
    var_value = c(b, x)
  ) %>% 
    select(ts_id, t, y, input_var, var_value, everything())
  
}



