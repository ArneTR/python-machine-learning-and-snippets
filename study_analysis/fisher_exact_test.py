## TODO: Must be transported to Python

# Also a fisher's-Exact Test test may be useful here, to check if the data is usable and the effect does not come from sampling
```{r}
library(perm)
perms <- chooseMatrix(6,3)
treatment_data <- c(38.2, 37.1, 37.6)
control_data <- c(36.4, 37.3, 36.0)
observed_test <- cbind(c(treatment_data, control_data))

treatment_avgs <- (1/3)*perms%*%observed_test
control_avgs <- (1/3)*(1-perms)%*%observed_test
test_statistics <- abs(treatment_avgs-control_avgs)

test_statistic_observed <- mean(treatment_data) - mean(control_data)
larger_than_observed <- (test_statistics >= test_statistic_observed)
#numbers in which the statistic exceeds the value in the observed date
sum(larger_than_observed)

fishers_p_value <- sum(larger_than_observed) / nrow(perms)

# this df is just for putting all relevant data in a neat table
df <- data.frame(perms,control_avgs,treatment_avgs,test_statistics)

```

