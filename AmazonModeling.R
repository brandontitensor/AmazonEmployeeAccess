#############
##LIBRARIES##
#############

library(tidymodels) #For the recipes
library(tidyverse) #Given for EDA
library(poissonreg) #For Poisson Regression
library(vroom) #For reading in data
library(DataExplorer)
library(glmnet)
library(mltools)
library(randomForest)
library(doParallel)
library(xgboost)
tidymodels_prefer()
conflicted::conflicts_prefer(yardstick::rmse)
library(rpart)
library(stacks) #For stacking
library(ggmosaic)


####################
##WORK IN PARALLEL##
####################

all_cores <- parallel::detectCores(logical = FALSE)
registerDoParallel(cores = all_cores)

########
##DATA##
########

my_data <- vroom("train.csv")
test_data <- vroom("test.csv")

#######
##EDA##
#######

DataExplorer::plot_histogram(my_data)
GGally::ggpairs(my_data)

plot_1 <- ggplot(data=my_data) +
  geom_mosaic(aes(x=product(RESOURCE), fill = ACTION))

##########
##RECIPE##
##########

my_recipe <- recipe(ACTION~., data=my_data) %>%
step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors5
  step_other(all_nominal_predictors(), threshold = .01) %>% # combines categorical values that occur <5% into an "other" value6
  step_dummy(all_nominal_predictors()) 

prepped_recipe <- prep(my_recipe, verbose = T)
bake_1 <- bake(prepped_recipe, new_data = NULL)


#######################
##LOGISTIC REGRESSION##
#######################

my_data$ACTION <- as.factor(my_data$ACTION)

logistic_mod <- logistic_reg() %>% 
  set_engine("glm")

logistic_workflow <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(logistic_mod) %>%
fit(data = my_data)

extract_fit_engine(logistic_workflow) %>% #Extracts model details from workflow
  summary()

logistic_predictions <- predict(logistic_workflow,
                              new_data=test_data,
                              type= "prob")
logistic_predictions <- cbind(test_data$id,logistic_predictions$.pred_1)

colnames(logistic_predictions) <- c("id","ACTION")
summary(logistic_predictions)

logistic_predictions <- as.data.frame(logistic_predictions)

vroom_write(logistic_predictions,"logistic_predictions.csv",',')



