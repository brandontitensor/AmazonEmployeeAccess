#############
##LIBRARIES##
#############

library(tidymodels) 
library(tidyverse)
library(vroom) 
library(glmnet)
library(randomForest)
library(doParallel)
library(xgboost)
tidymodels_prefer()
conflicted::conflicts_prefer(yardstick::rmse)
library(rpart)
library(stacks)
library(embed)


####################
##WORK IN PARALLEL##
####################

#all_cores <- parallel::detectCores(logical = FALSE)
#num_cores <- makePSOCKcluster(NUMBER OF CORES)
#registerDoParallel(cores = all_cores)

#stopCluster(num_cores)

########
##DATA##
########

my_data <- vroom("train.csv")
test_data <- vroom("test.csv")
my_data$ACTION <- as.factor(my_data$ACTION)

#######
##EDA##
#######

#DataExplorer::plot_histogram(my_data)
#GGally::ggpairs(my_data)

#plot_1 <- ggplot(data=my_data) +
#  geom_mosaic(aes(x=product(RESOURCE), fill = ACTION))

##########
##RECIPE##
##########

my_recipe <- recipe(ACTION~., data=my_data) %>%
step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors5
  step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value6
  #step_dummy(all_nominal_predictors()) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))

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

#######################
##PENALIZED LOGISTIC##
######################

plog_mod <- logistic_reg(mixture = tune(),
                         penalty = tune()) %>% 
  set_engine("glmnet")

plog_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(plog_mod)

tuning_grid <- grid_regular(penalty(),
                           mixture(),
                           levels = 5) ## L^2 total tuning possibilities13

## Split data for CV
folds <- vfold_cv(my_data, v = 3, repeats=1)


CV_results <- plog_workflow %>%
tune_grid(resamples=folds,
          grid=tuning_grid,
          metrics=metric_set(roc_auc, f_meas, sens, recall, spec,
                             precision, accuracy)) 
bestTune <- CV_results %>%
select_best("roc_auc")

final_plog_workflow <-
plog_workflow %>%
finalize_workflow(bestTune) %>%
fit(data=my_data)

## Predict
plog_predictions <- final_plog_workflow %>%
predict(new_data = test_data, type="prob")

plog_predictions <- cbind(test_data$id,plog_predictions$.pred_1)

colnames(plog_predictions) <- c("id","ACTION")


plog_predictions <- as.data.frame(plog_predictions)

vroom_write(plog_predictions,"plog_predictions.csv",',')

#################################
##RANDOM FOREST CLASSIFICATIONS##
#################################

RF_model <- rand_forest(mode = "classification",
                        mtry = tune(),
                        trees = 500,
                        min_n = tune()) %>% #Applies Linear Model
  set_engine("ranger")

RF_workflow <- workflow() %>% #Creates a workflow
  add_recipe(my_recipe) %>% #Adds in my recipe
  add_model(RF_model) 

tuning_grid_rf <- grid_regular(mtry(range = c(1,10)),
                               min_n(),
                               levels = 5)
folds_rf <- vfold_cv(my_data, v = 10, repeats=1)

CV_results_rf <- RF_workflow %>%
  tune_grid(resamples=folds_rf,
            grid=tuning_grid_rf,
            metrics=metric_set(roc_auc, f_meas, sens, recall, spec,
                               precision, accuracy))
bestTune_rf <- CV_results_rf %>%
  select_best("roc_auc")

final_rf_wf <- RF_workflow %>% 
  finalize_workflow(bestTune_rf) %>% 
  fit(data = my_data)


bike_predictions_RF<- final_rf_wf %>% 
  predict(new_data = test_data)

bike_predictions_RF[bike_predictions_RF < 0] <- 0
bike_predictions_RF <- cbind(test$datetime,bike_predictions_RF) #Adds back in the dattime variable for submission
bike_predictions_RF <- bike_predictions_RF %>% mutate(.pred=exp(.pred))
colnames(bike_predictions_RF) <- c("datetime","count") #Changes the labels for submission
bike_predictions_RF$datetime <- as.character(format(bike_predictions_RF$datetime)) 

vroom_write(bike_predictions_RF,"bike_predictions_RF.csv",',')




