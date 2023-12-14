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
library(discrim)
library(naivebayes)
library(kknn)
library(themis)


####################
##WORK IN PARALLEL##
####################

all_cores <- parallel::detectCores(logical = FALSE)
#num_cores <- makePSOCKcluster(NUMBER OF CORES)
registerDoParallel(cores = all_cores)

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
 # step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value6
  #step_dummy(all_nominal_predictors()) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_smote(all_outcomes(), neighbors=7)


prepped_recipe <- prep(my_recipe, verbose = T)
bake_1 <- bake(prepped_recipe, new_data = NULL)


# #######################
# ##LOGISTIC REGRESSION##
# #######################
# 
# my_data$ACTION <- as.factor(my_data$ACTION)
# 
# logistic_mod <- logistic_reg() %>%
#   set_engine("glm")
# 
# logistic_workflow <- workflow() %>%
# add_recipe(my_recipe) %>%
# add_model(logistic_mod) %>%
# fit(data = my_data)
# 
# extract_fit_engine(logistic_workflow) %>% #Extracts model details from workflow
#   summary()
# 
# logistic_predictions <- predict(logistic_workflow,
#                               new_data=test_data,
#                               type= "prob")
# logistic_predictions <- cbind(test_data$id,logistic_predictions$.pred_1)
# 
# colnames(logistic_predictions) <- c("id","ACTION")
# summary(logistic_predictions)
# 
# logistic_predictions <- as.data.frame(logistic_predictions)
# 
# vroom_write(logistic_predictions,"logistic_predictions_smote.csv",',')
# 
# #######################
# ##PENALIZED LOGISTIC##
# ######################
# 
# plog_mod <- logistic_reg(mixture = tune(),
#                          penalty = tune()) %>%
#   set_engine("glmnet")
# 
# plog_workflow <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(plog_mod)
# 
# tuning_grid <- grid_regular(penalty(),
#                            mixture(),
#                            levels = 5) ## L^2 total tuning possibilities13
# 
# ## Split data for CV
# folds <- vfold_cv(my_data, v = 3, repeats=1)
# 
# 
# CV_results <- plog_workflow %>%
# tune_grid(resamples=folds,
#           grid=tuning_grid,
#           metrics=metric_set(roc_auc, f_meas, sens, recall, spec,
#                              precision, accuracy))
# bestTune <- CV_results %>%
# select_best("roc_auc")
# 
# final_plog_workflow <-
# plog_workflow %>%
# finalize_workflow(bestTune) %>%
# fit(data=my_data)
# 
# ## Predict
# plog_predictions <- final_plog_workflow %>%
# predict(new_data = test_data, type="prob")
# 
# plog_predictions <- cbind(test_data$id,plog_predictions$.pred_1)
# 
# colnames(plog_predictions) <- c("id","ACTION")
# 
# 
# plog_predictions <- as.data.frame(plog_predictions)
# 
# vroom_write(plog_predictions,"plog_predictions_smote.csv",',')

#################################
##RANDOM FOREST CLASSIFICATIONS##
#################################

RF_model <- rand_forest(mode = "classification",
                        mtry = tune(),
                        trees = 1000,
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


RF_predictions <- final_rf_wf %>% 
  predict(new_data = test_data, type="prob")


RF_predictions <- cbind(test_data$id,RF_predictions$.pred_1)

colnames(RF_predictions) <- c("id","ACTION")


RF_predictions <- as.data.frame(RF_predictions)

vroom_write(RF_predictions,"RF_predictions_pca_smote1.csv",',')

###############
##NAIVE BAYES##
###############

NB_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
set_mode("classification") %>%
set_engine("naivebayes") 

NB_workflow <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(NB_model)

tuning_grid_NB <- grid_regular(Laplace(),
                               smoothness(),
                               levels = 5)
folds_NB <- vfold_cv(my_data, v = 10, repeats=1)

CV_results_NB <- NB_workflow %>%
  tune_grid(resamples=folds_NB,
            grid=tuning_grid_NB,
            metrics=metric_set(roc_auc, f_meas, sens, recall, spec,
                               precision, accuracy))
bestTune_NB <- CV_results_NB %>%
  select_best("roc_auc")

final_NB_wf <- NB_workflow %>% 
  finalize_workflow(bestTune_NB) %>% 
  fit(data = my_data)

NB_predictions <- final_NB_wf %>% 
predict(NB_workflow, new_data=test_data, type="prob")

NB_predictions <- cbind(test_data$id,NB_predictions$.pred_1)

colnames(NB_predictions) <- c("id","ACTION")

NB_predictions <- as.data.frame(NB_predictions)

vroom_write(NB_predictions,"NB_predictions_pca_smote.csv",',')

#######
##KNN##
#######

knn_model <- nearest_neighbor(neighbors= tune()) %>% 
  set_mode("classification") %>%
set_engine("kknn")

knn_wf <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(knn_model)

tuning_grid_knn <- grid_regular(neighbors(),
                               levels = 5)
folds_knn <- vfold_cv(my_data, v = 10, repeats=1)

CV_results_knn <- knn_wf %>%
  tune_grid(resamples=folds_knn,
            grid=tuning_grid_knn,
            metrics=metric_set(roc_auc, f_meas, sens, recall, spec,
                               precision, accuracy))
bestTune_knn <- CV_results_knn %>%
  select_best("roc_auc")

final_knn_wf <- knn_wf %>% 
  finalize_workflow(bestTune_knn) %>% 
  fit(data = my_data)

knn_predictions <- final_knn_wf %>% 
  predict(knn_wf, new_data=test_data, type="prob")

knn_predictions <- cbind(test_data$id,knn_predictions$.pred_1)

colnames(knn_predictions) <- c("id","ACTION")

knn_predictions <- as.data.frame(knn_predictions)

vroom_write(knn_predictions,"knn_predictions_pca_smote.csv",',')

#######
##SVM##
#######

svm_model <-svm_rbf(rbf_sigma=tune(), cost=tune()) %>% 
  set_mode("classification") %>%
  set_engine("kernlab")

svm_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svm_model)

tuning_grid_svm <- grid_regular(rbf_sigma(),
                                cost(),
                                levels = 5)
folds_svm <- vfold_cv(my_data, v = 10, repeats=1)

CV_results_svm <- svm_wf %>%
  tune_grid(resamples=folds_svm,
            grid=tuning_grid_svm,
            metrics=metric_set(roc_auc, f_meas, sens, recall, spec,
                               precision, accuracy))

bestTune_svm <- CV_results_svm %>%
  select_best("roc_auc")

final_svm_wf <- svm_wf %>% 
  finalize_workflow(bestTune_svm) %>% 
  fit(data = my_data)

svm_predictions <- final_svm_wf %>% 
  predict(svm_wf, new_data=test_data, type="prob")

svm_predictions <- cbind(test_data$id,svm_predictions$.pred_1)

colnames(svm_predictions) <- c("id","ACTION")

svm_predictions <- as.data.frame(svm_predictions)

vroom_write(svm_predictions,"svm_predictions_pca_smote.csv",',')









