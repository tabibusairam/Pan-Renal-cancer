library(survival)
library(glmnet)


require(gdata)
dat <- read.xls("Desktop/combined.xlsx")


#index <- sample(1:442, replace=F)


# index <- read.xls("Desktop/train_test_index.xlsx")
# index = index[,1]
#train_index = index[1:442]
#test_index = index[1:442]

vital <- dat[,c(2)]

train = dat
#[train_index,]
test = dat
#[test_index,]
status <- ifelse(vital >=2 ,1,0)

train_status = status
#[train_index]
test_status = status
#[test_index]

covariates <- c("total_area", "total_convex_area",  "total_perimeter", "total_filled_area", "total_major_axis", "total_minor_axis","total_peri_by_area", "main_region_area", "main_region_convex_area", "main_region_eccentricity", "main_extent", "main_region_solidity" , "main_region_perimeter" , "main_region_angle" , "main_region_peri_by_area" , "main_region_major_axis", "main_region_minor_axis" , "stage" , "gender")


# unicox_df = data.frame(var=NA, HR=NA, se=NA, z=NA, pv=NA)

# for(i in 1:length(covariates)){
#   var = covariates[i]
#   cox = coxph(Surv(time, status) ~ dat[,var], data = dat)
#   unicox_df[i,'var'] = var
#   unicox_df[i,2:5] = summary(cox)$coef[c(2:5)]
# }

# write.csv(unicox_df, file = "Desktop/p-values-univariate.csv")



#1,2,3,4,5,7,8

univ_formulas <- sapply(covariates,
                        function(x) as.formula(paste('Surv(time, status)~', x)))


univ_models <- lapply( univ_formulas, function(x){coxph(x, data = train)})


univ_results <- lapply(univ_models,
                       function(x){ 
                          x <- summary(x)
                          p.value<-signif(x$wald["pvalue"], digits=8)
                          wald.test<-signif(x$wald["test"], digits=8)
                          beta<-signif(x$coef[1], digits=8);#coeficient beta
                          HR <-signif(x$coef[2], digits=8);#exp(beta)
                          HR.confint.lower <- signif(x$conf.int[,"lower .95"], 8)
                          HR.confint.upper <- signif(x$conf.int[,"upper .95"],8)
                          HR <- paste0(HR, " (", 
                                       HR.confint.lower, "-", HR.confint.upper, ")")
                          res<-c(beta, HR, wald.test, p.value)
                          names(res)<-c("beta", "HR (95% CI for HR)", "wald.test", 
                                        "p.value")
                          return(res)
                          #return(exp(cbind(coef(x),confint(x))))
                         })

res <- t(as.data.frame(univ_results, check.names = FALSE))
as.data.frame(res)


write.csv(res, file = "Desktop/train_results.csv")




sig_features_train = train[,c(6,7,8,9,10,11,12,13,14,21)]
sig_features_test = test[,c(6,7,8,9,10,11,12,13,14,21)]

#   sig_features_train = train[,c(7,7)]
# sig_features_test = test[,c(7,7)]


  surv <- Surv(train[, "time"], train_status)

  cox.temp <- coxph(surv ~ ., data = dat[,c(7,7)])



  model2 <- glmnet(x = as.matrix(sig_features_train), y = as.matrix(surv),
                     family = "cox", alpha = 01, standardize=T)
  thresh = cv.glmnet(x = as.matrix(sig_features_train), y = as.matrix(surv),
              family = "cox", alpha = 1,nfolds = , standardize=T)$lambda.min

  print(thresh)

#rm(list = setdiff(ls(), lsf.str()))


#0.0172

surv_test <- Surv(test[, "time"], test_status)
risk <- predict(model2, newx = as.matrix(sig_features_test), thresh)
summary(coxph(surv_test ~ risk))

risk_high <- risk > median(risk, na.rm = T)
table(risk_high)

sf <- survfit(surv_test ~ risk_high)

plot(sf, col = c("green", "red"), mark = 3, xlab = "Time (Days)", ylab = "Survival Probability")
legend(3000, 0.9, legend = c("Low Risk", "High Risk"), col = c("green", "red"), lty = 1, bty = "n")


summary(coxph(surv_test ~ risk))
summary(coxph(surv_test ~ risk_high))
survdiff(surv_test ~ risk_high)
survdiff(surv_test ~ risk_high, rho = 2)




summary(coxph(surv_test ~ risk_high + test[,"gender"] + test[,"stage"]))
summary(coxph(surv_test ~ risk + test[,"gender"] + test[,"stage"]))




#####################################################################################################################################
actual_risk = matrix(, nrow = 471, ncol = 1)
actual_bin = matrix(, nrow = 471, ncol = 1)

for (x in c(1:471)){

	sig_features_train1 <- train[-x,(6:22)]


	train_status1 <- train_status[-x] 


	print(x)


	surv <- Surv(train[-x, "time"], train_status1)

	cox.temp <- coxph(surv ~ ., data = sig_features_train1)


	model2 <- glmnet(x = as.matrix(sig_features_train1), y = as.matrix(surv),
                   family = "cox", alpha = 01, standardize=T)
	thresh = cv.glmnet(x = as.matrix(sig_features_train1), y = as.matrix(surv),
            family = "cox", alpha = 1,nfolds = 10, standardize=T)$lambda.min

#0.0544
	print(thresh)
	surv_test <- Surv(test[, "time"], test_status)
	risk <- predict(model2, newx = as.matrix(sig_features_test), thresh)

	actual_risk[x] <- risk[x]

	risk_high <- risk > median(risk, na.rm = T)
	table(risk_high)

	actual_bin[x] <- risk_high[x]
}












