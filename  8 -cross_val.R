library(survival)
library(glmnet)


require(gdata)
dat <- read.xls("Desktop/combined.xlsx")


vital <- dat[,c(3)]

train = dat
#[train_index,]
test = dat
#[test_index,]
status <- ifelse(vital >=2 ,1,0)

train_status = status
#[train_index]
test_status = status
#[test_index]
##################################################################################################################
#4,5,6,7,8,9,10,11,12,16,18,19,20
#21,22,23,24,25,26
#4,5,6,7,12,16,18,19,22
alph = 27;
covariates <- c("total_area", "total_convex_area",  "total_perimeter", "total_filled_area", "total_major_axis", "total_minor_axis","total_peri_by_area", "main_region_area", "main_region_convex_area", "main_region_eccentricity", "main_extent", "main_region_solidity" , "main_region_perimeter" , "main_region_angle" , "main_region_peri_by_area" , "main_region_major_axis", "main_region_minor_axis" , "stage" , "gender")

covariates1 <- c("cell_total_area", "cell_total_convex_area",  "cell_total_perimeter", "cell_total_filled_area", "cell_total_major_axis", "cell_total_minor_axis","cell_total_peri_by_area")


# alph,alph
sig_features_train = train[,c(4,5,6,7,12,16,18,19,22)]
sig_features_test = test[,c(4,5,6,7,12,16,18,19,22)]


  surv <- Surv(train[, "time"], train_status)

  cox.temp <- coxph(surv ~ ., data = dat[,c(4,5,6,7,12,16,18,19,22)])


##################################################################################################################


actual_risk = matrix(, nrow = dim(dat)[1], ncol = 1)
actual_bin = matrix(, nrow = dim(dat)[1], ncol = 1)

for (x in c(1:dim(dat)[1])){
	sig_features_train1 <- dat[-x,c(4,5,6,7,12,16,18,19,22)]


	train_status1 <- train_status[-x] 


	print(x)


	surv <- Surv(train[-x, "time"], train_status1)

	cox.temp <- coxph(surv ~ ., data = dat[-x,c(4,5,6,7,12,16,18,19,22)])



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



summary(coxph(surv_test ~ actual_risk))

sf <- survfit(surv_test ~ actual_bin)

##################################################################################################################



tot = "Desktop/final-figures/"
tot = paste(tot,covariates1[alph-20],sep="")
tot_risk = paste(tot,'/actual_risk.csv',sep="")
tot_bin = paste(tot,'/actual_bin.csv',sep="")
tot_cox = paste(tot,'/unicox_df.csv',sep="")

write.csv(actual_bin, file = tot_bin)
write.csv(actual_risk, file = tot_risk)

unicox_df = data.frame(beta=NA, HR=NA, se=NA, z=NA, pv=NA)
unicox_df[1:5] = summary(coxph(surv_test ~ actual_risk))$coef[c(1:5)]

write.csv(unicox_df, file = tot_cox)





plot(sf, col = c("green", "red"), mark = 3, lwd=2, xlab = "Time (Days)", ylab = "Survival Probability")
legend(2700, 1, legend = c("Low Risk", "High Risk"), col = c("green", "red"), lty = 1,lwd =2, bty = "n")

pval = summary(coxph(surv_test ~ actual_risk))$coef[c(5)]
pval = toString(pval)

# prefix = 'P-value = '
# tot = paste(prefix,substr(pval,1,6))

covariates1[alph-20]
mtext(substitute(bold('Lasso Cox ')), side=3)

# temp=list()
# temp <- locator(1)
# temp("x") = 1100.368
# temp("y") = 0.1298919

text(temp,'P-value = 3.68e-6')








##################################################################################################################

# write.csv(actual_bin, file = "Desktop/final-figures/total_filled_area/actual_bin.csv")
# write.csv(actual_risk, file = "Desktop/final-figures/total_filled_area/actual_risk.csv")

# unicox_df = data.frame(beta=NA, HR=NA, se=NA, z=NA, pv=NA)
# unicox_df[1:5] = summary(coxph(surv_test ~ actual_risk))$coef[c(1:5)]

# write.csv(unicox_df, file = "Desktop/final-figures/total_filled_area/unicox_df.csv")




##################################################################################################################




#after getting saved



# library(survival)
# library(glmnet)


# require(gdata)
# dat <- read.xls("Desktop/final_clinical.xlsx")


# vital <- dat[,c(4)]

# train = dat
# #[train_index,]
# test = dat
# #[test_index,]
# status <- ifelse(vital >=2 ,1,0)

# train_status = status
# #[train_index]
# test_status = status




# surv_test <- Surv(test[, "time"], test_status)


# actual_bin <- read.csv("Desktop/final-figures/all-features/actual_bin.csv")

# actual_risk <- read.csv("Desktop/final-figures/all-features/actual_risk.csv")

# actual_bin = actual_bin[,2]
# actual_risk = actual_risk[,2]




# summary(coxph(surv_test ~ actual_risk))

# sf <- survfit(surv_test ~ actual_bin)

# plot(sf, col = c("green", "red"), mark = 3, lwd=3, xlab = "Time (Days)", ylab = "Survival Probability")
# legend(3000, 0.9, legend = c("Low Risk", "High Risk"), col = c("green", "red"), lty = 1,lwd =3, bty = "n")


# unicox_df = data.frame(beta=NA, HR=NA, se=NA, z=NA, pv=NA)
# unicox_df[1:5] = summary(coxph(surv_test ~ actual_risk))$coef[c(1:5)]

# write.csv(unicox_df, file = "Desktop/final-figures/all-features/unicox_df.csv")






# library(survival)
# library(glmnet)


# require(gdata)
# dat <- read.xls("Desktop/combined.xlsx")
# actual_risk <- read.csv("Desktop/final-figures/total_features/actual_risk.csv")
# actual_risk = actual_risk[,2]

# res.cox <- coxph(Surv(dat[,"time"], dat[,"status"]) ~ actual_risk + dat[,"age"]  + dat[,"gender"] + dat[,"stage"] + dat[,"grade"])

# summary(res.cox)






