#### model fit ####
# 1. split train-CV
# 2. create features X
# 3. fit regularized regression
# 4. plot learning curves for different lambda and polynomial order
# 5. pick best lambda and pol. order and submit

# regularized logistic regression: 
#       1. cost and gradiant function
#       2. make pol. features function
#       3. find theta that minimize cost
#       4. stonks


### Data and etc ----
source("functions.R")
titanic <- read.csv("data/titanic.csv")

### choosing some features ----
names(titanic)
titanic <- titanic[, which(names(titanic) %in% c("PassengerId", "Survived", "Pclass", "Sex", 
                                                 "Age", "Embarked",
                                                 "Family_size", "logFare", "ticket_t", 
                                                 "iTitles"))]

### splitting the dataset in train and cv ----
set.seed(12)
trainf <- titanic[!is.na(titanic$Survived), ]
train_examples <- sample(1:nrow(trainf), round(nrow(trainf)*0.7))
y = trainf$Survived %>% as.character %>% strtoi()
X = trainf[, which(names(trainf) %in% c("Pclass", "Sex", 
                                        "Age", "Embarked",
                                        "Family_size", "logFare", "ticket_t", 
                                        "iTitles"))]
X <- as.matrix(X)

# choose p ----
lambda = 0.01
p_space <- 1:50
cv_cost <- as.data.frame(matrix(p_space, nrow = 1))
train_cost <- as.data.frame(matrix(p_space, nrow = 1))
for (random_trial in 1:50){
        train_examples <- sample(1:nrow(trainf), round(nrow(trainf)*0.7))
        train_cost_trial = c()
        cv_cost_trial = c()
        for(p in p_space){
                Xp <- polFeatures(X, p = p)
                Xp <- normalize(Xp)
                Xcv <- Xp[-train_examples, ]
                ycv <- y[-train_examples]
                Xtrain <- Xp[train_examples, ]
                ytrain <- y[train_examples]
                theta = optim(rep(0, ncol(Xtrain) + 1), fn = regLogitCost, gr = regLogitGrad, 
                              lambda = lambda, X = Xtrain, y = ytrain,
                              method = "BFGS")
                train_cost_trial[p] <- regLogitCost(theta$par, lambda = 0, 
                                                    X = Xtrain, y = ytrain)
                cv_cost_trial[p] <- regLogitCost(theta$par, lambda = 0, X = Xcv, y = ycv)
        }
        cv_cost[random_trial, ] <- cv_cost_trial
        train_cost[random_trial, ] <- train_cost_trial
}

cv_cost_mean <- colMeans(cv_cost)
train_cost_mean <- colMeans(train_cost)

plot(x = p_space, y = cv_cost_mean, col = 'blue', type = 'l', 
     ylim = c(min(union(cv_cost_mean, train_cost_mean)), 
              max(union(cv_cost_mean, train_cost_mean))), 
     main = paste0("p = ", p), ylab = "MSE")
lines(x = p_space, y = train_cost_mean, col = 'green')
legend(x = max(p_space) - max(p_space)/6,
       y =  max(union(cv_cost_mean, train_cost_mean)) - 
               max(union(cv_cost_mean, train_cost_mean))/100,
       legend = c("train", "cv"), col = c("green", "blue"), lty = 1)

p_optm <- p_space[which(cv_cost_mean == min(cv_cost_mean))]
        
# see lambda ----
p = 50
Xp <- polFeatures(X, p = p)
Xp <- normalize(Xp)

lambda_space <- c(0, 0.001, 0.003, 0.008, 0.01, 0.03, 0.08, 0.1, 0.3, 0.8, 1, 1.2, 1.5)
cv_cost <- as.data.frame(matrix(lambda_space, nrow = 1))
train_cost <- as.data.frame(matrix(lambda_space, nrow = 1))

for (random_trial in 1:50){
        train_examples <- sample(1:nrow(trainf), round(nrow(trainf)*0.7))
        train_cost_trial = c()
        cv_cost_trial = c()
        for(i in 1:length(lambda_space)){
                lambda = lambda_space[i]
                Xcv <- Xp[-train_examples, ]
                ycv <- y[-train_examples]
                Xtrain <- Xp[train_examples, ]
                ytrain <- y[train_examples]
                theta = optim(rep(0, ncol(Xtrain) + 1), fn = regLogitCost, gr = regLogitGrad, 
                              lambda = lambda, X = Xtrain, y = ytrain,
                              method = "BFGS")
                train_cost_trial[i] <- regLogitCost(theta$par, lambda = 0, 
                                                    X = Xtrain, y = ytrain)
                cv_cost_trial[i] <- regLogitCost(theta$par, lambda = 0, X = Xcv, y = ycv)
        }
        cv_cost[random_trial, ] <- cv_cost_trial
        train_cost[random_trial, ] <- train_cost_trial
}

cv_cost <- colMeans(cv_cost)
train_cost <- colMeans(train_cost)

plot(x = lambda_space, y = cv_cost, col = 'blue', type = 'l', ylab = "MSE",
     ylim = c(min(union(cv_cost, train_cost)), 
              max(union(cv_cost, train_cost))), main = paste0("p = ", p))
lines(x = lambda_space, y = train_cost, col = 'green')
legend(x = max(lambda_space) - max(lambda_space)/6,
       y =  max(union(cv_cost, train_cost)) - max(union(cv_cost, train_cost))/100,
       legend = c("train", "cv"), col = c("green", "blue"), lty = 1)

optimal_lambda = lambda_space[which(cv_cost == min(cv_cost))]
print(paste0("optimal lambda = ", optimal_lambda))


# learning curves ----
lambda = 0.01
m_space = c(17, 30, seq(nrow(Xtrain)/10, nrow(Xtrain), 10)) %>% round()
cv_cost <- as.data.frame(matrix(m_space, nrow = 1))
train_cost <- as.data.frame(matrix(m_space, nrow = 1))
for (random_trial in 1:50){
        train_examples <- sample(1:nrow(trainf), round(nrow(trainf)*0.7))
        train_cost_trial = c()
        cv_cost_trial = c()
        for(i in 1:length(m_space)){
                m_sample = sample(1:nrow(Xtrain), m_space[i])
                theta_init = rep(0, ncol(Xtrain) + 1)
                Xm = Xtrain[m_sample, ]
                ym = ytrain[m_sample]
                theta = optim(theta_init, fn = regLogitCost, gr = regLogitGrad, 
                              lambda = lambda, X = Xm, y = ym,
                              method = "BFGS")
                train_cost_trial[i] <- regLogitCost(theta$par, lambda = 0, 
                                                    X = Xm, y = ym)
                cv_cost_trial[i] <- regLogitCost(theta$par, lambda = 0, X = Xcv, y = ycv)
        }
        cv_cost[random_trial, ] <- cv_cost_trial
        train_cost[random_trial, ] <- train_cost_trial
}

cv_cost <- colMeans(cv_cost)
train_cost <- colMeans(train_cost)

plot(x = m_space, y = cv_cost, col = 'blue', type = 'l', 
     ylim = c(min(union(cv_cost, train_cost)), 
              max(union(cv_cost, train_cost))), main = paste0("p = ", p))
lines(x = m_space, y = train_cost, col = 'green')
legend(x = max(m_space) - max(m_space)/6,
       y =  max(union(cv_cost, train_cost)) - max(union(cv_cost, train_cost))/100,
       legend = c("train", "cv"), col = c("green", "blue"), lty = 1)

# one fit ----
p = 50
lambda = 0.01
X <- titanic[, which(names(trainf) %in% c("Pclass", "Sex", 
                                        "Age", "Embarked",
                                        "Family_size", "logFare", "ticket_t", 
                                        "iTitles"))]
X <- as.matrix(X)
X <- polFeatures(X, p = p)
X <- normalize(X)
test <- X[titanic$Survived %>% is.na(),]
X <- X[!is.na(titanic$Survived), ]
theta_init = rep(0, ncol(X) + 1)
theta = optim(par = theta_init, fn = regLogitCost, gr = regLogitGrad, 
              lambda = lambda, X = X, y = y, method = "BFGS")

test <- cbind(test, rep(1, nrow(test)))
y_pred <- sigmoid(test%*%theta$par)

y_pred_class_50 <- ifelse(y_pred > 0.5, 1, 0)
test.send <- read.csv("data/test.csv")
test.send <- cbind(test.send$PassengerId, y_pred_class_50)
colnames(test.send) <- c("PassengerId", "Survived")
write.csv(test.send, "data/reg50_submission50_001.csv", row.names = F)

