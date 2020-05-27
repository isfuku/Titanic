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
set.seed(144)
trainf <- titanic[!is.na(titanic$Survived), ]
train_examples <- sample(1:nrow(trainf), round(nrow(trainf)*0.7))

# choose p ----
y = trainf$Survived %>% as.character %>% strtoi()
X = trainf[, which(names(trainf) %in% c("Pclass", "Sex", 
                                        "Age", "Embarked",
                                        "Family_size", "logFare", "ticket_t", 
                                        "iTitles"))]
X <- as.matrix(X)


theta_init = rep(0, ncol(X) + 1)
theta = optim(par = theta_init, fn = regLogitCost, gr = regLogitGrad, 
              lambda = 0, X = X, y = y)
theta$par
theta$value

lambda = 1
p_space <- 1:10
cv_cost <- as.data.frame(matrix(p_space, nrow = 1))
train_cost <- as.data.frame(matrix(p_space, nrow = 1))
for (random_trial in 1:10){
        train_examples <- sample(1:nrow(trainf), round(nrow(trainf)*0.7))
        train_cost_trial = c()
        cv_cost_trial = c()
        for(p in p_space){
                X <- polFeatures(X, p = p)
                X <- normalize(X)
                Xcv <- X[-train_examples, ]
                ycv <- y[-train_examples]
                Xtrain <- X[train_examples, ]
                ytrain <- y[train_examples]
                theta = optim(rep(0, ncol(X) + 1), fn = regLogitCost, gr = regLogitGrad, 
                              lambda = lambda, X = X, y = y)
                train_cost_trial[i] <- regLogitCost(theta$par, lambda = 0, X = X, y = y)
                cv_cost_trial[i] <- regLogitCost(theta$par, lambda = 0, X = Xcv, y = ycv)
        }
        cv_cost[random_trial, ] <- cv_cost_trial
        train_cost[random_trial, ] <- train_cost_trial
}

cv_cost_mean <- rowMeans(cv_cost)
train_cost_mean <- rowMeas(train_cost)

plot(x = p_space, y = cv_cost_mean, col = 'blue', type = 'l', 
     ylim = c(min(union(cv_cost_mean, train_cost_mean)), 
              max(union(cv_cost_mean, train_cost_mean))), main = paste0("p = ", p))
lines(x = p_space, y = train_cost, col = 'green')
legend(x = max(lambda_space) - max(lambda_space)/6,
       y =  max(union(cv_cost, train_cost)) - max(union(cv_cost, train_cost))/100,
       legend = c("train", "cv"), col = c("green", "blue"), lty = 1)

optimal_lambda = lambda_space[which(cv_cost == min(cv_cost))]
print(paste0("optimal lambda = ", optimal_lambda))

# p = 4, see lambda ----
y = trainf$Survived %>% as.character %>% strtoi()
X = trainf[, which(names(trainf) %in% c("Pclass", "Sex", 
                                        "Age", "Embarked",
                                        "Family_size", "logFare", "ticket_t", 
                                        "iTitles"))]
X <- as.matrix(X)
p = 4
X <- polFeatures(X, p = p)
X <- normalize(X)
Xcv <- X[-train_examples, ]
ycv <- y[-train_examples]
X <- X[train_examples, ]
y <- y[train_examples]

theta_init = rep(0, ncol(X) + 1)
theta = optim(par = theta_init, fn = regLogitCost, gr = regLogitGrad, 
              lambda = 0, X = X, y = y)
theta$par
theta$value

lambda_space <- c(0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6, 9, 14)
p_space <- 1:10
cv_cost <- c()
train_cost <- c()
for(i in 1:length(lambda_space)){
        theta_init = rep(0, ncol(X) + 1)
        theta = optim(par = theta_init, fn = regLogitCost, gr = regLogitGrad, 
                      lambda = lambda_space[i], X = X, y = y)
        train_cost[i] <- regLogitCost(theta$par, lambda = 0, X = X, y = y)
        cv_cost[i] <- regLogitCost(theta$par, lambda = 0, X = Xcv, y = ycv)
        print(theta$par)
}

plot(x = lambda_space, y = cv_cost, col = 'blue', type = 'l', 
     ylim = c(min(union(cv_cost, train_cost)), 
              max(union(cv_cost, train_cost))), main = paste0("p = ", p))
lines(x = lambda_space, y = train_cost, col = 'green')
legend(x = max(lambda_space) - max(lambda_space)/6,
       y =  max(union(cv_cost, train_cost)) - max(union(cv_cost, train_cost))/100,
       legend = c("train", "cv"), col = c("green", "blue"), lty = 1)

optimal_lambda = lambda_space[which(cv_cost == min(cv_cost))]
print(paste0("optimal lambda = ", optimal_lambda))


# learning curves ----
m_space = seq(nrow(X)/10, nrow(X), 10)
cv_cost <- c()
train_cost <- c()
for(i in 1:length(m_space)){
        theta_init = rep(0, ncol(X) + 1)
        theta = optim(par = theta_init, fn = regLogitCost, gr = regLogitGrad, 
                      lambda = optimal_lambda, X = X, y = y)
        m_sample = sample(1:nrow(X), m_space[i])
        Xm = X[m_sample, ]
        ym = y[m_sample]
        train_cost[i] <- regLogitCost(theta$par, lambda = 0, X = Xm, y = ym)
        cv_cost[i] <- regLogitCost(theta$par, lambda = 0, X = Xcv, y = ycv)
        print(theta$par)
}

plot(x = m_space, y = cv_cost, col = 'blue', type = 'l', 
     ylim = c(min(union(cv_cost, train_cost)), 
              max(union(cv_cost, train_cost))), main = paste0("p = ", p))
lines(x = m_space, y = train_cost, col = 'green')
legend(x = max(m_space) - max(m_space)/6,
       y =  max(union(cv_cost, train_cost)) - max(union(cv_cost, train_cost))/100,
       legend = c("train", "cv"), col = c("green", "blue"), lty = 1)

optimal_lambda = lambda_space[which(cv_cost == min(cv_cost))]
print(paste0("optimal lambda = ", optimal_lambda))


# one fit ----
y = trainf$Survived %>% as.character %>% strtoi()
X = trainf[, which(names(trainf) %in% c("Pclass", "Sex", 
                                        "Age", "Embarked",
                                        "Family_size", "logFare", "ticket_t", 
                                        "iTitles"))]
X <- as.matrix(X)
p = 4
X <- polFeatures(X, p = p)
X <- normalize(X)

theta_init = rep(0, ncol(X) + 1)
theta = optim(par = theta_init, fn = regLogitCost, gr = regLogitGrad, 
              lambda = 0.8, X = X, y = y)
theta$par
theta$value

test <- titanic[is.na(titanic$Survived), which(names(trainf) %in% c("Pclass", "Sex", 
                                             "Age", "Embarked",
                                             "Family_size", "logFare", "ticket_t", 
                                             "iTitles"))]
test <- as.matrix(test)
p = 4
test <- polFeatures(test, p)
test <- normalize(test)
test <- cbind(test, rep(1, nrow(test)))
y_pred <- sigmoid(test%*%theta$par)

y_pred_class_50 <- ifelse(y_pred > 0.5, 1, 0)
test.send <- read.csv("data/test.csv")
test.send <- cbind(test.send$PassengerId, y_pred_class_50)
colnames(test.send) <- c("PassengerId", "Survived")
write.csv(test.send, "data/reg50_submission.csv", row.names = F)

