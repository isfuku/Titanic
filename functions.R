# some libraries ----
library(ggplot2)
library(dplyr)
library(GGally)

# correlation matrix ----
make_cor_matrix <- function(titanic){
        cor_matrix <- data.frame()
        for (j in 1:ncol(titanic)){
                cor_vec <- c()
                v_name = c()
                if (class(titanic[, j]) %in% c("integer", "numeric")){
                        for (i in 1:ncol(titanic)){
                                if (class(titanic[, i]) %in% c("integer", "numeric")){
                                        cor_vec[i] <- cor(titanic[!is.na(titanic[,j]) &
                                                                          !is.na(titanic[,i]), j],
                                                          titanic[!is.na(titanic[,j]) &
                                                                          !is.na(titanic[,i]), i])
                                        v_name[i] <- names(titanic)[i]
                                }
                        }
                        
                        if (nrow(cor_matrix) == 0){
                                cor_matrix = matrix(cor_vec, 
                                                    nrow = 1) %>% as.data.frame()
                                k = 2
                                colnames(cor_matrix) <- v_name
                        }else{
                                cor_matrix[k, ] <- cor_vec
                                k = k + 1
                        }
                }
        }
        cor_matrix <- cor_matrix[, -which(is.na(names(cor_matrix)))]
        rownames(cor_matrix) <- colnames(cor_matrix)
        return(cor_matrix)
}
strReverse <- function(x){
        sapply(lapply(strsplit(x, NULL), rev), paste, collapse="")
}

# KNN classifier ----
distance <- function(x1, x2){
        sqrt(sum((x1 - x2)^2))
}

bad_knn <- function(x_point, y, X, k = 1){
        X = X[!is.na(y), ]
        y = y[!is.na(y)]
        lower = c()
        output <- c()
        for ( i in 1:nrow(X) ){
                edist = distance(x_point, X[i, ])
                if (length(lower) < k & !is.na(edist)){
                        lower[length(lower) + 1] = edist
                        output[length(lower)] <- y[i]
                }else if (!is.na(edist) & edist < max(lower)){
                        lower[which(lower == max(lower))] <- edist
                        output[which(lower == max(lower))] <- y[i]
                }
                
        }
        l = 1/(lower + 1)
        return( sum(l/sum(l) * output) )
}


# ticket type ----
tick_type <- function(ticket){
        ticket_vector <- strsplit(ticket, "")[[1]]
        ticket_vector <- setdiff(ticket_vector, c("."))
        if (ticket_vector[1] %in% strsplit("0123456789 ", "")[[1]]){
                return ("Numerical")
        }
        ticket = ""
        for (char in ticket_vector){
                if (char %in% strsplit("0123456789 /", "")[[1]]){
                        break
                }
                ticket = paste0(ticket, char)
        }
        return (ticket)
}

# x1, x2, survived plot ----
v1 = "logFare"
v2 = "Age"
biplot <- function(v1, v2){
        max_x = titanic[, which(names(titanic) == v1)] %>% max
        max_y = titanic[, which(names(titanic) == v2)] %>% max
        
        plot(titanic[titanic$Survived == 1, which(names(titanic) == v1)], 
             titanic[titanic$Survived == 1, which(names(titanic) == v2)], 
             col = 'darkgreen', pch = "+", xlab = v1, ylab = v2)
        points(titanic[titanic$Survived == 0, which(names(titanic) == v1)], 
               titanic[titanic$Survived == 0, which(names(titanic) == v2)], 
               col = 'red', pch = "+")
        legend(x = max_x-max_x/10, y =  max_y - max_y/10,
               legend = c("1", "0"), col = c("darkgreen", "red"), pch = "+")
        
}
biplot(v2, v1)

# sigmoid ----
sigmoid <- function(x){
        1/(1+exp(-x))
}
# Reg Logistic Cost and Grad ----

regLogitCost <- function(theta, lambda, X, y){
        ## Calculate the cost J(theta) for regularized logistic regression
        
        ## adding "bias" unit
        X <- as.matrix(X)
        X <- cbind(X, rep(1, nrow(X)))

        ## number of obs and features
        n = nrow(X)
        k = ncol(X)
        
        ## compute cost
        h = sigmoid(X%*%theta)
        J = 1/(2*n)*(sum((h - y)^2) + lambda*(sum(theta^2) - theta[k]^2))
        
        return(J)
}

regLogitGrad <- function(theta, lambda, X, y){
        ## Calculate the gradient of J(theta) for regularized logistic regression
        ## adding "bias" unit
        X <- as.matrix(X)
        X <- cbind(X, rep(1, nrow(X)))
        ## number of obs and features
        n = nrow(X)
        
        ## compute gradient
        h = sigmoid(X%*%theta)
        grad = 1/n*(t(X) %*% (h - y))
        grad[1:(k-1)] <- grad[1:(k-1)] + lambda/n*grad[1:(k-1)]
        
        return(grad)
}

# polynomial features ----
polFeatures <- function(X, p){
        k = ncol(X)
        n = nrow(X)
        X_pol = matrix(rep(0, k*p*n), nrow = n)
        for (j in 1:p){
                X_pol[,((j-1)*k+1):(j*k)] <- X^j
        }
        return (X_pol)
}

# normalizing features ----
normalize <- function(X){
        for (j in 1:ncol(X)){
                sj = max(X[, j]) - min(X[, j])
                X[, j] <- (X[, j] - mean(X[, j])) / sj
        }
        return (X)
}
normalize(X)
