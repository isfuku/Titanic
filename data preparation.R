## My approach: 
##      1. input missing values on features, when needed
##      2. create good features
##      3. Train - CV split (no need to split for test, since I can test on Kaggle)
##      4. fit a basic model (regularized logistic regression)
##      5. plot learning curves and see what to do next

#### Exploratory Data Analysis ####

# libraries ----
library(ggplot2)
library(dplyr)
library(GGally)
source("functions.R")

# data ----
train <- read.csv("data/train.csv")
test <- read.csv("data/test.csv")
test$Survived <- NA
titanic <- rbind(train, test)
head(titanic)
titanic$Survived <- as.factor(titanic$Survived)

# quick look at data  ----
for (j in 2:ncol(titanic)){
        print(paste0("Feature ", names(titanic)[j], ":"))
        if ( length(summary(titanic[, j])) < 8){
                print(summary(titanic[, j]))
        }
        
        cat(paste0("number of NAs: ", sum(is.na(titanic[, j])), "\n\n"))
}
titanic$Sex <- ifelse(titanic$Sex == "male", 1, 0)

# age: 263 NAs
# fare: 1 NA
# cabin: 1014 == ""
# embarked: 2 == ""

# Name ----
nm <- titanic$Name[1:5]

titles = sub(".*, ", "", titanic$Name) %>% strReverse() %>% 
        sub(pattern = ".* \\.", replacement = "") %>%
        strReverse()

ggplot()+
        geom_bar(aes(titles))+
        coord_flip()

# create few categories from these titles
titles %>% unique()

Mr <- titles == c("Ms") # mudar para Mr
Miss <- titles == "Mlle" # mudar para miss
Mrs <- titles %in% c("Lady", "Dona")
Military <- c("Major", "Col", "Capt")
Noble <- c("the Countess", "Sir", "Mme", "Jonkheer")
Others <- c("Don", "Dr", "Rev")

titles[Mr] <- "Mr"
titles[Miss] <- "Miss"
titles[Mrs] <- "Mrs"
titles[titles %in% Military] <- "Military"
titles[titles %in% Noble] <- "Noble"
titles[titles %in% Others] <- "Others"

titanic$Titles <- titles
ggplot(titanic[!is.na(titanic$Survived),])+
        geom_bar(aes(Titles, group = Survived, fill = Survived))+
        coord_flip()

titanic$last_name <- gsub("\\,.*","", titanic$Name)
titanic$iTitles <- titanic$Titles %>% as.character() %>% as.factor() %>% as.integer()

# family size ----
# sibsp: Number of Siblings/Spouses Aboard
# parch: Number of Parents/Children Aboard 

titanic$Family_size = titanic$SibSp + titanic$Parch
table(titanic$Family_size)
ggplot(titanic) + 
        geom_bar(aes(Family_size))
# family size = 4 now means family size >= 4
titanic$Family_size <- ifelse(titanic$Family_size >= 4, 4, titanic$Family_size) 

# age ----

# making a correlation matrix for the integer and numeric variables

cor_matrix <- make_cor_matrix(titanic)
cor_matrix

# let's choose as Age predictors some correlated features
age_na <- is.na(titanic$Age)
if(file.exists("Age_predict.csv")){
    Age_predict <- read.csv("Age_predict.csv")    
}else{ #deveria adicionar tbm o título
        age_preds <- colnames(cor_matrix)[which(cor_matrix$Age^2 > 0.15^2 & cor_matrix$Age != 1)]
        age_preds = c("Pclass", "SibSp",  "Parch",  "Fare")
        y = titanic$Age[!is.na(titanic$Age)]
        X = titanic[!is.na(titanic$Age), which(names(titanic) %in% 
                                                       age_preds)]
        to_pred <- titanic[is.na(titanic$Age), which(names(titanic) %in% 
                                                             age_preds)]
        
        Age_predicit <- c()
        for (i in 1:nrow(to_pred)){
                print(paste0("working on index: ", i, " of ", nrow(to_pred)))
                x_point <- to_pred[i,]
                Age_predicit[i] <- bad_knn(x_point, y, X, 5)
        }
        write.csv(Age_predicit, "Age_predict.csv", row.names = F)
}

titanic$Age[age_na] <- Age_predicit

# data viz
ggplot(titanic[!is.na(titanic$Survived),]) + 
        geom_density(aes(Age, group = Survived, fill = Survived), alpha = 0.5)


# embarked ----
##  "Miss Amelie boarded the Titanic at Southampton
## as maid to Mrs George Nelson Stone."
titanic[titanic$Embarked == "", ]
titanic$Embarked[titanic$Embarked == ""] <- "S"
titanic$Embarked <- titanic$Embarked %>% as.character() %>% as.factor() %>% as.integer()

# data viz
ggplot(titanic[!is.na(titanic$Survived),]) + 
        geom_bar(aes(Embarked, group = Survived, fill = Survived), alpha = 0.5)
titanic$Embarked 

# Fare ----
fare_preds <- colnames(cor_matrix)[which(cor_matrix$Fare^2 > 0.15^2 & cor_matrix$Fare != 1)]
input_fare <- function(obs_na){
        
        preds = titanic[!is.na(titanic$Fare) & titanic$Pclass == obs_na$Pclass &
                                titanic$SibSp == obs_na$SibSp &
                                titanic$Sex == obs_na$Sex & 
                                titanic$Age %in% (obs_na$Age - 10):(obs_na$Age+10) &
                                titanic$Parch == obs_na$Parch,]
        return(median(preds$Fare))  
}
titanic$Fare[is.na(titanic$Fare)] <- input_fare(titanic[is.na(titanic$Fare), ])

ggplot(titanic[!is.na(titanic$Survived),]) + 
        geom_density(aes(Fare, group = Survived, fill = Survived), alpha = 0.5)

# taking log to smooth (idea from kaggle)
titanic$logFare <- ifelse(titanic$Fare > 0, log(titanic$Fare), 0)

ggplot(titanic[!is.na(titanic$Survived),]) + 
        geom_density(aes(logFare, group = Survived, fill = Survived), alpha = 0.5)

# Cabin ----

titanic$cabin_l <- titanic$Cabin %>% substr(1,1)

# Ticket ----
titanic$Ticket <- as.character(titanic$Ticket)
titanic$ticket_t <- unlist(lapply(titanic$Ticket, FUN = tick_type))

# vizualisation
ggplot(titanic[!is.na(titanic$Survived),]) + 
        geom_bar(aes(ticket_t, group = Survived, fill = Survived)) +
        coord_flip()
titanic$ticket_t <- titanic$ticket_t %>% as.character() %>% as.factor() %>% as.integer()

# Some more Vizualisation ----
toPlot <- names(titanic)[c(3, 5, 6, 12, 14, 15, 18, 20)]

for(v1 in toPlot){
        for (v2 in toPlot){
                biplot(v1, v2)
        }
}
cor_matrix <- make_cor_matrix(titanic)
cor_matrix
ggcorr(titanic[, which(names(titanic) %in% toPlot)], method = c("everything", "pearson")) 


# Save file... ----
write.csv(titanic, "data/titanic.csv", row.names = FALSE)
