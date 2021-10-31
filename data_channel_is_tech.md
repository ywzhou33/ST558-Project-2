ST558 Project 2
================
Aries Zhou & Jiatao Wang
10/19/2021

## Introduction

This is a R project using the exploratory data analysis and supervised
statistical learning method to analyze a data set.  
This data set is called **Online News Popularity Data Set** and you can
access the data set
[here](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity)  
There are lots of measurements/heterogeneous features of articles,
including type of the data channel, number of images, number of videos,
number of links, counts of words in the title/content, when it is
published, summary statistics of polarity of positive/negative words and
etc…  
The **main goal** of this project is to use those features/explanatory
variables to predict the popularity(number of the shares in social
networks)  
Before conducting any method to fit the data with models, we want to do
some exploratory data analysis (including some summary statistics and
graphs) to visualize the data. And then, we will fit the data under
regression setting.  
Supervised learning methods that will be used in this project include:
linear regression, generalized linear model, lasso regression, random
forest regression, boosted method, or any other method that we will find
that could be applicable through our discovering of the data.

List of packages used:

``` r
library(dplyr)
library(tidyr)
library(ggcorrplot)
library(vcd)
library(caret)
library(class)
library(randomForest)
library(gbm)
library(readr)
library(leaps)
library(Matrix)
library(glmnet)
library(rmarkdown)
library(doParallel)
```

## Data Cleaning

### Data

Read in data and transpose data\_channel\_is\* and weekday\_is\* columns
into categorical columns.

``` r
# import data
pop <- read_csv("OnlineNewsPopularity.csv")

# check if there is any missing or NA values in the data set 
anyNA(pop) # returned FALSE, so no missing values 
```

    ## [1] FALSE

``` r
# convert the wide to long format (categorize data channel, and make them into one column)
new <- pop %>% pivot_longer(cols = data_channel_is_lifestyle:data_channel_is_world, names_to = "channel",values_to = 'logi.num.d') 
new_data <- new %>% filter(logi.num.d != 0) %>% select(-logi.num.d) # drop logical number

# merge those weekday columns into one.
Z <- new_data %>% pivot_longer(cols = weekday_is_monday:weekday_is_sunday, names_to = "weekday",values_to = 'logi.num.w') 
X <- Z %>% filter(logi.num.w != 0) %>% select(-logi.num.w) # drop logical numbers
```

Subset data on data channel of interest for analysis and set the params
to do automation.

``` r
pop.data <- X %>% filter(channel == params$channel) %>% select(-1:-2)
pop.data$is_weekend <- as.factor(pop.data$is_weekend)
pop.data$weekday <- as.factor(pop.data$weekday)
```

``` r
nrow(new_data)< nrow(pop)
```

    ## [1] TRUE

Since `nrow(new_data)< nrow(pop)` returned `TRUE`, indicating that there
are some observations that are not in the types of channel listed in the
data set.

## Exploratory Data Analysis

### Summarizations And Graphs

``` r
#check data structures.
str(pop.data)
```

    ## tibble [7,346 × 48] (S3: tbl_df/tbl/data.frame)
    ##  $ n_tokens_title              : num [1:7346] 13 10 12 11 8 13 11 8 8 12 ...
    ##  $ n_tokens_content            : num [1:7346] 1072 370 989 97 1207 ...
    ##  $ n_unique_tokens             : num [1:7346] 0.416 0.56 0.434 0.67 0.411 ...
    ##  $ n_non_stop_words            : num [1:7346] 1 1 1 1 1 ...
    ##  $ n_non_stop_unique_tokens    : num [1:7346] 0.541 0.698 0.572 0.837 0.549 ...
    ##  $ num_hrefs                   : num [1:7346] 19 2 20 2 24 21 20 5 5 22 ...
    ##  $ num_self_hrefs              : num [1:7346] 19 2 20 0 24 19 20 2 3 22 ...
    ##  $ num_imgs                    : num [1:7346] 20 0 20 0 42 20 20 1 1 28 ...
    ##  $ num_videos                  : num [1:7346] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ average_token_length        : num [1:7346] 4.68 4.36 4.62 4.86 4.72 ...
    ##  $ num_keywords                : num [1:7346] 7 9 9 7 8 10 7 10 9 9 ...
    ##  $ kw_min_min                  : num [1:7346] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kw_max_min                  : num [1:7346] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kw_avg_min                  : num [1:7346] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kw_min_max                  : num [1:7346] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kw_max_max                  : num [1:7346] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kw_avg_max                  : num [1:7346] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kw_min_avg                  : num [1:7346] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kw_max_avg                  : num [1:7346] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kw_avg_avg                  : num [1:7346] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ self_reference_min_shares   : num [1:7346] 545 8500 545 0 545 545 545 924 2500 545 ...
    ##  $ self_reference_max_shares   : num [1:7346] 16000 8500 16000 0 16000 16000 16000 924 2500 16000 ...
    ##  $ self_reference_avg_sharess  : num [1:7346] 3151 8500 3151 0 2830 ...
    ##  $ is_weekend                  : Factor w/ 2 levels "0","1": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ LDA_00                      : num [1:7346] 0.0286 0.0222 0.0222 0.4583 0.025 ...
    ##  $ LDA_01                      : num [1:7346] 0.0288 0.3067 0.1507 0.029 0.0252 ...
    ##  $ LDA_02                      : num [1:7346] 0.0286 0.0222 0.2434 0.0287 0.025 ...
    ##  $ LDA_03                      : num [1:7346] 0.0286 0.0222 0.0222 0.0297 0.025 ...
    ##  $ LDA_04                      : num [1:7346] 0.885 0.627 0.561 0.454 0.9 ...
    ##  $ global_subjectivity         : num [1:7346] 0.514 0.437 0.543 0.539 0.539 ...
    ##  $ global_sentiment_polarity   : num [1:7346] 0.281 0.0712 0.2986 0.1611 0.2883 ...
    ##  $ global_rate_positive_words  : num [1:7346] 0.0746 0.0297 0.0839 0.0309 0.0696 ...
    ##  $ global_rate_negative_words  : num [1:7346] 0.0121 0.027 0.0152 0.0206 0.0116 ...
    ##  $ rate_positive_words         : num [1:7346] 0.86 0.524 0.847 0.6 0.857 ...
    ##  $ rate_negative_words         : num [1:7346] 0.14 0.476 0.153 0.4 0.143 ...
    ##  $ avg_positive_polarity       : num [1:7346] 0.411 0.351 0.428 0.567 0.427 ...
    ##  $ min_positive_polarity       : num [1:7346] 0.0333 0.1364 0.1 0.4 0.1 ...
    ##  $ max_positive_polarity       : num [1:7346] 1 0.6 1 0.8 1 1 1 0.35 1 1 ...
    ##  $ avg_negative_polarity       : num [1:7346] -0.22 -0.195 -0.243 -0.125 -0.227 ...
    ##  $ min_negative_polarity       : num [1:7346] -0.5 -0.4 -0.5 -0.125 -0.5 -0.5 -0.5 -0.2 -0.5 -0.5 ...
    ##  $ max_negative_polarity       : num [1:7346] -0.05 -0.1 -0.05 -0.125 -0.05 -0.05 -0.05 -0.05 -0.125 -0.05 ...
    ##  $ title_subjectivity          : num [1:7346] 0.455 0.643 1 0.125 0.5 ...
    ##  $ title_sentiment_polarity    : num [1:7346] 0.136 0.214 0.5 0 0 ...
    ##  $ abs_title_subjectivity      : num [1:7346] 0.0455 0.1429 0.5 0.375 0 ...
    ##  $ abs_title_sentiment_polarity: num [1:7346] 0.136 0.214 0.5 0 0 ...
    ##  $ shares                      : num [1:7346] 505 855 891 3600 17100 2800 445 783 1500 1800 ...
    ##  $ channel                     : chr [1:7346] "data_channel_is_tech" "data_channel_is_tech" "data_channel_is_tech" "data_channel_is_tech" ...
    ##  $ weekday                     : Factor w/ 7 levels "weekday_is_friday",..: 2 2 2 2 2 2 2 2 2 2 ...

``` r
#summary stats for the response variable. 
summary(pop.data$shares)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##      36    1100    1700    3072    3000  663600

The distribution of the response variable (shares) is:  
\- **Right-skewed** if its mean is **greater** than its median.  
\- **Left-skewed** if its mean is **less** than its median.  
\- **Normal** if its mean **equals** to its median.

Check correlations.

``` r
#get all numeric variables without collinearity
pop.data.num <- select(pop.data, is.numeric) %>% mutate_all(~(scale(.) %>% as.vector)) 

# due to the large number of variables, try to get a best subset with the stepwise method. 
lm <- step(lm(shares ~ ., data = pop.data.num))
```

    ## Start:  AIC=-80.73
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + kw_min_min + kw_max_min + kw_avg_min + kw_min_max + 
    ##     kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_01 + LDA_02 + LDA_03 + LDA_04 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + rate_negative_words + 
    ##     avg_positive_polarity + min_positive_polarity + max_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_subjectivity + title_sentiment_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity
    ## 
    ## 
    ## Step:  AIC=-80.73
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + kw_min_min + kw_max_min + kw_avg_min + kw_min_max + 
    ##     kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_01 + LDA_02 + LDA_03 + LDA_04 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ## 
    ## Step:  AIC=-80.73
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + kw_min_min + kw_max_min + kw_avg_min + kw_min_max + 
    ##     kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - LDA_02                        1    0.0026 7181.1 -82.728
    ## - LDA_01                        1    0.0027 7181.1 -82.728
    ## - title_subjectivity            1    0.0086 7181.2 -82.722
    ## - max_positive_polarity         1    0.0089 7181.2 -82.722
    ## - n_unique_tokens               1    0.0149 7181.2 -82.716
    ## - global_sentiment_polarity     1    0.0169 7181.2 -82.714
    ## - LDA_03                        1    0.0475 7181.2 -82.682
    ## - global_rate_negative_words    1    0.0529 7181.2 -82.677
    ## - n_non_stop_unique_tokens      1    0.0853 7181.2 -82.644
    ## - min_positive_polarity         1    0.1057 7181.3 -82.623
    ## - abs_title_sentiment_polarity  1    0.1142 7181.3 -82.614
    ## - avg_positive_polarity         1    0.1197 7181.3 -82.609
    ## - global_subjectivity           1    0.1388 7181.3 -82.589
    ## - num_keywords                  1    0.1643 7181.3 -82.563
    ## - kw_min_max                    1    0.2351 7181.4 -82.491
    ## - rate_positive_words           1    0.2701 7181.4 -82.455
    ## - self_reference_min_shares     1    0.3456 7181.5 -82.377
    ## - self_reference_max_shares     1    0.4737 7181.6 -82.246
    ## - average_token_length          1    0.4759 7181.6 -82.244
    ## - n_non_stop_words              1    0.4802 7181.6 -82.240
    ## - kw_max_max                    1    0.7081 7181.9 -82.007
    ## - kw_min_avg                    1    0.7107 7181.9 -82.004
    ## - self_reference_avg_sharess    1    0.7567 7181.9 -81.957
    ## - n_tokens_title                1    1.1251 7182.3 -81.580
    ## - kw_max_min                    1    1.2117 7182.4 -81.492
    ## - title_sentiment_polarity      1    1.3333 7182.5 -81.367
    ## - abs_title_subjectivity        1    1.3382 7182.5 -81.362
    ## - global_rate_positive_words    1    1.6010 7182.7 -81.093
    ## <none>                                      7181.1 -80.731
    ## - kw_avg_min                    1    2.8758 7184.0 -79.790
    ## - kw_min_min                    1    3.2293 7184.4 -79.428
    ## - num_videos                    1    3.7504 7184.9 -78.895
    ## - num_imgs                      1    3.8278 7185.0 -78.816
    ## - kw_max_avg                    1    4.1538 7185.3 -78.483
    ## - LDA_00                        1    4.3593 7185.5 -78.273
    ## - kw_avg_max                    1    8.4710 7189.6 -74.071
    ## - max_negative_polarity         1   10.6132 7191.8 -71.882
    ## - n_tokens_content              1   12.2754 7193.4 -70.184
    ## - num_self_hrefs                1   14.2400 7195.4 -68.179
    ## - kw_avg_avg                    1   14.2762 7195.4 -68.142
    ## - avg_negative_polarity         1   16.0037 7197.2 -66.378
    ## - min_negative_polarity         1   19.8618 7201.0 -62.441
    ## - num_hrefs                     1   22.6119 7203.8 -59.636
    ## 
    ## Step:  AIC=-82.73
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + kw_min_min + kw_max_min + kw_avg_min + kw_min_max + 
    ##     kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_01 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_positive_words + global_rate_negative_words + 
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     max_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_subjectivity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - LDA_01                        1    0.0037 7181.2 -84.725
    ## - title_subjectivity            1    0.0085 7181.2 -84.720
    ## - max_positive_polarity         1    0.0087 7181.2 -84.719
    ## - n_unique_tokens               1    0.0155 7181.2 -84.712
    ## - global_sentiment_polarity     1    0.0167 7181.2 -84.711
    ## - LDA_03                        1    0.0455 7181.2 -84.682
    ## - global_rate_negative_words    1    0.0528 7181.2 -84.674
    ## - n_non_stop_unique_tokens      1    0.0840 7181.2 -84.642
    ## - min_positive_polarity         1    0.1061 7181.3 -84.620
    ## - abs_title_sentiment_polarity  1    0.1145 7181.3 -84.611
    ## - avg_positive_polarity         1    0.1194 7181.3 -84.606
    ## - global_subjectivity           1    0.1373 7181.3 -84.588
    ## - num_keywords                  1    0.1639 7181.3 -84.561
    ## - kw_min_max                    1    0.2354 7181.4 -84.488
    ## - rate_positive_words           1    0.2708 7181.4 -84.451
    ## - self_reference_min_shares     1    0.3477 7181.5 -84.373
    ## - self_reference_max_shares     1    0.4748 7181.6 -84.243
    ## - n_non_stop_words              1    0.4781 7181.6 -84.239
    ## - average_token_length          1    0.4796 7181.6 -84.238
    ## - kw_max_max                    1    0.7088 7181.9 -84.003
    ## - kw_min_avg                    1    0.7100 7181.9 -84.002
    ## - self_reference_avg_sharess    1    0.7593 7181.9 -83.952
    ## - n_tokens_title                1    1.1263 7182.3 -83.576
    ## - kw_max_min                    1    1.2098 7182.4 -83.491
    ## - title_sentiment_polarity      1    1.3328 7182.5 -83.365
    ## - abs_title_subjectivity        1    1.3381 7182.5 -83.360
    ## - global_rate_positive_words    1    1.6016 7182.8 -83.090
    ## <none>                                      7181.1 -82.728
    ## - kw_avg_min                    1    2.8832 7184.0 -81.780
    ## - kw_min_min                    1    3.2266 7184.4 -81.428
    ## - num_videos                    1    3.7478 7184.9 -80.895
    ## - num_imgs                      1    3.8537 7185.0 -80.787
    ## - kw_max_avg                    1    4.1623 7185.3 -80.472
    ## - LDA_00                        1    4.4011 7185.6 -80.228
    ## - kw_avg_max                    1    8.4730 7189.6 -76.066
    ## - max_negative_polarity         1   10.6262 7191.8 -73.866
    ## - n_tokens_content              1   12.2743 7193.4 -72.183
    ## - kw_avg_avg                    1   14.3889 7195.5 -70.024
    ## - num_self_hrefs                1   14.4544 7195.6 -69.957
    ## - avg_negative_polarity         1   16.0264 7197.2 -68.352
    ## - min_negative_polarity         1   19.8619 7201.0 -64.438
    ## - num_hrefs                     1   22.7419 7203.9 -61.501
    ## 
    ## Step:  AIC=-84.72
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + kw_min_min + kw_max_min + kw_avg_min + kw_min_max + 
    ##     kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_positive_words + global_rate_negative_words + 
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     max_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_subjectivity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - title_subjectivity            1    0.0087 7181.2 -86.716
    ## - max_positive_polarity         1    0.0087 7181.2 -86.716
    ## - n_unique_tokens               1    0.0155 7181.2 -86.709
    ## - global_sentiment_polarity     1    0.0165 7181.2 -86.708
    ## - LDA_03                        1    0.0481 7181.2 -86.675
    ## - global_rate_negative_words    1    0.0514 7181.2 -86.672
    ## - n_non_stop_unique_tokens      1    0.0847 7181.2 -86.638
    ## - min_positive_polarity         1    0.1062 7181.3 -86.616
    ## - abs_title_sentiment_polarity  1    0.1139 7181.3 -86.608
    ## - avg_positive_polarity         1    0.1192 7181.3 -86.603
    ## - global_subjectivity           1    0.1372 7181.3 -86.584
    ## - num_keywords                  1    0.1629 7181.3 -86.558
    ## - kw_min_max                    1    0.2363 7181.4 -86.483
    ## - rate_positive_words           1    0.2726 7181.4 -86.446
    ## - self_reference_min_shares     1    0.3478 7181.5 -86.369
    ## - self_reference_max_shares     1    0.4746 7181.6 -86.239
    ## - average_token_length          1    0.4784 7181.6 -86.235
    ## - n_non_stop_words              1    0.4805 7181.6 -86.233
    ## - kw_max_max                    1    0.7089 7181.9 -85.999
    ## - kw_min_avg                    1    0.7095 7181.9 -85.999
    ## - self_reference_avg_sharess    1    0.7595 7181.9 -85.948
    ## - n_tokens_title                1    1.1249 7182.3 -85.574
    ## - kw_max_min                    1    1.2159 7182.4 -85.481
    ## - title_sentiment_polarity      1    1.3322 7182.5 -85.362
    ## - abs_title_subjectivity        1    1.3377 7182.5 -85.356
    ## - global_rate_positive_words    1    1.5988 7182.8 -85.089
    ## <none>                                      7181.2 -84.725
    ## - kw_avg_min                    1    2.8926 7184.0 -83.766
    ## - kw_min_min                    1    3.2280 7184.4 -83.423
    ## - num_videos                    1    3.7457 7184.9 -82.894
    ## - num_imgs                      1    3.8647 7185.0 -82.772
    ## - kw_max_avg                    1    4.1881 7185.3 -82.441
    ## - LDA_00                        1    4.4327 7185.6 -82.191
    ## - kw_avg_max                    1    8.4693 7189.6 -78.066
    ## - max_negative_polarity         1   10.6225 7191.8 -75.866
    ## - n_tokens_content              1   12.2871 7193.4 -74.166
    ## - num_self_hrefs                1   14.4703 7195.6 -71.937
    ## - kw_avg_avg                    1   14.4811 7195.6 -71.926
    ## - avg_negative_polarity         1   16.0274 7197.2 -70.347
    ## - min_negative_polarity         1   19.8585 7201.0 -66.438
    ## - num_hrefs                     1   22.7549 7203.9 -63.484
    ## 
    ## Step:  AIC=-86.72
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + kw_min_min + kw_max_min + kw_avg_min + kw_min_max + 
    ##     kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_positive_words + global_rate_negative_words + 
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     max_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_sentiment_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - max_positive_polarity         1    0.0088 7181.2 -88.707
    ## - n_unique_tokens               1    0.0157 7181.2 -88.700
    ## - global_sentiment_polarity     1    0.0177 7181.2 -88.698
    ## - LDA_03                        1    0.0466 7181.2 -88.668
    ## - global_rate_negative_words    1    0.0522 7181.2 -88.662
    ## - n_non_stop_unique_tokens      1    0.0849 7181.2 -88.629
    ## - min_positive_polarity         1    0.1071 7181.3 -88.606
    ## - avg_positive_polarity         1    0.1186 7181.3 -88.594
    ## - global_subjectivity           1    0.1316 7181.3 -88.581
    ## - num_keywords                  1    0.1623 7181.3 -88.550
    ## - abs_title_sentiment_polarity  1    0.2176 7181.4 -88.493
    ## - kw_min_max                    1    0.2369 7181.4 -88.473
    ## - rate_positive_words           1    0.2716 7181.4 -88.438
    ## - self_reference_min_shares     1    0.3484 7181.5 -88.359
    ## - self_reference_max_shares     1    0.4762 7181.6 -88.228
    ## - average_token_length          1    0.4809 7181.6 -88.224
    ## - n_non_stop_words              1    0.4830 7181.6 -88.221
    ## - kw_min_avg                    1    0.7074 7181.9 -87.992
    ## - kw_max_max                    1    0.7108 7181.9 -87.989
    ## - self_reference_avg_sharess    1    0.7610 7181.9 -87.937
    ## - n_tokens_title                1    1.1221 7182.3 -87.568
    ## - kw_max_min                    1    1.2187 7182.4 -87.469
    ## - title_sentiment_polarity      1    1.3938 7182.6 -87.290
    ## - abs_title_subjectivity        1    1.4527 7182.6 -87.230
    ## - global_rate_positive_words    1    1.6077 7182.8 -87.071
    ## <none>                                      7181.2 -86.716
    ## - kw_avg_min                    1    2.8943 7184.1 -85.755
    ## - kw_min_min                    1    3.2328 7184.4 -85.409
    ## - num_videos                    1    3.7410 7184.9 -84.890
    ## - num_imgs                      1    3.8678 7185.0 -84.760
    ## - kw_max_avg                    1    4.1901 7185.4 -84.431
    ## - LDA_00                        1    4.4279 7185.6 -84.187
    ## - kw_avg_max                    1    8.4680 7189.6 -80.058
    ## - max_negative_polarity         1   10.6285 7191.8 -77.851
    ## - n_tokens_content              1   12.2784 7193.4 -76.166
    ## - kw_avg_avg                    1   14.4797 7195.6 -73.918
    ## - num_self_hrefs                1   14.4849 7195.6 -73.913
    ## - avg_negative_polarity         1   16.0503 7197.2 -72.315
    ## - min_negative_polarity         1   19.8560 7201.0 -68.432
    ## - num_hrefs                     1   22.7704 7203.9 -65.459
    ## 
    ## Step:  AIC=-88.71
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + kw_min_min + kw_max_min + kw_avg_min + kw_min_max + 
    ##     kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_positive_words + global_rate_negative_words + 
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - n_unique_tokens               1    0.0127 7181.2 -90.694
    ## - global_sentiment_polarity     1    0.0176 7181.2 -90.688
    ## - LDA_03                        1    0.0474 7181.2 -90.658
    ## - global_rate_negative_words    1    0.0512 7181.2 -90.654
    ## - n_non_stop_unique_tokens      1    0.0905 7181.3 -90.614
    ## - min_positive_polarity         1    0.0984 7181.3 -90.606
    ## - global_subjectivity           1    0.1365 7181.3 -90.567
    ## - num_keywords                  1    0.1621 7181.3 -90.541
    ## - avg_positive_polarity         1    0.1902 7181.4 -90.512
    ## - abs_title_sentiment_polarity  1    0.2165 7181.4 -90.485
    ## - kw_min_max                    1    0.2372 7181.4 -90.464
    ## - rate_positive_words           1    0.2766 7181.4 -90.424
    ## - self_reference_min_shares     1    0.3501 7181.5 -90.348
    ## - n_non_stop_words              1    0.4790 7181.6 -90.217
    ## - self_reference_max_shares     1    0.4802 7181.7 -90.215
    ## - average_token_length          1    0.4806 7181.7 -90.215
    ## - kw_min_avg                    1    0.7052 7181.9 -89.985
    ## - kw_max_max                    1    0.7062 7181.9 -89.984
    ## - self_reference_avg_sharess    1    0.7666 7181.9 -89.922
    ## - n_tokens_title                1    1.1245 7182.3 -89.556
    ## - kw_max_min                    1    1.2180 7182.4 -89.461
    ## - title_sentiment_polarity      1    1.4027 7182.6 -89.272
    ## - abs_title_subjectivity        1    1.4476 7182.6 -89.226
    ## - global_rate_positive_words    1    1.6574 7182.8 -89.011
    ## <none>                                      7181.2 -88.707
    ## - kw_avg_min                    1    2.8925 7184.1 -87.748
    ## - kw_min_min                    1    3.2289 7184.4 -87.404
    ## - num_videos                    1    3.7328 7184.9 -86.889
    ## - num_imgs                      1    3.9105 7185.1 -86.707
    ## - kw_max_avg                    1    4.1966 7185.4 -86.415
    ## - LDA_00                        1    4.4370 7185.6 -86.169
    ## - kw_avg_max                    1    8.4600 7189.6 -82.057
    ## - max_negative_polarity         1   10.6203 7191.8 -79.850
    ## - n_tokens_content              1   12.2762 7193.4 -78.159
    ## - kw_avg_avg                    1   14.4872 7195.7 -75.902
    ## - num_self_hrefs                1   14.5009 7195.7 -75.888
    ## - avg_negative_polarity         1   16.0419 7197.2 -74.315
    ## - min_negative_polarity         1   19.9014 7201.1 -70.376
    ## - num_hrefs                     1   22.7660 7203.9 -67.455
    ## 
    ## Step:  AIC=-90.69
    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + num_keywords + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_positive_words + global_rate_negative_words + 
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - global_sentiment_polarity     1    0.0170 7181.2 -92.676
    ## - LDA_03                        1    0.0457 7181.2 -92.647
    ## - global_rate_negative_words    1    0.0487 7181.2 -92.644
    ## - min_positive_polarity         1    0.1242 7181.3 -92.567
    ## - global_subjectivity           1    0.1374 7181.3 -92.553
    ## - num_keywords                  1    0.1585 7181.3 -92.531
    ## - avg_positive_polarity         1    0.1822 7181.4 -92.507
    ## - abs_title_sentiment_polarity  1    0.2171 7181.4 -92.472
    ## - kw_min_max                    1    0.2397 7181.4 -92.448
    ## - rate_positive_words           1    0.2787 7181.5 -92.408
    ## - self_reference_min_shares     1    0.3490 7181.5 -92.337
    ## - self_reference_max_shares     1    0.4785 7181.7 -92.204
    ## - n_non_stop_words              1    0.4903 7181.7 -92.192
    ## - average_token_length          1    0.5330 7181.7 -92.148
    ## - n_non_stop_unique_tokens      1    0.6675 7181.9 -92.011
    ## - kw_max_max                    1    0.7003 7181.9 -91.977
    ## - kw_min_avg                    1    0.7134 7181.9 -91.964
    ## - self_reference_avg_sharess    1    0.7653 7181.9 -91.911
    ## - n_tokens_title                1    1.1149 7182.3 -91.553
    ## - kw_max_min                    1    1.2224 7182.4 -91.443
    ## - title_sentiment_polarity      1    1.4038 7182.6 -91.258
    ## - abs_title_subjectivity        1    1.4447 7182.6 -91.216
    ## - global_rate_positive_words    1    1.6671 7182.9 -90.988
    ## <none>                                      7181.2 -90.694
    ## - kw_avg_min                    1    2.8977 7184.1 -89.730
    ## - kw_min_min                    1    3.2249 7184.4 -89.395
    ## - num_videos                    1    3.7212 7184.9 -88.888
    ## - num_imgs                      1    4.0657 7185.2 -88.536
    ## - kw_max_avg                    1    4.2212 7185.4 -88.377
    ## - LDA_00                        1    4.4362 7185.6 -88.157
    ## - kw_avg_max                    1    8.4490 7189.6 -84.056
    ## - max_negative_polarity         1   10.8361 7192.0 -81.617
    ## - num_self_hrefs                1   14.5158 7195.7 -77.860
    ## - kw_avg_avg                    1   14.5865 7195.8 -77.787
    ## - n_tokens_content              1   15.8025 7197.0 -76.546
    ## - avg_negative_polarity         1   16.0938 7197.3 -76.249
    ## - min_negative_polarity         1   20.0152 7201.2 -72.248
    ## - num_hrefs                     1   22.8307 7204.0 -69.376
    ## 
    ## Step:  AIC=-92.68
    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + num_keywords + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_03 + global_subjectivity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_sentiment_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - global_rate_negative_words    1    0.0384 7181.2 -94.637
    ## - LDA_03                        1    0.0452 7181.2 -94.630
    ## - min_positive_polarity         1    0.1399 7181.3 -94.533
    ## - num_keywords                  1    0.1573 7181.4 -94.515
    ## - global_subjectivity           1    0.1659 7181.4 -94.507
    ## - abs_title_sentiment_polarity  1    0.2175 7181.4 -94.454
    ## - avg_positive_polarity         1    0.2277 7181.4 -94.443
    ## - kw_min_max                    1    0.2390 7181.4 -94.432
    ## - rate_positive_words           1    0.2618 7181.5 -94.408
    ## - self_reference_min_shares     1    0.3473 7181.5 -94.321
    ## - n_non_stop_words              1    0.4754 7181.7 -94.190
    ## - self_reference_max_shares     1    0.4783 7181.7 -94.187
    ## - average_token_length          1    0.5538 7181.8 -94.110
    ## - n_non_stop_unique_tokens      1    0.6840 7181.9 -93.977
    ## - kw_max_max                    1    0.6967 7181.9 -93.964
    ## - kw_min_avg                    1    0.7166 7181.9 -93.943
    ## - self_reference_avg_sharess    1    0.7635 7182.0 -93.895
    ## - n_tokens_title                1    1.1095 7182.3 -93.541
    ## - kw_max_min                    1    1.2202 7182.4 -93.428
    ## - title_sentiment_polarity      1    1.4303 7182.6 -93.213
    ## - abs_title_subjectivity        1    1.4388 7182.6 -93.205
    ## - global_rate_positive_words    1    1.8637 7183.1 -92.770
    ## <none>                                      7181.2 -92.676
    ## - kw_avg_min                    1    2.8948 7184.1 -91.716
    ## - kw_min_min                    1    3.2185 7184.4 -91.385
    ## - num_videos                    1    3.7334 7184.9 -90.858
    ## - num_imgs                      1    4.0488 7185.2 -90.536
    ## - kw_max_avg                    1    4.2176 7185.4 -90.363
    ## - LDA_00                        1    4.4357 7185.6 -90.140
    ## - kw_avg_max                    1    8.4345 7189.6 -86.053
    ## - max_negative_polarity         1   10.9646 7192.2 -83.468
    ## - num_self_hrefs                1   14.5067 7195.7 -79.852
    ## - kw_avg_avg                    1   14.5739 7195.8 -79.783
    ## - n_tokens_content              1   15.8336 7197.0 -78.497
    ## - avg_negative_polarity         1   17.0479 7198.2 -77.258
    ## - min_negative_polarity         1   20.0226 7201.2 -74.223
    ## - num_hrefs                     1   22.9831 7204.2 -71.203
    ## 
    ## Step:  AIC=-94.64
    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + num_keywords + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_03 + global_subjectivity + global_rate_positive_words + 
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - LDA_03                        1    0.0461 7181.3 -96.590
    ## - min_positive_polarity         1    0.1532 7181.4 -96.480
    ## - num_keywords                  1    0.1582 7181.4 -96.475
    ## - global_subjectivity           1    0.1766 7181.4 -96.456
    ## - abs_title_sentiment_polarity  1    0.2138 7181.5 -96.418
    ## - avg_positive_polarity         1    0.2182 7181.5 -96.414
    ## - kw_min_max                    1    0.2389 7181.5 -96.393
    ## - self_reference_min_shares     1    0.3462 7181.6 -96.283
    ## - self_reference_max_shares     1    0.4794 7181.7 -96.147
    ## - average_token_length          1    0.5642 7181.8 -96.060
    ## - n_non_stop_unique_tokens      1    0.6775 7181.9 -95.944
    ## - kw_max_max                    1    0.7049 7181.9 -95.916
    ## - kw_min_avg                    1    0.7187 7182.0 -95.902
    ## - self_reference_avg_sharess    1    0.7641 7182.0 -95.855
    ## - n_non_stop_words              1    0.7849 7182.0 -95.834
    ## - n_tokens_title                1    1.1003 7182.3 -95.511
    ## - kw_max_min                    1    1.2145 7182.5 -95.395
    ## - title_sentiment_polarity      1    1.4066 7182.6 -95.198
    ## - abs_title_subjectivity        1    1.4570 7182.7 -95.147
    ## <none>                                      7181.2 -94.637
    ## - rate_positive_words           1    2.3334 7183.6 -94.250
    ## - global_rate_positive_words    1    2.8668 7184.1 -93.705
    ## - kw_avg_min                    1    2.8890 7184.1 -93.682
    ## - kw_min_min                    1    3.2203 7184.5 -93.343
    ## - num_videos                    1    3.7412 7185.0 -92.811
    ## - num_imgs                      1    4.0607 7185.3 -92.484
    ## - kw_max_avg                    1    4.2166 7185.5 -92.325
    ## - LDA_00                        1    4.4193 7185.7 -92.118
    ## - kw_avg_max                    1    8.4642 7189.7 -87.984
    ## - max_negative_polarity         1   11.0858 7192.3 -85.305
    ## - num_self_hrefs                1   14.5367 7195.8 -81.782
    ## - kw_avg_avg                    1   14.5845 7195.8 -81.733
    ## - n_tokens_content              1   15.8075 7197.0 -80.484
    ## - avg_negative_polarity         1   17.0116 7198.3 -79.256
    ## - min_negative_polarity         1   20.0157 7201.3 -76.190
    ## - num_hrefs                     1   22.9449 7204.2 -73.203
    ## 
    ## Step:  AIC=-96.59
    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + num_keywords + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + global_subjectivity + global_rate_positive_words + 
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - min_positive_polarity         1    0.1526 7181.4 -98.434
    ## - num_keywords                  1    0.1647 7181.4 -98.421
    ## - global_subjectivity           1    0.1786 7181.5 -98.407
    ## - abs_title_sentiment_polarity  1    0.2103 7181.5 -98.375
    ## - avg_positive_polarity         1    0.2126 7181.5 -98.372
    ## - kw_min_max                    1    0.2375 7181.5 -98.347
    ## - self_reference_min_shares     1    0.3457 7181.6 -98.236
    ## - self_reference_max_shares     1    0.4828 7181.8 -98.096
    ## - average_token_length          1    0.5740 7181.9 -98.003
    ## - n_non_stop_unique_tokens      1    0.6668 7182.0 -97.908
    ## - kw_max_max                    1    0.7024 7182.0 -97.871
    ## - kw_min_avg                    1    0.7503 7182.0 -97.822
    ## - self_reference_avg_sharess    1    0.7649 7182.0 -97.807
    ## - n_non_stop_words              1    0.7768 7182.1 -97.795
    ## - n_tokens_title                1    1.0880 7182.4 -97.477
    ## - kw_max_min                    1    1.1903 7182.5 -97.372
    ## - title_sentiment_polarity      1    1.4031 7182.7 -97.155
    ## - abs_title_subjectivity        1    1.4561 7182.7 -97.100
    ## <none>                                      7181.3 -96.590
    ## - rate_positive_words           1    2.3446 7183.6 -96.192
    ## - global_rate_positive_words    1    2.8573 7184.1 -95.668
    ## - kw_avg_min                    1    2.8690 7184.2 -95.656
    ## - kw_min_min                    1    3.2271 7184.5 -95.289
    ## - num_videos                    1    3.8615 7185.1 -94.641
    ## - num_imgs                      1    4.0526 7185.3 -94.445
    ## - kw_max_avg                    1    4.1755 7185.5 -94.320
    ## - LDA_00                        1    4.3793 7185.7 -94.111
    ## - kw_avg_max                    1    8.4312 7189.7 -89.970
    ## - max_negative_polarity         1   11.0916 7192.4 -87.253
    ## - kw_avg_avg                    1   14.6216 7195.9 -83.648
    ## - num_self_hrefs                1   14.6266 7195.9 -83.643
    ## - n_tokens_content              1   15.7784 7197.1 -82.467
    ## - avg_negative_polarity         1   17.0468 7198.3 -81.173
    ## - min_negative_polarity         1   20.0503 7201.3 -78.108
    ## - num_hrefs                     1   23.1525 7204.4 -74.944
    ## 
    ## Step:  AIC=-98.43
    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + num_keywords + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + global_subjectivity + global_rate_positive_words + 
    ##     rate_positive_words + avg_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS      AIC
    ## - num_keywords                  1    0.1551 7181.6 -100.275
    ## - global_subjectivity           1    0.1792 7181.6 -100.250
    ## - abs_title_sentiment_polarity  1    0.2002 7181.6 -100.229
    ## - kw_min_max                    1    0.2302 7181.7 -100.198
    ## - self_reference_min_shares     1    0.3436 7181.8 -100.082
    ## - self_reference_max_shares     1    0.4784 7181.9  -99.944
    ## - avg_positive_polarity         1    0.5035 7181.9  -99.919
    ## - average_token_length          1    0.5508 7182.0  -99.870
    ## - kw_max_max                    1    0.6938 7182.1  -99.724
    ## - n_non_stop_unique_tokens      1    0.7316 7182.2  -99.685
    ## - n_non_stop_words              1    0.7524 7182.2  -99.664
    ## - self_reference_avg_sharess    1    0.7612 7182.2  -99.655
    ## - kw_min_avg                    1    0.7671 7182.2  -99.649
    ## - n_tokens_title                1    1.0851 7182.5  -99.324
    ## - kw_max_min                    1    1.2015 7182.6  -99.205
    ## - title_sentiment_polarity      1    1.3866 7182.8  -99.015
    ## - abs_title_subjectivity        1    1.4223 7182.9  -98.979
    ## <none>                                      7181.4  -98.434
    ## - rate_positive_words           1    2.2918 7183.7  -98.090
    ## - global_rate_positive_words    1    2.7049 7184.1  -97.667
    ## - kw_avg_min                    1    2.8899 7184.3  -97.478
    ## - kw_min_min                    1    3.2319 7184.7  -97.128
    ## - num_videos                    1    3.8256 7185.3  -96.521
    ## - num_imgs                      1    4.1211 7185.6  -96.219
    ## - kw_max_avg                    1    4.2054 7185.6  -96.133
    ## - LDA_00                        1    4.5029 7185.9  -95.829
    ## - kw_avg_max                    1    8.4097 7189.8  -91.836
    ## - max_negative_polarity         1   11.3375 7192.8  -88.845
    ## - num_self_hrefs                1   14.6865 7196.1  -85.426
    ## - kw_avg_avg                    1   14.7267 7196.2  -85.385
    ## - n_tokens_content              1   16.6286 7198.1  -83.444
    ## - avg_negative_polarity         1   17.2422 7198.7  -82.817
    ## - min_negative_polarity         1   19.9408 7201.4  -80.064
    ## - num_hrefs                     1   23.4162 7204.9  -76.520
    ## 
    ## Step:  AIC=-100.28
    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + kw_min_min + kw_max_min + 
    ##     kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_00 + global_subjectivity + 
    ##     global_rate_positive_words + rate_positive_words + avg_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS      AIC
    ## - global_subjectivity           1    0.1725 7181.8 -102.099
    ## - kw_min_max                    1    0.2044 7181.8 -102.066
    ## - abs_title_sentiment_polarity  1    0.2057 7181.8 -102.065
    ## - self_reference_min_shares     1    0.3488 7181.9 -101.918
    ## - self_reference_max_shares     1    0.4861 7182.1 -101.778
    ## - avg_positive_polarity         1    0.5258 7182.1 -101.737
    ## - average_token_length          1    0.5880 7182.2 -101.674
    ## - kw_max_max                    1    0.6297 7182.2 -101.631
    ## - kw_min_avg                    1    0.6941 7182.3 -101.565
    ## - n_non_stop_unique_tokens      1    0.7051 7182.3 -101.554
    ## - self_reference_avg_sharess    1    0.7694 7182.4 -101.488
    ## - n_non_stop_words              1    0.8055 7182.4 -101.451
    ## - n_tokens_title                1    1.0209 7182.6 -101.231
    ## - kw_max_min                    1    1.1393 7182.7 -101.110
    ## - title_sentiment_polarity      1    1.3980 7183.0 -100.845
    ## - abs_title_subjectivity        1    1.4386 7183.0 -100.804
    ## <none>                                      7181.6 -100.275
    ## - rate_positive_words           1    2.3188 7183.9  -99.904
    ## - global_rate_positive_words    1    2.7386 7184.3  -99.474
    ## - kw_avg_min                    1    2.7880 7184.4  -99.424
    ## - kw_min_min                    1    3.2489 7184.8  -98.952
    ## - num_videos                    1    3.7957 7185.4  -98.393
    ## - num_imgs                      1    4.0433 7185.6  -98.140
    ## - kw_max_avg                    1    4.2412 7185.8  -97.938
    ## - LDA_00                        1    4.4849 7186.1  -97.689
    ## - kw_avg_max                    1    8.5285 7190.1  -93.556
    ## - max_negative_polarity         1   11.3240 7192.9  -90.701
    ## - kw_avg_avg                    1   14.5972 7196.2  -87.359
    ## - num_self_hrefs                1   14.9296 7196.5  -87.020
    ## - n_tokens_content              1   16.5178 7198.1  -85.398
    ## - avg_negative_polarity         1   17.2156 7198.8  -84.686
    ## - min_negative_polarity         1   19.9402 7201.5  -81.907
    ## - num_hrefs                     1   23.2666 7204.9  -78.514
    ## 
    ## Step:  AIC=-102.1
    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + kw_min_min + kw_max_min + 
    ##     kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_00 + global_rate_positive_words + 
    ##     rate_positive_words + avg_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS      AIC
    ## - abs_title_sentiment_polarity  1    0.1918 7182.0 -103.902
    ## - kw_min_max                    1    0.2080 7182.0 -103.886
    ## - self_reference_min_shares     1    0.3602 7182.1 -103.730
    ## - avg_positive_polarity         1    0.4002 7182.2 -103.689
    ## - self_reference_max_shares     1    0.4945 7182.3 -103.593
    ## - average_token_length          1    0.6066 7182.4 -103.478
    ## - kw_max_max                    1    0.6231 7182.4 -103.461
    ## - kw_min_avg                    1    0.7068 7182.5 -103.376
    ## - n_non_stop_unique_tokens      1    0.7171 7182.5 -103.365
    ## - self_reference_avg_sharess    1    0.7862 7182.6 -103.294
    ## - n_non_stop_words              1    0.9326 7182.7 -103.145
    ## - n_tokens_title                1    0.9963 7182.8 -103.080
    ## - kw_max_min                    1    1.1456 7182.9 -102.927
    ## - abs_title_subjectivity        1    1.3905 7183.2 -102.676
    ## - title_sentiment_polarity      1    1.3971 7183.2 -102.670
    ## <none>                                      7181.8 -102.099
    ## - rate_positive_words           1    2.1910 7184.0 -101.858
    ## - global_rate_positive_words    1    2.5902 7184.4 -101.450
    ## - kw_avg_min                    1    2.7972 7184.6 -101.238
    ## - kw_min_min                    1    3.2564 7185.0 -100.768
    ## - num_videos                    1    3.7833 7185.5 -100.230
    ## - num_imgs                      1    4.0817 7185.8  -99.925
    ## - kw_max_avg                    1    4.2902 7186.1  -99.711
    ## - LDA_00                        1    4.4204 7186.2  -99.578
    ## - kw_avg_max                    1    8.4518 7190.2  -95.459
    ## - max_negative_polarity         1   11.7102 7193.5  -92.130
    ## - kw_avg_avg                    1   14.6629 7196.4  -89.116
    ## - num_self_hrefs                1   15.0226 7196.8  -88.748
    ## - n_tokens_content              1   16.4171 7198.2  -87.325
    ## - avg_negative_polarity         1   18.1815 7199.9  -85.525
    ## - min_negative_polarity         1   19.9559 7201.7  -83.715
    ## - num_hrefs                     1   23.5857 7205.4  -80.013
    ## 
    ## Step:  AIC=-103.9
    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + kw_min_min + kw_max_min + 
    ##     kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_00 + global_rate_positive_words + 
    ##     rate_positive_words + avg_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_sentiment_polarity + 
    ##     abs_title_subjectivity
    ## 
    ##                              Df Sum of Sq    RSS      AIC
    ## - kw_min_max                  1    0.2023 7182.2 -105.695
    ## - self_reference_min_shares   1    0.3589 7182.3 -105.535
    ## - avg_positive_polarity       1    0.4621 7182.4 -105.430
    ## - self_reference_max_shares   1    0.4900 7182.4 -105.401
    ## - average_token_length        1    0.5921 7182.5 -105.297
    ## - kw_max_max                  1    0.6166 7182.6 -105.272
    ## - n_non_stop_unique_tokens    1    0.7121 7182.7 -105.174
    ## - kw_min_avg                  1    0.7213 7182.7 -105.165
    ## - self_reference_avg_sharess  1    0.7825 7182.7 -105.102
    ## - n_non_stop_words            1    0.9349 7182.9 -104.946
    ## - n_tokens_title              1    1.0017 7183.0 -104.878
    ## - kw_max_min                  1    1.1335 7183.1 -104.743
    ## - abs_title_subjectivity      1    1.1987 7183.2 -104.676
    ## - title_sentiment_polarity    1    1.2666 7183.2 -104.607
    ## <none>                                    7182.0 -103.902
    ## - rate_positive_words         1    2.1412 7184.1 -103.713
    ## - global_rate_positive_words  1    2.6505 7184.6 -103.192
    ## - kw_avg_min                  1    2.7684 7184.7 -103.071
    ## - kw_min_min                  1    3.2269 7185.2 -102.603
    ## - num_videos                  1    3.7706 7185.7 -102.047
    ## - num_imgs                    1    4.1079 7186.1 -101.702
    ## - kw_max_avg                  1    4.3246 7186.3 -101.480
    ## - LDA_00                      1    4.3970 7186.4 -101.406
    ## - kw_avg_max                  1    8.4681 7190.4  -97.246
    ## - max_negative_polarity       1   11.5980 7193.6  -94.049
    ## - kw_avg_avg                  1   14.6851 7196.6  -90.897
    ## - num_self_hrefs              1   14.9142 7196.9  -90.663
    ## - n_tokens_content            1   16.3906 7198.3  -89.157
    ## - avg_negative_polarity       1   18.0244 7200.0  -87.489
    ## - min_negative_polarity       1   19.9049 7201.9  -85.571
    ## - num_hrefs                   1   23.4388 7205.4  -81.967
    ## 
    ## Step:  AIC=-105.7
    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + kw_min_min + kw_max_min + 
    ##     kw_avg_min + kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_00 + global_rate_positive_words + 
    ##     rate_positive_words + avg_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_sentiment_polarity + 
    ##     abs_title_subjectivity
    ## 
    ##                              Df Sum of Sq    RSS      AIC
    ## - self_reference_min_shares   1    0.3766 7182.5 -107.310
    ## - avg_positive_polarity       1    0.4677 7182.6 -107.217
    ## - self_reference_max_shares   1    0.4894 7182.6 -107.195
    ## - average_token_length        1    0.5842 7182.7 -107.098
    ## - kw_max_max                  1    0.6812 7182.8 -106.999
    ## - n_non_stop_unique_tokens    1    0.7189 7182.9 -106.960
    ## - self_reference_avg_sharess  1    0.7903 7182.9 -106.887
    ## - kw_min_avg                  1    0.8940 7183.1 -106.781
    ## - n_non_stop_words            1    0.9325 7183.1 -106.742
    ## - n_tokens_title              1    1.0314 7183.2 -106.641
    ## - kw_max_min                  1    1.1370 7183.3 -106.533
    ## - abs_title_subjectivity      1    1.1950 7183.4 -106.473
    ## - title_sentiment_polarity    1    1.2741 7183.4 -106.392
    ## <none>                                    7182.2 -105.695
    ## - rate_positive_words         1    2.1471 7184.3 -105.500
    ## - global_rate_positive_words  1    2.6629 7184.8 -104.972
    ## - kw_avg_min                  1    2.7598 7184.9 -104.873
    ## - kw_min_min                  1    3.1950 7185.4 -104.428
    ## - num_videos                  1    3.7890 7185.9 -103.821
    ## - num_imgs                    1    4.1066 7186.3 -103.496
    ## - LDA_00                      1    4.3720 7186.5 -103.225
    ## - kw_max_avg                  1    4.3757 7186.5 -103.221
    ## - kw_avg_max                  1    9.8759 7192.0  -97.601
    ## - max_negative_polarity       1   11.5869 7193.7  -95.854
    ## - kw_avg_avg                  1   14.8326 7197.0  -92.540
    ## - num_self_hrefs              1   14.8911 7197.1  -92.480
    ## - n_tokens_content            1   16.3853 7198.5  -90.955
    ## - avg_negative_polarity       1   18.0191 7200.2  -89.288
    ## - min_negative_polarity       1   19.9303 7202.1  -87.339
    ## - num_hrefs                   1   23.3773 7205.5  -83.824
    ## 
    ## Step:  AIC=-107.31
    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + kw_min_min + kw_max_min + 
    ##     kw_avg_min + kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + global_rate_positive_words + rate_positive_words + 
    ##     avg_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_sentiment_polarity + abs_title_subjectivity
    ## 
    ##                              Df Sum of Sq    RSS      AIC
    ## - self_reference_max_shares   1    0.1219 7182.7 -109.186
    ## - avg_positive_polarity       1    0.4687 7183.0 -108.831
    ## - average_token_length        1    0.5843 7183.1 -108.713
    ## - self_reference_avg_sharess  1    0.6733 7183.2 -108.622
    ## - kw_max_max                  1    0.6855 7183.2 -108.609
    ## - n_non_stop_unique_tokens    1    0.7191 7183.3 -108.575
    ## - kw_min_avg                  1    0.8947 7183.4 -108.395
    ## - n_non_stop_words            1    0.9420 7183.5 -108.347
    ## - n_tokens_title              1    1.0461 7183.6 -108.240
    ## - kw_max_min                  1    1.1029 7183.6 -108.182
    ## - abs_title_subjectivity      1    1.1898 7183.7 -108.094
    ## - title_sentiment_polarity    1    1.2697 7183.8 -108.012
    ## <none>                                    7182.5 -107.310
    ## - rate_positive_words         1    2.1813 7184.7 -107.080
    ## - global_rate_positive_words  1    2.6820 7185.2 -106.568
    ## - kw_avg_min                  1    2.7044 7185.2 -106.545
    ## - kw_min_min                  1    3.1799 7185.7 -106.059
    ## - num_videos                  1    3.8159 7186.4 -105.409
    ## - num_imgs                    1    4.0682 7186.6 -105.151
    ## - LDA_00                      1    4.3179 7186.9 -104.895
    ## - kw_max_avg                  1    4.3636 7186.9 -104.849
    ## - kw_avg_max                  1    9.8821 7192.4  -99.210
    ## - max_negative_polarity       1   11.6101 7194.1  -97.446
    ## - kw_avg_avg                  1   14.8337 7197.4  -94.155
    ## - num_self_hrefs              1   15.4051 7197.9  -93.571
    ## - n_tokens_content            1   16.3908 7198.9  -92.566
    ## - avg_negative_polarity       1   18.0842 7200.6  -90.838
    ## - min_negative_polarity       1   20.0010 7202.5  -88.882
    ## - num_hrefs                   1   23.3745 7205.9  -85.443
    ## 
    ## Step:  AIC=-109.19
    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + kw_min_min + kw_max_min + 
    ##     kw_avg_min + kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_avg_sharess + LDA_00 + global_rate_positive_words + 
    ##     rate_positive_words + avg_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_sentiment_polarity + 
    ##     abs_title_subjectivity
    ## 
    ##                              Df Sum of Sq    RSS      AIC
    ## - avg_positive_polarity       1    0.4744 7183.1 -110.700
    ## - average_token_length        1    0.5905 7183.2 -110.582
    ## - kw_max_max                  1    0.6894 7183.3 -110.481
    ## - n_non_stop_unique_tokens    1    0.7228 7183.4 -110.446
    ## - kw_min_avg                  1    0.8947 7183.6 -110.271
    ## - n_non_stop_words            1    0.9489 7183.6 -110.215
    ## - n_tokens_title              1    1.0561 7183.7 -110.106
    ## - kw_max_min                  1    1.1211 7183.8 -110.039
    ## - self_reference_avg_sharess  1    1.1610 7183.8 -109.998
    ## - abs_title_subjectivity      1    1.1914 7183.8 -109.967
    ## - title_sentiment_polarity    1    1.2760 7183.9 -109.881
    ## <none>                                    7182.7 -109.186
    ## - rate_positive_words         1    2.1579 7184.8 -108.979
    ## - global_rate_positive_words  1    2.7006 7185.4 -108.424
    ## - kw_avg_min                  1    2.7331 7185.4 -108.391
    ## - kw_min_min                  1    3.1828 7185.8 -107.931
    ## - num_videos                  1    3.8286 7186.5 -107.271
    ## - num_imgs                    1    4.0345 7186.7 -107.060
    ## - LDA_00                      1    4.2897 7186.9 -106.800
    ## - kw_max_avg                  1    4.3734 7187.0 -106.714
    ## - kw_avg_max                  1   10.0020 7192.7 -100.963
    ## - max_negative_polarity       1   11.5610 7194.2  -99.371
    ## - kw_avg_avg                  1   14.8496 7197.5  -96.014
    ## - num_self_hrefs              1   15.9631 7198.6  -94.878
    ## - n_tokens_content            1   16.3684 7199.0  -94.464
    ## - avg_negative_polarity       1   18.0264 7200.7  -92.772
    ## - min_negative_polarity       1   19.9695 7202.6  -90.790
    ## - num_hrefs                   1   23.3927 7206.1  -87.300
    ## 
    ## Step:  AIC=-110.7
    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + kw_min_min + kw_max_min + 
    ##     kw_avg_min + kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_avg_sharess + LDA_00 + global_rate_positive_words + 
    ##     rate_positive_words + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_sentiment_polarity + abs_title_subjectivity
    ## 
    ##                              Df Sum of Sq    RSS      AIC
    ## - average_token_length        1    0.5021 7183.6 -112.187
    ## - kw_max_max                  1    0.6957 7183.8 -111.989
    ## - n_non_stop_words            1    0.7440 7183.9 -111.940
    ## - n_non_stop_unique_tokens    1    0.8252 7184.0 -111.857
    ## - kw_min_avg                  1    0.8603 7184.0 -111.821
    ## - n_tokens_title              1    1.0374 7184.2 -111.640
    ## - kw_max_min                  1    1.1197 7184.3 -111.555
    ## - title_sentiment_polarity    1    1.1472 7184.3 -111.527
    ## - self_reference_avg_sharess  1    1.1702 7184.3 -111.504
    ## - abs_title_subjectivity      1    1.2715 7184.4 -111.400
    ## <none>                                    7183.1 -110.700
    ## - rate_positive_words         1    2.1464 7185.3 -110.506
    ## - kw_avg_min                  1    2.7397 7185.9 -109.899
    ## - global_rate_positive_words  1    2.9460 7186.1 -109.688
    ## - kw_min_min                  1    3.2303 7186.4 -109.398
    ## - num_videos                  1    3.8082 7186.9 -108.807
    ## - num_imgs                    1    4.0373 7187.2 -108.573
    ## - LDA_00                      1    4.2657 7187.4 -108.339
    ## - kw_max_avg                  1    4.3395 7187.5 -108.264
    ## - kw_avg_max                  1    9.8647 7193.0 -102.619
    ## - max_negative_polarity       1   11.4764 7194.6 -100.973
    ## - kw_avg_avg                  1   14.7315 7197.9  -97.650
    ## - num_self_hrefs              1   15.7087 7198.8  -96.653
    ## - n_tokens_content            1   16.1511 7199.3  -96.202
    ## - avg_negative_polarity       1   17.9578 7201.1  -94.358
    ## - min_negative_polarity       1   20.0821 7203.2  -92.192
    ## - num_hrefs                   1   23.0108 7206.1  -89.205
    ## 
    ## Step:  AIC=-112.19
    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + kw_min_min + kw_max_min + kw_avg_min + kw_max_max + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_avg_sharess + 
    ##     LDA_00 + global_rate_positive_words + rate_positive_words + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity
    ## 
    ##                              Df Sum of Sq    RSS      AIC
    ## - n_non_stop_words            1    0.2880 7183.9 -113.892
    ## - kw_max_max                  1    0.7322 7184.4 -113.438
    ## - kw_min_avg                  1    0.7896 7184.4 -113.380
    ## - n_non_stop_unique_tokens    1    0.9407 7184.6 -113.225
    ## - kw_max_min                  1    1.0665 7184.7 -113.096
    ## - self_reference_avg_sharess  1    1.1231 7184.8 -113.039
    ## - title_sentiment_polarity    1    1.1845 7184.8 -112.976
    ## - n_tokens_title              1    1.1931 7184.8 -112.967
    ## - abs_title_subjectivity      1    1.2620 7184.9 -112.897
    ## <none>                                    7183.6 -112.187
    ## - rate_positive_words         1    2.1551 7185.8 -111.984
    ## - kw_avg_min                  1    2.6729 7186.3 -111.454
    ## - global_rate_positive_words  1    2.7913 7186.4 -111.333
    ## - kw_min_min                  1    3.2647 7186.9 -110.849
    ## - num_videos                  1    3.8590 7187.5 -110.242
    ## - num_imgs                    1    4.0713 7187.7 -110.025
    ## - LDA_00                      1    4.1253 7187.8 -109.970
    ## - kw_max_avg                  1    4.2824 7187.9 -109.809
    ## - kw_avg_max                  1    9.9584 7193.6 -104.011
    ## - max_negative_polarity       1   11.3628 7195.0 -102.577
    ## - kw_avg_avg                  1   14.5347 7198.2  -99.339
    ## - num_self_hrefs              1   15.3601 7199.0  -98.496
    ## - n_tokens_content            1   16.5937 7200.2  -97.238
    ## - avg_negative_polarity       1   17.9329 7201.6  -95.872
    ## - min_negative_polarity       1   19.9939 7203.6  -93.770
    ## - num_hrefs                   1   22.5668 7206.2  -91.146
    ## 
    ## Step:  AIC=-113.89
    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + num_imgs + num_videos + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_max_max + kw_avg_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_avg_sharess + LDA_00 + 
    ##     global_rate_positive_words + rate_positive_words + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_sentiment_polarity + 
    ##     abs_title_subjectivity
    ## 
    ##                              Df Sum of Sq    RSS      AIC
    ## - n_non_stop_unique_tokens    1    0.6577 7184.6 -115.220
    ## - kw_max_max                  1    0.7474 7184.7 -115.128
    ## - kw_min_avg                  1    0.7532 7184.7 -115.122
    ## - kw_max_min                  1    1.0359 7185.0 -114.833
    ## - self_reference_avg_sharess  1    1.1412 7185.1 -114.726
    ## - n_tokens_title              1    1.1523 7185.1 -114.714
    ## - title_sentiment_polarity    1    1.1721 7185.1 -114.694
    ## - abs_title_subjectivity      1    1.2702 7185.2 -114.594
    ## - rate_positive_words         1    1.8705 7185.8 -113.980
    ## <none>                                    7183.9 -113.892
    ## - kw_avg_min                  1    2.6100 7186.5 -113.224
    ## - global_rate_positive_words  1    2.9790 7186.9 -112.847
    ## - kw_min_min                  1    3.2591 7187.2 -112.561
    ## - num_videos                  1    3.8092 7187.7 -111.998
    ## - num_imgs                    1    3.8365 7187.8 -111.970
    ## - LDA_00                      1    4.0510 7188.0 -111.751
    ## - kw_max_avg                  1    4.1953 7188.1 -111.604
    ## - kw_avg_max                  1    9.8029 7193.7 -105.875
    ## - max_negative_polarity       1   11.5799 7195.5 -104.061
    ## - kw_avg_avg                  1   14.3163 7198.2 -101.268
    ## - num_self_hrefs              1   15.3975 7199.3 -100.164
    ## - avg_negative_polarity       1   18.2691 7202.2  -97.235
    ## - n_tokens_content            1   18.4192 7202.3  -97.082
    ## - min_negative_polarity       1   19.7744 7203.7  -95.700
    ## - num_hrefs                   1   22.5962 7206.5  -92.823
    ## 
    ## Step:  AIC=-115.22
    ## shares ~ n_tokens_title + n_tokens_content + num_hrefs + num_self_hrefs + 
    ##     num_imgs + num_videos + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + 
    ##     self_reference_avg_sharess + LDA_00 + global_rate_positive_words + 
    ##     rate_positive_words + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_sentiment_polarity + abs_title_subjectivity
    ## 
    ##                              Df Sum of Sq    RSS      AIC
    ## - kw_min_avg                  1    0.7313 7185.3 -116.472
    ## - kw_max_max                  1    0.7452 7185.3 -116.458
    ## - kw_max_min                  1    0.9820 7185.6 -116.216
    ## - n_tokens_title              1    1.1093 7185.7 -116.086
    ## - self_reference_avg_sharess  1    1.1376 7185.7 -116.057
    ## - abs_title_subjectivity      1    1.2520 7185.8 -115.940
    ## - title_sentiment_polarity    1    1.3247 7185.9 -115.866
    ## <none>                                    7184.6 -115.220
    ## - rate_positive_words         1    2.0317 7186.6 -115.143
    ## - kw_avg_min                  1    2.4985 7187.1 -114.666
    ## - kw_min_min                  1    3.2116 7187.8 -113.937
    ## - num_imgs                    1    3.2155 7187.8 -113.933
    ## - global_rate_positive_words  1    3.2181 7187.8 -113.930
    ## - num_videos                  1    3.7965 7188.4 -113.339
    ## - LDA_00                      1    4.0007 7188.6 -113.130
    ## - kw_max_avg                  1    4.1103 7188.7 -113.018
    ## - kw_avg_max                  1    9.5016 7194.1 -107.511
    ## - max_negative_polarity       1   11.6092 7196.2 -105.359
    ## - kw_avg_avg                  1   14.0433 7198.6 -102.875
    ## - num_self_hrefs              1   15.4998 7200.1 -101.389
    ## - avg_negative_polarity       1   18.1703 7202.8  -98.665
    ## - min_negative_polarity       1   19.9709 7204.6  -96.829
    ## - num_hrefs                   1   22.6058 7207.2  -94.143
    ## - n_tokens_content            1   25.6900 7210.3  -91.000
    ## 
    ## Step:  AIC=-116.47
    ## shares ~ n_tokens_title + n_tokens_content + num_hrefs + num_self_hrefs + 
    ##     num_imgs + num_videos + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_max_max + kw_avg_max + kw_max_avg + kw_avg_avg + self_reference_avg_sharess + 
    ##     LDA_00 + global_rate_positive_words + rate_positive_words + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity
    ## 
    ##                              Df Sum of Sq    RSS      AIC
    ## - kw_max_min                  1    0.7404 7186.1 -117.715
    ## - kw_max_max                  1    1.0959 7186.4 -117.352
    ## - n_tokens_title              1    1.1022 7186.4 -117.345
    ## - self_reference_avg_sharess  1    1.1443 7186.5 -117.302
    ## - abs_title_subjectivity      1    1.2259 7186.5 -117.219
    ## - title_sentiment_polarity    1    1.3092 7186.6 -117.134
    ## <none>                                    7185.3 -116.472
    ## - rate_positive_words         1    2.0117 7187.3 -116.416
    ## - kw_avg_min                  1    2.1618 7187.5 -116.262
    ## - global_rate_positive_words  1    3.3186 7188.6 -115.080
    ## - num_imgs                    1    3.3554 7188.7 -115.043
    ## - kw_min_min                  1    3.4478 7188.8 -114.948
    ## - kw_max_avg                  1    3.4608 7188.8 -114.935
    ## - num_videos                  1    3.8904 7189.2 -114.496
    ## - LDA_00                      1    4.0390 7189.4 -114.344
    ## - kw_avg_max                  1    9.8810 7195.2 -108.377
    ## - max_negative_polarity       1   11.6327 7196.9 -106.589
    ## - num_self_hrefs              1   16.2211 7201.5 -101.907
    ## - avg_negative_polarity       1   18.2569 7203.6  -99.831
    ## - kw_avg_avg                  1   19.0057 7204.3  -99.067
    ## - min_negative_polarity       1   20.0164 7205.3  -98.037
    ## - num_hrefs                   1   22.8571 7208.2  -95.141
    ## - n_tokens_content            1   26.2055 7211.5  -91.729
    ## 
    ## Step:  AIC=-117.72
    ## shares ~ n_tokens_title + n_tokens_content + num_hrefs + num_self_hrefs + 
    ##     num_imgs + num_videos + kw_min_min + kw_avg_min + kw_max_max + 
    ##     kw_avg_max + kw_max_avg + kw_avg_avg + self_reference_avg_sharess + 
    ##     LDA_00 + global_rate_positive_words + rate_positive_words + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity
    ## 
    ##                              Df Sum of Sq    RSS      AIC
    ## - n_tokens_title              1    1.1078 7187.2 -118.583
    ## - self_reference_avg_sharess  1    1.1078 7187.2 -118.583
    ## - kw_max_max                  1    1.1741 7187.2 -118.515
    ## - abs_title_subjectivity      1    1.2398 7187.3 -118.448
    ## - title_sentiment_polarity    1    1.3495 7187.4 -118.336
    ## <none>                                    7186.1 -117.715
    ## - rate_positive_words         1    2.0142 7188.1 -117.657
    ## - kw_max_avg                  1    2.7827 7188.8 -116.871
    ## - kw_avg_min                  1    2.9860 7189.0 -116.664
    ## - kw_min_min                  1    3.0787 7189.1 -116.569
    ## - global_rate_positive_words  1    3.3136 7189.4 -116.329
    ## - num_imgs                    1    3.3338 7189.4 -116.308
    ## - LDA_00                      1    3.9263 7190.0 -115.703
    ## - num_videos                  1    4.0255 7190.1 -115.601
    ## - kw_avg_max                  1    9.1410 7195.2 -110.377
    ## - max_negative_polarity       1   11.6730 7197.7 -107.792
    ## - num_self_hrefs              1   16.1222 7202.2 -103.253
    ## - kw_avg_avg                  1   18.2666 7204.3 -101.066
    ## - avg_negative_polarity       1   18.3900 7204.4 -100.940
    ## - min_negative_polarity       1   20.1325 7206.2  -99.164
    ## - num_hrefs                   1   23.2892 7209.3  -95.946
    ## - n_tokens_content            1   26.0693 7212.1  -93.114
    ## 
    ## Step:  AIC=-118.58
    ## shares ~ n_tokens_content + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + kw_min_min + kw_avg_min + kw_max_max + kw_avg_max + 
    ##     kw_max_avg + kw_avg_avg + self_reference_avg_sharess + LDA_00 + 
    ##     global_rate_positive_words + rate_positive_words + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_sentiment_polarity + 
    ##     abs_title_subjectivity
    ## 
    ##                              Df Sum of Sq    RSS      AIC
    ## - self_reference_avg_sharess  1    1.0709 7188.2 -119.489
    ## - kw_max_max                  1    1.2332 7188.4 -119.323
    ## - title_sentiment_polarity    1    1.2606 7188.4 -119.295
    ## - abs_title_subjectivity      1    1.6635 7188.8 -118.883
    ## <none>                                    7187.2 -118.583
    ## - rate_positive_words         1    1.9851 7189.1 -118.554
    ## - kw_max_avg                  1    2.5698 7189.7 -117.957
    ## - kw_avg_min                  1    3.0193 7190.2 -117.498
    ## - kw_min_min                  1    3.0707 7190.2 -117.445
    ## - global_rate_positive_words  1    3.3513 7190.5 -117.158
    ## - num_imgs                    1    3.4129 7190.6 -117.096
    ## - LDA_00                      1    3.8477 7191.0 -116.651
    ## - num_videos                  1    4.1253 7191.3 -116.368
    ## - kw_avg_max                  1    8.4953 7195.7 -111.905
    ## - max_negative_polarity       1   11.6400 7198.8 -108.695
    ## - num_self_hrefs              1   15.6499 7202.8 -104.605
    ## - kw_avg_avg                  1   17.5129 7204.7 -102.705
    ## - avg_negative_polarity       1   18.3730 7205.5 -101.828
    ## - min_negative_polarity       1   20.0086 7207.2 -100.161
    ## - num_hrefs                   1   22.4049 7209.6  -97.719
    ## - n_tokens_content            1   26.5553 7213.7  -93.491
    ## 
    ## Step:  AIC=-119.49
    ## shares ~ n_tokens_content + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + kw_min_min + kw_avg_min + kw_max_max + kw_avg_max + 
    ##     kw_max_avg + kw_avg_avg + LDA_00 + global_rate_positive_words + 
    ##     rate_positive_words + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_sentiment_polarity + abs_title_subjectivity
    ## 
    ##                              Df Sum of Sq    RSS      AIC
    ## - kw_max_max                  1    1.1832 7189.4 -120.279
    ## - title_sentiment_polarity    1    1.2509 7189.5 -120.210
    ## - abs_title_subjectivity      1    1.6804 7189.9 -119.771
    ## <none>                                    7188.2 -119.489
    ## - rate_positive_words         1    1.9802 7190.2 -119.465
    ## - kw_max_avg                  1    2.5863 7190.8 -118.846
    ## - kw_avg_min                  1    3.0490 7191.3 -118.373
    ## - kw_min_min                  1    3.0521 7191.3 -118.370
    ## - global_rate_positive_words  1    3.4382 7191.7 -117.976
    ## - num_imgs                    1    3.4423 7191.7 -117.972
    ## - LDA_00                      1    3.7965 7192.0 -117.610
    ## - num_videos                  1    4.1018 7192.3 -117.298
    ## - kw_avg_max                  1    8.0273 7196.3 -113.290
    ## - max_negative_polarity       1   11.7006 7199.9 -109.541
    ## - num_self_hrefs              1   15.6844 7203.9 -105.477
    ## - kw_avg_avg                  1   17.8284 7206.1 -103.291
    ## - avg_negative_polarity       1   18.6621 7206.9 -102.442
    ## - min_negative_polarity       1   20.2300 7208.5 -100.844
    ## - num_hrefs                   1   22.3122 7210.5  -98.722
    ## - n_tokens_content            1   26.7276 7215.0  -94.225
    ## 
    ## Step:  AIC=-120.28
    ## shares ~ n_tokens_content + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + kw_min_min + kw_avg_min + kw_avg_max + kw_max_avg + 
    ##     kw_avg_avg + LDA_00 + global_rate_positive_words + rate_positive_words + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity
    ## 
    ##                              Df Sum of Sq    RSS      AIC
    ## - title_sentiment_polarity    1    1.2693 7190.7 -120.983
    ## - abs_title_subjectivity      1    1.6864 7191.1 -120.557
    ## - rate_positive_words         1    1.9531 7191.4 -120.284
    ## <none>                                    7189.4 -120.279
    ## - kw_min_min                  1    1.9858 7191.4 -120.251
    ## - kw_max_avg                  1    2.9177 7192.3 -119.299
    ## - kw_avg_min                  1    3.0513 7192.5 -119.162
    ## - global_rate_positive_words  1    3.5349 7192.9 -118.668
    ## - num_imgs                    1    3.5518 7193.0 -118.651
    ## - LDA_00                      1    3.7505 7193.2 -118.448
    ## - num_videos                  1    4.1001 7193.5 -118.091
    ## - kw_avg_max                  1    6.9866 7196.4 -115.144
    ## - max_negative_polarity       1   11.6304 7201.0 -110.405
    ## - num_self_hrefs              1   15.8332 7205.2 -106.119
    ## - avg_negative_polarity       1   18.5232 7207.9 -103.377
    ## - kw_avg_avg                  1   20.0268 7209.4 -101.845
    ## - min_negative_polarity       1   20.1324 7209.5 -101.737
    ## - num_hrefs                   1   22.5236 7211.9  -99.301
    ## - n_tokens_content            1   27.0092 7216.4  -94.734
    ## 
    ## Step:  AIC=-120.98
    ## shares ~ n_tokens_content + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + kw_min_min + kw_avg_min + kw_avg_max + kw_max_avg + 
    ##     kw_avg_avg + LDA_00 + global_rate_positive_words + rate_positive_words + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     abs_title_subjectivity
    ## 
    ##                              Df Sum of Sq    RSS      AIC
    ## - rate_positive_words         1    1.7711 7192.5 -121.173
    ## <none>                                    7190.7 -120.983
    ## - kw_min_min                  1    2.0677 7192.8 -120.870
    ## - kw_max_avg                  1    3.0002 7193.7 -119.918
    ## - kw_avg_min                  1    3.0700 7193.8 -119.847
    ## - abs_title_subjectivity      1    3.1208 7193.8 -119.795
    ## - global_rate_positive_words  1    3.3674 7194.1 -119.543
    ## - num_imgs                    1    3.3786 7194.1 -119.532
    ## - LDA_00                      1    3.6368 7194.3 -119.268
    ## - num_videos                  1    4.1405 7194.8 -118.754
    ## - kw_avg_max                  1    7.1559 7197.8 -115.676
    ## - max_negative_polarity       1   11.4080 7202.1 -111.337
    ## - num_self_hrefs              1   16.0183 7206.7 -106.636
    ## - avg_negative_polarity       1   18.2129 7208.9 -104.400
    ## - min_negative_polarity       1   19.9921 7210.7 -102.587
    ## - kw_avg_avg                  1   20.5971 7211.3 -101.971
    ## - num_hrefs                   1   23.0291 7213.7  -99.494
    ## - n_tokens_content            1   27.4605 7218.1  -94.982
    ## 
    ## Step:  AIC=-121.17
    ## shares ~ n_tokens_content + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + kw_min_min + kw_avg_min + kw_avg_max + kw_max_avg + 
    ##     kw_avg_avg + LDA_00 + global_rate_positive_words + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + abs_title_subjectivity
    ## 
    ##                              Df Sum of Sq    RSS      AIC
    ## <none>                                    7192.5 -121.173
    ## - kw_min_min                  1    2.0946 7194.5 -121.034
    ## - kw_max_avg                  1    2.9798 7195.4 -120.131
    ## - kw_avg_min                  1    3.0906 7195.5 -120.018
    ## - abs_title_subjectivity      1    3.3797 7195.8 -119.722
    ## - LDA_00                      1    3.4724 7195.9 -119.628
    ## - num_imgs                    1    3.5289 7196.0 -119.570
    ## - num_videos                  1    4.0457 7196.5 -119.043
    ## - kw_avg_max                  1    7.1640 7199.6 -115.860
    ## - global_rate_positive_words  1    8.9743 7201.4 -114.013
    ## - max_negative_polarity       1   13.6989 7206.2 -109.195
    ## - num_self_hrefs              1   16.3885 7208.8 -106.454
    ## - min_negative_polarity       1   18.4051 7210.9 -104.399
    ## - avg_negative_polarity       1   19.0321 7211.5 -103.761
    ## - kw_avg_avg                  1   20.5668 7213.0 -102.198
    ## - num_hrefs                   1   22.4854 7214.9 -100.244
    ## - n_tokens_content            1   26.3746 7218.8  -96.285

``` r
lm$call[["formula"]] # model selected based on AIC. 
```

    ## shares ~ n_tokens_content + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + kw_min_min + kw_avg_min + kw_avg_max + kw_max_avg + 
    ##     kw_avg_avg + LDA_00 + global_rate_positive_words + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + abs_title_subjectivity
    ## <environment: 0x7f9b9af9b2a8>

``` r
# get the selected subset
num.s <- pop.data.num %>% select(n_tokens_content , n_non_stop_words , n_non_stop_unique_tokens , num_hrefs , num_videos , kw_avg_max , kw_min_avg , kw_max_avg , kw_avg_avg , self_reference_min_shares , self_reference_avg_sharess , abs_title_subjectivity, shares)

# check correlations
cor <- round(cor(num.s, use="complete.obs"), 2)

# select correlated variables without variables with collinearity (ex. kw_min_min and kw_max_min to kw_avg_min)
plot.s <- num.s %>% select(n_tokens_content, n_non_stop_unique_tokens, n_non_stop_words, num_hrefs, num_videos, kw_avg_avg, shares)

# plot a correlation plot
cor.plot <- round(cor(plot.s, use="complete.obs"), 2)
ggcorrplot(cor.plot, hc.order = TRUE, type = "lower", lab = TRUE)
```

<img src="./cor-1.png" style="display: block; margin: auto;" />

For this correlation plot, the color is red if two variables are
positively correlated and is blue if two variables are negatively
correlated.

Some tables for selected data channel of interest showing the counts and
percentage grouped by channel and weekday

``` r
#simple table displaying counts for different type of channel (all obs)
table(X$channel) 
```

    ## 
    ##           data_channel_is_bus data_channel_is_entertainment 
    ##                          6258                          7057 
    ##     data_channel_is_lifestyle        data_channel_is_socmed 
    ##                          2099                          2323 
    ##          data_channel_is_tech         data_channel_is_world 
    ##                          7346                          8427

``` r
#some summary stats grouped by channel 
C1 <- X %>% 
    group_by( channel ) %>% 
    summarise( percent = 100 * n() / nrow( new_data ),mean_shares = mean(shares), mean_images = mean(num_imgs),mean_video = mean(num_videos),mean_link = mean(num_hrefs))
knitr::kable(C1)
```

| channel                          |   percent | mean\_shares | mean\_images | mean\_video | mean\_link |
| :------------------------------- | --------: | -----------: | -----------: | ----------: | ---------: |
| data\_channel\_is\_bus           | 18.675022 |     3063.019 |     1.808405 |   0.6364653 |   9.356184 |
| data\_channel\_is\_entertainment | 21.059385 |     2970.487 |     6.317699 |   2.5458410 |  10.689670 |
| data\_channel\_is\_lifestyle     |  6.263802 |     3682.123 |     4.904717 |   0.4749881 |  13.419247 |
| data\_channel\_is\_socmed        |  6.932259 |     3629.383 |     4.290142 |   1.1175204 |  13.176065 |
| data\_channel\_is\_tech          | 21.921814 |     3072.283 |     4.434522 |   0.4471821 |   9.416825 |
| data\_channel\_is\_world         | 25.147717 |     2287.734 |     2.841225 |   0.5495431 |  10.195206 |

``` r
# using the subset data set containing weekday info in one column. 
table(pop.data$weekday)
```

    ## 
    ##    weekday_is_friday    weekday_is_monday  weekday_is_saturday    weekday_is_sunday 
    ##                  989                 1235                  525                  396 
    ##  weekday_is_thursday   weekday_is_tuesday weekday_is_wednesday 
    ##                 1310                 1474                 1417

``` r
C2 <- pop.data %>% 
    group_by( weekday ) %>% 
    summarise( percent = 100 * n() / nrow( X ),mean_shares = mean(shares), mean_images = mean(num_imgs),mean_video = mean(num_videos),mean_link = mean(num_hrefs))
knitr::kable(C2)
```

| weekday                |  percent | mean\_shares | mean\_images | mean\_video | mean\_link |
| :--------------------- | -------: | -----------: | -----------: | ----------: | ---------: |
| weekday\_is\_friday    | 2.951358 |     3050.813 |     3.990900 |   0.4226491 |   9.267947 |
| weekday\_is\_monday    | 3.685467 |     2821.483 |     5.428340 |   0.4202429 |   9.950607 |
| weekday\_is\_saturday  | 1.566696 |     3615.453 |     5.007619 |   0.3847619 |  12.230476 |
| weekday\_is\_sunday    | 1.181737 |     3935.687 |     4.229798 |   0.4974747 |  11.618687 |
| weekday\_is\_thursday  | 3.909281 |     2744.543 |     4.102290 |   0.4076336 |   8.632061 |
| weekday\_is\_tuesday   | 4.398687 |     2883.410 |     4.307327 |   0.4911805 |   8.884668 |
| weekday\_is\_wednesday | 4.228588 |     3362.784 |     4.162315 |   0.4876500 |   8.676782 |

``` r
table(pop.data$weekday, pop.data$channel)
```

    ##                       
    ##                        data_channel_is_tech
    ##   weekday_is_friday                     989
    ##   weekday_is_monday                    1235
    ##   weekday_is_saturday                   525
    ##   weekday_is_sunday                     396
    ##   weekday_is_thursday                  1310
    ##   weekday_is_tuesday                   1474
    ##   weekday_is_wednesday                 1417

``` r
table(pop.data$channel, pop.data$is_weekend)
```

    ##                       
    ##                           0    1
    ##   data_channel_is_tech 6425  921

``` r
C3 <- pop.data %>% group_by(is_weekend)%>% 
    summarise( percent = 100 * n() / nrow( X ),mean_shares = mean(shares), mean_images = mean(num_imgs),mean_video = mean(num_videos),mean_link = mean(num_hrefs))
knitr::kable(C3)
```

| is\_weekend |   percent | mean\_shares | mean\_images | mean\_video | mean\_link |
| :---------- | --------: | -----------: | -----------: | ----------: | ---------: |
| 0           | 19.173381 |     2974.685 |     4.400311 |   0.4491829 |   9.051206 |
| 1           |  2.748433 |     3753.143 |     4.673181 |   0.4332248 |  11.967427 |

### Graphical summaries

``` r
#Scatter plot for n_tokens_content v.s. Shares.
scatter.tc <- ggplot(data = pop.data, aes(x = n_tokens_content, y = shares))
scatter.tc + geom_point(aes(color = is_weekend)) + 
             geom_smooth(method = "lm") + 
             labs(title = "Number of Tokens in Content v.s. Shares", x = "Number of tokens in content", y = "shares") + 
             scale_color_discrete(name = "is_weekend")
```

<img src="./scatter1-1.png" style="display: block; margin: auto;" />

If the linear regression line shows an upward trend, then articles with
more words in content tend to be shared more often; if it shows a
downward trend, then articles with more words in content tend to be
shared less often.

``` r
#Scatter plot for Videos v.s. Shares.
scatter.video <- ggplot(pop.data,aes(x = num_videos, y =shares))
scatter.video + geom_point(aes(shape = is_weekend, color = weekday), size = 2) + 
                geom_smooth(method = "lm") + 
                labs(x = "Videos", y = "Shares", title = "Videos vs Shares ") +  
                scale_shape_manual(values = c(3:4))+
                scale_color_discrete(name = "weekday")+
                scale_shape_discrete(name="is_weekend")
```

<img src="./scatter2-1.png" style="display: block; margin: auto;" />

Similarly, if the linear regression line shows an upward trend, then
articles with more videos tend to be shared more often; if it shows a
downward trend, then articles with more videos tend to be shared less
often.

``` r
#Scatter plot for Number of Tokens in Content v.s. Number of Links.
scatter.stop <- ggplot(data = pop.data, aes(x = n_tokens_content, y = num_hrefs))
scatter.stop + geom_point(aes(color = is_weekend)) + 
               geom_smooth(method = "lm") + 
               labs(title = "Number of Tokens in Content v.s. Number of Links", 
                    x = "Number of Tokens in Content", 
                    y = "Number of Links") + 
               scale_color_discrete(name = "is_weekend")
```

<img src="./scatter3-1.png" style="display: block; margin: auto;" />

Observing from the plot, if the linear regression line is upward, there
is a positive correlation relationship between the number of words and
number of links in articles. The number of links increases as the number
of words increases in the articles. If the linear regression line is
downward, the result is the reverse.

#### General plots

This is a bar plot channel by weekend(is or not)

``` r
#bar plot channel by weekend
bar.weekend <-ggplot(X,aes(x = channel))
bar.weekend + geom_bar(aes(fill = as.factor(is_weekend)), position = "dodge") + 
              labs(x = "channel", y = "Count", title = "Channel by Weekend") +
              theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
              scale_fill_discrete(name = "is_weekend") 
```

<img src="./barplot1-1.png" style="display: block; margin: auto;" />
Observe the number of counts from the y-axis of this barplot, we can
compare whether there is more shares on the weekends or less shares on
weekends for each channels.

Boxplot for different channels.

``` r
box <- ggplot(X, aes(x = channel, y = shares))
box + geom_boxplot(position = "dodge") + 
      labs(x = "y", title = "Boxplot for popularity with channel type ") + 
      scale_x_discrete(name = "channel") + 
      geom_jitter(aes(color = as.factor(weekday))) + 
      scale_y_continuous() + 
      theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
      scale_color_discrete(name = "weekday")
```

<img src="./boxplot-1.png" style="display: block; margin: auto;" />
There are some outliers of shares if the points are away from the box.

This is the bar plot : channel by weekday(stacked bar).

``` r
s.bar <- ggplot(X, aes(x = weekday))
s.bar + geom_bar(aes(fill = as.factor(channel)), 
                 position = "stack",show.legend = NA) +         labs(x = "weekday") + 
        scale_fill_discrete(name = "channel") + 
        theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
        labs(title = "weekday by channel ")
```

<img src="./barplot2-1.png" style="display: block; margin: auto;" />

``` r
 # or  
g <- ggplot(X, aes(x = channel))
g + geom_bar(aes(fill = as.factor(weekday)),
             position = "stack",show.legend = NA) + 
    labs(x = "channel")+ 
    scale_fill_discrete(name = "weekday") + 
    theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1))+
    labs(title = " channel by weekday ")
```

<img src="./barplot2-2.png" style="display: block; margin: auto;" />

The stacked bar helps to understand the proportions between each
channels/weekdays by comparing the size of rectangles. (Different colors
represent different channels/weekdays.)

## Modeling(Regression Settings)

``` r
# Use parallel computing to speed up computations
cores <- detectCores()
cl <- makePSOCKcluster(cores-1)
registerDoParallel(cl)
```

Using 5-fold Cross-Validation.

``` r
ctrl <- trainControl(method = "cv", number = 5)
```

### Split the data set.

Before fitting any predictive models, we tried some methods that could
help reduce the dimension of data.  
We randomly selected some predictors of interest and perform the best
subset selection under the condition of least square linear regression.

``` r
# for the variable that can be used in the linear regression model. 
# try best subset selection, select number of variables using adjusted R^2, and mallow's cp, BIC,
 
final <- pop.data %>% select(n_tokens_content , n_non_stop_words , n_non_stop_unique_tokens , num_hrefs , num_imgs,num_keywords, num_videos , kw_avg_max , kw_min_avg , kw_max_avg , kw_avg_avg , self_reference_min_shares , self_reference_avg_sharess ,global_rate_positive_words,rate_positive_words, abs_title_subjectivity,abs_title_sentiment_polarity,shares)

set.seed(1033)

# split the subset data into training and testing set. Use p = 0.7.
train.index.sub <- createDataPartition(y = final$shares, p = 0.7, list = F)
train.sub <- final[train.index.sub, ] # training set
test.sub <- final[-train.index.sub, ] # test set

regression1 <- regsubsets(shares ~., data = train.sub, nvmax=17)
hh1<-summary(regression1)

# this is the indicators of the variables that are supposed to be included in the model each time(iteration)
knitr::kable(hh1$which)
```

| (Intercept) | n\_tokens\_content | n\_non\_stop\_words | n\_non\_stop\_unique\_tokens | num\_hrefs | num\_imgs | num\_keywords | num\_videos | kw\_avg\_max | kw\_min\_avg | kw\_max\_avg | kw\_avg\_avg | self\_reference\_min\_shares | self\_reference\_avg\_sharess | global\_rate\_positive\_words | rate\_positive\_words | abs\_title\_subjectivity | abs\_title\_sentiment\_polarity |
| :---------- | :----------------- | :------------------ | :--------------------------- | :--------- | :-------- | :------------ | :---------- | :----------- | :----------- | :----------- | :----------- | :--------------------------- | :---------------------------- | :---------------------------- | :-------------------- | :----------------------- | :------------------------------ |
| TRUE        | FALSE              | FALSE               | FALSE                        | TRUE       | FALSE     | FALSE         | FALSE       | FALSE        | FALSE        | FALSE        | FALSE        | FALSE                        | FALSE                         | FALSE                         | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | FALSE              | FALSE               | FALSE                        | TRUE       | FALSE     | FALSE         | FALSE       | FALSE        | FALSE        | FALSE        | TRUE         | FALSE                        | FALSE                         | FALSE                         | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | TRUE               | FALSE               | FALSE                        | TRUE       | FALSE     | FALSE         | FALSE       | FALSE        | FALSE        | FALSE        | TRUE         | FALSE                        | FALSE                         | FALSE                         | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | TRUE               | FALSE               | FALSE                        | TRUE       | TRUE      | FALSE         | FALSE       | FALSE        | FALSE        | FALSE        | TRUE         | FALSE                        | FALSE                         | FALSE                         | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | TRUE               | FALSE               | FALSE                        | TRUE       | TRUE      | FALSE         | FALSE       | FALSE        | FALSE        | FALSE        | TRUE         | FALSE                        | FALSE                         | TRUE                          | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | TRUE               | FALSE               | FALSE                        | TRUE       | TRUE      | FALSE         | FALSE       | TRUE         | FALSE        | TRUE         | TRUE         | FALSE                        | FALSE                         | FALSE                         | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | TRUE               | FALSE               | FALSE                        | TRUE       | TRUE      | FALSE         | FALSE       | TRUE         | FALSE        | TRUE         | TRUE         | FALSE                        | FALSE                         | TRUE                          | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | TRUE               | FALSE               | FALSE                        | TRUE       | TRUE      | FALSE         | FALSE       | TRUE         | FALSE        | TRUE         | TRUE         | FALSE                        | FALSE                         | TRUE                          | FALSE                 | TRUE                     | FALSE                           |
| TRUE        | TRUE               | FALSE               | FALSE                        | TRUE       | TRUE      | FALSE         | FALSE       | TRUE         | TRUE         | TRUE         | TRUE         | FALSE                        | FALSE                         | TRUE                          | FALSE                 | TRUE                     | FALSE                           |
| TRUE        | TRUE               | FALSE               | FALSE                        | TRUE       | TRUE      | FALSE         | TRUE        | TRUE         | TRUE         | TRUE         | TRUE         | FALSE                        | FALSE                         | TRUE                          | FALSE                 | TRUE                     | FALSE                           |
| TRUE        | TRUE               | FALSE               | FALSE                        | TRUE       | TRUE      | FALSE         | TRUE        | TRUE         | TRUE         | TRUE         | TRUE         | FALSE                        | TRUE                          | TRUE                          | FALSE                 | TRUE                     | FALSE                           |
| TRUE        | TRUE               | FALSE               | TRUE                         | TRUE       | TRUE      | FALSE         | TRUE        | TRUE         | TRUE         | TRUE         | TRUE         | FALSE                        | TRUE                          | TRUE                          | FALSE                 | TRUE                     | FALSE                           |
| TRUE        | TRUE               | FALSE               | TRUE                         | TRUE       | TRUE      | FALSE         | TRUE        | TRUE         | TRUE         | TRUE         | TRUE         | FALSE                        | TRUE                          | TRUE                          | TRUE                  | TRUE                     | FALSE                           |
| TRUE        | TRUE               | TRUE                | TRUE                         | TRUE       | TRUE      | FALSE         | TRUE        | TRUE         | TRUE         | TRUE         | TRUE         | FALSE                        | TRUE                          | TRUE                          | TRUE                  | TRUE                     | FALSE                           |
| TRUE        | TRUE               | TRUE                | TRUE                         | TRUE       | TRUE      | TRUE          | TRUE        | TRUE         | TRUE         | TRUE         | TRUE         | FALSE                        | TRUE                          | TRUE                          | TRUE                  | TRUE                     | FALSE                           |
| TRUE        | TRUE               | TRUE                | TRUE                         | TRUE       | TRUE      | TRUE          | TRUE        | TRUE         | TRUE         | TRUE         | TRUE         | TRUE                         | TRUE                          | TRUE                          | TRUE                  | TRUE                     | FALSE                           |
| TRUE        | TRUE               | TRUE                | TRUE                         | TRUE       | TRUE      | TRUE          | TRUE        | TRUE         | TRUE         | TRUE         | TRUE         | TRUE                         | TRUE                          | TRUE                          | TRUE                  | TRUE                     | TRUE                            |

A simple function that helps to get the model for the best subset
selection.

``` r
get_model_formula <- function(id, object, outcome){
  # get models data
  models <- hh1$which[id,-1]
  # Get model predictors
  predictors <- names(which(models == TRUE))
  predictors <- paste(predictors, collapse = "+")
  # Build model formula
  as.formula(paste0(outcome, "~", predictors))
}
```

Using mallow’s cp, BIC and Adjusted R^2, to do model selection.

``` r
gk <- data.frame(
  Adj.R2 = which.max(hh1$adjr2),
  CP = which.min(hh1$cp),
  BIC = which.min(hh1$bic)
)
knitr::kable(gk)
```

| Adj.R2 | CP | BIC |
| -----: | -: | --: |
|     12 |  9 |   1 |

``` r
par(mfrow=c(2,2))
plot(hh1$cp ,xlab="Number of Variables ",ylab="Mallow's Cp", type='b')
plot(hh1$adjr2 ,xlab="Number of Variables ",ylab="Adjusted R^2 ", type='b')
plot(hh1$bic,xlab="Number of Variables ",ylab="BIC ", type='b')

# using the mallow's cp to choose model size. 
best_subset_model <- get_model_formula(which.min(hh1$cp),models,"shares")
```

<img src="./select.bs-1.png" style="display: block; margin: auto;" />

After using the best subset selection, some important variables are:
`n_tokens_content`, `num_videos`, `n_non_stop_words`,
`n_non_stop_unique_tokens`, `self_reference_min_shares`, `kw_avg_avg`,
`abs_title_subjectivity`, and `kw_max_avg`.

### Linear Regression

Since using all predictors is time-consuming and hard to render in
automation, we use the random selected variables(p = 17) from the best
subset selection to fit the linear regression models.

``` r
# check the model selected by the best subset selection 
lm.fit1 <- train(best_subset_model, data = train.sub,
                 method = "lm", preProcess =c("center", "scale"), 
                 trControl = ctrl)

# Consider all variables in the best subset, pick a model using forward selection method.
lm2 <- step(lm(shares ~ . , data = train.sub), direction = "forward")

# fit the model chosen from the forward selection for all linear terms
lm.fit2 <- train(lm2$call[["formula"]], data = train.sub, 
                 method = "lm", preProcess =c("center", "scale"), 
                 trControl = ctrl)

# Consider variables selected with forward selection (with 2-way interactions).
lm3 <- step(lm(lm2$call[["formula"]], data = train.sub), scope = . ~.^2, direction = "both", use.start = TRUE)

# fit the model chosen from both forward and backward method for the interaction terms and linear terms. 
lm.fit3 <- train(lm3$call[["formula"]], data = train.sub, 
                 method = "lm", preProcess =c("center", "scale"), 
                 trControl = ctrl)
```

``` r
# create a table to compare the results of linear regression from training data 
lm.compare <- data.frame(models= c("lm.fit1", "lm.fit2","lm.fit3"), 
                         results = bind_rows(lm.fit1$results[2:4], lm.fit2$results[2:4], lm.fit3$results[2:4]))
knitr::kable(lm.compare) 
```

| models  | results.RMSE | results.Rsquared | results.MAE |
| :------ | -----------: | ---------------: | ----------: |
| lm.fit1 |     7892.152 |        0.0149577 |    2459.061 |
| lm.fit2 |     7990.044 |        0.0110626 |    2486.444 |
| lm.fit3 |     8809.201 |        0.0164781 |    3068.173 |

Check Linear Regression model performance on test set

``` r
# Best subset
pred.lm1 <- predict(lm.fit1, newdata = test.sub)
test.RMSE.lm1 <- RMSE(pred.lm1, test.sub$shares)

# Forward
pred.lm2 <- predict(lm.fit2, newdata = test.sub)
test.RMSE.lm2 <- RMSE(pred.lm2, test.sub$shares)

# both
pred.lm3 <- predict(lm.fit3, newdata = test.sub)
test.RMSE.lm3 <- RMSE(pred.lm3, test.sub$shares)
```

### Lasso Regression

Since lasso perform the variable selection, we tried to use Lasso
Regression(adding tuning parameter/ penalty)  
Lasso using all the predictors and get the test MSE

``` r
pop.data <- X %>% filter(channel == params$channel) %>% select(-1:-2)

# split data
train.index <- createDataPartition(y = pop.data$shares, p = 0.7, list = F)

train.lasso <- pop.data[train.index, ] # training set
test.lasso <- pop.data[-train.index, ] # test set

# using all predictors (52 predictors)
cv.out.full <- cv.glmnet(as.matrix(train.lasso[,-47:-48]), train.lasso$shares, alpha=1)

#MSE versus the log(lambda)
plot(cv.out.full,main = "tuning parameter selection for lasso(full predictors)")
```

<img src="./lasso-1.png" style="display: block; margin: auto;" />

``` r
best.lambda.full <- cv.out.full$lambda.min

#fitting the lasso regression 
lasso.fit.full <- glmnet(train.lasso[,-46:-48] ,train.lasso$shares, alpha = 1, lambda = best.lambda.full)
lasso.coef.full <- predict(lasso.fit.full, type = "coefficients")
print(lasso.coef.full)
```

    ## 46 x 1 sparse Matrix of class "dgCMatrix"
    ##                                        s0
    ## (Intercept)                  1908.2475224
    ## n_tokens_title                  .        
    ## n_tokens_content                0.7125094
    ## n_unique_tokens                 .        
    ## n_non_stop_words                .        
    ## n_non_stop_unique_tokens        .        
    ## num_hrefs                      38.2268329
    ## num_self_hrefs                -10.8132761
    ## num_imgs                        .        
    ## num_videos                     61.9155178
    ## average_token_length            .        
    ## num_keywords                    .        
    ## kw_min_min                      .        
    ## kw_max_min                      .        
    ## kw_avg_min                      .        
    ## kw_min_max                      .        
    ## kw_max_max                      .        
    ## kw_avg_max                      .        
    ## kw_min_avg                      .        
    ## kw_max_avg                      .        
    ## kw_avg_avg                      0.1575322
    ## self_reference_min_shares       .        
    ## self_reference_max_shares       .        
    ## self_reference_avg_sharess      .        
    ## is_weekend                      .        
    ## LDA_00                          .        
    ## LDA_01                          .        
    ## LDA_02                          .        
    ## LDA_03                          .        
    ## LDA_04                          .        
    ## global_subjectivity             .        
    ## global_sentiment_polarity       .        
    ## global_rate_positive_words      .        
    ## global_rate_negative_words      .        
    ## rate_positive_words             .        
    ## rate_negative_words             .        
    ## avg_positive_polarity           .        
    ## min_positive_polarity           .        
    ## max_positive_polarity           .        
    ## avg_negative_polarity           .        
    ## min_negative_polarity           .        
    ## max_negative_polarity           .        
    ## title_subjectivity              .        
    ## title_sentiment_polarity        .        
    ## abs_title_subjectivity          .        
    ## abs_title_sentiment_polarity    .

Lasso method using the 17 predictors.

``` r
#using selected predictors (17 predictors) 
#use k-fold cv to select best lambda for the lasso regression 
cv.out <- cv.glmnet(as.matrix(train.sub), train.sub$shares, alpha=1)
#MSE versus the log(lambda)
plot(cv.out,main = "tuning parameter selection for lasso(17 predictors)")
```

<img src="./unnamed-chunk-3-1.png" style="display: block; margin: auto;" />

``` r
best.lambda <- cv.out$lambda.min

#fitting the lasso regression 
lasso.fit.18 <- glmnet(train.sub[,-18] ,train.sub$shares, alpha = 1, lambda = best.lambda)
lasso.coef <- predict(lasso.fit.18, type = "coefficients")
print(lasso.coef)
```

    ## 18 x 1 sparse Matrix of class "dgCMatrix"
    ##                                        s0
    ## (Intercept)                  1848.5900415
    ## n_tokens_content                0.5648703
    ## n_non_stop_words                .        
    ## n_non_stop_unique_tokens        .        
    ## num_hrefs                      42.5857103
    ## num_imgs                        .        
    ## num_keywords                    .        
    ## num_videos                      .        
    ## kw_avg_max                      .        
    ## kw_min_avg                      .        
    ## kw_max_avg                      .        
    ## kw_avg_avg                      0.1961846
    ## self_reference_min_shares       .        
    ## self_reference_avg_sharess      .        
    ## global_rate_positive_words      .        
    ## rate_positive_words             .        
    ## abs_title_subjectivity          .        
    ## abs_title_sentiment_polarity    .

Check Lasso performance on test set.

``` r
# Using the 17 predictors
lasso.partial.pred <- predict(lasso.fit.18, newx= as.matrix(test.sub[,-18]))
test.RMSE.lasso.partial <- RMSE(lasso.partial.pred, test.sub$shares)

# Using all predictors (52 predictors)
lasso.pred.full <- predict(lasso.fit.full, newx= as.matrix(test.lasso[,-46:-48]))
test.RMSE.lasso.full <- RMSE(lasso.pred.full, test.lasso$shares)
```

### Random Forest Regression

Random Forest regression is used to de-correlate each model fitting. We
use the model the previously get from the forward and both selection
method to fit the random forest regression. Take computation limit, and
time consumption into account, there are only a few tuning parameters
set for the test.

``` r
# create data frame for tuning parameter
rf.tGrid <- expand.grid(mtry = seq(from = 1, to = 15, by = 1))

# train the Random Forest model
# use model selected by forward selection
rf.fit1 <- train(lm2$call[["formula"]], data = train.sub, 
             method = "rf", trControl = ctrl, 
             preProcess = c("center", "scale"), 
             tuneGrid = rf.tGrid)

# use model selected by both selection
rf.fit2 <- train(lm3$call[["formula"]], data = train.sub, 
             method = "rf", trControl = ctrl, 
             preProcess = c("center", "scale"), 
             tuneGrid = rf.tGrid)

# plot RMSE for each iteration
plot(rf.fit1$results$mtry, rf.fit1$results$RMSE, 
     xlab = "mtry",ylab = "RMSE",type = 'p',main = 'random forest')
```

<img src="./rf-1.png" style="display: block; margin: auto;" />

``` r
plot(rf.fit2$results$mtry, rf.fit2$results$RMSE, 
     xlab = "mtry",ylab = "RMSE",type = 'p',main = 'random forest')
```

<img src="./rf-2.png" style="display: block; margin: auto;" />

Check Random Forest model performance on test set.

``` r
# start model selected by forward
pred.rf1 <- predict(rf.fit1, newdata = test.sub)
test.RMSE.rf1 <- RMSE(pred.rf1, test.sub$shares)

# start model selected by both
pred.rf2 <- predict(rf.fit2, newdata = test.sub)
test.RMSE.rf2 <- RMSE(pred.rf2, test.sub$shares)
```

### Boosting model.(Stochastic Gradient Boosting)

Boosting is a slow learn method that learn from the previous fit each
time in order to prevent over fitting. We also use the model the
previously get from the forward and both selection method to fit the
boosted.  
Boosting tree have several tuning parameters, also, due to some
limitation, the number of tuning parameters and cross validation number
is set to be small.

``` r
# set tuning parameters
tune1 = c(25,50,100,150,200)
tune2 = c(1:10)
tune3 = 0.01
tune4 = 10
boos.grid <- expand.grid(n.trees = tune1, 
                         interaction.depth = tune2, 
                         shrinkage = tune3, 
                         n.minobsinnode = tune4)

# train the Boosted Tree model
# use model selected by forward selection
boostTreefit1 <- train(lm2$call[["formula"]], data = train.sub, 
                method = "gbm",
                preProcess = c("center","scale"),
                trControl = ctrl,
                tuneGrid = boos.grid)

par(mfrow=c(2,2))

plot(boostTreefit1$results$n.trees, boostTreefit1$results$RMSE, 
     xlab = "n.trees",ylab = "RMSE", type = 'p',main = 'boosted')
plot(boostTreefit1$results$interaction.depth, boostTreefit1$results$RMSE, 
     xlab = "subtrees",ylab = "RMSE",type = 'p',main = 'boosted')
plot(boostTreefit1$results$interaction.depth, boostTreefit1$results$Rsquared, 
     xlab = "subtrees",ylab = "R^2",type = 'p',main = 'boosted')

# use model selected by both selection
boostTreefit2 <- train(lm3$call[["formula"]], data = train.sub, 
                 method = "gbm",
                 preProcess = c("center","scale"),
                 trControl = ctrl,
                 tuneGrid = boos.grid)
```

<img src="./boost-1.png" style="display: block; margin: auto;" />

Check Boosted Tree model performance on test set.

``` r
# start model selected by forward
pred.boost1 <- predict(boostTreefit1 , newdata = test.sub)
test.RMSE.boost1 <- RMSE(pred.boost1, test.sub$shares)

# start model selected by both
pred.boost2 <- predict(boostTreefit2 , newdata = test.sub)
test.RMSE.boost2 <- RMSE(pred.boost2, test.sub$shares)
```

``` r
# done with parallel computing 
stopCluster(cl)
```

### Discussion and Model Selection

  - lm.fit1 is chosen by the best subset selection  
  - lm.fit2 is using forward selection to select variables of most
    interest.  
  - lm.fit3 is adding the interaction terms to the model fitting  
  - lasso.fit.full is using the lasso regression to fit the model for
    all predictors (it also perform variable selection)  
  - lasso.fit.18 is using the lasso regression to fit the model for
    random selected predictors(17 predictors )  
  - rf.fit1/rf.fit2 is using de-correlated method to reduce the
    variance  
  - boost.fit is using cross validation to select appropriate tuning
    parameter for the boosted model and use it for prediction.

This is a simple table containing these methods and the Root Mean Square
Error for each model fitting.

``` r
all.compare <- data.frame(models= c("lm.fit1", "lm.fit2","lm.fit3",
                                    "lasso.fit.full","lasso.fit.18",
                                    "rf.fit1","rf.fit2",
                                    "boostTreefit1","boostTreefit2"), 
                          test_RMSE = c(test.RMSE.lm1, test.RMSE.lm2, test.RMSE.lm3,
                                            test.RMSE.lasso.full, test.RMSE.lasso.partial,
                                            test.RMSE.rf1, test.RMSE.rf2, 
                                            test.RMSE.boost1, test.RMSE.boost2))
knitr::kable(all.compare) 
```

| models         | test\_RMSE |
| :------------- | ---------: |
| lm.fit1        |   4396.527 |
| lm.fit2        |   4399.189 |
| lm.fit3        |   6967.940 |
| lasso.fit.full |   4733.187 |
| lasso.fit.18   |   4385.367 |
| rf.fit1        |   4451.516 |
| rf.fit2        |   4448.395 |
| boostTreefit1  |   4384.499 |
| boostTreefit2  |   4380.764 |

Select model with lowest RMSE.

``` r
select <- all.compare %>% filter(test_RMSE == min(test_RMSE))
select
```

The model with lowest RMSE is the model. This model has the best
performance on test set.

### Automation of data channels

We need to read in libraries as well as some data set before knitting
the automation part.

``` r
channels <- unique(X$channel)
output_file <- paste0(channels,".md")

params = lapply(channels, FUN = function(x){list(channel = x)})

reports <- tibble(output_file, params)

library(rmarkdown)

apply(reports, MARGIN = 1,
      FUN = function(x){
        render(input = "./ST558_Project_2.Rmd",
               output_format = "github_document", 
               output_file = x[[1]], 
               params = x[[2]])
      })
```
