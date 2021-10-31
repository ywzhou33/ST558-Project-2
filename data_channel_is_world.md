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

    ## tibble [8,427 × 48] (S3: tbl_df/tbl/data.frame)
    ##  $ n_tokens_title              : num [1:8427] 10 9 12 9 11 11 11 11 9 12 ...
    ##  $ n_tokens_content            : num [1:8427] 231 1248 682 391 125 ...
    ##  $ n_unique_tokens             : num [1:8427] 0.636 0.49 0.46 0.51 0.675 ...
    ##  $ n_non_stop_words            : num [1:8427] 1 1 1 1 1 ...
    ##  $ n_non_stop_unique_tokens    : num [1:8427] 0.797 0.732 0.635 0.65 0.797 ...
    ##  $ num_hrefs                   : num [1:8427] 4 11 10 9 1 8 7 8 3 5 ...
    ##  $ num_self_hrefs              : num [1:8427] 1 0 0 2 1 6 6 0 1 2 ...
    ##  $ num_imgs                    : num [1:8427] 1 1 1 1 1 1 1 1 1 0 ...
    ##  $ num_videos                  : num [1:8427] 1 0 0 1 0 0 0 0 0 0 ...
    ##  $ average_token_length        : num [1:8427] 5.09 4.62 4.62 5.3 4.82 ...
    ##  $ num_keywords                : num [1:8427] 5 8 6 7 6 6 5 7 6 6 ...
    ##  $ kw_min_min                  : num [1:8427] 0 0 0 0 0 0 0 0 0 217 ...
    ##  $ kw_max_min                  : num [1:8427] 0 0 0 0 0 0 0 0 0 504 ...
    ##  $ kw_avg_min                  : num [1:8427] 0 0 0 0 0 ...
    ##  $ kw_min_max                  : num [1:8427] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kw_max_max                  : num [1:8427] 0 0 0 0 0 0 0 0 0 17100 ...
    ##  $ kw_avg_max                  : num [1:8427] 0 0 0 0 0 ...
    ##  $ kw_min_avg                  : num [1:8427] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kw_max_avg                  : num [1:8427] 0 0 0 0 0 ...
    ##  $ kw_avg_avg                  : num [1:8427] 0 0 0 0 0 ...
    ##  $ self_reference_min_shares   : num [1:8427] 0 0 0 704 16100 101 638 0 0 3100 ...
    ##  $ self_reference_max_shares   : num [1:8427] 0 0 0 704 16100 2600 3300 0 0 3100 ...
    ##  $ self_reference_avg_sharess  : num [1:8427] 0 0 0 704 16100 ...
    ##  $ is_weekend                  : Factor w/ 2 levels "0","1": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ LDA_00                      : num [1:8427] 0.04 0.025 0.0333 0.0288 0.0333 ...
    ##  $ LDA_01                      : num [1:8427] 0.04 0.2873 0.0333 0.0286 0.0333 ...
    ##  $ LDA_02                      : num [1:8427] 0.84 0.401 0.867 0.681 0.7 ...
    ##  $ LDA_03                      : num [1:8427] 0.04 0.2619 0.0333 0.0286 0.2 ...
    ##  $ LDA_04                      : num [1:8427] 0.04 0.025 0.0333 0.2334 0.0333 ...
    ##  $ global_subjectivity         : num [1:8427] 0.314 0.482 0.473 0.284 0.396 ...
    ##  $ global_sentiment_polarity   : num [1:8427] 0.0519 0.1024 0.0622 0.0333 0.2108 ...
    ##  $ global_rate_positive_words  : num [1:8427] 0.039 0.0385 0.0499 0.0179 0.048 ...
    ##  $ global_rate_negative_words  : num [1:8427] 0.0303 0.02083 0.03959 0.00512 0 ...
    ##  $ rate_positive_words         : num [1:8427] 0.562 0.649 0.557 0.778 1 ...
    ##  $ rate_negative_words         : num [1:8427] 0.438 0.351 0.443 0.222 0 ...
    ##  $ avg_positive_polarity       : num [1:8427] 0.298 0.404 0.343 0.15 0.281 ...
    ##  $ min_positive_polarity       : num [1:8427] 0.1 0.1 0.05 0.05 0.1 ...
    ##  $ max_positive_polarity       : num [1:8427] 0.5 1 0.6 0.35 0.6 0.7 0.8 1 0.5 1 ...
    ##  $ avg_negative_polarity       : num [1:8427] -0.238 -0.415 -0.22 -0.108 0 ...
    ##  $ min_negative_polarity       : num [1:8427] -0.5 -1 -0.6 -0.167 0 ...
    ##  $ max_negative_polarity       : num [1:8427] -0.1 -0.1 -0.05 -0.05 0 -0.05 -0.125 -0.1 -0.2 -0.1 ...
    ##  $ title_subjectivity          : num [1:8427] 0 0 0.75 0 0.45 ...
    ##  $ title_sentiment_polarity    : num [1:8427] 0 0 -0.25 0 0.4 ...
    ##  $ abs_title_subjectivity      : num [1:8427] 0.5 0.5 0.25 0.5 0.05 ...
    ##  $ abs_title_sentiment_polarity: num [1:8427] 0 0 0.25 0 0.4 ...
    ##  $ shares                      : num [1:8427] 710 2200 1600 598 1500 504 1800 1200 495 755 ...
    ##  $ channel                     : chr [1:8427] "data_channel_is_world" "data_channel_is_world" "data_channel_is_world" "data_channel_is_world" ...
    ##  $ weekday                     : Factor w/ 7 levels "weekday_is_friday",..: 2 2 2 2 2 2 2 2 2 6 ...

``` r
#summary stats for the response variable. 
summary(pop.data$shares)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##      35     827    1100    2288    1900  284700

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

    ## Start:  AIC=-233.1
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
    ## Step:  AIC=-233.1
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
    ## Step:  AIC=-233.1
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
    ## - num_keywords                  1    0.0001 8113.9 -235.10
    ## - num_self_hrefs                1    0.0099 8113.9 -235.09
    ## - kw_min_max                    1    0.0314 8113.9 -235.07
    ## - abs_title_subjectivity        1    0.0347 8113.9 -235.07
    ## - LDA_00                        1    0.0466 8113.9 -235.06
    ## - self_reference_max_shares     1    0.0681 8113.9 -235.03
    ## - max_positive_polarity         1    0.0730 8113.9 -235.03
    ## - LDA_01                        1    0.0820 8113.9 -235.02
    ## - global_rate_positive_words    1    0.1195 8114.0 -234.98
    ## - global_rate_negative_words    1    0.1503 8114.0 -234.95
    ## - title_subjectivity            1    0.2679 8114.1 -234.83
    ## - self_reference_min_shares     1    0.4404 8114.3 -234.65
    ## - avg_positive_polarity         1    0.4882 8114.3 -234.60
    ## - kw_max_max                    1    0.5256 8114.4 -234.56
    ## - self_reference_avg_sharess    1    0.6081 8114.5 -234.47
    ## - min_negative_polarity         1    0.6840 8114.5 -234.39
    ## - min_positive_polarity         1    0.6889 8114.6 -234.39
    ## - n_non_stop_unique_tokens      1    0.7650 8114.6 -234.31
    ## - avg_negative_polarity         1    1.0490 8114.9 -234.01
    ## - rate_positive_words           1    1.0786 8114.9 -233.99
    ## - n_unique_tokens               1    1.2626 8115.1 -233.79
    ## - global_sentiment_polarity     1    1.3644 8115.2 -233.69
    ## - kw_avg_max                    1    1.5327 8115.4 -233.51
    ## - num_videos                    1    1.5543 8115.4 -233.49
    ## - abs_title_sentiment_polarity  1    1.6279 8115.5 -233.41
    ## - kw_min_min                    1    1.6727 8115.5 -233.37
    ## <none>                                      8113.9 -233.10
    ## - LDA_03                        1    2.1047 8116.0 -232.92
    ## - n_tokens_content              1    2.8132 8116.7 -232.18
    ## - title_sentiment_polarity      1    3.3616 8117.2 -231.61
    ## - kw_avg_min                    1    4.2364 8118.1 -230.71
    ## - kw_max_min                    1    4.7678 8118.6 -230.15
    ## - max_negative_polarity         1    5.5992 8119.5 -229.29
    ## - n_non_stop_words              1    6.8582 8120.7 -227.99
    ## - global_subjectivity           1    8.6689 8122.5 -226.11
    ## - LDA_02                        1    8.6982 8122.6 -226.08
    ## - n_tokens_title                1   11.7030 8125.6 -222.96
    ## - kw_min_avg                    1   13.3297 8127.2 -221.27
    ## - num_hrefs                     1   13.7691 8127.6 -220.82
    ## - kw_max_avg                    1   16.5689 8130.4 -217.91
    ## - num_imgs                      1   26.3818 8140.2 -207.75
    ## - kw_avg_avg                    1   30.2883 8144.1 -203.71
    ## - average_token_length          1   30.4586 8144.3 -203.53
    ## 
    ## Step:  AIC=-235.1
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     kw_min_min + kw_max_min + kw_avg_min + kw_min_max + kw_max_max + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - num_self_hrefs                1    0.0103 8113.9 -237.09
    ## - kw_min_max                    1    0.0315 8113.9 -237.07
    ## - abs_title_subjectivity        1    0.0346 8113.9 -237.07
    ## - LDA_00                        1    0.0466 8113.9 -237.06
    ## - self_reference_max_shares     1    0.0681 8113.9 -237.03
    ## - max_positive_polarity         1    0.0730 8113.9 -237.03
    ## - LDA_01                        1    0.0830 8113.9 -237.02
    ## - global_rate_positive_words    1    0.1194 8114.0 -236.98
    ## - global_rate_negative_words    1    0.1503 8114.0 -236.95
    ## - title_subjectivity            1    0.2680 8114.1 -236.83
    ## - self_reference_min_shares     1    0.4403 8114.3 -236.65
    ## - avg_positive_polarity         1    0.4885 8114.3 -236.60
    ## - kw_max_max                    1    0.5352 8114.4 -236.55
    ## - self_reference_avg_sharess    1    0.6080 8114.5 -236.47
    ## - min_negative_polarity         1    0.6843 8114.5 -236.39
    ## - min_positive_polarity         1    0.6888 8114.6 -236.39
    ## - n_non_stop_unique_tokens      1    0.7649 8114.6 -236.31
    ## - avg_negative_polarity         1    1.0489 8114.9 -236.01
    ## - rate_positive_words           1    1.0816 8114.9 -235.98
    ## - n_unique_tokens               1    1.2626 8115.1 -235.79
    ## - global_sentiment_polarity     1    1.3645 8115.2 -235.69
    ## - num_videos                    1    1.5557 8115.4 -235.49
    ## - abs_title_sentiment_polarity  1    1.6286 8115.5 -235.41
    ## - kw_min_min                    1    1.6729 8115.5 -235.37
    ## - kw_avg_max                    1    1.7595 8115.6 -235.28
    ## <none>                                      8113.9 -235.10
    ## - LDA_03                        1    2.1106 8116.0 -234.91
    ## - n_tokens_content              1    2.8236 8116.7 -234.17
    ## - title_sentiment_polarity      1    3.3616 8117.2 -233.61
    ## - kw_avg_min                    1    4.2471 8118.1 -232.69
    ## - kw_max_min                    1    4.7787 8118.6 -232.14
    ## - max_negative_polarity         1    5.5993 8119.5 -231.29
    ## - n_non_stop_words              1    6.8873 8120.7 -229.96
    ## - global_subjectivity           1    8.6695 8122.5 -228.10
    ## - LDA_02                        1    8.7315 8122.6 -228.04
    ## - n_tokens_title                1   11.7247 8125.6 -224.94
    ## - kw_min_avg                    1   13.6927 8127.6 -222.90
    ## - num_hrefs                     1   13.7751 8127.6 -222.81
    ## - kw_max_avg                    1   16.5904 8130.5 -219.89
    ## - num_imgs                      1   26.3818 8140.2 -209.75
    ## - average_token_length          1   30.4616 8144.3 -205.53
    ## - kw_avg_avg                    1   30.6194 8144.5 -205.36
    ## 
    ## Step:  AIC=-237.09
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_imgs + num_videos + average_token_length + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - kw_min_max                    1    0.0292 8113.9 -239.06
    ## - abs_title_subjectivity        1    0.0348 8113.9 -239.06
    ## - LDA_00                        1    0.0481 8113.9 -239.04
    ## - max_positive_polarity         1    0.0756 8113.9 -239.02
    ## - self_reference_max_shares     1    0.0772 8113.9 -239.01
    ## - LDA_01                        1    0.0826 8114.0 -239.01
    ## - global_rate_positive_words    1    0.1193 8114.0 -238.97
    ## - global_rate_negative_words    1    0.1513 8114.0 -238.94
    ## - title_subjectivity            1    0.2649 8114.1 -238.82
    ## - self_reference_min_shares     1    0.4448 8114.3 -238.63
    ## - avg_positive_polarity         1    0.4937 8114.4 -238.58
    ## - kw_max_max                    1    0.5340 8114.4 -238.54
    ## - self_reference_avg_sharess    1    0.6260 8114.5 -238.44
    ## - min_negative_polarity         1    0.6917 8114.6 -238.38
    ## - min_positive_polarity         1    0.6972 8114.6 -238.37
    ## - n_non_stop_unique_tokens      1    0.7822 8114.7 -238.28
    ## - avg_negative_polarity         1    1.0552 8114.9 -238.00
    ## - rate_positive_words           1    1.0808 8115.0 -237.97
    ## - n_unique_tokens               1    1.2832 8115.2 -237.76
    ## - global_sentiment_polarity     1    1.3645 8115.2 -237.68
    ## - num_videos                    1    1.5469 8115.4 -237.49
    ## - abs_title_sentiment_polarity  1    1.6252 8115.5 -237.41
    ## - kw_min_min                    1    1.6715 8115.5 -237.36
    ## - kw_avg_max                    1    1.7602 8115.6 -237.27
    ## <none>                                      8113.9 -237.09
    ## - LDA_03                        1    2.1044 8116.0 -236.91
    ## - n_tokens_content              1    2.8232 8116.7 -236.16
    ## - title_sentiment_polarity      1    3.3582 8117.2 -235.61
    ## - kw_avg_min                    1    4.2470 8118.1 -234.68
    ## - kw_max_min                    1    4.7801 8118.7 -234.13
    ## - max_negative_polarity         1    5.6095 8119.5 -233.27
    ## - n_non_stop_words              1    6.8778 8120.7 -231.95
    ## - global_subjectivity           1    8.7245 8122.6 -230.04
    ## - LDA_02                        1    8.7724 8122.6 -229.99
    ## - n_tokens_title                1   11.7168 8125.6 -226.93
    ## - kw_min_avg                    1   13.8540 8127.7 -224.72
    ## - num_hrefs                     1   14.2256 8128.1 -224.33
    ## - kw_max_avg                    1   16.5917 8130.5 -221.88
    ## - num_imgs                      1   26.4626 8140.3 -211.66
    ## - average_token_length          1   30.4704 8144.3 -207.51
    ## - kw_avg_avg                    1   30.6385 8144.5 -207.33
    ## 
    ## Step:  AIC=-239.06
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_imgs + num_videos + average_token_length + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_max_max + kw_avg_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_00 + LDA_01 + LDA_02 + LDA_03 + 
    ##     global_subjectivity + global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - abs_title_subjectivity        1    0.0347 8113.9 -241.03
    ## - LDA_00                        1    0.0477 8113.9 -241.01
    ## - max_positive_polarity         1    0.0741 8114.0 -240.99
    ## - self_reference_max_shares     1    0.0757 8114.0 -240.99
    ## - LDA_01                        1    0.0835 8114.0 -240.98
    ## - global_rate_positive_words    1    0.1197 8114.0 -240.94
    ## - global_rate_negative_words    1    0.1525 8114.1 -240.91
    ## - title_subjectivity            1    0.2657 8114.2 -240.79
    ## - self_reference_min_shares     1    0.4430 8114.3 -240.60
    ## - avg_positive_polarity         1    0.4907 8114.4 -240.55
    ## - kw_max_max                    1    0.5747 8114.5 -240.47
    ## - self_reference_avg_sharess    1    0.6236 8114.5 -240.42
    ## - min_negative_polarity         1    0.6890 8114.6 -240.35
    ## - min_positive_polarity         1    0.7029 8114.6 -240.33
    ## - n_non_stop_unique_tokens      1    0.7869 8114.7 -240.25
    ## - avg_negative_polarity         1    1.0529 8115.0 -239.97
    ## - rate_positive_words           1    1.0894 8115.0 -239.93
    ## - n_unique_tokens               1    1.2895 8115.2 -239.72
    ## - global_sentiment_polarity     1    1.3705 8115.3 -239.64
    ## - num_videos                    1    1.5689 8115.5 -239.43
    ## - abs_title_sentiment_polarity  1    1.6257 8115.5 -239.38
    ## - kw_min_min                    1    1.6680 8115.6 -239.33
    ## <none>                                      8113.9 -239.06
    ## - LDA_03                        1    2.1251 8116.0 -238.86
    ## - kw_avg_max                    1    2.1933 8116.1 -238.79
    ## - n_tokens_content              1    2.8170 8116.7 -238.14
    ## - title_sentiment_polarity      1    3.3527 8117.3 -237.58
    ## - kw_avg_min                    1    4.2182 8118.1 -236.68
    ## - kw_max_min                    1    4.7550 8118.7 -236.13
    ## - max_negative_polarity         1    5.6063 8119.5 -235.24
    ## - n_non_stop_words              1    6.8644 8120.8 -233.94
    ## - global_subjectivity           1    8.7145 8122.6 -232.02
    ## - LDA_02                        1    8.7432 8122.6 -231.99
    ## - n_tokens_title                1   11.7293 8125.6 -228.89
    ## - num_hrefs                     1   14.2450 8128.1 -226.28
    ## - kw_min_avg                    1   16.0566 8130.0 -224.40
    ## - kw_max_avg                    1   16.6546 8130.6 -223.78
    ## - num_imgs                      1   26.4340 8140.3 -213.65
    ## - average_token_length          1   30.4613 8144.4 -209.49
    ## - kw_avg_avg                    1   30.6981 8144.6 -209.24
    ## 
    ## Step:  AIC=-241.03
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_imgs + num_videos + average_token_length + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_max_max + kw_avg_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_00 + LDA_01 + LDA_02 + LDA_03 + 
    ##     global_subjectivity + global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     title_sentiment_polarity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - LDA_00                        1    0.0466 8114.0 -242.98
    ## - max_positive_polarity         1    0.0757 8114.0 -242.95
    ## - self_reference_max_shares     1    0.0757 8114.0 -242.95
    ## - LDA_01                        1    0.0831 8114.0 -242.94
    ## - global_rate_positive_words    1    0.1277 8114.1 -242.90
    ## - global_rate_negative_words    1    0.1560 8114.1 -242.87
    ## - title_subjectivity            1    0.2311 8114.2 -242.79
    ## - self_reference_min_shares     1    0.4421 8114.4 -242.57
    ## - avg_positive_polarity         1    0.4931 8114.4 -242.52
    ## - kw_max_max                    1    0.5760 8114.5 -242.43
    ## - self_reference_avg_sharess    1    0.6233 8114.6 -242.38
    ## - min_negative_polarity         1    0.6842 8114.6 -242.32
    ## - min_positive_polarity         1    0.7133 8114.6 -242.29
    ## - n_non_stop_unique_tokens      1    0.7857 8114.7 -242.21
    ## - avg_negative_polarity         1    1.0612 8115.0 -241.93
    ## - rate_positive_words           1    1.0987 8115.0 -241.89
    ## - n_unique_tokens               1    1.2870 8115.2 -241.69
    ## - global_sentiment_polarity     1    1.3907 8115.3 -241.58
    ## - num_videos                    1    1.5718 8115.5 -241.40
    ## - abs_title_sentiment_polarity  1    1.6511 8115.6 -241.31
    ## - kw_min_min                    1    1.6683 8115.6 -241.29
    ## <none>                                      8113.9 -241.03
    ## - LDA_03                        1    2.1003 8116.0 -240.85
    ## - kw_avg_max                    1    2.1858 8116.1 -240.76
    ## - n_tokens_content              1    2.8201 8116.8 -240.10
    ## - title_sentiment_polarity      1    3.5423 8117.5 -239.35
    ## - kw_avg_min                    1    4.2284 8118.2 -238.64
    ## - kw_max_min                    1    4.7591 8118.7 -238.09
    ## - max_negative_polarity         1    5.6156 8119.6 -237.20
    ## - n_non_stop_words              1    6.8553 8120.8 -235.91
    ## - global_subjectivity           1    8.6855 8122.6 -234.01
    ## - LDA_02                        1    8.7499 8122.7 -233.94
    ## - n_tokens_title                1   12.0631 8126.0 -230.51
    ## - num_hrefs                     1   14.2731 8128.2 -228.22
    ## - kw_min_avg                    1   16.0513 8130.0 -226.37
    ## - kw_max_avg                    1   16.6509 8130.6 -225.75
    ## - num_imgs                      1   26.4399 8140.4 -215.61
    ## - average_token_length          1   30.4589 8144.4 -211.45
    ## - kw_avg_avg                    1   30.7072 8144.6 -211.20
    ## 
    ## Step:  AIC=-242.98
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_imgs + num_videos + average_token_length + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_max_max + kw_avg_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     title_sentiment_polarity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - self_reference_max_shares     1    0.0755 8114.1 -244.90
    ## - max_positive_polarity         1    0.0784 8114.1 -244.90
    ## - LDA_01                        1    0.1120 8114.1 -244.86
    ## - global_rate_positive_words    1    0.1248 8114.1 -244.85
    ## - global_rate_negative_words    1    0.1600 8114.1 -244.81
    ## - title_subjectivity            1    0.2287 8114.2 -244.74
    ## - self_reference_min_shares     1    0.4432 8114.4 -244.52
    ## - avg_positive_polarity         1    0.5068 8114.5 -244.45
    ## - kw_max_max                    1    0.5828 8114.6 -244.37
    ## - self_reference_avg_sharess    1    0.6232 8114.6 -244.33
    ## - min_negative_polarity         1    0.6875 8114.7 -244.26
    ## - min_positive_polarity         1    0.7285 8114.7 -244.22
    ## - n_non_stop_unique_tokens      1    0.7895 8114.8 -244.16
    ## - avg_negative_polarity         1    1.0630 8115.0 -243.88
    ## - rate_positive_words           1    1.0961 8115.1 -243.84
    ## - n_unique_tokens               1    1.2779 8115.3 -243.65
    ## - global_sentiment_polarity     1    1.3812 8115.4 -243.54
    ## - num_videos                    1    1.5865 8115.6 -243.33
    ## - abs_title_sentiment_polarity  1    1.6475 8115.6 -243.27
    ## - kw_min_min                    1    1.6623 8115.6 -243.25
    ## <none>                                      8114.0 -242.98
    ## - kw_avg_max                    1    2.2367 8116.2 -242.66
    ## - LDA_03                        1    2.3936 8116.4 -242.49
    ## - n_tokens_content              1    2.8466 8116.8 -242.02
    ## - title_sentiment_polarity      1    3.5399 8117.5 -241.30
    ## - kw_avg_min                    1    4.2490 8118.2 -240.57
    ## - kw_max_min                    1    4.7796 8118.8 -240.02
    ## - max_negative_polarity         1    5.6128 8119.6 -239.15
    ## - n_non_stop_words              1    6.8897 8120.9 -237.83
    ## - global_subjectivity           1    8.7377 8122.7 -235.91
    ## - LDA_02                        1   10.0752 8124.1 -234.52
    ## - n_tokens_title                1   12.1432 8126.1 -232.38
    ## - num_hrefs                     1   14.2357 8128.2 -230.21
    ## - kw_min_avg                    1   16.0916 8130.1 -228.28
    ## - kw_max_avg                    1   16.6591 8130.6 -227.69
    ## - num_imgs                      1   26.4594 8140.4 -217.54
    ## - average_token_length          1   30.4708 8144.5 -213.39
    ## - kw_avg_avg                    1   30.7102 8144.7 -213.14
    ## 
    ## Step:  AIC=-244.9
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_imgs + num_videos + average_token_length + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_max_max + kw_avg_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + self_reference_avg_sharess + 
    ##     LDA_01 + LDA_02 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_positive_words + global_rate_negative_words + 
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     max_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_subjectivity + title_sentiment_polarity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - max_positive_polarity         1    0.0773 8114.1 -246.82
    ## - LDA_01                        1    0.1113 8114.2 -246.78
    ## - global_rate_positive_words    1    0.1224 8114.2 -246.77
    ## - global_rate_negative_words    1    0.1601 8114.2 -246.73
    ## - title_subjectivity            1    0.2293 8114.3 -246.66
    ## - self_reference_min_shares     1    0.4408 8114.5 -246.44
    ## - avg_positive_polarity         1    0.5057 8114.6 -246.38
    ## - kw_max_max                    1    0.5904 8114.6 -246.29
    ## - min_negative_polarity         1    0.6802 8114.7 -246.19
    ## - min_positive_polarity         1    0.7257 8114.8 -246.15
    ## - n_non_stop_unique_tokens      1    0.8076 8114.9 -246.06
    ## - avg_negative_polarity         1    1.0560 8115.1 -245.80
    ## - rate_positive_words           1    1.0912 8115.1 -245.77
    ## - n_unique_tokens               1    1.2965 8115.4 -245.55
    ## - global_sentiment_polarity     1    1.3735 8115.4 -245.47
    ## - num_videos                    1    1.5598 8115.6 -245.28
    ## - self_reference_avg_sharess    1    1.5617 8115.6 -245.28
    ## - abs_title_sentiment_polarity  1    1.6511 8115.7 -245.19
    ## - kw_min_min                    1    1.6742 8115.7 -245.16
    ## <none>                                      8114.1 -244.90
    ## - kw_avg_max                    1    2.2447 8116.3 -244.57
    ## - LDA_03                        1    2.3925 8116.5 -244.42
    ## - n_tokens_content              1    2.8339 8116.9 -243.96
    ## - title_sentiment_polarity      1    3.5354 8117.6 -243.23
    ## - kw_avg_min                    1    4.2618 8118.3 -242.48
    ## - kw_max_min                    1    4.7966 8118.9 -241.92
    ## - max_negative_polarity         1    5.5900 8119.6 -241.10
    ## - n_non_stop_words              1    6.9088 8121.0 -239.73
    ## - global_subjectivity           1    8.7628 8122.8 -237.81
    ## - LDA_02                        1   10.0918 8124.1 -236.43
    ## - n_tokens_title                1   12.1692 8126.2 -234.27
    ## - num_hrefs                     1   14.1746 8128.2 -232.19
    ## - kw_min_avg                    1   16.0924 8130.2 -230.20
    ## - kw_max_avg                    1   16.6736 8130.7 -229.60
    ## - num_imgs                      1   26.3886 8140.4 -219.54
    ## - average_token_length          1   30.4935 8144.6 -215.29
    ## - kw_avg_avg                    1   30.6970 8144.8 -215.08
    ## 
    ## Step:  AIC=-246.82
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_imgs + num_videos + average_token_length + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_max_max + kw_avg_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + self_reference_avg_sharess + 
    ##     LDA_01 + LDA_02 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_positive_words + global_rate_negative_words + 
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_subjectivity + title_sentiment_polarity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - LDA_01                        1    0.1086 8114.2 -248.71
    ## - global_rate_positive_words    1    0.1466 8114.3 -248.67
    ## - global_rate_negative_words    1    0.1717 8114.3 -248.64
    ## - title_subjectivity            1    0.2243 8114.4 -248.59
    ## - avg_positive_polarity         1    0.4325 8114.6 -248.37
    ## - self_reference_min_shares     1    0.4401 8114.6 -248.36
    ## - kw_max_max                    1    0.5916 8114.7 -248.21
    ## - min_positive_polarity         1    0.6563 8114.8 -248.14
    ## - min_negative_polarity         1    0.6942 8114.8 -248.10
    ## - n_non_stop_unique_tokens      1    0.7853 8114.9 -248.00
    ## - avg_negative_polarity         1    1.0451 8115.2 -247.74
    ## - rate_positive_words           1    1.1398 8115.3 -247.64
    ## - n_unique_tokens               1    1.2506 8115.4 -247.52
    ## - global_sentiment_polarity     1    1.3721 8115.5 -247.40
    ## - self_reference_avg_sharess    1    1.5573 8115.7 -247.20
    ## - num_videos                    1    1.5749 8115.7 -247.19
    ## - abs_title_sentiment_polarity  1    1.6465 8115.8 -247.11
    ## - kw_min_min                    1    1.6751 8115.8 -247.08
    ## <none>                                      8114.1 -246.82
    ## - kw_avg_max                    1    2.2392 8116.4 -246.50
    ## - LDA_03                        1    2.3822 8116.5 -246.35
    ## - n_tokens_content              1    2.7659 8116.9 -245.95
    ## - title_sentiment_polarity      1    3.5243 8117.7 -245.16
    ## - kw_avg_min                    1    4.2338 8118.4 -244.43
    ## - kw_max_min                    1    4.7740 8118.9 -243.86
    ## - max_negative_polarity         1    5.5490 8119.7 -243.06
    ## - n_non_stop_words              1    7.0013 8121.1 -241.55
    ## - global_subjectivity           1    8.7042 8122.8 -239.78
    ## - LDA_02                        1   10.1172 8124.3 -238.32
    ## - n_tokens_title                1   12.1746 8126.3 -236.19
    ## - num_hrefs                     1   14.3924 8128.5 -233.89
    ## - kw_min_avg                    1   16.0961 8130.2 -232.12
    ## - kw_max_avg                    1   16.7331 8130.9 -231.46
    ## - num_imgs                      1   26.4889 8140.6 -221.35
    ## - average_token_length          1   30.6716 8144.8 -217.03
    ## - kw_avg_avg                    1   30.7425 8144.9 -216.95
    ## 
    ## Step:  AIC=-248.71
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_imgs + num_videos + average_token_length + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_max_max + kw_avg_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + self_reference_avg_sharess + 
    ##     LDA_02 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_positive_words + global_rate_negative_words + 
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_subjectivity + title_sentiment_polarity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - global_rate_positive_words    1    0.1447 8114.4 -250.56
    ## - global_rate_negative_words    1    0.1621 8114.4 -250.54
    ## - title_subjectivity            1    0.2215 8114.5 -250.48
    ## - avg_positive_polarity         1    0.4212 8114.7 -250.27
    ## - self_reference_min_shares     1    0.4473 8114.7 -250.24
    ## - kw_max_max                    1    0.5952 8114.8 -250.09
    ## - min_positive_polarity         1    0.6465 8114.9 -250.04
    ## - min_negative_polarity         1    0.6906 8114.9 -249.99
    ## - n_non_stop_unique_tokens      1    0.7935 8115.0 -249.88
    ## - avg_negative_polarity         1    1.0264 8115.3 -249.64
    ## - rate_positive_words           1    1.1298 8115.4 -249.53
    ## - n_unique_tokens               1    1.2736 8115.5 -249.38
    ## - global_sentiment_polarity     1    1.3810 8115.6 -249.27
    ## - self_reference_avg_sharess    1    1.5681 8115.8 -249.08
    ## - num_videos                    1    1.5691 8115.8 -249.08
    ## - abs_title_sentiment_polarity  1    1.6432 8115.9 -249.00
    ## - kw_min_min                    1    1.6997 8115.9 -248.94
    ## <none>                                      8114.2 -248.71
    ## - kw_avg_max                    1    2.1929 8116.4 -248.43
    ## - LDA_03                        1    2.2890 8116.5 -248.33
    ## - n_tokens_content              1    2.7511 8117.0 -247.85
    ## - title_sentiment_polarity      1    3.5520 8117.8 -247.02
    ## - kw_avg_min                    1    4.2041 8118.4 -246.34
    ## - kw_max_min                    1    4.7372 8119.0 -245.79
    ## - max_negative_polarity         1    5.5089 8119.8 -244.99
    ## - n_non_stop_words              1    7.0568 8121.3 -243.38
    ## - global_subjectivity           1    8.7261 8123.0 -241.65
    ## - LDA_02                        1   11.7577 8126.0 -238.51
    ## - n_tokens_title                1   12.2062 8126.5 -238.04
    ## - num_hrefs                     1   14.3613 8128.6 -235.81
    ## - kw_min_avg                    1   16.0024 8130.2 -234.10
    ## - kw_max_avg                    1   16.6307 8130.9 -233.45
    ## - num_imgs                      1   26.5429 8140.8 -223.19
    ## - kw_avg_avg                    1   30.6360 8144.9 -218.95
    ## - average_token_length          1   30.8923 8145.1 -218.69
    ## 
    ## Step:  AIC=-250.56
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_imgs + num_videos + average_token_length + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_max_max + kw_avg_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + self_reference_avg_sharess + 
    ##     LDA_02 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_subjectivity + title_sentiment_polarity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - title_subjectivity            1    0.2102 8114.6 -252.34
    ## - self_reference_min_shares     1    0.4526 8114.8 -252.09
    ## - kw_max_max                    1    0.5913 8115.0 -251.94
    ## - avg_positive_polarity         1    0.6149 8115.0 -251.92
    ## - min_positive_polarity         1    0.6340 8115.0 -251.90
    ## - min_negative_polarity         1    0.7120 8115.1 -251.82
    ## - n_non_stop_unique_tokens      1    0.7704 8115.2 -251.76
    ## - avg_negative_polarity         1    0.9295 8115.3 -251.59
    ## - global_rate_negative_words    1    0.9655 8115.4 -251.56
    ## - global_sentiment_polarity     1    1.2470 8115.6 -251.26
    ## - n_unique_tokens               1    1.2757 8115.7 -251.23
    ## - self_reference_avg_sharess    1    1.5541 8115.9 -250.94
    ## - num_videos                    1    1.5614 8115.9 -250.94
    ## - abs_title_sentiment_polarity  1    1.6419 8116.0 -250.85
    ## - kw_min_min                    1    1.7101 8116.1 -250.78
    ## <none>                                      8114.4 -250.56
    ## - LDA_03                        1    2.2403 8116.6 -250.23
    ## - kw_avg_max                    1    2.2515 8116.6 -250.22
    ## - rate_positive_words           1    2.5840 8117.0 -249.87
    ## - n_tokens_content              1    2.7036 8117.1 -249.75
    ## - title_sentiment_polarity      1    3.6294 8118.0 -248.79
    ## - kw_avg_min                    1    4.1937 8118.6 -248.20
    ## - kw_max_min                    1    4.7289 8119.1 -247.65
    ## - max_negative_polarity         1    5.3805 8119.8 -246.97
    ## - n_non_stop_words              1    6.9332 8121.3 -245.36
    ## - global_subjectivity           1    8.7472 8123.1 -243.48
    ## - LDA_02                        1   11.8388 8126.2 -240.27
    ## - n_tokens_title                1   12.1096 8126.5 -239.99
    ## - num_hrefs                     1   14.2618 8128.7 -237.76
    ## - kw_min_avg                    1   15.9731 8130.4 -235.99
    ## - kw_max_avg                    1   16.7696 8131.2 -235.16
    ## - num_imgs                      1   26.4526 8140.8 -225.13
    ## - kw_avg_avg                    1   30.8901 8145.3 -220.54
    ## - average_token_length          1   31.0965 8145.5 -220.32
    ## 
    ## Step:  AIC=-252.34
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_imgs + num_videos + average_token_length + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_max_max + kw_avg_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + self_reference_avg_sharess + 
    ##     LDA_02 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_sentiment_polarity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - self_reference_min_shares     1    0.4616 8115.1 -253.86
    ## - avg_positive_polarity         1    0.5812 8115.2 -253.74
    ## - kw_max_max                    1    0.5877 8115.2 -253.73
    ## - min_positive_polarity         1    0.6144 8115.2 -253.70
    ## - min_negative_polarity         1    0.7310 8115.3 -253.58
    ## - n_non_stop_unique_tokens      1    0.7608 8115.4 -253.55
    ## - global_rate_negative_words    1    0.9059 8115.5 -253.40
    ## - avg_negative_polarity         1    0.9179 8115.5 -253.39
    ## - global_sentiment_polarity     1    1.2317 8115.8 -253.06
    ## - n_unique_tokens               1    1.2739 8115.9 -253.02
    ## - num_videos                    1    1.5345 8116.1 -252.75
    ## - self_reference_avg_sharess    1    1.5778 8116.2 -252.70
    ## - kw_min_min                    1    1.7118 8116.3 -252.56
    ## - abs_title_sentiment_polarity  1    1.8880 8116.5 -252.38
    ## <none>                                      8114.6 -252.34
    ## - LDA_03                        1    2.2386 8116.8 -252.01
    ## - kw_avg_max                    1    2.2688 8116.9 -251.98
    ## - rate_positive_words           1    2.5333 8117.1 -251.71
    ## - n_tokens_content              1    2.7152 8117.3 -251.52
    ## - title_sentiment_polarity      1    3.6627 8118.3 -250.54
    ## - kw_avg_min                    1    4.2179 8118.8 -249.96
    ## - kw_max_min                    1    4.7476 8119.3 -249.41
    ## - max_negative_polarity         1    5.3719 8120.0 -248.76
    ## - n_non_stop_words              1    6.9707 8121.6 -247.10
    ## - global_subjectivity           1    8.5473 8123.1 -245.47
    ## - LDA_02                        1   11.8633 8126.5 -242.03
    ## - n_tokens_title                1   11.9734 8126.6 -241.91
    ## - num_hrefs                     1   14.2316 8128.8 -239.57
    ## - kw_min_avg                    1   15.9727 8130.6 -237.77
    ## - kw_max_avg                    1   16.8331 8131.4 -236.88
    ## - num_imgs                      1   26.4727 8141.1 -226.89
    ## - kw_avg_avg                    1   30.9784 8145.6 -222.23
    ## - average_token_length          1   31.0540 8145.7 -222.15
    ## 
    ## Step:  AIC=-253.86
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_imgs + num_videos + average_token_length + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_max_max + kw_avg_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_avg_sharess + LDA_02 + 
    ##     LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_sentiment_polarity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - avg_positive_polarity         1    0.5589 8115.6 -255.28
    ## - min_positive_polarity         1    0.5871 8115.6 -255.25
    ## - kw_max_max                    1    0.5940 8115.7 -255.24
    ## - min_negative_polarity         1    0.7371 8115.8 -255.09
    ## - n_non_stop_unique_tokens      1    0.7455 8115.8 -255.09
    ## - global_rate_negative_words    1    0.9053 8116.0 -254.92
    ## - avg_negative_polarity         1    0.9163 8116.0 -254.91
    ## - global_sentiment_polarity     1    1.2350 8116.3 -254.58
    ## - n_unique_tokens               1    1.2545 8116.3 -254.56
    ## - num_videos                    1    1.5727 8116.6 -254.23
    ## - self_reference_avg_sharess    1    1.6314 8116.7 -254.17
    ## - kw_min_min                    1    1.7075 8116.8 -254.09
    ## - abs_title_sentiment_polarity  1    1.9092 8117.0 -253.88
    ## <none>                                      8115.1 -253.86
    ## - LDA_03                        1    2.2491 8117.3 -253.53
    ## - kw_avg_max                    1    2.3212 8117.4 -253.45
    ## - rate_positive_words           1    2.5444 8117.6 -253.22
    ## - n_tokens_content              1    2.7345 8117.8 -253.02
    ## - title_sentiment_polarity      1    3.6976 8118.8 -252.02
    ## - kw_avg_min                    1    4.2064 8119.3 -251.49
    ## - kw_max_min                    1    4.7386 8119.8 -250.94
    ## - max_negative_polarity         1    5.3970 8120.5 -250.26
    ## - n_non_stop_words              1    6.9009 8122.0 -248.70
    ## - global_subjectivity           1    8.5004 8123.6 -247.04
    ## - LDA_02                        1   11.7509 8126.8 -243.67
    ## - n_tokens_title                1   11.9791 8127.0 -243.43
    ## - num_hrefs                     1   14.3616 8129.4 -240.96
    ## - kw_min_avg                    1   15.8828 8130.9 -239.38
    ## - kw_max_avg                    1   16.8644 8131.9 -238.37
    ## - num_imgs                      1   26.5493 8141.6 -228.34
    ## - average_token_length          1   30.9007 8146.0 -223.83
    ## - kw_avg_avg                    1   31.1259 8146.2 -223.60
    ## 
    ## Step:  AIC=-255.28
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_imgs + num_videos + average_token_length + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_max_max + kw_avg_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_avg_sharess + LDA_02 + 
    ##     LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_negative_words + rate_positive_words + min_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - min_positive_polarity         1    0.2388 8115.9 -257.03
    ## - kw_max_max                    1    0.5788 8116.2 -256.68
    ## - min_negative_polarity         1    0.7306 8116.3 -256.52
    ## - global_rate_negative_words    1    0.8309 8116.5 -256.42
    ## - n_non_stop_unique_tokens      1    0.8934 8116.5 -256.35
    ## - avg_negative_polarity         1    1.4051 8117.0 -255.82
    ## - n_unique_tokens               1    1.4205 8117.0 -255.81
    ## - num_videos                    1    1.5195 8117.1 -255.70
    ## - self_reference_avg_sharess    1    1.6369 8117.3 -255.58
    ## - kw_min_min                    1    1.6905 8117.3 -255.52
    ## - abs_title_sentiment_polarity  1    1.8498 8117.5 -255.36
    ## <none>                                      8115.6 -255.28
    ## - LDA_03                        1    2.2226 8117.8 -254.97
    ## - kw_avg_max                    1    2.2859 8117.9 -254.91
    ## - n_tokens_content              1    2.7879 8118.4 -254.38
    ## - title_sentiment_polarity      1    3.7265 8119.3 -253.41
    ## - rate_positive_words           1    3.9873 8119.6 -253.14
    ## - kw_avg_min                    1    4.1695 8119.8 -252.95
    ## - global_sentiment_polarity     1    4.4888 8120.1 -252.62
    ## - kw_max_min                    1    4.6951 8120.3 -252.41
    ## - max_negative_polarity         1    5.9539 8121.6 -251.10
    ## - n_non_stop_words              1    6.4542 8122.1 -250.58
    ## - global_subjectivity           1    8.3486 8124.0 -248.62
    ## - LDA_02                        1   11.6891 8127.3 -245.15
    ## - n_tokens_title                1   11.9900 8127.6 -244.84
    ## - num_hrefs                     1   14.1747 8129.8 -242.57
    ## - kw_min_avg                    1   15.8789 8131.5 -240.81
    ## - kw_max_avg                    1   16.8560 8132.5 -239.79
    ## - num_imgs                      1   26.3512 8142.0 -229.96
    ## - kw_avg_avg                    1   31.0643 8146.7 -225.09
    ## - average_token_length          1   31.4885 8147.1 -224.65
    ## 
    ## Step:  AIC=-257.03
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_imgs + num_videos + average_token_length + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_max_max + kw_avg_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_avg_sharess + LDA_02 + 
    ##     LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_negative_words + rate_positive_words + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_sentiment_polarity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - kw_max_max                    1     0.586 8116.4 -258.42
    ## - global_rate_negative_words    1     0.695 8116.6 -258.31
    ## - min_negative_polarity         1     0.707 8116.6 -258.30
    ## - n_non_stop_unique_tokens      1     1.028 8116.9 -257.96
    ## - avg_negative_polarity         1     1.393 8117.3 -257.59
    ## - num_videos                    1     1.518 8117.4 -257.45
    ## - self_reference_avg_sharess    1     1.607 8117.5 -257.36
    ## - kw_min_min                    1     1.698 8117.6 -257.27
    ## - n_unique_tokens               1     1.718 8117.6 -257.25
    ## - abs_title_sentiment_polarity  1     1.848 8117.7 -257.11
    ## <none>                                      8115.9 -257.03
    ## - LDA_03                        1     2.261 8118.1 -256.68
    ## - kw_avg_max                    1     2.313 8118.2 -256.63
    ## - n_tokens_content              1     2.793 8118.7 -256.13
    ## - title_sentiment_polarity      1     3.721 8119.6 -255.17
    ## - rate_positive_words           1     3.782 8119.6 -255.11
    ## - kw_avg_min                    1     4.208 8120.1 -254.66
    ## - global_sentiment_polarity     1     4.257 8120.1 -254.61
    ## - kw_max_min                    1     4.743 8120.6 -254.11
    ## - max_negative_polarity         1     5.994 8121.9 -252.81
    ## - n_non_stop_words              1     7.264 8123.1 -251.49
    ## - global_subjectivity           1     8.620 8124.5 -250.09
    ## - LDA_02                        1    11.645 8127.5 -246.95
    ## - n_tokens_title                1    11.988 8127.8 -246.59
    ## - num_hrefs                     1    13.947 8129.8 -244.56
    ## - kw_min_avg                    1    15.893 8131.8 -242.55
    ## - kw_max_avg                    1    16.928 8132.8 -241.47
    ## - num_imgs                      1    26.308 8142.2 -231.76
    ## - kw_avg_avg                    1    31.098 8147.0 -226.80
    ## - average_token_length          1    32.318 8148.2 -225.54
    ## 
    ## Step:  AIC=-258.42
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_imgs + num_videos + average_token_length + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_avg_sharess + LDA_02 + LDA_03 + 
    ##     global_subjectivity + global_sentiment_polarity + global_rate_negative_words + 
    ##     rate_positive_words + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_sentiment_polarity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - global_rate_negative_words    1     0.670 8117.1 -259.73
    ## - min_negative_polarity         1     0.738 8117.2 -259.66
    ## - n_non_stop_unique_tokens      1     0.964 8117.4 -259.42
    ## - kw_min_min                    1     1.333 8117.8 -259.04
    ## - avg_negative_polarity         1     1.435 8117.9 -258.93
    ## - num_videos                    1     1.542 8118.0 -258.82
    ## - n_unique_tokens               1     1.608 8118.1 -258.75
    ## - self_reference_avg_sharess    1     1.628 8118.1 -258.73
    ## - abs_title_sentiment_polarity  1     1.843 8118.3 -258.51
    ## - kw_avg_max                    1     1.856 8118.3 -258.50
    ## <none>                                      8116.4 -258.42
    ## - LDA_03                        1     1.989 8118.4 -258.36
    ## - n_tokens_content              1     2.838 8119.3 -257.48
    ## - rate_positive_words           1     3.716 8120.2 -256.57
    ## - title_sentiment_polarity      1     3.746 8120.2 -256.54
    ## - global_sentiment_polarity     1     4.252 8120.7 -256.01
    ## - kw_avg_min                    1     4.437 8120.9 -255.82
    ## - kw_max_min                    1     4.965 8121.4 -255.27
    ## - max_negative_polarity         1     6.034 8122.5 -254.16
    ## - n_non_stop_words              1     7.244 8123.7 -252.91
    ## - global_subjectivity           1     8.636 8125.1 -251.46
    ## - n_tokens_title                1    12.181 8128.6 -247.78
    ## - LDA_02                        1    12.201 8128.6 -247.76
    ## - num_hrefs                     1    14.044 8130.5 -245.85
    ## - kw_min_avg                    1    16.689 8133.1 -243.11
    ## - kw_max_avg                    1    17.136 8133.6 -242.65
    ## - num_imgs                      1    26.682 8143.1 -232.77
    ## - kw_avg_avg                    1    31.757 8148.2 -227.52
    ## - average_token_length          1    32.132 8148.6 -227.13
    ## 
    ## Step:  AIC=-259.73
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_imgs + num_videos + average_token_length + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_avg_sharess + LDA_02 + LDA_03 + 
    ##     global_subjectivity + global_sentiment_polarity + rate_positive_words + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - n_non_stop_unique_tokens      1     0.919 8118.0 -260.77
    ## - min_negative_polarity         1     1.120 8118.2 -260.56
    ## - kw_min_min                    1     1.401 8118.5 -260.27
    ## - num_videos                    1     1.489 8118.6 -260.18
    ## - self_reference_avg_sharess    1     1.606 8118.7 -260.06
    ## - n_unique_tokens               1     1.632 8118.7 -260.03
    ## - avg_negative_polarity         1     1.777 8118.9 -259.88
    ## <none>                                      8117.1 -259.73
    ## - LDA_03                        1     1.984 8119.1 -259.67
    ## - kw_avg_max                    1     2.006 8119.1 -259.65
    ## - abs_title_sentiment_polarity  1     2.046 8119.2 -259.60
    ## - n_tokens_content              1     2.736 8119.8 -258.89
    ## - rate_positive_words           1     3.185 8120.3 -258.42
    ## - title_sentiment_polarity      1     3.681 8120.8 -257.91
    ## - global_sentiment_polarity     1     4.139 8121.3 -257.43
    ## - kw_avg_min                    1     4.431 8121.5 -257.13
    ## - kw_max_min                    1     4.973 8122.1 -256.57
    ## - max_negative_polarity         1     5.889 8123.0 -255.62
    ## - n_non_stop_words              1     8.831 8125.9 -252.56
    ## - global_subjectivity           1    10.811 8127.9 -250.51
    ## - n_tokens_title                1    12.093 8129.2 -249.18
    ## - LDA_02                        1    12.108 8129.2 -249.17
    ## - num_hrefs                     1    13.642 8130.8 -247.58
    ## - kw_min_avg                    1    17.042 8134.2 -244.05
    ## - kw_max_avg                    1    17.600 8134.7 -243.47
    ## - num_imgs                      1    26.241 8143.4 -234.53
    ## - average_token_length          1    32.604 8149.7 -227.95
    ## - kw_avg_avg                    1    32.639 8149.8 -227.91
    ## 
    ## Step:  AIC=-260.77
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + num_hrefs + num_imgs + num_videos + average_token_length + 
    ##     kw_min_min + kw_max_min + kw_avg_min + kw_avg_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_avg_sharess + LDA_02 + 
    ##     LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     rate_positive_words + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_sentiment_polarity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - n_unique_tokens               1     0.779 8118.8 -261.96
    ## - min_negative_polarity         1     0.962 8119.0 -261.77
    ## - self_reference_avg_sharess    1     1.569 8119.6 -261.14
    ## - kw_min_min                    1     1.575 8119.6 -261.14
    ## - num_videos                    1     1.623 8119.7 -261.09
    ## - kw_avg_max                    1     1.843 8119.9 -260.86
    ## - avg_negative_polarity         1     1.886 8119.9 -260.82
    ## <none>                                      8118.0 -260.77
    ## - abs_title_sentiment_polarity  1     2.123 8120.2 -260.57
    ## - LDA_03                        1     2.164 8120.2 -260.53
    ## - rate_positive_words           1     3.276 8121.3 -259.37
    ## - n_tokens_content              1     3.687 8121.7 -258.95
    ## - title_sentiment_polarity      1     3.704 8121.7 -258.93
    ## - kw_avg_min                    1     4.352 8122.4 -258.26
    ## - global_sentiment_polarity     1     4.367 8122.4 -258.24
    ## - kw_max_min                    1     4.901 8122.9 -257.69
    ## - max_negative_polarity         1     6.406 8124.4 -256.13
    ## - n_non_stop_words              1     7.927 8126.0 -254.55
    ## - global_subjectivity           1    10.730 8128.8 -251.64
    ## - LDA_02                        1    11.800 8129.8 -250.53
    ## - n_tokens_title                1    12.247 8130.3 -250.07
    ## - num_hrefs                     1    16.118 8134.2 -246.06
    ## - kw_min_avg                    1    17.122 8135.2 -245.02
    ## - kw_max_avg                    1    17.687 8135.7 -244.43
    ## - num_imgs                      1    31.469 8149.5 -230.17
    ## - average_token_length          1    32.160 8150.2 -229.46
    ## - kw_avg_avg                    1    32.757 8150.8 -228.84
    ## 
    ## Step:  AIC=-261.96
    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_words + 
    ##     num_hrefs + num_imgs + num_videos + average_token_length + 
    ##     kw_min_min + kw_max_min + kw_avg_min + kw_avg_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_avg_sharess + LDA_02 + 
    ##     LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     rate_positive_words + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_sentiment_polarity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - min_negative_polarity         1     0.864 8119.7 -263.07
    ## - self_reference_avg_sharess    1     1.608 8120.4 -262.30
    ## - kw_min_min                    1     1.730 8120.5 -262.17
    ## - kw_avg_max                    1     1.783 8120.6 -262.12
    ## - num_videos                    1     1.801 8120.6 -262.10
    ## - avg_negative_polarity         1     1.906 8120.7 -261.99
    ## <none>                                      8118.8 -261.96
    ## - abs_title_sentiment_polarity  1     2.135 8120.9 -261.75
    ## - LDA_03                        1     2.262 8121.1 -261.62
    ## - rate_positive_words           1     3.310 8122.1 -260.53
    ## - title_sentiment_polarity      1     3.696 8122.5 -260.13
    ## - kw_avg_min                    1     4.382 8123.2 -259.42
    ## - global_sentiment_polarity     1     4.422 8123.2 -259.38
    ## - kw_max_min                    1     4.937 8123.7 -258.84
    ## - max_negative_polarity         1     6.805 8125.6 -256.90
    ## - n_tokens_content              1    10.952 8129.8 -252.60
    ## - global_subjectivity           1    11.175 8130.0 -252.37
    ## - n_non_stop_words              1    11.451 8130.3 -252.09
    ## - n_tokens_title                1    11.910 8130.7 -251.61
    ## - LDA_02                        1    12.185 8131.0 -251.33
    ## - num_hrefs                     1    16.146 8135.0 -247.22
    ## - kw_min_avg                    1    17.230 8136.0 -246.10
    ## - kw_max_avg                    1    17.835 8136.6 -245.47
    ## - num_imgs                      1    30.759 8149.6 -232.10
    ## - average_token_length          1    31.612 8150.4 -231.22
    ## - kw_avg_avg                    1    32.939 8151.8 -229.84
    ## 
    ## Step:  AIC=-263.07
    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_words + 
    ##     num_hrefs + num_imgs + num_videos + average_token_length + 
    ##     kw_min_min + kw_max_min + kw_avg_min + kw_avg_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_avg_sharess + LDA_02 + 
    ##     LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     rate_positive_words + avg_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - avg_negative_polarity         1     1.050 8120.7 -263.98
    ## - kw_min_min                    1     1.591 8121.3 -263.42
    ## - self_reference_avg_sharess    1     1.638 8121.3 -263.37
    ## - num_videos                    1     1.809 8121.5 -263.19
    ## - kw_avg_max                    1     1.903 8121.6 -263.09
    ## <none>                                      8119.7 -263.07
    ## - abs_title_sentiment_polarity  1     2.119 8121.8 -262.87
    ## - LDA_03                        1     2.258 8121.9 -262.73
    ## - rate_positive_words           1     2.841 8122.5 -262.12
    ## - title_sentiment_polarity      1     3.799 8123.5 -261.13
    ## - kw_avg_min                    1     4.306 8124.0 -260.60
    ## - global_sentiment_polarity     1     4.414 8124.1 -260.49
    ## - kw_max_min                    1     4.864 8124.5 -260.02
    ## - max_negative_polarity         1     5.971 8125.6 -258.87
    ## - n_tokens_content              1    10.151 8129.8 -254.54
    ## - global_subjectivity           1    11.561 8131.2 -253.08
    ## - n_non_stop_words              1    11.906 8131.6 -252.72
    ## - LDA_02                        1    11.971 8131.6 -252.65
    ## - n_tokens_title                1    11.977 8131.7 -252.65
    ## - num_hrefs                     1    16.194 8135.9 -248.28
    ## - kw_min_avg                    1    17.215 8136.9 -247.22
    ## - kw_max_avg                    1    17.854 8137.5 -246.56
    ## - num_imgs                      1    30.710 8150.4 -233.26
    ## - average_token_length          1    31.709 8151.4 -232.22
    ## - kw_avg_avg                    1    32.887 8152.6 -231.00
    ## 
    ## Step:  AIC=-263.98
    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_words + 
    ##     num_hrefs + num_imgs + num_videos + average_token_length + 
    ##     kw_min_min + kw_max_min + kw_avg_min + kw_avg_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_avg_sharess + LDA_02 + 
    ##     LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     rate_positive_words + max_negative_polarity + title_sentiment_polarity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - kw_min_min                    1     1.604 8122.3 -264.31
    ## - self_reference_avg_sharess    1     1.732 8122.5 -264.18
    ## - num_videos                    1     1.778 8122.5 -264.13
    ## - abs_title_sentiment_polarity  1     1.918 8122.6 -263.99
    ## <none>                                      8120.7 -263.98
    ## - kw_avg_max                    1     1.977 8122.7 -263.93
    ## - LDA_03                        1     2.202 8122.9 -263.69
    ## - rate_positive_words           1     2.440 8123.2 -263.45
    ## - global_sentiment_polarity     1     3.391 8124.1 -262.46
    ## - title_sentiment_polarity      1     3.963 8124.7 -261.87
    ## - kw_avg_min                    1     4.335 8125.1 -261.48
    ## - kw_max_min                    1     4.893 8125.6 -260.90
    ## - max_negative_polarity         1     5.073 8125.8 -260.72
    ## - global_subjectivity           1    10.689 8131.4 -254.89
    ## - n_tokens_content              1    11.601 8132.3 -253.95
    ## - LDA_02                        1    11.668 8132.4 -253.88
    ## - n_non_stop_words              1    11.738 8132.5 -253.81
    ## - n_tokens_title                1    12.011 8132.7 -253.52
    ## - num_hrefs                     1    15.623 8136.3 -249.78
    ## - kw_min_avg                    1    17.373 8138.1 -247.97
    ## - kw_max_avg                    1    18.053 8138.8 -247.27
    ## - num_imgs                      1    30.751 8151.5 -234.13
    ## - average_token_length          1    31.208 8151.9 -233.66
    ## - kw_avg_avg                    1    33.267 8154.0 -231.53
    ## 
    ## Step:  AIC=-264.31
    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_words + 
    ##     num_hrefs + num_imgs + num_videos + average_token_length + 
    ##     kw_max_min + kw_avg_min + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_avg_sharess + LDA_02 + LDA_03 + 
    ##     global_subjectivity + global_sentiment_polarity + rate_positive_words + 
    ##     max_negative_polarity + title_sentiment_polarity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - self_reference_avg_sharess    1     1.778 8124.1 -264.47
    ## - num_videos                    1     1.796 8124.1 -264.45
    ## - abs_title_sentiment_polarity  1     1.870 8124.2 -264.38
    ## <none>                                      8122.3 -264.31
    ## - rate_positive_words           1     2.596 8124.9 -263.62
    ## - LDA_03                        1     3.068 8125.4 -263.13
    ## - kw_avg_min                    1     3.392 8125.7 -262.80
    ## - global_sentiment_polarity     1     3.447 8125.8 -262.74
    ## - kw_max_min                    1     3.965 8126.3 -262.20
    ## - title_sentiment_polarity      1     4.180 8126.5 -261.98
    ## - kw_avg_max                    1     4.206 8126.5 -261.95
    ## - max_negative_polarity         1     5.195 8127.5 -260.93
    ## - global_subjectivity           1    10.640 8133.0 -255.28
    ## - LDA_02                        1    10.894 8133.2 -255.02
    ## - n_tokens_title                1    11.514 8133.8 -254.38
    ## - n_non_stop_words              1    11.961 8134.3 -253.91
    ## - n_tokens_content              1    12.130 8134.5 -253.74
    ## - num_hrefs                     1    15.397 8137.7 -250.35
    ## - kw_min_avg                    1    16.530 8138.9 -249.18
    ## - kw_max_avg                    1    17.827 8140.2 -247.84
    ## - num_imgs                      1    30.153 8152.5 -235.09
    ## - average_token_length          1    31.559 8153.9 -233.64
    ## - kw_avg_avg                    1    32.683 8155.0 -232.47
    ## 
    ## Step:  AIC=-264.47
    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_words + 
    ##     num_hrefs + num_imgs + num_videos + average_token_length + 
    ##     kw_max_min + kw_avg_min + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + LDA_02 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     rate_positive_words + max_negative_polarity + title_sentiment_polarity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - abs_title_sentiment_polarity  1     1.832 8125.9 -264.57
    ## - num_videos                    1     1.872 8126.0 -264.53
    ## <none>                                      8124.1 -264.47
    ## - rate_positive_words           1     2.529 8126.6 -263.85
    ## - LDA_03                        1     3.107 8127.2 -263.25
    ## - kw_avg_min                    1     3.349 8127.5 -263.00
    ## - global_sentiment_polarity     1     3.415 8127.5 -262.93
    ## - kw_max_min                    1     3.889 8128.0 -262.44
    ## - kw_avg_max                    1     4.106 8128.2 -262.21
    ## - title_sentiment_polarity      1     4.225 8128.3 -262.09
    ## - max_negative_polarity         1     5.205 8129.3 -261.07
    ## - global_subjectivity           1    10.900 8135.0 -255.17
    ## - LDA_02                        1    11.121 8135.2 -254.94
    ## - n_tokens_title                1    11.696 8135.8 -254.35
    ## - n_non_stop_words              1    12.211 8136.3 -253.81
    ## - n_tokens_content              1    12.392 8136.5 -253.63
    ## - num_hrefs                     1    15.350 8139.5 -250.56
    ## - kw_min_avg                    1    16.573 8140.7 -249.30
    ## - kw_max_avg                    1    17.821 8141.9 -248.00
    ## - num_imgs                      1    30.728 8154.8 -234.66
    ## - average_token_length          1    31.790 8155.9 -233.56
    ## - kw_avg_avg                    1    33.173 8157.3 -232.13
    ## 
    ## Step:  AIC=-264.57
    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_words + 
    ##     num_hrefs + num_imgs + num_videos + average_token_length + 
    ##     kw_max_min + kw_avg_min + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + LDA_02 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     rate_positive_words + max_negative_polarity + title_sentiment_polarity
    ## 
    ##                             Df Sum of Sq    RSS     AIC
    ## <none>                                   8125.9 -264.57
    ## - num_videos                 1     1.959 8127.9 -264.54
    ## - rate_positive_words        1     2.408 8128.3 -264.07
    ## - LDA_03                     1     3.249 8129.2 -263.20
    ## - kw_avg_min                 1     3.253 8129.2 -263.20
    ## - global_sentiment_polarity  1     3.371 8129.3 -263.07
    ## - kw_max_min                 1     3.803 8129.7 -262.63
    ## - kw_avg_max                 1     4.030 8130.0 -262.39
    ## - max_negative_polarity      1     5.167 8131.1 -261.21
    ## - title_sentiment_polarity   1     5.345 8131.3 -261.03
    ## - LDA_02                     1    11.360 8137.3 -254.80
    ## - n_non_stop_words           1    12.044 8138.0 -254.09
    ## - n_tokens_title             1    12.114 8138.1 -254.02
    ## - global_subjectivity        1    12.148 8138.1 -253.98
    ## - n_tokens_content           1    12.347 8138.3 -253.78
    ## - num_hrefs                  1    15.330 8141.3 -250.69
    ## - kw_min_avg                 1    16.549 8142.5 -249.43
    ## - kw_max_avg                 1    17.772 8143.7 -248.16
    ## - num_imgs                   1    31.260 8157.2 -234.22
    ## - average_token_length       1    31.793 8157.7 -233.66
    ## - kw_avg_avg                 1    33.171 8159.1 -232.24

``` r
lm$call[["formula"]] # model selected based on AIC. 
```

    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_words + 
    ##     num_hrefs + num_imgs + num_videos + average_token_length + 
    ##     kw_max_min + kw_avg_min + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + LDA_02 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     rate_positive_words + max_negative_polarity + title_sentiment_polarity
    ## <environment: 0x7f9b7eedc240>

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
    ##                 1305                 1356                  519                  567 
    ##  weekday_is_thursday   weekday_is_tuesday weekday_is_wednesday 
    ##                 1569                 1546                 1565

``` r
C2 <- pop.data %>% 
    group_by( weekday ) %>% 
    summarise( percent = 100 * n() / nrow( X ),mean_shares = mean(shares), mean_images = mean(num_imgs),mean_video = mean(num_videos),mean_link = mean(num_hrefs))
knitr::kable(C2)
```

| weekday                |  percent | mean\_shares | mean\_images | mean\_video | mean\_link |
| :--------------------- | -------: | -----------: | -----------: | ----------: | ---------: |
| weekday\_is\_friday    | 3.894360 |     2228.411 |     3.266667 |   0.5915709 |  10.329502 |
| weekday\_is\_monday    | 4.046553 |     2456.054 |     2.583333 |   0.5715339 |   9.942478 |
| weekday\_is\_saturday  | 1.548791 |     2760.202 |     2.736031 |   0.5549133 |  11.179191 |
| weekday\_is\_sunday    | 1.692032 |     2605.483 |     3.156966 |   0.5996473 |  10.894180 |
| weekday\_is\_thursday  | 4.682184 |     2394.008 |     2.773104 |   0.5003187 |  10.297642 |
| weekday\_is\_tuesday   | 4.613548 |     2220.135 |     2.718629 |   0.5595084 |  10.034929 |
| weekday\_is\_wednesday | 4.670248 |     1879.788 |     2.819808 |   0.5150160 |   9.778275 |

``` r
table(pop.data$weekday, pop.data$channel)
```

    ##                       
    ##                        data_channel_is_world
    ##   weekday_is_friday                     1305
    ##   weekday_is_monday                     1356
    ##   weekday_is_saturday                    519
    ##   weekday_is_sunday                      567
    ##   weekday_is_thursday                   1569
    ##   weekday_is_tuesday                    1546
    ##   weekday_is_wednesday                  1565

``` r
table(pop.data$channel, pop.data$is_weekend)
```

    ##                        
    ##                            0    1
    ##   data_channel_is_world 7341 1086

``` r
C3 <- pop.data %>% group_by(is_weekend)%>% 
    summarise( percent = 100 * n() / nrow( X ),mean_shares = mean(shares), mean_images = mean(num_imgs),mean_video = mean(num_videos),mean_link = mean(num_hrefs))
knitr::kable(C3)
```

| is\_weekend |   percent | mean\_shares | mean\_images | mean\_video | mean\_link |
| :---------- | --------: | -----------: | -----------: | ----------: | ---------: |
| 0           | 21.906893 |     2229.789 |     2.824275 |   0.5452936 |   10.07165 |
| 1           |  3.240824 |     2679.424 |     2.955801 |   0.5782689 |   11.03039 |

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
| TRUE        | FALSE              | FALSE               | FALSE                        | FALSE      | TRUE      | FALSE         | FALSE       | FALSE        | FALSE        | FALSE        | FALSE        | FALSE                        | FALSE                         | FALSE                         | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | FALSE              | FALSE               | FALSE                        | FALSE      | TRUE      | FALSE         | FALSE       | FALSE        | FALSE        | FALSE        | TRUE         | FALSE                        | FALSE                         | FALSE                         | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | FALSE              | FALSE               | FALSE                        | FALSE      | TRUE      | FALSE         | FALSE       | FALSE        | TRUE         | FALSE        | TRUE         | FALSE                        | FALSE                         | FALSE                         | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | FALSE              | FALSE               | FALSE                        | FALSE      | TRUE      | FALSE         | FALSE       | FALSE        | TRUE         | TRUE         | TRUE         | FALSE                        | FALSE                         | FALSE                         | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | FALSE              | FALSE               | FALSE                        | FALSE      | TRUE      | FALSE         | FALSE       | FALSE        | TRUE         | TRUE         | TRUE         | FALSE                        | FALSE                         | FALSE                         | FALSE                 | FALSE                    | TRUE                            |
| TRUE        | TRUE               | FALSE               | FALSE                        | TRUE       | TRUE      | FALSE         | FALSE       | FALSE        | TRUE         | TRUE         | TRUE         | FALSE                        | FALSE                         | FALSE                         | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | TRUE               | FALSE               | FALSE                        | TRUE       | TRUE      | FALSE         | FALSE       | FALSE        | TRUE         | TRUE         | TRUE         | FALSE                        | FALSE                         | FALSE                         | FALSE                 | FALSE                    | TRUE                            |
| TRUE        | TRUE               | FALSE               | FALSE                        | TRUE       | TRUE      | FALSE         | TRUE        | FALSE        | TRUE         | TRUE         | TRUE         | FALSE                        | FALSE                         | FALSE                         | FALSE                 | FALSE                    | TRUE                            |
| TRUE        | TRUE               | TRUE                | FALSE                        | TRUE       | TRUE      | FALSE         | TRUE        | FALSE        | TRUE         | TRUE         | TRUE         | FALSE                        | FALSE                         | TRUE                          | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | TRUE               | TRUE                | FALSE                        | TRUE       | TRUE      | FALSE         | TRUE        | FALSE        | TRUE         | TRUE         | TRUE         | FALSE                        | FALSE                         | TRUE                          | FALSE                 | FALSE                    | TRUE                            |
| TRUE        | TRUE               | TRUE                | FALSE                        | TRUE       | TRUE      | FALSE         | TRUE        | FALSE        | TRUE         | TRUE         | TRUE         | FALSE                        | TRUE                          | TRUE                          | FALSE                 | FALSE                    | TRUE                            |
| TRUE        | TRUE               | TRUE                | FALSE                        | TRUE       | TRUE      | FALSE         | TRUE        | TRUE         | TRUE         | TRUE         | TRUE         | FALSE                        | TRUE                          | TRUE                          | FALSE                 | FALSE                    | TRUE                            |
| TRUE        | TRUE               | TRUE                | TRUE                         | TRUE       | TRUE      | FALSE         | TRUE        | TRUE         | TRUE         | TRUE         | TRUE         | FALSE                        | TRUE                          | TRUE                          | FALSE                 | FALSE                    | TRUE                            |
| TRUE        | TRUE               | TRUE                | TRUE                         | TRUE       | TRUE      | TRUE          | TRUE        | TRUE         | TRUE         | TRUE         | TRUE         | FALSE                        | TRUE                          | TRUE                          | FALSE                 | FALSE                    | TRUE                            |
| TRUE        | TRUE               | TRUE                | TRUE                         | TRUE       | TRUE      | TRUE          | TRUE        | TRUE         | TRUE         | TRUE         | TRUE         | TRUE                         | TRUE                          | TRUE                          | FALSE                 | FALSE                    | TRUE                            |
| TRUE        | TRUE               | TRUE                | TRUE                         | TRUE       | TRUE      | TRUE          | TRUE        | TRUE         | TRUE         | TRUE         | TRUE         | TRUE                         | TRUE                          | TRUE                          | FALSE                 | TRUE                     | TRUE                            |
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
|     13 | 11 |   4 |

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
| lm.fit1 |     5420.317 |        0.0254268 |    1945.181 |
| lm.fit2 |     5589.535 |        0.0211355 |    1936.831 |
| lm.fit3 |     6647.994 |        0.0196451 |    2069.975 |

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
    ##                                         s0
    ## (Intercept)                   3125.3936329
    ## n_tokens_title                  13.0691315
    ## n_tokens_content                 .        
    ## n_unique_tokens                  .        
    ## n_non_stop_words                 .        
    ## n_non_stop_unique_tokens         .        
    ## num_hrefs                        .        
    ## num_self_hrefs                   .        
    ## num_imgs                        54.6083603
    ## num_videos                       .        
    ## average_token_length           -96.1354402
    ## num_keywords                     .        
    ## kw_min_min                       .        
    ## kw_max_min                       .        
    ## kw_avg_min                       .        
    ## kw_min_max                       .        
    ## kw_max_max                       .        
    ## kw_avg_max                       .        
    ## kw_min_avg                       .        
    ## kw_max_avg                       .        
    ## kw_avg_avg                       0.1007222
    ## self_reference_min_shares        .        
    ## self_reference_max_shares        .        
    ## self_reference_avg_sharess       .        
    ## is_weekend                       .        
    ## LDA_00                           .        
    ## LDA_01                           .        
    ## LDA_02                       -1924.9986989
    ## LDA_03                         848.6773659
    ## LDA_04                           .        
    ## global_subjectivity            618.0483521
    ## global_sentiment_polarity        .        
    ## global_rate_positive_words       .        
    ## global_rate_negative_words       .        
    ## rate_positive_words              .        
    ## rate_negative_words           -321.1512129
    ## avg_positive_polarity            .        
    ## min_positive_polarity            .        
    ## max_positive_polarity            .        
    ## avg_negative_polarity            .        
    ## min_negative_polarity            .        
    ## max_negative_polarity        -1510.5725762
    ## title_subjectivity               .        
    ## title_sentiment_polarity        45.2622908
    ## abs_title_subjectivity           .        
    ## abs_title_sentiment_polarity    20.6683321

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
    ## (Intercept)                  1673.0328633
    ## n_tokens_content                .        
    ## n_non_stop_words             -283.9494452
    ## n_non_stop_unique_tokens        .        
    ## num_hrefs                       4.8366362
    ## num_imgs                       80.2272371
    ## num_keywords                    .        
    ## num_videos                     16.3511866
    ## kw_avg_max                      .        
    ## kw_min_avg                      .        
    ## kw_max_avg                      .        
    ## kw_avg_avg                      0.2380648
    ## self_reference_min_shares       .        
    ## self_reference_avg_sharess      .        
    ## global_rate_positive_words      .        
    ## rate_positive_words             .        
    ## abs_title_subjectivity          .        
    ## abs_title_sentiment_polarity  225.6742730

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
| lm.fit1        |   6885.712 |
| lm.fit2        |   6886.144 |
| lm.fit3        |   6998.977 |
| lasso.fit.full |   5097.421 |
| lasso.fit.18   |   6916.549 |
| rf.fit1        |   6869.901 |
| rf.fit2        |   6875.201 |
| boostTreefit1  |   6891.053 |
| boostTreefit2  |   6893.944 |

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
