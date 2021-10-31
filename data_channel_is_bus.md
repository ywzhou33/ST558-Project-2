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

    ## tibble [6,258 × 48] (S3: tbl_df/tbl/data.frame)
    ##  $ n_tokens_title              : num [1:6258] 9 9 8 13 11 8 10 12 6 13 ...
    ##  $ n_tokens_content            : num [1:6258] 255 211 397 244 723 708 142 444 109 306 ...
    ##  $ n_unique_tokens             : num [1:6258] 0.605 0.575 0.625 0.56 0.491 ...
    ##  $ n_non_stop_words            : num [1:6258] 1 1 1 1 1 ...
    ##  $ n_non_stop_unique_tokens    : num [1:6258] 0.792 0.664 0.806 0.68 0.642 ...
    ##  $ num_hrefs                   : num [1:6258] 3 3 11 3 18 8 2 9 3 3 ...
    ##  $ num_self_hrefs              : num [1:6258] 1 1 0 2 1 3 1 8 2 2 ...
    ##  $ num_imgs                    : num [1:6258] 1 1 1 1 1 1 1 23 1 1 ...
    ##  $ num_videos                  : num [1:6258] 0 0 0 0 0 1 0 0 0 0 ...
    ##  $ average_token_length        : num [1:6258] 4.91 4.39 5.45 4.42 5.23 ...
    ##  $ num_keywords                : num [1:6258] 4 6 6 4 6 7 5 10 6 10 ...
    ##  $ kw_min_min                  : num [1:6258] 0 0 0 0 0 0 0 0 0 217 ...
    ##  $ kw_max_min                  : num [1:6258] 0 0 0 0 0 0 0 0 0 5700 ...
    ##  $ kw_avg_min                  : num [1:6258] 0 0 0 0 0 ...
    ##  $ kw_min_max                  : num [1:6258] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kw_max_max                  : num [1:6258] 0 0 0 0 0 0 0 0 0 17100 ...
    ##  $ kw_avg_max                  : num [1:6258] 0 0 0 0 0 ...
    ##  $ kw_min_avg                  : num [1:6258] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kw_max_avg                  : num [1:6258] 0 0 0 0 0 0 0 0 0 5700 ...
    ##  $ kw_avg_avg                  : num [1:6258] 0 0 0 0 0 ...
    ##  $ self_reference_min_shares   : num [1:6258] 0 918 0 2800 0 6100 0 585 821 0 ...
    ##  $ self_reference_max_shares   : num [1:6258] 0 918 0 2800 0 6100 0 1600 821 0 ...
    ##  $ self_reference_avg_sharess  : num [1:6258] 0 918 0 2800 0 ...
    ##  $ is_weekend                  : Factor w/ 2 levels "0","1": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ LDA_00                      : num [1:6258] 0.8 0.218 0.867 0.3 0.867 ...
    ##  $ LDA_01                      : num [1:6258] 0.05 0.0333 0.0333 0.05 0.0333 ...
    ##  $ LDA_02                      : num [1:6258] 0.0501 0.0334 0.0333 0.05 0.0333 ...
    ##  $ LDA_03                      : num [1:6258] 0.0501 0.0333 0.0333 0.05 0.0333 ...
    ##  $ LDA_04                      : num [1:6258] 0.05 0.6822 0.0333 0.5497 0.0333 ...
    ##  $ global_subjectivity         : num [1:6258] 0.341 0.702 0.374 0.332 0.375 ...
    ##  $ global_sentiment_polarity   : num [1:6258] 0.1489 0.3233 0.2125 -0.0923 0.1827 ...
    ##  $ global_rate_positive_words  : num [1:6258] 0.0431 0.0569 0.0655 0.0164 0.0636 ...
    ##  $ global_rate_negative_words  : num [1:6258] 0.01569 0.00948 0.01008 0.02459 0.0083 ...
    ##  $ rate_positive_words         : num [1:6258] 0.733 0.857 0.867 0.4 0.885 ...
    ##  $ rate_negative_words         : num [1:6258] 0.267 0.143 0.133 0.6 0.115 ...
    ##  $ avg_positive_polarity       : num [1:6258] 0.287 0.496 0.382 0.292 0.341 ...
    ##  $ min_positive_polarity       : num [1:6258] 0.0333 0.1 0.0333 0.1364 0.0333 ...
    ##  $ max_positive_polarity       : num [1:6258] 0.7 1 1 0.433 1 ...
    ##  $ avg_negative_polarity       : num [1:6258] -0.119 -0.467 -0.145 -0.456 -0.214 ...
    ##  $ min_negative_polarity       : num [1:6258] -0.125 -0.8 -0.2 -1 -0.6 -0.5 -0.3 0 -0.1 0 ...
    ##  $ max_negative_polarity       : num [1:6258] -0.1 -0.133 -0.1 -0.125 -0.1 ...
    ##  $ title_subjectivity          : num [1:6258] 0 0 0 0.7 0.5 ...
    ##  $ title_sentiment_polarity    : num [1:6258] 0 0 0 -0.4 0.5 ...
    ##  $ abs_title_subjectivity      : num [1:6258] 0.5 0.5 0.5 0.2 0 ...
    ##  $ abs_title_sentiment_polarity: num [1:6258] 0 0 0 0.4 0.5 ...
    ##  $ shares                      : num [1:6258] 711 1500 3100 852 425 3200 575 819 732 1200 ...
    ##  $ channel                     : chr [1:6258] "data_channel_is_bus" "data_channel_is_bus" "data_channel_is_bus" "data_channel_is_bus" ...
    ##  $ weekday                     : Factor w/ 7 levels "weekday_is_friday",..: 2 2 2 2 2 2 2 2 2 6 ...

``` r
#summary stats for the response variable. 
summary(pop.data$shares)
```

    ##     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
    ##      1.0    952.2   1400.0   3063.0   2500.0 690400.0

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

    ## Start:  AIC=-118.15
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
    ## Step:  AIC=-118.15
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + kw_min_min + kw_max_min + kw_avg_min + kw_min_max + 
    ##     kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + rate_negative_words + 
    ##     avg_positive_polarity + min_positive_polarity + max_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_subjectivity + title_sentiment_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - title_subjectivity            1    0.0016 6055.2 -120.14
    ## - LDA_02                        1    0.0053 6055.2 -120.14
    ## - n_non_stop_words              1    0.0100 6055.2 -120.14
    ## - global_rate_positive_words    1    0.0126 6055.2 -120.13
    ## - global_rate_negative_words    1    0.0841 6055.3 -120.06
    ## - rate_negative_words           1    0.0850 6055.3 -120.06
    ## - LDA_01                        1    0.1004 6055.3 -120.04
    ## - kw_max_min                    1    0.1183 6055.3 -120.03
    ## - rate_positive_words           1    0.1524 6055.4 -119.99
    ## - avg_negative_polarity         1    0.1689 6055.4 -119.97
    ## - self_reference_max_shares     1    0.2076 6055.4 -119.93
    ## - max_positive_polarity         1    0.2165 6055.4 -119.92
    ## - kw_min_avg                    1    0.2265 6055.4 -119.91
    ## - avg_positive_polarity         1    0.2316 6055.4 -119.91
    ## - min_positive_polarity         1    0.2426 6055.5 -119.90
    ## - kw_max_max                    1    0.2563 6055.5 -119.88
    ## - global_sentiment_polarity     1    0.2972 6055.5 -119.84
    ## - num_self_hrefs                1    0.3077 6055.5 -119.83
    ## - abs_title_sentiment_polarity  1    0.3336 6055.5 -119.80
    ## - kw_avg_max                    1    0.3552 6055.6 -119.78
    ## - self_reference_avg_sharess    1    0.3892 6055.6 -119.75
    ## - n_non_stop_unique_tokens      1    0.4435 6055.7 -119.69
    ## - n_tokens_content              1    0.4724 6055.7 -119.66
    ## - LDA_00                        1    0.4800 6055.7 -119.65
    ## - kw_avg_min                    1    0.5154 6055.7 -119.61
    ## - max_negative_polarity         1    0.6116 6055.8 -119.52
    ## - n_unique_tokens               1    0.6535 6055.9 -119.47
    ## - title_sentiment_polarity      1    0.6697 6055.9 -119.45
    ## - kw_min_min                    1    0.6711 6055.9 -119.45
    ## - n_tokens_title                1    1.1020 6056.3 -119.01
    ## - global_subjectivity           1    1.2256 6056.4 -118.88
    ## - num_hrefs                     1    1.5451 6056.8 -118.55
    ## <none>                                      6055.2 -118.15
    ## - min_negative_polarity         1    2.0688 6057.3 -118.01
    ## - num_imgs                      1    2.2385 6057.4 -117.83
    ## - num_keywords                  1    2.4792 6057.7 -117.58
    ## - num_videos                    1    2.5607 6057.8 -117.50
    ## - kw_min_max                    1    3.0108 6058.2 -117.04
    ## - abs_title_subjectivity        1    3.5817 6058.8 -116.45
    ## - average_token_length          1    4.5458 6059.8 -115.45
    ## - self_reference_min_shares     1    4.9274 6060.1 -115.06
    ## - LDA_03                        1    5.6816 6060.9 -114.28
    ## - kw_max_avg                    1   13.0208 6068.2 -106.70
    ## - kw_avg_avg                    1   16.6743 6071.9 -102.94
    ## 
    ## Step:  AIC=-120.15
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + kw_min_min + kw_max_min + kw_avg_min + kw_min_max + 
    ##     kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + rate_negative_words + 
    ##     avg_positive_polarity + min_positive_polarity + max_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - LDA_02                        1    0.0053 6055.2 -122.14
    ## - n_non_stop_words              1    0.0099 6055.2 -122.14
    ## - global_rate_positive_words    1    0.0122 6055.2 -122.13
    ## - global_rate_negative_words    1    0.0844 6055.3 -122.06
    ## - rate_negative_words           1    0.0848 6055.3 -122.06
    ## - LDA_01                        1    0.1003 6055.3 -122.04
    ## - kw_max_min                    1    0.1182 6055.3 -122.02
    ## - rate_positive_words           1    0.1522 6055.4 -121.99
    ## - avg_negative_polarity         1    0.1682 6055.4 -121.97
    ## - self_reference_max_shares     1    0.2084 6055.4 -121.93
    ## - max_positive_polarity         1    0.2169 6055.4 -121.92
    ## - kw_min_avg                    1    0.2254 6055.4 -121.91
    ## - avg_positive_polarity         1    0.2313 6055.4 -121.91
    ## - min_positive_polarity         1    0.2428 6055.5 -121.89
    ## - kw_max_max                    1    0.2550 6055.5 -121.88
    ## - global_sentiment_polarity     1    0.2957 6055.5 -121.84
    ## - num_self_hrefs                1    0.3080 6055.5 -121.83
    ## - kw_avg_max                    1    0.3537 6055.6 -121.78
    ## - self_reference_avg_sharess    1    0.3900 6055.6 -121.74
    ## - n_non_stop_unique_tokens      1    0.4429 6055.7 -121.69
    ## - n_tokens_content              1    0.4712 6055.7 -121.66
    ## - abs_title_sentiment_polarity  1    0.4806 6055.7 -121.65
    ## - LDA_00                        1    0.4809 6055.7 -121.65
    ## - kw_avg_min                    1    0.5148 6055.7 -121.61
    ## - max_negative_polarity         1    0.6108 6055.8 -121.51
    ## - n_unique_tokens               1    0.6535 6055.9 -121.47
    ## - kw_min_min                    1    0.6710 6055.9 -121.45
    ## - title_sentiment_polarity      1    0.6926 6055.9 -121.43
    ## - n_tokens_title                1    1.1005 6056.3 -121.01
    ## - global_subjectivity           1    1.2337 6056.4 -120.87
    ## - num_hrefs                     1    1.5447 6056.8 -120.55
    ## <none>                                      6055.2 -120.14
    ## - min_negative_polarity         1    2.0711 6057.3 -120.00
    ## - num_imgs                      1    2.2369 6057.4 -119.83
    ## - num_keywords                  1    2.4782 6057.7 -119.58
    ## - num_videos                    1    2.5591 6057.8 -119.50
    ## - kw_min_max                    1    3.0093 6058.2 -119.04
    ## - abs_title_subjectivity        1    4.2690 6059.5 -117.73
    ## - average_token_length          1    4.5443 6059.8 -117.45
    ## - self_reference_min_shares     1    4.9260 6060.1 -117.06
    ## - LDA_03                        1    5.6801 6060.9 -116.28
    ## - kw_max_avg                    1   13.0196 6068.2 -108.70
    ## - kw_avg_avg                    1   16.6733 6071.9 -104.94
    ## 
    ## Step:  AIC=-122.14
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + kw_min_min + kw_max_min + kw_avg_min + kw_min_max + 
    ##     kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_01 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_positive_words + global_rate_negative_words + 
    ##     rate_positive_words + rate_negative_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - n_non_stop_words              1    0.0102 6055.2 -124.13
    ## - global_rate_positive_words    1    0.0118 6055.2 -124.13
    ## - rate_negative_words           1    0.0848 6055.3 -124.05
    ## - global_rate_negative_words    1    0.0853 6055.3 -124.05
    ## - LDA_01                        1    0.0961 6055.3 -124.04
    ## - kw_max_min                    1    0.1205 6055.3 -124.02
    ## - rate_positive_words           1    0.1525 6055.4 -123.98
    ## - avg_negative_polarity         1    0.1679 6055.4 -123.97
    ## - self_reference_max_shares     1    0.2091 6055.4 -123.92
    ## - max_positive_polarity         1    0.2148 6055.4 -123.92
    ## - kw_min_avg                    1    0.2278 6055.4 -123.90
    ## - avg_positive_polarity         1    0.2315 6055.4 -123.90
    ## - min_positive_polarity         1    0.2452 6055.5 -123.89
    ## - kw_max_max                    1    0.2558 6055.5 -123.88
    ## - global_sentiment_polarity     1    0.2965 6055.5 -123.83
    ## - num_self_hrefs                1    0.3035 6055.5 -123.83
    ## - kw_avg_max                    1    0.3635 6055.6 -123.76
    ## - self_reference_avg_sharess    1    0.3919 6055.6 -123.73
    ## - n_non_stop_unique_tokens      1    0.4500 6055.7 -123.67
    ## - n_tokens_content              1    0.4745 6055.7 -123.65
    ## - abs_title_sentiment_polarity  1    0.4805 6055.7 -123.64
    ## - kw_avg_min                    1    0.5205 6055.7 -123.60
    ## - LDA_00                        1    0.5979 6055.8 -123.52
    ## - max_negative_polarity         1    0.6106 6055.8 -123.51
    ## - n_unique_tokens               1    0.6491 6055.9 -123.47
    ## - kw_min_min                    1    0.6705 6055.9 -123.45
    ## - title_sentiment_polarity      1    0.6962 6055.9 -123.42
    ## - n_tokens_title                1    1.1057 6056.3 -123.00
    ## - global_subjectivity           1    1.2316 6056.4 -122.87
    ## - num_hrefs                     1    1.5478 6056.8 -122.54
    ## <none>                                      6055.2 -122.14
    ## - min_negative_polarity         1    2.0719 6057.3 -122.00
    ## - num_imgs                      1    2.2370 6057.5 -121.83
    ## - num_keywords                  1    2.4780 6057.7 -121.58
    ## - num_videos                    1    2.5590 6057.8 -121.50
    ## - kw_min_max                    1    3.0148 6058.2 -121.03
    ## - abs_title_subjectivity        1    4.2844 6059.5 -119.71
    ## - average_token_length          1    4.5477 6059.8 -119.44
    ## - self_reference_min_shares     1    4.9212 6060.1 -119.06
    ## - LDA_03                        1    5.9880 6061.2 -117.95
    ## - kw_max_avg                    1   13.0380 6068.3 -110.68
    ## - kw_avg_avg                    1   16.7160 6071.9 -106.89
    ## 
    ## Step:  AIC=-124.13
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + num_keywords + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_01 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_positive_words + global_rate_negative_words + 
    ##     rate_positive_words + rate_negative_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - global_rate_positive_words    1    0.0123 6055.2 -126.12
    ## - global_rate_negative_words    1    0.0843 6055.3 -126.04
    ## - LDA_01                        1    0.0964 6055.3 -126.03
    ## - kw_max_min                    1    0.1201 6055.3 -126.00
    ## - rate_negative_words           1    0.1474 6055.4 -125.98
    ## - avg_negative_polarity         1    0.1669 6055.4 -125.96
    ## - self_reference_max_shares     1    0.2087 6055.4 -125.91
    ## - max_positive_polarity         1    0.2133 6055.4 -125.91
    ## - avg_positive_polarity         1    0.2293 6055.5 -125.89
    ## - kw_min_avg                    1    0.2304 6055.5 -125.89
    ## - min_positive_polarity         1    0.2455 6055.5 -125.88
    ## - kw_max_max                    1    0.2572 6055.5 -125.86
    ## - global_sentiment_polarity     1    0.2966 6055.5 -125.82
    ## - num_self_hrefs                1    0.2994 6055.5 -125.82
    ## - kw_avg_max                    1    0.3687 6055.6 -125.75
    ## - self_reference_avg_sharess    1    0.3902 6055.6 -125.73
    ## - rate_positive_words           1    0.3957 6055.6 -125.72
    ## - n_non_stop_unique_tokens      1    0.4402 6055.7 -125.67
    ## - n_tokens_content              1    0.4652 6055.7 -125.65
    ## - abs_title_sentiment_polarity  1    0.4811 6055.7 -125.63
    ## - kw_avg_min                    1    0.5196 6055.7 -125.59
    ## - LDA_00                        1    0.5986 6055.8 -125.51
    ## - max_negative_polarity         1    0.6108 6055.8 -125.50
    ## - n_unique_tokens               1    0.6497 6055.9 -125.46
    ## - kw_min_min                    1    0.6734 6055.9 -125.43
    ## - title_sentiment_polarity      1    0.6944 6055.9 -125.41
    ## - n_tokens_title                1    1.1035 6056.3 -124.99
    ## - global_subjectivity           1    1.2215 6056.4 -124.87
    ## - num_hrefs                     1    1.5656 6056.8 -124.51
    ## <none>                                      6055.2 -124.13
    ## - min_negative_polarity         1    2.0681 6057.3 -123.99
    ## - num_imgs                      1    2.2276 6057.5 -123.83
    ## - num_keywords                  1    2.4960 6057.7 -123.55
    ## - num_videos                    1    2.5525 6057.8 -123.49
    ## - kw_min_max                    1    3.0157 6058.2 -123.01
    ## - abs_title_subjectivity        1    4.2785 6059.5 -121.71
    ## - self_reference_min_shares     1    4.9301 6060.2 -121.04
    ## - average_token_length          1    5.0722 6060.3 -120.89
    ## - LDA_03                        1    5.9851 6061.2 -119.95
    ## - kw_max_avg                    1   13.0430 6068.3 -112.66
    ## - kw_avg_avg                    1   16.7241 6072.0 -108.87
    ## 
    ## Step:  AIC=-126.12
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + num_keywords + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_01 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_negative_words + rate_positive_words + rate_negative_words + 
    ##     avg_positive_polarity + min_positive_polarity + max_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - LDA_01                        1    0.0952 6055.3 -128.02
    ## - kw_max_min                    1    0.1209 6055.4 -127.99
    ## - rate_negative_words           1    0.1353 6055.4 -127.98
    ## - avg_negative_polarity         1    0.1572 6055.4 -127.95
    ## - self_reference_max_shares     1    0.2089 6055.4 -127.90
    ## - max_positive_polarity         1    0.2282 6055.5 -127.88
    ## - kw_min_avg                    1    0.2294 6055.5 -127.88
    ## - min_positive_polarity         1    0.2463 6055.5 -127.86
    ## - global_rate_negative_words    1    0.2470 6055.5 -127.86
    ## - kw_max_max                    1    0.2585 6055.5 -127.85
    ## - avg_positive_polarity         1    0.2889 6055.5 -127.82
    ## - num_self_hrefs                1    0.2999 6055.5 -127.81
    ## - global_sentiment_polarity     1    0.3054 6055.5 -127.80
    ## - kw_avg_max                    1    0.3645 6055.6 -127.74
    ## - self_reference_avg_sharess    1    0.3910 6055.6 -127.71
    ## - rate_positive_words           1    0.4273 6055.7 -127.67
    ## - n_non_stop_unique_tokens      1    0.4390 6055.7 -127.66
    ## - n_tokens_content              1    0.4694 6055.7 -127.63
    ## - abs_title_sentiment_polarity  1    0.4797 6055.7 -127.62
    ## - kw_avg_min                    1    0.5211 6055.8 -127.58
    ## - max_negative_polarity         1    0.6013 6055.8 -127.50
    ## - LDA_00                        1    0.6059 6055.8 -127.49
    ## - n_unique_tokens               1    0.6574 6055.9 -127.44
    ## - kw_min_min                    1    0.6730 6055.9 -127.42
    ## - title_sentiment_polarity      1    0.6957 6055.9 -127.40
    ## - n_tokens_title                1    1.0986 6056.3 -126.98
    ## - global_subjectivity           1    1.2155 6056.5 -126.86
    ## - num_hrefs                     1    1.5595 6056.8 -126.50
    ## <none>                                      6055.2 -126.12
    ## - min_negative_polarity         1    2.0679 6057.3 -125.98
    ## - num_imgs                      1    2.2162 6057.5 -125.83
    ## - num_keywords                  1    2.5037 6057.7 -125.53
    ## - num_videos                    1    2.5696 6057.8 -125.46
    ## - kw_min_max                    1    3.0096 6058.3 -125.01
    ## - abs_title_subjectivity        1    4.2812 6059.5 -123.69
    ## - self_reference_min_shares     1    4.9226 6060.2 -123.03
    ## - average_token_length          1    5.0618 6060.3 -122.89
    ## - LDA_03                        1    5.9767 6061.2 -121.94
    ## - kw_max_avg                    1   13.0911 6068.3 -114.60
    ## - kw_avg_avg                    1   16.8018 6072.0 -110.78
    ## 
    ## Step:  AIC=-128.02
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + num_keywords + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_negative_words + rate_positive_words + rate_negative_words + 
    ##     avg_positive_polarity + min_positive_polarity + max_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - rate_negative_words           1    0.1284 6055.5 -129.89
    ## - kw_max_min                    1    0.1352 6055.5 -129.88
    ## - avg_negative_polarity         1    0.1475 6055.5 -129.87
    ## - self_reference_max_shares     1    0.2040 6055.5 -129.81
    ## - max_positive_polarity         1    0.2230 6055.6 -129.79
    ## - kw_min_avg                    1    0.2437 6055.6 -129.77
    ## - min_positive_polarity         1    0.2542 6055.6 -129.76
    ## - global_rate_negative_words    1    0.2607 6055.6 -129.75
    ## - kw_max_max                    1    0.2640 6055.6 -129.75
    ## - avg_positive_polarity         1    0.2804 6055.6 -129.73
    ## - num_self_hrefs                1    0.3002 6055.6 -129.71
    ## - global_sentiment_polarity     1    0.3124 6055.6 -129.69
    ## - kw_avg_max                    1    0.3831 6055.7 -129.62
    ## - self_reference_avg_sharess    1    0.3849 6055.7 -129.62
    ## - rate_positive_words           1    0.4247 6055.8 -129.58
    ## - n_non_stop_unique_tokens      1    0.4469 6055.8 -129.56
    ## - n_tokens_content              1    0.4717 6055.8 -129.53
    ## - abs_title_sentiment_polarity  1    0.4903 6055.8 -129.51
    ## - LDA_00                        1    0.5114 6055.8 -129.49
    ## - kw_avg_min                    1    0.5521 6055.9 -129.45
    ## - max_negative_polarity         1    0.5922 6055.9 -129.41
    ## - n_unique_tokens               1    0.6499 6056.0 -129.35
    ## - kw_min_min                    1    0.6812 6056.0 -129.31
    ## - title_sentiment_polarity      1    0.6887 6056.0 -129.31
    ## - n_tokens_title                1    1.0970 6056.4 -128.88
    ## - global_subjectivity           1    1.2192 6056.6 -128.76
    ## - num_hrefs                     1    1.5426 6056.9 -128.42
    ## <none>                                      6055.3 -128.02
    ## - min_negative_polarity         1    2.0312 6057.4 -127.92
    ## - num_imgs                      1    2.2522 6057.6 -127.69
    ## - num_keywords                  1    2.4873 6057.8 -127.45
    ## - num_videos                    1    2.5779 6057.9 -127.36
    ## - kw_min_max                    1    3.0303 6058.4 -126.89
    ## - abs_title_subjectivity        1    4.3149 6059.7 -125.56
    ## - self_reference_min_shares     1    4.9463 6060.3 -124.91
    ## - average_token_length          1    5.0604 6060.4 -124.79
    ## - LDA_03                        1    5.8948 6061.2 -123.93
    ## - kw_max_avg                    1   13.0737 6068.4 -116.52
    ## - kw_avg_avg                    1   16.8000 6072.1 -112.68
    ## 
    ## Step:  AIC=-129.89
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + num_keywords + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - kw_max_min                    1    0.1410 6055.6 -131.74
    ## - avg_negative_polarity         1    0.1596 6055.6 -131.72
    ## - self_reference_max_shares     1    0.2050 6055.7 -131.67
    ## - max_positive_polarity         1    0.2159 6055.7 -131.66
    ## - avg_positive_polarity         1    0.2176 6055.7 -131.66
    ## - min_positive_polarity         1    0.2207 6055.7 -131.66
    ## - kw_min_avg                    1    0.2367 6055.7 -131.64
    ## - kw_max_max                    1    0.2633 6055.7 -131.61
    ## - num_self_hrefs                1    0.3396 6055.8 -131.53
    ## - kw_avg_max                    1    0.3741 6055.8 -131.50
    ## - rate_positive_words           1    0.3851 6055.8 -131.49
    ## - self_reference_avg_sharess    1    0.3886 6055.9 -131.48
    ## - abs_title_sentiment_polarity  1    0.4750 6055.9 -131.40
    ## - global_sentiment_polarity     1    0.4823 6055.9 -131.39
    ## - LDA_00                        1    0.4824 6055.9 -131.39
    ## - global_rate_negative_words    1    0.4897 6056.0 -131.38
    ## - n_tokens_content              1    0.5380 6056.0 -131.33
    ## - kw_avg_min                    1    0.5643 6056.0 -131.30
    ## - n_non_stop_unique_tokens      1    0.5871 6056.1 -131.28
    ## - n_unique_tokens               1    0.6034 6056.1 -131.26
    ## - max_negative_polarity         1    0.6239 6056.1 -131.24
    ## - kw_min_min                    1    0.6661 6056.1 -131.20
    ## - title_sentiment_polarity      1    0.7110 6056.2 -131.15
    ## - n_tokens_title                1    1.1588 6056.6 -130.69
    ## - global_subjectivity           1    1.3493 6056.8 -130.49
    ## - num_hrefs                     1    1.4838 6056.9 -130.35
    ## <none>                                      6055.5 -129.89
    ## - min_negative_polarity         1    2.0477 6057.5 -129.77
    ## - num_imgs                      1    2.3893 6057.9 -129.42
    ## - num_keywords                  1    2.4153 6057.9 -129.39
    ## - num_videos                    1    2.6245 6058.1 -129.17
    ## - kw_min_max                    1    3.0522 6058.5 -128.73
    ## - abs_title_subjectivity        1    4.3873 6059.9 -127.35
    ## - self_reference_min_shares     1    4.9335 6060.4 -126.79
    ## - LDA_03                        1    5.8001 6061.3 -125.89
    ## - average_token_length          1    6.1389 6061.6 -125.55
    ## - kw_max_avg                    1   12.9944 6068.5 -118.47
    ## - kw_avg_avg                    1   16.7145 6072.2 -114.64
    ## 
    ## Step:  AIC=-131.74
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + num_keywords + kw_min_min + 
    ##     kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_00 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_negative_words + 
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     max_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_sentiment_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - avg_negative_polarity         1    0.1593 6055.8 -133.57
    ## - avg_positive_polarity         1    0.2109 6055.8 -133.52
    ## - max_positive_polarity         1    0.2129 6055.8 -133.52
    ## - self_reference_max_shares     1    0.2157 6055.8 -133.52
    ## - min_positive_polarity         1    0.2296 6055.8 -133.50
    ## - kw_min_avg                    1    0.2662 6055.9 -133.47
    ## - kw_max_max                    1    0.2813 6055.9 -133.45
    ## - kw_avg_max                    1    0.2820 6055.9 -133.45
    ## - num_self_hrefs                1    0.3186 6055.9 -133.41
    ## - rate_positive_words           1    0.3940 6056.0 -133.33
    ## - self_reference_avg_sharess    1    0.3967 6056.0 -133.33
    ## - abs_title_sentiment_polarity  1    0.4652 6056.1 -133.26
    ## - global_sentiment_polarity     1    0.4921 6056.1 -133.23
    ## - global_rate_negative_words    1    0.4944 6056.1 -133.23
    ## - LDA_00                        1    0.5079 6056.1 -133.22
    ## - n_tokens_content              1    0.5265 6056.1 -133.20
    ## - n_non_stop_unique_tokens      1    0.5904 6056.2 -133.13
    ## - n_unique_tokens               1    0.5953 6056.2 -133.12
    ## - max_negative_polarity         1    0.6267 6056.2 -133.09
    ## - kw_min_min                    1    0.7210 6056.3 -133.00
    ## - title_sentiment_polarity      1    0.7403 6056.3 -132.97
    ## - n_tokens_title                1    1.1111 6056.7 -132.59
    ## - global_subjectivity           1    1.3420 6056.9 -132.35
    ## - num_hrefs                     1    1.4936 6057.1 -132.20
    ## <none>                                      6055.6 -131.74
    ## - min_negative_polarity         1    2.0583 6057.7 -131.61
    ## - num_keywords                  1    2.3418 6057.9 -131.32
    ## - num_imgs                      1    2.3740 6058.0 -131.29
    ## - num_videos                    1    2.5862 6058.2 -131.07
    ## - kw_min_max                    1    3.0307 6058.6 -130.61
    ## - kw_avg_min                    1    3.3837 6059.0 -130.24
    ## - abs_title_subjectivity        1    4.4173 6060.0 -129.18
    ## - self_reference_min_shares     1    4.9701 6060.6 -128.61
    ## - LDA_03                        1    5.7424 6061.3 -127.81
    ## - average_token_length          1    6.1365 6061.7 -127.40
    ## - kw_max_avg                    1   14.5284 6070.1 -118.74
    ## - kw_avg_avg                    1   18.4310 6074.0 -114.72
    ## 
    ## Step:  AIC=-133.58
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + num_keywords + kw_min_min + 
    ##     kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_00 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_negative_words + 
    ##     rate_positive_words + avg_positive_polarity + min_positive_polarity + 
    ##     max_positive_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - min_positive_polarity         1    0.2008 6056.0 -135.37
    ## - max_positive_polarity         1    0.2050 6056.0 -135.36
    ## - self_reference_max_shares     1    0.2162 6056.0 -135.35
    ## - avg_positive_polarity         1    0.2605 6056.0 -135.31
    ## - kw_min_avg                    1    0.2720 6056.0 -135.29
    ## - kw_avg_max                    1    0.2743 6056.0 -135.29
    ## - kw_max_max                    1    0.2795 6056.0 -135.29
    ## - num_self_hrefs                1    0.3296 6056.1 -135.24
    ## - rate_positive_words           1    0.3593 6056.1 -135.20
    ## - global_sentiment_polarity     1    0.3898 6056.2 -135.17
    ## - self_reference_avg_sharess    1    0.3979 6056.2 -135.16
    ## - abs_title_sentiment_polarity  1    0.4330 6056.2 -135.13
    ## - LDA_00                        1    0.4929 6056.3 -135.07
    ## - global_rate_negative_words    1    0.5075 6056.3 -135.05
    ## - max_negative_polarity         1    0.5641 6056.3 -134.99
    ## - n_unique_tokens               1    0.5934 6056.4 -134.96
    ## - n_non_stop_unique_tokens      1    0.6012 6056.4 -134.95
    ## - n_tokens_content              1    0.6368 6056.4 -134.92
    ## - kw_min_min                    1    0.7231 6056.5 -134.83
    ## - title_sentiment_polarity      1    0.7929 6056.6 -134.76
    ## - n_tokens_title                1    1.1075 6056.9 -134.43
    ## - global_subjectivity           1    1.2108 6057.0 -134.32
    ## - num_hrefs                     1    1.4544 6057.2 -134.07
    ## <none>                                      6055.8 -133.57
    ## - num_keywords                  1    2.3056 6058.1 -133.19
    ## - num_imgs                      1    2.3757 6058.1 -133.12
    ## - num_videos                    1    2.4999 6058.3 -132.99
    ## - kw_min_max                    1    3.0065 6058.8 -132.47
    ## - min_negative_polarity         1    3.2406 6059.0 -132.23
    ## - kw_avg_min                    1    3.3844 6059.1 -132.08
    ## - abs_title_subjectivity        1    4.4647 6060.2 -130.96
    ## - self_reference_min_shares     1    4.9674 6060.7 -130.44
    ## - LDA_03                        1    5.6981 6061.5 -129.69
    ## - average_token_length          1    6.1236 6061.9 -129.25
    ## - kw_max_avg                    1   14.5653 6070.3 -120.54
    ## - kw_avg_avg                    1   18.4424 6074.2 -116.55
    ## 
    ## Step:  AIC=-135.37
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + num_keywords + kw_min_min + 
    ##     kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_00 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_negative_words + 
    ##     rate_positive_words + avg_positive_polarity + max_positive_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - self_reference_max_shares     1    0.2139 6056.2 -137.15
    ## - kw_min_avg                    1    0.2765 6056.2 -137.08
    ## - kw_max_max                    1    0.2910 6056.3 -137.07
    ## - kw_avg_max                    1    0.2952 6056.3 -137.06
    ## - global_sentiment_polarity     1    0.3015 6056.3 -137.06
    ## - num_self_hrefs                1    0.3328 6056.3 -137.02
    ## - max_positive_polarity         1    0.3556 6056.3 -137.00
    ## - self_reference_avg_sharess    1    0.3989 6056.4 -136.96
    ## - rate_positive_words           1    0.4088 6056.4 -136.94
    ## - abs_title_sentiment_polarity  1    0.4386 6056.4 -136.91
    ## - n_unique_tokens               1    0.5000 6056.5 -136.85
    ## - LDA_00                        1    0.5323 6056.5 -136.82
    ## - max_negative_polarity         1    0.5877 6056.6 -136.76
    ## - n_tokens_content              1    0.6084 6056.6 -136.74
    ## - global_rate_negative_words    1    0.6387 6056.6 -136.71
    ## - n_non_stop_unique_tokens      1    0.6684 6056.6 -136.68
    ## - kw_min_min                    1    0.7372 6056.7 -136.61
    ## - avg_positive_polarity         1    0.7635 6056.7 -136.58
    ## - title_sentiment_polarity      1    0.7940 6056.8 -136.55
    ## - n_tokens_title                1    1.1180 6057.1 -136.21
    ## - global_subjectivity           1    1.1909 6057.2 -136.14
    ## - num_hrefs                     1    1.5200 6057.5 -135.80
    ## <none>                                      6056.0 -135.37
    ## - num_keywords                  1    2.3446 6058.3 -134.94
    ## - num_imgs                      1    2.3587 6058.3 -134.93
    ## - num_videos                    1    2.4888 6058.5 -134.80
    ## - kw_min_max                    1    3.0604 6059.0 -134.21
    ## - kw_avg_min                    1    3.3875 6059.4 -133.87
    ## - min_negative_polarity         1    3.4036 6059.4 -133.85
    ## - abs_title_subjectivity        1    4.5441 6060.5 -132.67
    ## - self_reference_min_shares     1    4.9633 6060.9 -132.24
    ## - LDA_03                        1    5.7720 6061.7 -131.41
    ## - average_token_length          1    6.0178 6062.0 -131.15
    ## - kw_max_avg                    1   14.5834 6070.5 -122.32
    ## - kw_avg_avg                    1   18.4688 6074.4 -118.31
    ## 
    ## Step:  AIC=-137.15
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + num_keywords + kw_min_min + 
    ##     kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     max_positive_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - kw_min_avg                    1    0.2472 6056.4 -138.89
    ## - num_self_hrefs                1    0.2712 6056.5 -138.87
    ## - kw_max_max                    1    0.2815 6056.5 -138.86
    ## - global_sentiment_polarity     1    0.2957 6056.5 -138.84
    ## - kw_avg_max                    1    0.2994 6056.5 -138.84
    ## - self_reference_avg_sharess    1    0.3101 6056.5 -138.83
    ## - max_positive_polarity         1    0.3697 6056.5 -138.76
    ## - rate_positive_words           1    0.4039 6056.6 -138.73
    ## - abs_title_sentiment_polarity  1    0.4366 6056.6 -138.70
    ## - n_unique_tokens               1    0.5116 6056.7 -138.62
    ## - LDA_00                        1    0.5326 6056.7 -138.60
    ## - max_negative_polarity         1    0.5979 6056.8 -138.53
    ## - n_tokens_content              1    0.6147 6056.8 -138.51
    ## - global_rate_negative_words    1    0.6350 6056.8 -138.49
    ## - n_non_stop_unique_tokens      1    0.6632 6056.8 -138.46
    ## - kw_min_min                    1    0.7416 6056.9 -138.38
    ## - title_sentiment_polarity      1    0.7725 6057.0 -138.35
    ## - avg_positive_polarity         1    0.7888 6057.0 -138.33
    ## - n_tokens_title                1    1.1332 6057.3 -137.98
    ## - global_subjectivity           1    1.1904 6057.4 -137.92
    ## - num_hrefs                     1    1.5053 6057.7 -137.59
    ## <none>                                      6056.2 -137.15
    ## - num_imgs                      1    2.3846 6058.6 -136.68
    ## - num_keywords                  1    2.3855 6058.6 -136.68
    ## - num_videos                    1    2.4415 6058.6 -136.62
    ## - kw_min_max                    1    3.1205 6059.3 -135.92
    ## - min_negative_polarity         1    3.3745 6059.6 -135.66
    ## - kw_avg_min                    1    3.5028 6059.7 -135.53
    ## - abs_title_subjectivity        1    4.5446 6060.7 -134.45
    ## - LDA_03                        1    5.7970 6062.0 -133.16
    ## - average_token_length          1    6.0433 6062.2 -132.91
    ## - kw_max_avg                    1   14.5623 6070.7 -124.12
    ## - self_reference_min_shares     1   17.5510 6073.7 -121.04
    ## - kw_avg_avg                    1   18.2733 6074.5 -120.29
    ## 
    ## Step:  AIC=-138.89
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + num_keywords + kw_min_min + 
    ##     kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     max_positive_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - num_self_hrefs                1    0.2525 6056.7 -140.63
    ## - kw_max_max                    1    0.2706 6056.7 -140.61
    ## - kw_avg_max                    1    0.3024 6056.7 -140.58
    ## - self_reference_avg_sharess    1    0.3117 6056.7 -140.57
    ## - global_sentiment_polarity     1    0.3137 6056.7 -140.57
    ## - max_positive_polarity         1    0.3761 6056.8 -140.50
    ## - rate_positive_words           1    0.3973 6056.8 -140.48
    ## - abs_title_sentiment_polarity  1    0.4241 6056.9 -140.45
    ## - n_unique_tokens               1    0.5347 6057.0 -140.34
    ## - max_negative_polarity         1    0.6004 6057.0 -140.27
    ## - global_rate_negative_words    1    0.6204 6057.0 -140.25
    ## - LDA_00                        1    0.6219 6057.0 -140.25
    ## - n_tokens_content              1    0.6273 6057.1 -140.24
    ## - n_non_stop_unique_tokens      1    0.6671 6057.1 -140.20
    ## - kw_min_min                    1    0.7460 6057.2 -140.12
    ## - avg_positive_polarity         1    0.7760 6057.2 -140.09
    ## - title_sentiment_polarity      1    0.7848 6057.2 -140.08
    ## - n_tokens_title                1    1.1324 6057.6 -139.72
    ## - global_subjectivity           1    1.2046 6057.6 -139.65
    ## - num_hrefs                     1    1.4972 6057.9 -139.34
    ## <none>                                      6056.4 -138.89
    ## - num_imgs                      1    2.3678 6058.8 -138.44
    ## - num_videos                    1    2.5157 6058.9 -138.29
    ## - num_keywords                  1    2.9772 6059.4 -137.82
    ## - min_negative_polarity         1    3.3826 6059.8 -137.40
    ## - kw_avg_min                    1    3.4707 6059.9 -137.31
    ## - kw_min_max                    1    3.5614 6060.0 -137.21
    ## - abs_title_subjectivity        1    4.5763 6061.0 -136.16
    ## - LDA_03                        1    5.9013 6062.3 -134.80
    ## - average_token_length          1    6.1619 6062.6 -134.53
    ## - kw_max_avg                    1   17.1562 6073.6 -123.19
    ## - self_reference_min_shares     1   17.5244 6074.0 -122.81
    ## - kw_avg_avg                    1   23.3400 6079.8 -116.82
    ## 
    ## Step:  AIC=-140.63
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_imgs + num_videos + 
    ##     average_token_length + num_keywords + kw_min_min + kw_avg_min + 
    ##     kw_min_max + kw_max_max + kw_avg_max + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     max_positive_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - kw_max_max                    1    0.2633 6056.9 -142.36
    ## - kw_avg_max                    1    0.3078 6057.0 -142.31
    ## - global_sentiment_polarity     1    0.3425 6057.0 -142.28
    ## - max_positive_polarity         1    0.3530 6057.0 -142.27
    ## - self_reference_avg_sharess    1    0.3690 6057.0 -142.25
    ## - abs_title_sentiment_polarity  1    0.4114 6057.1 -142.21
    ## - rate_positive_words           1    0.4372 6057.1 -142.18
    ## - n_unique_tokens               1    0.5625 6057.2 -142.05
    ## - LDA_00                        1    0.5655 6057.2 -142.05
    ## - max_negative_polarity         1    0.6093 6057.3 -142.00
    ## - n_tokens_content              1    0.6451 6057.3 -141.96
    ## - global_rate_negative_words    1    0.6484 6057.3 -141.96
    ## - n_non_stop_unique_tokens      1    0.6603 6057.3 -141.95
    ## - kw_min_min                    1    0.7255 6057.4 -141.88
    ## - avg_positive_polarity         1    0.7587 6057.4 -141.85
    ## - title_sentiment_polarity      1    0.7783 6057.5 -141.83
    ## - n_tokens_title                1    1.1152 6057.8 -141.48
    ## - global_subjectivity           1    1.1909 6057.9 -141.40
    ## <none>                                      6056.7 -140.63
    ## - num_hrefs                     1    2.1058 6058.8 -140.46
    ## - num_videos                    1    2.5294 6059.2 -140.02
    ## - num_imgs                      1    2.5908 6059.3 -139.95
    ## - num_keywords                  1    2.9248 6059.6 -139.61
    ## - min_negative_polarity         1    3.3264 6060.0 -139.19
    ## - kw_avg_min                    1    3.4376 6060.1 -139.08
    ## - kw_min_max                    1    3.5906 6060.3 -138.92
    ## - abs_title_subjectivity        1    4.6104 6061.3 -137.87
    ## - LDA_03                        1    5.8089 6062.5 -136.63
    ## - average_token_length          1    6.4408 6063.1 -135.98
    ## - kw_max_avg                    1   17.0817 6073.8 -125.01
    ## - self_reference_min_shares     1   17.2849 6074.0 -124.80
    ## - kw_avg_avg                    1   23.1995 6079.9 -118.70
    ## 
    ## Step:  AIC=-142.36
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_imgs + num_videos + 
    ##     average_token_length + num_keywords + kw_min_min + kw_avg_min + 
    ##     kw_min_max + kw_avg_max + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_avg_sharess + LDA_00 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_negative_words + 
    ##     rate_positive_words + avg_positive_polarity + max_positive_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - kw_avg_max                    1    0.1500 6057.1 -144.20
    ## - global_sentiment_polarity     1    0.3364 6057.3 -144.01
    ## - max_positive_polarity         1    0.3616 6057.3 -143.99
    ## - self_reference_avg_sharess    1    0.3853 6057.3 -143.96
    ## - abs_title_sentiment_polarity  1    0.4121 6057.4 -143.93
    ## - rate_positive_words           1    0.4402 6057.4 -143.90
    ## - LDA_00                        1    0.5901 6057.5 -143.75
    ## - max_negative_polarity         1    0.6068 6057.5 -143.73
    ## - n_non_stop_unique_tokens      1    0.6120 6057.6 -143.73
    ## - n_unique_tokens               1    0.6215 6057.6 -143.72
    ## - n_tokens_content              1    0.6405 6057.6 -143.70
    ## - global_rate_negative_words    1    0.6595 6057.6 -143.68
    ## - avg_positive_polarity         1    0.7687 6057.7 -143.56
    ## - title_sentiment_polarity      1    0.7868 6057.7 -143.54
    ## - n_tokens_title                1    1.0860 6058.0 -143.24
    ## - global_subjectivity           1    1.2045 6058.1 -143.11
    ## <none>                                      6056.9 -142.36
    ## - num_hrefs                     1    2.0778 6059.0 -142.21
    ## - kw_min_min                    1    2.3726 6059.3 -141.91
    ## - num_imgs                      1    2.5164 6059.5 -141.76
    ## - num_keywords                  1    2.6790 6059.6 -141.59
    ## - num_videos                    1    2.6956 6059.6 -141.57
    ## - min_negative_polarity         1    3.3353 6060.3 -140.91
    ## - kw_min_max                    1    3.3574 6060.3 -140.89
    ## - kw_avg_min                    1    3.4259 6060.4 -140.82
    ## - abs_title_subjectivity        1    4.6066 6061.5 -139.60
    ## - LDA_03                        1    6.0171 6063.0 -138.14
    ## - average_token_length          1    6.5635 6063.5 -137.58
    ## - kw_max_avg                    1   17.0362 6074.0 -126.78
    ## - self_reference_min_shares     1   17.2265 6074.2 -126.58
    ## - kw_avg_avg                    1   23.1591 6080.1 -120.48
    ## 
    ## Step:  AIC=-144.2
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_imgs + num_videos + 
    ##     average_token_length + num_keywords + kw_min_min + kw_avg_min + 
    ##     kw_min_max + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_avg_sharess + LDA_00 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_negative_words + 
    ##     rate_positive_words + avg_positive_polarity + max_positive_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - global_sentiment_polarity     1    0.3361 6057.4 -145.86
    ## - max_positive_polarity         1    0.3713 6057.5 -145.82
    ## - self_reference_avg_sharess    1    0.4013 6057.5 -145.79
    ## - abs_title_sentiment_polarity  1    0.4067 6057.5 -145.78
    ## - rate_positive_words           1    0.4275 6057.5 -145.76
    ## - n_non_stop_unique_tokens      1    0.5736 6057.7 -145.61
    ## - max_negative_polarity         1    0.6095 6057.7 -145.57
    ## - n_tokens_content              1    0.6149 6057.7 -145.57
    ## - global_rate_negative_words    1    0.6286 6057.7 -145.55
    ## - LDA_00                        1    0.6346 6057.7 -145.55
    ## - n_unique_tokens               1    0.6804 6057.8 -145.50
    ## - title_sentiment_polarity      1    0.7815 6057.9 -145.40
    ## - avg_positive_polarity         1    0.7822 6057.9 -145.40
    ## - n_tokens_title                1    1.1017 6058.2 -145.06
    ## - global_subjectivity           1    1.2016 6058.3 -144.96
    ## <none>                                      6057.1 -144.20
    ## - num_hrefs                     1    2.0672 6059.2 -144.07
    ## - num_imgs                      1    2.5680 6059.7 -143.55
    ## - num_keywords                  1    2.7416 6059.8 -143.37
    ## - num_videos                    1    2.8520 6059.9 -143.26
    ## - kw_min_min                    1    3.2678 6060.4 -142.83
    ## - kw_avg_min                    1    3.2997 6060.4 -142.79
    ## - kw_min_max                    1    3.3382 6060.4 -142.75
    ## - min_negative_polarity         1    3.4099 6060.5 -142.68
    ## - abs_title_subjectivity        1    4.5764 6061.7 -141.48
    ## - LDA_03                        1    6.3643 6063.5 -139.63
    ## - average_token_length          1    6.6911 6063.8 -139.29
    ## - self_reference_min_shares     1   17.1846 6074.3 -128.47
    ## - kw_max_avg                    1   19.4853 6076.6 -126.10
    ## - kw_avg_avg                    1   29.3446 6086.4 -115.96
    ## 
    ## Step:  AIC=-145.86
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_imgs + num_videos + 
    ##     average_token_length + num_keywords + kw_min_min + kw_avg_min + 
    ##     kw_min_max + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_avg_sharess + LDA_00 + LDA_03 + global_subjectivity + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     max_positive_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - rate_positive_words           1    0.1501 6057.6 -147.70
    ## - max_positive_polarity         1    0.2952 6057.7 -147.55
    ## - abs_title_sentiment_polarity  1    0.4275 6057.9 -147.41
    ## - self_reference_avg_sharess    1    0.4276 6057.9 -147.41
    ## - LDA_00                        1    0.5687 6058.0 -147.27
    ## - n_tokens_content              1    0.6385 6058.1 -147.20
    ## - n_non_stop_unique_tokens      1    0.6483 6058.1 -147.19
    ## - global_rate_negative_words    1    0.6612 6058.1 -147.17
    ## - n_unique_tokens               1    0.6658 6058.1 -147.17
    ## - title_sentiment_polarity      1    0.6916 6058.1 -147.14
    ## - max_negative_polarity         1    0.7287 6058.2 -147.10
    ## - global_subjectivity           1    1.0226 6058.5 -146.80
    ## - n_tokens_title                1    1.1511 6058.6 -146.67
    ## <none>                                      6057.4 -145.86
    ## - avg_positive_polarity         1    1.9711 6059.4 -145.82
    ## - num_hrefs                     1    1.9957 6059.4 -145.79
    ## - num_imgs                      1    2.6119 6060.0 -145.16
    ## - num_keywords                  1    2.6364 6060.1 -145.13
    ## - num_videos                    1    2.9052 6060.3 -144.85
    ## - kw_min_min                    1    3.2355 6060.7 -144.51
    ## - kw_min_max                    1    3.2872 6060.7 -144.46
    ## - kw_avg_min                    1    3.3388 6060.8 -144.41
    ## - min_negative_polarity         1    4.4683 6061.9 -143.24
    ## - abs_title_subjectivity        1    4.6676 6062.1 -143.04
    ## - LDA_03                        1    6.3329 6063.8 -141.32
    ## - average_token_length          1    6.3562 6063.8 -141.29
    ## - self_reference_min_shares     1   17.1465 6074.6 -130.17
    ## - kw_max_avg                    1   19.4315 6076.9 -127.81
    ## - kw_avg_avg                    1   29.1560 6086.6 -117.81
    ## 
    ## Step:  AIC=-147.7
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_imgs + num_videos + 
    ##     average_token_length + num_keywords + kw_min_min + kw_avg_min + 
    ##     kw_min_max + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_avg_sharess + LDA_00 + LDA_03 + global_subjectivity + 
    ##     global_rate_negative_words + avg_positive_polarity + max_positive_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - abs_title_sentiment_polarity  1    0.4335 6058.0 -149.25
    ## - self_reference_avg_sharess    1    0.4416 6058.0 -149.25
    ## - max_positive_polarity         1    0.5033 6058.1 -149.18
    ## - global_rate_negative_words    1    0.6038 6058.2 -149.08
    ## - LDA_00                        1    0.6259 6058.2 -149.05
    ## - n_unique_tokens               1    0.6679 6058.2 -149.01
    ## - n_tokens_content              1    0.6974 6058.3 -148.98
    ## - title_sentiment_polarity      1    0.7062 6058.3 -148.97
    ## - max_negative_polarity         1    0.7134 6058.3 -148.96
    ## - n_non_stop_unique_tokens      1    0.7146 6058.3 -148.96
    ## - n_tokens_title                1    1.1509 6058.7 -148.51
    ## - global_subjectivity           1    1.3800 6059.0 -148.28
    ## <none>                                      6057.6 -147.70
    ## - num_hrefs                     1    1.9489 6059.5 -147.69
    ## - avg_positive_polarity         1    2.1966 6059.8 -147.43
    ## - num_imgs                      1    2.6231 6060.2 -146.99
    ## - num_keywords                  1    2.7090 6060.3 -146.90
    ## - num_videos                    1    2.9656 6060.5 -146.64
    ## - kw_min_max                    1    3.2876 6060.9 -146.31
    ## - kw_min_min                    1    3.3185 6060.9 -146.27
    ## - kw_avg_min                    1    3.3479 6060.9 -146.24
    ## - min_negative_polarity         1    4.3359 6061.9 -145.22
    ## - abs_title_subjectivity        1    4.5529 6062.1 -145.00
    ## - LDA_03                        1    6.2365 6063.8 -143.26
    ## - average_token_length          1    6.2657 6063.8 -143.23
    ## - self_reference_min_shares     1   17.0798 6074.7 -132.08
    ## - kw_max_avg                    1   19.6118 6077.2 -129.47
    ## - kw_avg_avg                    1   29.4315 6087.0 -119.37
    ## 
    ## Step:  AIC=-149.25
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_imgs + num_videos + 
    ##     average_token_length + num_keywords + kw_min_min + kw_avg_min + 
    ##     kw_min_max + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_avg_sharess + LDA_00 + LDA_03 + global_subjectivity + 
    ##     global_rate_negative_words + avg_positive_polarity + max_positive_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_sentiment_polarity + 
    ##     abs_title_subjectivity
    ## 
    ##                              Df Sum of Sq    RSS     AIC
    ## - self_reference_avg_sharess  1    0.4265 6058.4 -150.81
    ## - max_positive_polarity       1    0.5127 6058.5 -150.72
    ## - global_rate_negative_words  1    0.6484 6058.7 -150.58
    ## - LDA_00                      1    0.6511 6058.7 -150.58
    ## - n_tokens_content            1    0.6832 6058.7 -150.55
    ## - max_negative_polarity       1    0.6878 6058.7 -150.54
    ## - n_non_stop_unique_tokens    1    0.6887 6058.7 -150.54
    ## - n_unique_tokens             1    0.7005 6058.7 -150.53
    ## - n_tokens_title              1    1.1758 6059.2 -150.04
    ## - global_subjectivity         1    1.4813 6059.5 -149.72
    ## - title_sentiment_polarity    1    1.4998 6059.5 -149.70
    ## <none>                                    6058.0 -149.25
    ## - num_hrefs                   1    1.9788 6060.0 -149.21
    ## - avg_positive_polarity       1    2.0998 6060.1 -149.08
    ## - num_imgs                    1    2.5934 6060.6 -148.57
    ## - num_keywords                1    2.7254 6060.7 -148.44
    ## - num_videos                  1    2.9548 6061.0 -148.20
    ## - kw_min_max                  1    3.2620 6061.3 -147.88
    ## - kw_min_min                  1    3.3330 6061.3 -147.81
    ## - kw_avg_min                  1    3.4200 6061.4 -147.72
    ## - abs_title_subjectivity      1    4.1640 6062.2 -146.95
    ## - min_negative_polarity       1    4.5790 6062.6 -146.53
    ## - LDA_03                      1    6.3400 6064.4 -144.71
    ## - average_token_length        1    6.5301 6064.5 -144.51
    ## - self_reference_min_shares   1   17.1126 6075.1 -133.60
    ## - kw_max_avg                  1   19.6609 6077.7 -130.98
    ## - kw_avg_avg                  1   29.5729 6087.6 -120.78
    ## 
    ## Step:  AIC=-150.81
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_imgs + num_videos + 
    ##     average_token_length + num_keywords + kw_min_min + kw_avg_min + 
    ##     kw_min_max + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     LDA_00 + LDA_03 + global_subjectivity + global_rate_negative_words + 
    ##     avg_positive_polarity + max_positive_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_sentiment_polarity + abs_title_subjectivity
    ## 
    ##                              Df Sum of Sq    RSS      AIC
    ## - max_positive_polarity       1     0.530 6059.0 -152.265
    ## - LDA_00                      1     0.629 6059.1 -152.163
    ## - global_rate_negative_words  1     0.631 6059.1 -152.160
    ## - n_non_stop_unique_tokens    1     0.683 6059.1 -152.107
    ## - n_tokens_content            1     0.687 6059.1 -152.103
    ## - max_negative_polarity       1     0.690 6059.1 -152.100
    ## - n_unique_tokens             1     0.722 6059.2 -152.067
    ## - n_tokens_title              1     1.164 6059.6 -151.610
    ## - global_subjectivity         1     1.492 6059.9 -151.271
    ## - title_sentiment_polarity    1     1.532 6060.0 -151.230
    ## <none>                                    6058.4 -150.813
    ## - num_hrefs                   1     1.973 6060.4 -150.775
    ## - avg_positive_polarity       1     2.118 6060.6 -150.626
    ## - num_imgs                    1     2.578 6061.0 -150.151
    ## - num_keywords                1     2.737 6061.2 -149.986
    ## - num_videos                  1     3.079 6061.5 -149.633
    ## - kw_avg_min                  1     3.228 6061.7 -149.479
    ## - kw_min_max                  1     3.262 6061.7 -149.444
    ## - kw_min_min                  1     3.345 6061.8 -149.358
    ## - abs_title_subjectivity      1     4.144 6062.6 -148.533
    ## - min_negative_polarity       1     4.639 6063.1 -148.023
    ## - LDA_03                      1     6.393 6064.8 -146.213
    ## - average_token_length        1     6.583 6065.0 -146.016
    ## - kw_max_avg                  1    19.355 6077.8 -132.852
    ## - kw_avg_avg                  1    30.120 6088.6 -121.778
    ## - self_reference_min_shares   1    62.339 6120.8  -88.749
    ## 
    ## Step:  AIC=-152.27
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_imgs + num_videos + 
    ##     average_token_length + num_keywords + kw_min_min + kw_avg_min + 
    ##     kw_min_max + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     LDA_00 + LDA_03 + global_subjectivity + global_rate_negative_words + 
    ##     avg_positive_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity
    ## 
    ##                              Df Sum of Sq    RSS      AIC
    ## - n_unique_tokens             1     0.502 6059.5 -153.746
    ## - global_rate_negative_words  1     0.564 6059.5 -153.683
    ## - max_negative_polarity       1     0.657 6059.6 -153.587
    ## - LDA_00                      1     0.708 6059.7 -153.534
    ## - n_tokens_content            1     0.799 6059.8 -153.440
    ## - n_non_stop_unique_tokens    1     0.901 6059.9 -153.335
    ## - n_tokens_title              1     1.195 6060.2 -153.032
    ## - global_subjectivity         1     1.498 6060.5 -152.719
    ## - title_sentiment_polarity    1     1.574 6060.5 -152.640
    ## - avg_positive_polarity       1     1.596 6060.6 -152.617
    ## <none>                                    6059.0 -152.265
    ## - num_hrefs                   1     2.090 6061.1 -152.107
    ## - num_imgs                    1     2.621 6061.6 -151.559
    ## - num_keywords                1     2.960 6061.9 -151.209
    ## - num_videos                  1     3.161 6062.1 -151.001
    ## - kw_avg_min                  1     3.225 6062.2 -150.935
    ## - kw_min_max                  1     3.240 6062.2 -150.920
    ## - kw_min_min                  1     3.379 6062.3 -150.776
    ## - abs_title_subjectivity      1     3.964 6062.9 -150.172
    ## - min_negative_polarity       1     4.803 6063.8 -149.306
    ## - average_token_length        1     6.163 6065.1 -147.903
    ## - LDA_03                      1     6.411 6065.4 -147.648
    ## - kw_max_avg                  1    19.465 6078.4 -134.193
    ## - kw_avg_avg                  1    30.312 6089.3 -123.036
    ## - self_reference_min_shares   1    62.165 6121.1  -90.385
    ## 
    ## Step:  AIC=-153.75
    ## shares ~ n_tokens_title + n_tokens_content + n_non_stop_unique_tokens + 
    ##     num_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + kw_min_min + kw_avg_min + kw_min_max + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + LDA_00 + LDA_03 + 
    ##     global_subjectivity + global_rate_negative_words + avg_positive_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_sentiment_polarity + 
    ##     abs_title_subjectivity
    ## 
    ##                              Df Sum of Sq    RSS      AIC
    ## - n_tokens_content            1     0.366 6059.8 -155.368
    ## - LDA_00                      1     0.610 6060.1 -155.116
    ## - global_rate_negative_words  1     0.632 6060.1 -155.093
    ## - max_negative_polarity       1     0.934 6060.4 -154.782
    ## - n_tokens_title              1     1.183 6060.7 -154.525
    ## - global_subjectivity         1     1.383 6060.9 -154.318
    ## - title_sentiment_polarity    1     1.509 6061.0 -154.188
    ## - avg_positive_polarity       1     1.620 6061.1 -154.073
    ## <none>                                    6059.5 -153.746
    ## - num_hrefs                   1     2.459 6061.9 -153.207
    ## - num_keywords                1     2.904 6062.4 -152.748
    ## - num_imgs                    1     3.003 6062.5 -152.646
    ## - kw_min_max                  1     3.189 6062.7 -152.454
    ## - kw_avg_min                  1     3.235 6062.7 -152.406
    ## - kw_min_min                  1     3.456 6062.9 -152.179
    ## - abs_title_subjectivity      1     3.812 6063.3 -151.811
    ## - num_videos                  1     3.866 6063.3 -151.755
    ## - min_negative_polarity       1     4.356 6063.8 -151.249
    ## - average_token_length        1     5.691 6065.2 -149.871
    ## - LDA_03                      1     6.611 6066.1 -148.923
    ## - n_non_stop_unique_tokens    1     9.531 6069.0 -145.911
    ## - kw_max_avg                  1    19.509 6079.0 -135.631
    ## - kw_avg_avg                  1    30.533 6090.0 -124.293
    ## - self_reference_min_shares   1    62.318 6121.8  -91.715
    ## 
    ## Step:  AIC=-155.37
    ## shares ~ n_tokens_title + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_imgs + num_videos + average_token_length + num_keywords + 
    ##     kw_min_min + kw_avg_min + kw_min_max + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + LDA_00 + LDA_03 + global_subjectivity + 
    ##     global_rate_negative_words + avg_positive_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_sentiment_polarity + abs_title_subjectivity
    ## 
    ##                              Df Sum of Sq    RSS      AIC
    ## - global_rate_negative_words  1     0.551 6060.4 -156.800
    ## - LDA_00                      1     0.684 6060.5 -156.662
    ## - max_negative_polarity       1     0.723 6060.6 -156.621
    ## - n_tokens_title              1     1.262 6061.1 -156.066
    ## - title_sentiment_polarity    1     1.480 6061.3 -155.840
    ## - global_subjectivity         1     1.525 6061.4 -155.793
    ## - avg_positive_polarity       1     1.529 6061.4 -155.789
    ## <none>                                    6059.8 -155.368
    ## - num_keywords                1     3.003 6062.8 -154.267
    ## - num_imgs                    1     3.099 6062.9 -154.169
    ## - kw_min_max                  1     3.190 6063.0 -154.075
    ## - kw_avg_min                  1     3.284 6063.1 -153.978
    ## - kw_min_min                  1     3.423 6063.3 -153.835
    ## - num_hrefs                   1     3.671 6063.5 -153.579
    ## - abs_title_subjectivity      1     3.820 6063.7 -153.425
    ## - num_videos                  1     4.034 6063.9 -153.204
    ## - average_token_length        1     5.432 6065.3 -151.761
    ## - min_negative_polarity       1     6.146 6066.0 -151.024
    ## - LDA_03                      1     6.370 6066.2 -150.794
    ## - n_non_stop_unique_tokens    1     9.714 6069.6 -147.344
    ## - kw_max_avg                  1    19.623 6079.5 -137.136
    ## - kw_avg_avg                  1    30.562 6090.4 -125.886
    ## - self_reference_min_shares   1    62.314 6122.2  -93.346
    ## 
    ## Step:  AIC=-156.8
    ## shares ~ n_tokens_title + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_imgs + num_videos + average_token_length + num_keywords + 
    ##     kw_min_min + kw_avg_min + kw_min_max + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + LDA_00 + LDA_03 + global_subjectivity + 
    ##     avg_positive_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity
    ## 
    ##                             Df Sum of Sq    RSS      AIC
    ## - max_negative_polarity      1     0.505 6060.9 -158.278
    ## - LDA_00                     1     0.651 6061.0 -158.128
    ## - n_tokens_title             1     1.251 6061.6 -157.508
    ## - title_sentiment_polarity   1     1.327 6061.7 -157.430
    ## - global_subjectivity        1     1.447 6061.8 -157.305
    ## - avg_positive_polarity      1     1.463 6061.9 -157.290
    ## <none>                                   6060.4 -156.800
    ## - num_keywords               1     2.915 6063.3 -155.790
    ## - num_imgs                   1     3.121 6063.5 -155.578
    ## - kw_min_max                 1     3.227 6063.6 -155.468
    ## - kw_avg_min                 1     3.307 6063.7 -155.385
    ## - kw_min_min                 1     3.444 6063.8 -155.244
    ## - num_hrefs                  1     3.456 6063.8 -155.233
    ## - abs_title_subjectivity     1     3.673 6064.1 -155.008
    ## - num_videos                 1     4.062 6064.4 -154.606
    ## - average_token_length       1     5.638 6066.0 -152.981
    ## - LDA_03                     1     6.374 6066.8 -152.221
    ## - min_negative_polarity      1    10.218 6070.6 -148.257
    ## - n_non_stop_unique_tokens   1    10.585 6071.0 -147.879
    ## - kw_max_avg                 1    19.769 6080.2 -138.419
    ## - kw_avg_avg                 1    30.797 6091.2 -127.079
    ## - self_reference_min_shares  1    62.242 6122.6  -94.856
    ## 
    ## Step:  AIC=-158.28
    ## shares ~ n_tokens_title + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_imgs + num_videos + average_token_length + num_keywords + 
    ##     kw_min_min + kw_avg_min + kw_min_max + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + LDA_00 + LDA_03 + global_subjectivity + 
    ##     avg_positive_polarity + min_negative_polarity + title_sentiment_polarity + 
    ##     abs_title_subjectivity
    ## 
    ##                             Df Sum of Sq    RSS      AIC
    ## - LDA_00                     1     0.655 6061.5 -159.602
    ## - n_tokens_title             1     1.261 6062.2 -158.976
    ## - title_sentiment_polarity   1     1.362 6062.3 -158.872
    ## - global_subjectivity        1     1.474 6062.4 -158.756
    ## - avg_positive_polarity      1     1.515 6062.4 -158.715
    ## <none>                                   6060.9 -158.278
    ## - num_keywords               1     2.901 6063.8 -157.284
    ## - num_imgs                   1     3.174 6064.1 -157.002
    ## - kw_min_max                 1     3.208 6064.1 -156.967
    ## - num_hrefs                  1     3.290 6064.2 -156.882
    ## - kw_avg_min                 1     3.300 6064.2 -156.872
    ## - kw_min_min                 1     3.501 6064.4 -156.664
    ## - abs_title_subjectivity     1     3.742 6064.6 -156.416
    ## - num_videos                 1     4.102 6065.0 -156.044
    ## - average_token_length       1     5.599 6066.5 -154.500
    ## - LDA_03                     1     6.500 6067.4 -153.570
    ## - min_negative_polarity      1    10.669 6071.6 -149.272
    ## - n_non_stop_unique_tokens   1    11.944 6072.8 -147.958
    ## - kw_max_avg                 1    19.801 6080.7 -139.867
    ## - kw_avg_avg                 1    30.888 6091.8 -128.467
    ## - self_reference_min_shares  1    63.674 6124.6  -94.876
    ## 
    ## Step:  AIC=-159.6
    ## shares ~ n_tokens_title + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_imgs + num_videos + average_token_length + num_keywords + 
    ##     kw_min_min + kw_avg_min + kw_min_max + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + LDA_03 + global_subjectivity + 
    ##     avg_positive_polarity + min_negative_polarity + title_sentiment_polarity + 
    ##     abs_title_subjectivity
    ## 
    ##                             Df Sum of Sq    RSS      AIC
    ## - n_tokens_title             1     1.160 6062.7 -160.404
    ## - title_sentiment_polarity   1     1.477 6063.0 -160.077
    ## - global_subjectivity        1     1.494 6063.0 -160.060
    ## - avg_positive_polarity      1     1.504 6063.1 -160.049
    ## <none>                                   6061.5 -159.602
    ## - num_keywords               1     2.657 6064.2 -158.859
    ## - num_imgs                   1     2.933 6064.5 -158.574
    ## - kw_min_max                 1     3.195 6064.7 -158.304
    ## - kw_avg_min                 1     3.313 6064.9 -158.183
    ## - num_hrefs                  1     3.539 6065.1 -157.949
    ## - kw_min_min                 1     3.559 6065.1 -157.929
    ## - abs_title_subjectivity     1     3.745 6065.3 -157.737
    ## - num_videos                 1     4.171 6065.7 -157.298
    ## - average_token_length       1     5.453 6067.0 -155.975
    ## - LDA_03                     1     5.848 6067.4 -155.568
    ## - min_negative_polarity      1    10.863 6072.4 -150.397
    ## - n_non_stop_unique_tokens   1    11.620 6073.2 -149.617
    ## - kw_max_avg                 1    20.148 6081.7 -140.835
    ## - kw_avg_avg                 1    31.702 6093.3 -128.957
    ## - self_reference_min_shares  1    63.793 6125.3  -96.086
    ## 
    ## Step:  AIC=-160.4
    ## shares ~ n_non_stop_unique_tokens + num_hrefs + num_imgs + num_videos + 
    ##     average_token_length + num_keywords + kw_min_min + kw_avg_min + 
    ##     kw_min_max + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     LDA_03 + global_subjectivity + avg_positive_polarity + min_negative_polarity + 
    ##     title_sentiment_polarity + abs_title_subjectivity
    ## 
    ##                             Df Sum of Sq    RSS      AIC
    ## - title_sentiment_polarity   1     1.414 6064.1 -160.945
    ## - avg_positive_polarity      1     1.495 6064.2 -160.861
    ## - global_subjectivity        1     1.500 6064.2 -160.856
    ## <none>                                   6062.7 -160.404
    ## - num_keywords               1     2.698 6065.4 -159.619
    ## - num_imgs                   1     2.965 6065.7 -159.344
    ## - abs_title_subjectivity     1     3.092 6065.8 -159.214
    ## - kw_avg_min                 1     3.135 6065.8 -159.169
    ## - kw_min_max                 1     3.136 6065.8 -159.168
    ## - num_hrefs                  1     3.339 6066.0 -158.958
    ## - kw_min_min                 1     3.350 6066.1 -158.947
    ## - num_videos                 1     4.312 6067.0 -157.955
    ## - LDA_03                     1     5.885 6068.6 -156.332
    ## - average_token_length       1     5.939 6068.6 -156.277
    ## - min_negative_polarity      1    11.010 6073.7 -151.050
    ## - n_non_stop_unique_tokens   1    11.793 6074.5 -150.243
    ## - kw_max_avg                 1    19.938 6082.6 -141.858
    ## - kw_avg_avg                 1    31.561 6094.3 -129.911
    ## - self_reference_min_shares  1    63.575 6126.3  -97.123
    ## 
    ## Step:  AIC=-160.94
    ## shares ~ n_non_stop_unique_tokens + num_hrefs + num_imgs + num_videos + 
    ##     average_token_length + num_keywords + kw_min_min + kw_avg_min + 
    ##     kw_min_max + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     LDA_03 + global_subjectivity + avg_positive_polarity + min_negative_polarity + 
    ##     abs_title_subjectivity
    ## 
    ##                             Df Sum of Sq    RSS      AIC
    ## - avg_positive_polarity      1     1.289 6065.4 -161.615
    ## - global_subjectivity        1     1.526 6065.6 -161.370
    ## <none>                                   6064.1 -160.945
    ## - abs_title_subjectivity     1     2.061 6066.2 -160.818
    ## - num_keywords               1     2.698 6066.8 -160.161
    ## - num_imgs                   1     2.868 6067.0 -159.986
    ## - kw_avg_min                 1     3.115 6067.2 -159.731
    ## - kw_min_max                 1     3.147 6067.3 -159.698
    ## - num_hrefs                  1     3.406 6067.5 -159.430
    ## - kw_min_min                 1     3.408 6067.5 -159.429
    ## - num_videos                 1     4.391 6068.5 -158.415
    ## - average_token_length       1     5.771 6069.9 -156.992
    ## - LDA_03                     1     5.774 6069.9 -156.989
    ## - min_negative_polarity      1    10.471 6074.6 -152.148
    ## - n_non_stop_unique_tokens   1    11.177 6075.3 -151.420
    ## - kw_max_avg                 1    20.115 6084.2 -142.221
    ## - kw_avg_avg                 1    31.938 6096.1 -130.072
    ## - self_reference_min_shares  1    63.359 6127.5  -97.899
    ## 
    ## Step:  AIC=-161.61
    ## shares ~ n_non_stop_unique_tokens + num_hrefs + num_imgs + num_videos + 
    ##     average_token_length + num_keywords + kw_min_min + kw_avg_min + 
    ##     kw_min_max + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     LDA_03 + global_subjectivity + min_negative_polarity + abs_title_subjectivity
    ## 
    ##                             Df Sum of Sq    RSS      AIC
    ## - global_subjectivity        1     0.810 6066.2 -162.780
    ## <none>                                   6065.4 -161.615
    ## - abs_title_subjectivity     1     1.985 6067.4 -161.567
    ## - num_keywords               1     2.561 6068.0 -160.973
    ## - num_imgs                   1     2.853 6068.3 -160.672
    ## - num_hrefs                  1     3.108 6068.5 -160.409
    ## - kw_avg_min                 1     3.127 6068.5 -160.390
    ## - kw_min_max                 1     3.183 6068.6 -160.332
    ## - kw_min_min                 1     3.373 6068.8 -160.136
    ## - num_videos                 1     4.279 6069.7 -159.201
    ## - LDA_03                     1     5.910 6071.3 -157.520
    ## - average_token_length       1     6.101 6071.5 -157.324
    ## - min_negative_polarity      1    10.411 6075.8 -152.882
    ## - n_non_stop_unique_tokens   1    10.579 6076.0 -152.710
    ## - kw_max_avg                 1    19.984 6085.4 -143.031
    ## - kw_avg_avg                 1    31.708 6097.1 -130.985
    ## - self_reference_min_shares  1    63.634 6129.0  -98.303
    ## 
    ## Step:  AIC=-162.78
    ## shares ~ n_non_stop_unique_tokens + num_hrefs + num_imgs + num_videos + 
    ##     average_token_length + num_keywords + kw_min_min + kw_avg_min + 
    ##     kw_min_max + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     LDA_03 + min_negative_polarity + abs_title_subjectivity
    ## 
    ##                             Df Sum of Sq    RSS      AIC
    ## <none>                                   6066.2 -162.780
    ## - abs_title_subjectivity     1     2.015 6068.2 -162.702
    ## - num_keywords               1     2.758 6069.0 -161.935
    ## - num_imgs                   1     2.986 6069.2 -161.700
    ## - kw_avg_min                 1     3.206 6069.4 -161.473
    ## - kw_min_max                 1     3.206 6069.4 -161.473
    ## - num_hrefs                  1     3.260 6069.5 -161.418
    ## - kw_min_min                 1     3.487 6069.7 -161.183
    ## - num_videos                 1     4.397 6070.6 -160.245
    ## - average_token_length       1     5.802 6072.0 -158.797
    ## - LDA_03                     1     6.019 6072.2 -158.573
    ## - n_non_stop_unique_tokens   1    11.910 6078.1 -152.505
    ## - min_negative_polarity      1    13.352 6079.6 -151.021
    ## - kw_max_avg                 1    20.635 6086.9 -143.528
    ## - kw_avg_avg                 1    32.798 6099.0 -131.036
    ## - self_reference_min_shares  1    64.128 6130.3  -98.972

``` r
lm$call[["formula"]] # model selected based on AIC. 
```

    ## shares ~ n_non_stop_unique_tokens + num_hrefs + num_imgs + num_videos + 
    ##     average_token_length + num_keywords + kw_min_min + kw_avg_min + 
    ##     kw_min_max + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     LDA_03 + min_negative_polarity + abs_title_subjectivity
    ## <environment: 0x7f9b9cc1a808>

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
    ##                  832                 1153                  243                  343 
    ##  weekday_is_thursday   weekday_is_tuesday weekday_is_wednesday 
    ##                 1234                 1182                 1271

``` r
C2 <- pop.data %>% 
    group_by( weekday ) %>% 
    summarise( percent = 100 * n() / nrow( X ),mean_shares = mean(shares), mean_images = mean(num_imgs),mean_video = mean(num_videos),mean_link = mean(num_hrefs))
knitr::kable(C2)
```

| weekday                |   percent | mean\_shares | mean\_images | mean\_video | mean\_link |
| :--------------------- | --------: | -----------: | -----------: | ----------: | ---------: |
| weekday\_is\_friday    | 2.4828409 |     2363.770 |     1.695914 |   0.6370192 |   9.012019 |
| weekday\_is\_monday    | 3.4407640 |     3887.436 |     1.889853 |   0.7380746 |   9.643539 |
| weekday\_is\_saturday  | 0.7251567 |     4426.897 |     2.057613 |   0.5390947 |  13.604938 |
| weekday\_is\_sunday    | 1.0235751 |     3543.784 |     1.956268 |   0.7871720 |  13.276968 |
| weekday\_is\_thursday  | 3.6824828 |     2885.192 |     1.772285 |   0.5745543 |   8.842788 |
| weekday\_is\_tuesday   | 3.5273053 |     2932.336 |     1.884941 |   0.7402707 |   8.740271 |
| weekday\_is\_wednesday | 3.7928976 |     2676.552 |     1.684500 |   0.4854445 |   8.521636 |

``` r
table(pop.data$weekday, pop.data$channel)
```

    ##                       
    ##                        data_channel_is_bus
    ##   weekday_is_friday                    832
    ##   weekday_is_monday                   1153
    ##   weekday_is_saturday                  243
    ##   weekday_is_sunday                    343
    ##   weekday_is_thursday                 1234
    ##   weekday_is_tuesday                  1182
    ##   weekday_is_wednesday                1271

``` r
table(pop.data$channel, pop.data$is_weekend)
```

    ##                      
    ##                          0    1
    ##   data_channel_is_bus 5672  586

``` r
C3 <- pop.data %>% group_by(is_weekend)%>% 
    summarise( percent = 100 * n() / nrow( X ),mean_shares = mean(shares), mean_images = mean(num_imgs),mean_video = mean(num_videos),mean_link = mean(num_hrefs))
knitr::kable(C3)
```

| is\_weekend |   percent | mean\_shares | mean\_images | mean\_video | mean\_link |
| :---------- | --------: | -----------: | -----------: | ----------: | ---------: |
| 0           | 16.926291 |     2975.514 |     1.788787 |   0.6315233 |   8.937059 |
| 1           |  1.748732 |     3909.990 |     1.998293 |   0.6843003 |  13.412969 |

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
| TRUE        | FALSE              | FALSE               | FALSE                        | FALSE      | FALSE     | FALSE         | FALSE       | FALSE        | FALSE        | FALSE        | FALSE        | TRUE                         | FALSE                         | FALSE                         | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | FALSE              | FALSE               | FALSE                        | FALSE      | FALSE     | FALSE         | FALSE       | FALSE        | FALSE        | FALSE        | TRUE         | TRUE                         | FALSE                         | FALSE                         | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | FALSE              | FALSE               | FALSE                        | FALSE      | FALSE     | FALSE         | FALSE       | FALSE        | FALSE        | TRUE         | TRUE         | TRUE                         | FALSE                         | FALSE                         | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | FALSE              | FALSE               | FALSE                        | FALSE      | FALSE     | FALSE         | TRUE        | FALSE        | FALSE        | TRUE         | TRUE         | TRUE                         | FALSE                         | FALSE                         | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | FALSE              | FALSE               | FALSE                        | FALSE      | FALSE     | FALSE         | TRUE        | TRUE         | FALSE        | TRUE         | TRUE         | TRUE                         | FALSE                         | FALSE                         | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | FALSE              | FALSE               | FALSE                        | FALSE      | TRUE      | FALSE         | TRUE        | TRUE         | FALSE        | TRUE         | TRUE         | TRUE                         | FALSE                         | FALSE                         | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | FALSE              | FALSE               | TRUE                         | FALSE      | TRUE      | FALSE         | TRUE        | TRUE         | FALSE        | TRUE         | TRUE         | TRUE                         | FALSE                         | FALSE                         | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | FALSE              | FALSE               | TRUE                         | TRUE       | TRUE      | FALSE         | TRUE        | TRUE         | FALSE        | TRUE         | TRUE         | TRUE                         | FALSE                         | FALSE                         | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | FALSE              | FALSE               | TRUE                         | FALSE      | TRUE      | FALSE         | TRUE        | TRUE         | FALSE        | TRUE         | TRUE         | TRUE                         | FALSE                         | FALSE                         | FALSE                 | TRUE                     | TRUE                            |
| TRUE        | FALSE              | FALSE               | TRUE                         | TRUE       | TRUE      | FALSE         | TRUE        | TRUE         | FALSE        | TRUE         | TRUE         | TRUE                         | FALSE                         | FALSE                         | FALSE                 | TRUE                     | TRUE                            |
| TRUE        | FALSE              | FALSE               | TRUE                         | TRUE       | TRUE      | FALSE         | TRUE        | TRUE         | FALSE        | TRUE         | TRUE         | TRUE                         | FALSE                         | FALSE                         | TRUE                  | TRUE                     | TRUE                            |
| TRUE        | FALSE              | FALSE               | TRUE                         | TRUE       | TRUE      | TRUE          | TRUE        | TRUE         | FALSE        | TRUE         | TRUE         | TRUE                         | FALSE                         | FALSE                         | TRUE                  | TRUE                     | TRUE                            |
| TRUE        | FALSE              | FALSE               | TRUE                         | TRUE       | TRUE      | TRUE          | TRUE        | TRUE         | FALSE        | TRUE         | TRUE         | TRUE                         | FALSE                         | TRUE                          | TRUE                  | TRUE                     | TRUE                            |
| TRUE        | FALSE              | FALSE               | TRUE                         | TRUE       | TRUE      | TRUE          | TRUE        | TRUE         | TRUE         | TRUE         | TRUE         | TRUE                         | FALSE                         | TRUE                          | TRUE                  | TRUE                     | TRUE                            |
| TRUE        | FALSE              | FALSE               | TRUE                         | TRUE       | TRUE      | TRUE          | TRUE        | TRUE         | TRUE         | TRUE         | TRUE         | TRUE                         | TRUE                          | TRUE                          | TRUE                  | TRUE                     | TRUE                            |
| TRUE        | TRUE               | TRUE                | TRUE                         | TRUE       | TRUE      | TRUE          | TRUE        | TRUE         | TRUE         | TRUE         | TRUE         | TRUE                         | FALSE                         | TRUE                          | TRUE                  | TRUE                     | TRUE                            |
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
|     12 |  6 |   3 |

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
| lm.fit1 |     16114.27 |        0.0087997 |    2926.846 |
| lm.fit2 |     16065.94 |        0.0112939 |    3035.084 |
| lm.fit3 |     84963.95 |        0.0198473 |    6792.559 |

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
    ## (Intercept)                   614.92222044
    ## n_tokens_title                  .         
    ## n_tokens_content                0.27814750
    ## n_unique_tokens                 .         
    ## n_non_stop_words                .         
    ## n_non_stop_unique_tokens        .         
    ## num_hrefs                       4.48650025
    ## num_self_hrefs                  .         
    ## num_imgs                        .         
    ## num_videos                     61.36133965
    ## average_token_length            .         
    ## num_keywords                    .         
    ## kw_min_min                      .         
    ## kw_max_min                      .         
    ## kw_avg_min                      0.07723132
    ## kw_min_max                      .         
    ## kw_max_max                      .         
    ## kw_avg_max                      .         
    ## kw_min_avg                      .         
    ## kw_max_avg                      .         
    ## kw_avg_avg                      0.49971880
    ## self_reference_min_shares       .         
    ## self_reference_max_shares       .         
    ## self_reference_avg_sharess      .         
    ## is_weekend                      .         
    ## LDA_00                          .         
    ## LDA_01                          .         
    ## LDA_02                          .         
    ## LDA_03                       5003.38611105
    ## LDA_04                          .         
    ## global_subjectivity             .         
    ## global_sentiment_polarity       .         
    ## global_rate_positive_words      .         
    ## global_rate_negative_words   1869.43165908
    ## rate_positive_words             .         
    ## rate_negative_words             .         
    ## avg_positive_polarity           .         
    ## min_positive_polarity           .         
    ## max_positive_polarity           .         
    ## avg_negative_polarity           .         
    ## min_negative_polarity        -591.90421580
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
    ## (Intercept)                  1.293705e+03
    ## n_tokens_content             .           
    ## n_non_stop_words             .           
    ## n_non_stop_unique_tokens     .           
    ## num_hrefs                    8.676282e+00
    ## num_imgs                     .           
    ## num_keywords                 .           
    ## num_videos                   9.416791e+01
    ## kw_avg_max                   .           
    ## kw_min_avg                   .           
    ## kw_max_avg                   .           
    ## kw_avg_avg                   5.013515e-01
    ## self_reference_min_shares    7.744044e-02
    ## self_reference_avg_sharess   2.563786e-03
    ## global_rate_positive_words   .           
    ## rate_positive_words          .           
    ## abs_title_subjectivity       .           
    ## abs_title_sentiment_polarity .

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
| lm.fit1        |   6819.835 |
| lm.fit2        |   6782.762 |
| lm.fit3        |  20436.816 |
| lasso.fit.full |  17465.330 |
| lasso.fit.18   |   6491.358 |
| rf.fit1        |   6406.412 |
| rf.fit2        |   6796.015 |
| boostTreefit1  |   6322.388 |
| boostTreefit2  |   6267.698 |

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
