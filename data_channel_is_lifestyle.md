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

    ## tibble [2,099 × 48] (S3: tbl_df/tbl/data.frame)
    ##  $ n_tokens_title              : num [1:2099] 8 10 11 10 8 11 10 6 12 11 ...
    ##  $ n_tokens_content            : num [1:2099] 960 187 103 243 204 315 1190 374 499 223 ...
    ##  $ n_unique_tokens             : num [1:2099] 0.418 0.667 0.689 0.619 0.586 ...
    ##  $ n_non_stop_words            : num [1:2099] 1 1 1 1 1 ...
    ##  $ n_non_stop_unique_tokens    : num [1:2099] 0.55 0.8 0.806 0.824 0.698 ...
    ##  $ num_hrefs                   : num [1:2099] 21 7 3 1 7 4 25 7 14 5 ...
    ##  $ num_self_hrefs              : num [1:2099] 20 0 1 1 2 4 24 0 1 3 ...
    ##  $ num_imgs                    : num [1:2099] 20 1 1 0 1 1 20 1 1 0 ...
    ##  $ num_videos                  : num [1:2099] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ average_token_length        : num [1:2099] 4.65 4.66 4.84 4.38 4.67 ...
    ##  $ num_keywords                : num [1:2099] 10 7 6 10 8 10 8 8 10 6 ...
    ##  $ kw_min_min                  : num [1:2099] 0 0 0 0 0 0 0 0 217 217 ...
    ##  $ kw_max_min                  : num [1:2099] 0 0 0 0 0 0 0 0 1500 1900 ...
    ##  $ kw_avg_min                  : num [1:2099] 0 0 0 0 0 ...
    ##  $ kw_min_max                  : num [1:2099] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kw_max_max                  : num [1:2099] 0 0 0 0 0 0 0 0 17100 17100 ...
    ##  $ kw_avg_max                  : num [1:2099] 0 0 0 0 0 ...
    ##  $ kw_min_avg                  : num [1:2099] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kw_max_avg                  : num [1:2099] 0 0 0 0 0 ...
    ##  $ kw_avg_avg                  : num [1:2099] 0 0 0 0 0 ...
    ##  $ self_reference_min_shares   : num [1:2099] 545 0 5000 0 0 6200 545 0 1300 6700 ...
    ##  $ self_reference_max_shares   : num [1:2099] 16000 0 5000 0 0 6200 16000 0 1300 16700 ...
    ##  $ self_reference_avg_sharess  : num [1:2099] 3151 0 5000 0 0 ...
    ##  $ is_weekend                  : Factor w/ 2 levels "0","1": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ LDA_00                      : num [1:2099] 0.0201 0.0286 0.4374 0.02 0.2115 ...
    ##  $ LDA_01                      : num [1:2099] 0.1147 0.0286 0.2004 0.02 0.0255 ...
    ##  $ LDA_02                      : num [1:2099] 0.02 0.0286 0.0335 0.02 0.0251 ...
    ##  $ LDA_03                      : num [1:2099] 0.02 0.0287 0.0334 0.02 0.0251 ...
    ##  $ LDA_04                      : num [1:2099] 0.825 0.885 0.295 0.92 0.713 ...
    ##  $ global_subjectivity         : num [1:2099] 0.514 0.477 0.424 0.518 0.652 ...
    ##  $ global_sentiment_polarity   : num [1:2099] 0.268 0.15 0.118 0.156 0.317 ...
    ##  $ global_rate_positive_words  : num [1:2099] 0.0802 0.0267 0.0291 0.0494 0.0735 ...
    ##  $ global_rate_negative_words  : num [1:2099] 0.01667 0.0107 0.00971 0.02058 0.0049 ...
    ##  $ rate_positive_words         : num [1:2099] 0.828 0.714 0.75 0.706 0.938 ...
    ##  $ rate_negative_words         : num [1:2099] 0.172 0.2857 0.25 0.2941 0.0625 ...
    ##  $ avg_positive_polarity       : num [1:2099] 0.402 0.435 0.278 0.333 0.422 ...
    ##  $ min_positive_polarity       : num [1:2099] 0.1 0.2 0.0333 0.1364 0.1 ...
    ##  $ max_positive_polarity       : num [1:2099] 1 0.7 0.5 0.6 1 0.5 1 0.8 0.5 0.5 ...
    ##  $ avg_negative_polarity       : num [1:2099] -0.224 -0.263 -0.125 -0.177 -0.4 ...
    ##  $ min_negative_polarity       : num [1:2099] -0.5 -0.4 -0.125 -0.312 -0.4 ...
    ##  $ max_negative_polarity       : num [1:2099] -0.05 -0.125 -0.125 -0.125 -0.4 -0.125 -0.05 -0.05 -0.1 -0.1 ...
    ##  $ title_subjectivity          : num [1:2099] 0 0 0.857 0 0 ...
    ##  $ title_sentiment_polarity    : num [1:2099] 0 0 -0.714 0 0 ...
    ##  $ abs_title_subjectivity      : num [1:2099] 0.5 0.5 0.357 0.5 0.5 ...
    ##  $ abs_title_sentiment_polarity: num [1:2099] 0 0 0.714 0 0 ...
    ##  $ shares                      : num [1:2099] 556 1900 5700 462 3600 343 507 552 1200 1900 ...
    ##  $ channel                     : chr [1:2099] "data_channel_is_lifestyle" "data_channel_is_lifestyle" "data_channel_is_lifestyle" "data_channel_is_lifestyle" ...
    ##  $ weekday                     : Factor w/ 7 levels "weekday_is_friday",..: 2 2 2 2 2 2 2 2 6 7 ...

``` r
#summary stats for the response variable. 
summary(pop.data$shares)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##      28    1100    1700    3682    3250  208300

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

    ## Start:  AIC=-9.32
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
    ## Step:  AIC=-9.32
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
    ## Step:  AIC=-9.32
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
    ##                                Df Sum of Sq    RSS      AIC
    ## - max_negative_polarity         1    0.0009 2005.8 -11.3230
    ## - title_sentiment_polarity      1    0.0010 2005.8 -11.3229
    ## - kw_max_max                    1    0.0151 2005.8 -11.3082
    ## - avg_negative_polarity         1    0.0170 2005.8 -11.3061
    ## - num_imgs                      1    0.0194 2005.8 -11.3036
    ## - kw_avg_min                    1    0.0220 2005.8 -11.3009
    ## - kw_max_min                    1    0.0252 2005.8 -11.2975
    ## - kw_min_max                    1    0.0334 2005.8 -11.2890
    ## - min_positive_polarity         1    0.0466 2005.8 -11.2752
    ## - LDA_00                        1    0.0754 2005.9 -11.2450
    ## - n_tokens_title                1    0.0868 2005.9 -11.2331
    ## - num_keywords                  1    0.0873 2005.9 -11.2326
    ## - global_rate_negative_words    1    0.0900 2005.9 -11.2298
    ## - min_negative_polarity         1    0.0930 2005.9 -11.2266
    ## - rate_positive_words           1    0.0934 2005.9 -11.2262
    ## - abs_title_sentiment_polarity  1    0.1032 2005.9 -11.2159
    ## - average_token_length          1    0.1371 2006.0 -11.1804
    ## - LDA_03                        1    0.1612 2006.0 -11.1553
    ## - title_subjectivity            1    0.1766 2006.0 -11.1391
    ## - global_subjectivity           1    0.2558 2006.1 -11.0562
    ## - kw_min_min                    1    0.2706 2006.1 -11.0408
    ## - self_reference_max_shares     1    0.5629 2006.4 -10.7350
    ## - global_rate_positive_words    1    0.6003 2006.4 -10.6958
    ## - global_sentiment_polarity     1    0.6301 2006.4 -10.6646
    ## - LDA_01                        1    0.7619 2006.6 -10.5268
    ## - LDA_02                        1    0.7822 2006.6 -10.5056
    ## - num_self_hrefs                1    0.8662 2006.7 -10.4176
    ## - n_unique_tokens               1    1.1241 2006.9 -10.1479
    ## - self_reference_avg_sharess    1    1.1257 2006.9 -10.1463
    ## - avg_positive_polarity         1    1.3339 2007.1  -9.9285
    ## <none>                                      2005.8  -9.3239
    ## - abs_title_subjectivity        1    1.9933 2007.8  -9.2391
    ## - max_positive_polarity         1    2.4522 2008.3  -8.7594
    ## - n_non_stop_words              1    2.5693 2008.4  -8.6370
    ## - kw_avg_max                    1    2.8085 2008.6  -8.3870
    ## - num_hrefs                     1    3.1343 2008.9  -8.0466
    ## - self_reference_min_shares     1    3.7638 2009.6  -7.3889
    ## - n_non_stop_unique_tokens      1    5.4791 2011.3  -5.5981
    ## - kw_max_avg                    1    6.0839 2011.9  -4.9670
    ## - kw_min_avg                    1    6.7033 2012.5  -4.3208
    ## - n_tokens_content              1    7.8768 2013.7  -3.0973
    ## - kw_avg_avg                    1   13.6166 2019.4   2.8772
    ## - num_videos                    1   14.8040 2020.6   4.1110
    ## 
    ## Step:  AIC=-11.32
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
    ##     min_negative_polarity + title_subjectivity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS      AIC
    ## - title_sentiment_polarity      1    0.0010 2005.8 -13.3220
    ## - kw_max_max                    1    0.0151 2005.8 -13.3073
    ## - num_imgs                      1    0.0200 2005.8 -13.3021
    ## - kw_avg_min                    1    0.0222 2005.8 -13.2998
    ## - kw_max_min                    1    0.0251 2005.8 -13.2968
    ## - avg_negative_polarity         1    0.0257 2005.8 -13.2961
    ## - kw_min_max                    1    0.0334 2005.8 -13.2880
    ## - min_positive_polarity         1    0.0469 2005.9 -13.2739
    ## - LDA_00                        1    0.0753 2005.9 -13.2442
    ## - n_tokens_title                1    0.0867 2005.9 -13.2323
    ## - num_keywords                  1    0.0876 2005.9 -13.2313
    ## - global_rate_negative_words    1    0.0896 2005.9 -13.2292
    ## - rate_positive_words           1    0.0937 2005.9 -13.2250
    ## - abs_title_sentiment_polarity  1    0.1029 2005.9 -13.2154
    ## - min_negative_polarity         1    0.1060 2005.9 -13.2121
    ## - average_token_length          1    0.1369 2006.0 -13.1798
    ## - LDA_03                        1    0.1607 2006.0 -13.1548
    ## - title_subjectivity            1    0.1760 2006.0 -13.1388
    ## - global_subjectivity           1    0.2698 2006.1 -13.0407
    ## - kw_min_min                    1    0.2704 2006.1 -13.0401
    ## - self_reference_max_shares     1    0.5632 2006.4 -12.7337
    ## - global_rate_positive_words    1    0.6013 2006.4 -12.6938
    ## - global_sentiment_polarity     1    0.6484 2006.5 -12.6446
    ## - LDA_01                        1    0.7616 2006.6 -12.5262
    ## - LDA_02                        1    0.7818 2006.6 -12.5050
    ## - num_self_hrefs                1    0.8654 2006.7 -12.4176
    ## - self_reference_avg_sharess    1    1.1256 2006.9 -12.1454
    ## - n_unique_tokens               1    1.1429 2007.0 -12.1274
    ## - avg_positive_polarity         1    1.3365 2007.2 -11.9249
    ## <none>                                      2005.8 -11.3230
    ## - abs_title_subjectivity        1    1.9931 2007.8 -11.2384
    ## - max_positive_polarity         1    2.4624 2008.3 -10.7478
    ## - n_non_stop_words              1    2.5737 2008.4 -10.6315
    ## - kw_avg_max                    1    2.8182 2008.6 -10.3759
    ## - num_hrefs                     1    3.1456 2009.0 -10.0338
    ## - self_reference_min_shares     1    3.7633 2009.6  -9.3886
    ## - n_non_stop_unique_tokens      1    5.5348 2011.3  -7.5390
    ## - kw_max_avg                    1    6.0975 2011.9  -6.9519
    ## - kw_min_avg                    1    6.7063 2012.5  -6.3168
    ## - n_tokens_content              1    7.8785 2013.7  -5.0946
    ## - kw_avg_avg                    1   13.6319 2019.4   0.8940
    ## - num_videos                    1   14.8135 2020.6   2.1217
    ## 
    ## Step:  AIC=-13.32
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
    ##     min_negative_polarity + title_subjectivity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS      AIC
    ## - kw_max_max                    1    0.0149 2005.8 -15.3064
    ## - num_imgs                      1    0.0203 2005.8 -15.3008
    ## - kw_avg_min                    1    0.0222 2005.8 -15.2987
    ## - kw_max_min                    1    0.0249 2005.8 -15.2959
    ## - avg_negative_polarity         1    0.0250 2005.8 -15.2958
    ## - kw_min_max                    1    0.0338 2005.8 -15.2866
    ## - min_positive_polarity         1    0.0465 2005.9 -15.2733
    ## - LDA_00                        1    0.0755 2005.9 -15.2430
    ## - n_tokens_title                1    0.0866 2005.9 -15.2314
    ## - num_keywords                  1    0.0870 2005.9 -15.2309
    ## - global_rate_negative_words    1    0.0903 2005.9 -15.2275
    ## - rate_positive_words           1    0.0939 2005.9 -15.2238
    ## - min_negative_polarity         1    0.1052 2005.9 -15.2119
    ## - abs_title_sentiment_polarity  1    0.1261 2005.9 -15.1901
    ## - average_token_length          1    0.1369 2006.0 -15.1788
    ## - LDA_03                        1    0.1617 2006.0 -15.1528
    ## - title_subjectivity            1    0.1759 2006.0 -15.1379
    ## - global_subjectivity           1    0.2694 2006.1 -15.0401
    ## - kw_min_min                    1    0.2704 2006.1 -15.0391
    ## - self_reference_max_shares     1    0.5623 2006.4 -14.7337
    ## - global_rate_positive_words    1    0.6017 2006.4 -14.6924
    ## - global_sentiment_polarity     1    0.6480 2006.5 -14.6440
    ## - LDA_01                        1    0.7628 2006.6 -14.5239
    ## - LDA_02                        1    0.7809 2006.6 -14.5049
    ## - num_self_hrefs                1    0.8673 2006.7 -14.4146
    ## - self_reference_avg_sharess    1    1.1250 2006.9 -14.1450
    ## - n_unique_tokens               1    1.1425 2007.0 -14.1267
    ## - avg_positive_polarity         1    1.3355 2007.2 -13.9249
    ## <none>                                      2005.8 -13.3220
    ## - abs_title_subjectivity        1    2.0073 2007.8 -13.2225
    ## - max_positive_polarity         1    2.4655 2008.3 -12.7435
    ## - n_non_stop_words              1    2.5729 2008.4 -12.6313
    ## - kw_avg_max                    1    2.8320 2008.6 -12.3605
    ## - num_hrefs                     1    3.1447 2009.0 -12.0338
    ## - self_reference_min_shares     1    3.7623 2009.6 -11.3885
    ## - n_non_stop_unique_tokens      1    5.5342 2011.3  -9.5386
    ## - kw_max_avg                    1    6.1064 2011.9  -8.9416
    ## - kw_min_avg                    1    6.7174 2012.5  -8.3042
    ## - n_tokens_content              1    7.8777 2013.7  -7.0944
    ## - kw_avg_avg                    1   13.6592 2019.5  -1.0766
    ## - num_videos                    1   14.8125 2020.6   0.1218
    ## 
    ## Step:  AIC=-15.31
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + kw_min_min + kw_max_min + kw_avg_min + kw_min_max + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + title_subjectivity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS      AIC
    ## - num_imgs                      1    0.0203 2005.8 -17.2852
    ## - kw_avg_min                    1    0.0234 2005.8 -17.2819
    ## - kw_max_min                    1    0.0242 2005.8 -17.2811
    ## - avg_negative_polarity         1    0.0260 2005.8 -17.2792
    ## - kw_min_max                    1    0.0391 2005.9 -17.2655
    ## - min_positive_polarity         1    0.0452 2005.9 -17.2591
    ## - LDA_00                        1    0.0754 2005.9 -17.2275
    ## - n_tokens_title                1    0.0880 2005.9 -17.2143
    ## - global_rate_negative_words    1    0.0911 2005.9 -17.2111
    ## - rate_positive_words           1    0.0953 2005.9 -17.2066
    ## - min_negative_polarity         1    0.1049 2005.9 -17.1966
    ## - num_keywords                  1    0.1058 2005.9 -17.1956
    ## - abs_title_sentiment_polarity  1    0.1246 2006.0 -17.1760
    ## - average_token_length          1    0.1367 2006.0 -17.1633
    ## - title_subjectivity            1    0.1778 2006.0 -17.1203
    ## - LDA_03                        1    0.1826 2006.0 -17.1153
    ## - global_subjectivity           1    0.2684 2006.1 -17.0256
    ## - kw_min_min                    1    0.3595 2006.2 -16.9302
    ## - self_reference_max_shares     1    0.5604 2006.4 -16.7200
    ## - global_rate_positive_words    1    0.6112 2006.4 -16.6669
    ## - global_sentiment_polarity     1    0.6511 2006.5 -16.6251
    ## - LDA_02                        1    0.7711 2006.6 -16.4996
    ## - LDA_01                        1    0.7716 2006.6 -16.4991
    ## - num_self_hrefs                1    0.8559 2006.7 -16.4109
    ## - self_reference_avg_sharess    1    1.1223 2007.0 -16.1323
    ## - n_unique_tokens               1    1.1345 2007.0 -16.1195
    ## - avg_positive_polarity         1    1.3375 2007.2 -15.9072
    ## <none>                                      2005.8 -15.3064
    ## - abs_title_subjectivity        1    2.0088 2007.8 -15.2053
    ## - max_positive_polarity         1    2.4673 2008.3 -14.7260
    ## - n_non_stop_words              1    2.5697 2008.4 -14.6190
    ## - num_hrefs                     1    3.1319 2009.0 -14.0315
    ## - kw_avg_max                    1    3.7433 2009.6 -13.3929
    ## - self_reference_min_shares     1    3.7552 2009.6 -13.3804
    ## - n_non_stop_unique_tokens      1    5.5207 2011.3 -11.5371
    ## - kw_max_avg                    1    6.1195 2011.9 -10.9124
    ## - kw_min_avg                    1    6.7027 2012.5 -10.3040
    ## - n_tokens_content              1    7.8650 2013.7  -9.0922
    ## - kw_avg_avg                    1   13.7652 2019.6  -2.9509
    ## - num_videos                    1   14.9008 2020.7  -1.7711
    ## 
    ## Step:  AIC=-17.29
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_videos + average_token_length + num_keywords + 
    ##     kw_min_min + kw_max_min + kw_avg_min + kw_min_max + kw_avg_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + title_subjectivity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS      AIC
    ## - kw_max_min                    1    0.0231 2005.9 -19.2610
    ## - kw_avg_min                    1    0.0251 2005.9 -19.2589
    ## - avg_negative_polarity         1    0.0289 2005.9 -19.2550
    ## - kw_min_max                    1    0.0390 2005.9 -19.2444
    ## - min_positive_polarity         1    0.0461 2005.9 -19.2369
    ## - LDA_00                        1    0.0716 2005.9 -19.2103
    ## - n_tokens_title                1    0.0871 2005.9 -19.1940
    ## - global_rate_negative_words    1    0.0927 2005.9 -19.1881
    ## - rate_positive_words           1    0.0971 2005.9 -19.1836
    ## - num_keywords                  1    0.1045 2006.0 -19.1759
    ## - min_negative_polarity         1    0.1090 2006.0 -19.1711
    ## - abs_title_sentiment_polarity  1    0.1279 2006.0 -19.1514
    ## - average_token_length          1    0.1294 2006.0 -19.1497
    ## - title_subjectivity            1    0.1849 2006.0 -19.0917
    ## - LDA_03                        1    0.2199 2006.1 -19.0551
    ## - global_subjectivity           1    0.2814 2006.1 -18.9907
    ## - kw_min_min                    1    0.3536 2006.2 -18.9151
    ## - self_reference_max_shares     1    0.5600 2006.4 -18.6993
    ## - global_rate_positive_words    1    0.6092 2006.5 -18.6477
    ## - global_sentiment_polarity     1    0.6463 2006.5 -18.6090
    ## - LDA_01                        1    0.7672 2006.6 -18.4825
    ## - LDA_02                        1    0.7727 2006.6 -18.4767
    ## - num_self_hrefs                1    0.8362 2006.7 -18.4103
    ## - self_reference_avg_sharess    1    1.1226 2007.0 -18.1108
    ## - n_unique_tokens               1    1.1668 2007.0 -18.0645
    ## - avg_positive_polarity         1    1.3575 2007.2 -17.8651
    ## <none>                                      2005.8 -17.2852
    ## - abs_title_subjectivity        1    2.0124 2007.9 -17.1804
    ## - max_positive_polarity         1    2.4867 2008.3 -16.6846
    ## - n_non_stop_words              1    2.5582 2008.4 -16.6098
    ## - num_hrefs                     1    3.2591 2009.1 -15.8775
    ## - kw_avg_max                    1    3.7560 2009.6 -15.3584
    ## - self_reference_min_shares     1    3.7595 2009.6 -15.3548
    ## - n_non_stop_unique_tokens      1    6.0962 2011.9 -12.9155
    ## - kw_max_avg                    1    6.2241 2012.1 -12.7821
    ## - kw_min_avg                    1    6.7322 2012.6 -12.2521
    ## - n_tokens_content              1   10.8475 2016.7  -7.9645
    ## - kw_avg_avg                    1   14.0467 2019.9  -4.6373
    ## - num_videos                    1   15.1023 2021.0  -3.5407
    ## 
    ## Step:  AIC=-19.26
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_videos + average_token_length + num_keywords + 
    ##     kw_min_min + kw_avg_min + kw_min_max + kw_avg_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_00 + LDA_01 + LDA_02 + LDA_03 + 
    ##     global_subjectivity + global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + title_subjectivity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS      AIC
    ## - avg_negative_polarity         1    0.0288 2005.9 -21.2308
    ## - kw_min_max                    1    0.0401 2005.9 -21.2190
    ## - min_positive_polarity         1    0.0461 2005.9 -21.2127
    ## - LDA_00                        1    0.0732 2005.9 -21.1844
    ## - n_tokens_title                1    0.0828 2006.0 -21.1744
    ## - global_rate_negative_words    1    0.0913 2006.0 -21.1655
    ## - rate_positive_words           1    0.0955 2006.0 -21.1610
    ## - min_negative_polarity         1    0.1084 2006.0 -21.1475
    ## - num_keywords                  1    0.1243 2006.0 -21.1309
    ## - average_token_length          1    0.1290 2006.0 -21.1260
    ## - abs_title_sentiment_polarity  1    0.1332 2006.0 -21.1215
    ## - title_subjectivity            1    0.1880 2006.1 -21.0643
    ## - LDA_03                        1    0.2246 2006.1 -21.0259
    ## - global_subjectivity           1    0.2801 2006.2 -20.9679
    ## - kw_min_min                    1    0.3448 2006.2 -20.9002
    ## - self_reference_max_shares     1    0.5601 2006.4 -20.6750
    ## - global_rate_positive_words    1    0.6092 2006.5 -20.6236
    ## - global_sentiment_polarity     1    0.6493 2006.5 -20.5816
    ## - LDA_02                        1    0.7596 2006.6 -20.4662
    ## - LDA_01                        1    0.7648 2006.6 -20.4608
    ## - num_self_hrefs                1    0.8329 2006.7 -20.3896
    ## - kw_avg_min                    1    0.9107 2006.8 -20.3082
    ## - self_reference_avg_sharess    1    1.1222 2007.0 -20.0870
    ## - n_unique_tokens               1    1.1549 2007.0 -20.0528
    ## - avg_positive_polarity         1    1.3590 2007.2 -19.8394
    ## <none>                                      2005.9 -19.2610
    ## - abs_title_subjectivity        1    2.0036 2007.9 -19.1654
    ## - max_positive_polarity         1    2.4971 2008.4 -18.6495
    ## - n_non_stop_words              1    2.5512 2008.4 -18.5930
    ## - num_hrefs                     1    3.2428 2009.1 -17.8703
    ## - self_reference_min_shares     1    3.7549 2009.6 -17.3354
    ## - kw_avg_max                    1    4.4133 2010.3 -16.6478
    ## - n_non_stop_unique_tokens      1    6.0736 2011.9 -14.9149
    ## - kw_max_avg                    1    6.6750 2012.5 -14.2876
    ## - kw_min_avg                    1    6.8714 2012.7 -14.0829
    ## - n_tokens_content              1   10.8399 2016.7  -9.9483
    ## - kw_avg_avg                    1   14.6990 2020.6  -5.9356
    ## - num_videos                    1   15.0825 2021.0  -5.5372
    ## 
    ## Step:  AIC=-21.23
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_videos + average_token_length + num_keywords + 
    ##     kw_min_min + kw_avg_min + kw_min_max + kw_avg_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_00 + LDA_01 + LDA_02 + LDA_03 + 
    ##     global_subjectivity + global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + min_negative_polarity + 
    ##     title_subjectivity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS      AIC
    ## - kw_min_max                    1    0.0379 2005.9 -23.1912
    ## - min_positive_polarity         1    0.0529 2006.0 -23.1755
    ## - LDA_00                        1    0.0760 2006.0 -23.1513
    ## - min_negative_polarity         1    0.0823 2006.0 -23.1447
    ## - n_tokens_title                1    0.0825 2006.0 -23.1445
    ## - rate_positive_words           1    0.0953 2006.0 -23.1311
    ## - average_token_length          1    0.1224 2006.0 -23.1028
    ## - num_keywords                  1    0.1229 2006.0 -23.1022
    ## - global_rate_negative_words    1    0.1239 2006.0 -23.1012
    ## - abs_title_sentiment_polarity  1    0.1276 2006.0 -23.0973
    ## - title_subjectivity            1    0.1845 2006.1 -23.0378
    ## - LDA_03                        1    0.2376 2006.1 -22.9822
    ## - global_subjectivity           1    0.3323 2006.2 -22.8831
    ## - kw_min_min                    1    0.3387 2006.2 -22.8764
    ## - self_reference_max_shares     1    0.5533 2006.5 -22.6519
    ## - global_rate_positive_words    1    0.6730 2006.6 -22.5268
    ## - LDA_02                        1    0.7515 2006.7 -22.4446
    ## - LDA_01                        1    0.7526 2006.7 -22.4435
    ## - num_self_hrefs                1    0.8186 2006.7 -22.3744
    ## - global_sentiment_polarity     1    0.8289 2006.7 -22.3637
    ## - kw_avg_min                    1    0.9241 2006.8 -22.2641
    ## - self_reference_avg_sharess    1    1.1106 2007.0 -22.0691
    ## - n_unique_tokens               1    1.1267 2007.0 -22.0522
    ## - avg_positive_polarity         1    1.4956 2007.4 -21.6664
    ## <none>                                      2005.9 -21.2308
    ## - abs_title_subjectivity        1    2.0088 2007.9 -21.1298
    ## - max_positive_polarity         1    2.5320 2008.4 -20.5830
    ## - n_non_stop_words              1    2.5465 2008.4 -20.5679
    ## - num_hrefs                     1    3.2294 2009.1 -19.8542
    ## - self_reference_min_shares     1    3.7370 2009.6 -19.3241
    ## - kw_avg_max                    1    4.4136 2010.3 -18.6175
    ## - n_non_stop_unique_tokens      1    6.0482 2012.0 -16.9114
    ## - kw_max_avg                    1    6.6640 2012.6 -16.2691
    ## - kw_min_avg                    1    6.8654 2012.8 -16.0590
    ## - n_tokens_content              1   10.8114 2016.7 -11.9480
    ## - kw_avg_avg                    1   14.7056 2020.6  -7.8988
    ## - num_videos                    1   15.0537 2021.0  -7.5372
    ## 
    ## Step:  AIC=-23.19
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_videos + average_token_length + num_keywords + 
    ##     kw_min_min + kw_avg_min + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_00 + LDA_01 + LDA_02 + LDA_03 + 
    ##     global_subjectivity + global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + min_negative_polarity + 
    ##     title_subjectivity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS      AIC
    ## - min_positive_polarity         1    0.0528 2006.0 -25.1359
    ## - LDA_00                        1    0.0713 2006.0 -25.1166
    ## - n_tokens_title                1    0.0802 2006.0 -25.1072
    ## - min_negative_polarity         1    0.0831 2006.0 -25.1042
    ## - rate_positive_words           1    0.0936 2006.0 -25.0933
    ## - average_token_length          1    0.1210 2006.1 -25.0645
    ## - global_rate_negative_words    1    0.1226 2006.1 -25.0629
    ## - abs_title_sentiment_polarity  1    0.1238 2006.1 -25.0616
    ## - num_keywords                  1    0.1511 2006.1 -25.0330
    ## - title_subjectivity            1    0.1800 2006.1 -25.0028
    ## - LDA_03                        1    0.2449 2006.2 -24.9349
    ## - kw_min_min                    1    0.3213 2006.3 -24.8550
    ## - global_subjectivity           1    0.3340 2006.3 -24.8417
    ## - self_reference_max_shares     1    0.5560 2006.5 -24.6095
    ## - global_rate_positive_words    1    0.6709 2006.6 -24.4892
    ## - LDA_01                        1    0.7625 2006.7 -24.3934
    ## - LDA_02                        1    0.7823 2006.7 -24.3728
    ## - num_self_hrefs                1    0.8147 2006.8 -24.3389
    ## - global_sentiment_polarity     1    0.8277 2006.8 -24.3253
    ## - kw_avg_min                    1    0.9416 2006.9 -24.2061
    ## - self_reference_avg_sharess    1    1.1155 2007.0 -24.0242
    ## - n_unique_tokens               1    1.1668 2007.1 -23.9706
    ## - avg_positive_polarity         1    1.4798 2007.4 -23.6433
    ## <none>                                      2005.9 -23.1912
    ## - abs_title_subjectivity        1    2.0087 2007.9 -23.0904
    ## - max_positive_polarity         1    2.5334 2008.5 -22.5419
    ## - n_non_stop_words              1    2.5406 2008.5 -22.5344
    ## - num_hrefs                     1    3.2559 2009.2 -21.7869
    ## - self_reference_min_shares     1    3.7466 2009.7 -21.2744
    ## - kw_avg_max                    1    4.3840 2010.3 -20.6088
    ## - n_non_stop_unique_tokens      1    6.1462 2012.1 -18.7697
    ## - kw_max_avg                    1    6.6388 2012.6 -18.2559
    ## - kw_min_avg                    1    7.8622 2013.8 -16.9803
    ## - n_tokens_content              1   10.7845 2016.7 -13.9366
    ## - kw_avg_avg                    1   14.7087 2020.6  -9.8562
    ## - num_videos                    1   15.0421 2021.0  -9.5099
    ## 
    ## Step:  AIC=-25.14
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_videos + average_token_length + num_keywords + 
    ##     kw_min_min + kw_avg_min + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_00 + LDA_01 + LDA_02 + LDA_03 + 
    ##     global_subjectivity + global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     max_positive_polarity + min_negative_polarity + title_subjectivity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - min_negative_polarity         1    0.0686 2006.1 -27.064
    ## - LDA_00                        1    0.0744 2006.1 -27.058
    ## - n_tokens_title                1    0.0814 2006.1 -27.051
    ## - rate_positive_words           1    0.0910 2006.1 -27.041
    ## - global_rate_negative_words    1    0.1159 2006.1 -27.015
    ## - abs_title_sentiment_polarity  1    0.1186 2006.1 -27.012
    ## - average_token_length          1    0.1365 2006.1 -26.993
    ## - num_keywords                  1    0.1442 2006.1 -26.985
    ## - title_subjectivity            1    0.1795 2006.2 -26.948
    ## - LDA_03                        1    0.2449 2006.2 -26.880
    ## - kw_min_min                    1    0.3210 2006.3 -26.800
    ## - global_subjectivity           1    0.3248 2006.3 -26.796
    ## - self_reference_max_shares     1    0.5579 2006.5 -26.552
    ## - global_rate_positive_words    1    0.6878 2006.7 -26.416
    ## - LDA_01                        1    0.7726 2006.8 -26.328
    ## - LDA_02                        1    0.7878 2006.8 -26.312
    ## - global_sentiment_polarity     1    0.7942 2006.8 -26.305
    ## - num_self_hrefs                1    0.8080 2006.8 -26.291
    ## - kw_avg_min                    1    0.9407 2006.9 -26.152
    ## - self_reference_avg_sharess    1    1.1156 2007.1 -25.969
    ## - n_unique_tokens               1    1.3066 2007.3 -25.769
    ## - avg_positive_polarity         1    1.5129 2007.5 -25.553
    ## <none>                                      2006.0 -25.136
    ## - abs_title_subjectivity        1    2.0311 2008.0 -25.012
    ## - max_positive_polarity         1    2.5066 2008.5 -24.515
    ## - n_non_stop_words              1    2.5967 2008.6 -24.421
    ## - num_hrefs                     1    3.2923 2009.3 -23.694
    ## - self_reference_min_shares     1    3.7449 2009.7 -23.221
    ## - kw_avg_max                    1    4.3773 2010.4 -22.561
    ## - n_non_stop_unique_tokens      1    6.4123 2012.4 -20.437
    ## - kw_max_avg                    1    6.6271 2012.6 -20.213
    ## - kw_min_avg                    1    7.8739 2013.9 -18.913
    ## - n_tokens_content              1   10.7842 2016.8 -15.882
    ## - kw_avg_avg                    1   14.6796 2020.7 -11.832
    ## - num_videos                    1   15.0019 2021.0 -11.497
    ## 
    ## Step:  AIC=-27.06
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_videos + average_token_length + num_keywords + 
    ##     kw_min_min + kw_avg_min + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_00 + LDA_01 + LDA_02 + LDA_03 + 
    ##     global_subjectivity + global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     max_positive_polarity + title_subjectivity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - LDA_00                        1    0.0742 2006.1 -28.986
    ## - n_tokens_title                1    0.0807 2006.1 -28.980
    ## - rate_positive_words           1    0.0898 2006.2 -28.970
    ## - global_rate_negative_words    1    0.1146 2006.2 -28.944
    ## - abs_title_sentiment_polarity  1    0.1197 2006.2 -28.939
    ## - num_keywords                  1    0.1380 2006.2 -28.920
    ## - average_token_length          1    0.1414 2006.2 -28.916
    ## - title_subjectivity            1    0.1797 2006.2 -28.876
    ## - LDA_03                        1    0.2424 2006.3 -28.811
    ## - global_subjectivity           1    0.2697 2006.3 -28.782
    ## - kw_min_min                    1    0.3316 2006.4 -28.717
    ## - self_reference_max_shares     1    0.5502 2006.6 -28.489
    ## - global_rate_positive_words    1    0.6317 2006.7 -28.403
    ## - global_sentiment_polarity     1    0.7299 2006.8 -28.301
    ## - LDA_01                        1    0.7607 2006.8 -28.268
    ## - num_self_hrefs                1    0.7998 2006.9 -28.227
    ## - LDA_02                        1    0.8065 2006.9 -28.220
    ## - kw_avg_min                    1    0.9351 2007.0 -28.086
    ## - self_reference_avg_sharess    1    1.1019 2007.2 -27.912
    ## - n_unique_tokens               1    1.2386 2007.3 -27.768
    ## - avg_positive_polarity         1    1.4553 2007.5 -27.542
    ## <none>                                      2006.1 -27.064
    ## - abs_title_subjectivity        1    2.0091 2008.1 -26.963
    ## - n_non_stop_words              1    2.6128 2008.7 -26.332
    ## - max_positive_polarity         1    2.6651 2008.7 -26.277
    ## - num_hrefs                     1    3.2394 2009.3 -25.677
    ## - self_reference_min_shares     1    3.7118 2009.8 -25.184
    ## - kw_avg_max                    1    4.4346 2010.5 -24.429
    ## - n_non_stop_unique_tokens      1    6.3580 2012.4 -22.422
    ## - kw_max_avg                    1    6.5975 2012.7 -22.172
    ## - kw_min_avg                    1    7.8219 2013.9 -20.896
    ## - n_tokens_content              1   10.7407 2016.8 -17.856
    ## - kw_avg_avg                    1   14.6342 2020.7 -13.807
    ## - num_videos                    1   14.9566 2021.0 -13.473
    ## 
    ## Step:  AIC=-28.99
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_videos + average_token_length + num_keywords + 
    ##     kw_min_min + kw_avg_min + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     global_rate_negative_words + rate_positive_words + avg_positive_polarity + 
    ##     max_positive_polarity + title_subjectivity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - n_tokens_title                1    0.0815 2006.2 -30.901
    ## - rate_positive_words           1    0.0886 2006.2 -30.894
    ## - global_rate_negative_words    1    0.1130 2006.2 -30.868
    ## - abs_title_sentiment_polarity  1    0.1163 2006.2 -30.865
    ## - num_keywords                  1    0.1424 2006.3 -30.837
    ## - average_token_length          1    0.1450 2006.3 -30.835
    ## - title_subjectivity            1    0.1795 2006.3 -30.799
    ## - LDA_03                        1    0.1999 2006.3 -30.777
    ## - global_subjectivity           1    0.2668 2006.4 -30.707
    ## - kw_min_min                    1    0.3562 2006.5 -30.614
    ## - self_reference_max_shares     1    0.5600 2006.7 -30.401
    ## - global_rate_positive_words    1    0.6388 2006.8 -30.318
    ## - global_sentiment_polarity     1    0.7266 2006.9 -30.226
    ## - num_self_hrefs                1    0.8119 2006.9 -30.137
    ## - LDA_01                        1    0.8245 2007.0 -30.124
    ## - LDA_02                        1    0.9038 2007.0 -30.041
    ## - kw_avg_min                    1    0.9116 2007.0 -30.033
    ## - self_reference_avg_sharess    1    1.1104 2007.2 -29.825
    ## - n_unique_tokens               1    1.2791 2007.4 -29.649
    ## - avg_positive_polarity         1    1.4700 2007.6 -29.449
    ## <none>                                      2006.1 -28.986
    ## - abs_title_subjectivity        1    2.0158 2008.2 -28.878
    ## - max_positive_polarity         1    2.6451 2008.8 -28.221
    ## - n_non_stop_words              1    2.6522 2008.8 -28.213
    ## - num_hrefs                     1    3.2872 2009.4 -27.550
    ## - self_reference_min_shares     1    3.7062 2009.8 -27.112
    ## - kw_avg_max                    1    4.3959 2010.5 -26.392
    ## - n_non_stop_unique_tokens      1    6.4918 2012.6 -24.205
    ## - kw_max_avg                    1    6.6748 2012.8 -24.014
    ## - kw_min_avg                    1    8.0017 2014.1 -22.631
    ## - n_tokens_content              1   10.7758 2016.9 -19.742
    ## - kw_avg_avg                    1   14.8113 2020.9 -15.546
    ## - num_videos                    1   14.9804 2021.1 -15.371
    ## 
    ## Step:  AIC=-30.9
    ## shares ~ n_tokens_content + n_unique_tokens + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_videos + 
    ##     average_token_length + num_keywords + kw_min_min + kw_avg_min + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_01 + LDA_02 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_positive_words + global_rate_negative_words + 
    ##     rate_positive_words + avg_positive_polarity + max_positive_polarity + 
    ##     title_subjectivity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - rate_positive_words           1    0.0872 2006.3 -32.810
    ## - global_rate_negative_words    1    0.1147 2006.3 -32.781
    ## - abs_title_sentiment_polarity  1    0.1198 2006.3 -32.776
    ## - average_token_length          1    0.1266 2006.3 -32.769
    ## - num_keywords                  1    0.1460 2006.4 -32.748
    ## - title_subjectivity            1    0.1852 2006.4 -32.707
    ## - LDA_03                        1    0.2030 2006.4 -32.689
    ## - global_subjectivity           1    0.2671 2006.5 -32.622
    ## - kw_min_min                    1    0.3379 2006.5 -32.548
    ## - self_reference_max_shares     1    0.5602 2006.8 -32.315
    ## - global_rate_positive_words    1    0.6351 2006.8 -32.237
    ## - global_sentiment_polarity     1    0.7461 2007.0 -32.121
    ## - num_self_hrefs                1    0.7864 2007.0 -32.079
    ## - LDA_01                        1    0.7969 2007.0 -32.068
    ## - LDA_02                        1    0.8874 2007.1 -31.973
    ## - kw_avg_min                    1    0.8945 2007.1 -31.966
    ## - self_reference_avg_sharess    1    1.1099 2007.3 -31.740
    ## - n_unique_tokens               1    1.2585 2007.5 -31.585
    ## - avg_positive_polarity         1    1.4459 2007.7 -31.389
    ## <none>                                      2006.2 -30.901
    ## - abs_title_subjectivity        1    1.9527 2008.2 -30.859
    ## - n_non_stop_words              1    2.6093 2008.8 -30.173
    ## - max_positive_polarity         1    2.6153 2008.8 -30.167
    ## - num_hrefs                     1    3.2240 2009.4 -29.531
    ## - self_reference_min_shares     1    3.7117 2009.9 -29.021
    ## - kw_avg_max                    1    4.3149 2010.5 -28.392
    ## - n_non_stop_unique_tokens      1    6.4503 2012.7 -26.163
    ## - kw_max_avg                    1    6.6954 2012.9 -25.908
    ## - kw_min_avg                    1    8.1021 2014.3 -24.441
    ## - n_tokens_content              1   10.8237 2017.0 -21.607
    ## - kw_avg_avg                    1   14.8013 2021.0 -17.472
    ## - num_videos                    1   14.9383 2021.2 -17.330
    ## 
    ## Step:  AIC=-32.81
    ## shares ~ n_tokens_content + n_unique_tokens + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_videos + 
    ##     average_token_length + num_keywords + kw_min_min + kw_avg_min + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_01 + LDA_02 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_positive_words + global_rate_negative_words + 
    ##     avg_positive_polarity + max_positive_polarity + title_subjectivity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - global_rate_negative_words    1    0.0287 2006.3 -34.780
    ## - abs_title_sentiment_polarity  1    0.1158 2006.4 -34.689
    ## - average_token_length          1    0.1164 2006.4 -34.688
    ## - num_keywords                  1    0.1464 2006.5 -34.657
    ## - title_subjectivity            1    0.1814 2006.5 -34.620
    ## - LDA_03                        1    0.2120 2006.5 -34.588
    ## - global_subjectivity           1    0.2723 2006.6 -34.525
    ## - kw_min_min                    1    0.3383 2006.6 -34.456
    ## - self_reference_max_shares     1    0.5576 2006.9 -34.227
    ## - global_rate_positive_words    1    0.5991 2006.9 -34.183
    ## - num_self_hrefs                1    0.7753 2007.1 -33.999
    ## - LDA_01                        1    0.8013 2007.1 -33.972
    ## - kw_avg_min                    1    0.9107 2007.2 -33.857
    ## - LDA_02                        1    0.9121 2007.2 -33.856
    ## - global_sentiment_polarity     1    1.0216 2007.3 -33.741
    ## - self_reference_avg_sharess    1    1.1060 2007.4 -33.653
    ## - n_unique_tokens               1    1.2393 2007.5 -33.514
    ## - avg_positive_polarity         1    1.6270 2007.9 -33.109
    ## <none>                                      2006.3 -32.810
    ## - abs_title_subjectivity        1    1.9558 2008.3 -32.765
    ## - max_positive_polarity         1    2.7182 2009.0 -31.968
    ## - num_hrefs                     1    3.2479 2009.5 -31.415
    ## - self_reference_min_shares     1    3.7022 2010.0 -30.940
    ## - kw_avg_max                    1    4.3086 2010.6 -30.307
    ## - n_non_stop_words              1    4.3820 2010.7 -30.230
    ## - n_non_stop_unique_tokens      1    6.3831 2012.7 -28.143
    ## - kw_max_avg                    1    6.6806 2013.0 -27.832
    ## - kw_min_avg                    1    8.1074 2014.4 -26.345
    ## - n_tokens_content              1   10.8052 2017.1 -23.536
    ## - kw_avg_avg                    1   14.7991 2021.1 -19.384
    ## - num_videos                    1   14.8864 2021.2 -19.293
    ## 
    ## Step:  AIC=-34.78
    ## shares ~ n_tokens_content + n_unique_tokens + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_videos + 
    ##     average_token_length + num_keywords + kw_min_min + kw_avg_min + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_01 + LDA_02 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_positive_words + avg_positive_polarity + max_positive_polarity + 
    ##     title_subjectivity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - abs_title_sentiment_polarity  1    0.1152 2006.4 -36.659
    ## - average_token_length          1    0.1244 2006.5 -36.650
    ## - num_keywords                  1    0.1524 2006.5 -36.620
    ## - title_subjectivity            1    0.1826 2006.5 -36.589
    ## - LDA_03                        1    0.2034 2006.5 -36.567
    ## - global_subjectivity           1    0.2600 2006.6 -36.508
    ## - kw_min_min                    1    0.3429 2006.7 -36.421
    ## - self_reference_max_shares     1    0.5572 2006.9 -36.197
    ## - global_rate_positive_words    1    0.6348 2007.0 -36.116
    ## - num_self_hrefs                1    0.7657 2007.1 -35.979
    ## - LDA_01                        1    0.8190 2007.2 -35.923
    ## - kw_avg_min                    1    0.9073 2007.2 -35.831
    ## - LDA_02                        1    0.9192 2007.2 -35.818
    ## - self_reference_avg_sharess    1    1.1041 2007.4 -35.625
    ## - n_unique_tokens               1    1.2365 2007.6 -35.487
    ## - global_sentiment_polarity     1    1.8093 2008.1 -34.888
    ## - avg_positive_polarity         1    1.8189 2008.2 -34.878
    ## <none>                                      2006.3 -34.780
    ## - abs_title_subjectivity        1    1.9840 2008.3 -34.705
    ## - max_positive_polarity         1    2.7494 2009.1 -33.906
    ## - num_hrefs                     1    3.2805 2009.6 -33.351
    ## - self_reference_min_shares     1    3.6933 2010.0 -32.920
    ## - kw_avg_max                    1    4.3139 2010.6 -32.272
    ## - n_non_stop_words              1    4.3840 2010.7 -32.198
    ## - n_non_stop_unique_tokens      1    6.3740 2012.7 -30.122
    ## - kw_max_avg                    1    6.6625 2013.0 -29.821
    ## - kw_min_avg                    1    8.1429 2014.5 -28.278
    ## - n_tokens_content              1   10.8092 2017.1 -25.502
    ## - kw_avg_avg                    1   14.7815 2021.1 -21.372
    ## - num_videos                    1   14.9003 2021.2 -21.249
    ## 
    ## Step:  AIC=-36.66
    ## shares ~ n_tokens_content + n_unique_tokens + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_videos + 
    ##     average_token_length + num_keywords + kw_min_min + kw_avg_min + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_01 + LDA_02 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_positive_words + avg_positive_polarity + max_positive_polarity + 
    ##     title_subjectivity + abs_title_subjectivity
    ## 
    ##                              Df Sum of Sq    RSS     AIC
    ## - title_subjectivity          1    0.0781 2006.5 -38.578
    ## - average_token_length        1    0.1313 2006.6 -38.522
    ## - num_keywords                1    0.1534 2006.6 -38.499
    ## - LDA_03                      1    0.2027 2006.7 -38.447
    ## - global_subjectivity         1    0.2845 2006.7 -38.362
    ## - kw_min_min                  1    0.3369 2006.8 -38.307
    ## - self_reference_max_shares   1    0.5479 2007.0 -38.086
    ## - global_rate_positive_words  1    0.6332 2007.1 -37.997
    ## - num_self_hrefs              1    0.7366 2007.2 -37.889
    ## - LDA_01                      1    0.8003 2007.2 -37.822
    ## - LDA_02                      1    0.9005 2007.3 -37.718
    ## - kw_avg_min                  1    0.9374 2007.4 -37.679
    ## - self_reference_avg_sharess  1    1.0924 2007.5 -37.517
    ## - n_unique_tokens             1    1.2510 2007.7 -37.351
    ## - avg_positive_polarity       1    1.7633 2008.2 -36.816
    ## - global_sentiment_polarity   1    1.8546 2008.3 -36.720
    ## <none>                                    2006.4 -36.659
    ## - abs_title_subjectivity      1    2.1317 2008.6 -36.431
    ## - max_positive_polarity       1    2.7369 2009.2 -35.798
    ## - num_hrefs                   1    3.2569 2009.7 -35.255
    ## - self_reference_min_shares   1    3.6785 2010.1 -34.815
    ## - kw_avg_max                  1    4.3225 2010.8 -34.142
    ## - n_non_stop_words            1    4.3788 2010.8 -34.084
    ## - n_non_stop_unique_tokens    1    6.4240 2012.9 -31.950
    ## - kw_max_avg                  1    6.6292 2013.1 -31.736
    ## - kw_min_avg                  1    8.1226 2014.6 -30.179
    ## - n_tokens_content            1   10.7970 2017.2 -27.395
    ## - kw_avg_avg                  1   14.7473 2021.2 -23.288
    ## - num_videos                  1   14.8649 2021.3 -23.166
    ## 
    ## Step:  AIC=-38.58
    ## shares ~ n_tokens_content + n_unique_tokens + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_videos + 
    ##     average_token_length + num_keywords + kw_min_min + kw_avg_min + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_01 + LDA_02 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_positive_words + avg_positive_polarity + max_positive_polarity + 
    ##     abs_title_subjectivity
    ## 
    ##                              Df Sum of Sq    RSS     AIC
    ## - average_token_length        1    0.1319 2006.7 -40.440
    ## - num_keywords                1    0.1582 2006.7 -40.412
    ## - LDA_03                      1    0.2168 2006.7 -40.351
    ## - global_subjectivity         1    0.3353 2006.9 -40.227
    ## - kw_min_min                  1    0.3387 2006.9 -40.223
    ## - self_reference_max_shares   1    0.5537 2007.1 -39.999
    ## - global_rate_positive_words  1    0.6555 2007.2 -39.892
    ## - num_self_hrefs              1    0.7383 2007.3 -39.806
    ## - LDA_01                      1    0.8019 2007.3 -39.739
    ## - LDA_02                      1    0.9097 2007.4 -39.626
    ## - kw_avg_min                  1    0.9370 2007.5 -39.598
    ## - self_reference_avg_sharess  1    1.1000 2007.6 -39.427
    ## - n_unique_tokens             1    1.2408 2007.8 -39.280
    ## - avg_positive_polarity       1    1.7915 2008.3 -38.704
    ## - global_sentiment_polarity   1    1.9039 2008.4 -38.587
    ## <none>                                    2006.5 -38.578
    ## - abs_title_subjectivity      1    2.3197 2008.8 -38.152
    ## - max_positive_polarity       1    2.7269 2009.2 -37.727
    ## - num_hrefs                   1    3.2965 2009.8 -37.132
    ## - self_reference_min_shares   1    3.6909 2010.2 -36.720
    ## - kw_avg_max                  1    4.3083 2010.8 -36.076
    ## - n_non_stop_words            1    4.5065 2011.0 -35.869
    ## - n_non_stop_unique_tokens    1    6.4019 2012.9 -33.891
    ## - kw_max_avg                  1    6.6382 2013.2 -33.645
    ## - kw_min_avg                  1    8.1712 2014.7 -32.047
    ## - n_tokens_content            1   10.7806 2017.3 -29.330
    ## - kw_avg_avg                  1   14.7761 2021.3 -25.177
    ## - num_videos                  1   14.8190 2021.3 -25.133
    ## 
    ## Step:  AIC=-40.44
    ## shares ~ n_tokens_content + n_unique_tokens + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_videos + 
    ##     num_keywords + kw_min_min + kw_avg_min + kw_avg_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     avg_positive_polarity + max_positive_polarity + abs_title_subjectivity
    ## 
    ##                              Df Sum of Sq    RSS     AIC
    ## - num_keywords                1    0.1694 2006.8 -42.263
    ## - LDA_03                      1    0.1982 2006.8 -42.232
    ## - global_subjectivity         1    0.3062 2007.0 -42.119
    ## - kw_min_min                  1    0.3400 2007.0 -42.084
    ## - self_reference_max_shares   1    0.5756 2007.2 -41.838
    ## - global_rate_positive_words  1    0.6543 2007.3 -41.755
    ## - LDA_01                      1    0.7733 2007.4 -41.631
    ## - num_self_hrefs              1    0.7862 2007.4 -41.617
    ## - LDA_02                      1    0.8316 2007.5 -41.570
    ## - kw_avg_min                  1    0.9252 2007.6 -41.472
    ## - n_unique_tokens             1    1.1128 2007.8 -41.276
    ## - self_reference_avg_sharess  1    1.1243 2007.8 -41.264
    ## - avg_positive_polarity       1    1.7257 2008.4 -40.635
    ## - global_sentiment_polarity   1    1.8998 2008.5 -40.453
    ## <none>                                    2006.7 -40.440
    ## - abs_title_subjectivity      1    2.3933 2009.0 -39.938
    ## - max_positive_polarity       1    2.7530 2009.4 -39.562
    ## - num_hrefs                   1    3.6694 2010.3 -38.605
    ## - self_reference_min_shares   1    3.7141 2010.4 -38.558
    ## - kw_avg_max                  1    4.2947 2011.0 -37.952
    ## - n_non_stop_unique_tokens    1    6.5491 2013.2 -35.600
    ## - kw_max_avg                  1    6.6603 2013.3 -35.484
    ## - n_non_stop_words            1    8.3151 2015.0 -33.760
    ## - kw_min_avg                  1    8.3266 2015.0 -33.748
    ## - n_tokens_content            1   10.8825 2017.5 -31.087
    ## - num_videos                  1   14.7590 2021.4 -27.058
    ## - kw_avg_avg                  1   14.8311 2021.5 -26.983
    ## 
    ## Step:  AIC=-42.26
    ## shares ~ n_tokens_content + n_unique_tokens + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_videos + 
    ##     kw_min_min + kw_avg_min + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     avg_positive_polarity + max_positive_polarity + abs_title_subjectivity
    ## 
    ##                              Df Sum of Sq    RSS     AIC
    ## - LDA_03                      1    0.1554 2007.0 -44.100
    ## - kw_min_min                  1    0.2610 2007.1 -43.990
    ## - global_subjectivity         1    0.2926 2007.1 -43.956
    ## - self_reference_max_shares   1    0.5726 2007.4 -43.664
    ## - global_rate_positive_words  1    0.6380 2007.5 -43.595
    ## - LDA_01                      1    0.7179 2007.5 -43.512
    ## - LDA_02                      1    0.8236 2007.7 -43.401
    ## - kw_avg_min                  1    0.8738 2007.7 -43.349
    ## - num_self_hrefs              1    0.8783 2007.7 -43.344
    ## - self_reference_avg_sharess  1    1.1182 2007.9 -43.093
    ## - n_unique_tokens             1    1.1461 2008.0 -43.064
    ## - avg_positive_polarity       1    1.7012 2008.5 -42.484
    ## - global_sentiment_polarity   1    1.8712 2008.7 -42.306
    ## <none>                                    2006.8 -42.263
    ## - abs_title_subjectivity      1    2.4007 2009.2 -41.753
    ## - max_positive_polarity       1    2.7976 2009.6 -41.338
    ## - num_hrefs                   1    3.5503 2010.4 -40.552
    ## - self_reference_min_shares   1    3.7160 2010.5 -40.379
    ## - kw_avg_max                  1    4.2356 2011.1 -39.837
    ## - kw_max_avg                  1    6.6494 2013.5 -37.319
    ## - n_non_stop_unique_tokens    1    6.7523 2013.6 -37.212
    ## - kw_min_avg                  1    8.1671 2015.0 -35.738
    ## - n_non_stop_words            1    8.2465 2015.1 -35.655
    ## - n_tokens_content            1   10.9320 2017.8 -32.859
    ## - num_videos                  1   14.6239 2021.5 -29.022
    ## - kw_avg_avg                  1   14.6746 2021.5 -28.970
    ## 
    ## Step:  AIC=-44.1
    ## shares ~ n_tokens_content + n_unique_tokens + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_videos + 
    ##     kw_min_min + kw_avg_min + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     self_reference_avg_sharess + LDA_01 + LDA_02 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     avg_positive_polarity + max_positive_polarity + abs_title_subjectivity
    ## 
    ##                              Df Sum of Sq    RSS     AIC
    ## - kw_min_min                  1    0.2194 2007.2 -45.871
    ## - global_subjectivity         1    0.3363 2007.3 -45.748
    ## - self_reference_max_shares   1    0.5736 2007.5 -45.500
    ## - global_rate_positive_words  1    0.6601 2007.6 -45.410
    ## - LDA_01                      1    0.7845 2007.8 -45.280
    ## - kw_avg_min                  1    0.8821 2007.9 -45.178
    ## - num_self_hrefs              1    0.8885 2007.9 -45.171
    ## - LDA_02                      1    0.9183 2007.9 -45.140
    ## - n_unique_tokens             1    1.0072 2008.0 -45.047
    ## - self_reference_avg_sharess  1    1.1191 2008.1 -44.930
    ## - avg_positive_polarity       1    1.8035 2008.8 -44.215
    ## <none>                                    2007.0 -44.100
    ## - global_sentiment_polarity   1    1.9484 2008.9 -44.063
    ## - abs_title_subjectivity      1    2.4666 2009.4 -43.522
    ## - max_positive_polarity       1    2.7500 2009.7 -43.226
    ## - self_reference_min_shares   1    3.7096 2010.7 -42.224
    ## - num_hrefs                   1    3.7742 2010.8 -42.156
    ## - kw_avg_max                  1    4.1268 2011.1 -41.788
    ## - n_non_stop_unique_tokens    1    6.6462 2013.6 -39.161
    ## - kw_max_avg                  1    7.4037 2014.4 -38.371
    ## - kw_min_avg                  1    8.5270 2015.5 -37.201
    ## - n_non_stop_words            1    8.7404 2015.7 -36.979
    ## - n_tokens_content            1   11.2438 2018.2 -34.373
    ## - num_videos                  1   14.9298 2021.9 -30.543
    ## - kw_avg_avg                  1   16.9032 2023.9 -28.496
    ## 
    ## Step:  AIC=-45.87
    ## shares ~ n_tokens_content + n_unique_tokens + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_videos + 
    ##     kw_avg_min + kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_01 + LDA_02 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_positive_words + avg_positive_polarity + max_positive_polarity + 
    ##     abs_title_subjectivity
    ## 
    ##                              Df Sum of Sq    RSS     AIC
    ## - global_subjectivity         1    0.3343 2007.5 -47.521
    ## - self_reference_max_shares   1    0.5865 2007.8 -47.257
    ## - global_rate_positive_words  1    0.6479 2007.8 -47.193
    ## - LDA_01                      1    0.7563 2008.0 -47.080
    ## - kw_avg_min                  1    0.8791 2008.1 -46.951
    ## - LDA_02                      1    0.8872 2008.1 -46.943
    ## - num_self_hrefs              1    0.9191 2008.1 -46.910
    ## - n_unique_tokens             1    1.0318 2008.2 -46.792
    ## - self_reference_avg_sharess  1    1.1271 2008.3 -46.692
    ## - avg_positive_polarity       1    1.7988 2009.0 -45.990
    ## <none>                                    2007.2 -45.871
    ## - global_sentiment_polarity   1    1.9275 2009.1 -45.856
    ## - abs_title_subjectivity      1    2.5344 2009.7 -45.222
    ## - max_positive_polarity       1    2.6977 2009.9 -45.051
    ## - self_reference_min_shares   1    3.7123 2010.9 -43.992
    ## - num_hrefs                   1    3.8863 2011.1 -43.810
    ## - kw_avg_max                  1    4.5040 2011.7 -43.166
    ## - n_non_stop_unique_tokens    1    6.7172 2013.9 -40.858
    ## - kw_max_avg                  1    7.4879 2014.7 -40.055
    ## - n_non_stop_words            1    8.6924 2015.9 -38.800
    ## - kw_min_avg                  1    8.9549 2016.2 -38.527
    ## - n_tokens_content            1   11.3863 2018.6 -35.997
    ## - num_videos                  1   14.8225 2022.0 -32.427
    ## - kw_avg_avg                  1   17.1434 2024.3 -30.019
    ## 
    ## Step:  AIC=-47.52
    ## shares ~ n_tokens_content + n_unique_tokens + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_videos + 
    ##     kw_avg_min + kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_01 + LDA_02 + global_sentiment_polarity + global_rate_positive_words + 
    ##     avg_positive_polarity + max_positive_polarity + abs_title_subjectivity
    ## 
    ##                              Df Sum of Sq    RSS     AIC
    ## - self_reference_max_shares   1    0.5646 2008.1 -48.931
    ## - LDA_01                      1    0.7615 2008.3 -48.725
    ## - global_rate_positive_words  1    0.9134 2008.5 -48.566
    ## - kw_avg_min                  1    0.9253 2008.5 -48.554
    ## - num_self_hrefs              1    0.9761 2008.5 -48.501
    ## - LDA_02                      1    0.9846 2008.5 -48.492
    ## - n_unique_tokens             1    1.0174 2008.5 -48.458
    ## - self_reference_avg_sharess  1    1.0858 2008.6 -48.386
    ## <none>                                    2007.5 -47.521
    ## - global_sentiment_polarity   1    1.9457 2009.5 -47.488
    ## - abs_title_subjectivity      1    2.5411 2010.1 -46.866
    ## - avg_positive_polarity       1    2.6958 2010.2 -46.704
    ## - max_positive_polarity       1    2.9089 2010.4 -46.482
    ## - self_reference_min_shares   1    3.6540 2011.2 -45.704
    ## - num_hrefs                   1    4.1746 2011.7 -45.161
    ## - kw_avg_max                  1    4.5759 2012.1 -44.742
    ## - n_non_stop_unique_tokens    1    6.5739 2014.1 -42.659
    ## - kw_max_avg                  1    7.6073 2015.1 -41.582
    ## - n_non_stop_words            1    8.4193 2016.0 -40.737
    ## - kw_min_avg                  1    8.9295 2016.5 -40.205
    ## - n_tokens_content            1   11.2804 2018.8 -37.760
    ## - num_videos                  1   14.9024 2022.4 -33.997
    ## - kw_avg_avg                  1   17.4711 2025.0 -31.333
    ## 
    ## Step:  AIC=-48.93
    ## shares ~ n_tokens_content + n_unique_tokens + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_videos + 
    ##     kw_avg_min + kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + self_reference_avg_sharess + 
    ##     LDA_01 + LDA_02 + global_sentiment_polarity + global_rate_positive_words + 
    ##     avg_positive_polarity + max_positive_polarity + abs_title_subjectivity
    ## 
    ##                              Df Sum of Sq    RSS     AIC
    ## - LDA_01                      1    0.7656 2008.9 -50.131
    ## - num_self_hrefs              1    0.7806 2008.9 -50.115
    ## - global_rate_positive_words  1    0.9376 2009.0 -49.951
    ## - kw_avg_min                  1    0.9437 2009.0 -49.945
    ## - LDA_02                      1    0.9789 2009.1 -49.908
    ## - n_unique_tokens             1    1.0070 2009.1 -49.878
    ## - self_reference_avg_sharess  1    1.7722 2009.9 -49.079
    ## <none>                                    2008.1 -48.931
    ## - global_sentiment_polarity   1    1.9911 2010.1 -48.851
    ## - abs_title_subjectivity      1    2.6230 2010.7 -48.191
    ## - avg_positive_polarity       1    2.7033 2010.8 -48.107
    ## - max_positive_polarity       1    2.8789 2011.0 -47.924
    ## - num_hrefs                   1    4.1702 2012.3 -46.576
    ## - kw_avg_max                  1    4.5891 2012.7 -46.139
    ## - self_reference_min_shares   1    6.3341 2014.4 -44.320
    ## - n_non_stop_unique_tokens    1    6.5864 2014.7 -44.058
    ## - kw_max_avg                  1    7.5698 2015.7 -43.033
    ## - n_non_stop_words            1    8.5183 2016.6 -42.046
    ## - kw_min_avg                  1    8.9475 2017.0 -41.599
    ## - n_tokens_content            1   11.3327 2019.4 -39.118
    ## - num_videos                  1   14.6746 2022.8 -35.648
    ## - kw_avg_avg                  1   17.4672 2025.6 -32.752
    ## 
    ## Step:  AIC=-50.13
    ## shares ~ n_tokens_content + n_unique_tokens + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_videos + 
    ##     kw_avg_min + kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + self_reference_avg_sharess + 
    ##     LDA_02 + global_sentiment_polarity + global_rate_positive_words + 
    ##     avg_positive_polarity + max_positive_polarity + abs_title_subjectivity
    ## 
    ##                              Df Sum of Sq    RSS     AIC
    ## - LDA_02                      1    0.8264 2009.7 -51.267
    ## - num_self_hrefs              1    0.8826 2009.7 -51.209
    ## - kw_avg_min                  1    0.9505 2009.8 -51.138
    ## - global_rate_positive_words  1    0.9871 2009.8 -51.099
    ## - n_unique_tokens             1    1.0533 2009.9 -51.030
    ## - self_reference_avg_sharess  1    1.7640 2010.6 -50.288
    ## - global_sentiment_polarity   1    1.8958 2010.8 -50.151
    ## <none>                                    2008.9 -50.131
    ## - abs_title_subjectivity      1    2.6495 2011.5 -49.364
    ## - avg_positive_polarity       1    2.7761 2011.6 -49.232
    ## - max_positive_polarity       1    2.9841 2011.8 -49.015
    ## - num_hrefs                   1    4.4321 2013.3 -47.505
    ## - kw_avg_max                  1    4.6235 2013.5 -47.305
    ## - self_reference_min_shares   1    6.3481 2015.2 -45.508
    ## - n_non_stop_unique_tokens    1    6.7373 2015.6 -45.103
    ## - kw_max_avg                  1    7.6354 2016.5 -44.168
    ## - kw_min_avg                  1    8.6633 2017.5 -43.098
    ## - n_non_stop_words            1    8.7416 2017.6 -43.017
    ## - n_tokens_content            1   11.3393 2020.2 -40.316
    ## - num_videos                  1   14.5572 2023.4 -36.975
    ## - kw_avg_avg                  1   17.6271 2026.5 -33.793
    ## 
    ## Step:  AIC=-51.27
    ## shares ~ n_tokens_content + n_unique_tokens + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_videos + 
    ##     kw_avg_min + kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + self_reference_avg_sharess + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     avg_positive_polarity + max_positive_polarity + abs_title_subjectivity
    ## 
    ##                              Df Sum of Sq    RSS     AIC
    ## - num_self_hrefs              1    0.7471 2010.4 -52.487
    ## - kw_avg_min                  1    1.0344 2010.7 -52.187
    ## - n_unique_tokens             1    1.0588 2010.8 -52.162
    ## - global_rate_positive_words  1    1.0722 2010.8 -52.148
    ## - self_reference_avg_sharess  1    1.7064 2011.4 -51.486
    ## - global_sentiment_polarity   1    1.8218 2011.5 -51.365
    ## <none>                                    2009.7 -51.267
    ## - abs_title_subjectivity      1    2.6611 2012.3 -50.490
    ## - avg_positive_polarity       1    2.9259 2012.6 -50.214
    ## - max_positive_polarity       1    3.0823 2012.8 -50.051
    ## - num_hrefs                   1    4.8526 2014.5 -48.205
    ## - kw_avg_max                  1    5.6371 2015.3 -47.388
    ## - self_reference_min_shares   1    6.2193 2015.9 -46.782
    ## - n_non_stop_unique_tokens    1    6.8601 2016.5 -46.115
    ## - kw_max_avg                  1    8.1516 2017.8 -44.771
    ## - kw_min_avg                  1    8.7093 2018.4 -44.191
    ## - n_non_stop_words            1    9.0314 2018.7 -43.856
    ## - n_tokens_content            1   11.2161 2020.9 -41.585
    ## - num_videos                  1   14.5042 2024.2 -38.173
    ## - kw_avg_avg                  1   19.1246 2028.8 -33.387
    ## 
    ## Step:  AIC=-52.49
    ## shares ~ n_tokens_content + n_unique_tokens + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_videos + kw_avg_min + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_avg_sharess + global_sentiment_polarity + 
    ##     global_rate_positive_words + avg_positive_polarity + max_positive_polarity + 
    ##     abs_title_subjectivity
    ## 
    ##                              Df Sum of Sq    RSS     AIC
    ## - n_unique_tokens             1    1.0374 2011.5 -53.404
    ## - global_rate_positive_words  1    1.0914 2011.5 -53.348
    ## - kw_avg_min                  1    1.0932 2011.5 -53.346
    ## <none>                                    2010.4 -52.487
    ## - global_sentiment_polarity   1    1.9343 2012.4 -52.469
    ## - self_reference_avg_sharess  1    1.9401 2012.4 -52.463
    ## - abs_title_subjectivity      1    2.6543 2013.1 -51.718
    ## - max_positive_polarity       1    3.0919 2013.5 -51.262
    ## - avg_positive_polarity       1    3.1639 2013.6 -51.186
    ## - num_hrefs                   1    4.2377 2014.7 -50.067
    ## - kw_avg_max                  1    5.3355 2015.8 -48.924
    ## - self_reference_min_shares   1    6.5554 2017.0 -47.654
    ## - n_non_stop_unique_tokens    1    6.9027 2017.3 -47.293
    ## - kw_max_avg                  1    8.5370 2019.0 -45.593
    ## - kw_min_avg                  1    8.9693 2019.4 -45.144
    ## - n_non_stop_words            1    9.4192 2019.8 -44.676
    ## - n_tokens_content            1   10.8918 2021.3 -43.146
    ## - num_videos                  1   14.4078 2024.8 -39.498
    ## - kw_avg_avg                  1   19.9436 2030.4 -33.768
    ## 
    ## Step:  AIC=-53.4
    ## shares ~ n_tokens_content + n_non_stop_words + n_non_stop_unique_tokens + 
    ##     num_hrefs + num_videos + kw_avg_min + kw_avg_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + self_reference_avg_sharess + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     avg_positive_polarity + max_positive_polarity + abs_title_subjectivity
    ## 
    ##                              Df Sum of Sq    RSS     AIC
    ## - kw_avg_min                  1    0.9971 2012.5 -54.364
    ## - global_rate_positive_words  1    1.1698 2012.6 -54.184
    ## - self_reference_avg_sharess  1    1.8653 2013.3 -53.459
    ## <none>                                    2011.5 -53.404
    ## - global_sentiment_polarity   1    2.2423 2013.7 -53.066
    ## - max_positive_polarity       1    2.3509 2013.8 -52.953
    ## - abs_title_subjectivity      1    2.7920 2014.3 -52.493
    ## - avg_positive_polarity       1    2.8045 2014.3 -52.480
    ## - num_hrefs                   1    3.7187 2015.2 -51.527
    ## - kw_avg_max                  1    5.3579 2016.8 -49.821
    ## - self_reference_min_shares   1    6.4071 2017.9 -48.729
    ## - kw_max_avg                  1    8.4904 2020.0 -46.563
    ## - kw_min_avg                  1    9.2741 2020.8 -45.749
    ## - n_non_stop_unique_tokens    1    9.2901 2020.8 -45.732
    ## - n_non_stop_words            1   10.5855 2022.1 -44.387
    ## - num_videos                  1   14.2581 2025.7 -40.578
    ## - n_tokens_content            1   18.8209 2030.3 -35.856
    ## - kw_avg_avg                  1   19.5627 2031.0 -35.089
    ## 
    ## Step:  AIC=-54.36
    ## shares ~ n_tokens_content + n_non_stop_words + n_non_stop_unique_tokens + 
    ##     num_hrefs + num_videos + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_avg_sharess + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     avg_positive_polarity + max_positive_polarity + abs_title_subjectivity
    ## 
    ##                              Df Sum of Sq    RSS     AIC
    ## - global_rate_positive_words  1    1.2270 2013.7 -55.085
    ## - self_reference_avg_sharess  1    1.8973 2014.4 -54.386
    ## <none>                                    2012.5 -54.364
    ## - global_sentiment_polarity   1    2.2408 2014.7 -54.028
    ## - max_positive_polarity       1    2.3204 2014.8 -53.945
    ## - abs_title_subjectivity      1    2.8701 2015.3 -53.373
    ## - avg_positive_polarity       1    2.9351 2015.4 -53.305
    ## - num_hrefs                   1    3.9540 2016.4 -52.244
    ## - kw_avg_max                  1    4.5083 2017.0 -51.667
    ## - self_reference_min_shares   1    6.3101 2018.8 -49.793
    ## - kw_min_avg                  1    9.3476 2021.8 -46.637
    ## - n_non_stop_unique_tokens    1    9.3637 2021.8 -46.620
    ## - n_non_stop_words            1   10.7187 2023.2 -45.214
    ## - kw_max_avg                  1   11.4058 2023.9 -44.501
    ## - num_videos                  1   13.8106 2026.3 -42.009
    ## - n_tokens_content            1   18.8429 2031.3 -36.802
    ## - kw_avg_avg                  1   19.4888 2032.0 -36.135
    ## 
    ## Step:  AIC=-55.08
    ## shares ~ n_tokens_content + n_non_stop_words + n_non_stop_unique_tokens + 
    ##     num_hrefs + num_videos + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_avg_sharess + 
    ##     global_sentiment_polarity + avg_positive_polarity + max_positive_polarity + 
    ##     abs_title_subjectivity
    ## 
    ##                              Df Sum of Sq    RSS     AIC
    ## - global_sentiment_polarity   1    1.1077 2014.8 -55.930
    ## - max_positive_polarity       1    1.6429 2015.3 -55.373
    ## - self_reference_avg_sharess  1    1.8633 2015.6 -55.143
    ## <none>                                    2013.7 -55.085
    ## - avg_positive_polarity       1    2.0071 2015.7 -54.994
    ## - abs_title_subjectivity      1    2.4242 2016.1 -54.559
    ## - num_hrefs                   1    3.8927 2017.6 -53.031
    ## - kw_avg_max                  1    4.5461 2018.2 -52.351
    ## - self_reference_min_shares   1    6.1519 2019.8 -50.682
    ## - kw_min_avg                  1    9.5414 2023.2 -47.163
    ## - n_non_stop_words            1    9.8500 2023.5 -46.842
    ## - n_non_stop_unique_tokens    1   10.0757 2023.8 -46.608
    ## - kw_max_avg                  1   11.6842 2025.4 -44.941
    ## - num_videos                  1   13.6549 2027.3 -42.899
    ## - n_tokens_content            1   19.2846 2033.0 -37.079
    ## - kw_avg_avg                  1   20.1069 2033.8 -36.230
    ## 
    ## Step:  AIC=-55.93
    ## shares ~ n_tokens_content + n_non_stop_words + n_non_stop_unique_tokens + 
    ##     num_hrefs + num_videos + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_avg_sharess + 
    ##     avg_positive_polarity + max_positive_polarity + abs_title_subjectivity
    ## 
    ##                              Df Sum of Sq    RSS     AIC
    ## - avg_positive_polarity       1    1.1843 2016.0 -56.697
    ## - self_reference_avg_sharess  1    1.8163 2016.6 -56.039
    ## <none>                                    2014.8 -55.930
    ## - max_positive_polarity       1    2.2034 2017.0 -55.636
    ## - abs_title_subjectivity      1    2.6648 2017.5 -55.156
    ## - num_hrefs                   1    3.6259 2018.4 -54.156
    ## - kw_avg_max                  1    4.1924 2019.0 -53.567
    ## - self_reference_min_shares   1    6.2143 2021.0 -51.466
    ## - n_non_stop_words            1    9.4596 2024.3 -48.099
    ## - kw_min_avg                  1    9.6608 2024.5 -47.890
    ## - n_non_stop_unique_tokens    1   10.4496 2025.2 -47.072
    ## - kw_max_avg                  1   11.7590 2026.6 -45.716
    ## - num_videos                  1   13.6743 2028.5 -43.733
    ## - n_tokens_content            1   20.1057 2034.9 -37.088
    ## - kw_avg_avg                  1   20.2597 2035.1 -36.929
    ## 
    ## Step:  AIC=-56.7
    ## shares ~ n_tokens_content + n_non_stop_words + n_non_stop_unique_tokens + 
    ##     num_hrefs + num_videos + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_avg_sharess + 
    ##     max_positive_polarity + abs_title_subjectivity
    ## 
    ##                              Df Sum of Sq    RSS     AIC
    ## - max_positive_polarity       1    1.1430 2017.1 -57.507
    ## - self_reference_avg_sharess  1    1.8515 2017.8 -56.770
    ## <none>                                    2016.0 -56.697
    ## - abs_title_subjectivity      1    2.7014 2018.7 -55.886
    ## - num_hrefs                   1    4.1133 2020.1 -54.419
    ## - kw_avg_max                  1    4.3885 2020.4 -54.133
    ## - self_reference_min_shares   1    6.2233 2022.2 -52.227
    ## - n_non_stop_words            1    8.7004 2024.7 -49.658
    ## - kw_min_avg                  1    9.5559 2025.5 -48.771
    ## - n_non_stop_unique_tokens    1   11.1408 2027.1 -47.129
    ## - kw_max_avg                  1   12.0817 2028.1 -46.155
    ## - num_videos                  1   13.6724 2029.7 -44.510
    ## - n_tokens_content            1   19.1680 2035.2 -38.834
    ## - kw_avg_avg                  1   20.7681 2036.8 -37.184
    ## 
    ## Step:  AIC=-57.51
    ## shares ~ n_tokens_content + n_non_stop_words + n_non_stop_unique_tokens + 
    ##     num_hrefs + num_videos + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_avg_sharess + 
    ##     abs_title_subjectivity
    ## 
    ##                              Df Sum of Sq    RSS     AIC
    ## <none>                                    2017.1 -57.507
    ## - self_reference_avg_sharess  1    1.9502 2019.1 -57.479
    ## - abs_title_subjectivity      1    2.8039 2019.9 -56.591
    ## - num_hrefs                   1    3.6149 2020.8 -55.749
    ## - kw_avg_max                  1    4.3537 2021.5 -54.982
    ## - self_reference_min_shares   1    6.4350 2023.6 -52.822
    ## - kw_min_avg                  1    9.5063 2026.6 -49.638
    ## - kw_max_avg                  1   11.8887 2029.0 -47.172
    ## - n_non_stop_unique_tokens    1   12.6673 2029.8 -46.367
    ## - n_non_stop_words            1   13.2371 2030.4 -45.778
    ## - num_videos                  1   13.4175 2030.5 -45.591
    ## - n_tokens_content            1   18.2157 2035.3 -40.637
    ## - kw_avg_avg                  1   20.2892 2037.4 -38.500

``` r
lm$call[["formula"]] # model selected based on AIC. 
```

    ## shares ~ n_tokens_content + n_non_stop_words + n_non_stop_unique_tokens + 
    ##     num_hrefs + num_videos + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_avg_sharess + 
    ##     abs_title_subjectivity
    ## <environment: 0x7f9b9afa01c0>

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
    ##                  305                  322                  182                  210 
    ##  weekday_is_thursday   weekday_is_tuesday weekday_is_wednesday 
    ##                  358                  334                  388

``` r
C2 <- pop.data %>% 
    group_by( weekday ) %>% 
    summarise( percent = 100 * n() / nrow( X ),mean_shares = mean(shares), mean_images = mean(num_imgs),mean_video = mean(num_videos),mean_link = mean(num_hrefs))
knitr::kable(C2)
```

| weekday                |   percent | mean\_shares | mean\_images | mean\_video | mean\_link |
| :--------------------- | --------: | -----------: | -----------: | ----------: | ---------: |
| weekday\_is\_friday    | 0.9101761 |     3025.869 |     3.940984 |   0.4229508 |   13.56721 |
| weekday\_is\_monday    | 0.9609072 |     4345.711 |     4.124224 |   0.6211180 |   12.22050 |
| weekday\_is\_saturday  | 0.5431215 |     4062.451 |     8.510989 |   0.6043956 |   17.15934 |
| weekday\_is\_sunday    | 0.6266786 |     3790.376 |     8.595238 |   0.4571429 |   17.43333 |
| weekday\_is\_thursday  | 1.0683378 |     3500.268 |     3.927374 |   0.4022346 |   11.76257 |
| weekday\_is\_tuesday   | 0.9967174 |     4152.494 |     3.562874 |   0.4580838 |   11.89222 |
| weekday\_is\_wednesday | 1.1578633 |     3173.180 |     4.677835 |   0.4252577 |   13.21392 |

``` r
table(pop.data$weekday, pop.data$channel)
```

    ##                       
    ##                        data_channel_is_lifestyle
    ##   weekday_is_friday                          305
    ##   weekday_is_monday                          322
    ##   weekday_is_saturday                        182
    ##   weekday_is_sunday                          210
    ##   weekday_is_thursday                        358
    ##   weekday_is_tuesday                         334
    ##   weekday_is_wednesday                       388

``` r
table(pop.data$channel, pop.data$is_weekend)
```

    ##                            
    ##                                0    1
    ##   data_channel_is_lifestyle 1707  392

``` r
C3 <- pop.data %>% group_by(is_weekend)%>% 
    summarise( percent = 100 * n() / nrow( X ),mean_shares = mean(shares), mean_images = mean(num_imgs),mean_video = mean(num_videos),mean_link = mean(num_hrefs))
knitr::kable(C3)
```

| is\_weekend |  percent | mean\_shares | mean\_images | mean\_video | mean\_link |
| :---------- | -------: | -----------: | -----------: | ----------: | ---------: |
| 0           | 5.094002 |     3628.255 |     4.066198 |   0.4633861 |   12.52665 |
| 1           | 1.169800 |     3916.696 |     8.556122 |   0.5255102 |   17.30612 |

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
| TRUE        | FALSE              | FALSE               | FALSE                        | FALSE      | FALSE     | FALSE         | TRUE        | FALSE        | FALSE        | FALSE        | FALSE        | FALSE                        | FALSE                         | FALSE                         | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | TRUE               | FALSE               | FALSE                        | FALSE      | FALSE     | FALSE         | TRUE        | FALSE        | FALSE        | FALSE        | FALSE        | FALSE                        | FALSE                         | FALSE                         | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | TRUE               | FALSE               | FALSE                        | FALSE      | FALSE     | FALSE         | TRUE        | FALSE        | FALSE        | FALSE        | TRUE         | FALSE                        | FALSE                         | FALSE                         | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | TRUE               | FALSE               | FALSE                        | FALSE      | FALSE     | FALSE         | TRUE        | FALSE        | FALSE        | FALSE        | TRUE         | TRUE                         | FALSE                         | FALSE                         | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | TRUE               | FALSE               | FALSE                        | FALSE      | FALSE     | FALSE         | TRUE        | FALSE        | TRUE         | TRUE         | TRUE         | FALSE                        | FALSE                         | FALSE                         | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | TRUE               | FALSE               | FALSE                        | FALSE      | FALSE     | FALSE         | TRUE        | TRUE         | TRUE         | TRUE         | TRUE         | FALSE                        | FALSE                         | FALSE                         | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | TRUE               | FALSE               | FALSE                        | FALSE      | FALSE     | FALSE         | TRUE        | TRUE         | TRUE         | TRUE         | TRUE         | TRUE                         | FALSE                         | FALSE                         | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | TRUE               | TRUE                | TRUE                         | FALSE      | FALSE     | FALSE         | TRUE        | TRUE         | TRUE         | TRUE         | TRUE         | FALSE                        | FALSE                         | FALSE                         | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | TRUE               | TRUE                | TRUE                         | FALSE      | FALSE     | FALSE         | TRUE        | TRUE         | TRUE         | TRUE         | TRUE         | TRUE                         | FALSE                         | FALSE                         | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | TRUE               | TRUE                | TRUE                         | TRUE       | FALSE     | FALSE         | TRUE        | TRUE         | TRUE         | TRUE         | TRUE         | TRUE                         | FALSE                         | FALSE                         | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | TRUE               | TRUE                | TRUE                         | TRUE       | FALSE     | FALSE         | TRUE        | TRUE         | TRUE         | TRUE         | TRUE         | TRUE                         | TRUE                          | FALSE                         | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | TRUE               | TRUE                | TRUE                         | TRUE       | FALSE     | FALSE         | TRUE        | TRUE         | TRUE         | TRUE         | TRUE         | TRUE                         | TRUE                          | FALSE                         | FALSE                 | TRUE                     | FALSE                           |
| TRUE        | TRUE               | TRUE                | TRUE                         | TRUE       | TRUE      | FALSE         | TRUE        | TRUE         | TRUE         | TRUE         | TRUE         | TRUE                         | TRUE                          | FALSE                         | FALSE                 | TRUE                     | FALSE                           |
| TRUE        | TRUE               | TRUE                | TRUE                         | TRUE       | TRUE      | FALSE         | TRUE        | TRUE         | TRUE         | TRUE         | TRUE         | TRUE                         | TRUE                          | FALSE                         | TRUE                  | TRUE                     | FALSE                           |
| TRUE        | TRUE               | TRUE                | TRUE                         | TRUE       | TRUE      | FALSE         | TRUE        | TRUE         | TRUE         | TRUE         | TRUE         | TRUE                         | TRUE                          | TRUE                          | TRUE                  | TRUE                     | FALSE                           |
| TRUE        | TRUE               | TRUE                | TRUE                         | TRUE       | TRUE      | FALSE         | TRUE        | TRUE         | TRUE         | TRUE         | TRUE         | TRUE                         | TRUE                          | TRUE                          | TRUE                  | TRUE                     | TRUE                            |
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
|     12 |  9 |   3 |

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
| lm.fit1 |     9425.336 |        0.0157720 |    3375.776 |
| lm.fit2 |     9192.519 |        0.0093419 |    3407.919 |
| lm.fit3 |     9528.363 |        0.0286152 |    3934.951 |

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
    ## (Intercept)                   3.664136e+03
    ## n_tokens_title                .           
    ## n_tokens_content              8.350624e-01
    ## n_unique_tokens               .           
    ## n_non_stop_words             -1.547236e+03
    ## n_non_stop_unique_tokens      .           
    ## num_hrefs                     8.092090e-01
    ## num_self_hrefs                .           
    ## num_imgs                      .           
    ## num_videos                    5.152645e+02
    ## average_token_length          .           
    ## num_keywords                  .           
    ## kw_min_min                    .           
    ## kw_max_min                    .           
    ## kw_avg_min                    .           
    ## kw_min_max                    .           
    ## kw_max_max                    .           
    ## kw_avg_max                    .           
    ## kw_min_avg                    .           
    ## kw_max_avg                    .           
    ## kw_avg_avg                    1.395324e-01
    ## self_reference_min_shares     6.654685e-02
    ## self_reference_max_shares     .           
    ## self_reference_avg_sharess    .           
    ## is_weekend                    .           
    ## LDA_00                        .           
    ## LDA_01                        .           
    ## LDA_02                       -1.161369e+03
    ## LDA_03                        .           
    ## LDA_04                        .           
    ## global_subjectivity           .           
    ## global_sentiment_polarity     .           
    ## global_rate_positive_words    .           
    ## global_rate_negative_words    .           
    ## rate_positive_words           .           
    ## rate_negative_words           .           
    ## avg_positive_polarity         .           
    ## min_positive_polarity         .           
    ## max_positive_polarity         .           
    ## avg_negative_polarity         .           
    ## min_negative_polarity         .           
    ## max_negative_polarity         .           
    ## title_subjectivity            .           
    ## title_sentiment_polarity      .           
    ## abs_title_subjectivity        2.477462e+02
    ## abs_title_sentiment_polarity  .

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
    ## (Intercept)                  1.769446e+03
    ## n_tokens_content             9.463247e-01
    ## n_non_stop_words             .           
    ## n_non_stop_unique_tokens     .           
    ## num_hrefs                    .           
    ## num_imgs                     .           
    ## num_keywords                 .           
    ## num_videos                   2.495265e+02
    ## kw_avg_max                   .           
    ## kw_min_avg                   .           
    ## kw_max_avg                   .           
    ## kw_avg_avg                   3.086405e-01
    ## self_reference_min_shares    2.296196e-02
    ## self_reference_avg_sharess   .           
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
| lm.fit1        |   6233.417 |
| lm.fit2        |   6214.566 |
| lm.fit3        |   7588.278 |
| lasso.fit.full |   9931.049 |
| lasso.fit.18   |   6232.371 |
| rf.fit1        |   6216.549 |
| rf.fit2        |   6264.584 |
| boostTreefit1  |   6239.874 |
| boostTreefit2  |   6247.875 |

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
