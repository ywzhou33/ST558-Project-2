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

    ## tibble [2,323 × 48] (S3: tbl_df/tbl/data.frame)
    ##  $ n_tokens_title              : num [1:2323] 8 8 9 10 9 9 10 7 8 6 ...
    ##  $ n_tokens_content            : num [1:2323] 257 218 1226 1121 168 ...
    ##  $ n_unique_tokens             : num [1:2323] 0.568 0.663 0.41 0.451 0.778 ...
    ##  $ n_non_stop_words            : num [1:2323] 1 1 1 1 1 ...
    ##  $ n_non_stop_unique_tokens    : num [1:2323] 0.671 0.688 0.617 0.629 0.865 ...
    ##  $ num_hrefs                   : num [1:2323] 9 14 10 15 6 3 19 11 4 24 ...
    ##  $ num_self_hrefs              : num [1:2323] 7 3 10 11 4 2 10 1 4 6 ...
    ##  $ num_imgs                    : num [1:2323] 0 11 1 1 11 1 8 1 1 1 ...
    ##  $ num_videos                  : num [1:2323] 1 0 1 0 0 0 0 0 0 0 ...
    ##  $ average_token_length        : num [1:2323] 4.64 4.44 4.39 4.79 4.68 ...
    ##  $ num_keywords                : num [1:2323] 9 10 7 6 9 6 6 7 4 8 ...
    ##  $ kw_min_min                  : num [1:2323] 0 0 0 0 217 217 217 217 217 217 ...
    ##  $ kw_max_min                  : num [1:2323] 0 0 0 0 690 690 690 4800 1900 737 ...
    ##  $ kw_avg_min                  : num [1:2323] 0 0 0 0 572 ...
    ##  $ kw_min_max                  : num [1:2323] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kw_max_max                  : num [1:2323] 0 0 0 0 17100 17100 17100 28000 28000 28000 ...
    ##  $ kw_avg_max                  : num [1:2323] 0 0 0 0 3110 ...
    ##  $ kw_min_avg                  : num [1:2323] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kw_max_avg                  : num [1:2323] 0 0 0 0 2322 ...
    ##  $ kw_avg_avg                  : num [1:2323] 0 0 0 0 832 ...
    ##  $ self_reference_min_shares   : num [1:2323] 1300 3900 992 757 6600 1800 1200 3500 4500 1600 ...
    ##  $ self_reference_max_shares   : num [1:2323] 2500 3900 4700 5400 6600 1800 3500 3500 15300 1600 ...
    ##  $ self_reference_avg_sharess  : num [1:2323] 1775 3900 2858 2796 6600 ...
    ##  $ is_weekend                  : Factor w/ 2 levels "0","1": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ LDA_00                      : num [1:2323] 0.4392 0.1993 0.0298 0.0355 0.0231 ...
    ##  $ LDA_01                      : num [1:2323] 0.0225 0.2477 0.1939 0.0338 0.0223 ...
    ##  $ LDA_02                      : num [1:2323] 0.0224 0.0201 0.0288 0.0336 0.0224 ...
    ##  $ LDA_03                      : num [1:2323] 0.0233 0.5127 0.7181 0.863 0.9096 ...
    ##  $ LDA_04                      : num [1:2323] 0.4926 0.0202 0.0293 0.0341 0.0226 ...
    ##  $ global_subjectivity         : num [1:2323] 0.4 0.522 0.408 0.497 0.638 ...
    ##  $ global_sentiment_polarity   : num [1:2323] 0.00741 0.29912 0.10661 0.15961 0.08798 ...
    ##  $ global_rate_positive_words  : num [1:2323] 0.0311 0.055 0.0228 0.0562 0.0714 ...
    ##  $ global_rate_negative_words  : num [1:2323] 0.0272 0.0183 0.0114 0.0134 0.0476 ...
    ##  $ rate_positive_words         : num [1:2323] 0.533 0.75 0.667 0.808 0.6 ...
    ##  $ rate_negative_words         : num [1:2323] 0.467 0.25 0.333 0.192 0.4 ...
    ##  $ avg_positive_polarity       : num [1:2323] 0.36 0.536 0.395 0.372 0.492 ...
    ##  $ min_positive_polarity       : num [1:2323] 0.0333 0.1 0.0625 0.0333 0.1 ...
    ##  $ max_positive_polarity       : num [1:2323] 0.6 1 1 1 1 0.35 1 1 0.55 0.8 ...
    ##  $ avg_negative_polarity       : num [1:2323] -0.393 -0.237 -0.258 -0.317 -0.502 ...
    ##  $ min_negative_polarity       : num [1:2323] -0.5 -0.25 -1 -0.8 -1 0 -0.6 -1 -0.7 -0.5 ...
    ##  $ max_negative_polarity       : num [1:2323] -0.125 -0.2 -0.1 -0.15 -0.15 0 -0.05 -0.05 -0.125 -0.05 ...
    ##  $ title_subjectivity          : num [1:2323] 0.667 0.5 0 0 1 ...
    ##  $ title_sentiment_polarity    : num [1:2323] -0.5 0.5 0 0 -1 ...
    ##  $ abs_title_subjectivity      : num [1:2323] 0.167 0 0.5 0.5 0.5 ...
    ##  $ abs_title_sentiment_polarity: num [1:2323] 0.5 0.5 0 0 1 ...
    ##  $ shares                      : num [1:2323] 2600 690 4800 851 4800 9200 1600 775 18200 1600 ...
    ##  $ channel                     : chr [1:2323] "data_channel_is_socmed" "data_channel_is_socmed" "data_channel_is_socmed" "data_channel_is_socmed" ...
    ##  $ weekday                     : Factor w/ 7 levels "weekday_is_friday",..: 2 2 2 2 7 7 7 5 1 1 ...

``` r
#summary stats for the response variable. 
summary(pop.data$shares)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##       5    1400    2100    3629    3800  122800

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

    ## Start:  AIC=-87.81
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
    ## Step:  AIC=-87.81
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
    ## - global_rate_negative_words    1    0.0001 2153.7 -89.808
    ## - self_reference_avg_sharess    1    0.0032 2153.7 -89.805
    ## - n_tokens_content              1    0.0093 2153.7 -89.798
    ## - average_token_length          1    0.0238 2153.7 -89.782
    ## - global_subjectivity           1    0.0285 2153.7 -89.777
    ## - global_rate_positive_words    1    0.0412 2153.7 -89.764
    ## - n_tokens_title                1    0.0453 2153.7 -89.759
    ## - global_sentiment_polarity     1    0.0545 2153.7 -89.749
    ## - avg_positive_polarity         1    0.0581 2153.7 -89.745
    ## - kw_max_max                    1    0.0625 2153.7 -89.741
    ## - kw_min_avg                    1    0.0892 2153.8 -89.712
    ## - rate_positive_words           1    0.1208 2153.8 -89.678
    ## - n_non_stop_words              1    0.1445 2153.8 -89.652
    ## - self_reference_max_shares     1    0.1486 2153.8 -89.648
    ## - rate_negative_words           1    0.2933 2154.0 -89.492
    ## - n_non_stop_unique_tokens      1    0.3172 2154.0 -89.466
    ## - kw_min_max                    1    0.4253 2154.1 -89.349
    ## - LDA_00                        1    0.4526 2154.1 -89.320
    ## - max_negative_polarity         1    0.5786 2154.3 -89.184
    ## - title_sentiment_polarity      1    0.7413 2154.4 -89.009
    ## - abs_title_subjectivity        1    0.9525 2154.6 -88.781
    ## - n_unique_tokens               1    1.0007 2154.7 -88.729
    ## - title_subjectivity            1    1.0143 2154.7 -88.714
    ## - min_positive_polarity         1    1.0281 2154.7 -88.699
    ## - kw_avg_max                    1    1.1245 2154.8 -88.595
    ## - avg_negative_polarity         1    1.2346 2154.9 -88.477
    ## - max_positive_polarity         1    1.5839 2155.3 -88.100
    ## - kw_min_min                    1    1.6335 2155.3 -88.047
    ## - LDA_02                        1    1.6865 2155.4 -87.990
    ## - num_hrefs                     1    1.6921 2155.4 -87.984
    ## <none>                                      2153.7 -87.808
    ## - num_keywords                  1    2.1348 2155.8 -87.507
    ## - self_reference_min_shares     1    2.2983 2156.0 -87.330
    ## - LDA_03                        1    2.3585 2156.0 -87.265
    ## - num_self_hrefs                1    2.5540 2156.2 -87.055
    ## - num_videos                    1    2.6569 2156.3 -86.944
    ## - abs_title_sentiment_polarity  1    3.2289 2156.9 -86.328
    ## - num_imgs                      1    3.7294 2157.4 -85.789
    ## - kw_avg_min                    1    3.7909 2157.5 -85.723
    ## - LDA_01                        1    3.8869 2157.6 -85.619
    ## - kw_max_min                    1    4.6781 2158.4 -84.768
    ## - min_negative_polarity         1    5.5132 2159.2 -83.869
    ## - kw_max_avg                    1    6.8733 2160.6 -82.406
    ## - kw_avg_avg                    1   14.3031 2168.0 -74.431
    ## 
    ## Step:  AIC=-89.81
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + kw_min_min + kw_max_min + kw_avg_min + kw_min_max + 
    ##     kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + self_reference_max_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     rate_positive_words + rate_negative_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - self_reference_avg_sharess    1    0.0032 2153.7 -91.804
    ## - n_tokens_content              1    0.0094 2153.7 -91.798
    ## - average_token_length          1    0.0242 2153.7 -91.782
    ## - global_subjectivity           1    0.0285 2153.7 -91.777
    ## - n_tokens_title                1    0.0454 2153.7 -91.759
    ## - global_sentiment_polarity     1    0.0591 2153.7 -91.744
    ## - avg_positive_polarity         1    0.0599 2153.7 -91.743
    ## - kw_max_max                    1    0.0627 2153.7 -91.740
    ## - global_rate_positive_words    1    0.0734 2153.8 -91.729
    ## - kw_min_avg                    1    0.0892 2153.8 -91.712
    ## - rate_positive_words           1    0.1210 2153.8 -91.677
    ## - n_non_stop_words              1    0.1445 2153.8 -91.652
    ## - self_reference_max_shares     1    0.1486 2153.8 -91.648
    ## - rate_negative_words           1    0.3084 2154.0 -91.475
    ## - n_non_stop_unique_tokens      1    0.3197 2154.0 -91.463
    ## - kw_min_max                    1    0.4253 2154.1 -91.349
    ## - LDA_00                        1    0.4536 2154.1 -91.319
    ## - max_negative_polarity         1    0.5785 2154.3 -91.184
    ## - title_sentiment_polarity      1    0.7446 2154.4 -91.005
    ## - abs_title_subjectivity        1    0.9531 2154.6 -90.780
    ## - n_unique_tokens               1    1.0090 2154.7 -90.720
    ## - title_subjectivity            1    1.0144 2154.7 -90.714
    ## - min_positive_polarity         1    1.0307 2154.7 -90.696
    ## - kw_avg_max                    1    1.1243 2154.8 -90.595
    ## - avg_negative_polarity         1    1.2434 2154.9 -90.467
    ## - max_positive_polarity         1    1.5859 2155.3 -90.098
    ## - kw_min_min                    1    1.6354 2155.3 -90.045
    ## - LDA_02                        1    1.6866 2155.4 -89.989
    ## - num_hrefs                     1    1.6924 2155.4 -89.983
    ## <none>                                      2153.7 -89.808
    ## - num_keywords                  1    2.1347 2155.8 -89.506
    ## - self_reference_min_shares     1    2.2982 2156.0 -89.330
    ## - LDA_03                        1    2.3584 2156.0 -89.265
    ## - num_self_hrefs                1    2.5545 2156.2 -89.054
    ## - num_videos                    1    2.6620 2156.3 -88.938
    ## - abs_title_sentiment_polarity  1    3.2339 2156.9 -88.322
    ## - num_imgs                      1    3.7293 2157.4 -87.789
    ## - kw_avg_min                    1    3.7932 2157.5 -87.720
    ## - LDA_01                        1    3.8890 2157.6 -87.617
    ## - kw_max_min                    1    4.6789 2158.4 -86.767
    ## - min_negative_polarity         1    5.5184 2159.2 -85.863
    ## - kw_max_avg                    1    6.8760 2160.6 -84.403
    ## - kw_avg_avg                    1   14.3039 2168.0 -76.430
    ## 
    ## Step:  AIC=-91.8
    ## shares ~ n_tokens_title + n_tokens_content + n_unique_tokens + 
    ##     n_non_stop_words + n_non_stop_unique_tokens + num_hrefs + 
    ##     num_self_hrefs + num_imgs + num_videos + average_token_length + 
    ##     num_keywords + kw_min_min + kw_max_min + kw_avg_min + kw_min_max + 
    ##     kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + self_reference_max_shares + LDA_00 + 
    ##     LDA_01 + LDA_02 + LDA_03 + global_subjectivity + global_sentiment_polarity + 
    ##     global_rate_positive_words + rate_positive_words + rate_negative_words + 
    ##     avg_positive_polarity + min_positive_polarity + max_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_subjectivity + title_sentiment_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - n_tokens_content              1    0.0092 2153.7 -93.794
    ## - average_token_length          1    0.0244 2153.7 -93.778
    ## - global_subjectivity           1    0.0283 2153.7 -93.774
    ## - n_tokens_title                1    0.0452 2153.7 -93.756
    ## - global_sentiment_polarity     1    0.0590 2153.7 -93.741
    ## - avg_positive_polarity         1    0.0604 2153.7 -93.739
    ## - kw_max_max                    1    0.0630 2153.8 -93.736
    ## - global_rate_positive_words    1    0.0741 2153.8 -93.724
    ## - kw_min_avg                    1    0.0910 2153.8 -93.706
    ## - rate_positive_words           1    0.1212 2153.8 -93.674
    ## - n_non_stop_words              1    0.1452 2153.8 -93.648
    ## - rate_negative_words           1    0.3087 2154.0 -93.471
    ## - n_non_stop_unique_tokens      1    0.3228 2154.0 -93.456
    ## - kw_min_max                    1    0.4271 2154.1 -93.344
    ## - LDA_00                        1    0.4552 2154.1 -93.313
    ## - max_negative_polarity         1    0.5783 2154.3 -93.181
    ## - self_reference_max_shares     1    0.6095 2154.3 -93.147
    ## - title_sentiment_polarity      1    0.7418 2154.4 -93.004
    ## - abs_title_subjectivity        1    0.9527 2154.6 -92.777
    ## - n_unique_tokens               1    1.0077 2154.7 -92.718
    ## - title_subjectivity            1    1.0134 2154.7 -92.712
    ## - min_positive_polarity         1    1.0312 2154.7 -92.692
    ## - kw_avg_max                    1    1.1222 2154.8 -92.594
    ## - avg_negative_polarity         1    1.2426 2154.9 -92.464
    ## - max_positive_polarity         1    1.5830 2155.3 -92.098
    ## - kw_min_min                    1    1.6386 2155.3 -92.038
    ## - LDA_02                        1    1.6843 2155.4 -91.988
    ## - num_hrefs                     1    1.6984 2155.4 -91.973
    ## <none>                                      2153.7 -91.804
    ## - num_keywords                  1    2.1440 2155.8 -91.493
    ## - LDA_03                        1    2.3566 2156.0 -91.264
    ## - num_self_hrefs                1    2.6505 2156.3 -90.947
    ## - num_videos                    1    2.6588 2156.3 -90.938
    ## - abs_title_sentiment_polarity  1    3.2310 2156.9 -90.322
    ## - num_imgs                      1    3.7351 2157.4 -89.779
    ## - kw_avg_min                    1    3.7965 2157.5 -89.713
    ## - LDA_01                        1    3.8890 2157.6 -89.613
    ## - kw_max_min                    1    4.6871 2158.4 -88.754
    ## - min_negative_polarity         1    5.5154 2159.2 -87.863
    ## - kw_max_avg                    1    6.8764 2160.6 -86.399
    ## - self_reference_min_shares     1    8.7440 2162.4 -84.392
    ## - kw_avg_avg                    1   14.3010 2168.0 -78.430
    ## 
    ## Step:  AIC=-93.79
    ## shares ~ n_tokens_title + n_unique_tokens + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + average_token_length + num_keywords + kw_min_min + 
    ##     kw_max_min + kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_00 + LDA_01 + LDA_02 + LDA_03 + 
    ##     global_subjectivity + global_sentiment_polarity + global_rate_positive_words + 
    ##     rate_positive_words + rate_negative_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - average_token_length          1    0.0254 2153.7 -95.767
    ## - global_subjectivity           1    0.0275 2153.7 -95.765
    ## - n_tokens_title                1    0.0477 2153.7 -95.743
    ## - global_sentiment_polarity     1    0.0562 2153.8 -95.734
    ## - kw_max_max                    1    0.0621 2153.8 -95.727
    ## - avg_positive_polarity         1    0.0623 2153.8 -95.727
    ## - global_rate_positive_words    1    0.0725 2153.8 -95.716
    ## - kw_min_avg                    1    0.0880 2153.8 -95.700
    ## - rate_positive_words           1    0.1192 2153.8 -95.666
    ## - n_non_stop_words              1    0.1576 2153.8 -95.624
    ## - rate_negative_words           1    0.3046 2154.0 -95.466
    ## - n_non_stop_unique_tokens      1    0.3137 2154.0 -95.456
    ## - kw_min_max                    1    0.4256 2154.1 -95.335
    ## - LDA_00                        1    0.4502 2154.1 -95.309
    ## - max_negative_polarity         1    0.5961 2154.3 -95.152
    ## - self_reference_max_shares     1    0.6077 2154.3 -95.139
    ## - title_sentiment_polarity      1    0.7507 2154.4 -94.985
    ## - abs_title_subjectivity        1    0.9539 2154.7 -94.766
    ## - title_subjectivity            1    1.0166 2154.7 -94.698
    ## - min_positive_polarity         1    1.0221 2154.7 -94.692
    ## - kw_avg_max                    1    1.1185 2154.8 -94.588
    ## - avg_negative_polarity         1    1.3244 2155.0 -94.366
    ## - n_unique_tokens               1    1.3350 2155.0 -94.355
    ## - max_positive_polarity         1    1.5743 2155.3 -94.097
    ## - kw_min_min                    1    1.6346 2155.3 -94.032
    ## - LDA_02                        1    1.6902 2155.4 -93.972
    ## - num_hrefs                     1    1.8495 2155.5 -93.800
    ## <none>                                      2153.7 -93.794
    ## - num_keywords                  1    2.1399 2155.8 -93.487
    ## - LDA_03                        1    2.3653 2156.1 -93.245
    ## - num_self_hrefs                1    2.6415 2156.3 -92.947
    ## - num_videos                    1    2.7670 2156.5 -92.812
    ## - abs_title_sentiment_polarity  1    3.2379 2156.9 -92.305
    ## - num_imgs                      1    3.7848 2157.5 -91.716
    ## - kw_avg_min                    1    3.7978 2157.5 -91.702
    ## - LDA_01                        1    3.8918 2157.6 -91.601
    ## - kw_max_min                    1    4.6898 2158.4 -90.741
    ## - min_negative_polarity         1    6.0183 2159.7 -89.312
    ## - kw_max_avg                    1    6.8973 2160.6 -88.367
    ## - self_reference_min_shares     1    8.7373 2162.4 -86.389
    ## - kw_avg_avg                    1   14.3324 2168.0 -80.387
    ## 
    ## Step:  AIC=-95.77
    ## shares ~ n_tokens_title + n_unique_tokens + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + num_keywords + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_min_max + kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     LDA_00 + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     rate_positive_words + rate_negative_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - global_subjectivity           1    0.0233 2153.7 -97.742
    ## - n_tokens_title                1    0.0455 2153.8 -97.718
    ## - global_sentiment_polarity     1    0.0552 2153.8 -97.708
    ## - avg_positive_polarity         1    0.0630 2153.8 -97.699
    ## - kw_max_max                    1    0.0637 2153.8 -97.698
    ## - global_rate_positive_words    1    0.0698 2153.8 -97.692
    ## - kw_min_avg                    1    0.0873 2153.8 -97.673
    ## - rate_positive_words           1    0.1134 2153.8 -97.645
    ## - n_non_stop_words              1    0.1329 2153.8 -97.624
    ## - n_non_stop_unique_tokens      1    0.2884 2154.0 -97.456
    ## - rate_negative_words           1    0.2963 2154.0 -97.448
    ## - kw_min_max                    1    0.4329 2154.2 -97.300
    ## - LDA_00                        1    0.4395 2154.2 -97.293
    ## - max_negative_polarity         1    0.6061 2154.3 -97.113
    ## - self_reference_max_shares     1    0.6088 2154.3 -97.111
    ## - title_sentiment_polarity      1    0.7597 2154.5 -96.948
    ## - abs_title_subjectivity        1    0.9577 2154.7 -96.734
    ## - min_positive_polarity         1    1.0077 2154.7 -96.680
    ## - title_subjectivity            1    1.0214 2154.7 -96.666
    ## - kw_avg_max                    1    1.1368 2154.8 -96.541
    ## - avg_negative_polarity         1    1.3279 2155.1 -96.335
    ## - max_positive_polarity         1    1.6031 2155.3 -96.039
    ## - n_unique_tokens               1    1.6315 2155.3 -96.008
    ## - kw_min_min                    1    1.6462 2155.4 -95.992
    ## - LDA_02                        1    1.7415 2155.5 -95.889
    ## <none>                                      2153.7 -95.767
    ## - num_hrefs                     1    2.0289 2155.8 -95.580
    ## - num_keywords                  1    2.1178 2155.8 -95.484
    ## - LDA_03                        1    2.3610 2156.1 -95.222
    ## - num_self_hrefs                1    2.6230 2156.3 -94.940
    ## - num_videos                    1    2.8490 2156.6 -94.696
    ## - abs_title_sentiment_polarity  1    3.2461 2157.0 -94.268
    ## - num_imgs                      1    3.7595 2157.5 -93.716
    ## - kw_avg_min                    1    3.7906 2157.5 -93.682
    ## - LDA_01                        1    3.8854 2157.6 -93.580
    ## - kw_max_min                    1    4.6855 2158.4 -92.719
    ## - min_negative_polarity         1    5.9993 2159.7 -91.305
    ## - kw_max_avg                    1    6.9710 2160.7 -90.260
    ## - self_reference_min_shares     1    8.7455 2162.5 -88.353
    ## - kw_avg_avg                    1   14.4661 2168.2 -82.216
    ## 
    ## Step:  AIC=-97.74
    ## shares ~ n_tokens_title + n_unique_tokens + n_non_stop_words + 
    ##     n_non_stop_unique_tokens + num_hrefs + num_self_hrefs + num_imgs + 
    ##     num_videos + num_keywords + kw_min_min + kw_max_min + kw_avg_min + 
    ##     kw_min_max + kw_max_max + kw_avg_max + kw_min_avg + kw_max_avg + 
    ##     kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     LDA_00 + LDA_01 + LDA_02 + LDA_03 + global_sentiment_polarity + 
    ##     global_rate_positive_words + rate_positive_words + rate_negative_words + 
    ##     avg_positive_polarity + min_positive_polarity + max_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_subjectivity + title_sentiment_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - n_tokens_title                1    0.0447 2153.8 -99.694
    ## - global_sentiment_polarity     1    0.0461 2153.8 -99.692
    ## - kw_max_max                    1    0.0614 2153.8 -99.676
    ## - global_rate_positive_words    1    0.0815 2153.8 -99.654
    ## - avg_positive_polarity         1    0.0840 2153.8 -99.651
    ## - kw_min_avg                    1    0.0891 2153.8 -99.646
    ## - rate_positive_words           1    0.1061 2153.8 -99.627
    ## - n_non_stop_words              1    0.1348 2153.9 -99.597
    ## - rate_negative_words           1    0.2837 2154.0 -99.436
    ## - n_non_stop_unique_tokens      1    0.2920 2154.0 -99.427
    ## - kw_min_max                    1    0.4265 2154.2 -99.282
    ## - LDA_00                        1    0.4483 2154.2 -99.258
    ## - self_reference_max_shares     1    0.6170 2154.4 -99.077
    ## - max_negative_polarity         1    0.6558 2154.4 -99.035
    ## - title_sentiment_polarity      1    0.7575 2154.5 -98.925
    ## - abs_title_subjectivity        1    0.9468 2154.7 -98.721
    ## - title_subjectivity            1    1.0007 2154.7 -98.663
    ## - min_positive_polarity         1    1.0334 2154.8 -98.628
    ## - kw_avg_max                    1    1.1383 2154.9 -98.515
    ## - avg_negative_polarity         1    1.4455 2155.2 -98.183
    ## - max_positive_polarity         1    1.5799 2155.3 -98.038
    ## - n_unique_tokens               1    1.6331 2155.4 -97.981
    ## - kw_min_min                    1    1.6366 2155.4 -97.977
    ## - LDA_02                        1    1.7250 2155.5 -97.882
    ## <none>                                      2153.7 -97.742
    ## - num_hrefs                     1    2.0642 2155.8 -97.517
    ## - num_keywords                  1    2.1359 2155.9 -97.439
    ## - LDA_03                        1    2.3608 2156.1 -97.197
    ## - num_self_hrefs                1    2.6127 2156.3 -96.926
    ## - num_videos                    1    2.8284 2156.6 -96.693
    ## - abs_title_sentiment_polarity  1    3.2655 2157.0 -96.222
    ## - kw_avg_min                    1    3.7874 2157.5 -95.660
    ## - num_imgs                      1    3.7994 2157.5 -95.648
    ## - LDA_01                        1    3.8783 2157.6 -95.563
    ## - kw_max_min                    1    4.6776 2158.4 -94.702
    ## - min_negative_polarity         1    5.9880 2159.7 -93.292
    ## - kw_max_avg                    1    6.9562 2160.7 -92.251
    ## - self_reference_min_shares     1    8.7289 2162.5 -90.346
    ## - kw_avg_avg                    1   14.4436 2168.2 -84.215
    ## 
    ## Step:  AIC=-99.69
    ## shares ~ n_unique_tokens + n_non_stop_words + n_non_stop_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + num_imgs + num_videos + num_keywords + 
    ##     kw_min_min + kw_max_min + kw_avg_min + kw_min_max + kw_max_max + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_00 + LDA_01 + LDA_02 + LDA_03 + 
    ##     global_sentiment_polarity + global_rate_positive_words + 
    ##     rate_positive_words + rate_negative_words + avg_positive_polarity + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS      AIC
    ## - global_sentiment_polarity     1    0.0449 2153.8 -101.645
    ## - kw_max_max                    1    0.0609 2153.8 -101.628
    ## - global_rate_positive_words    1    0.0761 2153.9 -101.612
    ## - avg_positive_polarity         1    0.0830 2153.9 -101.604
    ## - kw_min_avg                    1    0.0919 2153.9 -101.595
    ## - rate_positive_words           1    0.1020 2153.9 -101.584
    ## - n_non_stop_words              1    0.1395 2153.9 -101.543
    ## - rate_negative_words           1    0.2766 2154.1 -101.395
    ## - n_non_stop_unique_tokens      1    0.3044 2154.1 -101.365
    ## - kw_min_max                    1    0.4150 2154.2 -101.246
    ## - LDA_00                        1    0.4497 2154.2 -101.209
    ## - self_reference_max_shares     1    0.6211 2154.4 -101.024
    ## - max_negative_polarity         1    0.6641 2154.4 -100.978
    ## - title_sentiment_polarity      1    0.7584 2154.5 -100.876
    ## - title_subjectivity            1    0.9993 2154.8 -100.616
    ## - abs_title_subjectivity        1    1.0102 2154.8 -100.604
    ## - min_positive_polarity         1    1.0273 2154.8 -100.586
    ## - kw_avg_max                    1    1.1856 2155.0 -100.415
    ## - avg_negative_polarity         1    1.4731 2155.3 -100.105
    ## - max_positive_polarity         1    1.5874 2155.4  -99.982
    ## - n_unique_tokens               1    1.6031 2155.4  -99.965
    ## - kw_min_min                    1    1.6336 2155.4  -99.932
    ## - LDA_02                        1    1.7647 2155.6  -99.791
    ## <none>                                      2153.8  -99.694
    ## - num_hrefs                     1    2.0454 2155.8  -99.489
    ## - num_keywords                  1    2.1147 2155.9  -99.414
    ## - LDA_03                        1    2.3596 2156.2  -99.150
    ## - num_self_hrefs                1    2.6227 2156.4  -98.867
    ## - num_videos                    1    2.8391 2156.6  -98.634
    ## - abs_title_sentiment_polarity  1    3.2788 2157.1  -98.160
    ## - kw_avg_min                    1    3.7586 2157.5  -97.643
    ## - num_imgs                      1    3.7873 2157.6  -97.612
    ## - LDA_01                        1    3.9590 2157.7  -97.428
    ## - kw_max_min                    1    4.6482 2158.4  -96.686
    ## - min_negative_polarity         1    6.0500 2159.8  -95.178
    ## - kw_max_avg                    1    6.9592 2160.8  -94.200
    ## - self_reference_min_shares     1    8.7052 2162.5  -92.324
    ## - kw_avg_avg                    1   14.4374 2168.2  -86.174
    ## 
    ## Step:  AIC=-101.65
    ## shares ~ n_unique_tokens + n_non_stop_words + n_non_stop_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + num_imgs + num_videos + num_keywords + 
    ##     kw_min_min + kw_max_min + kw_avg_min + kw_min_max + kw_max_max + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_00 + LDA_01 + LDA_02 + LDA_03 + 
    ##     global_rate_positive_words + rate_positive_words + rate_negative_words + 
    ##     avg_positive_polarity + min_positive_polarity + max_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_subjectivity + title_sentiment_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS      AIC
    ## - avg_positive_polarity         1    0.0384 2153.9 -103.604
    ## - global_rate_positive_words    1    0.0487 2153.9 -103.593
    ## - kw_max_max                    1    0.0614 2153.9 -103.579
    ## - kw_min_avg                    1    0.0909 2153.9 -103.547
    ## - rate_positive_words           1    0.1033 2153.9 -103.534
    ## - n_non_stop_words              1    0.1401 2154.0 -103.494
    ## - rate_negative_words           1    0.2464 2154.1 -103.380
    ## - n_non_stop_unique_tokens      1    0.2968 2154.1 -103.325
    ## - kw_min_max                    1    0.4130 2154.2 -103.200
    ## - LDA_00                        1    0.4381 2154.3 -103.173
    ## - self_reference_max_shares     1    0.6159 2154.4 -102.981
    ## - title_sentiment_polarity      1    0.7319 2154.6 -102.856
    ## - max_negative_polarity         1    0.7414 2154.6 -102.846
    ## - title_subjectivity            1    1.0014 2154.8 -102.565
    ## - abs_title_subjectivity        1    1.0266 2154.9 -102.538
    ## - min_positive_polarity         1    1.1199 2154.9 -102.438
    ## - kw_avg_max                    1    1.1957 2155.0 -102.356
    ## - max_positive_polarity         1    1.5982 2155.4 -101.922
    ## - n_unique_tokens               1    1.6284 2155.5 -101.890
    ## - kw_min_min                    1    1.6356 2155.5 -101.882
    ## - avg_negative_polarity         1    1.7062 2155.5 -101.806
    ## - LDA_02                        1    1.7859 2155.6 -101.720
    ## <none>                                      2153.8 -101.645
    ## - num_hrefs                     1    2.0256 2155.9 -101.462
    ## - num_keywords                  1    2.1134 2155.9 -101.367
    ## - LDA_03                        1    2.3817 2156.2 -101.078
    ## - num_self_hrefs                1    2.6330 2156.5 -100.807
    ## - num_videos                    1    2.8826 2156.7 -100.538
    ## - abs_title_sentiment_polarity  1    3.2725 2157.1 -100.118
    ## - kw_avg_min                    1    3.7639 2157.6  -99.589
    ## - num_imgs                      1    3.7951 2157.6  -99.556
    ## - LDA_01                        1    3.9568 2157.8  -99.382
    ## - kw_max_min                    1    4.6516 2158.5  -98.634
    ## - min_negative_polarity         1    6.0309 2159.9  -97.150
    ## - kw_max_avg                    1    6.9454 2160.8  -96.166
    ## - self_reference_min_shares     1    8.7312 2162.6  -94.247
    ## - kw_avg_avg                    1   14.4285 2168.3  -88.135
    ## 
    ## Step:  AIC=-103.6
    ## shares ~ n_unique_tokens + n_non_stop_words + n_non_stop_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + num_imgs + num_videos + num_keywords + 
    ##     kw_min_min + kw_max_min + kw_avg_min + kw_min_max + kw_max_max + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_00 + LDA_01 + LDA_02 + LDA_03 + 
    ##     global_rate_positive_words + rate_positive_words + rate_negative_words + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS      AIC
    ## - global_rate_positive_words    1    0.0472 2153.9 -105.553
    ## - kw_max_max                    1    0.0621 2153.9 -105.537
    ## - kw_min_avg                    1    0.0916 2154.0 -105.505
    ## - rate_positive_words           1    0.0981 2154.0 -105.498
    ## - n_non_stop_words              1    0.1443 2154.0 -105.448
    ## - rate_negative_words           1    0.2389 2154.1 -105.346
    ## - n_non_stop_unique_tokens      1    0.3270 2154.2 -105.251
    ## - kw_min_max                    1    0.4190 2154.3 -105.152
    ## - LDA_00                        1    0.4394 2154.3 -105.130
    ## - self_reference_max_shares     1    0.6110 2154.5 -104.945
    ## - max_negative_polarity         1    0.7320 2154.6 -104.814
    ## - title_sentiment_polarity      1    0.7414 2154.6 -104.804
    ## - title_subjectivity            1    1.0056 2154.9 -104.519
    ## - abs_title_subjectivity        1    1.0155 2154.9 -104.509
    ## - kw_avg_max                    1    1.1847 2155.1 -104.326
    ## - n_unique_tokens               1    1.6046 2155.5 -103.874
    ## - kw_min_min                    1    1.6444 2155.5 -103.831
    ## - avg_negative_polarity         1    1.7008 2155.6 -103.770
    ## - LDA_02                        1    1.7855 2155.7 -103.679
    ## <none>                                      2153.9 -103.604
    ## - min_positive_polarity         1    1.8673 2155.7 -103.591
    ## - num_keywords                  1    2.0928 2156.0 -103.348
    ## - num_hrefs                     1    2.1032 2156.0 -103.336
    ## - LDA_03                        1    2.3955 2156.3 -103.022
    ## - num_self_hrefs                1    2.5981 2156.5 -102.803
    ## - num_videos                    1    2.8701 2156.7 -102.510
    ## - max_positive_polarity         1    2.9198 2156.8 -102.457
    ## - abs_title_sentiment_polarity  1    3.2402 2157.1 -102.112
    ## - kw_avg_min                    1    3.7899 2157.7 -101.520
    ## - num_imgs                      1    3.8441 2157.7 -101.462
    ## - LDA_01                        1    3.9809 2157.8 -101.314
    ## - kw_max_min                    1    4.6691 2158.5 -100.573
    ## - min_negative_polarity         1    6.0343 2159.9  -99.105
    ## - kw_max_avg                    1    6.9292 2160.8  -98.142
    ## - self_reference_min_shares     1    8.7354 2162.6  -96.202
    ## - kw_avg_avg                    1   14.4015 2168.3  -90.123
    ## 
    ## Step:  AIC=-105.55
    ## shares ~ n_unique_tokens + n_non_stop_words + n_non_stop_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + num_imgs + num_videos + num_keywords + 
    ##     kw_min_min + kw_max_min + kw_avg_min + kw_min_max + kw_max_max + 
    ##     kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_00 + LDA_01 + LDA_02 + LDA_03 + 
    ##     rate_positive_words + rate_negative_words + min_positive_polarity + 
    ##     max_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_subjectivity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS      AIC
    ## - kw_max_max                    1    0.0617 2154.0 -107.486
    ## - rate_positive_words           1    0.0871 2154.0 -107.459
    ## - kw_min_avg                    1    0.0992 2154.0 -107.446
    ## - n_non_stop_words              1    0.1500 2154.1 -107.391
    ## - rate_negative_words           1    0.2400 2154.2 -107.294
    ## - n_non_stop_unique_tokens      1    0.3147 2154.2 -107.214
    ## - LDA_00                        1    0.4267 2154.3 -107.093
    ## - kw_min_max                    1    0.4286 2154.3 -107.091
    ## - self_reference_max_shares     1    0.6180 2154.5 -106.886
    ## - max_negative_polarity         1    0.7425 2154.7 -106.752
    ## - title_sentiment_polarity      1    0.7431 2154.7 -106.752
    ## - title_subjectivity            1    0.9840 2154.9 -106.492
    ## - abs_title_subjectivity        1    1.0685 2155.0 -106.401
    ## - kw_avg_max                    1    1.1671 2155.1 -106.295
    ## - kw_min_min                    1    1.6589 2155.6 -105.764
    ## - avg_negative_polarity         1    1.6956 2155.6 -105.725
    ## - n_unique_tokens               1    1.7117 2155.6 -105.708
    ## - LDA_02                        1    1.8139 2155.7 -105.597
    ## - min_positive_polarity         1    1.8259 2155.7 -105.585
    ## <none>                                      2153.9 -105.553
    ## - num_hrefs                     1    2.0697 2156.0 -105.322
    ## - num_keywords                  1    2.1141 2156.0 -105.274
    ## - LDA_03                        1    2.4295 2156.3 -104.934
    ## - num_self_hrefs                1    2.6856 2156.6 -104.658
    ## - num_videos                    1    2.8267 2156.7 -104.506
    ## - max_positive_polarity         1    3.2117 2157.1 -104.092
    ## - abs_title_sentiment_polarity  1    3.2668 2157.2 -104.032
    ## - kw_avg_min                    1    3.7849 2157.7 -103.474
    ## - num_imgs                      1    3.8957 2157.8 -103.355
    ## - LDA_01                        1    3.9863 2157.9 -103.258
    ## - kw_max_min                    1    4.6581 2158.6 -102.535
    ## - min_negative_polarity         1    5.9925 2159.9 -101.099
    ## - kw_max_avg                    1    6.8822 2160.8 -100.142
    ## - self_reference_min_shares     1    8.7408 2162.7  -98.145
    ## - kw_avg_avg                    1   14.3600 2168.3  -92.117
    ## 
    ## Step:  AIC=-107.49
    ## shares ~ n_unique_tokens + n_non_stop_words + n_non_stop_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + num_imgs + num_videos + num_keywords + 
    ##     kw_min_min + kw_max_min + kw_avg_min + kw_min_max + kw_avg_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_00 + LDA_01 + LDA_02 + LDA_03 + 
    ##     rate_positive_words + rate_negative_words + min_positive_polarity + 
    ##     max_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_subjectivity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS      AIC
    ## - rate_positive_words           1    0.0866 2154.1 -109.393
    ## - kw_min_avg                    1    0.0871 2154.1 -109.392
    ## - n_non_stop_words              1    0.1509 2154.1 -109.324
    ## - rate_negative_words           1    0.2373 2154.2 -109.230
    ## - n_non_stop_unique_tokens      1    0.3170 2154.3 -109.145
    ## - LDA_00                        1    0.4185 2154.4 -109.035
    ## - kw_min_max                    1    0.5359 2154.5 -108.908
    ## - self_reference_max_shares     1    0.6110 2154.6 -108.827
    ## - title_sentiment_polarity      1    0.7377 2154.7 -108.691
    ## - max_negative_polarity         1    0.7395 2154.7 -108.689
    ## - title_subjectivity            1    0.9888 2155.0 -108.420
    ## - abs_title_subjectivity        1    1.0730 2155.1 -108.329
    ## - kw_avg_max                    1    1.1378 2155.1 -108.260
    ## - avg_negative_polarity         1    1.6909 2155.7 -107.663
    ## - n_unique_tokens               1    1.7016 2155.7 -107.652
    ## - LDA_02                        1    1.8103 2155.8 -107.535
    ## - min_positive_polarity         1    1.8204 2155.8 -107.524
    ## <none>                                      2154.0 -107.486
    ## - num_hrefs                     1    2.0356 2156.0 -107.292
    ## - num_keywords                  1    2.2592 2156.2 -107.051
    ## - LDA_03                        1    2.5763 2156.6 -106.710
    ## - kw_min_min                    1    2.7188 2156.7 -106.556
    ## - num_self_hrefs                1    2.7229 2156.7 -106.552
    ## - num_videos                    1    2.8354 2156.8 -106.430
    ## - max_positive_polarity         1    3.2007 2157.2 -106.037
    ## - abs_title_sentiment_polarity  1    3.2621 2157.2 -105.971
    ## - kw_avg_min                    1    3.7397 2157.7 -105.457
    ## - num_imgs                      1    3.8557 2157.8 -105.332
    ## - LDA_01                        1    3.9472 2157.9 -105.233
    ## - kw_max_min                    1    4.6127 2158.6 -104.517
    ## - min_negative_polarity         1    5.9735 2159.9 -103.053
    ## - kw_max_avg                    1    7.0000 2161.0 -101.949
    ## - self_reference_min_shares     1    8.7637 2162.7 -100.054
    ## - kw_avg_avg                    1   14.6069 2168.6  -93.786
    ## 
    ## Step:  AIC=-109.39
    ## shares ~ n_unique_tokens + n_non_stop_words + n_non_stop_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + num_imgs + num_videos + num_keywords + 
    ##     kw_min_min + kw_max_min + kw_avg_min + kw_min_max + kw_avg_max + 
    ##     kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_00 + LDA_01 + LDA_02 + LDA_03 + 
    ##     rate_negative_words + min_positive_polarity + max_positive_polarity + 
    ##     avg_negative_polarity + min_negative_polarity + max_negative_polarity + 
    ##     title_subjectivity + title_sentiment_polarity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS      AIC
    ## - kw_min_avg                    1    0.0839 2154.2 -111.302
    ## - n_non_stop_unique_tokens      1    0.3098 2154.4 -111.059
    ## - LDA_00                        1    0.4216 2154.5 -110.938
    ## - kw_min_max                    1    0.5304 2154.6 -110.821
    ## - self_reference_max_shares     1    0.6121 2154.7 -110.733
    ## - title_sentiment_polarity      1    0.7338 2154.8 -110.602
    ## - max_negative_polarity         1    0.7463 2154.8 -110.588
    ## - title_subjectivity            1    0.9950 2155.1 -110.320
    ## - abs_title_subjectivity        1    1.0654 2155.1 -110.244
    ## - kw_avg_max                    1    1.1371 2155.2 -110.167
    ## - rate_negative_words           1    1.1721 2155.2 -110.129
    ## - avg_negative_polarity         1    1.6787 2155.7 -109.583
    ## - n_unique_tokens               1    1.7295 2155.8 -109.529
    ## - min_positive_polarity         1    1.7884 2155.8 -109.465
    ## - LDA_02                        1    1.8097 2155.9 -109.442
    ## <none>                                      2154.1 -109.393
    ## - num_hrefs                     1    2.0390 2156.1 -109.195
    ## - num_keywords                  1    2.2778 2156.3 -108.938
    ## - LDA_03                        1    2.6109 2156.7 -108.579
    ## - num_self_hrefs                1    2.7237 2156.8 -108.458
    ## - kw_min_min                    1    2.7269 2156.8 -108.454
    ## - num_videos                    1    2.8341 2156.9 -108.339
    ## - max_positive_polarity         1    3.1447 2157.2 -108.004
    ## - abs_title_sentiment_polarity  1    3.2574 2157.3 -107.883
    ## - kw_avg_min                    1    3.7400 2157.8 -107.363
    ## - n_non_stop_words              1    3.7941 2157.9 -107.305
    ## - num_imgs                      1    3.8580 2157.9 -107.236
    ## - LDA_01                        1    3.9478 2158.0 -107.139
    ## - kw_max_min                    1    4.6121 2158.7 -106.424
    ## - min_negative_polarity         1    5.9449 2160.0 -104.991
    ## - kw_max_avg                    1    6.9967 2161.1 -103.860
    ## - self_reference_min_shares     1    8.7660 2162.8 -101.959
    ## - kw_avg_avg                    1   14.6015 2168.7  -95.699
    ## 
    ## Step:  AIC=-111.3
    ## shares ~ n_unique_tokens + n_non_stop_words + n_non_stop_unique_tokens + 
    ##     num_hrefs + num_self_hrefs + num_imgs + num_videos + num_keywords + 
    ##     kw_min_min + kw_max_min + kw_avg_min + kw_min_max + kw_avg_max + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + self_reference_max_shares + 
    ##     LDA_00 + LDA_01 + LDA_02 + LDA_03 + rate_negative_words + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS      AIC
    ## - n_non_stop_unique_tokens      1    0.3244 2154.5 -112.953
    ## - LDA_00                        1    0.3937 2154.5 -112.878
    ## - kw_min_max                    1    0.4733 2154.6 -112.792
    ## - self_reference_max_shares     1    0.6270 2154.8 -112.626
    ## - title_sentiment_polarity      1    0.7308 2154.9 -112.514
    ## - max_negative_polarity         1    0.7613 2154.9 -112.482
    ## - title_subjectivity            1    1.0083 2155.2 -112.215
    ## - abs_title_subjectivity        1    1.0812 2155.2 -112.137
    ## - kw_avg_max                    1    1.1676 2155.3 -112.044
    ## - rate_negative_words           1    1.2001 2155.3 -112.009
    ## - n_unique_tokens               1    1.6848 2155.8 -111.486
    ## - avg_negative_polarity         1    1.6951 2155.8 -111.475
    ## - min_positive_polarity         1    1.8060 2155.9 -111.356
    ## <none>                                      2154.2 -111.302
    ## - LDA_02                        1    1.9323 2156.1 -111.220
    ## - num_hrefs                     1    2.0500 2156.2 -111.093
    ## - num_keywords                  1    2.1972 2156.3 -110.934
    ## - num_self_hrefs                1    2.6576 2156.8 -110.438
    ## - kw_min_min                    1    2.7766 2156.9 -110.310
    ## - LDA_03                        1    2.8740 2157.0 -110.205
    ## - num_videos                    1    2.8952 2157.0 -110.182
    ## - max_positive_polarity         1    3.1321 2157.3 -109.927
    ## - abs_title_sentiment_polarity  1    3.2524 2157.4 -109.798
    ## - kw_avg_min                    1    3.6928 2157.8 -109.324
    ## - n_non_stop_words              1    3.8200 2158.0 -109.187
    ## - num_imgs                      1    3.8559 2158.0 -109.148
    ## - LDA_01                        1    3.9729 2158.1 -109.022
    ## - kw_max_min                    1    4.6347 2158.8 -108.310
    ## - min_negative_polarity         1    5.9628 2160.1 -106.881
    ## - self_reference_min_shares     1    8.7229 2162.9 -103.915
    ## - kw_max_avg                    1   12.6821 2166.8  -99.666
    ## - kw_avg_avg                    1   30.5720 2184.7  -80.566
    ## 
    ## Step:  AIC=-112.95
    ## shares ~ n_unique_tokens + n_non_stop_words + num_hrefs + num_self_hrefs + 
    ##     num_imgs + num_videos + num_keywords + kw_min_min + kw_max_min + 
    ##     kw_avg_min + kw_min_max + kw_avg_max + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + self_reference_max_shares + LDA_00 + 
    ##     LDA_01 + LDA_02 + LDA_03 + rate_negative_words + min_positive_polarity + 
    ##     max_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_subjectivity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS      AIC
    ## - LDA_00                        1    0.4316 2154.9 -114.487
    ## - kw_min_max                    1    0.5019 2155.0 -114.412
    ## - self_reference_max_shares     1    0.6224 2155.1 -114.282
    ## - title_sentiment_polarity      1    0.7523 2155.2 -114.142
    ## - max_negative_polarity         1    0.8609 2155.3 -114.025
    ## - title_subjectivity            1    0.9779 2155.4 -113.899
    ## - abs_title_subjectivity        1    1.0273 2155.5 -113.845
    ## - kw_avg_max                    1    1.0695 2155.5 -113.800
    ## - rate_negative_words           1    1.2001 2155.7 -113.659
    ## - avg_negative_polarity         1    1.6712 2156.1 -113.151
    ## - min_positive_polarity         1    1.7117 2156.2 -113.108
    ## <none>                                      2154.5 -112.953
    ## - num_hrefs                     1    1.9295 2156.4 -112.873
    ## - LDA_02                        1    1.9909 2156.5 -112.807
    ## - num_keywords                  1    2.3293 2156.8 -112.443
    ## - num_self_hrefs                1    2.6658 2157.1 -112.080
    ## - LDA_03                        1    2.8422 2157.3 -111.890
    ## - kw_min_min                    1    2.8507 2157.3 -111.881
    ## - num_videos                    1    2.9471 2157.4 -111.777
    ## - abs_title_sentiment_polarity  1    3.3601 2157.8 -111.333
    ## - n_non_stop_words              1    3.4979 2158.0 -111.184
    ## - num_imgs                      1    3.5463 2158.0 -111.132
    ## - kw_avg_min                    1    3.6727 2158.2 -110.996
    ## - max_positive_polarity         1    3.8026 2158.3 -110.856
    ## - LDA_01                        1    4.1059 2158.6 -110.530
    ## - kw_max_min                    1    4.6149 2159.1 -109.982
    ## - min_negative_polarity         1    5.7005 2160.2 -108.814
    ## - self_reference_min_shares     1    8.8823 2163.4 -105.395
    ## - kw_max_avg                    1   12.5026 2167.0 -101.511
    ## - n_unique_tokens               1   15.2306 2169.7  -98.588
    ## - kw_avg_avg                    1   30.2931 2184.8  -82.517
    ## 
    ## Step:  AIC=-114.49
    ## shares ~ n_unique_tokens + n_non_stop_words + num_hrefs + num_self_hrefs + 
    ##     num_imgs + num_videos + num_keywords + kw_min_min + kw_max_min + 
    ##     kw_avg_min + kw_min_max + kw_avg_max + kw_max_avg + kw_avg_avg + 
    ##     self_reference_min_shares + self_reference_max_shares + LDA_01 + 
    ##     LDA_02 + LDA_03 + rate_negative_words + min_positive_polarity + 
    ##     max_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_subjectivity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS      AIC
    ## - kw_min_max                    1    0.5794 2155.5 -115.863
    ## - self_reference_max_shares     1    0.6109 2155.5 -115.829
    ## - title_sentiment_polarity      1    0.7315 2155.6 -115.699
    ## - max_negative_polarity         1    0.8714 2155.8 -115.548
    ## - title_subjectivity            1    0.9521 2155.9 -115.461
    ## - abs_title_subjectivity        1    1.0381 2155.9 -115.369
    ## - kw_avg_max                    1    1.0797 2156.0 -115.324
    ## - rate_negative_words           1    1.1909 2156.1 -115.204
    ## - avg_negative_polarity         1    1.6555 2156.6 -114.703
    ## - num_hrefs                     1    1.7721 2156.7 -114.578
    ## <none>                                      2154.9 -114.487
    ## - min_positive_polarity         1    1.8930 2156.8 -114.448
    ## - num_keywords                  1    2.1519 2157.1 -114.169
    ## - num_self_hrefs                1    2.8287 2157.7 -113.440
    ## - kw_min_min                    1    2.8733 2157.8 -113.392
    ## - num_videos                    1    2.9072 2157.8 -113.356
    ## - abs_title_sentiment_polarity  1    3.4067 2158.3 -112.818
    ## - n_non_stop_words              1    3.4237 2158.3 -112.800
    ## - kw_avg_min                    1    3.6719 2158.6 -112.532
    ## - num_imgs                      1    3.6884 2158.6 -112.515
    ## - max_positive_polarity         1    3.8389 2158.7 -112.353
    ## - kw_max_min                    1    4.6308 2159.5 -111.501
    ## - min_negative_polarity         1    5.7530 2160.7 -110.294
    ## - LDA_02                        1    6.1717 2161.1 -109.844
    ## - LDA_01                        1    6.7213 2161.6 -109.253
    ## - LDA_03                        1    8.9189 2163.8 -106.893
    ## - self_reference_min_shares     1    9.0416 2163.9 -106.761
    ## - kw_max_avg                    1   12.5619 2167.5 -102.985
    ## - n_unique_tokens               1   14.9172 2169.8 -100.462
    ## - kw_avg_avg                    1   30.7024 2185.6  -83.624
    ## 
    ## Step:  AIC=-115.86
    ## shares ~ n_unique_tokens + n_non_stop_words + num_hrefs + num_self_hrefs + 
    ##     num_imgs + num_videos + num_keywords + kw_min_min + kw_max_min + 
    ##     kw_avg_min + kw_avg_max + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     self_reference_max_shares + LDA_01 + LDA_02 + LDA_03 + rate_negative_words + 
    ##     min_positive_polarity + max_positive_polarity + avg_negative_polarity + 
    ##     min_negative_polarity + max_negative_polarity + title_subjectivity + 
    ##     title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS     AIC
    ## - self_reference_max_shares     1     0.661 2156.2 -117.15
    ## - title_sentiment_polarity      1     0.790 2156.3 -117.01
    ## - max_negative_polarity         1     0.876 2156.4 -116.92
    ## - title_subjectivity            1     0.945 2156.4 -116.84
    ## - abs_title_subjectivity        1     1.005 2156.5 -116.78
    ## - rate_negative_words           1     1.224 2156.7 -116.54
    ## - avg_negative_polarity         1     1.599 2157.1 -116.14
    ## - num_hrefs                     1     1.746 2157.2 -115.98
    ## <none>                                      2155.5 -115.86
    ## - min_positive_polarity         1     1.962 2157.4 -115.75
    ## - kw_min_min                    1     2.340 2157.8 -115.34
    ## - num_keywords                  1     2.570 2158.1 -115.09
    ## - num_self_hrefs                1     2.921 2158.4 -114.72
    ## - num_videos                    1     2.949 2158.4 -114.69
    ## - n_non_stop_words              1     3.395 2158.9 -114.21
    ## - abs_title_sentiment_polarity  1     3.411 2158.9 -114.19
    ## - kw_avg_min                    1     3.683 2159.2 -113.90
    ## - num_imgs                      1     3.826 2159.3 -113.74
    ## - max_positive_polarity         1     3.932 2159.4 -113.63
    ## - kw_avg_max                    1     4.006 2159.5 -113.55
    ## - kw_max_min                    1     4.683 2160.2 -112.82
    ## - min_negative_polarity         1     5.640 2161.1 -111.79
    ## - LDA_02                        1     6.168 2161.7 -111.22
    ## - LDA_01                        1     6.955 2162.4 -110.38
    ## - LDA_03                        1     8.775 2164.3 -108.43
    ## - self_reference_min_shares     1     9.106 2164.6 -108.07
    ## - kw_max_avg                    1    13.237 2168.7 -103.64
    ## - n_unique_tokens               1    14.622 2170.1 -102.16
    ## - kw_avg_avg                    1    32.471 2187.9  -83.13
    ## 
    ## Step:  AIC=-117.15
    ## shares ~ n_unique_tokens + n_non_stop_words + num_hrefs + num_self_hrefs + 
    ##     num_imgs + num_videos + num_keywords + kw_min_min + kw_max_min + 
    ##     kw_avg_min + kw_avg_max + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     LDA_01 + LDA_02 + LDA_03 + rate_negative_words + min_positive_polarity + 
    ##     max_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_subjectivity + title_sentiment_polarity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS      AIC
    ## - title_sentiment_polarity      1     0.871 2157.0 -118.212
    ## - max_negative_polarity         1     0.908 2157.1 -118.172
    ## - title_subjectivity            1     0.967 2157.1 -118.108
    ## - abs_title_subjectivity        1     0.993 2157.1 -118.081
    ## - rate_negative_words           1     1.137 2157.3 -117.926
    ## - avg_negative_polarity         1     1.684 2157.8 -117.336
    ## - num_hrefs                     1     1.773 2157.9 -117.241
    ## <none>                                      2156.2 -117.150
    ## - min_positive_polarity         1     1.992 2158.1 -117.005
    ## - kw_min_min                    1     2.367 2158.5 -116.602
    ## - num_self_hrefs                1     2.493 2158.6 -116.466
    ## - num_keywords                  1     2.555 2158.7 -116.399
    ## - num_videos                    1     3.041 2159.2 -115.876
    ## - n_non_stop_words              1     3.473 2159.6 -115.412
    ## - abs_title_sentiment_polarity  1     3.478 2159.6 -115.406
    ## - kw_avg_min                    1     3.729 2159.9 -115.136
    ## - num_imgs                      1     3.734 2159.9 -115.130
    ## - kw_avg_max                    1     3.872 2160.0 -114.982
    ## - max_positive_polarity         1     4.016 2160.2 -114.828
    ## - kw_max_min                    1     5.016 2161.2 -113.752
    ## - min_negative_polarity         1     5.893 2162.0 -112.810
    ## - LDA_02                        1     6.117 2162.3 -112.569
    ## - LDA_01                        1     7.169 2163.3 -111.439
    ## - LDA_03                        1     9.244 2165.4 -109.212
    ## - kw_max_avg                    1    12.613 2168.8 -105.600
    ## - self_reference_min_shares     1    14.553 2170.7 -103.524
    ## - n_unique_tokens               1    14.713 2170.9 -103.352
    ## - kw_avg_avg                    1    32.295 2188.4  -84.614
    ## 
    ## Step:  AIC=-118.21
    ## shares ~ n_unique_tokens + n_non_stop_words + num_hrefs + num_self_hrefs + 
    ##     num_imgs + num_videos + num_keywords + kw_min_min + kw_max_min + 
    ##     kw_avg_min + kw_avg_max + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     LDA_01 + LDA_02 + LDA_03 + rate_negative_words + min_positive_polarity + 
    ##     max_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     max_negative_polarity + title_subjectivity + abs_title_subjectivity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS      AIC
    ## - max_negative_polarity         1     0.767 2157.8 -119.386
    ## - abs_title_subjectivity        1     1.182 2158.2 -118.939
    ## - title_subjectivity            1     1.292 2158.3 -118.821
    ## - avg_negative_polarity         1     1.456 2158.5 -118.644
    ## - rate_negative_words           1     1.475 2158.5 -118.624
    ## - num_hrefs                     1     1.845 2158.9 -118.225
    ## <none>                                      2157.0 -118.212
    ## - min_positive_polarity         1     2.022 2159.0 -118.035
    ## - kw_min_min                    1     2.270 2159.3 -117.768
    ## - num_self_hrefs                1     2.431 2159.4 -117.595
    ## - abs_title_sentiment_polarity  1     2.613 2159.6 -117.399
    ## - num_keywords                  1     2.645 2159.7 -117.365
    ## - num_videos                    1     2.855 2159.9 -117.139
    ## - n_non_stop_words              1     3.190 2160.2 -116.778
    ## - kw_avg_min                    1     3.763 2160.8 -116.162
    ## - kw_avg_max                    1     3.776 2160.8 -116.149
    ## - num_imgs                      1     3.841 2160.9 -116.079
    ## - max_positive_polarity         1     3.956 2161.0 -115.955
    ## - kw_max_min                    1     5.054 2162.1 -114.775
    ## - min_negative_polarity         1     5.693 2162.7 -114.088
    ## - LDA_02                        1     5.959 2163.0 -113.803
    ## - LDA_01                        1     7.326 2164.3 -112.335
    ## - LDA_03                        1     9.336 2166.3 -110.179
    ## - kw_max_avg                    1    12.481 2169.5 -106.809
    ## - n_unique_tokens               1    14.289 2171.3 -104.874
    ## - self_reference_min_shares     1    14.511 2171.5 -104.636
    ## - kw_avg_avg                    1    32.018 2189.0  -85.983
    ## 
    ## Step:  AIC=-119.39
    ## shares ~ n_unique_tokens + n_non_stop_words + num_hrefs + num_self_hrefs + 
    ##     num_imgs + num_videos + num_keywords + kw_min_min + kw_max_min + 
    ##     kw_avg_min + kw_avg_max + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     LDA_01 + LDA_02 + LDA_03 + rate_negative_words + min_positive_polarity + 
    ##     max_positive_polarity + avg_negative_polarity + min_negative_polarity + 
    ##     title_subjectivity + abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS      AIC
    ## - avg_negative_polarity         1     0.707 2158.5 -120.624
    ## - abs_title_subjectivity        1     1.142 2158.9 -120.157
    ## - rate_negative_words           1     1.193 2159.0 -120.102
    ## - title_subjectivity            1     1.212 2159.0 -120.081
    ## - num_hrefs                     1     1.830 2159.6 -119.416
    ## <none>                                      2157.8 -119.386
    ## - min_positive_polarity         1     1.873 2159.7 -119.370
    ## - kw_min_min                    1     2.229 2160.0 -118.988
    ## - num_self_hrefs                1     2.437 2160.2 -118.764
    ## - num_keywords                  1     2.515 2160.3 -118.680
    ## - abs_title_sentiment_polarity  1     2.619 2160.4 -118.568
    ## - num_videos                    1     2.841 2160.6 -118.329
    ## - n_non_stop_words              1     3.280 2161.1 -117.857
    ## - num_imgs                      1     3.666 2161.4 -117.442
    ## - kw_avg_min                    1     3.860 2161.6 -117.234
    ## - kw_avg_max                    1     3.885 2161.7 -117.207
    ## - max_positive_polarity         1     4.224 2162.0 -116.843
    ## - kw_max_min                    1     5.142 2162.9 -115.856
    ## - min_negative_polarity         1     5.248 2163.0 -115.743
    ## - LDA_02                        1     5.903 2163.7 -115.040
    ## - LDA_01                        1     7.305 2165.1 -113.534
    ## - LDA_03                        1     9.610 2167.4 -111.063
    ## - kw_max_avg                    1    12.547 2170.3 -107.917
    ## - n_unique_tokens               1    14.156 2171.9 -106.196
    ## - self_reference_min_shares     1    14.266 2172.1 -106.078
    ## - kw_avg_avg                    1    32.026 2189.8  -87.161
    ## 
    ## Step:  AIC=-120.62
    ## shares ~ n_unique_tokens + n_non_stop_words + num_hrefs + num_self_hrefs + 
    ##     num_imgs + num_videos + num_keywords + kw_min_min + kw_max_min + 
    ##     kw_avg_min + kw_avg_max + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     LDA_01 + LDA_02 + LDA_03 + rate_negative_words + min_positive_polarity + 
    ##     max_positive_polarity + min_negative_polarity + title_subjectivity + 
    ##     abs_title_subjectivity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS      AIC
    ## - abs_title_subjectivity        1     1.144 2159.6 -121.394
    ## - title_subjectivity            1     1.223 2159.7 -121.309
    ## - rate_negative_words           1     1.406 2159.9 -121.111
    ## - num_hrefs                     1     1.612 2160.1 -120.890
    ## <none>                                      2158.5 -120.624
    ## - min_positive_polarity         1     2.024 2160.5 -120.447
    ## - kw_min_min                    1     2.213 2160.7 -120.244
    ## - num_self_hrefs                1     2.407 2160.9 -120.036
    ## - num_keywords                  1     2.458 2160.9 -119.981
    ## - abs_title_sentiment_polarity  1     2.524 2161.0 -119.910
    ## - num_videos                    1     2.869 2161.4 -119.539
    ## - n_non_stop_words              1     3.421 2161.9 -118.945
    ## - num_imgs                      1     3.767 2162.3 -118.574
    ## - kw_avg_min                    1     3.865 2162.4 -118.468
    ## - kw_avg_max                    1     3.882 2162.4 -118.451
    ## - max_positive_polarity         1     4.106 2162.6 -118.210
    ## - kw_max_min                    1     5.130 2163.6 -117.110
    ## - LDA_02                        1     5.882 2164.4 -116.302
    ## - min_negative_polarity         1     6.174 2164.7 -115.989
    ## - LDA_01                        1     7.349 2165.8 -114.728
    ## - LDA_03                        1     9.489 2168.0 -112.435
    ## - kw_max_avg                    1    12.719 2171.2 -108.976
    ## - self_reference_min_shares     1    14.144 2172.6 -107.452
    ## - n_unique_tokens               1    17.378 2175.9 -103.997
    ## - kw_avg_avg                    1    32.132 2190.6  -88.299
    ## 
    ## Step:  AIC=-121.39
    ## shares ~ n_unique_tokens + n_non_stop_words + num_hrefs + num_self_hrefs + 
    ##     num_imgs + num_videos + num_keywords + kw_min_min + kw_max_min + 
    ##     kw_avg_min + kw_avg_max + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     LDA_01 + LDA_02 + LDA_03 + rate_negative_words + min_positive_polarity + 
    ##     max_positive_polarity + min_negative_polarity + title_subjectivity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS      AIC
    ## - title_subjectivity            1     0.655 2160.3 -122.690
    ## - num_hrefs                     1     1.444 2161.1 -121.841
    ## - rate_negative_words           1     1.446 2161.1 -121.838
    ## <none>                                      2159.6 -121.394
    ## - min_positive_polarity         1     1.930 2161.6 -121.319
    ## - abs_title_sentiment_polarity  1     2.219 2161.8 -121.008
    ## - kw_min_min                    1     2.223 2161.9 -121.004
    ## - num_keywords                  1     2.376 2162.0 -120.839
    ## - num_self_hrefs                1     2.604 2162.2 -120.594
    ## - num_videos                    1     2.910 2162.5 -120.266
    ## - n_non_stop_words              1     3.384 2163.0 -119.757
    ## - kw_avg_min                    1     3.771 2163.4 -119.341
    ## - num_imgs                      1     3.817 2163.4 -119.291
    ## - max_positive_polarity         1     3.887 2163.5 -119.216
    ## - kw_avg_max                    1     3.961 2163.6 -119.137
    ## - kw_max_min                    1     5.012 2164.7 -118.009
    ## - LDA_02                        1     5.793 2165.4 -117.171
    ## - min_negative_polarity         1     6.391 2166.0 -116.529
    ## - LDA_01                        1     7.483 2167.1 -115.358
    ## - LDA_03                        1     9.153 2168.8 -113.570
    ## - kw_max_avg                    1    12.872 2172.5 -109.589
    ## - self_reference_min_shares     1    14.282 2173.9 -108.081
    ## - n_unique_tokens               1    17.252 2176.9 -104.911
    ## - kw_avg_avg                    1    32.302 2191.9  -88.905
    ## 
    ## Step:  AIC=-122.69
    ## shares ~ n_unique_tokens + n_non_stop_words + num_hrefs + num_self_hrefs + 
    ##     num_imgs + num_videos + num_keywords + kw_min_min + kw_max_min + 
    ##     kw_avg_min + kw_avg_max + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     LDA_01 + LDA_02 + LDA_03 + rate_negative_words + min_positive_polarity + 
    ##     max_positive_polarity + min_negative_polarity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS      AIC
    ## - rate_negative_words           1     1.448 2161.7 -123.133
    ## - num_hrefs                     1     1.525 2161.8 -123.050
    ## <none>                                      2160.3 -122.690
    ## - min_positive_polarity         1     1.945 2162.2 -122.599
    ## - kw_min_min                    1     2.171 2162.5 -122.357
    ## - num_keywords                  1     2.244 2162.5 -122.277
    ## - num_self_hrefs                1     2.571 2162.9 -121.927
    ## - num_videos                    1     2.789 2163.1 -121.693
    ## - n_non_stop_words              1     3.407 2163.7 -121.029
    ## - num_imgs                      1     3.659 2163.9 -120.758
    ## - kw_avg_min                    1     3.778 2164.1 -120.630
    ## - max_positive_polarity         1     3.949 2164.2 -120.447
    ## - kw_avg_max                    1     4.065 2164.4 -120.322
    ## - kw_max_min                    1     5.031 2165.3 -119.286
    ## - LDA_02                        1     5.700 2166.0 -118.569
    ## - min_negative_polarity         1     6.549 2166.8 -117.658
    ## - LDA_01                        1     7.471 2167.8 -116.670
    ## - abs_title_sentiment_polarity  1     8.989 2169.3 -115.043
    ## - LDA_03                        1     9.007 2169.3 -115.024
    ## - kw_max_avg                    1    12.896 2173.2 -110.864
    ## - self_reference_min_shares     1    14.106 2174.4 -109.571
    ## - n_unique_tokens               1    17.264 2177.6 -106.199
    ## - kw_avg_avg                    1    32.377 2192.7  -90.133
    ## 
    ## Step:  AIC=-123.13
    ## shares ~ n_unique_tokens + n_non_stop_words + num_hrefs + num_self_hrefs + 
    ##     num_imgs + num_videos + num_keywords + kw_min_min + kw_max_min + 
    ##     kw_avg_min + kw_avg_max + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     LDA_01 + LDA_02 + LDA_03 + min_positive_polarity + max_positive_polarity + 
    ##     min_negative_polarity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS      AIC
    ## - num_hrefs                     1     1.391 2163.1 -123.638
    ## - min_positive_polarity         1     1.622 2163.4 -123.390
    ## <none>                                      2161.7 -123.133
    ## - num_keywords                  1     1.994 2163.7 -122.991
    ## - kw_min_min                    1     2.083 2163.8 -122.896
    ## - num_videos                    1     2.637 2164.4 -122.301
    ## - num_self_hrefs                1     2.893 2164.6 -122.026
    ## - num_imgs                      1     3.627 2165.4 -121.238
    ## - n_non_stop_words              1     3.828 2165.6 -121.023
    ## - kw_avg_min                    1     3.912 2165.7 -120.933
    ## - kw_avg_max                    1     4.108 2165.8 -120.723
    ## - max_positive_polarity         1     5.116 2166.8 -119.642
    ## - kw_max_min                    1     5.198 2166.9 -119.554
    ## - LDA_02                        1     5.690 2167.4 -119.027
    ## - LDA_01                        1     7.703 2169.4 -116.870
    ## - abs_title_sentiment_polarity  1     8.776 2170.5 -115.722
    ## - LDA_03                        1     8.894 2170.6 -115.595
    ## - min_negative_polarity         1    12.104 2173.8 -112.162
    ## - kw_max_avg                    1    12.601 2174.3 -111.632
    ## - self_reference_min_shares     1    14.063 2175.8 -110.070
    ## - n_unique_tokens               1    16.693 2178.4 -107.264
    ## - kw_avg_avg                    1    32.095 2193.8  -90.897
    ## 
    ## Step:  AIC=-123.64
    ## shares ~ n_unique_tokens + n_non_stop_words + num_self_hrefs + 
    ##     num_imgs + num_videos + num_keywords + kw_min_min + kw_max_min + 
    ##     kw_avg_min + kw_avg_max + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     LDA_01 + LDA_02 + LDA_03 + min_positive_polarity + max_positive_polarity + 
    ##     min_negative_polarity + abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS      AIC
    ## - min_positive_polarity         1    1.4811 2164.6 -124.048
    ## <none>                                      2163.1 -123.638
    ## - num_keywords                  1    1.8695 2165.0 -123.631
    ## - kw_min_min                    1    1.9878 2165.1 -123.505
    ## - num_videos                    1    2.5826 2165.7 -122.866
    ## - n_non_stop_words              1    3.7128 2166.8 -121.655
    ## - kw_avg_max                    1    3.9045 2167.0 -121.449
    ## - kw_avg_min                    1    3.9165 2167.1 -121.436
    ## - kw_max_min                    1    5.1963 2168.3 -120.065
    ## - max_positive_polarity         1    5.9640 2169.1 -119.242
    ## - num_imgs                      1    6.5839 2169.7 -118.578
    ## - num_self_hrefs                1    6.7046 2169.8 -118.449
    ## - LDA_02                        1    6.7705 2169.9 -118.379
    ## - LDA_01                        1    7.8651 2171.0 -117.207
    ## - abs_title_sentiment_polarity  1    8.4223 2171.6 -116.611
    ## - LDA_03                        1    9.3262 2172.5 -115.644
    ## - min_negative_polarity         1   11.1896 2174.3 -113.653
    ## - kw_max_avg                    1   11.9898 2175.1 -112.798
    ## - self_reference_min_shares     1   14.4995 2177.6 -110.119
    ## - n_unique_tokens               1   16.0793 2179.2 -108.434
    ## - kw_avg_avg                    1   31.1809 2194.3  -92.392
    ## 
    ## Step:  AIC=-124.05
    ## shares ~ n_unique_tokens + n_non_stop_words + num_self_hrefs + 
    ##     num_imgs + num_videos + num_keywords + kw_min_min + kw_max_min + 
    ##     kw_avg_min + kw_avg_max + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     LDA_01 + LDA_02 + LDA_03 + max_positive_polarity + min_negative_polarity + 
    ##     abs_title_sentiment_polarity
    ## 
    ##                                Df Sum of Sq    RSS      AIC
    ## <none>                                      2164.6 -124.048
    ## - kw_min_min                    1    2.1631 2166.8 -123.728
    ## - num_keywords                  1    2.2581 2166.9 -123.626
    ## - num_videos                    1    2.3237 2166.9 -123.556
    ## - kw_avg_max                    1    3.7385 2168.3 -122.040
    ## - n_non_stop_words              1    3.9271 2168.5 -121.838
    ## - kw_avg_min                    1    4.0939 2168.7 -121.659
    ## - kw_max_min                    1    5.3717 2170.0 -120.291
    ## - max_positive_polarity         1    6.1044 2170.7 -119.506
    ## - num_self_hrefs                1    6.4376 2171.1 -119.150
    ## - LDA_02                        1    6.7810 2171.4 -118.782
    ## - num_imgs                      1    7.3427 2171.9 -118.182
    ## - LDA_01                        1    7.8265 2172.4 -117.664
    ## - abs_title_sentiment_polarity  1    8.2423 2172.8 -117.220
    ## - LDA_03                        1    9.8479 2174.5 -115.504
    ## - min_negative_polarity         1   11.9325 2176.5 -113.278
    ## - kw_max_avg                    1   12.1734 2176.8 -113.021
    ## - self_reference_min_shares     1   14.8177 2179.4 -110.200
    ## - n_unique_tokens               1   21.2760 2185.9 -103.327
    ## - kw_avg_avg                    1   31.4682 2196.1  -92.521

``` r
lm$call[["formula"]] # model selected based on AIC. 
```

    ## shares ~ n_unique_tokens + n_non_stop_words + num_self_hrefs + 
    ##     num_imgs + num_videos + num_keywords + kw_min_min + kw_max_min + 
    ##     kw_avg_min + kw_avg_max + kw_max_avg + kw_avg_avg + self_reference_min_shares + 
    ##     LDA_01 + LDA_02 + LDA_03 + max_positive_polarity + min_negative_polarity + 
    ##     abs_title_sentiment_polarity
    ## <environment: 0x7f9b86a1e748>

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
    ##    weekday_is_friday    weekday_is_monday  weekday_is_saturday 
    ##                  332                  337                  180 
    ##    weekday_is_sunday  weekday_is_thursday   weekday_is_tuesday 
    ##                  137                  463                  458 
    ## weekday_is_wednesday 
    ##                  416

``` r
C2 <- pop.data %>% 
    group_by( weekday ) %>% 
    summarise( percent = 100 * n() / nrow( X ),mean_shares = mean(shares), mean_images = mean(num_imgs),mean_video = mean(num_videos),mean_link = mean(num_hrefs))
knitr::kable(C2)
```

| weekday                |   percent | mean\_shares | mean\_images | mean\_video | mean\_link |
| :--------------------- | --------: | -----------: | -----------: | ----------: | ---------: |
| weekday\_is\_friday    | 0.9907490 |     4012.880 |     3.262048 |   1.2710843 |   11.74699 |
| weekday\_is\_monday    | 1.0056699 |     4010.442 |     3.421365 |   1.3590504 |   11.20475 |
| weekday\_is\_saturday  | 0.5371531 |     3508.711 |     5.733333 |   0.5611111 |   20.98333 |
| weekday\_is\_sunday    | 0.4088332 |     4525.350 |     5.649635 |   1.1021898 |   13.43796 |
| weekday\_is\_thursday  | 1.3816771 |     3092.168 |     4.853132 |   1.0453564 |   13.41685 |
| weekday\_is\_tuesday   | 1.3667562 |     3503.290 |     5.126638 |   1.0021834 |   14.05459 |
| weekday\_is\_wednesday | 1.2414205 |     3508.510 |     3.194711 |   1.2524038 |   11.21394 |

``` r
table(pop.data$weekday, pop.data$channel)
```

    ##                       
    ##                        data_channel_is_socmed
    ##   weekday_is_friday                       332
    ##   weekday_is_monday                       337
    ##   weekday_is_saturday                     180
    ##   weekday_is_sunday                       137
    ##   weekday_is_thursday                     463
    ##   weekday_is_tuesday                      458
    ##   weekday_is_wednesday                    416

``` r
table(pop.data$channel, pop.data$is_weekend)
```

    ##                         
    ##                             0    1
    ##   data_channel_is_socmed 2006  317

``` r
C3 <- pop.data %>% group_by(is_weekend)%>% 
    summarise( percent = 100 * n() / nrow( X ),mean_shares = mean(shares), mean_images = mean(num_imgs),mean_video = mean(num_videos),mean_link = mean(num_hrefs))
knitr::kable(C3)
```

| is\_weekend |   percent | mean\_shares | mean\_images | mean\_video | mean\_link |
| :---------- | --------: | -----------: | -----------: | ----------: | ---------: |
| 0           | 5.9862728 |     3579.021 |     4.067797 |   1.1684945 |   12.45763 |
| 1           | 0.9459863 |     3948.079 |     5.697161 |   0.7949527 |   17.72240 |

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
| TRUE        | FALSE              | FALSE               | TRUE                         | FALSE      | FALSE     | FALSE         | FALSE       | FALSE        | FALSE        | FALSE        | FALSE        | FALSE                        | FALSE                         | FALSE                         | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | FALSE              | FALSE               | TRUE                         | FALSE      | FALSE     | FALSE         | FALSE       | FALSE        | FALSE        | FALSE        | TRUE         | FALSE                        | FALSE                         | FALSE                         | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | FALSE              | FALSE               | TRUE                         | FALSE      | FALSE     | FALSE         | FALSE       | FALSE        | FALSE        | FALSE        | TRUE         | TRUE                         | FALSE                         | FALSE                         | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | FALSE              | FALSE               | TRUE                         | FALSE      | TRUE      | FALSE         | FALSE       | FALSE        | FALSE        | FALSE        | TRUE         | TRUE                         | FALSE                         | FALSE                         | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | FALSE              | FALSE               | TRUE                         | FALSE      | TRUE      | FALSE         | FALSE       | TRUE         | FALSE        | TRUE         | TRUE         | FALSE                        | FALSE                         | FALSE                         | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | FALSE              | FALSE               | TRUE                         | FALSE      | TRUE      | FALSE         | FALSE       | TRUE         | FALSE        | TRUE         | TRUE         | FALSE                        | TRUE                          | FALSE                         | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | FALSE              | TRUE                | TRUE                         | FALSE      | TRUE      | FALSE         | FALSE       | TRUE         | FALSE        | TRUE         | TRUE         | TRUE                         | FALSE                         | FALSE                         | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | FALSE              | TRUE                | TRUE                         | TRUE       | TRUE      | FALSE         | FALSE       | TRUE         | FALSE        | TRUE         | TRUE         | FALSE                        | TRUE                          | FALSE                         | FALSE                 | FALSE                    | FALSE                           |
| TRUE        | FALSE              | TRUE                | TRUE                         | TRUE       | TRUE      | FALSE         | FALSE       | TRUE         | FALSE        | TRUE         | TRUE         | FALSE                        | TRUE                          | FALSE                         | TRUE                  | FALSE                    | FALSE                           |
| TRUE        | TRUE               | TRUE                | TRUE                         | TRUE       | TRUE      | FALSE         | FALSE       | TRUE         | FALSE        | TRUE         | TRUE         | FALSE                        | TRUE                          | FALSE                         | TRUE                  | FALSE                    | FALSE                           |
| TRUE        | TRUE               | TRUE                | TRUE                         | TRUE       | TRUE      | FALSE         | FALSE       | TRUE         | FALSE        | TRUE         | TRUE         | TRUE                         | TRUE                          | FALSE                         | TRUE                  | FALSE                    | FALSE                           |
| TRUE        | TRUE               | TRUE                | TRUE                         | TRUE       | TRUE      | TRUE          | FALSE       | TRUE         | FALSE        | TRUE         | TRUE         | TRUE                         | TRUE                          | FALSE                         | TRUE                  | FALSE                    | FALSE                           |
| TRUE        | TRUE               | TRUE                | TRUE                         | TRUE       | TRUE      | TRUE          | FALSE       | TRUE         | FALSE        | TRUE         | TRUE         | TRUE                         | TRUE                          | FALSE                         | TRUE                  | FALSE                    | TRUE                            |
| TRUE        | TRUE               | TRUE                | TRUE                         | TRUE       | TRUE      | TRUE          | FALSE       | TRUE         | FALSE        | TRUE         | TRUE         | TRUE                         | TRUE                          | TRUE                          | TRUE                  | FALSE                    | TRUE                            |
| TRUE        | TRUE               | TRUE                | TRUE                         | TRUE       | TRUE      | TRUE          | FALSE       | TRUE         | FALSE        | TRUE         | TRUE         | TRUE                         | TRUE                          | TRUE                          | TRUE                  | TRUE                     | TRUE                            |
| TRUE        | TRUE               | TRUE                | TRUE                         | TRUE       | TRUE      | TRUE          | TRUE        | TRUE         | FALSE        | TRUE         | TRUE         | TRUE                         | TRUE                          | TRUE                          | TRUE                  | TRUE                     | TRUE                            |
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
|     10 |  9 |   6 |

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
| lm.fit1 |     5310.986 |        0.0385173 |    2583.471 |
| lm.fit2 |     5346.692 |        0.0357747 |    2567.548 |
| lm.fit3 |     5640.343 |        0.0690678 |    2643.685 |

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
    ## (Intercept)                   2.664093e+03
    ## n_tokens_title                .           
    ## n_tokens_content              .           
    ## n_unique_tokens              -1.334608e+03
    ## n_non_stop_words              .           
    ## n_non_stop_unique_tokens      .           
    ## num_hrefs                    -2.281150e+00
    ## num_self_hrefs               -2.158107e+01
    ## num_imgs                      .           
    ## num_videos                    .           
    ## average_token_length          .           
    ## num_keywords                  .           
    ## kw_min_min                    .           
    ## kw_max_min                    .           
    ## kw_avg_min                    .           
    ## kw_min_max                   -4.403619e-04
    ## kw_max_max                    .           
    ## kw_avg_max                   -2.475318e-04
    ## kw_min_avg                    7.442306e-02
    ## kw_max_avg                    .           
    ## kw_avg_avg                    1.901993e-01
    ## self_reference_min_shares     1.323310e-02
    ## self_reference_max_shares     .           
    ## self_reference_avg_sharess    1.994104e-04
    ## is_weekend                    .           
    ## LDA_00                        1.199084e+03
    ## LDA_01                       -5.994352e+02
    ## LDA_02                       -2.945825e+02
    ## LDA_03                        .           
    ## LDA_04                        4.219829e+02
    ## global_subjectivity           .           
    ## global_sentiment_polarity     .           
    ## global_rate_positive_words    .           
    ## global_rate_negative_words    .           
    ## rate_positive_words           .           
    ## rate_negative_words           .           
    ## avg_positive_polarity         .           
    ## min_positive_polarity        -6.485942e+02
    ## max_positive_polarity         .           
    ## avg_negative_polarity         .           
    ## min_negative_polarity        -9.234570e+02
    ## max_negative_polarity         .           
    ## title_subjectivity            2.699642e+02
    ## title_sentiment_polarity      .           
    ## abs_title_subjectivity        .           
    ## abs_title_sentiment_polarity  1.072883e+03

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
    ##                                         s0
    ## (Intercept)                   5.459908e+03
    ## n_tokens_content              .           
    ## n_non_stop_words              3.420166e+02
    ## n_non_stop_unique_tokens     -3.926041e+03
    ## num_hrefs                    -5.521703e+00
    ## num_imgs                     -1.866280e+01
    ## num_keywords                  .           
    ## num_videos                    .           
    ## kw_avg_max                   -8.683598e-04
    ## kw_min_avg                    5.720191e-02
    ## kw_max_avg                    .           
    ## kw_avg_avg                    2.549434e-01
    ## self_reference_min_shares     1.304107e-02
    ## self_reference_avg_sharess    1.177825e-03
    ## global_rate_positive_words    .           
    ## rate_positive_words          -1.609645e+02
    ## abs_title_subjectivity        .           
    ## abs_title_sentiment_polarity  .

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
| lm.fit1        |   5313.519 |
| lm.fit2        |   5299.903 |
| lm.fit3        |   5411.260 |
| lasso.fit.full |   4970.102 |
| lasso.fit.18   |   5336.020 |
| rf.fit1        |   5225.837 |
| rf.fit2        |   5224.444 |
| boostTreefit1  |   5254.558 |
| boostTreefit2  |   5264.347 |

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
