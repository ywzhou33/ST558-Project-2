## ST558 Project 2  

The purpose of this repository is to create predictive models and automating Markdown reports. Analysis are done on the **Online News Popularity Data Set**. Further information about this data can be accessed [here](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity).   

The following variables are included in this data.  
 
0. `url`: URL of the article (non-predictive)   
1. `timedelta`: Days between the article publication and the dataset acquisition (non-predictive)   
2. `n_tokens_title`: Number of words in the title   
3. `n_tokens_content`: Number of words in the content   
4. `n_unique_tokens`: Rate of unique words in the content   
5. `n_non_stop_words`: Rate of non-stop words in the content   
6. `n_non_stop_unique_tokens`: Rate of unique non-stop words in the content   
7. `num_hrefs`: Number of links   
8. `num_self_hrefs`: Number of links to other articles published by Mashable   
9. `num_imgs`: Number of images   
10. `num_videos`: Number of videos   
11. `average_token_length`: Average length of the words in the content   
12. `num_keywords`: Number of keywords in the metadata   
13. `data_channel_is_lifestyle`: Is data channel 'Lifestyle'?   
14. `data_channel_is_entertainment`: Is data channel 'Entertainment'?   
15. `data_channel_is_bus`: Is data channel 'Business'?   
16. `data_channel_is_socmed`: Is data channel 'Social Media'?   
17. `data_channel_is_tech`: Is data channel 'Tech'?   
18. `data_channel_is_world`: Is data channel 'World'?   
19. `kw_min_min`: Worst keyword (min. shares)   
20. `kw_max_min`: Worst keyword (max. shares)   
21. `kw_avg_min`: Worst keyword (avg. shares)   
22. `kw_min_max`: Best keyword (min. shares)   
23. `kw_max_max`: Best keyword (max. shares)   
24. `kw_avg_max`: Best keyword (avg. shares)   
25. `kw_min_avg`: Avg. keyword (min. shares)   
26. `kw_max_avg`: Avg. keyword (max. shares)   
27. `kw_avg_avg`: Avg. keyword (avg. shares)   
28. `self_reference_min_shares`: Min. shares of referenced articles in Mashable   
29. `self_reference_max_shares`: Max. shares of referenced articles in Mashable   
30. `self_reference_avg_sharess`: Avg. shares of referenced articles in Mashable   
31. `weekday_is_monday`: Was the article published on a Monday?   
32. `weekday_is_tuesday`: Was the article published on a Tuesday?   
33. `weekday_is_wednesday`: Was the article published on a Wednesday?   
34. `weekday_is_thursday`: Was the article published on a Thursday?   
35. `weekday_is_friday`: Was the article published on a Friday?   
36. `weekday_is_saturday`: Was the article published on a Saturday?   
37. `weekday_is_sunday`: Was the article published on a Sunday?   
38. `is_weekend`: Was the article published on the weekend?   
39. `LDA_00`: Closeness to LDA topic 0   
40. `LDA_01`: Closeness to LDA topic 1   
41. `LDA_02`: Closeness to LDA topic 2   
42. `LDA_03`: Closeness to LDA topic 3   
43. `LDA_04`: Closeness to LDA topic 4   
44. `global_subjectivity`: Text subjectivity   
45. `global_sentiment_polarity`: Text sentiment polarity   
46. `global_rate_positive_words`: Rate of positive words in the content   
47. `global_rate_negative_words`: Rate of negative words in the content   
48. `rate_positive_words`: Rate of positive words among non-neutral tokens   
49. `rate_negative_words`: Rate of negative words among non-neutral tokens   
50. `avg_positive_polarity`: Avg. polarity of positive words   
51. `min_positive_polarity`: Min. polarity of positive words   
52. `max_positive_polarity`: Max. polarity of positive words   
53. `avg_negative_polarity`: Avg. polarity of negative words   
54. `min_negative_polarity`: Min. polarity of negative words   
55. `max_negative_polarity`: Max. polarity of negative words   
56. `title_subjectivity`: Title subjectivity   
57. `title_sentiment_polarity`: Title polarity   
58. `abs_title_subjectivity`: Absolute subjectivity level   
59. `abs_title_sentiment_polarity`: Absolute polarity level   
60. `shares`: Number of shares (the response variables)   

In this project, subsets by `data_channel_is_*` were produced for automating Markdown reports. Variables were selected from each subset by the best subset and stepwise selction methods. Predictive models including the linear regression models, lasso models, random forest models and boosted tree models were constructed using 5-fold cross-validation. These models were first constructed on training data set and than tested on test data set. The best model were selected based on lowest RMSE.   

### List of packages used:      

[__dplyr__](https://dplyr.tidyverse.org/) grammar of data manipulation  
[__tidyr__](https://tidyr.tidyverse.org/)  create tidy data  
[__ggcorrplot__](https://cran.r-project.org/web/packages/ggcorrplot/readme/README.html) Visualization of a correlation matrix using ggplot2  
[__vcd__](https://cran.r-project.org/web/packages/vcd/index.html) Visualizing Categorical Data  
[__caret__](https://cran.r-project.org/web/packages/caret/vignettes/caret.html) contains functions to streamline the model training process for complex regression and classification problems  
[__class__](https://cran.r-project.org/web/packages/class/index.html) Various functions for classification, including k-nearest neighbour, Learning Vector Quantization and Self-Organizing Maps.  
[__randomForest__](https://www.rdocumentation.org/packages/randomForest/versions/4.6-14/topics/randomForest) Classification and Regression with Random Forest  
[__gbm__](https://www.rdocumentation.org/packages/gbm/versions/2.1.8/topics/gbm)  Fits generalized boosted regression models  
[__readr__](https://readr.tidyverse.org/) provide a fast and friendly way to read rectangular data  
[__leaps__](https://www.rdocumentation.org/packages/leaps/versions/3.1/topics/leaps) Visualizing Categorical Data  
[__Matrix__](https://cran.r-project.org/web/packages/Matrix/index.html) Sparse and Dense Matrix Classes and Methods  
[__glmnet__](https://cran.r-project.org/web/packages/glmnet/index.html) Lasso and Elastic-Net Regularized Generalized Linear Models  
[__rmarkdown__](https://www.rdocumentation.org/packages/rmarkdown/versions/1.7) convert R Markdown documents into a variety of formats  
[__doParallel__](https://cran.r-project.org/web/packages/doParallel/index.html) Foreach Parallel Adaptor for the 'parallel' Package    


### Links to the generated analyses.  

The analysis for [Lifestyle articles is available here](data_channel_is_lifestyle.html).  
The analysis for [Entertainment articles is available here](data_channel_is_entertainment.html).  
The analysis for [Bus articles is available here](data_channel_is_bus.html).  
The analysis for [Socmed articles is available here](data_channel_is_socmed.html).  
The analysis for [Tech articles is available here](data_channel_is_tech.html).  
The analysis for [World articles is available here](data_channel_is_world.html).  

### Code used to create the analyses.

```{r, eval = FALSE}
channels <- unique(X$channel)
output.file <- paste0(channels,".md")

params = lapply(channels, FUN = function(x){list(channel = x)})

reports <- tibble(output.file, params)

library(rmarkdown)

apply(reports, MARGIN = 1,
      FUN = function(x){
        render(input = "./ST558_Project_2.Rmd",
               output_format = "github_document", 
               output_file = x[[1]], 
               params = x[[2]])
      })
```
