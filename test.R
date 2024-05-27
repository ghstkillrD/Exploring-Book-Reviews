# Load required libraries
library(tidyverse)
library(tm)
library(SnowballC)
library(topicmodels)
library(syuzhet)
library(wordcloud)
library(gridExtra)
library(Matrix)

# Load the dataset
book_reviews <- read.csv("MS4S09_CW_Book_Reviews.csv")


# Task A – Text Mining

# Text Preprocessing
corpus <- Corpus(VectorSource(book_reviews$Review_text))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stripWhitespace)

# Create Document-Term Matrix
dtm <- DocumentTermMatrix(corpus)


# Task B – Sentiment Analysis

# Sentiment Analysis
sentiment_scores <- get_nrc_sentiment(book_reviews$Review_text)
sentiment_scores$Review_ID <- seq_len(nrow(sentiment_scores))

# Visualize Sentiment Distribution
sentiment_distribution <- sentiment_scores %>%
  gather(sentiment, count, anger:trust) %>%
  group_by(sentiment) %>%
  summarise(total_count = sum(count))

# Bar plot for Sentiment Distribution
barplot(sentiment_distribution$total_count, names.arg = sentiment_distribution$sentiment,
        main = "Sentiment Distribution", xlab = "Sentiment", ylab = "Count", col = "skyblue")


# Task C – Topic Modelling

# Convert Document-Term Matrix to a Sparse Matrix
matrix <- sparseMatrix(i = dtm$i, j = dtm$j, x = dtm$v,
                              dims = dim(dtm), dimnames = dimnames(dtm))

# Create a topic model using Latent Dirichlet Allocation (LDA)
lda_model <- LDA(matrix, k = 5, method = "Gibbs", control = list(seed = 1234))

# Visualize topics
terms <- terms(lda_model, 10)
topics <- lapply(terms, paste, collapse = ", ")
flat_terms <- unlist(topics)
topic_lengths <- sapply(topics, length)
topics_df <- data.frame(Topic = rep(1:5, each = length(flat_terms) / 5), Terms = flat_terms)

# Plot Topic Terms
ggplot(topics_df, aes(x = Topic, y = reorder(Terms, Topic), label = Terms)) +
  geom_text(size = 3) +
  labs(title = "Top Terms per Topic", x = "Topic", y = "Terms") +
  theme_minimal()


# Task D – Further Exploration

# Wordcloud of most frequent words in reviews
word_freq <- Matrix::colSums(matrix)
word_freq <- sort(word_freq, decreasing = TRUE)
wordcloud(words = names(word_freq), freq = word_freq, min.freq = 10,
          max.words = 100, random.order = FALSE, colors = brewer.pal(8, "Dark2"))

# Average rating distribution
avg_rating_dist <- book_reviews %>%
  group_by(Title) %>%
  summarise(avg_rating = mean(Rating))

hist(avg_rating_dist$avg_rating, main = "Average Rating Distribution", xlab = "Average Rating", col = "lightblue")


