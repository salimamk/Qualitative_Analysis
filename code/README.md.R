# Install and load required packages
install.packages("tm")
install.packages("wordcloud2")
library(tm)
library(wordcloud2)
library(NLP)
library(tidyverse)

# Load the crude dataset
data("crude")
View(crude)

# View the structure of the dataset
summary(crude)

# Qualitative Analysis: Preprocess the Text
# Create a text corpus
corpus <- VCorpus(VectorSource(crude))  # Using VCorpus for flexibility

# Preprocess the text: remove punctuation, numbers, stop words, and convert to lowercase
corpus_clean <- corpus %>%
  tm_map(content_transformer(tolower)) %>%  # Convert to lowercase
  tm_map(removePunctuation) %>%             # Remove punctuation
  tm_map(removeNumbers) %>%                 # Remove numbers
  tm_map(removeWords, stopwords("en")) %>%  # Remove stop words
  tm_map(stripWhitespace)                   # Remove extra whitespaces

# Inspect cleaned text (first three documents)
inspect(corpus_clean[1:3])

# View the cleaned content of the first document
cat(content(corpus_clean[[1]]))

# View all cleaned text for the first three documents
for (i in 1:3) {
  cat("Document", i, ":\n")
  cat(content(corpus_clean[[i]]), "\n\n")
}

# Word Frequency Analysis
# Convert the corpus to a Term-Document Matrix
tdm <- TermDocumentMatrix(corpus_clean)
tdm_matrix <- as.matrix(tdm)

# Get word frequencies
word_freq <- sort(rowSums(tdm_matrix), decreasing = TRUE)

# Create a data frame of word frequencies
word_freq_df <- data.frame(word = names(word_freq), freq = word_freq)

# View the most frequent words
head(word_freq_df)

# Create a Word Cloud
wc<-wordcloud2(word_freq_df, size = 0.7, shape = "circle")
wc

# Save the word cloud plot to the 'figures' directory as an HTML file
library(htmlwidgets)
saveWidget(wc, "figures/WordCloud.html")

# Interpretation
# The word "oil" was the most frequent in the articles within the dataset
# and the word "last" is the least frequent.

#Sentiment Analysis
# Install and load sentiment analysis libraries
install.packages("syuzhet")
library(syuzhet)

# Extract raw text
raw_text <- sapply(corpus_clean, as.character)

# Perform sentiment analysis
sentiments <- get_nrc_sentiment(raw_text)

# Summarize sentiments
sentiment_summary <- colSums(sentiments)
print(sentiment_summary)

# Save the barplot to a PNG file in the 'figures' directory
png("figures/SentimentBarplot.png", width = 800, height = 600)

# Visualize sentiment with the barplot
barplot(sentiment_summary, col = rainbow(10), las = 2, main = "Sentiment Analysis")

# Close the PNG device to save the plot
dev.off()

# Interpretation of BarPlot
# The majority of articles showing a positive sentiment indicates that most of the content in the dataset reflects favorable opinions or attitudes.
# With a few showing negative sentiments of anger, disgust and fear.

#Topic Modeling -to identify the themes in the data

# Install and load required packages
install.packages("topicmodels")
library(topicmodels)
install.packages("wordcloud")
library(wordcloud)
install.packages("RColorBrewer")
library(RColorBrewer)

# Create a Document-Term Matrix
dtm <- DocumentTermMatrix(corpus_clean)

# Fit a Latent Dirichlet Allocation (LDA) model to discover topics
lda_model <- LDA(dtm, k = 3)  # 'k' is the number of topics

# Get the top terms for each topic
top_terms <- terms(lda_model, 10)  # Top 10 terms per topic

# Convert the top terms to a data frame
top_terms_df <- data.frame(
  term = as.vector(top_terms),
  topic = rep(1:3, each = 10)  # Create a 'topic' column for each term
)

top_terms_df

# Extract term frequencies from the Document-Term Matrix (DTM)
# Convert the DTM to a matrix and sum over the rows for each term
dtm_matrix <- as.matrix(dtm)

# Calculate the frequency of terms in the DTM
term_freqs <- colSums(dtm_matrix)

# Match the top terms from LDA model with the column names of the DTM
top_terms_df$frequency <- term_freqs[match(top_terms_df$term, colnames(dtm_matrix))]

# Check the updated dataframe with frequencies
head(top_terms_df)

# Create a word cloud for each topic based on term frequencies
wordcloud(words = top_terms_df$term, 
          freq = top_terms_df$frequency, 
          min.freq = 1, 
          scale = c(3, 0.5), 
          colors = brewer.pal(8, "Dark2"), 
          random.order = FALSE, 
          main = "Top Terms in Topics")

# Save the word cloud as a PNG image in the 'figures' directory
png("figures/TopicsWordCloud.png", width = 800, height = 600)
wordcloud(words = top_terms_df$term, 
          freq = top_terms_df$frequency, 
          min.freq = 1, 
          scale = c(3, 0.5), 
          colors = brewer.pal(8, "Dark2"), 
          random.order = FALSE, 
          main = "Top Terms in Topics")
dev.off()

# #Interpretation:
# "Oil" is the most frequent and important term in this topic, suggesting that:
# the theme revolves around the oil industry, production, or market-related discussions.This is accurate 
# given that the data is on articles on crude oil.