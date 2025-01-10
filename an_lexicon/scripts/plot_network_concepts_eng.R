
library(dplyr)
library(arrow)
library(jsonlite)
library(quanteda)
library(quanteda.textplots)


data <- read.csv("an_lexicon/output/data_lexicon_lookedup_flat.csv")

corpus <- data |>
  select(tokens, url) |>
  corpus(text_field = "tokens")

tokens <- tokens(corpus, remove_punct = TRUE, remove_numbers = TRUE)

# Combine compound words
compounds <- list(
  c("equal", "opportunity"),
  c("equal", "condition"),
  c("equal", "chance"),
  c("no", "one"),
  c("regardless", "of"),
  c("free", "from")
)

tokens <- tokens_compound(tokens, pattern = compounds, concatenator = "_")


# Remove stopwords and unnecessary tokens
stopwords_custom <- c(
  "the", "that", "a", "one", "its", "they", "it", "is", "and", "have",
  "of", "to", "in", "be", "for", "at", "as", "on", "will", "by", "from", "this",
  "she", "where", "into", "more", "if", "about", "so", "up", "out", "over",
  "than", "when", "such", "also", "not", "any", "could"
)

tokens <- tokens_remove(tokens, stopwords_custom) # stopwords("en")
tokens <- tokens_remove(tokens, c("s", "`"))



# Create FCM with whole paragraph as context
fcm <- fcm(tokens, context = "document", count = "frequency", tri = FALSE)


# Read in lexicon as nested list
lexicon <- jsonlite::fromJSON("an_lexicon/data/lexicon.json")

# flatten lexicon nested list to vector
lexicon_flat <- unlist(lexicon, recursive = FALSE)



# Select FCM co-occurring with flat lexicon_
fcm_lexicon <- fcm_select(fcm, lexicon_flat, case_insensitive = FALSE, verbose = TRUE)

# Convert and export to inspect FCM
fcm_lexicon_df <- convert(fcm_sort(fcm_lexicon), to = "data.frame")

openxlsx::write.xlsx(
  fcm_lexicon_df,
  "an_lexicon/output/fcm_lexicon.xlsx"
)


# Define colors depending on concept of lexicon
colors <- ifelse(
  featnames(fcm_lexicon) %in% lexicon[["individual"]], "#db7073",
  ifelse(featnames(fcm_lexicon) %in% lexicon[["collective"]], "#316196", "#309180")
)

# Create network plot with colored keywords
plot_network <- textplot_network(
  fcm_lexicon,
  min_freq = 0.7,
  omit_isolated = TRUE,
  # max.overlaps = 0,
  edge_alpha = 1,
  vertex_labelcolor = colors, vertex_color = colors
)



# count occurence in document of lexicon tokens
dfm <- dfm(tokens)
doc_freqs <- docfreq(dfm)
doc_freqs_lexicon <- doc_freqs[lexicon_flat]

doc_freqs_lexicon_prop <- round((doc_freqs_lexicon / nrow(data)) * 100, 2)

doc_freqs_lexicon_df <- 
  sapply(doc_freqs_lexicon, as.character) |>
  paste(" (", doc_freqs_lexicon_prop, "%)", sep = "") |>
  t() |>
  as.data.frame()

colnames(doc_freqs_lexicon_df) <- lexicon_flat


# Get all the feature names from the FCM co-occurring with the lexicon word
feature_names <- featnames(fcm)
top_n <- 100

lexicon_top_tokens_cooccur_ls <- lapply(lexicon_flat, function(lex_tok) {

  print(lex_tok)

  indice <- which(feature_names == lex_tok)
  fcm_lexicon_rows <- fcm[indice, ]
  cooccurrence_counts <- colSums(fcm_lexicon_rows)

  # Exclude keywords from co-occurring words
  cooccurrence_counts[lex_tok] <- 0
  cooccurrence_counts <- cooccurrence_counts[cooccurrence_counts > 0]

  if (length(cooccurrence_counts) == 0) {
    return(NULL)
  } 

  # Sort and select top co-occurring words
  sorted_cooccurrence <- sort(cooccurrence_counts, decreasing = TRUE)

  top_cooccurring_words <- sorted_cooccurrence[1:top_n] |> na.omit()

  # Combine top co-occurring tokens with count
  tokens = sapply(
      names(top_cooccurring_words), function(tok) paste0(tok, " (", top_cooccurring_words[tok], ")")
    )

  return(tokens)
})


filled_list <- lapply(lexicon_top_tokens_cooccur_ls, function(vec) {
  c(vec, rep(NA, top_n - length(vec)))
})

lexicon_top_tokens_cooccur <- as.data.frame(filled_list)


lexicon_row <- setNames(as.list(names(lexicon_top_tokens_cooccur)), names(lexicon_top_tokens_cooccur)) |>
  stringr::str_replace("[0-9]+", "") |> 
  t() |> 
  as.data.frame()


colnames(lexicon_row) <- lexicon_flat
colnames(lexicon_top_tokens_cooccur) <- lexicon_flat


# add to dataframe of top 100 tokens
lexicon_top_tokens_cooccur_count <- bind_rows(
  lexicon_row,
  doc_freqs_lexicon_df,
  lexicon_top_tokens_cooccur
) 

rownames(lexicon_top_tokens_cooccur_count) <- NULL


openxlsx::write.xlsx(
  lexicon_top_tokens_cooccur_count,
  "an_lexicon/output/lexicon_top_tokens_cooccur.xlsx"
)


dict <- dictionary(lexicon)

a <- dfm_lookup(dfm, dict)

b <- convert(a, to = "data.frame")


library(ggplot2)
library(reshape2)


df_long <- melt(b)


ggplot(df_long, aes(x=value, fill=variable)) +
  geom_density(alpha=0.5) +  # Use geom_histogram() for histogram
  labs(title="Distribution of Counts", x="Count", y="Density") +
  theme_minimal()

