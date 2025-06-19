# --- Install and load necessary packages ---
if (!requireNamespace("shiny", quietly = TRUE)) install.packages("shiny")
if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")
if (!requireNamespace("stringr", quietly = TRUE)) install.packages("stringr")
if (!requireNamespace("DT", quietly = TRUE)) install.packages("DT")
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2") # New package for plotting

library(shiny)
library(dplyr)
library(stringr)
library(DT)
library(ggplot2)

# --- Load prepared data ---
# Make sure 'prepared_corpus_data.rds' is in the same directory as app.R,
# or provide the correct path.
data_file <- "shiny_data_sample_results_combined_v7.rds"

if (!file.exists(data_file)) {
  stop(paste(data_file, "not found. Please run prepare_data.R first."))
}
corpus_data <- readRDS(data_file)

# Ensure 'concept' and 'coder' are factors for consistent plotting/grouping
corpus_data <- corpus_data |>
  mutate(concept = as.factor(concept), coder = as.factor(coder)) |>
  filter(!is.na(concept))

# --- Prepare summary data ---
corpus_data_summary <- corpus_data |>
  count(language, concept, coder)


# --- User Interface (UI) ---
ui <- fluidPage(
  titlePanel("Text Corpora Keyword Analyzer"),

  sidebarLayout(
    sidebarPanel(
      textAreaInput(
        "keywords",
        "Enter Keywords (comma separated):",
        value = "equal opportunity",
        rows = 3,
        placeholder = "e.g., equal opportunity, unique, freedom"
      ),
      helpText(
        "Keywords are case-insensitive. Use ',' to separate multiple keywords and phrases."
      ),

      radioButtons(
        "search_logic",
        "Keyword Search Logic:",
        choices = c("Any (OR)" = "or", "All (AND)" = "and"),
        selected = "or"
      ),

      selectInput(
        "language_filter",
        "Filter by Language:",
        choices = unique(corpus_data$language),
        selected = unique(corpus_data$language)[1]
      ),

      actionButton("analyze_button", "Analyze Text")
    ),

    mainPanel(
      tabsetPanel(
        tabPanel(
          "Summary by Category and Coder",
          radioButtons(
            "outputTypeCategoryCoder",
            "Choose Output Type:",
            choices = c("Table" = "table", "Plot" = "plot"),
            selected = "table",
            inline = TRUE
          ),
          uiOutput("categoryCoderOutput") # Dynamic UI for table or plot
        ),
        tabPanel(
          "Filtered Texts",
          DTOutput("filteredTextsTable")
        )
      )
    )
  )
)

# --- Server Logic ---
server <- function(input, output) {
  # Reactive expression to filter data based on keywords and language
  filtered_data <- eventReactive(input$analyze_button, {
    req(input$keywords) # Ensure keywords are entered

    keywords_raw <- trimws(str_split(input$keywords, pattern = ",")[[1]])
    keywords_clean <- keywords_raw[keywords_raw != ""] # Remove empty strings

    if (length(keywords_clean) == 0) {
      return(data.frame()) # Return empty if no valid keywords
    }

    # Create a regex pattern based on search logic
    if (input$search_logic == "or") {
      # For "OR" logic, match any of the keywords
      regex_pattern <- paste(keywords_clean, collapse = "|")
    } else {
      # For "AND" logic, ensure all keywords are present (can be in any order)
      regex_pattern <- "" # Not directly used as a single pattern for AND
    }

    # Apply language filter
    temp_data <- corpus_data |>
      filter(language == input$language_filter)

    # Apply keyword filtering
    if (input$search_logic == "or") {
      temp_data <- temp_data |>
        filter(str_detect(
          text,
          regex(regex_pattern, ignore_case = TRUE)
        ))
    } else {
      # "AND" logic
      for (kw in keywords_clean) {
        temp_data <- temp_data |>
          filter(str_detect(text, regex(kw, ignore_case = TRUE)))
      }
    }

    return(temp_data)
  })

  # --- Reactive expression for summary by Category and Coder ---
  summary_category_coder_data <- reactive({
    df <- filtered_data()
    if (nrow(df) == 0) {
      return(data.frame(
        concept = factor(),
        coder = factor(),
        Count = numeric(),
        Percent = numeric()
      ))
    }

    # Count texts from keyword filtered data by concept and coder
    df_summary <- count(df, concept, coder, name = "Count")

    # Add total texts for each coder and concept and calculate percent
    df_summary_combined <- corpus_data_summary |>
      filter(language == input$language_filter) |>
      full_join(df_summary, by = c("concept", "coder")) |>
      mutate(percent = (Count / n) * 100) |>
      rename(texts_classified = n, texts_keywords = Count) |>
      arrange(concept, coder)

    return(df_summary_combined)
  })

  # --- Output: Summary by Category and Coder (Table or Plot) ---
  output$summaryTableCategoryCoder <- renderTable({
    summary_category_coder_data() |>
      select(concept, coder, texts_classified, texts_keywords, percent)
  })

  output$plotCategoryCoder <- renderPlot({
    df_summary <- summary_category_coder_data()
    if (nrow(df_summary) == 0) {
      return(NULL)
    }
    ggplot(df_summary, aes(x = concept, y = percent, fill = coder)) +
      geom_bar(stat = "identity", position = "dodge") + # Use dodge for side-by-side bars
      labs(
        title = "Relative Frequency of Texts by Category and Coder",
        x = "Category",
        y = "Relative Frequency (%)",
        fill = "Coder"
      ) +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) # Rotate x-axis labels if needed
  })

  # --- Dynamic UI for Category and Coder Output ---
  output$categoryCoderOutput <- renderUI({
    if (input$outputTypeCategoryCoder == "table") {
      tableOutput("summaryTableCategoryCoder")
    } else {
      plotOutput("plotCategoryCoder")
    }
  })

  # --- Output: Filtered Texts Table ---
  output$filteredTextsTable <- renderDT({
    df <- filtered_data() |> select(!language)
    if (nrow(df) == 0) {
      return(datatable(
        data.frame(Message = "No texts found matching the criteria."),
        options = list(dom = 't')
      ))
    }
    datatable(df, options = list(pageLength = 10))
  })
}

# --- Run the application ---
shinyApp(ui = ui, server = server)
