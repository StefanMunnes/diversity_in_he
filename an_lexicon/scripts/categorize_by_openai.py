import polars as pl
import numpy as np
from openai import OpenAI
import time


with open("C:/Users/munnes/Documents/API_Keys/openai_20240613.txt", "r") as f:
    openai_key = f.read()

client = OpenAI(api_key=openai_key)


# load and prepare data
data_lexicon_lookedup = pl.read_parquet("an_lexicon/data/data_lexicon_lookedup.parquet")

data_ready = (
  data_lexicon_lookedup
  .filter(pl.col("concept_category").is_in(["Collective", "Individual", "Hybrid"]))
  .filter(pl.col("country").is_in(["usa", "uk"]))
  .with_columns(
    # create row number as index
      index = pl.col("url").cum_count()
  )
  )

# system prompt: read txt file with prompt
with open("an_llm/prompt_indiv_colle_v3.txt", "r") as f:
    prompt_system = f.read()


# loop over rows of data_ready, call API and store results in csv

# Create empty dataframe with two columns to add results to
data_results = pl.DataFrame(schema={"token": pl.String, "linprob": pl.Float64})


time_start = time.time()

for row in data_ready.iter_rows(named = True):

    # print progress
    print(f"Row: {row['index']}")

    message = [
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": row["text"]},
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=message,
        max_tokens=1,
        temperature=0,
        logprobs=True,
        top_logprobs=3,
    )

    # Extract the classified concept from the response object
    logprobs = response.choices[0].logprobs.content[0].top_logprobs

    rows = []

    # Loop over the top 3 log probabilities and extract the token and linear probability
    for item in logprobs:
        rows.append(
            {
                "token": item.token,
                "linprob": np.round(np.exp(item.logprob) * 100, 2),
            }
        )

    # Append the rows to the dataframe
    data_results = data_results.vstack(pl.DataFrame(rows))


time_end = time.time()
print(f"Time: {round((time_end - time_start) / 60)} minutes") # ~26 Minutes


concept_lab = {1: "Individual", 2: "Collective", 3: "Hybrid"}

data_results_wide = (
    data_results
    .with_columns(
        sequence = (pl.arange(0, data_results.shape[0]) % 3) + 1,
        group = pl.arange(0, data_results.shape[0]) // 3,
        token = pl.when(
            pl.col("token").cast(pl.Int64, strict=False).is_in([1, 2, 3])
        )
        .then(pl.col("token").cast(pl.Int64, strict=False))
        .otherwise(pl.lit(None))
        .replace_strict(concept_lab)
    )
    .pivot(values=["token", "linprob"], index="group", on="sequence")
)


# add columns to data_ready
data_results_combined = (
    pl
    .concat([data_ready, data_results_wide], how="horizontal")
    .with_columns(match = pl.col("concept_category") == pl.col("token_1"))
    .select([
        "text", "match", "concept_category", 
        "token_1", "linprob_1", "token_2", "linprob_2", "token_3", "linprob_3",
        "keywords", "tokens",
        "individual", "collective", 
        "individual_count", "collective_count", 
        "individual_prop", "collective_prop",
        "url", "domain"
    ])
)

data_results_combined.write_excel("an_llm/data_results_combined.xlsx")


data_results_combined["match"].value_counts()

