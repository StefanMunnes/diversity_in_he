import time
import polars as pl
import numpy as np
from openai import OpenAI

# Load API key and initialize client
with open("C:/Users/munnes/Documents/API_Keys/openai_20240613.txt", "r") as f:
    openai_key = f.read().strip()

client = OpenAI(api_key=openai_key)

# Determine variables (version, languages)
version = "v8"
languages = ["eng", "ger"]

for lang in languages:
    print(f"Processing {lang.upper()} data")

    if lang == "eng":
        df_list = []
        # Load and combine Excel files for USA, UK, and IND
        for country in ["usa", "uk", "ind"]:
            temp_pl = pl.read_excel(
                f"an_llm/data/handcoding/done/data_filtered_sample_{country}.xlsx"
            )
            df_list.append(temp_pl)
        data_ready = pl.concat(df_list, how="vertical").with_columns(
            index=pl.col("url").cum_count()
        )

    elif lang == "ger":
        data_ready = pl.read_excel(
            "an_llm/data/handcoding/done/data_filtered_sample_ger.xlsx"
        ).with_columns(index=pl.col("url").cum_count())

    else:
        print(f"Language '{lang}' is not supported.")
        continue

    # define language specific paths
    prompt_path = f"an_llm/prompts/prompt_indiv_colle_{version}_{lang}.txt"
    output_path = f"an_llm/data/data_results_{version}_{lang}.xlsx"

    # Read system prompt from the corresponding prompt file
    with open(prompt_path, "r", encoding="utf8") as f:
        prompt_system = f.read()

    # Create an empty dataframe for results
    data_results = pl.DataFrame(schema={"token": pl.String, "linprob": pl.Float64})
    time_start = time.time()

    # Iterate over each row in the data and process the text
    for row in data_ready.iter_rows(named=True):
        print(f"[{lang.upper()}] Processing row: {row['index']}")
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
        # Extract log probabilities from the API response
        logprobs = response.choices[0].logprobs.content[0].top_logprobs
        rows = []
        for item in logprobs:
            rows.append(
                {
                    "token": item.token,
                    "linprob": np.round(np.exp(item.logprob) * 100, 2),
                }
            )
        data_results = data_results.vstack(pl.DataFrame(rows))

    time_end = time.time()
    print(
        f"[{lang.upper()}] Total processing time: {round((time_end - time_start) / 60, 2)} minutes"
    )

    # Map numeric tokens to their corresponding labels and pivot the results dataframe
    concept_lab = {1: "Individual", 2: "Collective", 3: "Neutral/Irrelevant"}
    data_results_wide = data_results.with_columns(
        sequence=(pl.arange(0, data_results.shape[0]) % 3) + 1,
        group=pl.arange(0, data_results.shape[0]) // 3,
        token=pl.when(pl.col("token").cast(pl.Int64, strict=False).is_in([1, 2, 3]))
        .then(pl.col("token").cast(pl.Int64, strict=False))
        .otherwise(pl.lit(None))
        .replace_strict(concept_lab),
    ).pivot(values=["token", "linprob"], index="group", on="sequence")

    # Write the processed results to an Excel file
    data_results_wide.write_excel(output_path)
    print(f"[{lang.upper()}] Results written to {output_path}")
