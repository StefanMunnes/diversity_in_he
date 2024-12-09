import polars as pl
import ftfy


replacement_dict = {
    '√§': 'ä', #
    'ГӨ': 'ä', #
    'Ă¤': 'ä', #
    'ÃĊ': 'ä', #
    'ΟΛ': 'ä', #
    'ûÊ': 'ä', #
    'УЄ': 'ä', #
    'ÃĪ': 'ä', #
    'Ã€': 'ä', #
    'Ã¤': 'ä', #
    '├ż': 'ä', #
    'Ć¤': 'ä', #
    '√∂': 'ö', #
    'Ă¶': 'ö', #
    'ΟΕ': 'ö', #
    'ûÑ': 'ö', #
    'УЖ': 'ö', #
    'Ãķ': 'ö', #
    'Г¶': 'ö', #
    '├Č': 'ö', #
    'Ć¶': 'ö', #
    '√ľ': 'ü', #
    'Гј': 'ü', #
    'ĂĽ': 'ü', #
    'Ãỳ': 'ü', #
    'УМ': 'ü', #
    # 'û': 'ü', #
    'ΟΦ': 'ü', #
    'û¥': 'ü', #
    'Ãž': 'ü', #
    'ÃŒ': 'ü', #
    'Ã¼': 'ü', #
    '├╝': 'ü', #
    'uМҲ': 'ü', #
    'uäˆ': 'ü', #
    'Ć¼': 'ü', #
    '√Ą': 'Ä', #
    'Ã': 'Ä', #
    'Ã""': 'Ä', #
    'û""': 'Ä', #
    '√Ė': 'Ö', #
    'Ã¶': 'Ö', #
    'Ã': 'Ö', #
    'Ã–': 'Ö', #
    'Ć': 'Ö', #
    '√ú': 'Ü',
    'Ăś': 'Ü', # Übersicht
    '├£': 'Ü', #
    'Ã': 'Ü', #
    'Гң': 'Ü', #
    'Ο€': 'Ü', #
    'ûœ': 'Ü', #
    'Ãœ': 'Ü', #
    'Ć': 'Ü', #
    '√ü': 'ß', #
    'Гҹ': 'ß', #
    'Οü': 'ß', #
    'ûŸ': 'ß', #
    'УЖ': 'ß', #
    'ÃŸ': 'ß', #
    'Ã': 'ß', #
    '├¤': 'ß', #
    'У': 'ß', #
    'Ć': 'ß', #
    # 'û': 'ß', #
    '‚Äě': ' ', #
    'вҖһ': ' ', #
    '‚Äú': ' ', #
    'вҖ“': ' ', #
    'â€': ' ', #
    "'Äě": ' ', #
    "'Äú": ' ', #
    "'Äď": ' ', #
    "'Äô": ' ', #
    "'Äö": ' ', #
    "'Äė": ' ', #
    "'Ä¶": ' ', #
}

# Ã 149

def correct_umlaute(text):
    # Ensure the input is a string for replacement operation
    if isinstance(text, str):

        text = ftfy.fix_text(text)

        for wrong, right in replacement_dict.items():
            text = text.replace(wrong, right)

    return text


data_ger = pl.scan_csv("scraping/Germany/scraped_data.csv")

querry = (
    data_ger
    .with_columns(
        pl.col("text").map_elements(lambda t: correct_umlaute(t), return_dtype=pl.Utf8)
    )
)

data_ger = querry.collect()

data_ger.write_csv("scraping/Germany/scraped_data.csv")
