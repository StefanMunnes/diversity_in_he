import json

from HanTa import HanoverTagger as ht
tagger_en = ht.HanoverTagger('morphmodel_en.pgz')
tagger_de = ht.HanoverTagger('morphmodel_ger.pgz')


lexicon = {
  "individual": [
        "equal opportunity",
        "equal conditions",
        "equal chances",
        "participate",
        "participation",
        "contribute",
        "contribution",
        "individual",
        "individuality",
        "individualized",
        "individually",
        "self",
        "oneself",
        "member",
        "person",
        "personal",
        "personalize",
        "no one",
        "each",
        "every",
        "right",
        "freedom",
        "autonomy",
        "autonomous",
        "self-determination",
        "identity",
        "background",
        "unique",
        "uniqueness",
        "uniquely",
        "talent",
        "skill",
        "handicap",
        "regardless of",
        "free from",
        "growth",
        "achieve",
        "achiever",
        "achievement",
        "innovate",
        "innovation"
  ],
  "collective": [
        "justice",
        "equity",
        "community",
        "collaboration",
        "collaborative",
        "belonging",
        "unity",
        "cohesion",
        "public",
        "social",
        "multicultural",
        "tradition",
        "tolerance",
        "minority",
        "marginalized",
        "marginalization",
        "sustainability",
        "commitment",
        "solidarity",
        "allyship",
        "interpersonal",
        "interdependence"
  ],
  "neutral": [
        "global",
        "exchange",
        "network",
        "responsibility",
        "empower",
        "empowering",
        "empowered",
        "empowerment",
        "fair",
        "fairness",
        "fairly",
        "advocacy",
        "advance",
        "advancement",
        "excellence",
        "knowledge",
        "inclusion",
        "inclusive",
        "exclusive",
        "exclusion",
        "discrimination",
        "harassment",
        "diversity"
    ]
}


lexicon = {
    "individual": list(set([item[1] for item in tagger_en.tag_sent(lexicon["individual"])])),
    "collective": list(set([item[1] for item in tagger_en.tag_sent(lexicon["collective"])])),
    "neutral": list(set([item[1] for item in tagger_en.tag_sent(lexicon["neutral"])])),
}


# write lexicon to json file
with open("an_lexicon/data/lexicon.json", "w") as f:
    json.dump(lexicon, f, indent=4)
