import concepts

example_name = {'SetFit/sst2': 'text', 
                "fancyzhx/ag_news": 'text', 
                "fancyzhx/yelp_polarity": 'text', 
                "fancyzhx/dbpedia_14": 'content', 
                "Duyacquy/Single-label-medical-abstract": 'medical_abstract', 
                "Duyacquy/Legal-text": 'case_text', 
                "Duyacquy/Ecommerce-text": 'text',
                "Duyacquy/Pubmed-20k": 'abstract_text',
                "Duyacquy/UCI-drug": 'review'} # Done
concepts_from_labels = {'SetFit/sst2': ["negative","positive"], "fancyzhx/yelp_polarity": ["negative","positive"], "fancyzhx/ag_news": ["World", "Sports", "Business", "Sci/Tech"], "fancyzhx/dbpedia_14": [
        "company",
        "educational institution",
        "artist",
        "athlete",
        "office holder",
        "mean of transportation",
        "building",
        "natural place",
        "village",
        "animal",
        "plant",
        "album",
        "film",
        "written work"
    ], 
    "Duyacquy/Single-label-medical-abstract": ["1", "2", "3", "4", "5"], 
    "Duyacquy/Legal-text": ["applied", "cited", "considered", "followed", "referred to"], 
    "Duyacquy/Ecommerce-text": ["Household", "Books", "Electronics", "Clothing & Accessories"], 
    "Duyacquy/Stack-overflow-question": ["HQ", "LQ_EDIT", "LQ_CLOSE"],
    "Duyacquy/Pubmed-20k": ["CONCLUSION", "BACKGROUND", "METHODS", "RESULTS", "OBJECTIVE"],
    "Duyacquy/UCI-drug": ["1", "5", "10"]
} # Done

class_num = {'SetFit/sst2': 2, "fancyzhx/ag_news": 4, 
             "fancyzhx/yelp_polarity": 2, "fancyzhx/dbpedia_14": 14, 
             "Duyacquy/Single-label-medical-abstract": 5, "Duyacquy/Legal-text": 5, 
             "Duyacquy/Ecommerce-text": 4, "Duyacquy/Stack-overflow-question": 3,
             "Duyacquy/Pubmed-20k": 5, "Duyacquy/UCI-drug": 3} # Done

# Config for Roberta-Base baseline
finetune_epoch = {'SetFit/sst2': 3, "fancyzhx/ag_news": 2, "fancyzhx/yelp_polarity": 2, "fancyzhx/dbpedia_14": 2}
finetune_mlp_epoch = {'SetFit/sst2': 30, "fancyzhx/ag_news": 5, "fancyzhx/yelp_polarity": 3, "fancyzhx/dbpedia_14": 3}

# Config for CBM training
concept_set = {'SetFit/sst2': concepts.sst2, 
               "fancyzhx/yelp_polarity": concepts.yelpp, 
               "fancyzhx/ag_news": concepts.agnews, 
               "fancyzhx/dbpedia_14": concepts.dbpedia, 
               "Duyacquy/Single-label-medical-abstract": concepts.med_abs, 
               "Duyacquy/Legal-text": concepts.legal, 
               "Duyacquy/Ecommerce-text": concepts.ecom, 
               "Duyacquy/Stack-overflow-question": concepts.stackoverflow,
               "Duyacquy/Pubmed-20k": concepts.pubmed,
               "Duyacquy/UCI-drug": concepts.drug} # Done

cbl_epochs = {'SetFit/sst2': 10, "fancyzhx/ag_news": 10,
              "fancyzhx/yelp_polarity": 10, "fancyzhx/dbpedia_14": 10,
              "Duyacquy/Single-label-medical-abstract": 10, "Duyacquy/Legal-text": 10,
              "Duyacquy/Ecommerce-text": 10, "Duyacquy/Stack-overflow-question": 10,
              "Duyacquy/Pubmed-20k": 10, "Duyacquy/UCI-drug": 10} # Done

#Duyacquy/Single-label-medical-abstract
#TimSchopf/medical_abstracts
# ok

#Duyacquy/Pubmed-20k
#pietrolesci/pubmed-200k-rct
# abstract_text, target
# ok

#Duyacquy/Ecommerce-text
#darklord1611/ecom_categories
# ok


#Duyacquy/UCI-drug
#dd-n-kk/uci-drug-review-cleaned
# review, rating
#ok


#Duyacquy/Legal-text
#darklord1611/legal_citations

#Duyacquy/Stack-overflow-question
#darklord1611/stackoverflow_question_ratings

dataset_config = {
    "Duyacquy/Single-label-medical-abstract": {
        "text_column": "medical_abstract",
        "label_column": "condition_label"
    },
    "SetFit/20_newsgroups": {
        "text_column": "text",
        "label_column": "label"
    },
    "JuliaTsk/yahoo-answers": {
        "text_column": "question title",
        "label_column": "class id"
    },
    "fancyzhx/ag_news": {
        "text_column": "text",
        "label_column": "label"
    },
    "fancyzhx/dbpedia_14": {
        "text_column": "content",
        "label_column": "label"
    },
    "SetFit/sst2": {
        "text_column": "text",
        "label_column": "label"
    },
    "fancyzhx/yelp_polarity": {
        "text_column": "text",
        "label_column": "label"
    },
    "Duyacquy/Pubmed-20k": {
        "text_column": "abstract_text",
        "label_column": "target"
    },
    "Duyacquy/UCI-drug": {
        "text_column": "review",
        "label_column": "rating"
    },
    "Duyacquy/Legal-text": {
        "text_column": "case_text",
        "label_column": "case_outcome"
    },
    "Duyacquy/Ecommerce-text": {
        "text_column": "text",
        "label_column": "label"
    },
    "Duyacquy/Stack-overflow-question": {
        "text_column": "Text",
        "label_column": "Y"
    }
}
