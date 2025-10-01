import os
import re
import google.generativeai as genai
from pydantic import BaseModel, Field
from typing import List



medical_prompt_1 = """
Here are some examples of key features that are often present in
a patient who has neoplasms. Each feature is shown between the tag 
<example></example>.
<example>Shortness of breath.</example>
<example>Drenching night sweats.</example>
<example>Abnormal lumps or bumps.</example>
<example>Changes in bowel or bladder habits.</example>
List 30 other different important features that are often present
when having neoplasms. Need to follow the template above, i.e. 
<example>features</example>.
"""


medical_prompt_2 = """
Here are some examples of key features that are often present in
a patient who has digestive system diseases. Each feature is shown between the tag 
<example></example>.
<example>abdominal pain or cramping.</example>
<example>nausea and vomiting.</example>
<example>bloating and gas.</example> 
<example>heartburn.</example> 
List 30 other different important features that are often present
when having digestive system diseases. Need to follow the template above, i.e. 
<example>features</example>.
"""

medical_prompt_3 = """
Here are some examples of key features that are often present in
a patient who has nervous system diseases. Each feature is shown between the tag 
<example></example>.
<example>Persistent or sudden onset of a headache.</example>
<example>Loss of feeling or tingling.</example>
<example>Weakness or loss of muscle strength.
</example> 
<example>Loss of sight or double vision.</example> 
List 30 other different important features that are often present
when having nervous system diseases. Need to follow the template above, i.e. 
<example>features</example>.
"""

medical_prompt_4 = """
Here are some examples of key features that are often present in
a patient who has cardiovascular diseases. Each feature is shown between the tag 
<example></example>.
<example>Shortness of breath.</example> 
<example>Pain in legs when walking.</example> 
<example>Bulging neck veins.</example> 
<example>Abnormal heart sounds.</example> 
List 30 other different important features that are often present
when having cardiovascular diseases. Need to follow the template above, i.e. 
<example>features</example>.
"""


medical_prompt_5 = """
Here are some examples of key features that are often present in
a patient who has general pathological diseases. Each feature is shown between the tag 
<example></example>.
<example>Unexpected blood loss.</example> 
<example>Elevated body temperature.</example> 
<example>Dehydration.</example> 
<example>Increased pulse and respiration.</example> 
List 30 other different important features that are often present
when having general pathological diseases. Need to follow the template above, i.e. 
<example>features</example>.
"""


ag_news_prompt_1 = """
Here are some examples of key features that are often present in
worldwide news. Each feature is shown between the tag <example></example>.
<example>words related to country and place.</example>
<example>political stunts taken by governments.</example>
<example>global issues.</example>
<example>words related to war, conflict.</example>
List other important features that are often present in worldwide news. Need to follow the template above, i.e. <example>features</example>.
"""

ag_news_prompt_2 = """
Here are some examples of key features that are often present
in sport news. Each feature is shown between the tag <example></example>.
<example>name of sports stars.</example>
<example>words related to game, competition.</example>
<example>ball games like baseball, basketball.</example>
<example>name of sport teams.</example>
List other important features that are often present in sport
news. Need to follow the template above, i.e. <example>features</example>.
"""

ag_news_prompt_3 = """
Here are some examples of key features that are often present in
business and financial news. Each feature is shown between the
tag <example></example>.
<example>words related to currency, money.</example>
<example>the numerical amount of dollars.</example>
<example>the symbol like $.</example>
<example>words related to stock, Portfolio.</example>
List other important features that are often present in business
and financial news. Need to follow the template above, i.e. <example>features</example>.
"""


ag_news_prompt_4 = """
Here are some examples of key features that are often present in
news related to science and technology. Each feature is shown
between the tag <example></example>.
<example>name of scientists or the word scientists.</example>
<example>words related to technical devices.</example>
<example>words related to universe, space, planet.</example>
<example>words related to the natural landscape.</example>
List other important features that are often present in news
related to science and technology. Need to follow the template
above, i.e. <example>features</example>.
"""


dbpedia_prompt_1 = """
Here are some examples of key features that are often present when
introducing a company. Each feature is shown between the tag
<example></example>.
<example>the name of the company.</example>
<example>the location of the company</example>
<example>the founding year of the company</example>
<example>words related to organization, group.</example>
List 30 other important features that are often present when introducing a company. Need to follow the template above, i.e.
<example>features</example>.
"""

dbpedia_prompt_2 = """
Here are some examples of key features that are often present
when introducing an educational institution. Each feature is shown
between the tag <example></example>.
<example>the name of the school.</example>
<example>the location of the school</example>
<example>the founding year of the school</example>
<example>words related to college, university.</example>
List 30 other important features that are often present when introducing an educational institution. Need to follow the template
above, i.e. <example>features</example>.
"""

dbpedia_prompt_3 = """
Here are some examples of key features that are often present
when introducing an artist. Each feature is shown between the tag
<example></example>.
<example>the artist’s name.</example>
<example>the artist’s works</example>
<example>the artist’s born date</example>
<example>words related to music, painting.</example>
List 30 other important features that are often present when introducing an artist. Need to follow the template above, i.e. <example>features</example>.
"""

dbpedia_prompt_4 = """
Here are some examples of key features that are often present when
introducing an athlete or sports star. Each feature is shown between
the tag <example></example>.
<example>the athlete’s or sports stars’ name.</example>
<example>the sport the athlete plays (e.g. football, basketball).</example>
<example>the athlete’s or sports stars’ born date</example>
<example>words related to ball games, competition.</example>
List 30 other important features that are often present when introducing an athlete or sports star. Need to follow the template above,
i.e. <example>features</example>.
"""

dbpedia_prompt_5 = """
Here are some examples of key features that are often present when
introducing an office holder. Each feature is shown between the
tag <example></example>.
<example>the office holder’s name.</example>
<example>the office holder’s position.</example>
<example>the office holder’s born date</example>
<example>words related to politician, businessman.</example>
List 30 other important features that are often present when introducing an office holder. Need to follow the template above, i.e.
<example>features</example>.
"""

dbpedia_prompt_6 = """
Here are some examples of key features that are often present
when introducing transportation. Each feature is shown between
the tag <example></example>.
<example>the model type of the transportation or vehicle.</example>
<example>the production date of the transportation or vehicle.</example>
<example>the functions of the transportation or vehicle.</example>
<example>words related to ship, car, train.</example>
List 30 other important features that are often present when
introducing transportation. Need to follow the template above, i.e.
<example>features</example>.
"""


dbpedia_prompt_7 = """
Here are some examples of key features that are often present
when introducing a building. Each feature is shown between the
tag <example></example>.
<example>the name of the building.</example>
<example>the built date of the building.</example>
<example>the location of the building.</example>
<example>words related to the type of the building (e.g. church,
historic house, park, resort).</example>
List 30 other important features that are often present when introducing a building. Need to follow the template above, i.e. <example>features</example>.
"""

dbpedia_prompt_8 = """
Here are some examples of key features that are often present when
introducing a natural place. Each feature is shown between the tag
<example></example>.
<example>the name of the natural place.</example>
<example>the length or height of the natural place.</example>
<example>the location of the natural place.</example>
<example>words related to mountain, river.</example>
List 30 other important features that are often present when introducing a natural place. Need to follow the template above, i.e.
<example>features</example>.
"""


dbpedia_prompt_9 = """
Here are some examples of key features that are often present
when introducing a village. Each feature is shown between the tag
<example></example>.
<example>the name of the village.</example>
<example>the population of the village.</example>
<example>the census of the village.</example>
<example>words related to district, families.</example>
List 30 other important features that are often present when introducing a village. Need to follow the template above, i.e. <example>features</example>.
"""

dbpedia_prompt_10 = """
Here are some examples of key features that are often present when
introducing a kind of animal. Each feature is shown between the
tag <example></example>.
<example>the species of the animal.</example>
<example>the habitat of the animal.</example>
<example>the type of the animal (e.g. bird, insect,
moth).</example>
<example>words related to genus, family.</example>
List 30 other important features that are often present when introducing a kind of animal. Need to follow the template above, i.e.
<example>features</example>.
"""

dbpedia_prompt_11 = """
Here are some examples of key features that are often present when
introducing a kind of plant. Each feature is shown between the tag
<example></example>.
<example>the name of the plant.</example>
<example>the genus or family of plant.</example>
<example>the place where the plant was found.</example>
<example>words related to grass, herb, flower.</example>
List 30 other important features that are often present when introducing a kind of plant. Need to follow the template above, i.e.
<example>features</example>.
"""

dbpedia_prompt_12 = """
Here are some examples of key features that are often present
when introducing an album. Each feature is shown between the tag
<example></example>.
<example>the name of the album.</example>
<example>the type of music, instrument.</example>
<example>the release date of the album.</example>
<example>words related to band, studio.</example>
List 30 other important features that are often present when introducing an album. Need to follow the template above, i.e. <example>features</example>.
"""

dbpedia_prompt_13 = """
Here are some examples of key features that are often present
when introducing a film. Each feature is shown between the tag
<example></example>.
<example>the name of the film.</example>
<example>the maker or producer of the film.</example>
<example>the type of the film (e.g. drama, science fiction, comedy,
cartoon, animation).</example>
<example>words related to TV, video.</example>
List 30 other important features that are often present when introducing a film. Need to follow the template above, i.e. <example>features</example>.
"""

dbpedia_prompt_14 = """
Here are some examples of key features that are often present when
introducing a written work. Each feature is shown between the tag
<example></example>.
<example>the name of the written work.</example>
<example>the author of the film.</example>
<example>the type of the written work (e.g. novel, manga, journal).</example>
<example>words related to book.</example>
List 30 other important features that are often present when introducing a written work. Need to follow the template above, i.e.
<example>features</example>.
"""


legal_prompt_1 = """
Here are some examples of key features that are often present when a higher court affirms the decision of a lower court. Each feature is shown between the tag <example></example>.
<example>Appellate court judgment explicitly stating no error by lower tribunal.</example>
<example>Confirmation of the original decision's validity.</example>
<example>Lower court's ruling stands.</example>
<example>No new trial or re-evaluation ordered.</example>
<example>Appellate court's opinion upholding the prior verdict.</example>
List 30 other important features that are often present when a higher court affirms a lower court's decision. Need to follow the template above, i.e.,
<example>features</example>.
"""


legal_prompt_2 = """
Here are some examples of key features that are often present when a court applies the principles or rules of a previously decided legal case to the current set of facts. Each feature is shown between the tag <example></example>.
<example>Legal principle from an earlier ruling used to resolve current dispute.</example>
<example>Facts of the current matter determined to fit prior established rule.</example>
<example>Court's reasoning directly mirrors precedent's methodology.</example>
<example>New ruling consistent with a specific prior holding.</example>
<example>Resolution of the case relies on established legal framework from another decision.</example>
List 30 other important features that are often present when a court applies a previously decided legal case. Need to follow the template above, i.e.
<example>features</example>.
"""

legal_prompt_3 = """
Here are some examples of key features that are often present when a court approves a specific part of a prior decision, an argument, or a legal principle from a previously decided case. Each feature is shown between the tag <example></example>.
<example>Judicial endorsement of a prior legal interpretation.</example>
<example>Later court expresses agreement with earlier case's reasoning.</example>
<example>Acceptance of a specific legal test or standard from a prior ruling.</example>
<example>Endorsement of a lower court's analysis on a particular point.</example>
<example>Confirmation of the correctness of a legal position.</example>
List 30 other important features that are often present when a court approves a specific aspect of a previously decided legal case or precedent. Need to follow the template above, i.e.,
<example>features</example>.
"""

legal_prompt_4 = """
Here are some examples of key features that are often present when a court cites a previously decided legal case or authority. Each feature is shown between the tag <example></example>.
<example>Footnote or parenthetical reference to an earlier opinion.</example>
<example>Inclusion of a prior case name and reporter information.</example>
<example>Reference to a previous decision within the body of the text.</example>
<example>Acknowledgement of a source for a legal proposition.</example>
<example>Inclusion on a list of relevant authorities.</example>
List 30 other important features that are often present when a court cites a previously decided legal case. Need to follow the template above, i.e.,
<example>features</example>.
"""

legal_prompt_5 = """
Here are some examples of key features that are often present when a court has considered a previously decided legal case or a legal principle, indicating it has taken it into account during its deliberations. Each feature is shown between the tag <example></example>.
<example>Discussion of a prior case's facts or holding in the court's opinion.</example>
<example>Analysis of how a prior ruling might bear on the present issue.</example>
<example>Explicit mention of reviewing a specific precedent during deliberation.</example>
<example>Acknowledgement that a relevant case was evaluated for applicability.</example>
<example>Assessment of a prior decision's persuasive value.</example>
List 30 other important features that are often present when a court has considered a previously decided legal case or principle. Need to follow the template above, i.e.,
<example>features</example>.
"""

legal_prompt_6 = """
Here are some examples of key features that are often present when a court discusses a previously decided legal case, engaging in an in-depth examination or analysis of its facts, reasoning, or implications. Each feature is shown between the tag <example></example>.
<example>Detailed recounting of an earlier case's factual background.</example>
<example>Analysis of the judicial reasoning employed in a prior ruling.</example>
<example>Exploration of the implications or scope of a precedent.</example>
<example>Explanation of the similarities or differences with another decision.</example>
<example>Commentary on the significance or evolution of a legal concept from an earlier case.</example>
List 30 other important features that are often present when a court discusses a previously decided legal case. Need to follow the template above, i.e.,
<example>features</example>.
"""

legal_prompt_7 = """
Here are some examples of key features that are often present when a court distinguishes a previously decided legal case, stating that it is not applicable to the current situation due to fundamental differences in facts or legal issues. Each feature is shown between the tag <example></example>.
<example>Identification of material factual differences from a prior case.</example>
<example>Explanation of why a previous rule does not govern the current situation.</example>
<example>Highlighting unique circumstances that prevent direct application of a precedent.</example>
<example>Court explains why a seemingly similar earlier case is not controlling.</example>
<example>Statement that the principle from the earlier case is inapplicable due to specific variations.</example>
List 30 other important features that are often present when a court distinguishes a previously decided legal case. Need to follow the template above, i.e.,
<example>features</example>.
"""

legal_prompt_8 = """
Here are some examples of key features that are often present when a court follows the direct holding, rule, or reasoning of a previously decided legal case because the current case presents substantially similar facts or legal issues. Each feature is shown between the tag <example></example>.
<example>Adherence to the precise legal rule established in an earlier ruling.</example>
<example>Court reaches the same legal conclusion based on similar facts and precedent.</example>
<example>Implementation of a binding rule from a higher court.</example>
<example>Application of the established doctrine from a prior decision.</example>
<example>Court's reasoning directly mirrors the path laid out by an earlier, relevant case.</example>
List 30 other important features that are often present when a court follows a previously decided legal case. Need to follow the template above, i.e.,
<example>features</example>.
"""

legal_prompt_9 = """
Here are some examples of key features that are often present when a court refers to a previously decided legal case or authority, making a mention or general acknowledgment without necessarily engaging in deep analysis or application. Each feature is shown between the tag <example></example>.
<example>Mention of a prior case in legal arguments or opinions.</example>
<example>Passing acknowledgment of an earlier decision.</example>
<example>Casual mention of a related legal authority.</example>
<example>Brief nod to a previous ruling without detailed analysis.</example>
<example>Indication that a prior case is part of the background legal landscape.</example>
List 30 other important features that are often present when a court refers to a previously decided legal case. Need to follow the template above, i.e.,
<example>features</example>.
"""

legal_prompt_10 = """
Here are some examples of key features that are often present when a court identifies a previously decided legal case as related to the current matter due to shared legal topics, similar factual contexts, or a common lineage within a body of law. Each feature is shown between the tag <example></example>.
<example>The current issue shares common legal themes with an earlier ruling.</example>
<example>Legal questions in the current case touch upon areas covered by a previous decision.</example>
<example>Prior judgment addresses similar types of facts or legal relationships.</example>
<example>The subject matter of the case has tangential connections to an earlier opinion.</example>
<example>Overlap in the area of law being litigated with a previous case.</example>
List 30 other important features that are often present when a court identifies a previously decided legal case as related to the current matter. Need to follow the template above, i.e.,
<example>features</example>.
"""

ecommerce_prompt_1 = """
Here are some examples of key features that are often present when describing an e-commerce product that falls into the Household category. Each feature is shown between the tag <example></example>.
<example>Often related to interior decor or functional home utility.</example>
<example>Physical dimensions (length, width, height) are frequently important.</example>
<example>Materials like wood, metal, plastic, or fabric are common in construction.</example>
<example>Can involve setup or assembly for full use.</example>
<example>Primary purpose is to furnish, maintain, or enhance living spaces.</example>
List other 30 important features that are often present when describing an e-commerce product specifically for the Household category. Need to follow the template above, i.e.
<example>features</example>.
"""


ecommerce_prompt_2 = """
Here are some examples of key features that are often present when describing an e-commerce product that falls into the Books category. Each feature is shown between the tag <example></example>.
<example>Features an author and a publisher.</example>
<example>Identified by an International Standard identifier (e.g., ISBN).</example>
<example>Available in various formats like physical bound copies or digital files.</example>
<example>Categorized by subject matter or narrative style.</example>
<example>Contains written or illustrated content intended for reading.</example>
List other 30 important features that are often present when describing an e-commerce product specifically for the Books category. Need to follow the template above, i.e.
<example>features</example>.
"""

ecommerce_prompt_3 = """
Here are some examples of key features that are often present when describing an e-commerce product that falls into the Electronics category. Each feature is shown between the tag <example></example>.
<example>Relies on electrical power for operation.</example>
<example>Detailed technical specifications (e.g., RAM, storage, connectivity) are crucial.</example>
<example>Often includes processors, circuits, or digital displays.</example>
<example>Frequently requires compatibility with other devices or software.</example>
<example>Provides functions such as computing, communication, or audio/visual output.</example>
List other 30 important features that are often present when describing an e-commerce product specifically for the Electronics category. Need to follow the template above, i.e.
<example>features</example>.

"""

ecommerce_prompt_4 = """
Here are some examples of key features that are often present when describing an e-commerce product that falls into the Clothing & Accessories category. Each feature is shown between the tag <example></example>.
<example>Primary features include specific sizes and fits.</example>
<example>Made from various textiles like cotton, polyester, or leather.</example>
<example>Associated with a gender or age group for wear.</example>
<example>Offers a wide range of colors and patterns.</example>
<example>Designed to be worn on the body or carried as a personal item.</example>
List other 30 important features that are often present when describing an e-commerce product specifically for the Clothing & Accessories category. Need to follow the template above, i.e.
<example>features</example>.
"""


stackoverflow_prompt_1 = """
Here are some examples of key features that are often present when describing a Stack Overflow question that is classified as HQ: High-quality posts without a single edit. Each feature is shown between the tag <example></example>.
<example>Receives multiple upvotes shortly after posting.</example>
<example>Attracts highly relevant and accurate answers quickly.</example>
<example>Code examples are well-formatted, complete, and reproducible.</example>
<example>The problem statement is universally understood on first reading.</example>
<example>All necessary context and error messages are clearly provided upfront.</example>
List other 20 important features that are often present when describing a Stack Overflow question that is classified as HQ. Need to follow the template above, i.e.
<example>features</example>.
"""


stackoverflow_prompt_2 = """
Here are some examples of key features that are often present when describing a Stack Overflow question that is classified as LQ_EDIT: Low-quality posts with a negative score, and multiple community edits, but remain open. Each feature is shown between the tag <example></example>.
<example>Has a score of 0 or negative (e.g., -1, -2) for a period.</example>
<example>Displays a history of several edits made by users other than the original author.</example>
<example>Initial comments often request clarification or additional information.</example>
<example>The question's original tags were vague or incorrect, requiring correction.</example>
<example>It was initially hard to read due to poor formatting, which was later fixed by others.</example>
List other 20 important features that are often present when describing a Stack Overflow question that is classified as LQ_EDIT. Need to follow the template above, i.e.
<example>features</example>.
"""


stackoverflow_prompt_3 = """
Here are some examples of key features that are often present when describing a Stack Overflow question that is classified as LQ_CLOSE: Low-quality posts that were closed by the community without a single edit. Each feature is shown between the tag <example></example>.
<example>Shows a "Closed" banner prominently, often with a reason like "duplicate" or "off-topic."</example>
<example>The question history shows zero edits by anyone (including the author).</example>
<example>Comments often point to the question being a duplicate or not suitable for the platform.</example>
<example>Does not attract any answers, or only highly speculative/non-solution answers.</example>
<example>Asks for broad advice, opinions, or solutions to entire projects rather than specific problems.</example>
List other 20 important features that are often present when describing a Stack Overflow question that is classified as LQ_CLOSE. Need to follow the template above, i.e.
<example>features</example>.
"""

pubmed_prompt_background = """
Here are some examples of key features that are often present when a sentence serves as BACKGROUND in a medical abstract. Each feature is shown between the tag <example></example>.
<example>Establishing the prevalence or significance of a medical condition.</example>
<example>Reference to existing knowledge or a summary of prior studies.</example>
<example>Identification of a gap, controversy, or an unanswered question in the current literature.</example>
<example>Description of the standard treatment, current practice, or existing diagnostic method.</example>
<example>Statement about the underlying mechanism of a disease.</example>
List 30 other important features that are often present when a sentence provides BACKGROUND information in a medical abstract. Need to follow the template above, i.e.,
<example>features</example>.
"""

pubmed_prompt_objective = """
Here are some examples of key features that are often present when a sentence states the OBJECTIVE of a medical study. Each feature is shown between the tag <example></example>.
<example>Explicit statement of the study's primary goal using phrases like "the aim was to".</example>
<example>Declaration of the main hypothesis to be tested.</example>
<example>Use of infinitive verbs to describe the purpose (e.g., "to determine," "to investigate," "to compare").</example>
<example>Defining the primary endpoint or main outcome of interest for the study.</example>
<example>Stating the specific interventions or populations being evaluated.</example>
List 30 other important features that are often present when a sentence outlines the OBJECTIVE of a medical study. Need to follow the template above, i.e.,
<example>features</example>.
"""

pubmed_prompt_methods = """
Here are some examples of key features that are often present when a sentence describes the METHODS of a medical study. Each feature is shown between the tag <example></example>.
<example>Description of the study design (e.g., randomized controlled trial, double-blind, cohort study).</example>
<example>Details on participant selection criteria (inclusion/exclusion).</example>
<example>Explanation of the intervention administered to the treatment group.</example>
<example>Specification of the control group (e.g., placebo, standard care).</example>
<example>Mention of the statistical tests used for data analysis.</example>
List 30 other important features that are often present when a sentence details the METHODS of a medical study. Need to follow the template above, i.e.,
<example>features</example>.
"""

pubmed_prompt_results = """
Here are some examples of key features that are often present when a sentence reports the RESULTS of a medical study. Each feature is shown between the tag <example></example>.
<example>Reporting of specific numerical data, such as percentages, means, or counts.</example>
<example>Use of statistical language, like p-values or confidence intervals (CI).</example>
<example>Direct comparison of outcomes between the study's intervention and control groups.</example>
<example>Phrases indicating findings, such as "was significantly higher" or "showed no difference."</example>
<example>Statement of the main findings without interpretation or implications.</example>
List 30 other important features that are often present when a sentence presents the RESULTS of a medical study. Need to follow the template above, i.e.,
<example>features</example>.
"""

pubmed_prompt_conclusions = """
Here are some examples of key features that are often present when a sentence states the CONCLUSIONS of a medical study. Each feature is shown between the tag <example></example>.
<example>Interpretation of what the main results mean.</example>
<example>Statement of the clinical or practical implications of the findings.</example>
<example>Phrases indicating a summary, such as "in conclusion," or "our findings suggest that."</example>
<example>Comparison of the study's findings with previous research or established knowledge.</example>
<example>Suggestion for future research or acknowledgment of the study's limitations.</example>
List 30 other important features that are often present when a sentence draws CONCLUSIONS from a medical study. Need to follow the template above, i.e.,
<example>features</example>.
"""

drug_prompt_rating_1 = """
Here are some examples of key features that are often present when a user gives a drug a very low rating (e.g., 1 out of 10). Each feature is shown between the tag <example></example>.
<example>The drug had no effect on the condition.</example>
<example>Severe or unbearable side effects were experienced.</example>
<example>The user's condition became worse after taking the medication.</example>
<example>Strong expression of regret or disappointment about using the drug.</example>
<example>Explicit warning for others not to take this medication.</example>
List 30 other important features that are often present when a user gives a drug a very negative rating. Need to follow the template above, i.e.,
<example>features</example>.
"""

drug_prompt_rating_5 = """
Here are some examples of key features that are often present when a user gives a drug a mediocre or mixed rating (e.g., 5 out of 10). Each feature is shown between the tag <example></example>.
<example>The drug provided only minimal or slight relief.</example>
<example>Positive effects were offset by noticeable but tolerable side effects.</example>
<example>The effectiveness of the drug was inconsistent or wore off quickly.</example>
<example>Uncertainty about whether the drug was truly effective.</example>
<example>Comparison to another drug that worked better or worse.</example>
List 30 other important features that are often present when a user gives a drug a mixed or average rating. Need to follow the template above, i.e.,
<example>features</example>.
"""

drug_prompt_rating_10 = """
Here are some examples of key features that are often present when a user gives a drug a very high rating (e.g., 10 out of 10). Each feature is shown between the tag <example></example>.
<example>The drug was described as a "life-saver" or "miracle."</example>
<example>Complete or significant relief from symptoms was achieved.</example>
<example>There were no noticeable side effects experienced.</example>
<example>The user reported a major improvement in their quality of life.</example>
<example>Strong recommendation for others with the same condition.</example>
List 30 other important features that are often present when a user gives a drug a very positive rating. Need to follow the template above, i.e.,
<example>features</example>.
"""

class DrugReviewPrompt(BaseModel):
    system_prompt: str = Field(
        default="You are an expert in analyzing patient feedback, your job is to identify key themes and features in drug reviews that distinguish different levels of user satisfaction."
    )
    labels_to_prompts: dict[str, str] = Field(
        default={
            "1": drug_prompt_rating_1,
            "5": drug_prompt_rating_5,
            "10": drug_prompt_rating_10,
        }
    )

class PubmedPrompt(BaseModel):
    system_prompt: str = Field(
        default="You are an expert scientific editor and researcher, your job is to identify key linguistic and structural features that distinguish different sections of a medical research abstract.")
    labels_to_prompts: dict[str, str] = Field(
        default={
            "BACKGROUND": pubmed_prompt_background,
            "OBJECTIVE": pubmed_prompt_objective,
            "METHODS": pubmed_prompt_methods,
            "RESULTS": pubmed_prompt_results,
            "CONCLUSIONS": pubmed_prompt_conclusions,
        }
    )


class MedAbsPrompt(BaseModel):
    system_prompt: str = Field(
        default="You are an expert Diagnostic Specialist, your job is to identify some key features that will distinguish different medical conditions")
    labels_to_prompts: dict[str, str] = Field(
        default={
            "neoplasms": medical_prompt_1,
            "digestive_system_diseases": medical_prompt_2,
            "nervous_system_diseases": medical_prompt_3,
            "cardiovascular_diseases": medical_prompt_4,
            "general_pathological_diseases": medical_prompt_5,
        }
    )

class AgNewsPrompt(BaseModel):
    system_prompt: str = Field(
        default="You are an expert News Specialist, your job is to identify some key features that will distinguish different news categories")
    labels_to_prompts: dict[str, str] = Field(
        default={
            "world": ag_news_prompt_1,
            "sports": ag_news_prompt_2,
            "business": ag_news_prompt_3,
            "sci_tech": ag_news_prompt_4,
        }
    )

class DbpediaPrompt(BaseModel):
    system_prompt: str = Field(
        default="You are an expert Dbpedia Specialist, your job is to identify some key features that will distinguish different dbpedia categories")
    labels_to_prompts: dict[str, str] = Field(
        default={
            "company": dbpedia_prompt_1,
            "educational institution": dbpedia_prompt_2,
            "artist": dbpedia_prompt_3,
            "athlete": dbpedia_prompt_4,
            "office holder": dbpedia_prompt_5,
            "transportation": dbpedia_prompt_6,
            "building": dbpedia_prompt_7,
            "natural place": dbpedia_prompt_8,
            "village": dbpedia_prompt_9,
            "animal": dbpedia_prompt_10,
            "plant": dbpedia_prompt_11,
            "album": dbpedia_prompt_12,
            "film": dbpedia_prompt_13,
            "written work": dbpedia_prompt_14,
        }
    )

class LegalPrompt(BaseModel):
    system_prompt: str = Field(
        default="You are an expert Legal Specialist, your job is to identify some key features that will distinguish a specific relationship or action taken by a court regarding a prior legal authority")
    labels_to_prompts: dict[str, str] = Field(
        default={
            "affirmed": legal_prompt_1,
            "applied": legal_prompt_2,
            "approved": legal_prompt_3,
            "cited": legal_prompt_4,
            "considered": legal_prompt_5,
            "discussed": legal_prompt_6,
            "distinguished": legal_prompt_7,
            "followed": legal_prompt_8,
            "referred to": legal_prompt_9,
            "related": legal_prompt_10,
        }
    )

class EcommercePrompt(BaseModel):
    system_prompt: str = Field(
        default="You are an expert E-commerce Specialist, your job is to identify some key features that will distinguish different e-commerce product categories")
    labels_to_prompts: dict[str, str] = Field(
        default={
            "household": ecommerce_prompt_1,
            "books": ecommerce_prompt_2,
            "electronics": ecommerce_prompt_3,
            "clothing_accessories": ecommerce_prompt_4,
        }
    )


class StackOverflowPrompt(BaseModel):
    system_prompt: str = Field(
        default="You are an expert Stack Overflow Specialist, your job is to identify some key features that will distinguish different Stack Overflow question categories")
    labels_to_prompts: dict[str, str] = Field(
        default={
            "HQ": stackoverflow_prompt_1,
            "LQ_EDIT": stackoverflow_prompt_2,
            "LQ_CLOSE": stackoverflow_prompt_3,
        }
    )

def configure_llm():
    """Cấu hình API của Google Generative AI bằng cách sử dụng key được đặt trực tiếp trong code."""
    API_KEY = "AIzaSyCItfzA_PvQV4YkOck-RsP6fmtxTJBhxAE" 

    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
        raise ValueError(
            "Lỗi: Vui lòng thay thế 'YOUR_API_KEY_HERE' bằng API Key Gemini thực của bạn "
            "trong hàm configure_llm() để tiếp tục."
        )
        
    genai.configure(api_key=API_KEY)
    print("Đã cấu hình thành công API của Google Generative AI.")


def call_llm_for_features(prompt_text: str, model) -> List[str]:
    """
    Args:
        prompt_text: Toàn bộ văn bản prompt để gửi đến LLM.
        model: Đối tượng mô hình GenerativeModel đã được khởi tạo.

    Returns:
        Một danh sách các chuỗi, mỗi chuỗi là một đặc điểm do LLM tạo ra.
    """
    try:
        # Bước 1: Gọi LLM để nó SINH RA nội dung dựa trên prompt.
        response = model.generate_content(prompt_text)
        
        # Bước 2: TRÍCH XUẤT các đặc điểm từ bên trong các thẻ <example>
        # của văn bản mà LLM đã trả về. re.DOTALL cho phép '.' khớp với cả ký tự xuống dòng.
        features = re.findall(r'<example>(.*?)</example>', response.text, re.DOTALL)
        
        # Loại bỏ khoảng trắng thừa ở đầu/cuối mỗi đặc điểm đã trích xuất.
        cleaned_features = [feature.strip() for feature in features]
        return cleaned_features
    except Exception as e:
        print(f"-> Đã xảy ra lỗi khi gọi LLM hoặc xử lý phản hồi: {e}")
        return []

def save_list_to_json(data: List[str], full_filepath: str):
    """
    Lưu một danh sách các chuỗi vào một tệp JSON.
    Tự động tạo thư mục nếu nó chưa tồn tại.
    """
    try:
        directory = os.path.dirname(full_filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Đã tạo thư mục: '{directory}'")

        # Mở tệp và ghi đối tượng Python (list) vào dưới dạng JSON
        with open(full_filepath, 'w', encoding='utf-8') as f:
            # json.dump để ghi list vào file
            # indent=4 để file JSON được định dạng đẹp, dễ đọc
            # ensure_ascii=False để đảm bảo các ký tự Unicode (nếu có) được ghi đúng
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        print(f"\nThành công! Đã lưu {len(data)} đặc điểm vào tệp JSON tại đường dẫn:")
        print(f"==> {full_filepath}")

    except Exception as e:
        print(f"\nĐã xảy ra lỗi khi lưu tệp JSON: {e}")
        print("Vui lòng kiểm tra lại đường dẫn và quyền ghi của thư mục.")

import json

def main():
    """Hàm chính điều phối toàn bộ quá trình."""
    output_directory = r"D:\Lap\iSE\cb-llm-v0\concept_gen"
    # Thay đổi tên tệp đầu ra thành .js on
    output_filename = "generated_medical_features.json"
    full_output_path = os.path.join(output_directory, output_filename)
    
    try:
        configure_llm()
        prompt_data = MedAbsPrompt()
        
        # Lưu ý: 'gemini-2.0-flash-lite' không phải là một model hợp lệ tại thời điểm này.
        # Sử dụng 'gemini-1.5-flash-latest' là lựa chọn nhanh và hiệu quả.
        llm_model = genai.GenerativeModel('gemini-2.0-flash-lite')

        all_generated_features = []

        print("\nBắt đầu quá trình tạo các đặc điểm y tế...")
        print("-" * 50)

        for label, user_prompt in prompt_data.labels_to_prompts.items():
            print(f"Đang tạo đặc điểm cho: '{label}'...")
            full_prompt = f"{prompt_data.system_prompt}\n\n{user_prompt}"
            features = call_llm_for_features(full_prompt, llm_model)
            
            if features:
                print(f"-> Hoàn tất. Tạo được {len(features)} đặc điểm.")
                all_generated_features.extend(features)
            else:
                print(f"-> Không tạo được đặc điểm nào cho '{label}'.")

        print("-" * 50)
        
        if all_generated_features:
            # Gọi hàm lưu file JSON mới
            save_list_to_json(all_generated_features, full_output_path)
        else:
            print("Không có đặc điểm nào được tạo. Sẽ không tạo tệp output.")

    except ValueError as ve:
        print(ve)
    except Exception as e:
        print(f"Đã xảy ra một lỗi không mong muốn trong quá trình thực thi: {e}")

if __name__ == "__main__":
    main()