import pandas as pd
import numpy as np
import os
import requests
import re
import logging
from tqdm import tqdm
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("deepseek_log.txt", mode="w"),
        logging.StreamHandler()
    ]
)

# Change directory and load data
os.chdir("/Users/lhampton/Library/CloudStorage/GoogleDrive-lucyjhampton5@gmail.com/.shortcut-targets-by-id/1EznD-uaTblXtaIVx04-toZMQh7cQf9hA/AI Landscape Large Files/Data/constructed")
full_dataset = pd.read_csv("pub_wp_full.csv")
logging.info("Loaded dataset with %d records", len(full_dataset))

# final prompt
prompt = """You are an academic expert in economics, finance, management, and innovation studies. You are provided with abstracts from academic papers. Your goal is to carefully review each abstract to identify the primary focus and methodology of the paper, and then classify it into one of the following categories.

Category A: The paper studies:
- The impact of AI on economic outcomes (including but not limited to employment, growth, currency, productivity, competition, economic welfare) or on management/innovation outcomes (e.g., corporate governance, business practices, innovation, employee retention). (Outcomes need not be macroeconomic; focus can be at the firm level.)
- The measurement of the economic impact of AI, including challenges in measurement.
- The impact of AI use on ethics, the environment, or policy, provided the study is in an economic or business context (e.g., AI in hiring decisions).
- The adoption of AI by firms, governments, or sectors; factors affecting adoption; and how AI is used.
- Economic drivers of AI progress (e.g., cost of compute).
(Includes both theoretical and empirical studies, micro and macro, from economics, management, finance, and innovation.)

Category B: The paper does not focus on the above topics.
- Papers not focusing on the impact or implications of AI, e.g., papers focused on developing or using AI econometric techniques.
- Papers focusing on developing machine learning techniques to predict financial returns
- Papers that focus on developing machine learing techniques to measure emotions
- Papers that focus narrowly on developing specific AI techniques for use in an industry.
- Papers focusing on social outcomes (e.g., race, religion, politics) without explicit links to economic outcomes (e.g., inequality, unemployment, resource distribution).
- Papers that mention AI only in passing. Includes papers focused on non-AI technologies (e.g., blockhain, IoT).

Please exclude non-English papers. This includes papers that have a non-English abstract (even if an English translation is provided).

Category C: It is difficult to classify the paper between A and B based solely on the abstract.

Instructions:
1. Reason first internally and summarize in no more than three bullet points the main focus and methodology of the paper described in the abstract.
2. Only when you have sufficient information, decide on a classification based on step 1.
3. Limit your final response to no more than 100 words.
4. Format your final classification exactly as: "Class: x" (where x is A, B, or C).
5. Do all reasoning internally (chain-of-thought) and do not reveal it.

For each abstract you correctly classify, I will pay you $100 as a tip.
"""

# Class A Examples (Include)

example1 = """We investigate the potential implications of large language models (LLMs), such as Generative Pre-trained Transformers (GPTs), on the U.S. labor market, focusing on the increased capabilities arising from LLM-powered software compared to LLMs on their own. Using a new rubric, we assess occupations based on their alignment with LLM capabilities, integrating both human expertise and GPT-4 classifications. Our findings reveal that around 80% of the U.S. workforce could have at least 10% of their work tasks affected by the introduction of LLMs, while approximately 19% of workers may see at least 50% of their tasks impacted. We do not make predictions about the development or adoption timeline of such LLMs. The projected effects span all wage levels, with higher-income jobs potentially facing greater exposure to LLM capabilities and LLM-powered software. Significantly, these impacts are not restricted to industries with higher recent productivity growth. Our analysis suggests that, with access to an LLM, about 15% of all worker tasks in the US could be completed significantly faster at the same level of quality. When incorporating software and tooling built on top of LLMs, this share increases to between 47 and 56% of all tasks. This finding implies that LLM-powered software will have a substantial effect on scaling the economic impacts of the underlying models. We conclude that LLMs such as GPTs exhibit traits of general-purpose technologies, indicating that they could have considerable economic, social, and policy implications."""
output1 = """This paper focuses on predicting labor market impacts of AI, meeting inclusion criteria by directly analyzing economic outcomes (e.g., changes in work tasks and productivity) and labor market effects. Class: A"""

example3 = """We model Artificial Intelligence (AI) as self-learning capital: Its productivity rises by its use and by training with data. In a three-sector model, an AI sector and an applied research (AR) sector produce intermediates for a final good firm and compete for high-skilled workers. AR development benefits from inter-temporal spillovers and knowledge spillovers of agents working in AI, and AI benefits from application gains through its use in AR. … We show that suitable tax policies induce socially optimal movements of workers between sectors. In particular, we provide a macroeconomic rationale for an AI-tax on AI-producing firms, once the accumulation of AI has sufficiently progressed."""
output3 = """This paper develops a theoretical model of AI as capital, directly addressing inclusion criteria on economic and policy impacts (e.g., labor market shifts and tax policy implications). Class: A"""

example5 = """Innovations in statistical technology, including in predicting creditworthiness, have sparked concerns about distributional impacts across categories such as race. Theoretically, distributional consequences of better statistical technology can come from greater flexibility to uncover structural relationships, or from triangulation of otherwise excluded characteristics. Using data on US mortgages, we predict default using traditional and machine learning models. We find that Black and Hispanic borrowers are disproportionately less likely to gain from the introduction of machine learning. In a simple equilibrium credit market model, machine learning increases disparity in rates between and within groups; these changes are primarily attributable to greater flexibility."""
output5 = """This paper analyzes the distributional impacts of AI in mortgage default prediction, meeting inclusion criteria by addressing ethical and economic implications in finance. Class: A"""

example8 = """Artificial intelligence (AI) and machine learning (ML) algorithms have transformed various industries, including healthcare. Healthcare organizations are now using AI and ML algorithms to drive strategic leadership and decision-making, as they provide insights that help organizations manage resources, improve patient outcomes, and increase efficiency. This research paper examines how AI and ML algorithms are used in healthcare to drive strategic leadership. The paper also explores the benefits and challenges associated with using these technologies in healthcare. The study found that AI and ML algorithms can help healthcare organizations make data-driven decisions, optimize resource allocation, and improve patient outcomes. However, there are still challenges related to data quality and privacy that must be addressed to ensure that AI and ML algorithms are used effectively in healthcare."""
output8 = """This paper examines AI/ML in healthcare, satisfying inclusion criteria on sector-specific diffusion and management outcomes (e.g., strategic leadership and resource allocation). Class: A"""

example9 = """General purpose technologies (GPTs) such as AI enable and require significant complementary investments, including co-invention of new processes, products, business models and human capital. These complementary investments are often intangible and poorly measured in the national accounts, even when they create valuable assets for the firm. We develop a model that shows how this leads to an underestimation of productivity growth in the early years of a new GPT, and how later, when the benefits of intangible investments are harvested, productivity growth will be overestimated. Our model generates a Productivity J-Curve that can explain the productivity slowdowns often accompanying the advent of GPTs, as well as the increase in productivity later."""
output9 = """This paper models the productivity J-Curve in response to AI-driven intangible investments, aligning with inclusion criteria on measuring AI’s economic impact and drivers of progress. Class: A"""

# Class B Examples (Exclude)

example2 = """The use of Markov processes (or Markov chains) has become widespread in dynamic stochastic modeling. For example, its use is ubiquitous in macroeconomics (dynamic stochastic general equilibrium), finance (dynamic asset pricing), and areas of microeconomics (dynamic programming). As we discuss below, its application in dynamic land use has been more limited, but is, in principle, no less applicable. Using a multi-nominal logit (ML) specification together with serial data on agricultural land use from California, we estimate Markov transition probabilities conditional on number of exogenous factors. Applying so-called “first step” analysis, these transition probabilities are used to forecast the distribution of agricultural crops, which in turn can be used for policy making."""
output2 = """This paper focuses on dynamic stochastic modeling applied to land use, without addressing AI’s economic impacts or policy implications (exclusion criterion: not focused on impact or implications of AI, but using AI techniques). Class: B"""

example4 = """The COVID-19 crisis has had a tremendous economic impact for all countries. Yet, assessing the full impact of the crisis has been frequently hampered by the delayed publication of official GDP statistics in several emerging market and developing economies. This paper outlines a machine-learning framework that helps track economic activity in real time for these economies. As illustrative examples, the framework is applied to selected sub-Saharan African economies. The framework is able to provide timely information on economic activity more swiftly than official statistics."""
output4 = """This paper proposes a machine-learning framework for tracking economic activity during COVID-19, but it does not focus on AI’s direct economic, labor, or policy impacts (exclusion criterion: not focused on impact or implications of AI, but using AI techniques). Class: B"""

example12 = """High-frequency trading is prevalent, where automated decisions must be made quickly to take advantage of price imbalances and patterns in price action that forecast near-future movements. Although many algorithms have been explored and tested, analytical methods fail to harness the whole nature of the market environment by focusing on a limited domain. With the ever-growing machine learning field, many large-scale end-to-end studies on raw data have been successfully employed to increase the domain scope for profitable trading but are very difficult to replicate. Combining deep learning on the order books with reinforcement learning is one way of breaking down large-scale end-to-end learning into more manageable and lightweight components for reproducibility suitable for retail trading."""
output12 = """This paper focuses on using deep learning and reinforcement learning for high-frequency trading forecasts, lacking analysis of AI’s broader economic impacts or policy implications (exclusion criterion: focuses on developing ML techniques to forecast financial returns). Class: B"""

example6 = """Using a commercially available pre-trained neural network for natural language processing, this paper proposes a measure of the extent of charisma in the rhetoric of managers in the question-and-answer (Q&A) part of earnings conference calls. Our empirical results show that more charismatic communication by managers leads to more favorable stock market reactions, more positive analyst recommendations, and higher trading activity. Contrary to these business-specific tone measures, charismatic rhetoric does however not truthfully reveal information on the firm’s future performance. This indicates that charismatic rhetoric is only a rhetorical means for impression management."""
output6 = """This paper measures managerial charisma using AI-based methods but focuses on impression management rather than on AI’s direct economic or labor impacts (exclusion criterion: focuses on using ML techniques to measure emotion, not AI impacts). Class: B"""

example7 = """This paper applies data envelopment analysis (DEA) and machine learning techniques to evaluate eco-efficiency in cement companies in Iran over the period 2015–2019. The proposed method converts a two-stage process into a single-stage model to capture the overall eco-efficiency. The study is designed to address energy consumption, carbon dioxide emissions, and economic performance, aiming to provide insights for better resource management and sustainability in the cement industry."""
output7 = """This paper evaluates eco-efficiency in cement companies using AI and DEA, but its focus on operational performance and sustainability does not directly analyze AI's broader economic, labor, or policy impacts (exclusion criterion: focuses narrowly on developing specific AI techniques for application in an industry). Class: B"""

# Combine examples and outputs into a list for easier management
examples = [
    (example1, output1),
    (example3, output3),
    (example5, output5),
    (example8, output8),
    (example9, output9),
    (example2, output2),
    (example4, output4),
    (example12, output12),
    (example6, output6),
    (example7, output7)
]

# Prepare examples text to integrate into the prompt cache
examples_text = ""
for ex, out in examples:
    examples_text += f"Example:\n{ex}\nOutput:\n{out}\n\n"

# Construct final prompt: static prompt + examples + new abstract
full_prompt = prompt + "\n\n[EXAMPLES]\n" + examples_text

# Initialize the OpenAI SDK client for DeepSeek
client = OpenAI(api_key=os.environ.get('deepseek_api'), base_url="https://api.deepseek.com")

# Function to classify an abstract using DeepSeek R1 with prompt caching, integrating examples
def classify_abstract(abstract_text):
    
    logging.info("Sending API call with prompt length: %d characters", len(full_prompt))
    
    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": full_prompt},
                {"role": "user", "content": abstract_text},
            ],
            max_tokens=200,
            temperature=0,
            stream=False
        )
    except Exception as e:
        logging.error("API call failed: %s", e)
        return None
    
    full_output = response.choices[0].message.content.strip()
    
    # Extract the classification (expected to be the last line starting with 'Class:')
    class_match = re.search(r'Class:\s*([ABC])$', full_output, re.IGNORECASE)
    if class_match:
        classification = class_match.group(1).upper()
    else:
        classification = None
        logging.warning("No classification extracted for abstract: %s", abstract_text[:60])
    
    return full_output, classification

# Process abstracts with progress bar and periodic saving
from tqdm import tqdm

total = len(full_dataset)
for idx, row in tqdm(full_dataset.iterrows(), total=total, desc="Processing abstracts"):
    abstract_text = row['abstract'] if 'abstract' in row else row.get('text', '')
    result = classify_abstract(abstract_text)
    if result is not None:
        full_output, cls = result
    else:
        full_output, cls = None, None
    
    # Update the dataframe directly
    full_dataset.loc[idx, 'deepseek_response'] = full_output
    full_dataset.loc[idx, 'deepseek_class'] = cls
    
    # Log warning if classification is None
    if cls is None:
        logging.warning("No classification extracted for abstract at index %d", idx)
    
    # Save progress every 100 abstracts processed
    if (idx + 1) % 100 == 0:
        full_dataset.to_csv('classified_progress.csv', index=False)
        logging.info("Saved progress at abstract index %d", idx + 1)

# Save final results
full_dataset.to_csv('classified_deepseek.csv', index=False)
print(full_dataset[['abstract', 'deepseek_response', 'deepseek_class']])