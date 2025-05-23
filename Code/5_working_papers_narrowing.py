# %%
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import time
import logging
from tqdm import tqdm
import re
import os

# %%
username = os.getlogin()
if username =="lucyjhampton":
    local_path = r"/Users/lucyjhampton/Library/CloudStorage/GoogleDrive-lucyjhampton5@gmail.com/My Drive/AI Landscape Collaboration"
    os.chdir(local_path)
if username == "lhampton":
    local_path = r"/Users/lhampton/Documents/Local_Projects/ailandscapeproject"
    os.chdir(local_path)
    data_path = r"/Users/lhampton/Library/CloudStorage/GoogleDrive-lucyjhampton5@gmail.com/.shortcut-targets-by-id/1EznD-uaTblXtaIVx04-toZMQh7cQf9hA/AI Landscape Large Files/Data/original/working_papers"

# %%

NBER_no_duplicates = pd.read_csv(f'{data_path}/nber/NBER_no_duplicates.csv')
NBER_no_duplicates['Source'] = "NBER"

SSRN_no_duplicates = pd.read_csv(f'{data_path}/ssrn/SSRN_no_duplicates.csv')
SSRN_no_duplicates['Source'] = "SSRN"

working_papers = pd.concat([NBER_no_duplicates, SSRN_no_duplicates], ignore_index=True)

working_papers = working_papers.drop(columns = {"type", "journal", "M1", "M2", "AB", "endref", "cond", "nber_is_published", "nber_in_ssrn", "keywords", "searchterm", "source", "ssrn_is_published", "tow", "vol", "access", "note", "url", "date"})
working_papers = working_papers.reset_index(level=None, drop=True)

working_papers['ID'] = working_papers.index

# working_papers.to_csv(f'{local_path}/Data/working_papers.csv')
# working_papers
# %%

example1 = """Large scale projects increasingly operate in complicated settings whilst drawing on an array of complex data-points, which require precise analysis for accurate control and interventions to mitigate possible project failure. Coupled with a growing tendency to rely on new information systems and processes in change projects, 90% of megaprojects globally fail to achieve their planned objectives. Renewed interest in the concept of Artificial Intelligence (AI) against a backdrop of disruptive technological innovations, seeks to enhance project managers‚Äô cognitive capacity through the project lifecycle and enhance project excellence. However, despite growing interest there remains limited empirical insights on project managers‚Äô ability to leverage AI for cognitive load enhancement in complex settings. As such this research adopts an exploratory sequential linear mixed methods approach to address unresolved empirical issues on transient adaptations of AI in complex projects, and the impact on cognitive load enhancement. Initial thematic findings from semi-structured interviews with domain experts, suggest that in order to leverage AI technologies and processes for sustainable cognitive load enhancement with complex data over time, project managers require improved knowledge and access to relevant technologies that mediate data processes in complex projects, but equally reflect application across different project phases. These initial findings support further hypothesis testing through a larger quantitative study incorporating structural equation modelling to examine the relationship between artificial intelligence and project managers‚Äô cognitive load with project data in complex contexts."""
output1 = """The paper focuses on the adoption of AI in megaprojects, an economic outcome. It uses a linear mixed method approach to identify the impacts of AI adoption on cognitive load enhancement in complex settings. Class: A"""

example2 = """Using Italian data from Twitter, we employ textual data and machine learning techniques to build new real-time measures of consumers' inflation expectations. First, we select some relevant keywords to identify tweets related to prices and expectations thereof. Second, we build a set of daily measures of inflation expectations on the selected tweets combining the Latent Dirichlet Allocation (LDA) with a dictionary-based approach, using manually labelled bi-grams and tri-grams. Finally, we show that the Twitter-based indicators are highly correlated with both monthly survey-based and daily market-based inflation expectations.Our new indicators provide additional information beyond the market-based expectations, the professional forecasts, and the realized inflation, and anticipate consumers' expectations proving to be a good real-time proxy. Results suggest that Twitter can be a new timely source to elicit beliefs."""
output2 = """This paper focuses on using building a measure of inflation expectations using Twitter data and machine-learning techniques. It does not focus on the economic impact of AI itself; AI is just used as a method, and so is not relevant. Class: B"""

example3 = """In this article, a class of mean-variance portfolio selection problems with constant risk aversion is investigated by means of closed-loop equilibrium strategies. Thanks to the non-Markovian setting, two delicate kinds of equilibrium strategies are introduced and both of them obviously reduce to the existing counterpart in the Markovian case. To explicitly represent the equilibrium strategy, a class of backward stochastic Riccati system is introduced, and its solvability is carefully discussed for the first time in the literature. Different from the current literature, the spectacular role of random interest rates in the model is firstly indicated by several interesting phenomena, and the new deeper relations between closed-loop, open-loop equilibrium strategies are shown as well. Finally, numerical analysis via deep learning method is present to illustrate the novel theoretical findings."""
output3 = """ This paper focuses on solving a class of mean-variance portfolio selection problems using closed-loop equilibrium strategies and deep learning techniques. The paper does not focus on the impact of AI, it is only used as a method here, and so is not relevant. Class: B"""

example4 = """Do non-traditional digital trace data and traditional survey data yield similar estimates of the impact of a cash transfer program? In a randomized controlled trial of Togo‚Äôs COVID-19 Novissi program, endline survey data indicate positive treatment effects on beneficiary food security, mental health, and self-perceived economic status. However, impact estimates based on mobile phone data ‚Äì processed with machine learning to predict beneficiary welfare ‚Äì do not yield similar results, even though related data and methods do accurately predict wealth and consumption in prior cross-sectional analysis in Togo. This limitation likely arises from the underlying difficulty of using mobile phone data to predict short-term changes in wellbeing within a rural population with fairly homogeneous baseline levels of poverty. We discuss the implications of these results for using new digital data sources in impact evaluation."""
output4 = """This paper focuses on evaluating the use of digital trace data as an alternative to traditional survey data as a method of investigating the impact of cash transfer programs. It does not focus on the impact of AI itself; AI is just used to process the mobile phone data, and so it is not relevant. Class: B"""

example5 = """Our study explored the rise of public companies competing to launch large language models (LLMs) in the Chinese stock market after ChatGPTs' success. We analyzed 25 companies listed on the Chinese Stock Exchange and discovered that the cumulative abnormal return (CAR) was high up to 3% before LLMs' release, indicating a positive view from insiders. However, CAR dropped to around 1.5% after their release. Early LLM releases had better market reactions, especially those focused on customer service, design, and education. Conversely, LLMs dedicated to IT and civil service received negative feedback."""
output5 = """This paper focuses on estimating the cumulative abnormal return following adoption of LLMs in the Chinese stock market, and how this differs by the time and area of adoption. The paper focuses on the impact of AI on companies' financial returns, which is relevant. Class: A"""

example6 = """In order to create a well-functioning internal market for Artificial Intelligence (AI)-systems, the European Commission recently proposed the Artificial Intelligence Act. However, this legislative proposal pays limited attention to the health-specific risks the use of AI poses to patients‚Äô rights. This article outlines that fundamental rights impacts associated with AI such as discrimination, diminished privacy and opaque decision-making are exacerbated in the context of health and may threaten the protection of foundational values and core patients‚Äô rights. However, while the EU is facilitating and promoting the use and availability of AI in the health sector in Europe via the Digital Single Market, it is unclear whether it can provide the concomitant patients‚Äô rights protection. This article theorises the Europeanisation of health AI by exploring legal challenges through a patients‚Äô rights lens in order to determine if the European regulatory approach for AI provides for sufficient protection to patients‚Äô rights."""
output6 = """The paper focuses on the ethical impacts of the adoption of AI in health, focusing on impacts on fundamental rights and policy implications. This is relevant to the policy and ethical implications of AI arising through economic activity (adoption in the healthcare sector) so is relevant. Class: A"""

example7 = """The paper proposes an explainable AI model that can be used in credit risk management and, in particular, in measuring the risks that arise when credit is borrowed employing credit scoring platforms. The model applies similarity networks to Shapley values, so that AI predictions are grouped according to the similarity in the underlying explanatory variables.The empirical analysis of 15,000 small and medium companies asking for credit reveals that both risky and not risky borrowers can be grouped according to a set of similar financial characteristics, which can be employed to explain and understand their credit score and, therefore, to predict their future behaviour."""
output7 = """The paper focuses on developing an explainable AI model for use in credit risk management. As the focus of the paper is on developing a model rather than evaluating the impact of AI, it is beyond our scope. Class: B"""

example8 = """This paper evidences the explanatory power of managers‚Äô uncertainty for cross-sectional stock returns. I introduce a novel measure of the degree of managers‚Äô uncertain beliefs about future states: manager uncertainty (MU), defined as the count of the word ‚Äúuncertainty‚Äù over the sum of the count of the word ‚Äúuncertainty‚Äù and the count of the word ‚Äúrisk‚Äù in filings and conference calls. I find that manager‚Äôs level of uncertainty reveals valuation information about real options and thereby has significantly negative explanatory power for cross-sectional stock returns. Beyond existing market-based uncertainty measures, the manager uncertainty measure has incremental pricing power by capturing information frictions between managers‚Äô reported uncertainty and investors‚Äô perception of uncertainty. Moreover, a short-long portfolio sorted by manager uncertainty has a significantly positive premium and cannot be spanned by existing factor models. An application on COVID-19 uncertainty shows consistent results."""
output8 = """This paper investigates the relationship between manager uncertainty, measured using filings and conference calls, and stock returns. The paper is not focused on the impact of AI, therefore it is not relevant. Class: B"""

example9 = """Drawing insights from the field of innovation economics, we discuss the likely competitive environment shaping generative AI advances. Central to our analysis are the concepts of appropriability‚Äîwhether firms in the industry are able to control the knowledge generated by their innovations‚Äîand complementary assets‚Äîwhether effective entry requires access to specialized infrastructure and capabilities to which incumbent firms can ration access. While the rapid improvements in AI foundation models promise transformative impacts across broad sectors of the economy, we argue that tight control over complementary assets will likely result in a concentrated market structure, as in past episodes of technological upheaval. We suggest the likely paths through which incumbent firms may restrict entry, confining newcomers to subordinate roles and stifling broad sectoral innovation. We conclude with speculations regarding how this oligopolistic future might be averted. Policy interventions aimed at fractionalizing or facilitating shared access to complementary assets might help preserve competition and incentives for extending the generative AI frontier. Ironically, the best hopes for a vibrant open source AI ecosystem might rest on the presence of a ‚Äúrogue‚Äù technology giant, who might choose openness and engagement with smaller firms as a strategic weapon wielded against other incumbents."""
output9 = """This paper investigates the likely competitive environment shaping advances in generative AI, including factors affecting the likely market structure. As it focuses on the drivers of AI development as well as the policy implications of AI, it is relevant. Class: A"""

example10 = """Machine learning algorithms can find predictive signals that researchers fail to notice; yet they are notoriously hard-to-interpret. How can we extract theoretical insights from these black boxes? History provides a clue. Facing a similar problem ‚Äì how to extract theoretical insights from their intuitions ‚Äì researchers often turned to ‚Äúanomalies:‚Äù constructed examples that highlight flaws in an existing theory and spur the development of new ones. Canonical examples include the Allais paradox and the Kahneman-Tversky choice experiments for expected utility theory. We suggest anomalies can extract theoretical insights from black box predictive algorithms. We develop procedures to automatically generate anomalies for an existing theory when given a predictive algorithm. We cast anomaly generation as an adversarial game between a theory and a falsifier, the solutions to which are anomalies: instances where the black box algorithm predicts - were we to collect data - we would likely observe violations of the theory. As an illustration, we generate anomalies for expected utility theory using a large, publicly available dataset on real lottery choices. Based on an estimated neural network that predicts lottery choices, our procedures recover known anomalies and discover new ones for expected utility theory. In incentivized experiments, subjects violate expected utility theory on these algorithmically generated anomalies; moreover, the violation rates are similar to observed rates for the Allais paradox and Common ratio effect."""
output10 = """This paper focuses on how researchers can extract theoretical insights from hard-to-interpret machine learning algorithms. As this paper is primarily concerned with AI in a research methodology rather than AI's economic impacts, it is not relevant. Class: B"""

example11 = """Timely and accurate measurement of AI use by firms is both challenging and crucial for understanding the impacts of AI on the U.S. economy. We provide new, real-time estimates of current and expected future use of AI for business purposes based on the Business Trends and Outlook Survey for September 2023 to February 2024. During this period, bi-weekly estimates of AI use rate rose from 3.7% to 5.4%, with an expected rate of about 6.6% by early Fall 2024. The fraction of workers at businesses that use AI is higher, especially for large businesses and in the Information sector. AI use is higher in large firms but the relationship between AI use and firm size is non-monotonic. In contrast, AI use is higher in young firms although, on an employment-weighted basis, is U-shaped in firm age. Common uses of AI include marketing automation, virtual agents, and data/text analytics. AI users often utilize AI to substitute for worker tasks and equipment/software, but few report reductions in employment due to AI use. Many firms undergo organizational changes to accommodate AI, particularly by training staff, developing new workflows, and purchasing cloud services/storage. AI users also exhibit better overall performance and higher incidence of employment expansion compared to other businesses. The most common reason for non-adoption is the inapplicability of AI to the business."""
output11 = """This paper focuses on how to measure AI use by firms in a timely and accurate way. As it focuses on the measurement of AI adoption, one of the specified outcomes, it is relevant. Class: A"""

example12 = """We argue that comprehensive out-of-sample (OOS) evaluation using statistical decision theory (SDT) should replace the current practice of K-fold and Common Task Framework validation in machine learning (ML) research. SDT provides a formal framework for performing comprehensive OOS evaluation across all possible (1) training samples, (2) populations that may generate training data, and (3) populations of prediction interest. Regarding feature (3), we emphasize that SDT requires the practitioner to directly confront the possibility that the future may not look like the past and to account for a possible need to extrapolate from one population to another when building a predictive algorithm. SDT is simple in abstraction, but it is often computationally demanding to implement. We discuss progress in tractable implementation of SDT when prediction accuracy is measured by mean square error or by misclassification rate. We summarize research studying settings in which the training data will be generated from a subpopulation of the population of prediction interest. We consider conditional prediction with alternative restrictions on the state space of possible populations that may generate training data. We present an illustrative application of the methodology to the problem of predicting patient illness to inform clinical decision making. We conclude by calling on ML researchers to join with econometricians and statisticians in expanding the domain within which implementation of SDT is tractable."""
output12 = """This paper aims to justify using comprehensive out-of sample evaluation using statistical decision theory as a replacement for k-fold and Common Task Framework validation in machine learning and discusses a way of making it less computationally demanding. As the focus is on the methodology of machine learning rather than its economic impacts, it is not relevant. Class: B"""

example13 = """This paper studies how and why households adjust their spending, saving, and borrowing in response to transitory income shocks. We leverage new large-scale survey data to first quantitatively assess households‚Äô intertemporal marginal propensities to consume (MPCs) and deleverage (MPDs) (the ‚Äúhow‚Äù), and second to dive into the motivations and decision-making processes across households (the ‚Äúwhy‚Äù). The combination of the quantitative estimation of household response dynamics with a qualitative exploration of the mental models employed during financial decisions provides a more complete view of household behavior. Our findings are as follows: First, we validate the reliability of surveys in predicting actual economic behaviors using a new approach called cross-validation, which compares the responses to hypothetical financial scenarios with observed actions from past studies. Participants‚Äô predicted reactions closely align with real-life behaviors. Second, we show that MPCs are significantly higher immediately following an income shock and diminish over time, with cumulative MPCs over a year showing significant variability. However, MPDs play a critical role in household financial adjustments and display significantly more cross-sectional heterogeneity. Neither is easily explained by socioeconomic or financial characteristics alone, and the explanatory power is improved by adding psychological factors, past experiences, and expectations. Third, using specifically-designed survey questions, we find that there is a broad range of motivations behind households‚Äô financial decisions and identify four household types using machine learning: Strongly Constrained, Precautionary, Quasi-Smoothers, and Spenders. Similar financial actions stem from diverse reasons, challenging the predictability of financial behavior solely based on socioeconomic and financial characteristics. Finally, we use our findings to address some puzzles in household finance."""
output13 = """This paper focuses on studying the household saving, spending and borrowing response to transitory income shocks. The authors use large-scale survey data and machine learning techniques in their analysis. Because the paper only uses machine learning as a method and does not focus on the economic impact of ML itself, it is not relevant. Class: B"""

example14 = """We consider an environment in which there is substantial uncertainty about the potential adverse external effects of AI algorithms. We find that subjecting algorithm implementation to regulatory approval or mandating testing is insufficient to implement the social optimum. When testing costs are low, a combination of mandatory testing for external effects and making developers liable for the adverse external effects of their algorithms comes close to implementing the social optimum even when developers have limited liability."""
output14 = """This paper considers the optimal policy response to the adverse effects of AI algorithms, including regulatory approval, mandatory testing and liability for developers. As it is focused on the policy implications of AI adoption, it is relevant. Class: B"""
# %%
prompt = "You are an academic expert in economics, finance, management and innovation studies. You are provided with abstracts from academic papers. Your goal is to carefully review each abstract to identify the primary focus and methodology of the paper. Based on this analysis, classify each paper into one of the following categories. \n\nCategory A: The paper studies: \n- The impact of AI on economic (including but not limited to employment, growth, currency, productivity, competition, economic welfare) or management/innovation (such as corporate governance, business practices, innovation and employee retention) outcomes. Note that these outcomes do not have to be macroeconomic, e.g., the paper can focus on a specific firm's productivity or financial decisions. \n- The measurement of the economic impact of AI, including measurement challenges. \n- AI’s implications for ethics, the environment, or policy; provided they focus on how these issues relate to economic activity, e.g., the relationship between AI, racial inequality and income inequality. \n- The adoption of AI by firms, the government, or sectors; what affects adoption decisions; and how AI is used. \n- Economic drivers of AI progress, such as cost of compute. \nProvided they meet the earlier requirements, include both theoretical and empirical, micro and macro studies from the fields of economics, management, finance, and innovation (including sector-specific studies). \n\nCategory B: The paper does not focus on the above topics. Category B includes papers that focus on developing machine learning techniques to predict financial returns. Category B also includes papers that focus on social outcomes such as race, religion and politics where the paper does not make explicit links to economic outcomes, such as economic inequality, unemployment or resource distribution. Also include papers that are not about the impact or implications of AI, e.g., those that focus on developing or using AI econometric techniques other than to analyse the impact of implications of AI. \n\nCategory C: It is difficult to classify the paper between A and B based on the abstract. \n\nInstructions: 1. Reason first, and summarise in no more than three bullet points, the main focus and methodology of the paper outlined in the abstract. 2. Only when you have enough information, decide on a classification based on 1. Do not use more than 100 words in your response. When giving your final classification, format it as 'Class: x' where x is your suggested classification. For each abstract you correctly classify, I will pay you $10 as a tip."
print(prompt)

#%%
## HOW MANY TOKENS?

import tiktoken

# define token count function
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# apply to papers
working_papers['no_tokens'] = working_papers['abstract'].apply(lambda x: num_tokens_from_string(str(x), "gpt-4o"))

## estimate cost
Total = working_papers['no_tokens'].sum()*0.12
print(f"Number of paper tokens: {Total}")
examples = ''.join([globals()[f'example{i}'] for i in range(1, 15)])
outputs = ''.join([globals()[f'output{i}'] for i in range(1, 15)])
prompt_tokens = num_tokens_from_string(prompt, "gpt-4o") + num_tokens_from_string(examples, "gpt-4o") + num_tokens_from_string(examples, "gpt-4o")
prompt_tokens = prompt_tokens/5
print(f"Number of prompt tokens: {prompt_tokens}")
est_output_tokens = 250
cost = (2.5*prompt_tokens*6000/10**6) + (2.5*Total/10**6) + (10*est_output_tokens*6000/10**6)
print(f"Estimated total cost in $: {cost}")

#%%
## PROMPT GPT-4o MINI

from openai import OpenAI
from dotenv import load_dotenv
import time
import logging
from tqdm import tqdm
import re

# set up logging
logging.basicConfig(level = logging.INFO)

# Load progress from the last run (if any)
try:
    classed_wps = pd.read_csv(f"{local_path}/Temp/wp_classification/classed_wps.csv")
    processed_IDs = set(classed_wps['ID'])
    data_dict = classed_wps.to_dict('list')
except FileNotFoundError:
    classed_wps = pd.DataFrame(columns=['abstract', 'ID', 'doi', 'GPT_response', 'GPT_class', 'Input_tokens', 'Output_tokens', 'reason_stop', 'system_fingerprint'])
    processed_IDs = set()
    data_dict = {
        'abstract': [],
        'ID': [],
        'doi': [],
        'GPT_response': [],
        'GPT_class': [],
        'Input_tokens': [],
        'Output_tokens': [],
        'reason_stop': [],
        'system_fingerprint': []
    }

# set up api
if username == "lhampton":
    env_location = '/Users/lhampton/Documents/env files/AI Landscape/.env'
else:
    env_location = '/Users/lucyjhampton/Documents/env files/AI Landscape/.env'
load_dotenv(env_location)
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()
client.api_key = api_key

# loop through abstracts
for index, row in tqdm(working_papers.iterrows(), desc = 'Processing abstracts'):
    if row['ID'] in processed_IDs:
        continue
        
    else:
        if pd.isnull(row['abstract']):
            data_dict['abstract'].append(row['abstract'])
            data_dict['ID'].append(row['ID'])
            data_dict['doi'].append(row['doi'])
            data_dict['GPT_response'].append('')
            data_dict['GPT_class'].append('')
            data_dict['Input_tokens'].append('')
            data_dict['Output_tokens'].append('')
            data_dict['reason_stop'].append('')
            data_dict['system_fingerprint'].append('')
        else:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=300,
                temperature=0,
                seed=2345,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": example1},
                    {"role": "assistant", "content": output1},
                    {"role": "user", "content": example2},
                    {"role": "assistant", "content": output2},
                    {"role": "user", "content": example3}, 
                    {"role": "assistant", "content": output3},
                    {"role": "user", "content": example4},
                    {"role": "assistant", "content": output4},
                    {"role": "user", "content": example5},
                    {"role": "assistant", "content": output5},
                    {"role": "user", "content": example6},
                    {"role": "assistant", "content": output6},
                    {"role": "user", "content": example7},
                    {"role": "assistant", "content": output7},
                    {"role": "user", "content": example8},
                    {"role": "assistant", "content": output8},
                    {"role": "user", "content": example9},
                    {"role": "assistant", "content": output9},
                    {"role": "user", "content": example10},
                    {"role": "assistant", "content": output10},
                    {"role": "user", "content": example11},
                    {"role": "assistant", "content": output11},
                    {"role": "user", "content": example12},
                    {"role": "assistant", "content": output12},
                    {"role": "user", "content": example13},
                    {"role": "assistant", "content": output13},
                    {"role": "user", "content": example14},
                    {"role": "assistant", "content": output14},
                    {"role": "user", "content": row['abstract']}
                ]
            )
        
            # store data in df
            data_dict['abstract'].append(row['abstract'])
            data_dict['ID'].append(row['ID'])
            data_dict['doi'].append(row['doi'])
            message = response.choices[0].message.content
            data_dict['GPT_response'].append(message)
            try:
                data_dict['GPT_class'].append(re.search(r'Class:\s*([A-Z])', message).group(1))
            except (ValueError, AttributeError) as e:
                data_dict['GPT_class'].append('')
            data_dict['Input_tokens'].append(response.usage.prompt_tokens)
            data_dict['Output_tokens'].append(response.usage.completion_tokens)
            data_dict['reason_stop'].append(response.choices[0].finish_reason)
            data_dict['system_fingerprint'].append(response.system_fingerprint)
    
        # create df out of dict
        classed_wps = pd.DataFrame(data_dict)
        
        # save progress, add url to processed url list
        if (index + 1) % 10 == 0:
            try:
                if len(classed_wps) - len(pd.read_csv(f"{local_path}/Temp/wp_classification/classed_wps.csv")) == 10:
                    new_papers = classed_wps.tail(10)
                    new_papers.to_csv(f"{local_path}/Temp/wp_classification/classed_wps.csv", index=False, mode='a', header=False)
                    logging.info("Saved progress at iteration {}".format(index + 1))
                else:
                    logging.info("Warning: trying to overwrite with blank at iteration {}".format(index + 1))
                    break
            except FileNotFoundError:
                classed_wps.to_csv(f"{local_path}/Temp/wp_classification/classed_wps.csv", index=False, header = True)
        
        # add to list of processed urls
        processed_IDs.add(row['ID'])

    # # sleep to observe rate limit (500 requests per minute)
    # time.sleep(0.01)

# Display the DataFrame
classed_wps = pd.DataFrame(data_dict)
classed_wps.to_csv(f"{local_path}/Temp/wp_classification/classed_wps.csv", index=False)
classed_wps
# %%

for_john_A = classed_wps[classed_wps['GPT_class'] == "A"].sample(100)
for_john_B = classed_wps[classed_wps['GPT_class'] == "B"].sample(100)
for_john_A.to_csv('for_john_A.csv')
for_john_B.to_csv('for_john_B.csv')

for_lucy_A = classed_wps[classed_wps['GPT_class'] == "A"].sample(100)
for_lucy_B = classed_wps[classed_wps['GPT_class'] == "B"].sample(100)
for_lucy_A.to_csv('for_lucy_A.csv')
for_lucy_B.to_csv('for_lucy_B.csv')

# %%

## Narrow further
classed_wps = pd.read_csv(f"{local_path}/Temp/wp_classification/classed_wps.csv")
As = classed_wps[classed_wps['GPT_class'] == "A"]
As = As.reset_index(level=None, drop=True)

prompt2 = """You are an academic expert in economics, finance, management, and innovation studies. You are provided with abstracts from academic papers. Your goal is to carefully review each abstract to identify the primary focus and methodology of the paper. Based on this analysis, classify each paper into one of the following categories:

### Categories:
**Category A**: The paper studies:
- The *impact* of AI on the economy (e.g., employment, growth, productivity, competition) or management/innovation outcomes (e.g., corporate governance, business practices, innovation, employee retention). Note: The outcomes can be micro-level (e.g., specific firm productivity) or macro-level.
- The *measurement* of the impact of AI on the economy, including challenges in measurement. (Do not confuse this with using AI itself to aid economic measurement.)
- AI's implications for *ethics*, the *environment*, or *policy*—provided the paper explicitly links these issues to economic activity (e.g., AI and income inequality).
- The *adoption* of AI by firms, governments, or sectors; and factors influencing adoption decisions. 
- The *economic drivers* of AI progress (e.g., cost of compute).

**Category B**: The paper does not focus on the above topics. Examples include:
- Papers focused on *developing* machine learning techniques to predict financial returns.
- Papers focused on *developing* or *using* AI econometric methods where the study does not examine the impacts of AI itself. For instance, do not include papers that focus on the technical aspects of adoption, e.g., the specific algorithms used.
- Papers discussing *social outcomes* (e.g., race, religion, politics) without explicit links to economic outcomes (e.g., unemployment, inequality).
- Studies that are not about AI, but other tech.

**Category C**: It is difficult to classify the paper between A and B based on the abstract.

---

### Instructions:
1. Carefully identify and summarise the main focus and methodology of the paper in no more than three bullet points. Clearly distinguish *what the paper studies* (its focus) from *how it studies it* (its methodology).
2. Apply the following decision process:
   - **Step 1**: Does the paper *explicitly focus* on the *impact* or *implications* of AI for the economy, management, or innovation? If yes, consider Category A. If no, proceed to Step 2.
   - **Step 2**: Does the paper focus on topics unrelated to the impact of AI on the economy, innovation or management, such as social outcomes without economic links, econometrics, or the development of AI methods? If yes, assign Category B.
   - **Step 3**: If the abstract lacks sufficient clarity, assign Category C.
3. Highlight disqualifying factors where relevant (e.g., "Focuses on ML econometrics, not impacts").
4. Conclude your analysis with the classification formatted as `Class: X`, where X is A, B, or C. Do not exceed 100 words.
"""
#%%
## PROMPT GPT-4o to narrow further

# set up logging
logging.basicConfig(level = logging.INFO)

# Load progress from the last run (if any)
try:
    classed_wps_2 = pd.read_csv(f"{local_path}/Temp/wp_classification/classed_wps_2.csv")
    processed_IDs_2 = set(classed_wps_2['ID'])
    data_dict_2 = classed_wps_2.to_dict('list')
except FileNotFoundError:
    classed_wps_2 = pd.DataFrame(columns=['abstract', 'ID', 'doi', 'GPT_response', 'GPT_class', 'Input_tokens', 'Output_tokens', 'reason_stop', 'system_fingerprint'])
    processed_IDs_2 = set()
    data_dict_2 = {
        'abstract': [],
        'ID': [],
        'doi': [],
        'GPT_response': [],
        'GPT_class': [],
        'Input_tokens': [],
        'Output_tokens': [],
        'reason_stop': [],
        'system_fingerprint': []
    }

# set up api
if username == "lhampton":
    env_location = '/Users/lhampton/Documents/env files/AI Landscape/.env'
else:
    env_location = '/Users/lucyjhampton/Documents/env files/AI Landscape/.env'
load_dotenv(env_location)
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()
client.api_key = api_key

# loop through abstracts
for index, row in tqdm(As.iterrows(), desc = 'Processing abstracts'):
    if row['ID'] in processed_IDs_2:
        continue
        
    else:
        if pd.isnull(row['abstract']):
            data_dict_2['abstract'].append(row['abstract'])
            data_dict_2['ID'].append(row['ID'])
            data_dict_2['doi'].append(row['doi'])
            data_dict_2['GPT_response'].append('')
            data_dict_2['GPT_class'].append('')
            data_dict_2['Input_tokens'].append('')
            data_dict_2['Output_tokens'].append('')
            data_dict_2['reason_stop'].append('')
            data_dict_2['system_fingerprint'].append('')
        else:
            response = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=300,
                temperature=0,
                seed=2345,
                messages=[
                    {"role": "system", "content": prompt2},
                    {"role": "user", "content": example1},
                    {"role": "assistant", "content": output1},
                    {"role": "user", "content": example2},
                    {"role": "assistant", "content": output2},
                    {"role": "user", "content": example3}, 
                    {"role": "assistant", "content": output3},
                    {"role": "user", "content": example4},
                    {"role": "assistant", "content": output4},
                    {"role": "user", "content": example5},
                    {"role": "assistant", "content": output5},
                    {"role": "user", "content": example6},
                    {"role": "assistant", "content": output6},
                    {"role": "user", "content": example7},
                    {"role": "assistant", "content": output7},
                    {"role": "user", "content": example8},
                    {"role": "assistant", "content": output8},
                    {"role": "user", "content": example9},
                    {"role": "assistant", "content": output9},
                    {"role": "user", "content": example10},
                    {"role": "assistant", "content": output10},
                    {"role": "user", "content": example11},
                    {"role": "assistant", "content": output11},
                    {"role": "user", "content": example12},
                    {"role": "assistant", "content": output12},
                    {"role": "user", "content": example13},
                    {"role": "assistant", "content": output13},
                    {"role": "user", "content": example14},
                    {"role": "assistant", "content": output14},
                    {"role": "user", "content": row['abstract']}
                ]
            )
        
            # store data in df
            data_dict_2['abstract'].append(row['abstract'])
            data_dict_2['ID'].append(row['ID'])
            data_dict_2['doi'].append(row['doi'])
            message = response.choices[0].message.content
            data_dict_2['GPT_response'].append(message)
            try:
                data_dict_2['GPT_class'].append(re.search(r'Class:\s*([A-Z])', message).group(1))
            except (ValueError, AttributeError) as e:
                data_dict_2['GPT_class'].append('')
            data_dict_2['Input_tokens'].append(response.usage.prompt_tokens)
            data_dict_2['Output_tokens'].append(response.usage.completion_tokens)
            data_dict_2['reason_stop'].append(response.choices[0].finish_reason)
            data_dict_2['system_fingerprint'].append(response.system_fingerprint)
    
        # create df out of dict
        classed_wps_2 = pd.DataFrame(data_dict_2)
        
        # save progress, add url to processed url list
        if (index + 1) % 10 == 0:
            try:
                if len(classed_wps_2) - len(pd.read_csv(f"{local_path}/Temp/wp_classification/classed_wps_2.csv")) == 10:
                    new_papers_2 = classed_wps_2.tail(10)
                    new_papers_2.to_csv(f"{local_path}/Temp/wp_classification/classed_wps_2.csv", index=False, mode='a', header=False)
                    logging.info("Saved progress at iteration {}".format(index + 1))
                else:
                    logging.info("Warning: trying to overwrite with blank at iteration {}".format(index + 1))
                    break
            except FileNotFoundError:
                classed_wps_2.to_csv(f"{local_path}/Temp/wp_classification/classed_wps_2.csv", index=False, header = True)
        
        # add to list of processed urls
        processed_IDs_2.add(row['ID'])

    # # sleep to observe rate limit (500 requests per minute)
    # time.sleep(0.01)

# Display the DataFrame
classed_wps_2 = pd.DataFrame(data_dict_2)
classed_wps_2.to_csv(f"{local_path}/Temp/wp_classification/classed_wps_2.csv", index=False)
classed_wps_2

# %%
import pandas

classed_wps_2 = pd.read_csv(f'{local_path}/Temp/wp_classification/classed_wps_2.csv')

for_john_A_2 = classed_wps_2[classed_wps_2['GPT_class'] == "A"].sample(100)
for_john_B_2 = classed_wps_2[classed_wps_2['GPT_class'] == "B"].sample(100)
for_john_A_2.to_csv('for_john_A_2.csv')
for_john_B_2.to_csv('for_john_B_2.csv')

for_lucy_A_2 = classed_wps_2[classed_wps_2['GPT_class'] == "A"].sample(100)
for_lucy_B_2 = classed_wps_2[classed_wps_2['GPT_class'] == "B"].sample(100)
for_lucy_A_2.to_csv('for_lucy_A_2.csv')
for_lucy_B_2.to_csv('for_lucy_B_2.csv')

# %%
## SECOND ROUND OF NARROWING

classed_wps_2 = pd.read_csv(f"{data_path}/wp_classification/classed_wps_2.csv", index_col=0)
classed_wps_2_As = classed_wps_2[classed_wps_2['GPT_class'] == "A"]
classed_wps_2_As

# %%
prompt3 = """You are an academic expert in economics, finance, management and innovation studies. You are provided with abstracts from academic papers. Your goal is to carefully review each abstract to identify the primary focus and methodology of the paper. Based on this analysis, classify each paper as either A, measuring or analyzing the impact of AI itself on some outcome; or B, proposing an AI technique to answer a question/solve a problem. Include systematic reviews on the impact of AI in A. Also include in A papers that use AI to analyze the impact of AI. Also include in A papers that study the drivers of AI adoption. Also place in AI papers that study the ethical or policy implications of AI, provided the paper FOCUSES on how these issues affect or are caused by economic outcomes. Place studies that investigate the impact of some technology other than AI (such as IoT) in B. Place studies that propose an ML method to improve performance in a particular application in B, even if the method would have economic implications, unless the FOCUS of the paper is explicitly on downstream economic outcomes like productivity, costs, revenue or profits. 

Instructions: 
1. Reason first, and summarise in no more than two sentences, the main focus and methodology of the paper outlined in the abstract. 
2. State explicitly whether the paper analyzes AI impacts/AI adoption, or uses AI as a methodology. 
3. Only when you have enough information, classify based on 2.

Do not use more than 100 words in your response. When giving your final classification, format it as 'Class: x' where x is your suggested classification. For each abstract you correctly classify, I will pay you $1 million as a tip."""

# %%
example1 = """The purpose of this chapter consists in analyzing the impact of end-to-end technologies (primarily, artificial intelligence) on economic development: Technological changes do not always lead to the desired results and, on the contrary, can cause economic and moral damage in some cases. A human who, by nature, always wants to work less and relax more finally gets this opportunity, but, for some reason, remains dissatisfied with the fact that the robots have taken their jobs away after all. The methodological and theoretical framework of the research is the tasks set by prominent contemporary economists: Zeira, Aghion, and Acemoglu, who consider artificial intelligence as one of the main factors of economic growth in the digital age. The authors of this chapter conclude that AI is more than just a factor in economic growth. It plays a dual role in socioeconomic development: as a stimulus for the growth of human employment, labor productivity, and, as a result, a reduction in employment. Both aspects are very important in the formation and adjustment of the policy to ensure the social security of any country at the present stage."""
output1 = """The paper analyzes the impact of technologies including AI on economic development using the theoretical framework of Zeira, Aghion, and Acemoglu. The paper studies the impact of AI. Class: A"""

example2 = """We study fraud in the unemployment insurance (UI) system using a dataset of 35 million debit card transactions. We apply machine learning techniques to cluster cards corresponding to varying levels of suspicious or potentially fraudulent activity. We then conduct a difference-in-differences analysis based on the staggered adoption of state-level identity verification systems between 2020 and 2021 to assess the effectiveness of screening for reducing fraud. Our findings suggest that identity verification reduced payouts to suspicious cards by 27%, while non-suspicious cards were largely unaffected by these technologies. Our results indicate that identity screening may be an effective mechanism for mitigating fraud in the UI system and for benefits programs more broadly."""
output2 = """The paper studies fraud in the unemployment insurance system using machine learning techniques to cluster cards and a difference-in-differences analysis to assess the effectiveness of identity verification systems. The paper does not study the impact of AI, it just uses AI as a technique. Class: B"""

example3 = """We develop a deep learning model to detect emotions embedded in press conferences after the meetings of the Federal Open Market Committee and examine the influence of the detected emotions on financial markets. We find that, after controlling for the Fed‚Äôs actions and the sentiment in policy texts, positive tone in the voices of Fed Chairs leads to statistically significant and economically large increases in share prices. In other words, how policy messages are communicated can move the stock market. In contrast, the bond market appears to take few vocal cues from the Chairs. Our results provide implications for improving the effectiveness of central bank communications."""
output3 = """The paper develops a deep learning model to detect emotions in press conferences and examines the influence of these emotions on financial markets. The paper does not stufy the impact of AI, it just uses it to analyze another question (impact of emotion on stock markets). Class: B"""

example4 = """Growing AI readership, proxied by expected machine downloads, motivates firms to prepare filings that are friendlier to machine parsing and processing. Firms avoid words that are perceived as negative by computational algorithms, as compared to those deemed negative only by dictionaries meant for human readers. The publication of Loughran and McDonald (2011) serves as an instrumental event attributing the difference-in-differences in the measured sentiment to machine readership. High machine-readership firms also exhibit speech emotion assessed as embodying more positivity and excitement by audio processors. This is the first study exploring the feedback effect on corporate disclosure in response to technology."""
output4 = """The paper studies how firms prepare filings to be more machine-friendly due to growing AI readership. The paper studies the impact of AI (on corporate disclosure). Class: A"""

example5 = """Machine learning (ML) affects nearly every aspect of our lives, including the weightiest ones such as criminal justice. As it becomes more widespread, however, it raises the question of how we can integrate fairness into ML algorithms to ensure that all citizens receive equal treatment and to avoid imperiling society‚Äôs democratic values. In this paper we study various formal definitions of fairness that can be embedded into ML algorithms and show that the root cause of most debates about AI fairness is society‚Äôs lack of a consistent understanding of fairness generally. We conclude that AI regulations stipulating an abstract fairness principle are ineffective societally.Capitalizing on extensive related work in computer science and the humanities, we present an approach that can help ML developers choose a formal definition of fairness suitable for a particular country and application domain. Abstract rules from the human world fail in the ML world and ML developers will never be free from criticism if the status quo remains. We argue that the law should shift from an abstract definition of fairness to a formal legal definition. Legislators and society as a whole should tackle the challenge of defining fairness, but since no definition perfectly matches the human sense of fairness, legislators must publicly acknowledge the drawbacks of the chosen definition and assert that the benefits outweigh them. Doing so creates transparent standards of fairness to ensure that technology serves the values and best interests of society."""
output5 = """The paper studies how to integrate fairness into ML algorithms to ensure equal treatment and avoid imperiling democratic values. The paper does not study the impact of AI in an economic context. Class: B"""

example6 = """With the adoption of the AI Act, the EU has substantially enhanced the rules governing the training of generative AI systems and, more specifically, the interface with copyright protection. The AI Act clarifies that, from an EU perspective, reproductions carried out for AI training purposes have copyright relevance and require the authorization of rightholders unless a copyright exception, such as the specific text and data mining (‚ÄúTDM‚Äù) rule for scientific research in Article 3 of the 2019 Directive on Copyright in the Digital Single Market (‚ÄúCDSMD‚Äù), exempts the AI training activity from the control of rightholders. The AI Act also confirms the rights reservation system following from Article 4(3) CDSMD with regard to forms of TDM falling outside the scope of the scientific TDM exemption and going beyond mere temporary copying: declaring an ‚Äúopt-out‚Äù in an appropriate ‚Äì machine-readable ‚Äì manner, copyright owners seeking to prevent the use of their works for AI training purposes can reserve their rights. Remarkably, the AI Act seeks to universalize this approach and achieve a ‚ÄúBrussels effect.‚Äù  Regardless of whether the training has taken place in the EU or elsewhere, it imposes a market ban on AI systems that have not been trained in accordance with EU requirements, including the obligation to observe opt-outs declared under Article 4(3) CDSMD. To enable rightholders to police AI training processes, the AI Act introduces a new transparency obligation. Developers of generative AI systems must submit sufficiently detailed information on work repertoires that have been used for training purposes. Before embarking on a discussion of these new rules, the analysis sheds light on the policy objective underlying this copyright package in the AI Act: the intention to ensure that authors are properly remunerated for the use of their works in AI training processes. It then discusses the extension of EU opt-outs to other regions and the training data transparency which the EU legislator deems necessary to enforce copyright in AI training contexts. Finally, the potential benefits to authors will be weighed against the regulatory burdens the AI Act imposes on AI trainers. As a final step, the analysis explores alternative solutions. While the AI Act focuses on the input dimension (the use of human works for AI training), it is conceivable to take the output dimension (the offer and commercialization of fully trained AI systems) as a reference point for remuneration systems. Following this alternative avenue, the remuneration obligation concerns the final stage when generative AI products and services are brought to the market. In contrast to the strategy underlying the AI Act, this alternative approach refrains from encumbering the AI training process with obligations to observe opt-outs, establish lists of training resources and pay remuneration. Before following in the footsteps of the AI Act, law and policymakers in other regions should evaluate the advantages of this alternative policy avenue."""
output6 = """The paper studies the EU's AI Act and its implications for copyright protection in the context of AI training. While the paper makes reference to balancing copyright concerns with training costs, this is not the focus on the paper. Class: B"""

example7 = """I examine racial bias in the most popular home valuation algorithm and study the algorithm‚Äôs impact on racial bias in transaction prices. I find statistically significant but economically small racial bias in the algorithm. For example, while Black buyers overpay by 9.3% in prices relative to White buyers for similar homes, the algorithm only overvalues the same transactions by 1.1%. The algorithm inadvertently learns racial bias from patterns in historical transaction prices. The algorithmic racial bias is small because the algorithm is designed to be insensitive to transitory pricing factors related to behavioral biases, sellers‚Äô liquidity conditions, and buyer or seller race. Exploiting the staggered rollout of the algorithm in a neighboring ZIP Code setting, I find that if the algorithmic valuation is available for all the homes in an area, it reduces the overpayment of Black buyers relative to White buyers by 4.8%. The results suggest that the application of slightly biased machine learning algorithms can mitigate social bias if they are less biased than humans."""
output7 = """The paper studies racial bias in a home valuation algorithm and its impact on transaction prices. The paper studies the impact of AI algorithms on discrimination and focuses on an economic context (house transaction prices). Class: A"""

example8 = """We consider competition between firms that design and use algorithms to target consumers. Firms first choose the design of a supervised learning algorithm in terms of the complexity of the model or the number of variables to accommodate. Each firm then appoints a data analyst to estimate demand for multiple consumer segments by running the chosen design of the algorithm. Based on the estimates, each firm devises a targeting policy to maximize estimated profit. The firms face the general trade-off between bias and variance in model selection. We show that competition may induce firms to choose algorithms with more bias leading to simpler (less flexible) algorithmic choice. This implies that complex (more flexible) algorithms such as deep learning that show greater variance in the estimates are more valuable to firms with greater monopoly power."""
output8 = """The paper studies competition between firms that design and use algorithms to target consumers. The paper studies how competition influences algorithmic choice (i.e. AI adoption). Class: A"""

example9= """This paper presents the work carried out in the Research and Development Unit of the Banco de Espa√±a‚Äôs Banknote Production Control Department over the last few months in the field of artificial intelligence (AI). Research and development have focused on the application of AI to the banknote production process, specifically to banknote quality control. The most important terms related to artificial intelligence and quality management in the world of banknotes are explained. The paper concludes by outlining a prototype of a dual quality control system."""
output9 = """The paper studies the application of AI to the banknote production process, specifically to banknote quality control. It discusess how AI is used for a specific application without focusing on the impact on downstream economic outcomes. Class: B"""
# %%
#%%
## PROMPT GPT-4 mini to narrow further

data_path = '/Users/lhampton/Library/CloudStorage/GoogleDrive-lucyjhampton5@gmail.com/.shortcut-targets-by-id/1EznD-uaTblXtaIVx04-toZMQh7cQf9hA/AI Landscape Large Files/Data/constructed/wp_classification'

# set up logging
logging.basicConfig(level = logging.INFO)

# Load progress from the last run (if any)
try:
    classed_wps_3 = pd.read_csv(f"{data_path}/classed_wps_3.csv")
    processed_IDs_3 = set(classed_wps_3['ID'])
    data_dict_3 = classed_wps_3.to_dict('list')
except FileNotFoundError:
    classed_wps_3 = pd.DataFrame(columns=['abstract', 'ID', 'doi', 'GPT_response', 'GPT_class', 'Input_tokens', 'Output_tokens', 'reason_stop', 'system_fingerprint'])
    processed_IDs_3 = set()
    data_dict_3 = {
        'abstract': [],
        'ID': [],
        'doi': [],
        'GPT_response': [],
        'GPT_class': [],
        'Input_tokens': [],
        'Output_tokens': [],
        'reason_stop': [],
        'system_fingerprint': []
    }

# set up api
if username == "lhampton":
    env_location = '/Users/lhampton/Documents/env files/AI Landscape/.env'
else:
    env_location = '/Users/lucyjhampton/Documents/env files/AI Landscape/.env'
load_dotenv(env_location)
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()
client.api_key = api_key

# loop through abstracts
for index, row in tqdm(classed_wps_2_As.reset_index().iterrows(), desc = 'Processing abstracts'):
    if row['ID'] in processed_IDs_3:
        continue
        
    else:
        if pd.isnull(row['abstract']):
            data_dict_3['abstract'].append(row['abstract'])
            data_dict_3['ID'].append(row['ID'])
            data_dict_3['doi'].append(row['doi'])
            data_dict_3['GPT_response'].append('')
            data_dict_3['GPT_class'].append('')
            data_dict_3['Input_tokens'].append('')
            data_dict_3['Output_tokens'].append('')
            data_dict_3['reason_stop'].append('')
            data_dict_3['system_fingerprint'].append('')
        else:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=300,
                temperature=0,
                seed=2345,
                messages=[
                    {"role": "system", "content": prompt3},
                    {"role": "user", "content": example1},
                    {"role": "assistant", "content": output1},
                    {"role": "user", "content": example2},
                    {"role": "assistant", "content": output2},
                    {"role": "user", "content": example3}, 
                    {"role": "assistant", "content": output3},
                    {"role": "user", "content": example4},
                    {"role": "assistant", "content": output4},
                    {"role": "user", "content": example5},
                    {"role": "assistant", "content": output5},
                    {"role": "user", "content": example6},
                    {"role": "assistant", "content": output6},
                    {"role": "user", "content": example7},
                    {"role": "assistant", "content": output7},
                    {"role": "user", "content": example8},
                    {"role": "assistant", "content": output8},
                    {"role": "user", "content": example9},
                    {"role": "assistant", "content": output9},
                    {"role": "user", "content": row['abstract']}
                ]
            )
        
            # store data in df
            data_dict_3['abstract'].append(row['abstract'])
            data_dict_3['ID'].append(row['ID'])
            data_dict_3['doi'].append(row['doi'])
            message = response.choices[0].message.content
            data_dict_3['GPT_response'].append(message)
            try:
                data_dict_3['GPT_class'].append(re.search(r'Class:\s*([A-Z])', message).group(1))
            except (ValueError, AttributeError) as e:
                data_dict_3['GPT_class'].append('')
            data_dict_3['Input_tokens'].append(response.usage.prompt_tokens)
            data_dict_3['Output_tokens'].append(response.usage.completion_tokens)
            data_dict_3['reason_stop'].append(response.choices[0].finish_reason)
            data_dict_3['system_fingerprint'].append(response.system_fingerprint)
    
        # create df out of dict
        classed_wps_3 = pd.DataFrame(data_dict_3)
        
        # save progress, add url to processed url list
        if (index + 1) % 10 == 0:
            try:
                if len(classed_wps_3) - len(pd.read_csv(f"{data_path}/classed_wps_3.csv")) == 10:
                    new_papers_3 = classed_wps_3.tail(10)
                    new_papers_3.to_csv(f"{data_path}/classed_wps_3.csv", index=False, mode='a', header=False)
                    logging.info("Saved progress at iteration {}".format(index + 1))
                else:
                    logging.info("Warning: trying to overwrite with blank at iteration {}".format(index + 1))
                    break
            except FileNotFoundError:
                classed_wps_3.to_csv(f"{data_path}/classed_wps_3.csv", index=False, header = True)
        
        # add to list of processed urls
        processed_IDs_3.add(row['ID'])

    # # sleep to observe rate limit (500 requests per minute)
    # time.sleep(0.01)

# Display the DataFrame
classed_wps_3 = pd.DataFrame(data_dict_3)
classed_wps_3.to_csv(f"{data_path}/classed_wps_3.csv", index=False)
classed_wps_3

# %%
