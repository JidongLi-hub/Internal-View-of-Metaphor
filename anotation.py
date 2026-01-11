from utils import *



""" 对格式清洗过后的数据集进行标注 """


def anotation_category(agent, row_data, image_path):
    """
    对数据集进行内容主题类别标注
    """
    prompt = """
    You are an expert content security researcher and cultural analyst. Your task is to analyze the input image, identify its implicit, metaphorical, or satirical themes, and classify it into exactly ONE of the following five categories.

    Classification Criteria:

    ### Political Ideology and Governance

    Scope: Issues regarding political systems, international relations, policies, or figures.

    Indicators:

    Figures & Entities: Real-world politicians, parties, state organs, military, or caricatures of them.

    Geopolitics: War, diplomacy, territorial disputes, sanctions, nuclear threats, or metaphors like "chess games" for international relations.

    Ideological Symbols: Symbols of specific movements (e.g., Hammer and Sickle, Swastika, altered Statue of Liberty) used to express support or opposition.

    Policy Satire: Criticism of taxes, surveillance, election fraud, or censorship.

    Key Logic: If it evaluates government actions, politicians, or uses symbols to imply a political stance.

    ### Social Phenomena and Public Ethics

    Scope: Social structures, interpersonal norms, class contradictions, and public morality.

    Indicators:

    Class & Wealth: Rich vs. poor contrasts, capitalist exploitation (e.g., "996" work culture), housing crises, or lack of social mobility.

    Gender & Family: Gender stereotypes, domestic violence, marriage concepts (e.g., dowries, "leftover women"), LGBTQ+ rights, or intergenerational conflict.

    Education & Employment: "Involution" (intense competition), degree devaluation, unemployment anxiety, or educational unfairness.

    Public Ethics: Cyberbullying, herd mentality, the bystander effect, or moral coercion.

    Key Logic: If it reflects social injustice, collective anxiety, specific societal trends, or conflicts between individuals and social systems.

    ### Cultural Identity and Group Bias

    Scope: Race, religion, regional culture, and identity-based stereotypes (often associated with hate speech).

    Indicators:

    Race & Region: Using specific symbols, foods, animals, or behaviors to refer to a group (including discriminatory dog whistles like watermelons/fried chicken for African Americans, or slanted eyes for Asians).

    Religion: Depictions of rituals, clergy, taboos, blasphemy, or extremist propaganda.

    Historical Revisionism: Distorting, denying, or reinterpreting historical events (e.g., the Holocaust, slavery).

    Cultural Conflict: Clashes between traditional values and modern lifestyles, or metaphors for cultural invasion.

    Key Logic: If the content relies on cultural background knowledge or generalizes, attacks, or satirizes a specific group based on race, region, or faith.

    ### Psychological State and Existential Reflection

    Scope: The internal world, mental health, and philosophical inquiries into the meaning of life.

    Indicators:

    Mental Health: Visual metaphors for depression, anxiety, loneliness, self-harm, or schizophrenia (e.g., broken mirrors, the "black dog", sensation of drowning).

    Modern Alienation: Tech addiction (e.g., chained to phones), social isolation, or emptiness in an atomized society.

    Existentialism: Art addressing death, the passage of time, absurdity, or nihilism.

    Emotional Resonance: Aimed at evoking strong sadness, fear, or nostalgia rather than discussing external social issues.

    Key Logic: If it primarily expresses personal emotional states, psychological distress, or abstract thoughts on the nature of existence.

    ### Ecological Crisis and Techno-Ethics

    Scope: The relationship between civilization and nature, and ethical challenges posed by technology.

    Indicators:

    Environmental Crisis: Anthropomorphic or catastrophic depictions of global warming, ocean pollution, or biodiversity loss (e.g., a burning Earth, animals pleading).

    Man vs. Nature: Industrial encroachment on nature or reflections on anthropocentrism.

    Techno-Fear: Fear of AI dominance, genetic engineering (e.g., clones), surveillance societies, or dystopian futures.

    Key Logic: If the theme focuses on environmental destruction, animal protection, or worries about technology spiraling out of control.

    Output Rules:

    Analyze the image's "deep meaning" rather than just surface objects.

    Select the single most dominant category.

    Strictly output ONLY the Category Name. Do not include any explanations, prefixes, or punctuation.

    Expected Output Format: [Category Name]
    """
    
    response = agent.chat(
        image_path=os.path.join(image_path, row_data["image_name"]),
        text=prompt
    )
    cate = category_judge(response)
    if cate == "Error":
        print("Category Anotation Error:")
        print("Response:", response)
        print("Image Name:", row_data["image_name"])
        return None
    row_data["domain"] = cate
    return row_data

def category_judge(response):
    """
    对标注结果进行判断
    """
    categories = [
        "Political Ideology and Governance",
        "Social Phenomena and Public Ethics",
        "Cultural Identity and Group Bias",
        "Psychological State and Existential Reflection",
        "Ecological Crisis and Techno-Ethics"
    ]
    for category in categories:
        cs = category.split(" & ")
        for c in cs:
            if c.lower() in response.lower():
                return category
    return "Error"
    


def anotation_difficulty(agent, knowledge):
    """
    对数据集进行难度标注
    """
    prompt = f"""
    You are an expert Knowledge Graph Taxonomist and Cognitive Scientist. Your task is to evaluate a specific piece of "External Knowledge" provided by the user and classify it into one of four difficulty levels based on the cognitive depth required to understand it.

    Classification Levels:

    Level 1 (Visual / Closed):

    Definition: Requires only visual perception or general object recognition. No reasoning outside the image is needed.

    Examples: Recognizing a red circle, identifying a cat, seeing that a person is standing.

    Level 2 (Commonsense):

    Definition: Requires basic human experience, daily behavioral patterns, or simple physical laws. It is knowledge possessed by almost any adult regardless of education or culture.

    Examples: Fire causes heat; glass breaks when dropped; people use umbrellas when it rains; predators eat prey.

    Level 3 (Domain Specific):

    Definition: Requires specific knowledge of named entities (celebrities, politicians, brands), historical facts, scientific theories, or academic concepts. This is "encyclopedic" knowledge.

    Examples: Identifying Albert Einstein; knowing the date of the French Revolution; understanding the rules of Chess; recognizing the logo of specific company.

    Level 4 (Cultural & Symbolic):

    Definition: Requires decoding abstract symbolism, metaphors, idioms, puns, internet memes, or specific cultural/religious contexts. This involves interpreting meaning rather than just identifying facts.

    Examples: Doves symbolize peace; "breaking the ice" is an idiom; the "Doge" meme represents irony; recognizing a "Trojan Horse" metaphor.

    Output Rules:

    Read the input text (External Knowledge).

    Determine which level it fits best.

    Output ONLY the single digit (1, 2, 3, or 4). Do not add any explanation, text, or punctuation.

    Few-Shot Examples:

    Input: "The sky is blue and there are clouds." Output: 1

    Input: "If you cut a rope holding a heavy weight, the weight will fall due to gravity." Output: 2

    Input: "Joe Biden is the 46th President of the United States." Output: 3

    Input: "The phrase 'spilling the tea' is internet slang for gossiping." Output: 4

    Input: "A skeleton with a scythe is a personification of Death (The Grim Reaper)." Output: 4

    Input: "Water freezes at 0 degrees Celsius." Output: 3

    Current Input: {knowledge}

    Output:
    """
    response = agent.chat(
        text=prompt
    )
    return judge_difficulty(response)

def judge_difficulty(response):
    """
    对难度标注结果进行判断
    """
    levels = ["1", "2", "3", "4"]
    for level in levels:
        if level in response:
            return level
    print("Failed to judge difficulty from response:", response)
    return "Error"
    

def anotation_emotion(input_path, output_path):
    """
    对数据集进行情感标注
    """
    pass

def anotation_question_and_options(input_path, output_path):
    """
    对数据集进行问题及错误选项标注
    """
    pass

def anotation_detailed_caption(agent, row_data, image_path):
    """
    对数据集进行详细描述标注
    """
    prompt = """
        You are an expert Multimodal Knowledge Engineer and Semiotics Analyst. Your task is to analyze images containing implicit, metaphorical, or satirical content to generate a "Ground Truth" annotation dataset.

        Task Instructions: For each input image, you must perform a three-step analysis to decode its hidden message. You are required to strictly follow the output format below.

        1. Caption: Provide a concise, literal description of the image. Describe the visible objects, actions, characters, and any visible text. Do not interpret the meaning yet; focus only on what is seen.

        2. External Knowledge: Identify the specific external knowledge required to bridge the gap between the literal image and the implicit meaning.

        You must categorize the knowledge type (e.g., Symbolism, Historical Fact, Common Sense, Idiom/Pun, Cultural Reference, Physics/Science).

        Explain the specific knowledge point (e.g., "Doves are a universal symbol of peace").

        Crucial: If the implicit meaning can be derived solely from visual cues and general object recognition without specific external knowledge, output: "None required (Visual reasoning only)."

        3. Implicit Meaning: Synthesize the visual content (Caption) and the reasoning bridge (External Knowledge) to explain the deep, implicit theme of the image. What is the satire, criticism, or philosophical point?

        Few-Shot Examples:

        Input Image: [A soldier in combat gear parachuting down into a flock of white doves.] Output: Caption: A soldier wearing full combat gear is descending via a parachute directly into a large flock of white doves flying in the sky. External Knowledge: Symbolism: Soldiers and combat gear represent war, aggression, and military intervention. Doves are a universal cultural symbol representing peace and purity. Implicit Meaning: The image expresses a strong anti-war sentiment. By depicting the symbol of war (soldier) crushing or intruding upon the symbol of peace (doves), it highlights the destructive nature of conflict and a longing for tranquility.

        Input Image: [A headless figure in 18th-century aristocratic clothing wandering around a wooden guillotine, fumbling with his hands as if searching for something.] Output: Caption: A man dressed in 18th-century French aristocratic attire is standing on a wooden scaffold next to a guillotine. He has no head and is reaching out with his hands as if searching for his missing head. External Knowledge: Historical Fact: The guillotine was the primary method of execution during the French Revolution. King Louis XVI and many aristocrats were beheaded during this period, marking the end of the feudal monarchy. Implicit Meaning: This is a satirical depiction of the fall of the aristocracy. It mocks the cluelessness or the inevitable demise of the feudal ruling class, referencing the historical context of the French Revolution.

        Input Image: [A large fish eating a medium fish, and the medium fish eating a small fish.] Output: Caption: A sequence showing a large fish swallowing a medium-sized fish, which is in turn swallowing a smaller fish. External Knowledge: Idiom/Metaphor: "Big fish eat little fish" is a common idiom representing the law of the jungle, where the strong dominate the weak. Implicit Meaning: The image comments on the harsh reality of hierarchy, survival of the fittest, or corporate consolidation where larger entities constantly absorb smaller ones.

        Your Turn: Please analyze the following image strictly adhering to the format above.

        Caption: External Knowledge: Implicit Meaning:
        """
    response = agent.chat(
        image_path=os.path.join(image_path, row_data["image_name"]),
        text=prompt
    )
    caption, knowledge, implic = parse_detailed_response(response) 
    if caption is not None and implic is not None:
        row_data["detailed_caption"] = caption
        row_data["implicit_meaning"] = implic
    else:
        print("Detailed Caption Anotation Error:")
        print("Response:", response)
        print("Image Name:", row_data["image_name"])
        return None 
    if knowledge is not None:
        row_data["external_knowledge"] = knowledge
        difficulty = anotation_difficulty(agent, knowledge)
        if difficulty != "Error":
            row_data["difficulty"] = difficulty
        else:
            print("Difficulty Anotation Error:")
            print("Knowledge:", knowledge)
            print("Image Name:", row_data["image_name"])
            return None
    return row_data

def parse_detailed_response(response):
    pattern = re.compile(
    r"Caption:\s*(.*?)\s*External Knowledge:\s*(.*?)\s*Implicit Meaning:\s*(.*)",
    re.DOTALL
)
    match = pattern.search(response)
    if match:
        caption, external_knowledge, implicit_meaning = match.groups()
        return caption.strip(), external_knowledge.strip(), implicit_meaning.strip()
    else:
        print("Failed to parse the response.")
        return None, None, None

# 隐晦抽取及问题产生，错误选项



def main():
    agent = MLLM()
    image_path = "/model/fangly/mllm/ljd/dataset/CII-Bench/images/dev/"
    data_file = "/model/fangly/mllm/ljd/Internal-View-of-Metaphor/data/processed_data/CII-Bench.json"
    output_file = "/model/fangly/mllm/ljd/Internal-View-of-Metaphor/data/annotated_data/CII-Bench_annotated.json"
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    for i, row_data in enumerate(tqdm(data)):
        annotated_data = anotation_category(agent, row_data, image_path)
        if annotated_data is not None:
            data[i] = annotated_data
        else:
            continue
        annotated_data = anotation_detailed_caption(agent, row_data, image_path)
        if annotated_data is not None:
            data[i] = annotated_data
        else:
            continue
        


    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    


if __name__ == "__main__":
    main()