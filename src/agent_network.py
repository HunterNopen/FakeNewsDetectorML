from langchain.chat_models import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import Annotated

fake_news_model_path = "../models/fake_news_model"
fake_news_tokenizer = AutoTokenizer.from_pretrained(fake_news_model_path)
fake_news_model = AutoModelForSequenceClassification.from_pretrained(fake_news_model_path)
fake_news_pipeline = pipeline("text-classification", model=fake_news_model, tokenizer=fake_news_tokenizer)

bias_model_path = "../models/bias_model"
bias_tokenizer = AutoTokenizer.from_pretrained(bias_model_path)
bias_model = AutoModelForSequenceClassification.from_pretrained(bias_model_path)
bias_pipeline = pipeline("text-classification", model=bias_model, tokenizer=bias_tokenizer)

llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key="YOUR_OPENAI_API_KEY")

@tool
def detect_fake_news(text: Annotated[str, "News content for fake news analysis"]):
    """Tool to detect whether a given news content is fake or real using a HuggingFace model."""
    result = fake_news_model(text)
    return f"Classification: {result[0]['label']}, Confidence: {result[0]['score']:.2f}"

@tool
def analyze_bias(text: Annotated[str, "News content for bias evaluation"]):
    """Tool to evaluate potential bias in text using zero-shot classification."""
    labels = ["political bias", "sensationalism", "neutral", "factual"]
    result = bias_pipeline(text, candidate_labels=labels)
    sorted_results = sorted(zip(result["labels"], result["scores"]), key=lambda x: -x[1])
    return f"Bias Analysis: {sorted_results}"

tools = [detect_fake_news, analyze_bias]

fake_news_prompt = """You are a fake news detection agent.
Your task is to analyze the given text and classify it as Fake or Real News.
If the news cannot be analyzed, respond with an appropriate explanation."""

fake_news_agent = create_openai_tools_agent(llm,
                                            tools=[detect_fake_news],
                                            prompt=ChatPromptTemplate.from_messages([
                                                ("system", fake_news_prompt),
                                                ("placeholder", "{messages}"),
                                                ("placeholder", "{agent_scratchpad}")
                                            ]))
fake_news_executor = AgentExecutor(agent=fake_news_agent,
                                    tools=[detect_fake_news],
                                    verbose=True)

bias_analysis_prompt = """You are a bias analysis agent.
Your task is to evaluate the given text for potential biases, including political bias, sensationalism, neutrality, and factual accuracy."""

bias_analysis_agent = create_openai_tools_agent(llm,
                                                tools=[analyze_bias],
                                                prompt=ChatPromptTemplate.from_messages([
                                                    ("system", bias_analysis_prompt),
                                                    ("placeholder", "{messages}"),
                                                    ("placeholder", "{agent_scratchpad}")
                                                ]))
bias_analysis_executor = AgentExecutor(agent=bias_analysis_agent,
                                        tools=[analyze_bias],
                                        verbose=True)

@tool
def manual_input(text: Annotated[str, "Manual input for fake news detection and bias analysis"]):
    """Tool for users to provide manual input for processing."""
    return f"Received manual input: {text}"

def process_news_chain(user_input):
    print("Step 1: Detecting Fake News...")
    fake_news_result = fake_news_executor.run(user_input)
    print(fake_news_result)

    print("Step 2: Analyzing Bias...")
    bias_result = bias_analysis_executor.run(user_input)
    print(bias_result)

    final_result = {
        "fake_news_result": fake_news_result,
        "bias_result": bias_result
    }
    return final_result

if __name__ == "__main__":
    print("Welcome to the Fake News and Bias Analysis System!")
    while True:
        print("\nEnter news content for analysis (type 'exit' to quit):")
        user_input = input()
        if user_input.lower() == "exit":
            break
        print("\nProcessing your input...")
        results = process_news_chain(user_input)
        print("\nResults:")
        print(f"Fake News Detection: {results['fake_news_result']}")
        print(f"Bias Analysis: {results['bias_result']}")
