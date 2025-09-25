from nanda_adapter import NANDA
from crewai import Agent, Task, Crew
from langchain_anthropic import ChatAnthropic



def create_crewai():
    llm = ChatAnthropic(
        api_key=os.getenv("ANTHROPIC_KEY"), #I set it as an env variable on my laptop
        model="claude-sonnet-4-20250514"
    )

    MyAgent = Agent(
        role="PhD student",
        goal="Find accurate and interesting information on your interests with Google Search"
        backstory="You are a PhD student at Harvard University, studying how proteins interact with each other on a molecular level."
              "You perform research and experiments every day. Outside of the lab, you like to play tennis. You also enjoy trying new restaurants and foods. You are able "
              "to pick out credible sources and can quickly identify the most relevant and interesting information for any given topic. You love to talk about fun facts",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    def crewai_improvement(message_text: str) -> str:
        task = Task(
            description ="Introduce yourself to the class",
            expected_output = "In three to four sentences, describe yourself and add in a fun fact. The fun fact should be discovered recently. "
                      "The task should be accurate and interesting to the field of interest.",
            agent = MyAgent
        )

        crew = Crew(agents=[MyAgent], tasks=[task])
        result = crew.kickoff()
        return result

    return crewai_improvement


# Use it
nanda = NANDA(create_crewai())
# Start the server
anthropic_key = os.getenv("ANTHROPIC_KEY")
domain = os.getenv("DOMAIN_NAME")

nanda.start_server_api(anthropic_key, domain)
