from langchain_community.llms import Ollama
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
import datetime

# Setup LLM
llm = Ollama(model="llama2")

# Create Tools
def get_time(input=""):
    """Returns current time"""
    return datetime.datetime.now().strftime("%I:%M %p")

def calculator(input):
    """Does simple math"""
    try:
        return str(eval(input))
    except:
        return "Error in calculation"

tools = [
    Tool(
        name="Time",
        func=get_time,
        description="Useful for when you need to know the current time"
    ),
    Tool(
        name="Calculator",
        func=calculator,
        description="Useful for math calculations. Input should be a math expression like '5*3'"
    ),
]

# Create prompt template
template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)

# Create agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

print("🤖 Agent with Tools Ready!")
print("Try: 'what time is it?' or 'calculate 15 * 7'")
print("-" * 60)

while True:
    user_input = input("\n💬 You: ").strip()
    
    if user_input.lower() in ['quit', 'exit']:
        print("👋 Bye!")
        break
    
    try:
        response = agent_executor.invoke({"input": user_input})
        print(f"\n🤖 Agent: {response['output']}")
    except Exception as e:
        print(f"\n❌ Error: {e}")