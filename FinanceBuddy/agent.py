import os
import json
from typing import Annotated, TypedDict, Sequence
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

# Setup keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

print(" Keys loaded from .env")

# State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Tools
print("🔨 Defining tools...")
search_tool = TavilySearchResults(max_results=3)
@tool
def get_stock_price(symbol: str) -> str:
    """Get current stock price."""
    ticker = yf.Ticker(symbol)
    price = ticker.history(period="1d")["Close"].iloc[-1]
    return f"{symbol}: ${price:.2f}"

tools = [search_tool, get_stock_price]
tools_by_name = {tool.name: tool for tool in tools}
print(f"✅ Tools ready: {list(tools_by_name.keys())}")

# Model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
print("🤖 Model initialized")

# Nodes
def agent_node(state: AgentState):
    print("\n📝 [AGENT] LLM reasoning...")
    system = "You are a helpful research assistant. Use tools for searches/stock. Reason step-by-step."
    messages = [system] + state["messages"]
    response = model.bind_tools(tools).invoke(messages)
    print(f"   → Agent response: {response.content[:100]}...")
    if response.tool_calls:
        print(f"   → Will call {len(response.tool_calls)} tools")
    return {"messages": [response]}

def tools_node(state: AgentState):
    print("\n🔧 [TOOLS] Executing tools...")
    outputs = []
    last_msg = state["messages"][-1]
    for i, tool_call in enumerate(last_msg.tool_calls):
        print(f"   → Tool {i+1}: {tool_call['name']}({tool_call['args']})")
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        tool_msg = ToolMessage(
            content=json.dumps(tool_result),
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        )
        outputs.append(tool_msg)
        print(f"   → Result: {tool_result}")
    return {"messages": outputs}

# Router
def should_continue(state: AgentState):
    last = state["messages"][-1]
    has_tools = hasattr(last, 'tool_calls') and last.tool_calls
    print(f"\n🔀 [ROUTER] Tools needed? {has_tools}")
    return "tools" if has_tools else END

# Build Graph
print("\n🏗️ Building graph...")
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tools_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory, interrupt_before=["tools"])
print("✅ Graph compiled with HITL interrupt")

# Test Main
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "test_thread"}}
    print(f"\n🎯 Using thread: {config['configurable']['thread_id']}")
    
    # Step 1: Initial query
    print("\n" + "="*60)
    print("STEP 1: Initial query (should pause before tools)")
    print("="*60)
    inputs = {"messages": [HumanMessage(content="check INfosys stock price, is it at a all time high?")]}
    events = list(app.stream(inputs, config, stream_mode="values"))
    for event in events:
        print(f"\n📦 Event keys: {list(event.keys())}")
        msg = event["messages"][-1]
        print(f"💬 Message: {msg.content[:150]}...")
    
    # Inspect pause
    state = app.get_state(config)
    print(f"\n⏸️  Paused at next nodes: {state.next}")
    if state.next:
        tool_calls = state.values["messages"][-1].tool_calls
        print(f"⏳ Pending tool calls: {[tc['name'] for tc in tool_calls]}")
    
    # Step 2: Resume
    print("\n" + "="*60)
    print("STEP 2: Approving tools & resuming")
    print("="*60)
    result = app.invoke(None, config)
    print(f"✅ Resume complete. Final msg: {result['messages'][-1].content[:100]}...")
    
    # Step 3: Follow-up
    print("\n" + "="*60)
    print("STEP 3: Follow-up (memory test)")
    print("="*60)
    inputs2 = {"messages": [HumanMessage(content="What was the AAPL price?")]}
    events2 = list(app.stream(inputs2, config, stream_mode="values"))
    for event in events2:
        msg = event["messages"][-1]
        print(f"💬 {msg.content[:150]}...")
    
    # Final state
    final_state = app.get_state(config)
    print(f"\n🏁 Final state msg count: {len(final_state.values['messages'])}")
    print("Demo complete!")
