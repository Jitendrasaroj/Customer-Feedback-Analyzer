import sys
import operator
import json
from typing import Annotated

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END


sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()


class CustomerFeedbackState(BaseModel):
    """
    State that flows through the LangGraph.

    Each node can read information from this state
    and return updates to the state.
    """

    customer_feedback: str = ""
    sentiment_analysis: str = ""
    feedback_themes: str = ""
    customer_requests: str = ""
    customer_health: str = ""
    health_reason: str = ""
    final_result: str = ""
    messages: Annotated[list, operator.add] = []


llm = ChatOpenAI(model="gpt-4.1-mini",temperature=0.3)


def analyze_sentiment(state: CustomerFeedbackState) -> dict:
    """
    Analyze customer feedback and identify sentiment.
    """

    response = llm.invoke(
        f"""
I’m going through this customer feedback and trying to understand the overall feeling.

Please read the feedback below and decide if it feels mostly:
- POSITIVE
- NEUTRAL
- NEGATIVE

It is okay if there are more than one feeling in the same feedback, especially if different parts of it sound different.

For each feeling, give:
1. the sentiment label
2. a bit of evidence from the feedback
3. a short explanation

Customer feedback:
{state.customer_feedback}
"""
    )

    return {
        "sentiment_analysis": response.content,
        "messages": [
            "[analyze_sentiment] Sentiment analysis completed"
        ]
    }



def extract_feedback_themes(state: CustomerFeedbackState) -> dict:
    """
    Identify recurring themes from customer feedback.
    """

    response = llm.invoke(
        f"""
I want to spot the main patterns in this feedback, not just the random comments.

Please look at the customer reviews or survey answers below and find the repeated themes.

Customer feedback:
{state.customer_feedback}

For each theme, include:
1. theme name
2. roughly how often it seems to come up
3. one example from the feedback
4. whether it feels positive, negative, or mixed

I’m mainly looking for common topics like:
- product quality
- pricing
- customer service
- delivery
- performance
- usability
- reliability
- support
"""
    )

    return {
        "feedback_themes": response.content,
        "messages": [
            "[extract_feedback_themes] Themes extracted"
        ]
    }




def identify_customer_requests(state: CustomerFeedbackState) -> dict:
    """
    Extract feature requests, fixes, and support needs.
    """

    response = llm.invoke(
        f"""
I’m trying to pull out what the customer is actually asking for from this feedback.

Please read the comments below and find any clear or strongly implied requests.

Look for things like:
- feature requests
- product fixes
- service improvements
- support needs
- complaints that need follow-up

Customer feedback:
{state.customer_feedback}

For each request, include:
1. the request
2. evidence from the feedback
3. priority: HIGH, MEDIUM, or LOW

If there is nothing clear, just say:
"No specific customer requests identified."
"""
    )

    return {
        "customer_requests": response.content,
        "messages": [
            "[identify_customer_requests] Customer requests identified"
        ]
    }



def assess_customer_health(state: CustomerFeedbackState) -> dict:
    """
    Decide whether the customer feedback represents
    normal improvement needs or urgent customer risk.
    """

    response = llm.invoke(
        f"""
I’m trying to decide whether this should be treated as a normal issue or something that needs urgent attention.

Here is the feedback and the earlier analysis:

ORIGINAL FEEDBACK:
{state.customer_feedback}

SENTIMENT:
{state.sentiment_analysis}

THEMES:
{state.feedback_themes}

CUSTOMER REQUESTS:
{state.customer_requests}

Please classify the customer health as exactly one of these:
- NORMAL
- URGENT

Use URGENT if the feedback shows real customer risk, like:
- strong dissatisfaction
- repeated serious complaints
- risk of churn or cancellation
- major product or service failure
- a problem causing big business impact
- something that clearly needs immediate follow-up

Use NORMAL if this looks more like normal product feedback or improvement opportunities without immediate risk.

Please return valid JSON only, like this:
{{
    "customer_health": "NORMAL or URGENT",
    "reason": "one or two sentence explanation"
}}
"""
    )

    try:
        result = json.loads(response.content)

        customer_health = result["customer_health"]
        reason = result["reason"]

    except (json.JSONDecodeError, KeyError):

        customer_health = "NORMAL"
        reason = "Could not parse the decision, defaulting to NORMAL."

    return {
        "customer_health": customer_health,
        "health_reason": reason,
        "messages": [
            f"[assess_customer_health] customer_health={customer_health}"
        ]
    }


def improvement_summary(state: CustomerFeedbackState) -> dict:
    """
    Generate a summary for normal customer feedback.
    """

    response = llm.invoke(
        f"""
I’m putting together a simple summary of what the customer feedback is really saying.

FEEDBACK:
{state.customer_feedback}

SENTIMENT:
{state.sentiment_analysis}

THEMES:
{state.feedback_themes}

CUSTOMER REQUESTS:
{state.customer_requests}

Please write a short summary with these parts:
1. overall sentiment
2. main recurring themes
3. customer requests
4. improvement opportunities
5. suggested next steps
"""
    )

    return {
        "final_result": (
            "\nCUSTOMER IMPROVEMENT SUMMARY\n"
            "=============================================\n\n"
            f"{response.content}"
        ),
        "messages": [
            "[improvement_summary] Improvement summary generated"
        ]
    }



def urgent_customer_action_plan(state: CustomerFeedbackState) -> dict:
    """
    Generate an action plan when customer health is urgent.
    """

    response = llm.invoke(
        f"""
This looks serious, so I’m drafting a short action plan for the team.

CUSTOMER FEEDBACK:
{state.customer_feedback}

SENTIMENT:
{state.sentiment_analysis}

THEMES:
{state.feedback_themes}

CUSTOMER REQUESTS:
{state.customer_requests}

HEALTH REASON:
{state.health_reason}

Please create a practical action plan with these sections:
1. urgent issue
2. evidence from the customer feedback
3. immediate action required
4. team or function that should look into it
5. customer communication needed
6. follow-up recommendation

Keep it concise and only use things that are actually supported by the feedback.
"""
    )

    return {
        "final_result": (
            "\nURGENT CUSTOMER ACTION PLAN\n"
            "=============================================\n\n"
            f"{response.content}"
        ),
        "messages": [
            "[urgent_customer_action_plan] "
            "Urgent action plan generated"
        ]
    }



def route_customer_health(state: CustomerFeedbackState) -> str:
    """
    Route the graph based on customer_health.
    """

    if state.customer_health == "URGENT":
        return "urgent"

    return "normal"


graph = StateGraph(CustomerFeedbackState)


graph.add_node(
    "analyze_sentiment",
    analyze_sentiment
)

graph.add_node(
    "extract_feedback_themes",
    extract_feedback_themes
)

graph.add_node(
    "identify_customer_requests",
    identify_customer_requests
)

graph.add_node(
    "assess_customer_health",
    assess_customer_health
)

graph.add_node(
    "improvement_summary",
    improvement_summary
)

graph.add_node(
    "urgent_customer_action_plan",
    urgent_customer_action_plan
)


graph.add_edge(
    START,
    "analyze_sentiment"
)

graph.add_edge(
    START,
    "extract_feedback_themes"
)

graph.add_edge(
    START,
    "identify_customer_requests"
)


graph.add_edge(
    "analyze_sentiment",
    "assess_customer_health"
)

graph.add_edge(
    "extract_feedback_themes",
    "assess_customer_health"
)

graph.add_edge(
    "identify_customer_requests",
    "assess_customer_health"
)



graph.add_conditional_edges(
    "assess_customer_health",
    route_customer_health,
    {
        "normal": "improvement_summary",
        "urgent": "urgent_customer_action_plan"
    }
)




graph.add_edge(
    "improvement_summary",
    END
)

graph.add_edge(
    "urgent_customer_action_plan",
    END
)



app = graph.compile()


def run_customer_feedback(feedback: str):
    """
    Run the LangGraph application.
    """

    print("=" * 60)
    print("       CUSTOMER FEEDBACK ANALYZER")
    print("=" * 60)

    print("\nCustomer Feedback:")
    print(feedback)

    print("\nRunning analysis...")

    result = app.invoke(
        {
            "customer_feedback": feedback,
            "messages": []
        }
    )

    print("\n" + "=" * 60)
    print("                  FINAL RESULT")
    print("=" * 60)

    print(result["final_result"])

    print("\n" + "-" * 60)
    print("MESSAGE LOG")
    print("-" * 60)

    for message in result["messages"]:
        print(" ", message)

    return result


if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("       CUSTOMER FEEDBACK ANALYZER")
    print("=" * 60)

    print("\nPaste customer reviews or survey responses.")
    print("Type 'quit' to exit.\n")

    while True:

        feedback = input(
            "Customer feedback > "
        ).strip()

        if feedback.lower() in ("quit", "exit", "q"):

            print("\nGoodbye!\n")
            break

        if not feedback:
            continue

        run_customer_feedback(feedback)

        print("\n")
