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
You are a customer feedback sentiment specialist.

Analyze the following customer feedback:

{state.customer_feedback}

Classify the feedback into:

- POSITIVE
- NEUTRAL
- NEGATIVE

You can identify multiple sentiments if the feedback contains
different comments.

For each sentiment provide:

1. The sentiment
2. Supporting evidence from the feedback
3. A short explanation

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
You are a customer feedback theme specialist.

Analyze these customer reviews or survey responses:

{state.customer_feedback}

Identify recurring product or service themes.

For each theme provide:

1. Theme name
2. Number or approximate frequency of mentions if possible
3. Example evidence from the feedback
4. Whether the theme is positive, negative, or mixed

Focus on recurring themes such as:

- Product quality
- Pricing
- Customer service
- Delivery
- Performance
- Usability
- Reliability
- Support

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
You are a customer request specialist.

Analyze the following customer feedback:

{state.customer_feedback}

Extract explicit or strongly implied customer requests.

Look for:

- Feature requests
- Product fixes
- Service improvements
- Support needs
- Complaints requiring follow-up

For each request provide:

1. Request
2. Evidence from the feedback
3. Priority: HIGH, MEDIUM, or LOW

If there are no clear requests, say:

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
You are a customer success decision system.

Evaluate the customer's overall health using the analysis below.

ORIGINAL FEEDBACK:
{state.customer_feedback}

SENTIMENT:
{state.sentiment_analysis}

THEMES:
{state.feedback_themes}

CUSTOMER REQUESTS:
{state.customer_requests}

Classify customer health as exactly one of:

NORMAL
URGENT

Use URGENT when the feedback indicates significant customer risk,
such as:

- Strong dissatisfaction
- Repeated serious complaints
- Threat of cancellation or churn
- Critical product/service failure
- Unresolved issue causing major business impact
- Escalation requiring immediate customer attention

Use NORMAL when the feedback represents normal product/service
improvement opportunities without immediate customer risk.

Return ONLY valid JSON:

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
You are a customer experience analyst.

Create an improvement summary from the following customer feedback.

FEEDBACK:
{state.customer_feedback}

SENTIMENT:
{state.sentiment_analysis}

THEMES:
{state.feedback_themes}

CUSTOMER REQUESTS:
{state.customer_requests}

Create a concise summary containing:

1. Overall sentiment
2. Main recurring themes
3. Customer requests
4. Improvement opportunities
5. Suggested next steps

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
You are a customer success manager handling an urgent customer issue.

Create an urgent customer action plan.

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

Create an action plan containing:

1. Urgent issue
2. Evidence from customer feedback
3. Immediate action required
4. Team or function that should investigate
5. Customer communication needed
6. Follow-up recommendation

Keep the plan practical and concise.

Do not invent facts that are not supported by the feedback.
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
