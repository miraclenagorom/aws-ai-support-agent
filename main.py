"""
Customer Support AI Agent

Configure via environment variables (see .env.example), then run locally:
  uv run main.py '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'

Deploy to AgentCore:
  agentcore deploy

Invoke deployed agent:
  agentcore invoke '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import shutil
from contextlib import ExitStack
from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory import MemoryClient
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamable_http_client
import argparse
import json
import os
import asyncio
import boto3
from strands.hooks import (
    HookProvider, AfterInvocationEvent, HookRegistry, MessageAddedEvent,
)
import logging
import uuid
from typing import Dict
from bedrock_agentcore.tools.code_interpreter_client import code_session
from strands_tools.browser import AgentCoreBrowser


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("CSAI_Agent")
logger.setLevel(logging.INFO)


def _fix_playwright_driver():
    """Zips built on Windows drop Unix exec bits; restore it on Playwright's node binary."""
    try:
        import playwright  # type: ignore
        node = os.path.join(os.path.dirname(
            playwright.__file__), "driver", "node")
        if os.path.exists(node) and not os.access(node, os.X_OK):
            try:
                os.chmod(node, 0o755)
            except OSError:
                tmp_node = "/tmp/playwright-node"
                shutil.copy(node, tmp_node)
                os.chmod(tmp_node, 0o755)
                os.environ["PLAYWRIGHT_NODEJS_PATH"] = tmp_node
    except Exception as e:
        logger.warning(f"Playwright driver fix skipped: {e}")


_fix_playwright_driver()

# 1 — App Initialisation ───────────────────────────────────────────────
app = BedrockAgentCoreApp()

# Suppress interactive tool-consent prompts (required in headless deployments).
os.environ["BYPASS_TOOL_CONSENT"] = "true"

# 2 — Configuration (from environment variables) ───────────────────────
GATEWAY_URL = os.environ.get("GATEWAY_URL", "")   # AgentCore Gateway MCP endpoint
KB_ID = os.environ.get("KB_ID", "")               # Bedrock Knowledge Base ID
REGION = os.environ.get("AWS_REGION", "us-east-1")
MEMORY_ID = os.environ.get("MEMORY_ID", "")       # AgentCore Memory ID

# 3 — Model and Clients ────────────────────────────────────────────────
model_id = "global.amazon.nova-2-lite-v1:0"

model = BedrockModel(model_id=model_id)
memory_client = MemoryClient(region_name=REGION)
_bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)

SYSTEM_PROMPT = """You are a helpful, professional customer support assistant \
for an e-commerce platform.

Use your tools:
- Order tracking and refund tools for order status and returns. Always call the \
tool rather than guessing order details.
- search_knowledge_base for product information, return policies, warranty, \
loyalty tiers, and order status definitions.
- calculate_loyalty_discount for any loyalty points or discount calculation.
- The browser tool when asked to visit a web page.

If a message begins with "Customer Context:", it contains remembered facts and \
preferences about this customer. Use them (for example their name and \
communication style) without repeating the block back to them.

Be accurate and concise. If you do not know something, say so."""


# 4 — Namespace Helper ─────────────────────────────────────────────────
def get_namespaces(mem_client: MemoryClient, memory_id: str) -> Dict:
    """Return a dict mapping strategy type → namespace template string."""
    strategies = mem_client.get_memory_strategies(memory_id)
    namespaces = {}
    for s in strategies:
        templates = s.get("namespaceTemplates") or s.get("namespaces") or []
        if templates:
            key = s.get("type") or s.get("memoryStrategyType") or s.get("name")
            namespaces[key] = templates[0]
    return namespaces


# 5 — Memory Hook ──────────────────────────────────────────────────────
class MemoryHook(HookProvider):
    """Long-term memory hook for the customer support agent."""

    def __init__(self, actor_id, session_id, memory_client, memory_id):
        self.actor_id = actor_id
        self.session_id = session_id
        self.memory_client = memory_client
        self.memory_id = memory_id
        self.namespaces = get_namespaces(memory_client, memory_id)
        self._original_query = None

    def retrieve_customer_context(self, event: MessageAddedEvent):
        """Retrieve relevant memories and prepend them to the user message."""
        messages = event.agent.messages
        if not messages or messages[-1].get("role") != "user":
            return
        content = messages[-1].get("content", [])
        if not content or "toolResult" in content[0] or "text" not in content[0]:
            return
        user_query = content[0]["text"]
        self._original_query = user_query
        try:
            all_context = []
            for context_type, template in self.namespaces.items():
                memories = self.memory_client.retrieve_memories(
                    memory_id=self.memory_id,
                    namespace=template.format(actorId=self.actor_id),
                    query=user_query,
                    top_k=5,
                )
                for m in memories:
                    if isinstance(m, dict):
                        text = m.get("content", {}).get("text", "").strip()
                        if text:
                            all_context.append(f"[{context_type}] {text}")
            if all_context:
                messages[-1]["content"][0]["text"] = (
                    "Customer Context:\n"
                    + "\n".join(all_context)
                    + f"\n\n{user_query}"
                )
        except Exception as e:
            print(f"Memory retrieval failed: {e}")

    def save_support_interaction(self, event: AfterInvocationEvent):
        """Save the completed turn to memory after the agent responds."""
        try:
            customer_query, agent_response = None, None
            for msg in reversed(event.agent.messages):
                content = msg.get("content", [])
                if not content or "text" not in content[0]:
                    continue
                if msg["role"] == "assistant" and agent_response is None:
                    agent_response = content[0]["text"]
                elif msg["role"] == "user" and "toolResult" not in content[0]:
                    customer_query = self._original_query or content[0]["text"]
                    break
            if customer_query and agent_response:
                self.memory_client.create_event(
                    memory_id=self.memory_id,
                    actor_id=self.actor_id,
                    session_id=self.session_id,
                    messages=[(customer_query, "USER"),
                              (agent_response, "ASSISTANT")],
                )
        except Exception as e:
            print(f"Memory save failed: {e}")

    def register_hooks(self, registry: HookRegistry) -> None:  # type: ignore
        """Register both memory callbacks."""
        registry.add_callback(
            MessageAddedEvent, self.retrieve_customer_context)
        registry.add_callback(AfterInvocationEvent,
                              self.save_support_interaction)


# 6 — Knowledge Base Tool ─────────────────────────────────────────────
@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the Amazon product catalog and support knowledge base.
    Use this for product specifications, return policies, warranty
    information, loyalty program details, and order status definitions.

    Args:
        query: The question or topic to search for

    Returns:
        Relevant information retrieved from the knowledge base
    """
    if not KB_ID:
        return "Knowledge base not configured."
    try:
        resp = _bedrock_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query},
        )
        results = resp.get("retrievalResults", [])
        if not results:
            return "No relevant information found in the knowledge base."
        chunks = [r["content"]["text"]
                  for r in results if r.get("content", {}).get("text")]
        return "\n---\n".join(chunks)
    except Exception as e:
        return f"Error searching knowledge base: {e}"


# 7 — Loyalty Discount Tool (Code Interpreter) ────────────────────────
@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard",
) -> str:
    """
    Calculate the loyalty discount for a customer order using the
    AgentCore Code Interpreter. Runs exact arithmetic in a secure sandbox.

    Args:
        loyalty_points:   Customer's current points balance
        tier:             Customer tier — Silver, Gold, or Platinum
        order_total:      Order total in USD
        product_category: standard, device, or fresh

    Returns:
        Full discount breakdown and final price
    """
    tier = tier.strip().title()
    code = (
        f"loyalty_points = {int(loyalty_points)}\n"
        f"tier = {tier!r}\n"
        f"order_total = {float(order_total)}\n"
        f"product_category = {product_category!r}\n"
    ) + '''
import json
earn_rates = {"standard": 1, "device": 2, "fresh": 5}
tier_rates = {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}
POINT_VALUE = 0.01  # $ per point (100 points = $1)

max_points = int(order_total * 0.5 / POINT_VALUE)
points_redeemed = (min(loyalty_points, max_points) // 500) * 500
points_discount = points_redeemed * POINT_VALUE
subtotal = order_total - points_discount
tier_rate = tier_rates.get(tier, 0.0)
tier_discount = subtotal * tier_rate
final_total = subtotal - tier_discount
total_savings = order_total - final_total
points_earned = int(final_total * earn_rates.get(product_category, 1))
remaining_points = loyalty_points - points_redeemed

print(json.dumps({
    "points_redeemed": points_redeemed,
    "points_discount": round(points_discount, 2),
    "tier": tier,
    "tier_discount_rate": tier_rate,
    "tier_discount": round(tier_discount, 2),
    "final_total": round(final_total, 2),
    "total_savings": round(total_savings, 2),
    "points_earned": points_earned,
    "remaining_points": remaining_points,
}))
'''

    try:
        with code_session(REGION) as code_client:
            response = code_client.invoke(
                "executeCode",
                {"code": code, "language": "python", "clearContext": True},
            )
            for event in response["stream"]:
                return json.dumps(event["result"])
        return "Code Interpreter returned no result."
    except Exception as e:
        rate = {"Silver": 0.0, "Gold": 0.10, "Platinum": 0.15}.get(tier, 0.0)
        discount = order_total * rate
        return json.dumps({
            "note": "Code Interpreter unavailable; tier discount only",
            "tier_discount_rate": rate,
            "tier_discount": round(discount, 2),
            "final_total": round(order_total - discount, 2),
            "error": str(e),
        })


# 8 — Agent Entrypoint ─────────────────────────────────────────────────
@app.entrypoint
async def invoke(payload, context=None):
    """
    Main handler called by AgentCore for every incoming request.

    Expected payload keys:
      prompt      (str, required) — the customer's message
      customer_id (str, optional) — unique customer identifier
      session_id  (str, optional) — session identifier; generated if absent
    """
    user_input = payload.get("prompt", "")
    actor_id = payload.get("customer_id", "guest")
    session_id = payload.get("session_id") or str(uuid.uuid4())

    try:
        memory_hook = MemoryHook(actor_id, session_id,
                                 memory_client, MEMORY_ID)
        agent_core_browser = AgentCoreBrowser(region=REGION)
        tools = [search_knowledge_base, calculate_loyalty_discount,
                 agent_core_browser.browser]

        mcp_client = MCPClient(lambda: streamable_http_client(GATEWAY_URL))
        gateway_note = ""
        with ExitStack() as stack:
            try:
                stack.enter_context(mcp_client)
                gateway_tools = mcp_client.list_tools_sync()
                tools.extend(gateway_tools)
                logger.info(
                    "Gateway connected successfully. Loaded %d tools.",
                    len(gateway_tools),
                )
            except TimeoutError:
                logger.exception("Gateway tool loading timed out")
                gateway_note = (
                    "\n\nNote: the order and refund services timed out, so I "
                    "could not use them for this request. Please try again shortly."
                )
            except ConnectionError:
                logger.exception("Gateway connection failed")
                gateway_note = (
                    "\n\nNote: I could not reach the order and refund services "
                    "right now. Please try again in a few minutes."
                )
            except Exception as exc:
                logger.exception("Gateway tool loading failed: %s", exc)
                gateway_note = (
                    "\n\nNote: the order and refund services are temporarily "
                    "unavailable, so I could not use them for this request. "
                    "Please try again later or contact support."
                )

            agent = Agent(
                model=model,
                tools=tools,
                hooks=[memory_hook],
                system_prompt=SYSTEM_PROMPT,
            )
            response = agent(user_input)
        return response.message["content"][0]["text"] + gateway_note
    except Exception as e:
        return f"Sorry, I ran into an error processing your request: {e}"


# ── CLI entry point ──────────────────────────────────────────────────────────
def main():
    """Run one invocation from the command line for local testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=str)
    args = parser.parse_args()
    response = asyncio.run(invoke(json.loads(args.payload)))
    print(response)


if __name__ == "__main__":
    app.run()
    # Uncomment the line below and comment app.run() for local CLI testing:
    # main()
