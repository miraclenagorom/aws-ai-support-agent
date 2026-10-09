# AI Customer Support Agent on AWS Bedrock AgentCore

## Project Overview
An AI customer support agent for an e-commerce scenario, built with the Strands Agents SDK and deployed on Amazon Bedrock AgentCore. It answers product and policy questions from a knowledge base, tracks orders and handles refunds through a gateway, calculates loyalty discounts in a sandbox, and remembers customer context across sessions.

## Architecture
| Component | Service | Purpose |
|---|---|---|
| Agent runtime | Bedrock AgentCore Runtime | Hosts and serves the agent |
| Model | Amazon Nova 2 Lite (Bedrock) | Reasoning and responses |
| Knowledge | Bedrock Knowledge Base (S3-backed) | Product catalog, return policy, warranty, loyalty tiers |
| Memory | AgentCore Memory | Long-term customer context, retrieved and saved via a hook |
| Tools | AgentCore Gateway (MCP) | Order tracking and refund tools |
| Calculations | AgentCore Code Interpreter | Exact loyalty discount arithmetic in a sandbox |
| Web access | AgentCore Browser | Visits web pages on request |

## Key Features
- **Knowledge retrieval:** `search_knowledge_base` queries the Bedrock Knowledge Base for product and policy answers.
- **Long-term memory:** a `MemoryHook` retrieves relevant customer memories before each turn and saves each completed interaction afterwards.
- **Sandboxed calculations:** `calculate_loyalty_discount` runs discount logic in the Code Interpreter, with a tier-only fallback if the sandbox is unavailable.
- **Gateway error handling:** timeouts, connection failures, and other gateway errors are caught in `invoke()`, and the agent still responds with a clear note instead of failing.

## Setup
1. Create your own AWS resources in one region: an S3 bucket with your product data, a Bedrock Knowledge Base, an AgentCore Memory, and an AgentCore Gateway.
2. Copy `.env.example` to `.env` and set the values, then export them in your shell:
   ```bash
   export GATEWAY_URL="..."
   export KB_ID="..."
   export MEMORY_ID="..."
   export AWS_REGION="us-east-1"
   ```
3. Run locally (uncomment `main()` at the bottom of `main.py` and comment out `app.run()`):
   ```bash
   uv run main.py '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'
   ```
4. Deploy and invoke:
   ```bash
   agentcore deploy
   agentcore invoke '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'
   ```

## Challenges Solved
- Recreated the Knowledge Base after a failed data source.
- Resolved a sandbox policy deny on memory creation.
- Fixed dependency packaging for the Linux runtime by using a Linux-compiled `requirements.txt`.
- Added Gateway error handling after a failed first review.

## Limitations
- Built on a starter template as part of a Udacity AWS AI course project.
- Built in a time-limited lab environment. The original AWS resources no longer exist, and no resource IDs or URLs are included in this repo.
- Running it requires your own AWS account, resources, and configuration.

## Author
**Miracle Maduabuchi**
[GitHub](https://github.com/miraclenagorom)
