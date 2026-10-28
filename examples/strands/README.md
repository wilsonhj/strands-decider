# Strands decider inside a Strands agent

A worked example of strands decider inside a [Strands](https://github.com/strands-agents/sdk-python)
agent. The agent runs locally on the default Bedrock model plus the locally served decider. It
runs from a clone of the repository and expects a server on port 8099 unless
`STRANDS_DECIDER_URL` is set.

The example's client talks plain HTTP with no authentication and no AWS SigV4 signing, so it
targets a decider server you run yourself (locally or on a private network). It cannot call a
SageMaker endpoint, which requires signed requests; adding signing is left to you. Start the
server with `--strict-window`: a prompt that does not fit the model's window is then refused
rather than shortened, and the gate blocks the call instead of judging a cut-off one.

**Prerequisites:** Strands drives the agent with an LLM call to Amazon Bedrock, so you need
AWS credentials with Bedrock access in your environment before you run the example. See the
[Strands Bedrock model provider docs](https://strandsagents.com/latest/documentation/docs/user-guide/concepts/model-providers/amazon-bedrock/)
for the supported credential options.

```bash
git clone https://github.com/strands-labs/strands-decider && cd strands-decider
pip install -e . strands-agents
strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8099 --strict-window

python examples/strands/tool_call_intervention.py
```

On an Apple-silicon Mac, `pip install -e ".[mlx]" strands-agents` and `--device mlx` on the
`serve` line run the decider through MLX
([Serving on a Mac with MLX](../../docs/inference.md#serving-on-a-mac-with-mlx)).

The scenario is deliberately small. The agent has one `get_weather` tool and a system prompt
that makes it eager, and the user asks "*What's the weather?*" without saying where. The agent
guesses a city and calls the tool anyway. Before that call runs, strands decider reads the
conversation and the proposed call and answers two narrow yes/no questions about it:

1. Are the argument values grounded in facts the user actually provided?
2. Is it premature to call this tool now, before clarifying with the user?

A few lines of plain Python turn those probabilities into a decision, and the agent goes back
to ask which city you meant instead of confidently reporting the weather somewhere nobody
mentioned.

The policy fails closed. The call runs only when the decider is confident that the arguments are
grounded and confident that the call is not premature. A probability in the `UNCERTAIN` band
(0.35 to 0.65 here) is neither, so it sends the model back to the user with `Guide`. If the
server cannot be reached, refuses the request, or returns something that is not a probability,
the call is blocked with `Deny`, as is a call too large to fit the model's window. The proposed
call is placed first in the state and the conversation after it, cut from its old end to fit, so
a long conversation never costs the gate the tool name and arguments it is judging. Text from
the conversation and the arguments is untrusted: markup that looks like the prompt's own tags is
defused before it is sent.

The seam is Strands' intervention system. You write an `InterventionHandler` with a
`before_tool_call` method and pass it to the agent, and it runs before any tool executes. What
you return is a typed action rather than text: the example uses `Proceed` to let the call run
and `Guide` to hand the model back its turn with feedback, which is what makes this different
from a hard allow/deny gate — `Guide` corrects the course instead of refusing. The handler is
an ordinary Python class and Strands has no opinion about what goes inside it, so the same
shape holds whether you call a decision model, a policy engine, or another agent.

Why classify instead of letting the agent check itself? The gate is a local forward pass in a
couple of hundred milliseconds, the conversation never leaves the machine, and each verdict is
a number your `if` statements branch on rather than prose the agent can negotiate with.

> This example is an illustration rather than a recommendation: the questions, the thresholds
> and the policy were all picked by hand. The point is that a decision this cheap can sit in a
> path where an LLM call never could.
