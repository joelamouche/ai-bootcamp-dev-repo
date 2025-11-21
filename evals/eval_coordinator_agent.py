from langsmith import Client

from api.agent.agents import coordinator_agent
from api.agent.graph import State
from time import sleep


ACC_THRESHOLD = 0.6
SLEEP_TIME = 5

client = Client()


def next_agent_evaluator(run, example):

    next_agent_match = run.outputs["coordinator_agent"]["next_agent"] == example.outputs["next_agent"] 
    final_answer_match = run.outputs["coordinator_agent"]["final_answer"] == example.outputs["coordinator_final_answer"]
    
    return next_agent_match and final_answer_match



results = client.evaluate(
    lambda x: coordinator_agent(State(messages=x["messages"])),
    data="coordinator-eval-dataset",
    evaluators=[
        next_agent_evaluator
    ],
    experiment_prefix="coordinator-eval-dataset"
)


print(f"Sleeping for {SLEEP_TIME} seconds...")
sleep(SLEEP_TIME)


results_resp = client.read_project(
    project_name = results.experiment_name,
    include_stats = True
)

avg_metric = results_resp.feedback_stats.get("next_agent_evaluator").get("avg")
errors = results_resp.feedback_stats.get("next_agent_evaluator").get("errors")

if avg_metric >= ACC_THRESHOLD:
    output_messge = f"✅ Success: {avg_metric}"
else:
    output_messge = f"❌ Failure: {avg_metric}"

if errors > 0:
    raise AssertionError(f"There were {errors} errors while running evaluations")
elif avg_metric >= ACC_THRESHOLD:
    print(output_messge, flush=True)
else:
    raise AssertionError(output_messge)