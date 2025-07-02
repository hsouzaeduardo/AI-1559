from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import ListSortOrder

project = AIProjectClient(
    credential=DefaultAzureCredential(),
    endpoint="https://ai-dio-eastus-dev-001.services.ai.azure.com/api/projects/projeto-dio")

agent = project.agents.get_agent("asst_4F60Y6ctgz476SvZojeetJY8")

thread = project.agents.threads.get("thread_3DneuvbiqULBaGwUR2PGsfgG")

message = project.agents.messages.create(
    thread_id=thread.id,
    role="user",
    content="Hi Agent752"
)

run = project.agents.runs.create_and_process(
    thread_id=thread.id,
    agent_id=agent.id)

if run.status == "failed":
    print(f"Run failed: {run.last_error}")
else:
    messages = project.agents.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING)

    for message in messages:
        if message.text_messages:
            print(f"{message.role}: {message.text_messages[-1].text.value}")