import asyncio
import os
from dataclasses import dataclass, asdict
from typing import Any

from dotenv import load_dotenv
from github import Github, Auth, GithubException
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow, AgentOutput, ToolCall, ToolCallResult
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI

load_dotenv(override=True)

MODEL = "gpt-4o-mini"

llm = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
    model=MODEL
)

auth = Auth.Token(os.getenv("GITHUB_TOKEN"))
git = Github(auth=auth)
repository = os.getenv("REPOSITORY")
pr_no = os.getenv("PR_NUMBER")


@dataclass()
class PR:
    author: str
    title: str
    state: str
    body: str
    diff_url: str
    commit_shas: list[str]


@dataclass()
class ChangedFile:
    filename: str
    status: str
    additions: int
    deletions: int
    changes: int
    patch: str


def get_pr_details(pr_no: int) -> dict[str, Any]:
    """Function tool that returns pr details for the given number"""
    try:
        repo = git.get_repo(repository)
        pr = repo.get_pull(pr_no)
        commit_shas = []
        for commit in pr.get_commits():
            commit_shas.append(commit.sha)
        pull_request = PR(
            author=pr.user.login,
            title=pr.title,
            body=pr.body or "",
            state=pr.state,
            diff_url=pr.diff_url,
            commit_shas=commit_shas
        )
        return asdict(pull_request)
    except (GithubException, Exception):
        return {}


def get_pr_commit_details(head_sha: str) -> list[dict[str, Any]]:
    """Function tool that returns pr commit details for the given head_sha"""
    try:
        repo = git.get_repo(repository)
        commit = repo.get_commit(head_sha)
        changed_files: list[ChangedFile] = []
        for f in commit.files:
            changed_files.append(ChangedFile(
                filename=f.filename,
                status=f.status,
                additions=f.additions,
                deletions=f.deletions,
                changes=f.changes,
                patch=f.patch or ""
            ))
        return [asdict(cf) for cf in changed_files]
    except (GithubException, Exception):
        return []


def get_file_contents(path: str) -> str:
    """Function tool that returns the string contents of a file on GitHub"""
    try:
        repo = git.get_repo(repository)
        file = repo.get_contents(path)
        raw_contents = getattr(file, "decoded_content", b"")
        return raw_contents.decode("utf-8", errors="replace")
    except (GithubException, Exception):
        return ""


async def add_summary_to_state(ctx: Context, summary: str) -> str:
    """Function tool that adds the PR review summary to the state"""
    async with ctx.store.edit_state() as ctx_state:
        ctx_state["state"]["gathered_contexts"] = summary
    return f'Gathered contexts set to "{summary}"'


async def add_draft_comment_to_state(ctx: Context, draft_comment: str) -> str:
    """Function tool that adds a draft comment to the state"""
    async with ctx.store.edit_state() as ctx_state:
        ctx_state["state"]["draft_comment"] = draft_comment
    return f'Draft comment set to "{draft_comment}"'


async def add_final_review_to_state(ctx: Context, final_review: str) -> str:
    """Function tool that adds the final review of the PR to the state"""
    async with ctx.store.edit_state() as ctx_state:
        ctx_state["state"]["final_review"] = final_review
    return f'Final review set to "{final_review}"'


def post_comment(pr_no: int, comment: str) -> str:
    """Function tool that posts a comment to a PR on GitHub"""
    try:
        repo = git.get_repo(repository)
        pr = repo.get_pull(pr_no)
        pr.create_review(body=comment)
        return "Successfully posted a comment on GitHub"
    except (GithubException, Exception) as e:
        return f"Error posting PR comment to GitHub: {e}"


ctx_agent_tool_functions = [
    get_pr_details,
    get_pr_commit_details,
    get_file_contents,
    add_summary_to_state
]
ctx_agent_tools = [FunctionTool.from_defaults(fn) for fn in ctx_agent_tool_functions]

ctx_agent_system_prompt = """
You are the context gathering ContextAgent. When gathering context, you MUST gather:
    - The details: author, title, body, diff_url, state and head_sha;
    - Changed files;
    - Any requested for files;
Once you gather the requested info, you MUST hand control back to the CommentorAgent.
"""
context_agent = FunctionAgent(
    name="ContextAgent",
    description="Gathers context on a specific Pull Request and stores it in state",
    llm=llm,
    system_prompt=ctx_agent_system_prompt,
    tools=ctx_agent_tools,
    can_handoff_to=["CommentorAgent"]
)

commentor_agent_tools = [FunctionTool.from_defaults(add_draft_comment_to_state)]
commentor_agent_system_prompt = """
You are the CommentorAgent that writes review comments for pull requests as a human reviewer would. 
Ensure you do the following for a thorough review: 
    - Request for the PR details, changed files, and any other repo files you may need from the ContextAgent. 
    - Once you have asked for all the necessary information, write a good ~200-300 word review in markdown format detailing:
        - What is good about the PR?
        - Did the author follow ALL contribution rules? What is missing?
        - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this.
        - Are new endpoints documented? - use the diff to determine this. 
        - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement.
    - If you need any additional details, you must hand off to the ContextAgent.
    - You should directly address the author. So your comments should sound like:
 "Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?"
You MUST hand off to the ReviewAndPostingAgent once you are done drafting a review.
"""
commentor_agent = FunctionAgent(
    name="CommentorAgent",
    description="Uses the context gathered by the ContextAgent to draft a Pull Request review comment",
    llm=llm,
    system_prompt=commentor_agent_system_prompt,
    tools=commentor_agent_tools,
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"]
)

review_and_posting_agent_tools = [FunctionTool.from_defaults(add_final_review_to_state), FunctionTool.from_defaults(post_comment)]
review_and_posting_agent_system_prompt = """
You are the ReviewAndPostingAgent. You must use the CommentorAgent to create a review comment. 
Once a review is generated, you need to run a final check and post it to GitHub.
The review MUST:
   - Be a ~200-300 word review in markdown format.
   - Specify what is good about the PR:
   - Did the author follow ALL contribution rules? What is missing?
   - Are there notes on test availability for new functionality? If there are new models, are there migrations for them?
   - Are there notes on whether new endpoints were documented?
   - Are there suggestions on which lines could be improved upon? Are these lines quoted?
If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns.
When you are satisfied, post the review to GitHub.  
"""
review_and_posting_agent = FunctionAgent(
    name="ReviewAndPostingAgent",
    description="Reviews the draft_comment and requests rewrites if necessary, before finally posting it to GitHub",
    llm=llm,
    system_prompt=review_and_posting_agent_system_prompt,
    tools=review_and_posting_agent_tools,
    can_handoff_to=["CommentorAgent"]
)

workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_and_posting_agent],
    root_agent=review_and_posting_agent.name,
    initial_state={
        "gathered_contexts": "",
        "draft_comment": "",
        "final_review": ""
    }
)


async def main():
    query = f"Write a review for PR: {pr_no}"
    prompt = RichPromptTemplate(query)

    ctx = Context(workflow_agent)
    handler = workflow_agent.run(prompt.format(), ctx=ctx)

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\n\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools:", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")


if __name__ == "__main__":
    asyncio.run(main())
    git.close()
