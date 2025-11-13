from dotenv import load_dotenv
import os
import asyncio
# from IPython.display import display, Image as IPImage  # Only works in Jupyter
import base64

load_dotenv()

api_key = os.environ['GOOGLE_API_KEY']


import uuid
from google.genai import types

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from google.adk.apps.app import App, ResumabilityConfig
from google.adk.tools.function_tool import FunctionTool



retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

BULK_REQUEST_THRESHOLD = 1

mcp_image_server = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="mcp-server-gemini-image-generator",
            args=[],
            env={
                "GEMINI_API_KEY": os.environ['GOOGLE_API_KEY'],
                "OUTPUT_IMAGE_PATH": os.environ['OUTPUT_IMAGE_PATH']
            },
            tool_filter=["generate_image_from_text"],
        ),
        timeout=30,
    )
)

def create_image_order(num_images: int, prompt: str, tool_context: ToolContext) -> dict:
    if num_images <= BULK_REQUEST_THRESHOLD:
        return{
            "status": "approved",
            "order_id": f"ORD-{num_images}-AUTO",
            "num_images": num_images,
            "image object": prompt,
            "message" : f"Order auto-approved: {num_images} images"
        }
    if not tool_context.tool_confirmation:
        tool_context.request_confirmation(
            hint=f"Large order: {num_images} images. Do you want to approve?",
            payload={"num_images": num_images}
        )
        return {
            "status": "pending",
            "message": f"Order for {num_images} images requires approval"
        }
    if tool_context.tool_confirmation.confirmed:
        return{
            "status": "approved",
            "order_id": f"ORD-{num_images}-HUMAN",
            "num_images": num_images,
            "image object": prompt,
            "message": f"Order approved: {num_images} images of {prompt}"
        }
    else:
        return{
            "status": "rejected",
            "message": f"Order rejected: {num_images} images of {prompt}"
        }    

image_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="image_agent",
    instruction="""You are a photograph coordinator assistant.
  
  When users request to generate images:
   1. Use the create_image_order tool with the number of images given
   2. If the order status is 'pending', inform the user that approval is required and wait for explicit approval
   3. "If the order status is 'approved', call the generate_image_from_text tool multiple times (once for each approved image) to generate all requested images with the given prompt
   4. After receiving the final result, provide a clear summary including:
      - Order status (approved/rejected)
      - Order ID (if available)
      - Number of images and image prompt
   5. Keep responses concise but informative""",
    tools=[FunctionTool(create_image_order), mcp_image_server]
)

image_generation_app = App(
    name="image_coordinator",
    root_agent=image_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)

session_service = InMemorySessionService()

image_runner = Runner(
    app=image_generation_app,
    session_service=session_service
)

def check_for_approval(events):
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if (
                    part.function_call
                    and part.function_call.name == "adk_request_confirmation"
                ):
                    return {
                        "approval_id": part.function_call.id,
                        "invocation_id": event.invocation_id,
                    }
    return None

def print_agent_response(events):
    """Print agent's text responses from events."""
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"Agent > {part.text}") 

                    
def create_approval_response(approval_info, approved):
    """Create approval response message."""
    confirmation_response = types.FunctionResponse(
        id=approval_info["approval_id"],
        name="adk_request_confirmation",
        response={"confirmed": approved},
    )
    return types.Content(
        role="user", parts=[types.Part(function_response=confirmation_response)]
    )

async def run_imaging_workflow(query: str, auto_approve: bool = True):
    """Runs a shipping workflow with approval handling.

    Args:
        query: User's shipping request
        auto_approve: Whether to auto-approve large orders (simulates human decision)
    """

    print(f"\n{'='*60}")
    print(f"User > {query}\n")

    # Generate unique session ID
    session_id = f"order_{uuid.uuid4().hex[:8]}"

    # Create session
    await session_service.create_session(
        app_name="image_coordinator", user_id="test_user", session_id=session_id
    )

    query_content = types.Content(role="user", parts=[types.Part(text=query)])
    events = []

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------
    # STEP 1: Send initial request to the Agent. If num_containers > 5, the Agent returns the special `adk_request_confirmation` event
    async for event in image_runner.run_async(
        user_id="test_user", session_id=session_id, new_message=query_content
    ):
        events.append(event)

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------
    # STEP 2: Loop through all the events generated and check if `adk_request_confirmation` is present.
    approval_info = check_for_approval(events)

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------
    # STEP 3: If the event is present, it's a large order - HANDLE APPROVAL WORKFLOW
    if approval_info:
        print(f"⏸️  Pausing for approval...")
        human_decision = input("Would you like to approve of this order? (y/n)")
        if (human_decision.lower()=='y' or human_decision.lower()=='yes'):
            auto_approve = True
        else:
            auto_approve = False
        print(f"🤔 Human Decision: {'APPROVE ✅' if auto_approve else 'REJECT ❌'}\n")

        # PATH A: Resume the agent by calling run_async() again with the approval decision
        async for event in image_runner.run_async(
            user_id="test_user",
            session_id=session_id,
            new_message=create_approval_response(
                approval_info, auto_approve
            ),  # Send human decision here
            invocation_id=approval_info[
                "invocation_id"
            ],  # Critical: same invocation_id tells ADK to RESUME
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(f"Agent > {part.text}")

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------
    else:
        # PATH B: If the `adk_request_confirmation` is not present - no approval needed - order completed immediately.
        print_agent_response(events)

    print(f"{'='*60}\n")


async def main():
    """Main entry point for the script."""
    await run_imaging_workflow("Create 1 images of a photorealistic hand puppet with yarn for hair, 2 buttons for eyes, make it yellow and orange striped with a simple bedroom background blurred out")


if __name__ == "__main__":
    asyncio.run(main())
