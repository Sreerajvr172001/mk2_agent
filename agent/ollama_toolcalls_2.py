# agent_decision_3.py — ReAct agent with Ollama native tool calling

import json
import ollama

# ── TOOL DEFINITIONS ──────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_detections",
            "description": "Get current YOLO object detections from the robot camera. Returns a list of detected objects with their labels, confidence scores, position (left/center/right) and size (large_close/medium/small_far).",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move_forward",
            "description": "Move the robot forward for one second.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "stop_robot",
            "description": "Stop the robot immediately. Use when a direct obstacle is detected ahead.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "turn_left",
            "description": "Rotate the robot left by 30 degrees.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "turn_right",
            "description": "Rotate the robot right by 30 degrees.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "goal_reached",
            "description": "Call this when the navigation goal has been completed successfully.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
]

# ── SIMULATED TOOL IMPLEMENTATIONS ────────────────────────────────────────────
# These are stubs that return fake data for now.
# In Step 2 (ROS2 integration), each of these will call your actual
# ROS2 topics and services instead.

# Change this between runs to simulate different camera states
SIMULATED_SCENE = "person_left"  # options: "clear", "person_ahead", "person_left", "person_right"

def get_detections() -> dict:
    scenes = {
        "clear": {
            "detections": [],
            "frame_width": 320,
            "frame_height": 240
        },
        "person_ahead": {
            "detections": [
                {
                    "label": "person",
                    "confidence": 0.94,
                    "position_classification": "center",
                    "size_classification": "large_close",
                    "is_direct_obstacle": True
                }
            ],
            "frame_width": 320,
            "frame_height": 240
        },
        "person_left": {
            "detections": [
                {
                    "label": "person",
                    "confidence": 0.94,
                    "position_classification": "left",
                    "size_classification": "large_close",
                    "is_direct_obstacle": False
                }
            ],
            "frame_width": 320,
            "frame_height": 240
        },
        "person_right": {
            "detections": [
                {
                    "label": "person",
                    "confidence": 0.91,
                    "position_classification": "right",
                    "size_classification": "medium",
                    "is_direct_obstacle": False
                }
            ],
            "frame_width": 320,
            "frame_height": 240
        },
    }
    return scenes.get(SIMULATED_SCENE, scenes["clear"])

def move_forward() -> dict:
    print("    >>> [ROBOT] Moving forward")
    return {"status": "ok", "action_taken": "move_forward"}

def stop_robot() -> dict:
    print("    >>> [ROBOT] Stopping")
    return {"status": "ok", "action_taken": "stop"}

def turn_left() -> dict:
    print("    >>> [ROBOT] Turning left 30 degrees")
    return {"status": "ok", "action_taken": "turn_left"}

def turn_right() -> dict:
    print("    >>> [ROBOT] Turning right 30 degrees")
    return {"status": "ok", "action_taken": "turn_right"}

def goal_reached() -> dict:
    print("    >>> [ROBOT] Goal reached!")
    return {"status": "ok", "action_taken": "goal_reached"}

# Map tool names to actual functions
TOOL_REGISTRY = {
    "get_detections": get_detections,
    "move_forward":   move_forward,
    "stop_robot":     stop_robot,
    "turn_left":      turn_left,
    "turn_right":     turn_right,
    "goal_reached":   goal_reached,
}

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are the navigation agent for MK2, an autonomous mobile robot.

Your goal is to navigate forward safely. You must avoid obstacles.

You have these tools available:
  get_detections() — check what the camera currently sees
  move_forward()   — move the robot forward one step
  stop_robot()     — stop immediately if blocked
  turn_left()      — rotate left to avoid obstacle on the right
  turn_right()     — rotate right to avoid obstacle on the left
  goal_reached()   — call this after successfully moving forward

Your reasoning loop:
  1. Always call get_detections() first to observe the scene.
  2. If is_direct_obstacle=true → call stop_robot().
  3. If obstacle is on the left → call turn_right().
  4. If obstacle is on the right → call turn_left().
  5. If no obstacles → call move_forward(), then goal_reached().

Always observe before acting. Never skip get_detections().
"""

# ── AGENT LOOP ────────────────────────────────────────────────────────────────

def run_agent(goal: str, max_steps: int = 10):
    print(f"\n{'=' * 60}")
    print(f"  GOAL: {goal}")
    print(f"  SCENE: {SIMULATED_SCENE}")
    print(f"{'=' * 60}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": goal}
    ]

    for step in range(max_steps):
        print(f"\n--- Step {step + 1} ---")

        response = ollama.chat(
            model="llama3.2:3b",
            messages=messages,
            tools=TOOLS,
            options={"temperature": 0, "num_gpu": 0}
        )

        message = response["message"]

        # ── If model produced text instead of tool call, force it back ────────
        if not message.get("tool_calls"):
            print(f"[AGENT TEXT]: {message.get('content', '')}")
            print("[RETRY]: Model produced text — forcing tool call...")

            # Append what the model said, then demand a tool call explicitly
            messages.append(message)
            messages.append({
                "role": "user",
                "content": "You must call a tool now. Do not write text. "
                           "Call the appropriate tool based on your reasoning above."
            })
            continue  # retry this step

        # ── Normal tool call path ─────────────────────────────────────────────
        # Append raw message object — do NOT reconstruct it
        messages.append(message)

        for tool_call in message["tool_calls"]:
            tool_name = tool_call["function"]["name"]
            tool_args = tool_call["function"]["arguments"]

            print(f"[TOOL CALL]: {tool_name}({tool_args})")

            if tool_name in TOOL_REGISTRY:
                result = TOOL_REGISTRY[tool_name](**tool_args)
            else:
                result = {"error": f"Unknown tool: {tool_name}"}

            print(f"[RESULT]: {json.dumps(result)}")

            # Append tool result — minimal format, no name field
            messages.append({
                "role": "tool",
                "content": json.dumps(result),
            })

            if tool_name in ("goal_reached", "stop_robot"):
                print(f"\n[AGENT LOOP COMPLETE]: terminated by {tool_name}")
                return

    print("\n[AGENT LOOP]: max steps reached")

if __name__ == "__main__":
    run_agent("Navigate forward safely, avoiding any obstacles in your path.")