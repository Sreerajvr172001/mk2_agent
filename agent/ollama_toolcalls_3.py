"""
MK2 Agentic AI — Step 1 v3: Stateful ReAct Agent with Native Tool Calling
--------------------------------------------------------------------------
Changes from v2:
  - SceneSimulator: stateful scene that changes after robot actions
  - Retry mechanism with MAX_RETRIES counter and safe STOP fallback
  - num_gpu=0: LLM runs on CPU, full VRAM free for YOLO TensorRT
  - Direct message append (no reconstruction) for correct tool-call context
  - Retry prompt is explicit about which tool to call

Run:
  python3 agent_decision_3.py

Change INITIAL_SCENE to test all branches:
  "clear"         → should MOVE_FORWARD → goal_reached
  "person_ahead"  → should STOP immediately
  "person_left"   → should TURN_RIGHT x2 → MOVE_FORWARD → goal_reached
  "person_right"  → should TURN_LEFT x2  → MOVE_FORWARD → goal_reached
"""

import json
import ollama

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

INITIAL_SCENE = "clear"   # change this to test other scenarios
MAX_STEPS     = 15
MAX_RETRIES   = 2

# ── SCENE SIMULATOR ───────────────────────────────────────────────────────────

class SceneSimulator:
    """
    Stateful scene simulator. Tracks robot actions and updates the
    scene accordingly so the agent doesn't loop forever.

    In Step 2 (ROS2), this entire class is replaced by live data
    from /detections. The tool function signatures stay identical.
    """

    def __init__(self, initial_scene: str):
        self.scene      = initial_scene
        self.turn_count = 0

    def get_detections(self) -> dict:
        # After enough turns, the obstacle clears
        if self.scene in ("person_left", "person_right"):
            if self.turn_count >= 2:
                self.scene = "clear"

        scenes = {
            "clear": {
                "detections": [],
                "frame_width": 320,
                "frame_height": 240,
            },
            "person_ahead": {
                "detections": [
                    {
                        "label": "person",
                        "confidence": 0.94,
                        "position_classification": "center",
                        "size_classification": "large_close",
                        "is_direct_obstacle": True,
                    }
                ],
                "frame_width": 320,
                "frame_height": 240,
            },
            "person_left": {
                "detections": [
                    {
                        "label": "person",
                        "confidence": 0.94,
                        "position_classification": "left",
                        "size_classification": "large_close",
                        "is_direct_obstacle": False,
                    }
                ],
                "frame_width": 320,
                "frame_height": 240,
            },
            "person_right": {
                "detections": [
                    {
                        "label": "person",
                        "confidence": 0.91,
                        "position_classification": "right",
                        "size_classification": "medium",
                        "is_direct_obstacle": False,
                    }
                ],
                "frame_width": 320,
                "frame_height": 240,
            },
        }
        return scenes.get(self.scene, scenes["clear"])

    def on_turn(self):
        self.turn_count += 1

    def on_move_forward(self):
        # Moving forward resets turn count
        self.turn_count = 0


simulator = SceneSimulator(INITIAL_SCENE)

# ── TOOL IMPLEMENTATIONS ──────────────────────────────────────────────────────

def get_detections() -> dict:
    result = simulator.get_detections()
    return result

def move_forward() -> dict:
    print("    >>> [ROBOT] Moving forward")
    simulator.on_move_forward()
    return {"status": "ok", "action_taken": "move_forward"}

def stop_robot() -> dict:
    print("    >>> [ROBOT] Stopping")
    return {"status": "ok", "action_taken": "stop"}

def turn_left() -> dict:
    print("    >>> [ROBOT] Turning left 30 degrees")
    simulator.on_turn()
    return {"status": "ok", "action_taken": "turn_left"}

def turn_right() -> dict:
    print("    >>> [ROBOT] Turning right 30 degrees")
    simulator.on_turn()
    return {"status": "ok", "action_taken": "turn_right"}

def goal_reached() -> dict:
    print("    >>> [ROBOT] Goal reached!")
    return {"status": "ok", "action_taken": "goal_reached"}

TOOL_REGISTRY = {
    "get_detections": get_detections,
    "move_forward":   move_forward,
    "stop_robot":     stop_robot,
    "turn_left":      turn_left,
    "turn_right":     turn_right,
    "goal_reached":   goal_reached,
}

# ── TOOL DEFINITIONS (Ollama schema) ─────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_detections",
            "description": (
                "Get current YOLO object detections from the robot camera. "
                "Returns a list of detected objects with label, confidence, "
                "position_classification (left/center/right), "
                "size_classification (large_close/medium/small_far), "
                "and is_direct_obstacle (true if blocking the path ahead)."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_forward",
            "description": "Move the robot forward for one step. Only call this when get_detections() shows no obstacles.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_robot",
            "description": "Stop the robot immediately. Call this when is_direct_obstacle=true.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "turn_left",
            "description": "Rotate the robot left by 30 degrees. Call this when an obstacle is on the right.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "turn_right",
            "description": "Rotate the robot right by 30 degrees. Call this when an obstacle is on the left.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "goal_reached",
            "description": "Call this after successfully moving forward. Signals that navigation is complete.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are the navigation agent for MK2, an autonomous mobile robot.
Your goal is to navigate forward safely, avoiding all obstacles.

You have these tools:
  get_detections() — check what the camera currently sees
  move_forward()   — move the robot forward one step
  stop_robot()     — stop immediately if path is blocked
  turn_left()      — rotate left 30 degrees (use when obstacle is on the RIGHT)
  turn_right()     — rotate right 30 degrees (use when obstacle is on the LEFT)
  goal_reached()   — call this after successfully moving forward

Your reasoning loop — follow this exactly:

  STEP 1: Always call get_detections() first to observe the scene.

  STEP 2: Read the detections carefully:
    - If any detection has is_direct_obstacle=true:
        Call stop_robot(). Navigation is blocked.
    - If a detection has position_classification="left":
        Call turn_right(). Obstacle is on the left, turn away from it.
    - If a detection has position_classification="right":
        Call turn_left(). Obstacle is on the right, turn away from it.
    - If detections list is empty, or all detections have
      size_classification="small_far":
        Call move_forward(). Path is clear.

  STEP 3: After move_forward(), call goal_reached().

  STEP 4: After any turn, call get_detections() again to re-check the scene.

IMPORTANT: You must always call a tool. Never respond with text only.
"""

# ── AGENT LOOP ────────────────────────────────────────────────────────────────

def run_agent(goal: str):
    print(f"\n{'=' * 60}")
    print(f"  GOAL: {goal}")
    print(f"  INITIAL SCENE: {INITIAL_SCENE}")
    print(f"{'=' * 60}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": goal},
    ]

    step = 0
    while step < MAX_STEPS:
        step += 1
        print(f"\n--- Step {step} ---")

        # ── Inner retry loop ──────────────────────────────────────────────────
        retries = 0
        while True:
            response = ollama.chat(
                model="qwen2.5:3b",
                messages=messages,
                tools=TOOLS,
                options={
                    "temperature": 0,
                    "num_gpu": 0,   # LLM on CPU — keeps full VRAM for YOLO
                },
            )

            message = response["message"]

            # Model produced text instead of a tool call
            if not message.get("tool_calls"):
                text = message.get("content", "").strip()
                print(f"[AGENT TEXT]: {text}")

                retries += 1
                if retries > MAX_RETRIES:
                    print("[FATAL]: Model refused tool calls after max retries. Stopping robot.")
                    stop_robot()
                    return

                print(f"[RETRY {retries}/{MAX_RETRIES}]: Forcing tool call...")

                # Keep the model's reasoning in context, then demand a tool call
                messages.append(message)
                messages.append({
                    "role": "user",
                    "content": (
                        "You reasoned correctly. Now you must call a tool. "
                        "Do not write text. Call the tool that matches your reasoning above. "
                        "If you said turn_right, call turn_right(). "
                        "If you said turn_left, call turn_left(). "
                        "If you said move_forward, call move_forward(). "
                        "If you said stop, call stop_robot()."
                    ),
                })
                continue  # retry

            # Got a valid tool call — break out of retry loop
            break

        # ── Execute tool calls ────────────────────────────────────────────────
        # Append raw message object directly — do not reconstruct
        messages.append(message)

        terminal = False
        for tool_call in message["tool_calls"]:
            tool_name = tool_call["function"]["name"]
            tool_args = tool_call["function"]["arguments"]

            print(f"[TOOL CALL]: {tool_name}({tool_args})")

            if tool_name in TOOL_REGISTRY:
                result = TOOL_REGISTRY[tool_name](**tool_args)
            else:
                result = {"error": f"Unknown tool '{tool_name}'"}
                print(f"[ERROR]: {result}")

            print(f"[RESULT]: {json.dumps(result)}")

            # Append tool result
            messages.append({
                "role": "tool",
                "content": json.dumps(result),
            })

            # Check for terminal actions
            if tool_name in ("goal_reached", "stop_robot"):
                terminal = True

        if terminal:
            print(f"\n[AGENT LOOP COMPLETE]: terminated by {tool_name}")
            return

    print(f"\n[AGENT LOOP]: max steps ({MAX_STEPS}) reached without completion")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_agent("Navigate forward safely, avoiding any obstacles in your path.")