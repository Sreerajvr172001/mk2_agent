"""
MK2 Agentic AI — Step 1: Standalone Agent Loop
------------------------------------------------
This script is the foundation of the agentic AI layer for MK2.
It takes a detection payload (simulating what the YOLO ROS2 node publishes
to /detections), sends it to llama3.2:3b via the Ollama Python client,
and receives a structured JSON navigation decision back.

No ROS2 here yet — that comes in Step 2. The goal of this step is to:
  1. Prove the LLM can reliably reason about detection data
  2. Learn how to enforce structured JSON output through prompting alone
  3. Establish the exact input/output contract that Step 2 will plug into

Prerequisites:
  pip install ollama
  ollama pull llama3.2:3b
"""

import json
import ollama

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
# This is the most important part of the agent. The system prompt defines:
#   - What role the LLM is playing (navigation reasoning agent for a robot)
#   - What input it receives (detection data from YOLO)
#   - Exactly what output format it must produce (strict JSON, no prose)
#   - The decision logic it should apply
#
# Why enforce JSON through the prompt rather than a library?
# Ollama supports a `format='json'` parameter that constrains output to valid
# JSON, but it doesn't enforce the *schema* (which keys exist, what values mean).
# The system prompt does both — it tells the model what the JSON must contain
# AND constrains it to produce only that. We use format='json' as a safety net
# on top of the prompt, not instead of it.
#
# Why not just use prose decisions?
# In Step 2 this output will be parsed by a ROS2 node and turned into robot
# commands. Prose like "you should probably stop" cannot be parsed. A dict with
# a defined 'action' key can be. The contract must be strict from day one.

SYSTEM_PROMPT = """
You are the navigation reasoning agent for MK2, an autonomous mobile robot.

Your job is to interpret object detection data from the robot's camera and
decide what action the robot should take next.

You will receive a JSON object describing what the YOLO vision model currently
detects in the robot's field of view. Each detection has:
  - "label": the class name of the detected object (e.g. "person", "chair")
  - "confidence": detection confidence from 0.0 to 1.0
  - "bbox_center_x": horizontal center of the bounding box (0 = left edge, 320 = right edge of a 320px wide frame)
  - "bbox_center_y": vertical center of the bounding box (0 = top edge, 240 = bottom edge of a 240px tall frame)
  - "bbox_width": width of the bounding box in pixels
  - "bbox_height": height of the bounding box in pixels

Frame size: 320x240 pixels. The robot moves forward along the center of the frame.

Decision rules you must apply:
  - If any detection with confidence > 0.7 occupies more than 25% of the frame width
    AND is centered within the middle third of the frame (bbox_center_x between 107 and 213),
    the object is a close obstacle directly ahead. Action: STOP.
  - If a high-confidence detection is present but offset to the left (bbox_center_x < 107),
    the path may be clear to the right. Action: TURN_RIGHT.
  - If a high-confidence detection is present but offset to the right (bbox_center_x > 213),
    the path may be clear to the left. Action: TURN_LEFT.
  - If no detections are present, or all detections are low confidence (< 0.7),
    the path is clear. Action: MOVE_FORWARD.
  - If the situation is ambiguous or multiple conflicting obstacles exist: Action: STOP.

You must respond with ONLY a valid JSON object. No explanation, no prose, no markdown.
The JSON must have exactly these keys:
  - "action": one of "MOVE_FORWARD", "STOP", "TURN_LEFT", "TURN_RIGHT"
  - "reason": one sentence explaining why this action was chosen
  - "primary_obstacle": the label of the most relevant detected object,
     or null if no relevant detection exists
  - "confidence_in_decision": your confidence in this decision from 0.0 to 1.0
"""

# ── SIMULATED DETECTION PAYLOAD ───────────────────────────────────────────────
# This mirrors exactly the data structure that will come from your ROS2
# /detections topic in Step 2. Each dict here corresponds to one Detection2D
# message from your yolo_detector.py — same fields, same coordinate system.
#
# In Step 2, this list will be built dynamically from the incoming
# Detection2DArray message instead of being hardcoded here.
#
# Try changing these values and re-running to see how the agent responds.
# Some scenarios to test:
#   - Person directly ahead, large and close: should STOP
#   - Person far to the left, small: should TURN_RIGHT
#   - Empty list: should MOVE_FORWARD
#   - Low confidence detections only: should MOVE_FORWARD

SIMULATED_DETECTIONS = [
    {
        "label": "person",
        "confidence": 0.94,
        "bbox_center_x": 50.0,   # near center horizontally — directly ahead
        "bbox_center_y": 50.0,   # near center vertically
        "bbox_width": 10.0,      # 110/320 = ~34% of frame width — large, close
        "bbox_height": 10.0,
    },
    {
        "label": "chair",
        "confidence": 0.61,       # below 0.7 threshold — should be ignored
        "bbox_center_x": 280.0,
        "bbox_center_y": 200.0,
        "bbox_width": 40.0,
        "bbox_height": 55.0,
    },
]


def format_detections_for_prompt(detections: list) -> str:
    """
    Converts the detection list into a JSON string to send as the user message.
    Keeping this as a separate function means Step 2 can call it with live
    detections from ROS2 with no changes to this logic.
    """
    if not detections:
        payload = {"detections": [], "frame_width": 320, "frame_height": 240}
    else:
        payload = {
            "detections": detections,
            "frame_width": 320,
            "frame_height": 240,
        }
    return json.dumps(payload, indent=2)


def query_agent(detections: list) -> dict:
    """
    Sends the detection payload to llama3.2:3b and returns the parsed decision.

    Parameters
    ----------
    detections : list of dicts, each representing one YOLO detection

    Returns
    -------
    dict with keys: action, reason, primary_obstacle, confidence_in_decision
    Returns an error dict if the LLM response cannot be parsed.
    """
    user_message = format_detections_for_prompt(detections)

    # format='json' tells Ollama to constrain the output to valid JSON syntax.
    # It does NOT enforce our schema — that's what the system prompt does.
    # Together they give us: valid JSON (format param) with the right keys (prompt).
    response = ollama.chat(
        model='llama3.2:3b',
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user',   'content': user_message},
        ],
        format='json',
        options={
            # temperature=0 makes the output deterministic.
            # For a safety-critical navigation decision, you want the same input
            # to always produce the same output. Creativity is not useful here.
            'temperature': 0,
        }
    )

    raw_content = response['message']['content']

    try:
        decision = json.loads(raw_content)
    except json.JSONDecodeError as e:
        # If parsing fails despite format='json', return a safe fallback.
        # In Step 2 this will cause the ROS2 node to publish STOP, which is
        # always the safe default when the agent cannot reason.
        print(f'[AGENT ERROR] Failed to parse LLM response as JSON: {e}')
        print(f'[AGENT ERROR] Raw response was: {raw_content}')
        return {
            'action': 'STOP',
            'reason': 'Agent reasoning failed — defaulting to safe stop.',
            'primary_obstacle': None,
            'confidence_in_decision': 0.0,
        }

    return decision


def main():
    print('=' * 60)
    print('MK2 Agentic AI — Step 1: Standalone Agent Loop')
    print('=' * 60)

    print('\n[INPUT] Detection payload being sent to agent:')
    print(format_detections_for_prompt(SIMULATED_DETECTIONS))

    print('\n[AGENT] Querying llama3.2:3b...')
    decision = query_agent(SIMULATED_DETECTIONS)

    print('\n[DECISION] Agent response:')
    print(json.dumps(decision, indent=2))

    print('\n[SUMMARY]')
    print(f"  Action   : {decision.get('action', 'UNKNOWN')}")
    print(f"  Obstacle : {decision.get('primary_obstacle', 'None')}")
    print(f"  Reason   : {decision.get('reason', 'N/A')}")
    print(f"  Certainty: {decision.get('confidence_in_decision', 0.0)}")
    print('=' * 60)


if __name__ == '__main__':
    main()
