"""
MK2 Agentic AI — Step 1: Standalone Agent Loop (v2)
----------------------------------------------------
Changes from v1:
  - Added classify_detection() which pre-computes geometry in Python
    before sending to the LLM. This fixes the arithmetic hallucination
    issue seen in v1 where llama3.2:3b ignored coordinate changes.
  - format_detections_for_prompt() now enriches each detection with:
      size_classification   : "large_close" | "medium" | "small_far"
      position_classification: "left" | "center" | "right"
      frame_coverage_percent: float (e.g. 34.4)
      is_direct_obstacle    : bool (Python decides this, not the LLM)
  - System prompt decision rules simplified — no arithmetic required.
    The LLM reads semantic labels and reasons from them, not raw numbers.

Architecture note:
  Layer 1 — YOLO26s        : pixels → bounding boxes
  Layer 2 — classify_detection() : coordinates → semantic facts  (THIS FILE)
  Layer 3 — LLM agent      : semantic facts → navigation decision (THIS FILE)

  Python does perception preprocessing. LLM does reasoning and deciding.
  This is the correct division of labour for a 3B model on a laptop GPU.

Prerequisites:
  pip install ollama
  ollama pull llama3.2:3b
"""

import json
import ollama

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

FRAME_WIDTH  = 320
FRAME_HEIGHT = 240

# Horizontal thirds used to classify left / center / right
LEFT_BOUNDARY  = FRAME_WIDTH / 3        # 106.7px
RIGHT_BOUNDARY = FRAME_WIDTH * 2 / 3   # 213.3px

# Size thresholds as a fraction of frame width
LARGE_THRESHOLD  = 0.25   # bbox_width / FRAME_WIDTH > 0.25 → large_close
MEDIUM_THRESHOLD = 0.10   # bbox_width / FRAME_WIDTH > 0.10 → medium

# Minimum confidence to consider a detection high-confidence
CONFIDENCE_THRESHOLD = 0.7

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
# Decision rules are expressed in terms of the pre-computed semantic labels,
# not raw coordinates. The LLM does not need to do any arithmetic.
#
# The LLM's job here is genuine reasoning:
#   - Which detection is the most relevant threat?
#   - What is the safest action given multiple detections?
#   - Why is that action the right one?
#
# Python classifies. LLM decides.

SYSTEM_PROMPT = """
You are the navigation reasoning agent for MK2, an autonomous mobile robot.

Your job is to interpret pre-classified object detection data from the robot's
camera and decide what action the robot should take next.

You will receive a JSON object with a list of detections. Each detection has
been pre-classified with the following fields:
  - "label": object class name (e.g. "person", "chair")
  - "confidence": detection confidence from 0.0 to 1.0
  - "frame_coverage_percent": how much of the frame width this object occupies
  - "size_classification": one of:
      "large_close" — object covers more than 25% of the frame width (close, large threat)
      "medium"      — object covers 10-25% of the frame width
      "small_far"   — object covers less than 10% of the frame width (distant, small)
  - "position_classification": one of:
      "left"   — object is in the left third of the frame (bbox center x < 107)
      "center" — object is in the middle third of the frame
      "right"  — object is in the right third of the frame (bbox center x > 213)
  - "is_direct_obstacle": true if the object is large_close, centered, and
    confidence > 0.7. This means it is a direct obstacle blocking the robot's path.

Decision rules you must apply:
  - If any detection has is_direct_obstacle=true:
      Action: STOP. The path ahead is blocked.
  - If no direct obstacles exist but a high-confidence (>0.7) detection is
    position_classification="left":
      Action: TURN_RIGHT. Obstacle is to the left, path may be clear to the right.
  - If no direct obstacles exist but a high-confidence (>0.7) detection is
    position_classification="right":
      Action: TURN_LEFT. Obstacle is to the right, path may be clear to the left.
  - If no detections exist, or all detections are confidence < 0.7,
    or all detections are size_classification="small_far":
      Action: MOVE_FORWARD. Path is clear.
  - If multiple conflicting high-confidence obstacles exist with no clear path:
      Action: STOP. When in doubt, stop.

You must respond with ONLY a valid JSON object. No explanation, no prose, no markdown.
The JSON must have exactly these keys:
  - "action": one of "MOVE_FORWARD", "STOP", "TURN_LEFT", "TURN_RIGHT"
  - "reason": one sentence explaining why this action was chosen
  - "primary_obstacle": the label of the most relevant detected object,
     or null if no relevant detection exists
  - "confidence_in_decision": your confidence in this decision from 0.0 to 1.0
"""

# ── GEOMETRY PREPROCESSING ────────────────────────────────────────────────────

def classify_detection(det: dict) -> dict:
    """
    Pre-computes spatial geometry for one detection before sending to the LLM.

    This function is the boundary between raw sensor data and semantic meaning.
    It converts pixel coordinates into human-readable labels that the LLM can
    reason about reliably — without needing to do arithmetic itself.

    Parameters
    ----------
    det : dict with keys: label, confidence, bbox_center_x, bbox_width
          (bbox_center_y and bbox_height are kept but not used for classification)

    Returns
    -------
    The original dict enriched with four new keys:
      size_classification, position_classification,
      frame_coverage_percent, is_direct_obstacle
    """
    cx = det['bbox_center_x']
    bw = det['bbox_width']
    conf = det['confidence']

    # ── Size classification ───────────────────────────────────────────────────
    frame_fraction = bw / FRAME_WIDTH
    if frame_fraction > LARGE_THRESHOLD:
        size_class = 'large_close'
    elif frame_fraction > MEDIUM_THRESHOLD:
        size_class = 'medium'
    else:
        size_class = 'small_far'

    # ── Position classification ───────────────────────────────────────────────
    if cx < LEFT_BOUNDARY:
        position_class = 'left'
    elif cx > RIGHT_BOUNDARY:
        position_class = 'right'
    else:
        position_class = 'center'

    # ── Direct obstacle flag ──────────────────────────────────────────────────
    # Python makes this binary determination so the LLM doesn't have to.
    # An object is a direct obstacle only if all three conditions are met:
    #   1. Large enough to be a real threat (not a distant blip)
    #   2. Centered in the frame (in the robot's forward path)
    #   3. High confidence (not a misdetection)
    is_direct_obstacle = (
        size_class == 'large_close'
        and position_class == 'center'
        and conf > CONFIDENCE_THRESHOLD
    )

    return {
        **det,
        'size_classification':    size_class,
        'position_classification': position_class,
        'frame_coverage_percent': round(frame_fraction * 100, 1),
        'is_direct_obstacle':     is_direct_obstacle,
    }


def format_detections_for_prompt(detections: list) -> str:
    """
    Enriches each detection with pre-computed geometry, then serialises
    the whole payload to a JSON string for the LLM user message.

    This function is the exact same interface as v1 — Step 2 can call it
    with live ROS 2 detections with no changes.
    """
    enriched = [classify_detection(d) for d in detections] if detections else []
    payload = {
        'detections': enriched,
        'frame_width': FRAME_WIDTH,
        'frame_height': FRAME_HEIGHT,
    }
    return json.dumps(payload, indent=2)


# ── AGENT QUERY ───────────────────────────────────────────────────────────────

def query_agent(detections: list) -> dict:
    """
    Sends the enriched detection payload to llama3.2:3b and returns the
    parsed navigation decision.

    Parameters
    ----------
    detections : list of raw detection dicts (label, confidence, bbox fields)
                 classify_detection() is called internally — do not pre-enrich.

    Returns
    -------
    dict with keys: action, reason, primary_obstacle, confidence_in_decision
    Returns a safe STOP fallback dict if the LLM response cannot be parsed.
    """
    user_message = format_detections_for_prompt(detections)

    response = ollama.chat(
        model='llama3.2:3b',
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user',   'content': user_message},
        ],
        format='json',
        options={
            'temperature': 0,   # deterministic — same input always gives same output
        }
    )

    raw_content = response['message']['content']

    try:
        decision = json.loads(raw_content)
    except json.JSONDecodeError as e:
        print(f'[AGENT ERROR] Failed to parse LLM response as JSON: {e}')
        print(f'[AGENT ERROR] Raw response was: {raw_content}')
        return {
            'action': 'STOP',
            'reason': 'Agent reasoning failed — defaulting to safe stop.',
            'primary_obstacle': None,
            'confidence_in_decision': 0.0,
        }

    return decision


# ── TEST SCENARIOS ────────────────────────────────────────────────────────────
# These four scenarios cover all four expected actions.
# Expected outputs are annotated so you can verify correctness at a glance.

TEST_SCENARIOS = [
    {
        'name': 'Scenario 1 — Person directly ahead, large and close',
        'expected_action': 'STOP',
        'detections': [
            {
                'label': 'person',
                'confidence': 0.94,
                'bbox_center_x': 158.0,   # center (107 < 158 < 213)
                'bbox_center_y': 120.0,
                'bbox_width': 110.0,      # 110/320 = 34.4% → large_close
                'bbox_height': 180.0,
            },
            {
                'label': 'chair',
                'confidence': 0.61,       # below 0.7 → ignored
                'bbox_center_x': 280.0,
                'bbox_center_y': 200.0,
                'bbox_width': 40.0,
                'bbox_height': 55.0,
            },
        ],
    },
    {
        'name': 'Scenario 2 — Person to the left, large but offset',
        'expected_action': 'TURN_RIGHT',
        'detections': [
            {
                'label': 'person',
                'confidence': 0.94,
                'bbox_center_x': 50.0,    # left (50 < 107)
                'bbox_center_y': 120.0,
                'bbox_width': 110.0,      # large_close but NOT centered → not direct obstacle
                'bbox_height': 180.0,
            },
        ],
    },
    {
        'name': 'Scenario 3 — Person to the left, medium size',
        'expected_action': 'TURN_RIGHT',
        'detections': [
            {
                'label': 'person',
                'confidence': 0.94,
                'bbox_center_x': 50.0,    # left
                'bbox_center_y': 50.0,
                'bbox_width': 50.0,       # 50/320 = 15.6% → medium
                'bbox_height': 50.0,
            },
        ],
    },
    {
        'name': 'Scenario 4 — Person tiny and far, no real threat',
        'expected_action': 'MOVE_FORWARD',
        'detections': [
            {
                'label': 'person',
                'confidence': 0.94,
                'bbox_center_x': 50.0,
                'bbox_center_y': 50.0,
                'bbox_width': 10.0,       # 10/320 = 3.1% → small_far
                'bbox_height': 10.0,
            },
        ],
    },
    {
        'name': 'Scenario 5 — Empty frame, nothing detected',
        'expected_action': 'MOVE_FORWARD',
        'detections': [],
    },
    {
        'name': 'Scenario 6 — Person to the right, large',
        'expected_action': 'TURN_LEFT',
        'detections': [
            {
                'label': 'person',
                'confidence': 0.94,
                'bbox_center_x': 270.0,   # right (270 > 213)
                'bbox_center_y': 120.0,
                'bbox_width': 90.0,       # 90/320 = 28.1% → large_close but right → not direct obstacle
                'bbox_height': 150.0,
            },
        ],
    },
]


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_scenario(scenario: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {scenario['name']}")
    print(f"  Expected: {scenario['expected_action']}")
    print('=' * 60)

    # Show the enriched payload so you can verify classify_detection() output
    enriched_payload = format_detections_for_prompt(scenario['detections'])
    print('\n[ENRICHED INPUT sent to LLM]:')
    print(enriched_payload)

    print('\n[AGENT] Querying llama3.2:3b...')
    decision = query_agent(scenario['detections'])

    # Pass/fail check
    actual = decision.get('action', 'UNKNOWN')
    status = '✓ PASS' if actual == scenario['expected_action'] else '✗ FAIL'

    print(f'\n[DECISION]:')
    print(json.dumps(decision, indent=2))
    print(f'\n[RESULT]: {status}  (expected={scenario["expected_action"]}, got={actual})')


def main():
    print('MK2 Agentic AI — Step 1 v2: Geometry-enriched agent loop')
    print('Running all test scenarios...')

    for scenario in TEST_SCENARIOS:
        run_scenario(scenario)

    print(f'\n{"=" * 60}')
    print('All scenarios complete.')
    print('If all show ✓ PASS, Step 2 (ROS 2 integration) can begin.')
    print('=' * 60)


if __name__ == '__main__':
    main()
