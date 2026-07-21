import json
from tests.test_dialogue_context_v29 import services

_, train, dialogue = services()
train.train("Механик дал роботу болт.", independent_key="scenario-a")
hid = dialogue.create()["hive"]["id"]

# Turn 1: multi-GAP
r1 = dialogue.query(hid, "Кому и кто дал болт?")
print("Turn 1 STATUS:", r1["answer"]["status"])
print("Turn 1:", r1["answer"]["surface"])
print()

# Turn 2: Кому?
r2 = dialogue.query(hid, "Кому?")
print("Turn 2 STATUS:", r2["answer"]["status"])
print("Turn 2:", r2["answer"]["surface"])
tr2 = r2.get("trace", {})
gd2 = tr2.get("gap_release_diagnostic")
if gd2:
    print("  GAP release decision:", gd2.get("decision"))
    print("  reason:", gd2.get("decision_reason"))
    print("  selected_participant_node_id:", gd2.get("selected_participant_node_id"))
    print("  selected_score:", gd2.get("selected_score"))
    print("  second_score:", gd2.get("second_score"))
    for c in gd2.get("candidates", []):
        print(f"    {c['resolved_surface']} score={c['final_score']:.3f}")
else:
    print("  NO gap_release_diagnostic in trace")
print()

# Turn 3: Кто?
r3 = dialogue.query(hid, "Кто?")
print("Turn 3 STATUS:", r3["answer"]["status"])
print("Turn 3:", r3["answer"]["surface"])
tr3 = r3.get("trace", {})
gd3 = tr3.get("gap_release_diagnostic")
if gd3:
    print("  GAP release decision:", gd3.get("decision"))
    print("  reason:", gd3.get("decision_reason"))
    print("  selected_participant_node_id:", gd3.get("selected_participant_node_id"))
    print("  selected_score:", gd3.get("selected_score"))
    for c in gd3.get("candidates", []):
        print(f"    {c['resolved_surface']} score={c['final_score']:.3f}")
else:
    print("  NO gap_release_diagnostic in trace")
print()

# Turn 4: Что?
r4 = dialogue.query(hid, "Что?")
print("Turn 4 STATUS:", r4["answer"]["status"])
print("Turn 4:", r4["answer"]["surface"])
tr4 = r4.get("trace", {})
gd4 = tr4.get("gap_release_diagnostic")
if gd4:
    print("  GAP release decision:", gd4.get("decision"))
    print("  reason:", gd4.get("decision_reason"))
    print("  selected_participant_node_id:", gd4.get("selected_participant_node_id"))
    print("  selected_score:", gd4.get("selected_score"))
    print("  second_score:", gd4.get("second_score"))
    for c in gd4.get("candidates", []):
        print(f"    {c['resolved_surface']} score={c['final_score']:.3f} accept={c['accepted']}")
else:
    print("  NO gap_release_diagnostic in trace")