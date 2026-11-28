"""The dashboard's data layer: attack type, detecting layer, ground truth."""

from feed import LiveState


def _v(drone, t, track, attack_type=None, verdict=None):
    return {"drone": drone, "t_session": t, "track": track,
            "verdict": verdict or {"A": "MITM", "B-RF": "ML_COMPROMISED",
                                   "B-IF": "ANOMALY"}[track],
            "attack_type": attack_type, "stage": "rule" if track == "A" else "ml",
            "reason": "x: evidence", "onset": 100, "detection_delay_s": t - 100}


def _state():
    s = LiveState()
    for name in ("Drone1", "Drone2", "Drone3"):
        s.apply_telemetry({"about": name, "role": "target", "t_session": 1,
                           "exchange_type": "x", "trust": {"Trust_Score": 0.9}})
    return s


def test_rule_names_the_type_even_after_ml():
    s = _state()
    s.apply_verdict(_v("Drone1", 120, "B-RF", "Sybil"))
    s.apply_verdict(_v("Drone1", 130, "A", "MITM"))
    inc = s.incidents["Drone1"]
    assert inc.attack_type == "MITM"            # the rule wins
    assert inc.caught_by == "ML, then Rule"
    assert inc.alert_t == 120 and inc.alert_delay_s == 20


def test_anomaly_alone_is_a_watch_with_no_type():
    s = _state()
    s.apply_verdict(_v("Drone2", 150, "B-IF"))
    inc = s.incidents["Drone2"]
    assert inc.severity == "watch"
    assert inc.attack_type is None
    assert inc.caught_by == "ML (anomaly)"


def test_swarm_puts_alerted_drones_first_with_type():
    s = _state()
    s.apply_verdict(_v("Drone3", 140, "A", "DataManipulation",
                       verdict="DATA_MANIPULATION"))
    top = s.swarm_table()[0]
    assert top["drone"] == "Drone3"
    assert top["attack_type"] == "Data manipulation"
    assert top["caught_by"] == "Rule"


def test_scorecard_against_ground_truth():
    s = _state()
    s.attack_type, s.gt_victim = "Data Manipulation", "Drone3"
    s.apply_verdict(_v("Drone3", 140, "A", "DataManipulation"))
    s.apply_verdict(_v("Drone1", 160, "B-RF", "MITM"))
    sc = s.run_scorecard()
    assert sc["caught"] == ["Drone3"] and sc["named"] == ["Drone3"]
    assert sc["false_alerts"] == ["Drone1"]
    assert s.is_attacker("Drone1") is False


def test_live_mode_has_no_ground_truth():
    s = _state()
    assert s.run_scorecard() is None
    assert s.is_attacker("Drone1") is None
