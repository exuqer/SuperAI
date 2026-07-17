from server.v2.capacity import capacity_pressure, get_working_occupancy


def test_working_capacity_uses_one_unit_and_pressure_curve():
    cells = [{"component_class": "memory_source"}] * 6 + [{"component_class": "role_candidate"}] * 5 + [{"component_class": "context"}] * 4
    capacity = get_working_occupancy(cells, 24)

    assert capacity["active_total"] == 15
    assert capacity["memory_sources"] == 6
    assert capacity["reasoning_cells"] == 5
    assert capacity["context_cells"] == 4
    assert capacity["occupancy"] == .625
    assert capacity_pressure(12, 24) == 0
    assert round(capacity_pressure(18, 24), 6) == .140625
    assert capacity_pressure(24, 24) == 1
