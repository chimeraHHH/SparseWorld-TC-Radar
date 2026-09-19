from tools.gpu_capacity import CapacityWindow


def test_short_allocation_gaps_never_admit_training():
    window = CapacityWindow()
    assert not window.observe(140000, 0.)
    assert not window.observe(140000, 59.)
    assert not window.observe(35000, 59.5)
    assert not window.observe(140000, 60.)
    assert not window.observe(140000, 119.)
    assert window.observe(140000, 120.)


def test_small_resident_context_can_coexist_with_full_batch_headroom():
    window = CapacityWindow()
    assert not window.observe(143000, 10.)
    assert window.observe(143000, 70.)
    assert not window.observe(131999, 71.)
    assert not window.observe(132000, 72.)
    assert window.observe(132000, 132.)
