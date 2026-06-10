"""Tests for the Titman-QP reconstruction backend (qp_reconstruct.py),
a Python port of registry-ipd's reconstructArmQP."""
import numpy as np

from qp_reconstruct import reconstruct_arm_qp


def _km_at(grid, t, e):
    ev = np.unique(t[e == 1]); s = 1.0; out = np.ones_like(grid, float)
    steps_t = [0.0]; steps_s = [1.0]
    for tt in ev:
        n = int(np.sum(t >= tt)); d = int(np.sum((t == tt) & (e == 1)))
        if n > 0:
            s *= (1 - d / n)
        steps_t.append(tt); steps_s.append(s)
    steps_t = np.array(steps_t); steps_s = np.array(steps_s)
    for i, g in enumerate(grid):
        k = np.searchsorted(steps_t, g, side="right") - 1
        out[i] = steps_s[max(k, 0)]
    return out


def test_qp_recovers_event_count_and_tracks_curve():
    t = np.array([0, 1, 2, 3, 4, 5, 6.0])
    s = np.array([1, .85, .72, .61, .52, .45, .40])
    N, E = 200, 84
    tt, ee = reconstruct_arm_qp(t, s, total_n=N, total_events=E, follow_up_max=6.0)
    assert abs(int(ee.sum()) - E) <= 2                  # event-count constraint
    assert tt.size == N                                 # N bodies, none invented
    rec = _km_at(t, tt, ee)
    assert np.max(np.abs(rec - s)) < 0.03               # KM tracks the curve


def test_qp_no_event_count_degrades_to_curve_only_ceiling():
    """With no total_events the QP uses E0 = N(1-S_K) (curve-only ceiling)."""
    t = np.array([0, 2, 4, 6.0]); s = np.array([1, .8, .6, .5]); N = 100
    tt, ee = reconstruct_arm_qp(t, s, total_n=N, total_events=None)
    # curve-only ceiling: ~all non-survivors are events -> ~N*(1-S_K) = 50
    assert 40 <= int(ee.sum()) <= 55
    assert np.all(np.diff(np.sort(tt)) >= 0)


def test_qp_monotone_and_bounded_outputs():
    t = np.array([0, 1, 2, 3.0]); s = np.array([1, .9, .7, .55]); N = 50
    tt, ee = reconstruct_arm_qp(t, s, total_n=N, total_events=20)
    assert set(np.unique(ee).tolist()) <= {0, 1}
    assert (tt >= 0).all() and tt.size == N


def test_qp_handles_sparse_anchors_fast():
    """Many curve points must stay fast (O(n) sensitivity recurrence)."""
    t = np.linspace(0, 10, 400)
    s = np.clip(np.exp(-0.1 * t), 0, 1)
    tt, ee = reconstruct_arm_qp(t, s, total_n=300, total_events=180, follow_up_max=10.0)
    assert tt.size == 300 and abs(int(ee.sum()) - 180) <= 3
