"""Pure utilities for latest-causal, per-sensor radar observations.

Velocity fields are the SDK's processed ego-compensated velocities. They are
not an independently retained raw Doppler measurement.
"""
import numpy as np

VELOCITY_COLUMNS = (3, 4, 7)


def latest_causal_sweep(get_record, first_token, reference_us):
    """Find the latest record at/before the anchor, using metadata only.

    nuScenes sample.data is not assumed to be on either side of the LiDAR
    anchor. Check both directions; a future record is never read as points.
    Return the selected record and the next timestamp (a maximality witness).
    """
    if not first_token:
        return None, None
    current = get_record(first_token)
    visited = {current['token']}
    while current['timestamp'] > reference_us:
        previous = current.get('prev', '')
        if not previous:
            return None, int(current['timestamp'])
        item = get_record(previous)
        if item['token'] in visited or item['timestamp'] >= current['timestamp']:
            raise ValueError('Nonmonotone or cyclic radar sample_data chain')
        visited.add(item['token'])
        current = item
    # Start a fresh traversal: the valid next boundary may have been visited
    # while walking backward from a future sample.data record.
    visited = {current['token']}
    while current.get('next', ''):
        item = get_record(current['next'])
        if item['token'] in visited or item['timestamp'] <= current['timestamp']:
            raise ValueError('Nonmonotone or cyclic radar sample_data chain')
        if item['timestamp'] > reference_us:
            return current, int(item['timestamp'])
        visited.add(item['token'])
        current = item
    return current, None


def geometry_filter_and_sanitize(features, xy_limit):
    """Filter without consulting velocity, then replace invalid velocity pairs.

    Membership, RCS, age and LOS are identical for the geometry/velocity arms.
    Invalid processed velocities are represented as zero in both cache views.
    """
    features = np.asarray(features)
    nonvelocity = [0, 1, 2, 5, 6, 8, 9]
    keep = (np.isfinite(features[:, nonvelocity]).all(1)
            & (np.abs(features[:, :2]) <= xy_limit).all(1))
    points = features[keep].astype(np.float32, copy=True)
    invalid = ~np.isfinite(points[:, VELOCITY_COLUMNS]).all(1)
    points[np.ix_(invalid, VELOCITY_COLUMNS)] = 0.
    return points, int(invalid.sum())


def velocity_view(points, mode):
    if mode == 'processed':
        return points
    if mode != 'zero':
        raise ValueError('velocity_mode must be processed or zero')
    points = points.copy()
    points[:, VELOCITY_COLUMNS] = 0.
    return points
