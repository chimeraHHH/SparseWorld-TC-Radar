"""Require sustained memory headroom before launching an independent GPU job."""
import subprocess


class CapacityWindow:
    def __init__(self, minimum_free_mib=132000, quiet_seconds=60):
        if minimum_free_mib <= 0 or quiet_seconds <= 0:
            raise ValueError('Capacity and observation interval must be positive')
        self.minimum_free_mib = minimum_free_mib
        self.quiet_seconds = quiet_seconds
        self.sufficient_since = None

    def observe(self, free_mib, now):
        if free_mib < self.minimum_free_mib:
            self.sufficient_since = None
            return False
        if self.sufficient_since is None:
            self.sufficient_since = now
        return now - self.sufficient_since >= self.quiet_seconds


def memory_snapshot(uuid):
    output = subprocess.check_output([
        'nvidia-smi', '-i', uuid, '--query-gpu=memory.free,memory.total',
        '--format=csv,noheader,nounits'], text=True).strip()
    free, total = [int(value.strip()) for value in output.split(',')]
    if not 0 <= free <= total:
        raise ValueError('Invalid GPU memory snapshot: ' + output)
    return dict(free_mib=free, total_mib=total)
