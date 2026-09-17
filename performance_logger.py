import time
import csv
from pathlib import Path

class PerformanceLogger:
    def __init__(self, output_path='results/perf_log.csv'):
        self.records = []
        self.output_path = Path(output_path)
        self._frame_start = None

    def frame_start(self):
        self._frame_start = time.perf_counter()

    def stage(self, stage_name):
        """Record timestamp for a named pipeline stage."""
        t = time.perf_counter()
        self.records.append({
            'stage': stage_name,
            'elapsed_ms': (t - self._frame_start) * 1000
        })

    def save(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['stage','elapsed_ms'])
            writer.writeheader()
            writer.writerows(self.records)
