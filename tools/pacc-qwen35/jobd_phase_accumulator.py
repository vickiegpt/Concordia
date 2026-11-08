"""Bounded-memory phase accounting for long-running hardware captures."""
from collections import Counter, deque
import math

PHASES = ("dst_prepare", "input_copy", "compute", "output_copy", "cleanup",
          "completion_publish", "device_before_publish")

class PhaseAccumulator:
    def __init__(self, start_index, build):
        self.start = self.high = start_index
        self.required_build = build
        self.indices, self.recent = set(), deque()
        self.unique = self.complete = self.invalid = self.late = 0
        self.pending = {}
        self.completed_indices = {}
        self.completed_order = deque()
        self.builds = Counter()
        self.hist = {phase: Counter() for phase in PHASES}

    def observe(self, record):
        index, seq, status, job, aux = (record[k] for k in ("index","seq","status","job","aux"))
        if index <= self.start or index in self.indices:
            return
        if index < self.high-4096:
            self.late += 1
            return
        self.indices.add(index);self.recent.append(index);self.unique += 1
        self.high = max(self.high,index)
        if len(self.recent)>4096:
            self.indices.remove(self.recent.popleft())
        if job != 5 or not (status in (0x5140,0x5330) or 0x5310 <= status <= 0x5316):
            return
        if seq in self.completed_indices:
            # Physical ring order can deliver the start after all seven phases.
            if status == 0x5140 and index < self.completed_indices[seq]:
                return
            raise RuntimeError("duplicate phase or reused completed sequence")
        row = self.pending.setdefault(seq,{})
        row["last_index"] = max(index,row.get("last_index",0))
        if status == 0x5330: row["build"]=aux
        if 0x5310 <= status <= 0x5316: row[PHASES[status-0x5310]]=aux
        if len(self.pending)>4096:
            raise RuntimeError("phase capture has too many incomplete submissions")
        if "build" in row and all(phase in row for phase in PHASES):
            self.complete += 1;self.builds[hex(row["build"])]+=1
            if any(row[phase]>=0xffffffff for phase in PHASES): self.invalid += 1
            else:
                for phase in PHASES: self.hist[phase][row[phase]]+=1
            del self.pending[seq]
            self.completed_indices[seq] = row["last_index"]
            self.completed_order.append(seq)
            if len(self.completed_order)>4096:
                del self.completed_indices[self.completed_order.popleft()]

    def report(self, returncode, timed_out):
        gaps=self.high-self.start-self.unique
        build_matches=bool(self.builds) and set(self.builds)=={hex(self.required_build)}
        stats={}
        for phase,hist in self.hist.items():
            if not hist: continue
            count=sum(hist.values())
            def percentile(fraction):
                target=math.ceil(count*fraction);seen=0
                for value,n in sorted(hist.items()):
                    seen+=n
                    if seen>=target:return value
            stats[phase]=dict(sum_us=sum(v*n for v,n in hist.items()),p50_us=percentile(.5),
                              p99_us=percentile(.99),max_us=max(hist))
        return dict(complete_submissions=self.complete,incomplete_submission_count=len(self.pending),
                    incomplete_sample=list(self.pending.items())[:8],saturated_submissions=self.invalid,
                    submission_build_counts=dict(self.builds),build_matches=build_matches,
                    observed_index_gap_count=gaps,late_records=self.late,
                    first_index_exclusive=self.start,last_index=self.high,unique_indices=self.unique,
                    phases_us=stats,returncode=returncode,timed_out=timed_out,
                    capture_valid=returncode==0 and not timed_out and self.complete>0 and
                    not self.pending and not self.invalid and gaps==0 and self.late==0 and build_matches)
