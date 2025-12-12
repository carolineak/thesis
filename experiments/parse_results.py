#!/usr/bin/env python3
import re
from typing import Dict, List, Any


def parse_result_file(path: str) -> Dict[str, List[Any]]:
    """
    Parse a text result file containing multiple LU experiments and return
    a dictionary of lists with parsed values.

    Each key maps to a list with one entry per test block.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    # Initialise dict-of-lists for all fields we care about
    data: Dict[str, List[Any]] = {
        "name": [],
        "matrix_n": [],
        "matrix_m": [],
        "L_ops": [],
        "U_ops": [],
        "Schur_ops": [],
        "Total_ops": [],
        "L_time": [],
        "U_time": [],
        "Schur_time": [],
        "Total_time": [],
        "fillin_n": [],
        "fillin_m": [],
        "lu_time": [],
        "test_passed": [],
        "rel_error": [],
    }

    i = 0
    n_lines = len(lines)

    def append_record(record: Dict[str, Any]) -> None:
        """Append a single test record into the dict-of-lists, filling in None where missing."""
        for key in data.keys():
            data[key].append(record.get(key, None))

    while i < n_lines:
        line = lines[i].strip()

        # Start of a new test
        if line.startswith("Test on"):
            record: Dict[str, Any] = {}

            # Test name (everything after "Test on ")
            record["name"] = line[len("Test on "):].strip()

            i += 1

            # ---------------- Matrix size ----------------
            while i < n_lines and not lines[i].strip().startswith("Matrix size:"):
                i += 1
            if i < n_lines:
                m = re.search(r"Matrix size:\s*(\d+)\s*x\s*(\d+)", lines[i])
                if m:
                    record["matrix_n"] = int(m.group(1))
                    record["matrix_m"] = int(m.group(2))
            i += 1

            # ---------------- Number of block matrix operations ----------------
            while i < n_lines and not lines[i].strip().startswith("Number of block matrix operations"):
                i += 1
            if i < n_lines:
                i += 1  # skip header line
                while i < n_lines and lines[i].strip().startswith(("L solves", "U solves", "Schur updates", "Total")):
                    s = lines[i].strip()
                    num_match = re.search(r":\s*([0-9.eE+-]+)", s)
                    if num_match:
                        val = int(float(num_match.group(1)))
                        if s.startswith("L solves"):
                            record["L_ops"] = val
                        elif s.startswith("U solves"):
                            record["U_ops"] = val
                        elif s.startswith("Schur updates"):
                            record["Schur_ops"] = val
                        elif s.startswith("Total"):
                            record["Total_ops"] = val
                    i += 1

            # ---------------- Time taken (seconds) ----------------
            while i < n_lines and not lines[i].strip().startswith("Time taken (seconds):"):
                i += 1
            if i < n_lines:
                i += 1  # skip header line
                while i < n_lines and lines[i].strip().startswith(("L solves", "U solves", "Schur updates", "Total")):
                    s = lines[i].strip()
                    num_match = re.search(r":\s*([0-9.eE+-]+)", s)
                    if num_match:
                        val = float(num_match.group(1))
                        if s.startswith("L solves"):
                            record["L_time"] = val
                        elif s.startswith("U solves"):
                            record["U_time"] = val
                        elif s.startswith("Schur updates"):
                            record["Schur_time"] = val
                        elif s.startswith("Total"):
                            record["Total_time"] = val
                    i += 1

            # ---------------- Fill-in matrix size ----------------
            while i < n_lines and not lines[i].strip().startswith("Fill-in matrix size:"):
                i += 1
            if i < n_lines:
                m = re.search(r"Fill-in matrix size:\s*(\d+)\s*x\s*(\d+)", lines[i])
                if m:
                    record["fillin_n"] = int(m.group(1))
                    record["fillin_m"] = int(m.group(2))
            i += 1

            # ---------------- Test passed / failed & relative error ----------------
            passed = None
            rel_err = None
            lu_time = None

            while i < n_lines and not lines[i].strip().startswith("Test on"):
                s = lines[i].strip()

                # TEST PASSED / FAILED line
                if s.startswith("TEST "):
                    if "PASSED" in s:
                        passed = True
                    elif "FAILED" in s:
                        passed = False
                    m = re.search(r"is\s*([0-9.eE+-]+)", s)
                    if m:
                        rel_err = float(m.group(1))

                # Time spend computing LU
                if s.startswith("Time spend computing LU:"):
                    m = re.search(r"Time spend computing LU:\s*([0-9.eE+-]+)", s)
                    if m:
                        lu_time = float(m.group(1))

                # stop scanning when we hit the next test or end
                if i + 1 < n_lines and lines[i + 1].strip().startswith("Test on"):
                    i += 1
                    break

                i += 1

            record["test_passed"] = passed
            record["rel_error"] = rel_err
            record["lu_time"] = lu_time

            append_record(record)
        else:
            i += 1

    return data


if __name__ == "__main__":
    import sys
    from pprint import pprint

    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <result_file.txt>")
        sys.exit(1)

    results = parse_result_file(sys.argv[1])
    pprint(results)
