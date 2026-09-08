"""Build the first-reconstruction diagnostic against the current NCFV libraries."""
import json
import argparse
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, default=Path(__file__).with_name("first_reconstruction_probe.cpp"))
parser.add_argument("--output", type=Path, default=Path("/tmp/ncfv_first_reconstruction_probe"))
args = parser.parse_args()
build = root / "build"
database = json.loads((build / "compile_commands.json").read_text())
entry = next(item for item in database if item["file"].endswith("/app/NCFV/NCFV.cpp"))
object_file = str(args.output.resolve()) + ".o"
executable = str(args.output.resolve())
command = shlex.split(entry["command"])
command[command.index("-o") + 1] = object_file
command[command.index("-c") + 1] = str(args.source.resolve())
subprocess.run(command, cwd=build, check=True)
lines = subprocess.check_output(["ninja", "-t", "commands", "NCFV"], cwd=build, text=True).splitlines()
link = shlex.split(next(line for line in reversed(lines) if " -o app/NCFV.exe " in line))
if link[:2] != [":", "&&"] or link[-2:] != ["&&", ":"]:
    raise RuntimeError("Unexpected Ninja link command; refusing to execute shell syntax")
link = link[2:-2]
link[link.index("-o") + 1] = executable
link = [object_file if item.endswith("/app/NCFV/NCFV.cpp.o") else item for item in link]
subprocess.run(link, cwd=build, check=True)
print(executable)
