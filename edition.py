#!/usr/bin/env python3
"""
Create a summary of the substantive musical changes between two MuseScore .mscx states.

The intended interpretation is:
- edition: OSQ's upstream file (the initial state)
- source: MALerLab's final OMR reverse engineered state

Unlike MALerLab who focus on OMR,
the focus here is on editorial changes.
As such this script deliberately ignores many
changes, notably including layout and engraving.

The initial implementation concentrates on:
- add/remove notes and/or rests
- changed pitch and/or duration

Typical use
from inside a clone of MALerLab/ossq-omr that has the OpenScore/StringQuartets repository as a remote:
For one score (with the given ID example)
```
python edition.py 7313978 --out changes
```
For every score listed in data/scores.yaml
```
python edition.py --out changes
```

Options:
- The OpenScore side is the remote's default branch unless a specific `--openscore-ref` is given;
- The MALerLab side is origin/omr-dev unless a specific `--malerlab-ref` is given.
- In `run-all` mode scores missing from either side are skipped and reported (one failing score does not stop the rest).

No requirements (Python's standard library only).

The output is one Markdown file per requested score.

By:

This is a contribution from Mark Gotham of OpenScore,
to the MALerLab fork of the OpenScore repo.
How's that for open research?


"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

from collections import defaultdict
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from fractions import Fraction
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# MuseScore duration handling
# ---------------------------------------------------------------------------

# Durations are expressed in quarter-note units.
DURATION_QUARTERS = {
    "long": Fraction(16),
    "breve": Fraction(8),
    "whole": Fraction(4),
    "half": Fraction(2),
    "quarter": Fraction(1),
    "eighth": Fraction(1, 2),
    "16th": Fraction(1, 4),
    "32nd": Fraction(1, 8),
    "64th": Fraction(1, 16),
    "128th": Fraction(1, 32),
    "256th": Fraction(1, 64),
}


def dot_count(element: ET.Element) -> int:
    """
    Number of augmentation dots.

    MuseScore stores this as <dots>N</dots> on the Chord/Rest.  (<Dot> elements
    only appear on a Note, and only to carry styling.)
    """
    try:
        return int(element.findtext("dots") or 0)
    except ValueError:
        return 0


def duration_from_element(element: ET.Element) -> Fraction | None:
    """
    Nominal duration (quarter-note units) of a Chord/Rest, ignoring tuplets.

    MuseScore normally stores durationType. Measure rests have
    durationType "measure" plus an explicit <duration> such as 4/4.
    """
    duration_type = element.findtext("durationType")
    if duration_type in DURATION_QUARTERS:
        value = DURATION_QUARTERS[duration_type]
        dots = dot_count(element)
        if dots:
            value *= Fraction(2) - Fraction(1, 2**dots)
        return value

    duration = element.findtext("duration")
    if duration and "/" in duration:
        try:
            numerator, denominator = duration.split("/", 1)
            return Fraction(int(numerator), int(denominator)) * 4
        except ValueError:
            pass

    return None


def duration_label(element: ET.Element) -> str:
    """Human-readable nominal duration."""
    duration_type = element.findtext("durationType") or "unknown"
    dots = dot_count(element)
    if dots == 1:
        return f"{duration_type} dotted"
    if dots == 2:
        return f"{duration_type} double-dotted"
    if dots > 2:
        return f"{duration_type} {dots}-dotted"
    return duration_type


# ---------------------------------------------------------------------------
# Musical representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Note:
    pitch: int

    @property
    def name(self) -> str:
        return midi_name(self.pitch)


@dataclass(frozen=True)
class Event:
    score_path: str
    staff_id: str
    part_name: str
    measure: str               # display label
    measure_index: int         # 1-based position within the staff
    voice: int
    onset: Fraction            # quarters from bar start (grace: its anchor's)
    duration: Fraction | None  # sounding length, tuplet-scaled; None if unknown
    duration_text: str
    kind: str                  # "chord" or "rest"
    notes: tuple[Note, ...]
    grace: bool = False
    grace_rank: int = 0        # <0 before the anchor note, >0 after it


def midi_name(midi: int) -> str:
    """
    Convert a MIDI pitch number to a simple chromatic pitch name.

    This intentionally reports sounding/MIDI pitch.  It does not attempt to
    reconstruct MuseScore's notational spelling (enharmonic accidentals).
    """
    names = [
        "C", "C♯", "D", "D♯", "E", "F",
        "F♯", "G", "G♯", "A", "A♯", "B"
    ]

    octave = midi // 12 - 1
    return f"{names[midi % 12]}{octave}"


def parse_pitch(note_element: ET.Element) -> int | None:
    text = note_element.findtext("pitch")
    if text is None:
        return None

    try:
        return int(text)
    except ValueError:
        return None


def parse_staff_part_names(root: ET.Element) -> dict[str, str]:
    """
    Map MuseScore Staff IDs to the containing Part's trackName.

    A Part can contain more than one Staff (e.g. piano), so several staff IDs
    can legitimately map to the same part name.
    """
    result: dict[str, str] = {}

    for part in root.findall(".//Part"):
        name = (
            part.findtext("trackName")
            or part.findtext("./Instrument/trackName")
            or "Unknown part"
        ).strip()

        for staff in part.findall("./Staff"):
            staff_id = staff.get("id")
            if staff_id:
                result[staff_id] = name

    return result


def iter_master_staves(root: ET.Element) -> Iterable[ET.Element]:
    """
    Yield score-level Staff elements containing measures.

    We deliberately restrict this to direct children of Score. This avoids
    accidentally interpreting Staff definitions inside Part elements as
    musical staves.
    """
    score = root.find("Score")
    if score is None:
        return

    for child in score:
        if child.tag == "Staff":
            yield child



GRACE_TAGS = {"acciaccatura", "appoggiatura"}


def grace_tag(element: ET.Element) -> str | None:
    """
    MuseScore marks grace notes with a child tag on the Chord:
    <acciaccatura/>, <appoggiatura/>, <grace4/>, <grace16/>, <grace8after/>...
    """
    for child in element:
        if child.tag in GRACE_TAGS or child.tag.startswith("grace"):
            return child.tag
    return None


def tuplet_ratio(element: ET.Element) -> Fraction:
    try:
        normal = int(element.findtext("normalNotes") or 1)
        actual = int(element.findtext("actualNotes") or 1)
        return Fraction(normal, actual)
    except (ValueError, ZeroDivisionError):
        return Fraction(1)


def parse_voice(voice: ET.Element, ctx: dict) -> list[Event]:
    """
    Parse one voice of one measure.

    * Tuplets: <Tuplet> ... <endTuplet/> are siblings of the chords they
      scale; durations are multiplied by normalNotes/actualNotes.
    * Grace notes take no time.  Each one is anchored to a real chord: the
      next one for before-graces, the previous one for "...after" graces, and
      given a grace_rank so it has its own unique slot.
    """
    events: list[Event] = []
    onset = Fraction(0)
    last_main = Fraction(0)
    after_rank = 0
    tuplets: list[Fraction] = []
    pending: list[Event] = []          # before-graces awaiting an anchor

    def flush(anchor: Fraction) -> None:
        n = len(pending)
        for i, g in enumerate(pending):
            events.append(replace(g, onset=anchor, grace_rank=-(n - i)))
        pending.clear()

    for element in voice:
        if element.tag == "Tuplet":
            tuplets.append(tuplet_ratio(element))
            continue
        if element.tag == "endTuplet":
            if tuplets:
                tuplets.pop()
            continue
        if element.tag not in ("Chord", "Rest"):
            continue

        nominal = duration_from_element(element)
        text = duration_label(element)
        ratio = Fraction(1)
        for t in tuplets:
            ratio *= t
        duration = nominal * ratio if nominal is not None else None
        if ratio != 1:
            text += f" (x{ratio})"

        grace = grace_tag(element)
        if grace:
            text += " grace"

        if element.tag == "Rest":
            kind, notes = "rest", ()
        else:
            pitches = [
                Note(p) for p in map(parse_pitch, element.findall("./Note"))
                if p is not None
            ]
            kind = "chord"
            notes = tuple(sorted(pitches, key=lambda n: n.pitch))

        event = Event(
            onset=onset, duration=duration, duration_text=text,
            kind=kind, notes=notes, grace=bool(grace), **ctx,
        )

        if grace is None:
            flush(onset)
            events.append(event)
            last_main = onset
            after_rank = 0
            if duration is not None:
                onset += duration
        elif grace.endswith("after"):
            after_rank += 1
            events.append(replace(event, onset=last_main, grace_rank=after_rank))
        else:
            pending.append(event)

    flush(onset)
    return events


def parse_mscx(xml_bytes: bytes, score_path: str) -> list[Event]:
    """
    Parse the musical content of an MSCX file.

    The parser deliberately ignores layout/engraving/metadata elements.
    """
    root = ET.fromstring(xml_bytes)

    part_names = parse_staff_part_names(root)
    events: list[Event] = []

    for staff in iter_master_staves(root):
        staff_id = staff.get("id", "?")
        part_name = part_names.get(staff_id, f"Staff {staff_id}")

        for measure_index, measure in enumerate(
            staff.findall("./Measure"), start=1
        ):
            label = measure.get("number") or str(measure_index)
            voices = measure.findall("./voice") or [measure]

            for voice_index, voice in enumerate(voices, start=1):
                events.extend(parse_voice(voice, dict(
                    score_path=score_path,
                    staff_id=staff_id,
                    part_name=part_name,
                    measure=label,
                    measure_index=measure_index,
                    voice=voice_index,
                )))

    return events


# ---------------------------------------------------------------------------
# Git access
# ---------------------------------------------------------------------------

def git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def git_text(repo: Path, *args: str) -> str:
    return git(repo, *args).decode("utf-8", errors="replace")


def git_file(repo: Path, ref: str, path: str) -> bytes:
    return git(repo, "show", f"{ref}:{path}")


def list_mscx(repo: Path, ref: str) -> set[str]:
    output = git_text(repo, "-c", "core.quotePath=false", "ls-tree", "-r", "--name-only", ref)
    return {
        line.strip()
        for line in output.splitlines()
        if line.strip().endswith(".mscx")
    }


# ---------------------------------------------------------------------------
# Alignment and diff
# ---------------------------------------------------------------------------
#
# Events are NOT matched by absolute onset.  A single changed duration (or an
# inserted/removed note) would shift every later onset in the bar, and every
# following note would be reported as "removed here, added there".  Instead:
#
#   1. measures of each staff are aligned with a sequence diff on content;
#   2. within a paired measure, each voice's events are aligned the same way;
#   3. only the genuinely unmatched stretches are examined for modifications.

@dataclass
class Change:
    measure: str
    part: str
    voice: int
    change_type: str
    edition: str
    source: str
    beat: str = ""


def beat_text(event: Event) -> str:
    text = f"{float(event.onset) + 1:.3g}"
    return f"{text} (grace)" if event.grace else text


def location(event: Event) -> dict:
    return dict(
        measure=event.measure,
        part=event.part_name,
        voice=event.voice,
        beat=beat_text(event),
    )


def event_signature(event: Event) -> tuple:
    return (
        event.kind,
        event.duration,
        event.grace,
        tuple(n.pitch for n in event.notes),
    )


def pitch_list(event: Event) -> list[int]:
    return [note.pitch for note in event.notes]


def format_pitches(pitches: Iterable[int]) -> str:
    return ", ".join(midi_name(p) for p in pitches)


def describe(event: Event) -> str:
    if event.kind == "rest":
        return "rest"
    return f"chord ({format_pitches(pitch_list(event))})"


def compare_notes(edition: Event, source: Event) -> list[Change]:
    """Compare two events that occupy the same slot."""
    changes: list[Change] = []
    loc = location(source)

    if edition.kind != source.kind:
        return [Change(**loc, change_type="Change event",
                       edition=describe(edition), source=describe(source))]

    if edition.duration != source.duration:
        changes.append(Change(**loc, change_type="Change duration",
                              edition=edition.duration_text,
                              source=source.duration_text))

    if edition.kind == "rest":
        return changes

    old_remaining = pitch_list(edition)
    new_remaining = pitch_list(source)

    # Exact matches cancel first.
    for pitch in pitch_list(edition):
        if pitch in new_remaining:
            old_remaining.remove(pitch)
            new_remaining.remove(pitch)

    # Pair what is left as pitch changes; the surplus is add/remove.
    while old_remaining and new_remaining:
        changes.append(Change(
            **loc, change_type="Change pitch",
            edition=midi_name(old_remaining.pop(0)),
            source=midi_name(new_remaining.pop(0)),
        ))
    for pitch in old_remaining:
        changes.append(Change(**loc, change_type="Remove note",
                              edition=midi_name(pitch), source="—"))
    for pitch in new_remaining:
        changes.append(Change(**loc, change_type="Add note",
                              edition="—", source=midi_name(pitch)))
    return changes


def event_as_change(event: Event, source: bool) -> list[Change]:
    """A wholly added (source=True) or removed event."""
    loc = location(event)
    if event.kind == "rest":
        return [Change(
            **loc,
            change_type="Add rest" if source else "Remove rest",
            edition="—" if source else event.duration_text,
            source=event.duration_text if source else "—",
        )]
    return [
        Change(
            **loc,
            change_type="Add note" if source else "Remove note",
            edition="—" if source else note.name,
            source=note.name if source else "—",
        )
        for note in event.notes
    ]


def compare_voice(old: list[Event], new: list[Event]) -> list[Change]:
    changes: list[Change] = []
    matcher = SequenceMatcher(
        None,
        [event_signature(e) for e in old],
        [event_signature(e) for e in new],
        autojunk=False,
    )
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        # Within an unmatched stretch, pair events that share a slot
        # (same onset and grace rank): those are modifications in place.
        candidates: dict[tuple, list[Event]] = defaultdict(list)
        for e in new[j1:j2]:
            candidates[(e.onset, e.grace_rank)].append(e)

        paired: set[int] = set()
        for e in old[i1:i2]:
            slot = candidates.get((e.onset, e.grace_rank))
            if slot:
                partner = slot.pop(0)
                paired.add(id(partner))
                changes.extend(compare_notes(e, partner))
            else:
                changes.extend(event_as_change(e, source=False))
        for e in new[j1:j2]:
            if id(e) not in paired:
                changes.extend(event_as_change(e, source=True))
    return changes


def compare_measure(
    old: dict[int, list[Event]], new: dict[int, list[Event]]
) -> list[Change]:
    changes: list[Change] = []
    for voice in sorted(set(old) | set(new)):
        changes.extend(compare_voice(old.get(voice, []), new.get(voice, [])))
    return changes


def measure_signature(voices: dict[int, list[Event]]) -> tuple:
    return tuple(
        (voice, tuple(event_signature(e) for e in events))
        for voice, events in sorted(voices.items())
    )


def group_measures(events: list[Event]):
    staves: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for e in events:
        staves[e.staff_id][e.measure_index][e.voice].append(e)
    return staves


def compare_events(
    edition_events: list[Event],
    source_events: list[Event],
) -> list[Change]:
    old_staves = group_measures(edition_events)
    new_staves = group_measures(source_events)
    changes: list[Change] = []

    for staff_id in sorted(set(old_staves) | set(new_staves),
                           key=measure_sort_key):
        old = old_staves.get(staff_id, {})
        new = new_staves.get(staff_id, {})
        old_idx, new_idx = sorted(old), sorted(new)

        matcher = SequenceMatcher(
            None,
            [measure_signature(old[i]) for i in old_idx],
            [measure_signature(new[j]) for j in new_idx],
            autojunk=False,
        )
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            overlap = min(i2 - i1, j2 - j1)
            for k in range(overlap):
                changes.extend(compare_measure(
                    old[old_idx[i1 + k]], new[new_idx[j1 + k]]))
            # Whole measures present on only one side.
            for i in old_idx[i1 + overlap:i2]:
                for events in old[i].values():
                    for e in events:
                        changes.extend(event_as_change(e, source=False))
            for j in new_idx[j1 + overlap:j2]:
                for events in new[j].values():
                    for e in events:
                        changes.extend(event_as_change(e, source=True))

    return changes


def measure_sort_key(value: str):
    """
    Sort ordinary measure numbers numerically where possible, while retaining
    things such as '12a' or '23-24'.
    """
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def render_markdown(
    score_path: str,
    edition_ref: str,
    source_ref: str,
    changes: list[Change],
) -> str:
    lines = [
        f"# Changes to `{Path(score_path).name}`",
        "",
        "## Provenance",
        "",
        f"- **Edition:** `{edition_ref}` — upstream / first state",
        f"- **Source:** `{source_ref}` — MALerLab / final state",
        f"- **Score:** `{score_path}`",
        "",
        "This report describes substantive musical differences between "
        "the upstream edition and the final MALerLab source state. "
        "Layout and engraving changes are intentionally excluded. "
        "Beat is counted in quarter notes from the start of the bar.",
        "",
    ]

    if not changes:
        lines.extend(
            [
                "## Changes",
                "",
                "No substantive note/rest changes were detected by the "
                "current parser.",
                "",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "## Changes",
            "",
            "| Measure | Beat | Part | Voice | Change | Edition | Source |",
            "|---:|---:|---|---:|---|---|---|",
        ]
    )

    for change in changes:
        lines.append(
            "| "
            f"{markdown_escape(change.measure)} | "
            f"{markdown_escape(change.beat)} | "
            f"{markdown_escape(change.part)} | "
            f"{change.voice} | "
            f"{markdown_escape(change.change_type)} | "
            f"{markdown_escape(change.edition)} | "
            f"{markdown_escape(change.source)} |"
        )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def git_remote_urls(repo: Path) -> dict[str, str]:
    output = git_text(repo, "remote", "-v")
    remotes: dict[str, str] = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) >= 2 and (len(fields) < 3 or fields[2] == "(fetch)"):
            remotes.setdefault(fields[0], fields[1])
    return remotes


def detect_openscore_remote(repo: Path) -> str | None:
    for name, url in git_remote_urls(repo).items():
        normalized = url.lower().rstrip("/").removesuffix(".git")
        if "openscore" in normalized and "stringquartets" in normalized:
            return name
    return None


def remote_default_ref(repo: Path, remote: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), "symbolic-ref", "--quiet",
         f"refs/remotes/{remote}/HEAD"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip().removeprefix("refs/remotes/")
    return None



SCORES_YAML = "data/scores.yaml"


def read_score_ids(repo: Path, ref: str) -> list[str]:
    """
    Return every top-level key of data/scores.yaml (e.g. "7313978").

    Read from the MALerLab ref so the list matches the scores being compared;
    fall back to the working tree.  The keys are the only unindented
    "<digits>:" lines, so no YAML library is needed.
    """
    try:
        text = git_text(repo, "show", f"{ref}:{SCORES_YAML}")
    except subprocess.CalledProcessError:
        path = repo / SCORES_YAML
        if not path.exists():
            raise FileNotFoundError(
                f"{SCORES_YAML} not found in {ref} or in {repo}"
            )
        text = path.read_text(encoding="utf-8")

    ids: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^['\"]?(\d+)['\"]?\s*:\s*(#.*)?$", line)
        if match:
            ids.append(match.group(1))
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a score in OpenScore and MALerLab repositories. "
                    f"If no score ID is given, compare every score listed in "
                    f"{SCORES_YAML}."
    )
    parser.add_argument(
        "score_id", nargs="?",
        help="Score ID (for example, 7313978 or sq7313978). "
             f"Omit to run all scores in {SCORES_YAML}.",
    )
    parser.add_argument(
        "--malerlab-repo", type=Path, default=Path("."),
        help="Local MALerLab clone (default: current directory).",
    )
    parser.add_argument(
        "--openscore-repo", type=Path,
        help="Local OpenScore clone. If omitted, uses the OpenScore remote "
             "inside the MALerLab clone.",
    )
    parser.add_argument(
        "--openscore-ref",
        help="OpenScore commit/ref (default: detected remote's default branch).",
    )
    parser.add_argument(
        "--malerlab-ref", default="origin/omr-dev",
        help="MALerLab commit/ref (default: origin/omr-dev).",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("change_reports"),
        help="Output directory (default: change_reports).",
    )
    return parser.parse_args()


@dataclass
class Context:
    malerlab_repo: Path
    malerlab_ref: str
    openscore_repo: Path
    openscore_ref: str
    openscore_files: set[str]
    malerlab_files: set[str]
    out: Path


def build_context(args: argparse.Namespace) -> Context | int:
    """Resolve repos and refs once.  Returns an exit code on failure."""
    malerlab_repo = args.malerlab_repo.resolve()
    if not (malerlab_repo / ".git").exists():
        print(f"Not a Git repository: {malerlab_repo}", file=sys.stderr)
        return 2

    remote_name = None
    if args.openscore_repo:
        openscore_repo = args.openscore_repo.resolve()
    else:
        remote_name = detect_openscore_remote(malerlab_repo)
        if remote_name is None:
            print(
                "Could not identify an OpenScore remote in the MALerLab "
                "clone. Add the OpenScore remote or supply --openscore-repo.",
                file=sys.stderr,
            )
            return 2
        # A remote is represented by its remote-tracking refs in this clone.
        openscore_repo = malerlab_repo

    if not (openscore_repo / ".git").exists():
        print(f"Not a Git repository: {openscore_repo}", file=sys.stderr)
        return 2

    openscore_ref = args.openscore_ref
    if openscore_ref is None:
        if remote_name:
            openscore_ref = remote_default_ref(openscore_repo, remote_name)
            if openscore_ref is None:
                # Common fallback if remote HEAD has not been configured.
                for candidate in (f"{remote_name}/main", f"{remote_name}/master"):
                    if subprocess.run(
                        ["git", "-C", str(openscore_repo), "rev-parse",
                         "--verify", candidate],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    ).returncode == 0:
                        openscore_ref = candidate
                        break
        else:
            openscore_ref = "HEAD"

    if not openscore_ref:
        print("Could not determine OpenScore's default branch. Supply "
              "--openscore-ref.", file=sys.stderr)
        return 2

    git(openscore_repo, "rev-parse", "--verify", openscore_ref)
    git(malerlab_repo, "rev-parse", "--verify", args.malerlab_ref)

    return Context(
        malerlab_repo=malerlab_repo,
        malerlab_ref=args.malerlab_ref,
        openscore_repo=openscore_repo,
        openscore_ref=openscore_ref,
        # Listing the tree is slow-ish, so do it once, not per score.
        openscore_files=list_mscx(openscore_repo, openscore_ref),
        malerlab_files=list_mscx(malerlab_repo, args.malerlab_ref),
        out=args.out,
    )


def compare_score(
    ctx: Context, score_id: str, brief: bool = False
) -> tuple[int, str]:
    """
    Compare one score and write its report.

    Returns (status, message): 0 = compared, 1 = skipped (missing/ambiguous).
    ParseError / CalledProcessError propagate to the caller.
    """
    filename = f"sq{score_id}.mscx"
    openscore_matches = sorted(
        p for p in ctx.openscore_files if Path(p).name == filename)
    malerlab_matches = sorted(
        p for p in ctx.malerlab_files if Path(p).name == filename)

    if not openscore_matches or not malerlab_matches:
        missing = []
        if not openscore_matches:
            missing.append(f"OpenScore ({ctx.openscore_ref})")
        if not malerlab_matches:
            missing.append(f"MALerLab ({ctx.malerlab_ref})")
        return 1, f"{filename} not found in " + " and ".join(missing) + "."
    if len(openscore_matches) != 1 or len(malerlab_matches) != 1:
        return 1, (
            f"Ambiguous score ID {score_id}. OpenScore matches: "
            + ", ".join(openscore_matches) + "; MALerLab matches: "
            + ", ".join(malerlab_matches)
        )

    openscore_path = openscore_matches[0]
    malerlab_path = malerlab_matches[0]
    openscore_events = parse_mscx(
        git_file(ctx.openscore_repo, ctx.openscore_ref, openscore_path),
        openscore_path,
    )
    malerlab_events = parse_mscx(
        git_file(ctx.malerlab_repo, ctx.malerlab_ref, malerlab_path),
        malerlab_path,
    )
    changes = compare_events(openscore_events, malerlab_events)

    ctx.out.mkdir(parents=True, exist_ok=True)
    output_path = ctx.out / f"sq{score_id}.changes.md"
    output_path.write_text(
        render_markdown(
            malerlab_path,
            f"OpenScore/{ctx.openscore_ref}",
            f"MALerLab/{ctx.malerlab_ref}",
            changes,
        ),
        encoding="utf-8",
    )
    if brief:
        return 0, f"{len(changes)} changes -> {output_path}"
    return 0, (
        f"Compared {openscore_path} (OpenScore) with {malerlab_path} "
        f"(MALerLab): {len(changes)} changes -> {output_path}"
    )


def main() -> int:
    args = parse_args()

    if args.score_id is not None:
        score_id = args.score_id.removeprefix("sq")
        if not score_id.isdigit():
            print("Score ID must be numeric (optionally prefixed with 'sq').",
                  file=sys.stderr)
            return 2
    else:
        score_id = None

    try:
        ctx = build_context(args)
        if isinstance(ctx, int):
            return ctx

        # Single score: behave exactly as before.
        if score_id is not None:
            status, message = compare_score(ctx, score_id)
            print(message, file=sys.stderr if status else sys.stdout)
            return status

        # Run all.
        score_ids = read_score_ids(ctx.malerlab_repo, ctx.malerlab_ref)
        if not score_ids:
            print(f"No score IDs found in {SCORES_YAML}.", file=sys.stderr)
            return 2

    except (ET.ParseError, subprocess.CalledProcessError,
            FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            print(exc.stderr.decode("utf-8", errors="replace"), file=sys.stderr)
        return 2

    compared, skipped, failed = 0, [], []
    total = len(score_ids)
    for n, sid in enumerate(score_ids, start=1):
        print(f"[{n}/{total}] sq{sid} ", end="", flush=True)
        # One bad score must not abort the batch.
        try:
            status, message = compare_score(ctx, sid, brief=True)
        except (ET.ParseError, subprocess.CalledProcessError) as exc:
            detail = str(exc)
            if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
                detail += " " + exc.stderr.decode("utf-8", "replace").strip()
            print(f"ERROR: {detail}")
            failed.append((sid, detail))
            continue
        print(message)
        if status == 0:
            compared += 1
        else:
            skipped.append((sid, message))

    print(
        f"\nDone: {compared} compared, {len(skipped)} skipped, "
        f"{len(failed)} failed (of {total}). Reports in {ctx.out}/"
    )
    for sid, message in skipped:
        print(f"  skipped sq{sid}: {message}", file=sys.stderr)
    for sid, detail in failed:
        print(f"  failed  sq{sid}: {detail}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
