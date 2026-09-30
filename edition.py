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
- added / removed notes
- changed pitch
- added / removed rests
- changed duration

Typical use:

    python summarize_mscx_changes.py \
        --repo /path/to/ossq-omr \
        --edition-ref <FIRST_REF> \
        --source-ref <FINAL_REF> \
        --out changes

If --path is omitted, all sq*.mscx files present in both refs are compared.

No requirements (Python's standard library only).

The output is one Markdown file per requested score.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET

from collections import defaultdict
from dataclasses import dataclass
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


def duration_from_element(element: ET.Element) -> Fraction | None:
    """
    Return the nominal duration of a Chord/Rest.

    MuseScore normally stores durationType. Some measure rests also have an
    explicit <duration> such as 4/4; that is used as a fallback.
    """
    duration_type = element.findtext("durationType")
    if duration_type in DURATION_QUARTERS:
        value = DURATION_QUARTERS[duration_type]

        dots = len(element.findall("dot"))
        if dots:
            # 1 dot = 1 + 1/2
            # 2 dots = 1 + 1/2 + 1/4
            multiplier = Fraction(2) - Fraction(1, 2**dots)
            value *= multiplier

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
    """Human-readable duration."""
    duration_type = element.findtext("durationType") or "unknown"
    dots = len(element.findall("dot"))

    if dots == 1:
        return f"{duration_type} dotted"
    if dots == 2:
        return f"{duration_type} double-dotted"

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
    measure: str
    voice: int
    onset: Fraction
    duration: Fraction | None
    duration_text: str
    kind: str                  # "chord" or "rest"
    notes: tuple[Note, ...]


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
            measure_number = measure.get("number") or str(measure_index)

            voices = measure.findall("./voice")

            # In many MSCX files there is no <voice> wrapper for the
            # single/default voice.
            if not voices:
                voices = [measure]

            for voice_index, voice in enumerate(voices, start=1):
                onset = Fraction(0)

                # We only care about Chord and Rest. Everything else in a
                # measure (LayoutBreak, Dynamic, Harmony, etc.) is ignored.
                for element in voice:
                    if element.tag not in ("Chord", "Rest"):
                        continue

                    duration = duration_from_element(element)
                    duration_text = duration_label(element)

                    if element.tag == "Rest":
                        event = Event(
                            score_path=score_path,
                            staff_id=staff_id,
                            part_name=part_name,
                            measure=measure_number,
                            voice=voice_index,
                            onset=onset,
                            duration=duration,
                            duration_text=duration_text,
                            kind="rest",
                            notes=(),
                        )
                    else:
                        pitches = []

                        for note_element in element.findall("./Note"):
                            pitch = parse_pitch(note_element)
                            if pitch is not None:
                                pitches.append(Note(pitch))

                        event = Event(
                            score_path=score_path,
                            staff_id=staff_id,
                            part_name=part_name,
                            measure=measure_number,
                            voice=voice_index,
                            onset=onset,
                            duration=duration,
                            duration_text=duration_text,
                            kind="chord",
                            notes=tuple(sorted(pitches, key=lambda n: n.pitch)),
                        )

                    events.append(event)

                    # Grace notes do not consume rhythmic time.
                    is_grace = any(
                        child.tag.startswith("grace")
                        for child in element
                    )

                    if duration is not None and not is_grace:
                        onset += duration

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
    output = git_text(repo, "ls-tree", "-r", "--name-only", ref)
    return {
        line.strip()
        for line in output.splitlines()
        if line.strip().endswith(".mscx")
        and "/scores/" in f"/{line}"
    }


# ---------------------------------------------------------------------------
# Event matching and diff
# ---------------------------------------------------------------------------

@dataclass
class Change:
    measure: str
    part: str
    voice: int
    change_type: str
    edition: str
    source: str


def event_key(event: Event) -> tuple:
    """
    Location key for a musical event.

    We intentionally do not include pitch here: pitch is one of the things
    we want to detect as changed.
    """
    return (
        event.staff_id,
        event.measure,
        event.voice,
        event.onset,
    )


def pitch_list(event: Event) -> list[int]:
    return [note.pitch for note in event.notes]


def format_pitches(pitches: Iterable[int]) -> str:
    return ", ".join(midi_name(p) for p in pitches)


def compare_notes(
    edition: Event,
    source: Event,
) -> list[Change]:
    changes: list[Change] = []

    location = dict(
        measure=source.measure,
        part=source.part_name,
        voice=source.voice,
    )

    if edition.kind != source.kind:
        changes.append(
            Change(
                **location,
                change_type="Change event",
                edition=(
                    f"{edition.kind} "
                    f"({format_pitches(pitch_list(edition))})"
                    if edition.kind == "chord"
                    else "rest"
                ),
                source=(
                    f"{source.kind} "
                    f"({format_pitches(pitch_list(source))})"
                    if source.kind == "chord"
                    else "rest"
                ),
            )
        )
        return changes

    if edition.duration != source.duration:
        changes.append(
            Change(
                **location,
                change_type="Change duration",
                edition=edition.duration_text,
                source=source.duration_text,
            )
        )

    if edition.kind == "rest":
        return changes

    old_pitches = pitch_list(edition)
    new_pitches = pitch_list(source)

    # Exact matches cancel first.
    old_remaining = old_pitches.copy()
    new_remaining = new_pitches.copy()

    for pitch in old_pitches:
        if pitch in new_remaining:
            old_remaining.remove(pitch)
            new_remaining.remove(pitch)

    # If the number of remaining notes is equal, pair them as pitch changes.
    while old_remaining and new_remaining:
        old_pitch = old_remaining.pop(0)
        new_pitch = new_remaining.pop(0)

        changes.append(
            Change(
                **location,
                change_type="Change pitch",
                edition=midi_name(old_pitch),
                source=midi_name(new_pitch),
            )
        )

    for pitch in old_remaining:
        changes.append(
            Change(
                **location,
                change_type="Remove note",
                edition=midi_name(pitch),
                source="—",
            )
        )

    for pitch in new_remaining:
        changes.append(
            Change(
                **location,
                change_type="Add note",
                edition="—",
                source=midi_name(pitch),
            )
        )

    return changes


def compare_events(
    edition_events: list[Event],
    source_events: list[Event],
) -> list[Change]:
    """
    Compare events by staff/measure/voice/onset.

    This is deliberately conservative. If event structure is ambiguous, the
    output favours explicit add/remove events over pretending to know that a
    particular note was transformed into another.
    """
    old = defaultdict(list)
    new = defaultdict(list)

    for event in edition_events:
        old[event_key(event)].append(event)

    for event in source_events:
        new[event_key(event)].append(event)

    changes: list[Change] = []

    all_keys = sorted(
        set(old) | set(new),
        key=lambda k: (
            k[0],
            measure_sort_key(k[1]),
            k[2],
            k[3],
        ),
    )

    for key in all_keys:
        old_events = old[key]
        new_events = new[key]

        # Straightforward one-to-one case.
        if len(old_events) == 1 and len(new_events) == 1:
            changes.extend(compare_notes(old_events[0], new_events[0]))
            continue

        # Events appearing only in the edition.
        for event in old_events:
            changes.extend(event_as_change(event, source=False))

        # Events appearing only in the source.
        for event in new_events:
            changes.extend(event_as_change(event, source=True))

        # Multiple events at exactly the same location are intentionally not
        # guessed at. This is unusual, but can arise with unusual MSCX data.
        if old_events and new_events:
            changes.append(
                Change(
                    measure=(new_events[0] if new_events else old_events[0]).measure,
                    part=(new_events[0] if new_events else old_events[0]).part_name,
                    voice=(new_events[0] if new_events else old_events[0]).voice,
                    change_type="Unresolved musical difference",
                    edition=f"{len(old_events)} event(s)",
                    source=f"{len(new_events)} event(s)",
                )
            )

    return changes


def event_as_change(event: Event, source: bool) -> list[Change]:
    """
    Convert a wholly added/removed event into editorial changes.
    """
    location = dict(
        measure=event.measure,
        part=event.part_name,
        voice=event.voice,
    )

    if event.kind == "rest":
        return [
            Change(
                **location,
                change_type="Add rest" if source else "Remove rest",
                edition="—" if source else event.duration_text,
                source=event.duration_text if source else "—",
            )
        ]

    result = []

    for pitch in event.notes:
        result.append(
            Change(
                **location,
                change_type="Add note" if source else "Remove note",
                edition="—" if source else midi_name(pitch.pitch),
                source=midi_name(pitch.pitch) if source else "—",
            )
        )

    return result


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
        "Layout and engraving changes are intentionally excluded.",
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
            "| Measure | Part | Voice | Change | Edition | Source |",
            "|---:|---|---:|---|---|---|",
        ]
    )

    for change in changes:
        lines.append(
            "| "
            f"{markdown_escape(change.measure)} | "
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize musical changes between two MSCX Git states."
    )

    parser.add_argument(
        "--repo",
        type=Path,
        required=True,
        help="Path to the local Git clone.",
    )

    parser.add_argument(
        "--edition-ref",
        required=True,
        help="Git ref representing the upstream/first state.",
    )

    parser.add_argument(
        "--source-ref",
        required=True,
        help="Git ref representing the final MALerLab/source state.",
    )

    parser.add_argument(
        "--path",
        action="append",
        help=(
            "Specific .mscx path to compare. May be supplied more than once. "
            "If omitted, all common scores are compared."
        ),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("change_reports"),
        help="Output directory (default: change_reports).",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()

    if not (repo / ".git").exists():
        print(f"Not a Git repository: {repo}", file=sys.stderr)
        return 2

    try:
        # Verify refs before doing any work.
        git(repo, "rev-parse", "--verify", args.edition_ref)
        git(repo, "rev-parse", "--verify", args.source_ref)

        if args.path:
            paths = set(args.path)
        else:
            edition_paths = list_mscx(repo, args.edition_ref)
            source_paths = list_mscx(repo, args.source_ref)
            paths = edition_paths & source_paths

        if not paths:
            print("No common .mscx files found.", file=sys.stderr)
            return 1

        args.out.mkdir(parents=True, exist_ok=True)

        total_changes = 0

        for path in sorted(paths):
            try:
                edition_xml = git_file(repo, args.edition_ref, path)
                source_xml = git_file(repo, args.source_ref, path)

                edition_events = parse_mscx(edition_xml, path)
                source_events = parse_mscx(source_xml, path)

                changes = compare_events(
                    edition_events,
                    source_events,
                )

                output_name = (
                    Path(path).name.removesuffix(".mscx")
                    + ".changes.md"
                )
                output_path = args.out / output_name

                output_path.write_text(
                    render_markdown(
                        path,
                        args.edition_ref,
                        args.source_ref,
                        changes,
                    ),
                    encoding="utf-8",
                )

                total_changes += len(changes)

                print(
                    f"{path}: {len(changes)} changes -> {output_path}"
                )

            except (ET.ParseError, subprocess.CalledProcessError) as exc:
                print(
                    f"ERROR processing {path}: {exc}",
                    file=sys.stderr,
                )

        print(
            f"\nProcessed {len(paths)} score(s); "
            f"reported {total_changes} change(s)."
        )

    except subprocess.CalledProcessError as exc:
        print(
            "Git command failed. Make sure the refs exist in the local clone "
            "and that this is a complete repository clone.",
            file=sys.stderr,
        )
        if exc.stderr:
            print(
                exc.stderr.decode("utf-8", errors="replace"),
                file=sys.stderr,
            )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
