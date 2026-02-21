#!/usr/bin/env python3
"""Convert a Discord message SQLite database to a plain-text corpus file.

Output format: UTF-8 plain text, one utterance per line.
This is the most universally compatible format for NLP pipelines —
readable by our own build_vocab/train scripts, NLTK, Gensim, word2vec,
Hugging Face datasets (with a trivial one-liner loader), and any other
tool that accepts a text corpus.

Usage
-----
    python scripts/discord_to_corpus.py messages.sqlite
    python scripts/discord_to_corpus.py messages.sqlite -o corpus.txt
    python scripts/discord_to_corpus.py messages.sqlite -o corpus.txt --stats

The source database is never modified.

Cleaning pipeline (applied in order)
--------------------------------------
1.  Skip non-text rows        — SQLite allows any type in a TEXT column;
                                 only typeof='text' rows are processed.
2.  Skip bot commands         — messages whose first non-space character is
                                 !, /, or . are typically bot invocations.
3.  Remove code blocks        — ```...``` blocks are dropped entirely;
                                 code is not useful for word prediction.
4.  Remove inline code        — `...` backtick spans are dropped entirely;
                                 identifiers / commands add noise.
5.  Remove spoiler markup     — ||text|| → text (keep the hidden content).
6.  Remove Discord mentions   — <@id>, <@!id>, <@&id> dropped entirely.
7.  Remove channel refs       — <#id> dropped entirely.
8.  Remove custom emoji       — <:name:id> and <a:name:id> dropped entirely.
9.  Remove URLs               — http(s):// links dropped entirely.
10. Strip blockquotes         — leading "> " stripped from each line.
11. Strip Markdown formatting — **bold**, *italic*, __under__, ~~strike~~
                                 markup characters removed, text kept.
12. Normalise whitespace      — runs of whitespace collapsed to single space,
                                 result stripped of leading/trailing space.
13. Require ≥ MIN_ALPHA_WORDS words composed purely of ASCII letters
                                 (default: 2).  This drops emoji-only messages,
                                 single-word utterances, and fragments that
                                 lost all meaning after earlier steps.

Nothing is lowercased here — the downstream vocabulary builder handles that,
so the raw file stays useful for case-sensitive tools too.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

MIN_ALPHA_WORDS: int = 2          # minimum purely-alphabetic words to keep
DEFAULT_OUTPUT   = "corpus.txt"   # output filename if -o not given

# Characters that indicate a bot command prefix
BOT_COMMAND_PREFIXES = frozenset("!/.")


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

# Compiled once at module load for speed across 1M+ messages
_RE_CODE_BLOCK   = re.compile(r'```.*?```',          re.DOTALL)
_RE_INLINE_CODE  = re.compile(r'`[^`\n]+`')
_RE_SPOILER      = re.compile(r'\|\|(.+?)\|\|',      re.DOTALL)
_RE_MENTION_USER = re.compile(r'<@!?\d+>')
_RE_MENTION_ROLE = re.compile(r'<@&\d+>')
_RE_CHANNEL      = re.compile(r'<#\d+>')
_RE_CUSTOM_EMOJI = re.compile(r'<a?:[a-zA-Z0-9_]+:\d+>')
_RE_URL          = re.compile(r'https?://\S+')
# Bare URLs lacking a protocol: word.tld/something  (must have a path segment
# to avoid false-positives on  "e.g." or "U.S.")
_RE_BARE_URL     = re.compile(r'\b\w+\.\w{2,6}/\S*')
_RE_BLOCKQUOTE   = re.compile(r'(?m)^>\s?')          # leading "> " per line
_RE_BOLD         = re.compile(r'\*\*(.+?)\*\*',      re.DOTALL)
_RE_ITALIC_STAR  = re.compile(r'\*(.+?)\*',          re.DOTALL)
_RE_ITALIC_UNDER = re.compile(r'__(.+?)__',          re.DOTALL)
_RE_UNDERLINE    = re.compile(r'_(.+?)_',             re.DOTALL)
_RE_STRIKE       = re.compile(r'~~(.+?)~~',           re.DOTALL)
_RE_WHITESPACE   = re.compile(r'[ \t\r\n]+')


_DROP_BOT      = "bot_command"
_DROP_EMPTY    = "empty_after_clean"
_DROP_FEW      = "too_few_alpha_words"


def clean(raw: str) -> tuple[str, None] | tuple[None, str]:
    """Apply the full cleaning pipeline to a single message.

    Returns ``(cleaned_text, None)`` if the message should be kept, or
    ``(None, drop_reason)`` if it should be discarded.  Having a single
    canonical implementation here means callers never drift out of sync.
    """
    # Step 2: bot commands
    stripped = raw.lstrip()
    if stripped and stripped[0] in BOT_COMMAND_PREFIXES:
        return None, _DROP_BOT

    text = raw

    # Step 3: code blocks (drop content entirely, including unclosed opening ```)
    text = _RE_CODE_BLOCK.sub('', text)
    text = text.replace('```', '')  # remove any unclosed fence left behind

    # Step 4: inline code (drop content entirely)
    text = _RE_INLINE_CODE.sub('', text)

    # Step 5: spoilers — keep the hidden text
    text = _RE_SPOILER.sub(r'\1', text)

    # Steps 6-9: Discord-specific syntax
    text = _RE_MENTION_USER.sub('', text)
    text = _RE_MENTION_ROLE.sub('', text)
    text = _RE_CHANNEL.sub('', text)
    text = _RE_CUSTOM_EMOJI.sub('', text)
    text = _RE_URL.sub('', text)
    # Bare URLs lacking a protocol: word.tld/path  (path segment required
    # to avoid false-positives on abbreviations like "e.g." or "U.S.")
    text = _RE_BARE_URL.sub('', text)

    # Step 10: blockquotes
    text = _RE_BLOCKQUOTE.sub('', text)

    # Step 11: Markdown — strip markup, keep text
    # Bold must come before italic (** before *)
    text = _RE_BOLD.sub(r'\1', text)
    text = _RE_ITALIC_STAR.sub(r'\1', text)
    text = _RE_ITALIC_UNDER.sub(r'\1', text)
    text = _RE_UNDERLINE.sub(r'\1', text)
    text = _RE_STRIKE.sub(r'\1', text)

    # Step 12: normalise whitespace
    text = _RE_WHITESPACE.sub(' ', text).strip()

    if not text:
        return None, _DROP_EMPTY

    # Step 13: require a minimum number of purely-alphabetic words
    alpha_words = [w for w in text.split() if w.isalpha()]
    if len(alpha_words) < MIN_ALPHA_WORDS:
        return None, _DROP_FEW

    return text, None


# ---------------------------------------------------------------------------
# Main conversion logic
# ---------------------------------------------------------------------------

def convert(
    db_path: Path,
    output_path: Path,
    table: str = "messages",
    content_col: str = "contents",
    verbose: bool = True,
    print_stats: bool = False,
) -> dict:
    """Read messages from db_path, clean them, and write corpus to output_path.

    Returns a dict of statistics.
    """
    stats = {
        "input_total_rows": 0,
        "skipped_non_text": 0,
        "skipped_empty_input": 0,
        "dropped_bot_command": 0,
        "dropped_empty_after_clean": 0,
        "dropped_too_few_alpha_words": 0,
        "kept": 0,
    }

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)  # read-only
    cur = conn.cursor()

    # Count total rows for progress reporting
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    stats["input_total_rows"] = cur.fetchone()[0]

    if verbose:
        print(f"Reading from {db_path}  ({stats['input_total_rows']:,} total rows)")
        print(f"Writing to   {output_path}")
        print()

    # Fetch only text-typed, non-empty rows — this avoids bringing integer/real
    # values (message IDs stored in wrong column, etc.) into the pipeline.
    cur.execute(
        f"SELECT {content_col} FROM {table} "
        f"WHERE typeof({content_col}) = 'text' AND {content_col} != ''"
    )
    stats["skipped_non_text"] = (
        stats["input_total_rows"]
        - cur.execute(
            f"SELECT COUNT(*) FROM {table} "
            f"WHERE typeof({content_col}) = 'text' AND {content_col} != ''"
        ).fetchone()[0]
    )
    # Re-run the actual fetch
    cur.execute(
        f"SELECT {content_col} FROM {table} "
        f"WHERE typeof({content_col}) = 'text' AND {content_col} != ''"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    counters: dict[str, int] = {
        "kept": 0,
        _DROP_BOT:   0,
        _DROP_EMPTY: 0,
        _DROP_FEW:   0,
    }

    with output_path.open('w', encoding='utf-8') as out:
        for i, (content,) in enumerate(cur):
            text, reason = clean(content)
            if text is not None:
                out.write(text)
                out.write('\n')
                counters["kept"] += 1
            else:
                counters[reason] += 1

            if verbose and (i + 1) % 100_000 == 0:
                print(f"  processed {i+1:,} …  kept {counters['kept']:,}")

    conn.close()

    total_text = sum(counters.values())
    stats["kept"]                        = counters["kept"]
    stats["dropped_bot_command"]         = counters[_DROP_BOT]
    stats["dropped_empty_after_clean"]   = counters[_DROP_EMPTY]
    stats["dropped_too_few_alpha_words"] = counters[_DROP_FEW]
    stats["skipped_non_text"]            = stats["input_total_rows"] - total_text

    if verbose:
        _print_stats(stats)

    if print_stats:
        stats_path = output_path.with_suffix('.stats.json')
        stats_path.write_text(json.dumps(stats, indent=2))
        print(f"\nStats written to {stats_path}")

    return stats


def _print_stats(stats: dict) -> None:
    total = stats["input_total_rows"]
    kept  = stats["kept"]

    print()
    print("─" * 50)
    print(f"  Total rows in database:      {total:>10,}")
    print(f"  Skipped (non-text / empty):  {stats['skipped_non_text']:>10,}  "
          f"({100*stats['skipped_non_text']/max(total,1):.1f}%)")
    print(f"  Dropped (bot commands):      {stats['dropped_bot_command']:>10,}  "
          f"({100*stats['dropped_bot_command']/max(total,1):.1f}%)")
    print(f"  Dropped (empty after clean): {stats['dropped_empty_after_clean']:>10,}  "
          f"({100*stats['dropped_empty_after_clean']/max(total,1):.1f}%)")
    print(f"  Dropped (< {MIN_ALPHA_WORDS} alpha words):  {stats['dropped_too_few_alpha_words']:>10,}  "
          f"({100*stats['dropped_too_few_alpha_words']/max(total,1):.1f}%)")
    print(f"  ─────────────────────────────────────────")
    print(f"  Kept:                        {kept:>10,}  "
          f"({100*kept/max(total,1):.1f}%)")
    print("─" * 50)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a Discord message SQLite DB to a plain-text corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "database",
        metavar="DB",
        help="Path to the SQLite database file (opened read-only)",
    )
    parser.add_argument(
        "-o", "--output",
        metavar="FILE",
        default=None,
        help=f"Output corpus file (default: <db stem>.corpus.txt)",
    )
    parser.add_argument(
        "--table",
        default="messages",
        help="Table name (default: messages)",
    )
    parser.add_argument(
        "--content-col",
        default="contents",
        dest="content_col",
        help="Column containing message text (default: contents)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Write a JSON stats file alongside the corpus",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args(argv)

    db_path = Path(args.database)
    if not db_path.exists():
        print(f"ERROR: database not found: {db_path}", file=sys.stderr)
        return 1

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = db_path.with_suffix('.corpus.txt')

    convert(
        db_path=db_path,
        output_path=output_path,
        table=args.table,
        content_col=args.content_col,
        verbose=not args.quiet,
        print_stats=args.stats,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
