# Security

piiscanner is pointed at databases full of personal data, so a security
problem in it matters more than most. Please report one privately, not in a
public issue.

**Email:** shubhamjjw14@gmail.com, with "piiscanner security" in the subject.

Say what you found, how to reproduce it, and which version
(`piiscanner --version`). You will get a reply within a week. Please give a
fix a reasonable time before you make it public.

## In scope

Anything that breaks one of the guarantees in the README:

- a write reaching the database (any statement other than a read)
- a raw sampled value written to a findings file, a report or a log
- a network connection to anything other than the database being scanned
- a scan that reads far more than its sample bounds

## Not in scope

Findings that are wrong (a column reported or missed). Those are bugs, and
welcome as ordinary issues -- with the column's name and type, never its
values.
