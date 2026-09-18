# Content safety lexicons

Each `.txt` file represents one category; the filename is the category
name. A category's severity is defined in the
`content_safety.category_severity` section of `config/settings.yaml`.

## Format

- One **term root** per line (the form before any suffix)
- Lines starting with `#` are comments
- Roots must be at least 4 characters long (shorter roots produce false
  positives)
- Suffixes are tolerated automatically: the root `idiot` also catches
  derivatives like `idiotic`, `idiots` (root + up to 6 extra characters)
- Character substitution (a→@, i→1), inter-letter separators (i.d.i.o.t),
  and letter repetition (idiooot) are normalized by the scanner; there is
  no need to also write out these variants

## Empty files

A category left empty is considered **inactive** and is reported under
`inactive_categories` in the scan result. The system does not error, but
that category goes unmeasured — this is shown explicitly in the report.

## Categories requiring organizational approval

The content of `religious_insult.txt` and similar categories carrying
cultural context **must be defined by the organization**. The line
between an academic or critical statement about religion and an insult
is not a technical decision. It is recommended that written guidance
consisting of acceptable and unacceptable examples be obtained before
this file is filled in.
