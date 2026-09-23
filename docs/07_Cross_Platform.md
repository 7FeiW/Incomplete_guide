# Cross-Platform Development

Research code often moves between Windows, macOS, Linux, and HPC systems. This
section explains how to keep paths, commands, environments, encodings, and
tests portable across those environments.

<!-- TODO: Add practical cross-platform guidance and examples. -->

## What to Ask an LLM AGNET

Avoid: “Make this run everywhere.”

Ask instead:

```text
Inspect <scope> for operating-system assumptions in paths, shell commands,
encodings, line endings, file permissions, optional dependencies, and tests.
Compare them with the platforms we need to support: <platforms>. Report each
issue with the affected path, why it fails on a platform, and the smallest
portable alternative or a clearly documented limitation. Do not change commands,
dependencies, CI configuration, or platform support promises until I approve.
```
