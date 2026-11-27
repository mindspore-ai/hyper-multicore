---
name: megakernel-dumper
description: >-
  Use HyperMulticore Dumper to locate MegaKernel numerical differences across two runs.
  Applies when configuring Kernel or Task dump, interpreting Native logs and summary
  CSV files, or comparing aligned Torch snapshots.
---

# MegaKernel Dumper

Read the repository's [diagnostics guide](../../../docs/diagnostics.md#采集张量快照) for configuration and supported operators.

1. Pair the Native and MegaKernel calls by model location, training step, microbatch,
   direction, and shard. Check that the jobs used the same weights, inputs, and route.
   A Dumper invocation number belongs to its own capture.
2. Start with Kernel dump on the affected call. Use `csv + log` for stage triage
   against Native prints, or `tensor + csv` when exact comparison is likely.
   Use `stages` and `io` to select consecutive stage ports. Narrow the capture to
   the relevant ranks and steps.
3. Compare the same stage port and logical range. In MegaMoE, use expert IDs and row
   spans to align replicated Native rows with local MegaMoE rows. Check token order
   before comparing elements. Use the recorded capture phase to distinguish a value
   at Kernel completion from a Task boundary value. Record unknown coverage or missing shards as such.
4. Report the earliest supported mismatch and the evidence. Printed values and statistics
   can flag a discrepancy. Use `dumper.load_tensor` and `dumper.compare_tensors` on the
   aligned CPU snapshots for an element count, coordinate, and both values.
5. Choose Task dump only when a stage is already implicated and a task-local input or
   output will resolve the next question. Prefer the downstream input after dependencies
   for collaborative GMM outputs. Use the same run's trace or artifact to identify the worker.
   Check the reported coverage before interpreting compact or truncated records.
