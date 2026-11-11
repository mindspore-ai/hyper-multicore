# A5 MegaMoE and the external model boundary

This branch extends the public standalone package. Default automatic-token BF16 use,
EIO/EOI weight layout, the framework adapter selector and the source-locked build remain.
An optional `local_num_tokens` constructor argument selects a fixed workload for downstream
integrations that already determine their input after sequence parallelism. Omitting it
preserves the upstream automatic-token behavior.

## Build and activate

For a machine with no former developer workspace or dependency cache, follow
[the Git-only reconstruction guide](rebuilding_from_git.md). The qualified source
and its patches are tracked; old runtime archives are not build inputs.

The qualification stack is Python 3.11.10, torch 2.12.0+cpu, torch_npu 2.12.0.post3
and CANN 9.2.0. Activate that environment before building:

```bash
bash build.sh --soc-list ascend950 --jobs 8 --native-only
source activate.bash
```

A source checkout owns its Python code and `build/native/payload` output. Copy both to
the destination and source its activation file again; no wheel installation or former
build path is required. Activation never builds, downloads or installs anything.
A5 and A2/A3 use separate private SHMEM backend families; do not combine them in one
SDK selection. No system CANN installation is overwritten. Build products stay in
ignored `build/` and `dist/` directories. The source-build payload selects the matching
framework ABI. Importing the Python classes alone is lazy.

BF16 retains `MegaMoeExperts` and the existing `aclnnHyperMegaMoe`/Grad interface.
MX uses `MXFP8MegaMoeExperts` with `aclnnHyperMegaMoeMxfp8` and
`aclnnHyperMegaMoeMxfp8Grad`. Only the primary forward/backward MX operators are built;
historical standalone ALL/single-vector/roles/solver diagnostic entries are excluded.
The qualified window2 worker keeps its existing expert/task boundaries.

## Required transport and recommended controls

A5 production MegaMoE tasks require `HYPER_MULTICORE_SHMEM_DATA_ENGINE=mte_udma`.
An uninitialized UDMA bank is rejected before launch. A2/A3 BF16 retains MTE support.

The deployed MX policy is configured before constructing the executor:

```bash
export HYPER_MULTICORE_SHMEM_DATA_ENGINE=mte_udma
export HYPER_MULTICORE_SHMEM_UDMA_QP_NUM=1
export HYPER_MULTICORE_SHMEM_HEAP_SIZE=1073741824
export HYPER_MULTICORE_MX_AIV_POLICY=window2
export HYPER_MULTICORE_MX_GMM_TILE=t6
export HYPER_MULTICORE_MX_COMM_SPLIT=512
export HYPER_MULTICORE_MX_READY_DEPTH=8
export HYPER_MULTICORE_MX_SWIGLU_ROW_ONLY=1
export HYPER_MULTICORE_MX_NOGRAD_NO_CLONE=1
export HYPER_MULTICORE_MOE_COUNT_POLICY=histc
export HYPER_MULTICORE_MOE_SHARE_EXECUTION_RESOURCES=1
export SHMEM_LOG_LEVEL=ERROR
```

BF16 A5 scheduled execution shares QP1 transport planning with MX. Select before constructing it:

```bash
export HYPER_MULTICORE_BF16_GMM_POLICY=blaze_bwd_lanes_a5
export HYPER_MULTICORE_BF16_ACTIVATION_POLICY=a5
export HYPER_MULTICORE_BF16_NOGRAD_NO_COPY=1
export HYPER_MULTICORE_BF16_SCHEDULE=window2
```

The standalone BF16 default remains `legacy_odd`/generic, including automatic tokens
and EIO/EOI layouts. MX policy alone does not change BF16 scheduling. Legacy BF16
retains its existing QP32 requirement. The explicit A5 window2/half16 paths require
32 physical AIC and QP1; no arbitrary A2/A3 scheduling change is implied.
At small EP each remote peer has one owner; spare odd AIV perform independent
self copies. At EP64 all odd owner lanes are occupied and the established mapping
is retained. Do not infer32 simultaneous remote payload movers from32 configured
communication lanes, and do not remove the shared-producer safety check.

BF16 and MX reuse the same transport planner/owner validator, common split control,
Put/Get completion and source-lifetime protocol. Their mathematical DAGs and joins
remain precision-specific. QP1 does not imply a slower dtype or packed FP8 token
communication; routed payloads in both paths are BF16.

Set `HYPER_MULTICORE_MEGAMOE_DISPATCH_MODE=pull` for Pull; combine remains Push.
Push accepts receive-capacity factors. Pull uses source/routed symmetric buffers,
live receive allocations and pointer-free plans against a route-independent logical
bound. It does not reserve the logical bound as a receive tensor. Metadata byte offsets
remain uint64; descriptor dimensions retain their uint32 bounds.

Communication split 4096 is retained for historical cluster reproduction. The normal
recommended downstream communication split is 512. Ready-depth 0/1/8 and no-grad row-only
remain explicit controls; they do not reorder communication tasks. Row-only affects
only no-grad SwiGLU forward. Retained forward/recompute and backward retain needed axes.

## Model-owned parameters

The package has no downstream-model dependency. MX accepts original torch tensor roots exposing
`col_data`, `col_scale`, `row_data` and `row_scale`. Physical data is E4M3FN; scales
are packed uint8 E8M0 pairs for the original 32-element groups. Forward borrows current
column storage, and backward resolves both axes again after FSDP gathers. Logical
shapes and physical scale layouts are validated without BF16 weight conversion.

Downstream adapters own model-specific tensor types, forward-direction checks, TP/SP
input length, routing, shared experts, checkpoint hooks and loss. Any legacy environment
aliases belong in those adapters. This standalone library reads `HYPER_MULTICORE_*`
controls and never assumes a model repository location.

## Source and validation boundary

Standard kernel closure still uses locked source adapters. Selected BF16 Blaze headers
come from installed CANN 9.2; selected MX Blaze headers retain their separately recorded
ops-transformer source commit. JSON receipts include original/adapted SHA256. Do not
infer identity with a native runtime branch merely from operator names.

SHMEM v1.7.0 is built privately with explicit Mesh selection and the upstream d942c05
queue publication/cache flush repair. Device inline consumers are rebuilt, not only
host libraries. Production fault observers, startup probes and per-task diagnostic
logging are excluded. Generic internal profiling is opt-in and separate from normal jobs.

The sibling experiment archive records source/build identities, the original A5-base
audit, CPU gates and targeted device/whole-model outcomes. A local pass is not a new
EP64 or multi-node run. Preserve the existing numerical/lifecycle gates when extending
this branch and use matched normal jobs for performance claims.

## Migration qualification

The final source build completed on the pinned stack. CPU qualification passed321
cases plus1860 subtests. Installed-wheel BF16 matched the upstream reference for full
outputs, all gradients and optimizer updates in EIO/EOI layouts with balanced, empty,
skewed and single-destination routing. MX Push/Pull full-output and four-gradient
hashes matched the frozen embedded implementation. Whole-empty receiving ranks were
also checked for exact-zero inactive expert gradients.

Downstream native MX, external Mega MX and external Mega BF16 each completed a three-step
four-device functional job. A selected training step/layer/micro capture exported
forward, recompute and backward with zero dropped records and continued training.
The final installed wheel also completed a replay after its source build directory
was moved away. These are functional migration gates, not steady-state speedup claims.
No new multi-node or A3 device run is implied.


## Optional BF16 half-Cube scheduling

`HYPER_MULTICORE_BF16_SCHEDULE=half16` is an explicit, default-disabled alternative
to window2. Physical AIC0–15 handle zero-based local experts0/2/4, while AIC16–31
handle1/3/5, each in Up -> activation-ready -> Down order before its next expert.
Each expert has16 logical lanes with complete tile coverage; no output is dropped.
Forward and recomputation change; backward remains32-lane. AIV transport/activation
ownership and completion stay unchanged. Cube ready lookahead is disabled inside
each half group to preserve its strict sequence; vector lookahead keeps its setting.

Prior local module data did not show a half16 gain; it is retained for explicit
workload comparisons rather than promoted as a default. Profile task counts/owners
per expert, not only the metadata reporting32 active physical Cube workers.
A shorter Combine WAIT is not proof of faster complete MoE or step time.

The shared communication split control now applies to BF16 as well. Environment
Pull ignores downstream Push-reservation factors, while an explicit independent
BF16 `dispatch_mode='pull'` call without that environment selection keeps the
original capacity-argument validation. Preserve source/ctx/stream lifetime gates.

After applying these source changes, rebuild with the documented native build
command. Old build products do not acquire the new16-lane/CSR/device VF code by
copying only Python files. No former workspace or binary archive is required;
no system CANN file is changed. Source-only commits intentionally contain no
compiled objects or temporary fault diagnostics.
