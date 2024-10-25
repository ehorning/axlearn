# Copyright © 2024 Apple Inc.
"""Runtime and compiler options for JAX/XLA."""
# This module must not depend on any jax/axlearn modules so that
# importing this module does not result in initializing jax.
import re
from typing import Union


def default_xla_options(
    *, instance_type: str, num_slices: int, backend: str
) -> dict[str, Union[str, bool]]:
    """Return the default flags for the given instance type and backend.

    These options can be passed to `jitted_fn.lower(...).compile(compiler_options=...)`
    or converted to flags using `xla_flags_from_options` and passed to
    `LIBTPU_INIT_ARGS` (only works on TPU) or `XLA_FLAGS` (works on any platform including TPU)
    before importing jax.

    Args:
        instance_type: A specifier for the ML accelerator. E.g., "tpu-v5p-2048".
        num_slices: The number of slices of the given instance type.
        backend: The jax backend. E.g., "tpu".

    Returns:
        A dictionary with the XLA flags and values.

    Raises:
        NotImplementedError if the instance type and backend combination is not supported.
    """
    if backend != "tpu":
        raise NotImplementedError(backend)
    version = infer_tpu_version(infer_tpu_type(instance_type))
    options = dict(
        xla_tpu_spmd_rng_bit_generator_unsafe=True,  # SPMD partition-aware RngBitGenerator.
        xla_tpu_enable_latency_hiding_scheduler="true",  # Try to schedule ops efficiently.
        xla_tpu_perform_spmd_cse_prevention="false",
        # b/229655601: prevent OOM on gpt2-small-repeat.
    )
    if version == "v4":
        options.update(
            # Per maggioni@google.com, the following flags are not supported by V3.
            # These flags are enabled by default starting on v5.
            xla_enable_async_all_gather="true",  # Allow async all-gather.
            xla_enable_async_collective_permute="true",  # Allow async collective permute.
        )
    if version == "v6e":
        options.update(
            # improved performance for v6e
            xla_tpu_scoped_vmem_limit_kib="98304",
            # maxtext xla flags
            xla_tpu_enable_async_collective_fusion="true",
            xla_tpu_enable_async_collective_fusion_fuse_all_gather="true",
            xla_tpu_enable_async_collective_fusion_multiple_steps="true",
            xla_tpu_overlap_compute_collective_tc="true",
            xla_enable_async_all_gather="true",
            # Flag to enable some advanced scheduling features.
            xla_tpu_enable_all_experimental_scheduler_features="true",
            # Flag to enable memory tracking scheduling. The default AUTO only enables
            # it in some situations. Not needed if
            # xla_tpu_enable_all_experimental_scheduler_features is set to true already.
            xla_tpu_enable_scheduler_memory_pressure_tracking="ENABLED",
            # Flag controlling the maximum number of overlapping host offloadings.
            xla_tpu_host_transfer_overlap_limit=24,
            # Flag to enable the aggressive removal of opt-barriers.
            xla_tpu_aggressive_opt_barrier_removal="ENABLED",
            # Flag to enable more aggressive scheduling for async ops, such as pushing
            # the async start to the beginning of the loop body.
            xla_lhs_prioritize_async_depth_over_stall="ENABLED",
            # For multi-slice configurations,
            # Flag to enable pipelining of cross-DCN all-gathers.
            xla_tpu_enable_ag_backward_pipelining="true",
            xla_should_allow_loop_variant_parameter_in_chain="ENABLED",
            xla_should_add_loop_invariant_op_in_chain="ENABLED",
            # Flag controlling the maximum number of overlapping cross-DCN send/recv.
            xla_max_concurrent_host_send_recv=100,
            # If you are seeing OOM (out-of-memory) error, or bad performance when HBM memory
            # usage is close to HBM capacity, tuning these two flags might help:
            # Flag controlling the HBM memory limit as a percentage of the total HBM size.
            # Default value is 95. Can tune up or down to give more or less memory for the
            # scheduler. The scheduler favors more on less memory usage when it's under
            # memory pressure, instead of hiding latency by overlapping more computations
            # and communications.
            # xla_tpu_scheduler_percent_shared_memory_limit=xx,
            # Flag controlling the number of times the scheduler is run if the scheduled
            # peak memory usage exceeds the initial memory limit, by setting memory limit
            # to 90% of the previous memory limit each time. Default value is 1. Sometimes
            # when the scheduler thinks it goes out memory, it may not actually happen due
            # to other factors controlled by other compiler passes, or the initial memory
            # limit is already set too low. Cutting the memory limit to 90% of previous one
            # though, may make the scheduler weighting too much on the memory usage instead
            # of latency side.
            xla_latency_hiding_scheduler_rerun=0,
        )
    if num_slices > 1:
        # Support multiple TPU slices connected over a data center network.
        options.update(
            # For collectives across multiple slices.
            xla_tpu_enable_megascale_barrier="true",
            # Per rwitten@google.com the following two flags allow gradient all-reduce to happen
            # concurrently with gradient computation for the following layer.
            xla_tpu_enable_data_parallel_all_reduce_opt="true",
            xla_tpu_data_parallel_opt_different_sized_ops="true",
        )

    # Validate options. Will never fail if this function is implemented correctly.
    # for k, v in options.items():
    #     assert v in [True, False, "true", "false"], (k, v)

    return options


def xla_flags_from_options(xla_options: dict[str, Union[str, bool]]) -> str:
    """Convert an XLA options dict suitable for
    `jitted_fn.lower(...).compile(compiler_options=xla_options)`
    to XLA flags suitable for the `XLA_FLAGS` environment variable.
    """
    flags = []
    for k, v in xla_options.items():
        if isinstance(v, bool):
            v = "1" if v else "0"
        flags.append(f"--{k}={v}")
    return " ".join(flags)


class NotTpuError(ValueError):
    pass


def infer_tpu_type(instance_type: str) -> str:
    """Infers tpu type (e.g. v4-8) from instance type (e.g. tpu-v4-8 or v4-8)."""
    if not (instance_type and re.fullmatch(r"(tpu-)?v.+-\d+", instance_type)):
        raise NotTpuError(f"Invalid TPU instance: {instance_type}")
    return instance_type.replace("tpu-", "")


def infer_tpu_version(tpu_type: str) -> str:
    """Infer TPU version from the TPU type.

    Args:
        tpu_type: A string of the format {version}-{cores}.

    Returns:
        Inferred TPU version string.

    Raises:
        ValueError: if the TPU version string is unknown.
    """
    tpu_type = infer_tpu_type(tpu_type)
    tpu_version = tpu_type.rsplit("-", 1)[0]  # split from the last occurrence of '-'
    if tpu_version not in _TPU_VERSIONS:
        raise ValueError(f"Unknown TPU version {tpu_version}. Expected one of {_TPU_VERSIONS}")
    return tpu_version


_TPU_VERSIONS = ("v3", "v4", "v5litepod", "v5p", "v6e")
