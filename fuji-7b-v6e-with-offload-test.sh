#!/bin/bash
set -e

CLUSTER=<cluster name>
PROJECT_ID=<project id>
ZONE=<zone>
INSTANCE_TYPE=tpu-v6e-16
NUM_REPLICAS=1

export BASTION_TIER=disabled

# Clean up any prior JobSet with the same name
kubectl delete jobset "$USER-fuji-7b-with-offload" --ignore-not-found=true
kubectl wait --for=delete pod -l jobset.sigs.k8s.io/jobset-name="$USER-fuji-7b-with-offload" --timeout=60s 2>/dev/null || true

# Bundle local workspace to Artifact Registry with tag matching the job name
# (Required on Cloudtop VMs because running_from_vm()=True skips bundling inside `launch run`)
axlearn gcp bundle --name="$USER-fuji-7b-with-offload" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --bundler_spec=allow_dirty=True \
        --bundler_type=artifactregistry \
        --bundler_spec=dockerfile=Dockerfile \
        --bundler_spec=image=tpu \
        --bundler_spec=target=tpu

TRAINER_DIR="gs://${PROJECT_ID}-axlearn/${USER}-v6e-7b-with-offload-$(date +"%Y%m%d%H%M%S")"
echo "=== Launching WITH-OFFLOAD run (trainer_dir: ${TRAINER_DIR}) ==="

# Launch training run WITH optimizer state offloading (offloads 52.88 GB of AdamW state to pinned_host)
axlearn gcp launch run --cluster="$CLUSTER" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --runner_name=gke_tpu_single \
        --name="$USER-fuji-7b-with-offload" \
        --instance_type="$INSTANCE_TYPE" \
        --num_replicas="$NUM_REPLICAS" \
        --bundler_spec=allow_dirty=True \
        --bundler_type=artifactregistry --bundler_spec=image=tpu \
        --bundler_spec=dockerfile=Dockerfile --bundler_spec=target=tpu \
        -- python3 -m axlearn.common.launch_trainer_main \
        --module=text.gpt.c4_trainer \
        --config=fuji-7B-v2-flash-offload-opt-single-host \
        --trainer_dir="$TRAINER_DIR" \
        --data_dir=gs://axlearn-public/tensorflow_datasets \
        --jax_backend=tpu \
        --mesh_selector=tpu-v6e-16 \
        --trace_at_steps=3 \
        --trainer_log_every_n_steps=1 \
        --max_step=10 \
        --alsologtostderr \
        --log_dir=/output

# Clean up completed JobSet so the cluster slice is free
kubectl delete jobset "$USER-fuji-7b-with-offload" --ignore-not-found=true
