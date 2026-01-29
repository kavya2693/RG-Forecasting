"""
Submit Spike Feature Test to Vertex AI
======================================
Compares production baseline vs production + spike features.
"""

from google.cloud import aiplatform
import argparse
from datetime import datetime


def submit_spike_test_job(
    project_id: str,
    region: str,
    tier: str,
    fold: str,
    bucket: str = None,
    display_name: str = None
):
    """Submit spike feature comparison job to Vertex AI."""

    if display_name is None:
        display_name = f"spike-test-{tier}-{fold}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    print(f"Submitting Vertex AI spike test job: {display_name}")
    print(f"  Tier: {tier}")
    print(f"  Fold: {fold}")

    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)

    # Define the custom training job
    job = aiplatform.CustomJob(
        display_name=display_name,
        worker_pool_specs=[
            {
                "machine_spec": {
                    "machine_type": "n1-highmem-8",  # 8 vCPUs, 52 GB RAM
                },
                "replica_count": 1,
                "python_package_spec": {
                    "executor_image_uri": "us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-12.py310:latest",
                    "package_uris": [f"gs://{bucket}/training_code/trainer-spike-0.1.tar.gz"],
                    "python_module": "trainer.train_lgbm_vertex_with_spikes",
                    "args": [
                        "--tier", tier,
                        "--fold", fold,
                        "--project", project_id,
                        "--output_table", f"{project_id}.forecasting.val_pred_spike_test_{tier.lower()}"
                    ]
                }
            }
        ]
    )

    # Run the job (async)
    job.run(sync=False)

    print(f"\nJob submitted: {job.resource_name}")
    print(f"View in console: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={project_id}")

    return job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-id', type=str, default='myforecastingsales')
    parser.add_argument('--region', type=str, default='us-central1')
    parser.add_argument('--bucket', type=str, default='myforecastingsales-data')
    parser.add_argument('--tier', type=str, default='T1_MATURE',
                        choices=['T1_MATURE', 'T2_GROWING', 'T3_COLD_START'])
    parser.add_argument('--fold', type=str, default='C1')
    parser.add_argument('--display-name', type=str, default=None)

    args = parser.parse_args()

    submit_spike_test_job(
        project_id=args.project_id,
        region=args.region,
        tier=args.tier,
        fold=args.fold,
        bucket=args.bucket,
        display_name=args.display_name
    )


if __name__ == "__main__":
    main()
