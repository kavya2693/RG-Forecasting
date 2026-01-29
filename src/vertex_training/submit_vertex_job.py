"""
Submit Vertex AI Custom Training Job
====================================
"""

from google.cloud import aiplatform
import argparse
from datetime import datetime


def submit_training_job(
    project_id: str,
    region: str,
    bucket: str,
    display_name: str = None
):
    """Submit a custom training job to Vertex AI."""

    if display_name is None:
        display_name = f"c1b1-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    print(f"Submitting Vertex AI training job: {display_name}")

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
                    "package_uris": [f"gs://{bucket}/training_code/trainer-0.1.tar.gz"],
                    "python_module": "trainer.train_c1b1_sharded",
                    "args": [
                        "--bucket", bucket,
                        "--train-prefix", "training_data/f1_train/",
                        "--val-prefix", "training_data/f1_val/",
                        "--sku-attr-path", "training_data/sku_list_attribute.csv",
                        "--output-path", f"models/c1b1_{datetime.now().strftime('%Y%m%d')}",
                        "--threshold", "0.7"
                    ]
                }
            }
        ]
    )

    # Run the job
    job.run(sync=False)

    print(f"Job submitted: {job.resource_name}")
    print(f"View in console: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={project_id}")

    return job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-id', type=str, default='myforecastingsales')
    parser.add_argument('--region', type=str, default='us-central1')
    parser.add_argument('--bucket', type=str, default='myforecastingsales-data')
    parser.add_argument('--display-name', type=str, default=None)

    args = parser.parse_args()

    submit_training_job(
        project_id=args.project_id,
        region=args.region,
        bucket=args.bucket,
        display_name=args.display_name
    )


if __name__ == "__main__":
    main()
