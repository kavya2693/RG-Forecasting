"""
Submit Optimized A-Items Training Job to Vertex AI
===================================================
Target: 62% A-items daily accuracy

Optimization parameters:
- calibration_factor = 1.25
- num_leaves = 511 for A-items
- threshold = 0.45 for A-items
- learning_rate = 0.012
- n_estimators = 1200
"""

from google.cloud import aiplatform
from google.cloud import storage
import argparse
from datetime import datetime
import subprocess
import os


def package_and_upload(bucket_name: str):
    """Package the trainer and upload to GCS."""
    print("Packaging trainer module...")

    # Get the vertex_training directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Build the package
    result = subprocess.run(
        ['python', 'setup.py', 'sdist'],
        cwd=script_dir,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Error building package: {result.stderr}")
        raise RuntimeError("Failed to build package")

    # Find the tarball
    dist_dir = os.path.join(script_dir, 'dist')
    tarballs = [f for f in os.listdir(dist_dir) if f.endswith('.tar.gz')]
    if not tarballs:
        raise RuntimeError("No tarball found in dist/")

    tarball = sorted(tarballs)[-1]  # Get the latest
    tarball_path = os.path.join(dist_dir, tarball)
    print(f"  Built: {tarball}")

    # Upload to GCS
    print(f"  Uploading to gs://{bucket_name}/training_code/trainer-optimized.tar.gz...")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob('training_code/trainer-optimized.tar.gz')
    blob.upload_from_filename(tarball_path)
    print("  Upload complete!")

    return f"gs://{bucket_name}/training_code/trainer-optimized.tar.gz"


def submit_training_job(
    project_id: str,
    region: str,
    bucket: str,
    display_name: str = None,
    machine_type: str = "n1-highmem-32"
):
    """Submit the optimized A-items training job to Vertex AI."""

    if display_name is None:
        display_name = f"optimized-a-items-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    print(f"\n{'='*60}")
    print(f"SUBMITTING OPTIMIZED A-ITEMS TRAINING JOB")
    print(f"{'='*60}")
    print(f"Display name: {display_name}")
    print(f"Machine type: {machine_type}")
    print(f"Target: 62% A-items daily accuracy")
    print(f"\nOptimization parameters:")
    print(f"  - num_leaves = 511 for A-items")
    print(f"  - learning_rate = 0.012")
    print(f"  - n_estimators = 1200")
    print(f"  - calibration_factor = 1.25")
    print(f"  - threshold = 0.45")

    # Package and upload
    package_uri = package_and_upload(bucket)

    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)

    # Define output path with timestamp
    output_path = f"optimized_a_items_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Define the custom training job
    job = aiplatform.CustomJob(
        display_name=display_name,
        worker_pool_specs=[
            {
                "machine_spec": {
                    "machine_type": machine_type,  # 32 vCPUs, 208 GB RAM
                },
                "replica_count": 1,
                "python_package_spec": {
                    "executor_image_uri": "us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-12.py310:latest",
                    "package_uris": [package_uri],
                    "python_module": "trainer.train_optimized_a_items",
                    "args": [
                        "--bucket", bucket,
                        "--train-prefix", "baseline2/f1_train/",
                        "--val-prefix", "baseline2/f1_val/",
                        "--output-path", output_path
                    ]
                }
            }
        ]
    )

    # Run the job
    print(f"\nSubmitting job...")
    job.run(sync=False)

    print(f"\n{'='*60}")
    print(f"JOB SUBMITTED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"Job name: {job.resource_name}")
    print(f"Output path: gs://{bucket}/{output_path}/")
    print(f"\nView in console:")
    print(f"  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={project_id}")
    print(f"\nMonitor with:")
    print(f"  gcloud ai custom-jobs describe {job.resource_name.split('/')[-1]} --region={region}")
    print(f"\nExpected outputs:")
    print(f"  - gs://{bucket}/{output_path}/optimized_a_items_metrics.json")
    print(f"  - gs://{bucket}/{output_path}/forecast_168day_optimized.csv")

    return job


def main():
    parser = argparse.ArgumentParser(description="Submit optimized A-items training to Vertex AI")
    parser.add_argument('--project-id', type=str, default='myforecastingsales',
                        help='GCP project ID')
    parser.add_argument('--region', type=str, default='us-central1',
                        help='GCP region')
    parser.add_argument('--bucket', type=str, default='myforecastingsales-data',
                        help='GCS bucket name')
    parser.add_argument('--display-name', type=str, default=None,
                        help='Custom job display name')
    parser.add_argument('--machine-type', type=str, default='n1-highmem-32',
                        help='Machine type (default: n1-highmem-32 for 208GB RAM)')

    args = parser.parse_args()

    submit_training_job(
        project_id=args.project_id,
        region=args.region,
        bucket=args.bucket,
        display_name=args.display_name,
        machine_type=args.machine_type
    )


if __name__ == "__main__":
    main()
