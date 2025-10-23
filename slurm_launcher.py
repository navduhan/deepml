#!/usr/bin/env python3
"""
SLURM Job Launcher for Protein Localization Prediction

This script launches SLURM jobs for training models with different features.
Useful for running multiple experiments in parallel on HPC clusters.

Usage:
    python slurm_launcher.py

Author: Naveen Duhan
Date: 2025-01-17
"""

import os
import subprocess
from pathlib import Path
import config


def submit_slurm_job(job_name, output_name, command, dependency=''):
    """
    Submit a job to SLURM scheduler.
    
    Args:
        job_name (str): Name of the job
        output_name (str): Base name for output files
        command (str): Command to execute
        dependency (str): Job ID dependency (optional)
    
    Returns:
        str: Job ID assigned by SLURM
    """
    if dependency:
        dependency = f'--dependency=afterok:{dependency} --kill-on-invalid-dep=yes'
    
    sbatch_command = (
        f"sbatch "
        f"-J {job_name} "
        f"-o {output_name}.out "
        f"-e {output_name}.err "
        f"--gres={config.SLURM_GPU} "
        f"--nodelist {config.SLURM_NODELIST} "
        f"-t {config.SLURM_TIME} "
        f"--partition {config.SLURM_PARTITION} "
        f"--wrap='{command}' "
        f"{dependency}"
    )
    
    print(f"Submitting job: {job_name}")
    print(f"Command: {command}")
    print(sbatch_command)
    
    sbatch_response = subprocess.getoutput(sbatch_command)
    print(sbatch_response)
    
    # Extract job ID from response
    job_id = sbatch_response.split(' ')[-1].strip()
    return job_id


def main():
    """Launch SLURM jobs for all configured features."""
    
    print("=" * 80)
    print("SLURM JOB LAUNCHER - Protein Classification")
    print("=" * 80)
    
    # Ensure SLURM output directory exists
    slurm_dir = config.SLURM_DIR
    slurm_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSLURM Configuration:")
    print(f"  Partition: {config.SLURM_PARTITION}")
    print(f"  Node: {config.SLURM_NODELIST}")
    print(f"  GPU: {config.SLURM_GPU}")
    print(f"  Time Limit: {config.SLURM_TIME}")
    print(f"  Output Directory: {slurm_dir}")
    
    # Get project root directory
    project_root = config.BASE_DIR
    
    print(f"\nSubmitting jobs for {len(config.FEATURE_CONFIGS)} features...")
    print("-" * 80)
    
    job_ids = {}
    
    for feature_name, vector_size in config.FEATURE_CONFIGS.items():
        # Create job name
        job_name = f"protloc_{feature_name}"
        
        # Create output file base name
        output_base = slurm_dir / feature_name
        
        # Create command
        command = (
            f"cd {project_root} && "
            f"python train.py {feature_name} {vector_size}"
        )
        
        # Submit job
        job_id = submit_slurm_job(job_name, output_base, command)
        job_ids[feature_name] = job_id
        
        print(f"  ✓ Job submitted for {feature_name} (Job ID: {job_id})")
        print()
    
    print("-" * 80)
    print(f"\n✅ Successfully submitted {len(job_ids)} jobs!")
    
    # Save job IDs to file
    job_ids_file = slurm_dir / "job_ids.txt"
    with open(job_ids_file, 'w') as f:
        f.write("Feature\tJob_ID\n")
        for feature, job_id in job_ids.items():
            f.write(f"{feature}\t{job_id}\n")
    
    print(f"\nJob IDs saved to: {job_ids_file}")
    
    # Print monitoring commands
    print("\nMonitoring Commands:")
    print(f"  Check job status: squeue -u $USER")
    print(f"  Check specific job: squeue -j <job_id>")
    print(f"  Cancel job: scancel <job_id>")
    print(f"  Cancel all jobs: scancel -u $USER")
    print(f"  View output: tail -f {slurm_dir}/<feature>.out")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
