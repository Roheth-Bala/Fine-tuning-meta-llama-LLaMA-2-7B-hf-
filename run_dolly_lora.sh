#!/bin/bash

#SBATCH --job-name=dolly_lora_llama2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=17:00:00
#SBATCH --output=/tmp/%u_%j.out
#SBATCH --error=/tmp/%u_%j.err

echo "Job Started: $(date)"
echo "Running on node: $(hostname)"

# 1. Temporary working directory
JOB_TMP_DIR=$(mktemp -d /tmp/${USER}_job${SLURM_JOB_ID}_XXXX)
echo "Created temporary directory: $JOB_TMP_DIR"

# 2. Copy script and potentially other files (e.g., requirements.txt)
cp train_dolly_lora.py $JOB_TMP_DIR/
# cp requirements.txt $JOB_TMP_DIR/ # Uncomment if you have a requirements file
cd $JOB_TMP_DIR

# 3. Use absolute path to conda Python (AVOIDS ENV ISSUES)
PYTHON_EXEC="/home/e/e1415353/my_projects/miniconda3/envs/ml/bin/python"
echo "Using Python: $($PYTHON_EXEC --version)"
$PYTHON_EXEC -c "import transformers; print('Transformers version:', transformers.__version__)"
$PYTHON_EXEC -c "import peft; print('PEFT version:', peft.__version__)"
$PYTHON_EXEC -c "import trl; print('TRL version:', trl.__version__)"

# 4. Log directory
LOG_DIR="/home/e/e1415353/my_projects/Assignment7Folder/Assignment7/" # Update this path as needed
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/dolly_lora_llama2_${SLURM_JOB_ID}.log"

# 5. Run training
echo "Executing train_dolly_lora.py..."
$PYTHON_EXEC train_dolly_lora.py 2>&1 | tee "$LOG_FILE"

# 6. Copy final results back (optional, adjust paths as needed)
# Example: copy the results folder if it's created in the job directory
if [ -d "./results" ]; then
    RESULT_DIR="$LOG_DIR/results_${SLURM_JOB_ID}"
    mkdir -p "$RESULT_DIR"
    cp -r ./results/* "$RESULT_DIR/"
    echo "Results copied to: $RESULT_DIR"
fi

# 7. Copy SLURM logs
cp "/tmp/${USER}_${SLURM_JOB_ID}.out" "$LOG_DIR/dolly_lora_llama2_${SLURM_JOB_ID}.slurm.out"
cp "/tmp/${USER}_${SLURM_JOB_ID}.err" "$LOG_DIR/dolly_lora_llama2_${SLURM_JOB_ID}.slurm.err"

# 8. Cleanup
rm -rf "$JOB_TMP_DIR"
echo "Job Finished: $(date)"
