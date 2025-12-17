import subprocess
import sys
from pathlib import Path
import time

BASE_DIR = Path(__file__).parent
TRAIN_SCRIPT = BASE_DIR / "scripts" / "train_nbe.py"
REPORT_SCRIPT = BASE_DIR / "scripts" / "generate_report.py"
EXPERIMENTS_ROOT = BASE_DIR / "experiments"

# Paramètres d'entraînement
COMMON_TRAIN_ARGS = [
    "--train_size", "2000",
    "--val_size", "500",
    "--epochs", "20",
    "--batch_size", "16",
    "--hidden_dim", "64",
    "--device", "cuda",
    "--patience", "5"
]

def run_command(cmd, task_name):
    """Exécute une commande shell et gère les erreurs."""
    print(f"{task_name}...")
    try:
        subprocess.run(cmd, check=True)
        print(f"{task_name} terminé.")
    except subprocess.CalledProcessError as e:
        print(f"\nError during {task_name}")
        print(f"Error {e.returncode}")
       

def run_study_pipeline(study_name, specific_args):
    """
    Orchestre le pipeline complet pour une étude.
    """

    study_dir = EXPERIMENTS_ROOT / study_name
    ckpt_dir = study_dir / "checkpoints"
    log_dir = study_dir / "logs"
    report_dir = study_dir / "report"

    if study_dir.exists():
        print("Directory already exist")
    
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--checkpoint_dir", str(ckpt_dir),
        "--log_dir", str(log_dir),
        "--output_dir", str(report_dir),
    ] + COMMON_TRAIN_ARGS + specific_args

    run_command(train_cmd, f"Train {study_name}")


    model_path = ckpt_dir / "final_model.pt"

    if not model_path.exists():
        print(f"No existing model")
        return

    report_cmd = [
        sys.executable, str(REPORT_SCRIPT),
        "--checkpoint", str(model_path),
        "--output", str(report_dir),
        "--N", "500",
        "--K_val", "100"
    ] + specific_args 

    run_command(report_cmd, f"Report {study_name}")

def main():
    if not TRAIN_SCRIPT.exists():
        print(f"Erreur : Script introuvable {TRAIN_SCRIPT}")
        return

    start_global = time.time()

    # Comparing encoders
    run_study_pipeline("1_Encoder_MLP", [
        "--encoder", "mlp",
        "--phi_layers", "128,64"
    ])

    run_study_pipeline("1_Encoder_CNN", [
        "--encoder", "cnn",
        "--channels", "16,32,64"
    ])

    run_study_pipeline("1_Encoder_RNN", [
        "--encoder", "rnn",
        "--rnn_type", "LSTM",
        "--rnn_layers", "1",
        "--bidirectional"
    ])

    # Comparing aggregation
    run_study_pipeline("3_Agg_Mean", [
        "--encoder", "mlp",
        "--aggregation", "mean"
    ])

    run_study_pipeline("3_Agg_MeanStd", [
        "--encoder", "mlp",
        "--aggregation", "meanstd"
    ])

if __name__ == "__main__":
    main()