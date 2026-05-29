import argparse
import os
import warnings
from sklearn.exceptions import ConvergenceWarning
from config import Config
from pipeline.orchestrator import NestedCVOrchestrator

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ['XDG_CACHE_HOME'] = '/storage/v-jinpewang/az_workspace/zhanglin/reproduction/specml/tabpfn_cache'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shared-max-features",
        type=int,
        default=Config.SHARED_MAX_FEATURES,
        help="Override Config.SHARED_MAX_FEATURES for the current run."
    )
    args = parser.parse_args()
    if args.shared_max_features <= 0:
        parser.error("--shared-max-features must be a positive integer.")
    return args

def suppress_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message="^Objective did not converge.*")
    warnings.filterwarnings("ignore", category=UserWarning, message="Using a target size .*different to the input size.*")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in cast")
    warnings.filterwarnings("ignore", message=".*A worker stopped while some jobs were given to the executor.*")

if __name__ == "__main__":
    args = parse_args()
    suppress_warnings()
    Config.SHARED_MAX_FEATURES = args.shared_max_features
    
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    DATA_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "data/Raman_spectroscopy_data_preprocessed.csv"))
    OUTPUT_XLSX_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "nestedcv_oof_predictions.xlsx"))
    TARGET_METALS = ["118Sn (KED)", "209Bi (KED)"]

    
    print(f"[*] Data Source: {DATA_PATH}")
    print(f"[*] Output Target: {OUTPUT_XLSX_PATH}")
    print(f"[*] SHARED_MAX_FEATURES: {Config.SHARED_MAX_FEATURES}")
    print(f"[*] TARGET_SHARED_MAX_FEATURES: {getattr(Config, 'TARGET_SHARED_MAX_FEATURES', {})}")

    runner = NestedCVOrchestrator(
        data_path=DATA_PATH,
        output_path=OUTPUT_XLSX_PATH,
        target_metals=TARGET_METALS
    )
    runner.run()
