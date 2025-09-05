from pathlib import Path
import pandas as pd

BASE_DIR = (Path(__file__).resolve().parent / "bcio" / "Behaviour").resolve()

FILES = {
    "behaviour": BASE_DIR / "bcio_behaviour.xlsx",
    "relations": BASE_DIR / "bcio_behaviour_rels.xlsx",   
}

def _read_excel(path: Path, sheet=None):
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    return pd.read_excel(path, sheet_name=sheet, engine="openpyxl")

def load_all():
    """Trả về dict:
       {
         'behaviour': {sheet_name: df, ...},
         'relations': {sheet_name: df, ...} hoặc None
       }
    """
    data = {"behaviour": _read_excel(FILES["behaviour"], sheet=None)}
    data["relations"] = _read_excel(FILES["relations"], sheet=None) if FILES["relations"].exists() else None
    return data
