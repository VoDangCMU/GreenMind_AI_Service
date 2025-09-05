import pandas as pd
from .loader import load_all

def get_behaviour_hierarchy(data):
    """Trả về dataframe chứa ParentID, ParentLabel, ChildID, ChildLabel"""
    sheetname = list(data["behaviour"].keys())[0]
    df = data["behaviour"][sheetname].copy()

    df.columns = [str(c).strip().lower() for c in df.columns]

    id_col = next((c for c in df.columns if "id" in c), None)
    label_col = next((c for c in df.columns if "label" in c or "name" in c), None)
    parent_col = next((c for c in df.columns if "parent" in c), None)

    if not (id_col and label_col and parent_col):
        raise RuntimeError("Không tìm thấy cột ID/Label/Parent trong file behaviour Excel!")

    id2label = dict(zip(df[id_col], df[label_col]))

    rows = []
    for _, row in df.iterrows():
        child_id, parent_id = row[id_col], row[parent_col]
        child_label = id2label.get(child_id, str(child_id))
        parent_label = id2label.get(parent_id, str(parent_id)) if pd.notna(parent_id) else None
        rows.append({
            "ParentID": parent_id,
            "ParentLabel": parent_label,
            "ChildID": child_id,
            "ChildLabel": child_label
        })

    return pd.DataFrame(rows)

def print_parent_children(df):
    grouped = df.groupby(["ParentID", "ParentLabel"])
    for (pid, plabel), group in grouped:
        if pd.isna(pid):  
            continue
        print(f"\n{plabel} ({pid})")
        for _, row in group.iterrows():
            print(f"   └── {row['ChildLabel']} ({row['ChildID']})")

if __name__ == "__main__":
    pd.set_option("display.max_columns", 0)
    pd.set_option("display.width", 160)

    data = load_all()
    df = get_behaviour_hierarchy(data)
    print_parent_children(df)
