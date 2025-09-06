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


def get_leaf_behaviours(data):
    """Trả về DataFrame chứa ID và Label của các leaf behaviour"""
    # Lấy sheet đầu tiên (thường chỉ có 1 sheet chính)
    sheetname = list(data["behaviour"].keys())[0]
    df = data["behaviour"][sheetname].copy()

    # Chuẩn hoá tên cột
    df.columns = [str(c).strip().lower() for c in df.columns]

    id_col = next((c for c in df.columns if "id" in c), None)
    label_col = next((c for c in df.columns if "label" in c or "name" in c), None)
    parent_col = next((c for c in df.columns if "parent" in c), None)

    if not (id_col and label_col and parent_col):
        raise RuntimeError("Không tìm thấy cột ID/Label/Parent trong file behaviour Excel!")

    # Tìm các ID là cha
    parent_ids = set(df[parent_col].dropna())

    # Các leaf = ID không nằm trong parent_ids
    leaf_df = df[~df[id_col].isin(parent_ids)][[id_col, label_col]].reset_index(drop=True)

    return leaf_df

if __name__ == "__main__":
    pd.set_option("display.max_columns", 0)
    pd.set_option("display.width", 160)

    data = load_all()
    df = get_behaviour_hierarchy(data)
    print_parent_children(df)
