import numpy as np
import pandas as pd
import re
from typing import List, Optional, Tuple

class DataLoader:
    def remove_columns_with_m(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        for column in df_copy.columns:
            if df_copy[column].astype(str).str.contains('m').any():
                df_copy.drop(column, axis=1, inplace=True)
        return df_copy

    def load_data(self, excel_file: str, class_zero: int) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        df = pd.read_excel(excel_file)
        original_columns = df.columns.tolist()
        new_columns = [str(i) for i in range(1, len(original_columns) + 1)]
        rename_dict = dict(zip(original_columns, new_columns))
        df_indexed = df.rename(columns=rename_dict)

        df_cleaned_indexed = self.remove_columns_with_m(df_indexed)
        hmdb_ids = df_cleaned_indexed.iloc[-1].copy()
        samples_df = df_cleaned_indexed.iloc[:-1].copy()

        zeros = np.zeros(class_zero, dtype=int)
        ones = np.ones(samples_df.shape[0] - class_zero, dtype=int)
        combined_array = np.concatenate((zeros, ones))
        labels = pd.Series(combined_array)

        return samples_df.astype(float), labels, hmdb_ids

    def get_metabolite_info(self, hmdb_df: pd.DataFrame, hmdb_ids: pd.DataFrame) -> pd.DataFrame:
        normalized_ids = []
        id_list = hmdb_ids.values.flatten()
        for hmdb_id in id_list:
            if pd.isna(hmdb_id) or str(hmdb_id).strip() == "":
                continue
            parts = [part.strip() for part in re.split(r"[,;\s]+", str(hmdb_id)) if part.strip()]
            for part in parts:
                if part.startswith('HMDB'):
                    normalized_ids.append(part)
                else:
                    normalized_ids.append(f"HMDB{part}")

        filtered_df = hmdb_df[hmdb_df['hmdb_id'].isin(normalized_ids)].copy()
        return filtered_df