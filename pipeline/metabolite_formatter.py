import pandas as pd
import re
from typing import List

class MetaboliteFormatter:
    def format_metabolite_info_for_llm(self, metabolite_df: pd.DataFrame) -> str:
        header = f"Found {len(metabolite_df)} records for the specified HMDBID."
        separator = "=" * 50
        all_metabolites_info = []

        for _, metabolite in metabolite_df.iterrows():
            metabolite_info = []
            metabolite_info.append(f"## Metabolite: {metabolite.get('name', 'Unknown')} ({metabolite.get('hmdb_id', 'Unknown')})")
            if 'synonyms' in metabolite and isinstance(metabolite['synonyms'], list) and len(metabolite['synonyms']) > 0:
                synonyms_list = metabolite['synonyms'][:2]
                metabolite_info.append(f"**Synonyms**: {', '.join(synonyms_list)}")
            if 'description' in metabolite and metabolite['description']:
                metabolite_info.append(f"### Description\n{metabolite['description']}")
            all_metabolites_info.append("\n".join(metabolite_info))

        formatted_text = f"{header}\n{separator}\n" + f"\n{separator}\n".join(all_metabolites_info)
        return formatted_text

    def split_metabolite_blocks(self, formatted_info: str) -> List[str]:
        blocks = re.split(r"={5,}", formatted_info)
        metabolite_blocks = [b.strip() for b in blocks if "## Metabolite:" in b]
        return metabolite_blocks