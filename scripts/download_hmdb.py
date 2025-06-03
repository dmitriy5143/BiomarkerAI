import os
import requests
import zipfile
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm

def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    with open(dest_path, "wb") as f:
        for chunk in tqdm(response.iter_content(chunk_size=8192),
                          total=total_size // 8192,
                          unit="KB",
                          desc="Downloading"):
            if chunk:
                f.write(chunk)
    print(f"\nФайл успешно скачан в {dest_path}")

def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Архив {zip_path} успешно распакован в {extract_to}")

def parse_hmdb_xml(xml_file):

    metabolites = []
    context = ET.iterparse(xml_file, events=('end',))
    count = 0

    for event, elem in tqdm(context, desc="Парсинг HMDB"):
        if elem.tag.endswith('metabolite'):
            metabolite_data = {}

            accession = elem.find('.//{http://www.hmdb.ca}accession')
            if accession is not None:
                metabolite_data['hmdb_id'] = accession.text

            name = elem.find('.//{http://www.hmdb.ca}name')
            if name is not None:
                metabolite_data['name'] = name.text

            description = elem.find('.//{http://www.hmdb.ca}description')
            if description is not None and description.text:
                metabolite_data['description'] = description.text

            status = elem.find('.//{http://www.hmdb.ca}status')
            if status is not None:
                metabolite_data['status'] = status.text

            formula = elem.find('.//{http://www.hmdb.ca}chemical_formula')
            if formula is not None:
                metabolite_data['formula'] = formula.text

            mono_mass = elem.find('.//{http://www.hmdb.ca}monisotopic_molecular_weight')
            if mono_mass is not None and mono_mass.text:
                try:
                    metabolite_data['monoisotopic_mass'] = float(mono_mass.text)
                except ValueError:
                    metabolite_data['monoisotopic_mass'] = None

            avg_mass = elem.find('.//{http://www.hmdb.ca}average_molecular_weight')
            if avg_mass is not None and avg_mass.text:
                try:
                    metabolite_data['average_molecular_weight'] = float(avg_mass.text)
                except ValueError:
                    metabolite_data['average_molecular_weight'] = None

            smiles = elem.find('.//{http://www.hmdb.ca}smiles')
            if smiles is not None:
                metabolite_data['smiles'] = smiles.text

            state = elem.find('.//{http://www.hmdb.ca}state')
            if state is not None:
                metabolite_data['state'] = state.text

            biospecimen_locs = []
            for loc in elem.findall('.//{http://www.hmdb.ca}biospecimen'):
                if loc.text is not None:
                    biospecimen_locs.append(loc.text)
            if biospecimen_locs:
                metabolite_data['biospecimen_locations'] = biospecimen_locs

            cellular_locs = []
            for loc in elem.findall('.//{http://www.hmdb.ca}cellular'):
                if loc.text is not None:
                    cellular_locs.append(loc.text)
            if cellular_locs:
                metabolite_data['cellular_locations'] = cellular_locs

            tissue_locs = []
            for loc in elem.findall('.//{http://www.hmdb.ca}tissue'):
                if loc.text is not None:
                    tissue_locs.append(loc.text)
            if tissue_locs:
                metabolite_data['tissue_locations'] = tissue_locs

            diseases = []
            for disease in elem.findall('.//{http://www.hmdb.ca}disease'):
                disease_info = {}
                disease_name = disease.find('.//{http://www.hmdb.ca}name')
                if disease_name is not None and disease_name.text is not None:
                    disease_info['name'] = disease_name.text

                omim_id = disease.find('.//{http://www.hmdb.ca}omim_id')
                if omim_id is not None and omim_id.text:
                    disease_info['omim_id'] = omim_id.text

                disease_refs = []
                for ref in disease.findall('.//{http://www.hmdb.ca}reference'):
                    ref_info = {}
                    pubmed_id = ref.find('.//{http://www.hmdb.ca}pubmed_id')
                    if pubmed_id is not None and pubmed_id.text:
                        ref_info['pubmed_id'] = pubmed_id.text

                    ref_text = ref.find('.//{http://www.hmdb.ca}reference_text')
                    if ref_text is not None and ref_text.text:
                        ref_info['text'] = ref_text.text

                    if ref_info:
                        disease_refs.append(ref_info)

                if disease_refs:
                    disease_info['references'] = disease_refs

                if disease_info:
                    diseases.append(disease_info)

            if diseases:
                metabolite_data['diseases'] = diseases

            pathways = []
            for pathway in elem.findall('.//{http://www.hmdb.ca}pathway'):
                pathway_info = {}
                pathway_name = pathway.find('.//{http://www.hmdb.ca}name')
                if pathway_name is not None and pathway_name.text is not None:
                    pathway_info['name'] = pathway_name.text

                kegg_map_id = pathway.find('.//{http://www.hmdb.ca}kegg_map_id')
                if kegg_map_id is not None and kegg_map_id.text:
                    pathway_info['kegg_map_id'] = kegg_map_id.text

                if pathway_info:
                    pathways.append(pathway_info)

            if pathways:
                metabolite_data['pathways'] = pathways

            normal_concentrations = []
            for conc in elem.findall('.//{http://www.hmdb.ca}normal_concentrations/concentration'):
                conc_info = {}
                biospecimen = conc.find('.//{http://www.hmdb.ca}biospecimen')
                if biospecimen is not None and biospecimen.text:
                    conc_info['biospecimen'] = biospecimen.text

                conc_value = conc.find('.//{http://www.hmdb.ca}concentration_value')
                if conc_value is not None and conc_value.text:
                    conc_info['value'] = conc_value.text

                conc_units = conc.find('.//{http://www.hmdb.ca}concentration_units')
                if conc_units is not None and conc_units.text:
                    conc_info['units'] = conc_units.text

                patient_age = conc.find('.//{http://www.hmdb.ca}patient_age') or conc.find('.//{http://www.hmdb.ca}subject_age')
                if patient_age is not None and patient_age.text:
                    conc_info['age'] = patient_age.text

                patient_sex = conc.find('.//{http://www.hmdb.ca}patient_sex') or conc.find('.//{http://www.hmdb.ca}subject_sex')
                if patient_sex is not None and patient_sex.text:
                    conc_info['sex'] = patient_sex.text

                patient_info = conc.find('.//{http://www.hmdb.ca}patient_information')
                if patient_info is not None and patient_info.text:
                    conc_info['patient_information'] = patient_info.text

                subject_condition = conc.find('.//{http://www.hmdb.ca}subject_condition')
                if subject_condition is not None and subject_condition.text:
                    conc_info['condition'] = subject_condition.text

                comment = conc.find('.//{http://www.hmdb.ca}comment')
                if comment is not None and comment.text:
                    conc_info['comment'] = comment.text

                conc_refs = []
                for ref in conc.findall('.//{http://www.hmdb.ca}reference'):
                    ref_info = {}
                    pubmed_id = ref.find('.//{http://www.hmdb.ca}pubmed_id')
                    if pubmed_id is not None and pubmed_id.text:
                        ref_info['pubmed_id'] = pubmed_id.text

                    ref_text = ref.find('.//{http://www.hmdb.ca}reference_text')
                    if ref_text is not None and ref_text.text:
                        ref_info['text'] = ref_text.text

                    if ref_info:
                        conc_refs.append(ref_info)

                if conc_refs:
                    conc_info['references'] = conc_refs

                if conc_info:
                    normal_concentrations.append(conc_info)

            if normal_concentrations:
                metabolite_data['normal_concentrations'] = normal_concentrations

            abnormal_concentrations = []
            for conc in elem.findall('.//{http://www.hmdb.ca}abnormal_concentrations/concentration'):
                conc_info = {}
                biospecimen = conc.find('.//{http://www.hmdb.ca}biospecimen')
                if biospecimen is not None and biospecimen.text:
                    conc_info['biospecimen'] = biospecimen.text

                conc_value = conc.find('.//{http://www.hmdb.ca}concentration_value')
                if conc_value is not None and conc_value.text:
                    conc_info['value'] = conc_value.text

                conc_units = conc.find('.//{http://www.hmdb.ca}concentration_units')
                if conc_units is not None and conc_units.text:
                    conc_info['units'] = conc_units.text

                patient_age = conc.find('.//{http://www.hmdb.ca}patient_age') or conc.find('.//{http://www.hmdb.ca}subject_age')
                if patient_age is not None and patient_age.text:
                    conc_info['age'] = patient_age.text

                patient_sex = conc.find('.//{http://www.hmdb.ca}patient_sex') or conc.find('.//{http://www.hmdb.ca}subject_sex')
                if patient_sex is not None and patient_sex.text:
                    conc_info['sex'] = patient_sex.text

                patient_info = conc.find('.//{http://www.hmdb.ca}patient_information')
                if patient_info is not None and patient_info.text:
                    conc_info['patient_information'] = patient_info.text

                subject_condition = conc.find('.//{http://www.hmdb.ca}subject_condition')
                if subject_condition is not None and subject_condition.text:
                    conc_info['condition'] = subject_condition.text

                comment = conc.find('.//{http://www.hmdb.ca}comment')
                if comment is not None and comment.text:
                    conc_info['comment'] = comment.text

                conc_refs = []
                for ref in conc.findall('.//{http://www.hmdb.ca}reference'):
                    ref_info = {}
                    pubmed_id = ref.find('.//{http://www.hmdb.ca}pubmed_id')
                    if pubmed_id is not None and pubmed_id.text:
                        ref_info['pubmed_id'] = pubmed_id.text

                    ref_text = ref.find('.//{http://www.hmdb.ca}reference_text')
                    if ref_text is not None and ref_text.text:
                        ref_info['text'] = ref_text.text

                    if ref_info:
                        conc_refs.append(ref_info)

                if conc_refs:
                    conc_info['references'] = conc_refs

                if conc_info:
                    abnormal_concentrations.append(conc_info)

            if abnormal_concentrations:
                metabolite_data['abnormal_concentrations'] = abnormal_concentrations

            exp_properties = []
            for prop in elem.findall('.//{http://www.hmdb.ca}experimental_properties/property'):
                prop_info = {}
                kind = prop.find('.//{http://www.hmdb.ca}kind')
                if kind is not None and kind.text:
                    prop_info['kind'] = kind.text

                value = prop.find('.//{http://www.hmdb.ca}value')
                if value is not None and value.text:
                    prop_info['value'] = value.text

                source = prop.find('.//{http://www.hmdb.ca}source')
                if source is not None and source.text:
                    prop_info['source'] = source.text

                if prop_info:
                    exp_properties.append(prop_info)

            if exp_properties:
                metabolite_data['experimental_properties'] = exp_properties

            pred_properties = []
            for prop in elem.findall('.//{http://www.hmdb.ca}predicted_properties/property'):
                prop_info = {}
                kind = prop.find('.//{http://www.hmdb.ca}kind')
                if kind is not None and kind.text:
                    prop_info['kind'] = kind.text

                value = prop.find('.//{http://www.hmdb.ca}value')
                if value is not None and value.text:
                    prop_info['value'] = value.text

                source = prop.find('.//{http://www.hmdb.ca}source')
                if source is not None and source.text:
                    prop_info['source'] = source.text

                if prop_info:
                    pred_properties.append(prop_info)

            if pred_properties:
                metabolite_data['predicted_properties'] = pred_properties

            synonyms = []
            for syn in elem.findall('.//{http://www.hmdb.ca}synonym'):
                if syn.text is not None:
                    synonyms.append(syn.text)
            if synonyms:
                metabolite_data['synonyms'] = synonyms

            taxonomy = {}

            kingdom = elem.find('.//{http://www.hmdb.ca}taxonomy/kingdom')
            if kingdom is not None and kingdom.text:
                taxonomy['kingdom'] = kingdom.text

            super_class = elem.find('.//{http://www.hmdb.ca}taxonomy/super_class')
            if super_class is not None and super_class.text:
                taxonomy['super_class'] = super_class.text

            class_elem = elem.find('.//{http://www.hmdb.ca}taxonomy/class')
            if class_elem is not None and class_elem.text:
                taxonomy['class'] = class_elem.text

            sub_class = elem.find('.//{http://www.hmdb.ca}taxonomy/sub_class')
            if sub_class is not None and sub_class.text:
                taxonomy['sub_class'] = sub_class.text

            direct_parent = elem.find('.//{http://www.hmdb.ca}taxonomy/direct_parent')
            if direct_parent is not None and direct_parent.text:
                taxonomy['direct_parent'] = direct_parent.text

            alt_parents = []
            for parent in elem.findall('.//{http://www.hmdb.ca}taxonomy/alternative_parent'):
                if parent.text is not None:
                    alt_parents.append(parent.text)
            if alt_parents:
                taxonomy['alternative_parents'] = alt_parents

            substituents = []
            for subst in elem.findall('.//{http://www.hmdb.ca}taxonomy/substituent'):
                if subst.text is not None:
                    substituents.append(subst.text)
            if substituents:
                taxonomy['substituents'] = substituents

            if taxonomy:
                metabolite_data['taxonomy'] = taxonomy

            kegg_id = elem.find('.//{http://www.hmdb.ca}kegg_id')
            if kegg_id is not None and kegg_id.text is not None:
                metabolite_data['kegg_id'] = kegg_id.text

            iupac_name = elem.find('.//{http://www.hmdb.ca}iupac_name')
            if iupac_name is not None and iupac_name.text is not None:
                metabolite_data['iupac_name'] = iupac_name.text

            proteins = []
            for protein in elem.findall('.//{http://www.hmdb.ca}protein'):
                protein_info = {}

                name = protein.find('.//{http://www.hmdb.ca}name')
                if name is not None and name.text:
                    protein_info['name'] = name.text

                gene_name = protein.find('.//{http://www.hmdb.ca}gene_name')
                if gene_name is not None and gene_name.text:
                    protein_info['gene_name'] = gene_name.text

                protein_type = protein.find('.//{http://www.hmdb.ca}protein_type')
                if protein_type is not None and protein_type.text:
                    protein_info['protein_type'] = protein_type.text

                if protein_info:
                    proteins.append(protein_info)

            if proteins:
                metabolite_data['proteins'] = proteins

            synthesis_reference = elem.find('.//{http://www.hmdb.ca}synthesis_reference')
            if synthesis_reference is not None and synthesis_reference.text:
                metabolite_data['synthesis_reference'] = synthesis_reference.text

            if 'hmdb_id' in metabolite_data:
                metabolites.append(metabolite_data)

            elem.clear()

            count += 1
            if count % 10000 == 0:
                print(f"Обработано {count} метаболитов")

    df = pd.DataFrame(metabolites)
    print(f"Всего обработано {len(df)} метаболитов")
    return df

def main():
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Директория {data_dir} создана")
    
    zip_url = "http://www.hmdb.ca/system/downloads/current/hmdb_metabolites.zip"
    zip_path = os.path.join(data_dir, "hmdb_metabolites.zip")
    
    print("Скачивание архива HMDB...")
    download_file(zip_url, zip_path)
    
    print("Распаковка архива...")
    unzip_file(zip_path, data_dir)
    
    xml_file = os.path.join(data_dir, "hmdb_metabolites.xml")
    if not os.path.exists(xml_file):
        print(f"Ошибка: Файл {xml_file} не найден после распаковки.")
        return
    
    print("Парсинг XML-файла...")
    df = parse_hmdb_xml(xml_file)
    pickle_path = os.path.join(data_dir, "hmdb_database.pkl")
    df.to_pickle(pickle_path)
    print(f"Результат парсинга сохранен в {pickle_path}")

if __name__ == "__main__":
    main()