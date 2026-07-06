from reactome2py import analysis
import os
import pandas as pd

class Marker_ori:
    def __init__(self, marker_list, p_value):
        self.markers = ','.join(marker_list)
        self.p_value = p_value
        self.result = self.enrichment_analysis()

    def enrichment_analysis(self):
        """Enrichment analysis performed on all the pathways.
        
        First all the hit pathways are obtained. Then, it is determined
        which of them are significant (p_value < threshold).

        Returns
        -------
        dict
            Dictionary of significant pathways, where stids are keys
            and the values stored are p_value and significance of
            each pathway
        """
        result = analysis.identifiers(ids=self.markers, interactors=False, page_size='1', page='1',
                                      species='Homo Sapiens', sort_by='ENTITIES_FDR', order='ASC',
                                      resource='TOTAL', p_value='1', include_disease=False, min_entities=None,
                                      max_entities=None, projection=True)
        token = result['summary']['token']
        token_result = analysis.token(token, species='Homo sapiens', page_size='-1', page='-1', sort_by='ENTITIES_FDR',
                                      order='ASC', resource='TOTAL', p_value='1', include_disease=False,
                                      min_entities=None, max_entities=None)
        info = [(p['stId'], p['entities']['pValue']) for p in token_result['pathways']]
        pathway_significance = {}
        for stid, p_val in info:
            significance = 'significant' if p_val < self.p_value else 'non-significant'
            pathway_significance[stid] = {'p_value': round(p_val, 4), 'significance': significance}
        return pathway_significance

class Marker:
    def __init__(self, marker_list, p_value,
                 pathways_mapping_file="data/processed/pathways_mapped_all_genes.tsv",
                 save_dir="reactome_embedding/results/enrichment"):
        self.marker_list = marker_list
        self.markers = set(marker_list)
        self.p_value = p_value
        self.save_dir = save_dir

        os.makedirs(save_dir, exist_ok=True)

        # Load pathway → genes mapping
        self.pathway_to_genes = self.load_pathway_mapping(pathways_mapping_file)

        # Run enrichment
        self.result, self.pathway_stats = self.enrichment_analysis()

        # Save CSVs
        self.save_results_csv()

    def load_pathway_mapping(self, mapping_file):
        df = pd.read_csv(mapping_file, sep="\t", low_memory=False)
        mapping = {}
        for _, row in df.iterrows():
            genes = [g for g in row[1:] if pd.notna(g)]
            mapping[row["PathwayID"]] = genes
        return mapping

    def enrichment_analysis(self):
        # Run Reactome analysis
        result = analysis.identifiers(
            ids=",".join(self.marker_list),
            interactors=False,
            page_size='1', page='1',
            species='Homo Sapiens',
            sort_by='ENTITIES_FDR',
            order='ASC',
            resource='TOTAL',
            p_value='1',
            include_disease=False,
            min_entities=None,
            max_entities=None,
            projection=True
        )
        token = result['summary']['token']
        token_result = analysis.token(
            token,
            species='Homo sapiens',
            page_size='-1',
            page='-1',
            sort_by='ENTITIES_FDR',
            order='ASC',
            resource='TOTAL',
            p_value='1',
            include_disease=False,
            min_entities=None,
            max_entities=None,
        )

        pathway_results = {}
        pathway_stats = {}

        for p in token_result['pathways']:
            stid = p['stId']
            name = p.get('name', "")
            entities = p['entities']
            p_val = float(entities.get('pValue', 1.0))
            fdr = entities.get('fdr')
            found = entities.get('found')
            total = entities.get('total')
            hit_ratio = entities.get('ratio')
            significance = 'significant' if p_val < self.p_value else 'non-significant'

            # Try Reactome exp first
            hit_genes = entities.get('exp', [])
            # Fallback: intersect input markers with local mapping
            if not hit_genes:
                hit_genes = list(self.markers.intersection(self.pathway_to_genes.get(stid, [])))
            hit_genes_str = ",".join(hit_genes)

            # Save full info
            pathway_results[stid] = {
                **p,
                "p_value": p_val,  # no rounding
                "significance": significance,
                "hit_genes": hit_genes_str
            }

            # Save stats for separate CSV
            pathway_stats[stid] = {
                "stId": stid,
                "name": name,
                "p_value": p_val,   # no rounding
                "fdr": fdr if fdr is not None else None,
                "found": found,
                "total": total,
                "hit_ratio": hit_ratio,
                "hit_genes": hit_genes_str,
                "significance": significance
            }

        return pathway_results, pathway_stats

    def save_results_csv(self):
        # Simple CSV (stId + p_value + significance)
        simple_csv = os.path.join(self.save_dir, "enrichment_simple.csv")
        simple_df = pd.DataFrame([
            {"stId": stid,
             "p_value": f"{info['p_value']:.6E}",
             "significance": info["significance"]}
            for stid, info in self.result.items()
        ])
        simple_df.to_csv(simple_csv, index=False)

        # Expanded CSV (all stats, with formatted p_value and fdr)
        expanded_csv = os.path.join(self.save_dir, "enrichment_expanded.csv")
        expanded_df = pd.DataFrame(self.pathway_stats.values())

        if "p_value" in expanded_df.columns:
            expanded_df["p_value"] = expanded_df["p_value"].apply(lambda x: f"{x:.6E}")
        if "fdr" in expanded_df.columns:
            expanded_df["fdr"] = expanded_df["fdr"].apply(lambda x: f"{x:.6E}" if pd.notna(x) else None)

        expanded_df.to_csv(expanded_csv, index=False)

        print(f"✅ Saved simple CSV: {simple_csv}")
        print(f"✅ Saved expanded CSV: {expanded_csv}")
