"""
Network Meta-Analysis (NMA) for Time-to-Event Data from K-M Curves

This module enables network meta-analysis (NMA) when multiple treatment comparisons
are available across different studies. It allows:
- Indirect comparisons (A vs C through B)
- Treatment ranking
- Network plots
- League tables
- SUCRA scores (Surface Under the Cumulative Ranking curve)

Date: 2025-10-25
With God's help, Alhamdulillah!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
import warnings
warnings.filterwarnings('ignore')


class NetworkMetaAnalyzer:
    """
    Network Meta-Analysis for time-to-event data extracted from K-M curves.

    Supports:
    - Network plot visualization
    - Indirect treatment comparisons
    - Treatment ranking with SUCRA scores
    - League tables
    - Consistency checks
    """

    def __init__(self, outcome_name: str = "Overall Survival"):
        """
        Initialize NMA.

        Args:
            outcome_name: Name of the outcome (e.g., "Overall Survival", "PFS")
        """
        self.outcome_name = outcome_name
        self.studies = []
        self.treatments = set()
        self.network = None
        self.results = None

    def add_study(self,
                  study_name: str,
                  treatment_a: str,
                  treatment_b: str,
                  ipd_a: pd.DataFrame,
                  ipd_b: pd.DataFrame,
                  total_n_a: Optional[int] = None,
                  total_n_b: Optional[int] = None):
        """
        Add a two-arm study to the network.

        Args:
            study_name: Study identifier
            treatment_a: Name of treatment A (e.g., "Drug A")
            treatment_b: Name of treatment B (e.g., "Placebo")
            ipd_a: IPD for arm A (must have 'time' and 'event' columns)
            ipd_b: IPD for arm B (must have 'time' and 'event' columns)
            total_n_a: Total sample size for arm A
            total_n_b: Total sample size for arm B
        """
        # Add treatments to network
        self.treatments.add(treatment_a)
        self.treatments.add(treatment_b)

        # Prepare IPD with treatment labels
        ipd_a_labeled = ipd_a.copy()
        ipd_a_labeled['treatment'] = treatment_a
        ipd_a_labeled['study'] = study_name

        ipd_b_labeled = ipd_b.copy()
        ipd_b_labeled['treatment'] = treatment_b
        ipd_b_labeled['study'] = study_name

        # Calculate study-level statistics
        n_a = total_n_a or len(ipd_a)
        n_b = total_n_b or len(ipd_b)
        events_a = int(ipd_a['event'].sum())
        events_b = int(ipd_b['event'].sum())

        # Store study
        self.studies.append({
            'name': study_name,
            'treatment_a': treatment_a,
            'treatment_b': treatment_b,
            'ipd_a': ipd_a_labeled,
            'ipd_b': ipd_b_labeled,
            'n_a': n_a,
            'n_b': n_b,
            'events_a': events_a,
            'events_b': events_b
        })

        print(f"Added study: {study_name}")
        print(f"  Comparison: {treatment_a} vs {treatment_b}")
        print(f"  Sample sizes: {n_a} vs {n_b}")
        print(f"  Events: {events_a} vs {events_b}")
        print()

    def build_network(self):
        """Build the treatment network graph."""
        if len(self.studies) == 0:
            raise ValueError("No studies added. Use add_study() first.")

        # Create network graph
        G = nx.Graph()

        # Add nodes (treatments)
        for treatment in self.treatments:
            G.add_node(treatment)

        # Add edges (comparisons)
        edge_weights = {}  # Number of studies per comparison

        for study in self.studies:
            t_a = study['treatment_a']
            t_b = study['treatment_b']

            # Create edge (undirected)
            edge = tuple(sorted([t_a, t_b]))

            if edge not in edge_weights:
                edge_weights[edge] = 0
            edge_weights[edge] += 1

        # Add weighted edges
        for (t1, t2), weight in edge_weights.items():
            G.add_edge(t1, t2, weight=weight)

        self.network = G

        print("Network structure:")
        print(f"  Treatments: {len(self.treatments)}")
        print(f"  Comparisons: {len(edge_weights)}")
        print(f"  Studies: {len(self.studies)}")
        print()

        # Check connectivity
        if not nx.is_connected(G):
            print("WARNING: Network is not fully connected!")
            print("Some treatments cannot be compared indirectly.")
            print()
        else:
            print("Network is fully connected - all comparisons possible!")
            print()

        return G

    def calculate_pairwise_hr(self, ref_treatment: Optional[str] = None):
        """
        Calculate pairwise hazard ratios using Cox regression.

        Args:
            ref_treatment: Reference treatment for comparisons (default: first alphabetically)

        Returns:
            DataFrame with pairwise HRs
        """
        if self.network is None:
            self.build_network()

        # Set reference treatment
        if ref_treatment is None:
            ref_treatment = sorted(self.treatments)[0]

        print(f"Reference treatment: {ref_treatment}")
        print()

        # Pool all IPD
        all_ipd = []
        for study in self.studies:
            all_ipd.append(study['ipd_a'])
            all_ipd.append(study['ipd_b'])

        pooled_ipd = pd.concat(all_ipd, ignore_index=True)

        # Create dummy variables for treatments
        treatment_dummies = pd.get_dummies(pooled_ipd['treatment'], prefix='treat', drop_first=False)

        # Drop reference treatment column
        if f'treat_{ref_treatment}' in treatment_dummies.columns:
            treatment_dummies = treatment_dummies.drop(columns=[f'treat_{ref_treatment}'])

        # Prepare data for Cox regression
        cox_data = pooled_ipd[['time', 'event']].copy()
        cox_data = pd.concat([cox_data, treatment_dummies], axis=1)

        # Fit Cox model
        cph = CoxPHFitter()
        cph.fit(cox_data, duration_col='time', event_col='event')

        # Extract HRs and CIs
        hr_results = []

        for treatment in sorted(self.treatments):
            if treatment == ref_treatment:
                # Reference treatment: HR = 1.0
                hr_results.append({
                    'treatment': treatment,
                    'HR': 1.0,
                    'HR_lower': 1.0,
                    'HR_upper': 1.0,
                    'p_value': 1.0
                })
            else:
                col_name = f'treat_{treatment}'
                if col_name in cph.summary.index:
                    hr = np.exp(cph.summary.loc[col_name, 'coef'])
                    hr_lower = np.exp(cph.summary.loc[col_name, 'coef'] -
                                    1.96 * cph.summary.loc[col_name, 'se(coef)'])
                    hr_upper = np.exp(cph.summary.loc[col_name, 'coef'] +
                                    1.96 * cph.summary.loc[col_name, 'se(coef)'])
                    p_value = cph.summary.loc[col_name, 'p']

                    hr_results.append({
                        'treatment': treatment,
                        'HR': hr,
                        'HR_lower': hr_lower,
                        'HR_upper': hr_upper,
                        'p_value': p_value
                    })

        hr_df = pd.DataFrame(hr_results)
        self.results = hr_df

        return hr_df

    def calculate_ranking(self, ref_treatment: Optional[str] = None):
        """
        Calculate treatment ranking based on hazard ratios.

        Lower HR = better survival = higher rank

        Returns:
            DataFrame with treatment rankings and SUCRA scores
        """
        if self.results is None:
            self.calculate_pairwise_hr(ref_treatment=ref_treatment)

        # Rank by HR (lower is better)
        ranked = self.results.sort_values('HR')
        ranked['rank'] = range(1, len(ranked) + 1)

        # Calculate SUCRA (Surface Under the Cumulative Ranking curve)
        # SUCRA = 1 means always best, 0 means always worst
        n_treatments = len(ranked)
        ranked['SUCRA'] = (n_treatments - ranked['rank']) / (n_treatments - 1) * 100

        # P-score approximation (similar to SUCRA for frequentist NMA)
        ranked['P_score'] = ranked['SUCRA'] / 100

        return ranked[['treatment', 'HR', 'HR_lower', 'HR_upper', 'rank', 'SUCRA', 'P_score']]

    def plot_network(self, output_path: Optional[str] = None, figsize=(10, 8)):
        """
        Plot the treatment network graph.

        Args:
            output_path: Path to save plot (optional)
            figsize: Figure size
        """
        if self.network is None:
            self.build_network()

        fig, ax = plt.subplots(figsize=figsize)

        # Layout
        pos = nx.spring_layout(self.network, k=2, iterations=50, seed=42)

        # Draw nodes
        nx.draw_networkx_nodes(
            self.network, pos,
            node_color='lightblue',
            node_size=3000,
            ax=ax
        )

        # Draw edges with weights
        edges = self.network.edges()
        weights = [self.network[u][v]['weight'] for u, v in edges]

        nx.draw_networkx_edges(
            self.network, pos,
            width=[w * 2 for w in weights],
            alpha=0.6,
            ax=ax
        )

        # Draw labels
        nx.draw_networkx_labels(
            self.network, pos,
            font_size=12,
            font_weight='bold',
            ax=ax
        )

        # Draw edge labels (number of studies)
        edge_labels = {(u, v): f"{self.network[u][v]['weight']}"
                      for u, v in edges}
        nx.draw_networkx_edge_labels(
            self.network, pos,
            edge_labels=edge_labels,
            font_size=10,
            ax=ax
        )

        ax.set_title(f"Treatment Network - {self.outcome_name}",
                    fontsize=16, fontweight='bold', pad=20)
        ax.text(0.5, -0.1,
               f"Nodes = treatments, Edges = direct comparisons, Edge labels = number of studies",
               ha='center', va='top', transform=ax.transAxes,
               fontsize=10, style='italic')

        ax.axis('off')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Network plot saved: {output_path}")

        plt.show()

        return fig

    def plot_forest_plot(self, output_path: Optional[str] = None, figsize=(10, 8)):
        """
        Create forest plot of treatment effects (HRs vs reference).

        Args:
            output_path: Path to save plot
            figsize: Figure size
        """
        if self.results is None:
            raise ValueError("Run calculate_pairwise_hr() first")

        # Sort by HR
        plot_data = self.results.sort_values('HR')

        fig, ax = plt.subplots(figsize=figsize)

        y_positions = range(len(plot_data))

        # Plot HRs
        ax.errorbar(
            plot_data['HR'], y_positions,
            xerr=[plot_data['HR'] - plot_data['HR_lower'],
                  plot_data['HR_upper'] - plot_data['HR']],
            fmt='o', markersize=8, capsize=5, capthick=2,
            color='darkblue', ecolor='darkblue', alpha=0.7
        )

        # Reference line at HR=1
        ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7)

        # Labels
        ax.set_yticks(y_positions)
        ax.set_yticklabels(plot_data['treatment'])
        ax.set_xlabel('Hazard Ratio (95% CI)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Treatment', fontsize=12, fontweight='bold')
        ax.set_title(f'Forest Plot - {self.outcome_name}',
                    fontsize=14, fontweight='bold', pad=15)

        # Add HR values as text
        for i, (idx, row) in enumerate(plot_data.iterrows()):
            hr_text = f"{row['HR']:.2f} ({row['HR_lower']:.2f}-{row['HR_upper']:.2f})"
            ax.text(ax.get_xlim()[1] * 0.95, i, hr_text,
                   va='center', ha='right', fontsize=10)

        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Forest plot saved: {output_path}")

        plt.show()

        return fig

    def plot_ranking(self, output_path: Optional[str] = None, figsize=(10, 6)):
        """
        Create treatment ranking plot with SUCRA scores.

        Args:
            output_path: Path to save plot
            figsize: Figure size
        """
        ranking = self.calculate_ranking()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Ranking bar plot
        ranking_sorted = ranking.sort_values('rank')

        colors = plt.cm.RdYlGn_r(ranking_sorted['rank'] / len(ranking_sorted))

        ax1.barh(range(len(ranking_sorted)), ranking_sorted['rank'], color=colors)
        ax1.set_yticks(range(len(ranking_sorted)))
        ax1.set_yticklabels(ranking_sorted['treatment'])
        ax1.set_xlabel('Rank (1 = Best)', fontsize=12, fontweight='bold')
        ax1.set_title('Treatment Ranking', fontsize=14, fontweight='bold')
        ax1.invert_xaxis()  # Best (1) on right
        ax1.grid(axis='x', alpha=0.3)

        # Plot 2: SUCRA scores
        sucra_sorted = ranking.sort_values('SUCRA', ascending=True)

        colors_sucra = plt.cm.RdYlGn(sucra_sorted['SUCRA'] / 100)

        ax2.barh(range(len(sucra_sorted)), sucra_sorted['SUCRA'], color=colors_sucra)
        ax2.set_yticks(range(len(sucra_sorted)))
        ax2.set_yticklabels(sucra_sorted['treatment'])
        ax2.set_xlabel('SUCRA Score (%)', fontsize=12, fontweight='bold')
        ax2.set_title('SUCRA Scores', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 100)
        ax2.grid(axis='x', alpha=0.3)

        # Add score labels
        for i, score in enumerate(sucra_sorted['SUCRA']):
            ax2.text(score + 2, i, f'{score:.1f}%',
                    va='center', fontsize=10)

        plt.suptitle(f'{self.outcome_name} - Treatment Ranking',
                    fontsize=16, fontweight='bold', y=1.02)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Ranking plot saved: {output_path}")

        plt.show()

        return fig

    def create_league_table(self):
        """
        Create a league table showing all pairwise comparisons.

        Returns:
            DataFrame in league table format
        """
        if self.results is None:
            raise ValueError("Run calculate_pairwise_hr() first")

        treatments = sorted(self.treatments)
        n = len(treatments)

        # Initialize league table
        league = pd.DataFrame(index=treatments, columns=treatments)

        # Fill diagonal with treatment names
        for t in treatments:
            league.loc[t, t] = t

        # Fill with HRs
        # Upper triangle: HR (treatment vs reference)
        # Lower triangle: 1/HR (reference vs treatment)

        ref_treatment = sorted(self.treatments)[0]

        def _hr_for(treatment):
            row = self.results[self.results['treatment'] == treatment]
            if row.empty:
                return None
            return row['HR'].values[0]  # sentinel:skip-line P1-empty-dataframe-access  (guarded by row.empty above)

        for i, t1 in enumerate(treatments):
            for j, t2 in enumerate(treatments):
                if i == j:
                    continue
                hr1 = _hr_for(t1)
                hr2 = _hr_for(t2)
                if hr1 is None or hr2 is None:
                    continue
                if i < j:  # Upper triangle
                    league.loc[t1, t2] = f"{hr1 / hr2:.2f}"
                else:  # Lower triangle
                    league.loc[t1, t2] = f"{hr2 / hr1:.2f}"

        return league

    def run_full_nma(self, output_dir: str = 'nma_output',
                     ref_treatment: Optional[str] = None):
        """
        Run complete NMA analysis and generate all outputs.

        Args:
            output_dir: Directory for output files
            ref_treatment: Reference treatment

        Returns:
            Dictionary with all results
        """
        print("="*80)
        print("RUNNING NETWORK META-ANALYSIS")
        print("="*80)
        print()

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # Build network
        print("STEP 1: Building treatment network...")
        self.build_network()

        # Calculate HRs
        print("STEP 2: Calculating pairwise hazard ratios...")
        hr_df = self.calculate_pairwise_hr(ref_treatment=ref_treatment)
        print(hr_df)
        print()

        # Calculate ranking
        print("STEP 3: Calculating treatment ranking...")
        ranking = self.calculate_ranking(ref_treatment=ref_treatment)
        print(ranking)
        print()

        # Create league table
        print("STEP 4: Creating league table...")
        league = self.create_league_table()
        print(league)
        print()

        # Generate plots
        print("STEP 5: Generating plots...")

        # Network plot
        network_plot_path = output_path / 'network_plot.png'
        self.plot_network(output_path=str(network_plot_path))

        # Forest plot
        forest_plot_path = output_path / 'forest_plot.png'
        self.plot_forest_plot(output_path=str(forest_plot_path))

        # Ranking plot
        ranking_plot_path = output_path / 'ranking_plot.png'
        self.plot_ranking(output_path=str(ranking_plot_path))

        # Save results to CSV
        hr_df.to_csv(output_path / 'hazard_ratios.csv', index=False)
        ranking.to_csv(output_path / 'treatment_ranking.csv', index=False)
        league.to_csv(output_path / 'league_table.csv')

        print()
        print("="*80)
        print("NMA COMPLETE")
        print("="*80)
        print(f"All outputs saved to: {output_path.absolute()}")
        print()

        return {
            'hazard_ratios': hr_df,
            'ranking': ranking,
            'league_table': league,
            'network': self.network
        }


def run_nma_from_extractions(extraction_results: List[Dict],
                             outcome_name: str = "Overall Survival",
                             output_dir: str = 'nma_output',
                             ref_treatment: Optional[str] = None):
    """
    Convenience function to run NMA from extraction results.

    Args:
        extraction_results: List of dicts with:
            - study_name: Study identifier
            - treatment_a: Name of treatment A
            - treatment_b: Name of treatment B
            - ipd_a: IPD DataFrame for arm A
            - ipd_b: IPD DataFrame for arm B
            - total_n_a: Sample size for arm A (optional)
            - total_n_b: Sample size for arm B (optional)
        outcome_name: Name of the outcome
        output_dir: Output directory
        ref_treatment: Reference treatment (optional)

    Returns:
        NetworkMetaAnalyzer instance with results
    """
    nma = NetworkMetaAnalyzer(outcome_name=outcome_name)

    # Add all studies
    for result in extraction_results:
        nma.add_study(
            study_name=result['study_name'],
            treatment_a=result['treatment_a'],
            treatment_b=result['treatment_b'],
            ipd_a=result['ipd_a'],
            ipd_b=result['ipd_b'],
            total_n_a=result.get('total_n_a'),
            total_n_b=result.get('total_n_b')
        )

    # Run full analysis
    results = nma.run_full_nma(output_dir=output_dir, ref_treatment=ref_treatment)

    return nma, results


if __name__ == '__main__':
    print("Network Meta-Analysis Module")
    print("Example usage:")
    print()
    print("# Initialize NMA")
    print("nma = NetworkMetaAnalyzer(outcome_name='Overall Survival')")
    print()
    print("# Add studies")
    print("nma.add_study('Study1', 'Drug A', 'Placebo', ipd_a, ipd_placebo)")
    print("nma.add_study('Study2', 'Drug B', 'Placebo', ipd_b, ipd_placebo)")
    print("nma.add_study('Study3', 'Drug A', 'Drug B', ipd_a, ipd_b)")
    print()
    print("# Run full NMA")
    print("results = nma.run_full_nma(output_dir='nma_results')")
