#!/usr/bin/env python3
"""
Download K-M curve PDFs from open-access sources for validation testing.

Strategy:
1. Search PubMed Central for oncology K-M curves
2. Download 40 diverse PDFs (different journals, cancer types, styles)
3. Organize by source and cancer type
"""
import requests
import time
import json
from pathlib import Path
from typing import List, Dict
from defusedxml import ElementTree as ET
class KMPDFDownloader:
    """Downloader for K-M curve PDFs from open-access sources."""

    def __init__(self, output_dir: str = "test_pdfs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Create subdirectories by source
        (self.output_dir / 'pmc').mkdir(exist_ok=True)
        (self.output_dir / 'nejm').mkdir(exist_ok=True)
        (self.output_dir / 'other').mkdir(exist_ok=True)

        self.downloaded = []
        self.failed = []

    def search_pmc_for_km_curves(self, max_results: int = 50) -> List[Dict]:
        """
        Search PubMed Central for papers with Kaplan-Meier curves.

        Returns list of PMC IDs and metadata.
        """
        print("Searching PubMed Central for K-M curve papers...")

        # PMC E-utilities API
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

        # Simplified search query for better results
        # Search for: Kaplan-Meier AND oncology/cancer AND recent years
        query = (
            'Kaplan-Meier AND (cancer OR oncology OR survival) '
            'AND ("2020"[PDAT] : "2024"[PDAT])'
        )

        params = {
            'db': 'pmc',
            'term': query,
            'retmax': max_results,
            'retmode': 'json',
            'sort': 'relevance'
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

            pmc_ids = data.get('esearchresult', {}).get('idlist', [])
            print(f"Found {len(pmc_ids)} papers in PMC")

            # Get metadata for each paper
            results = []
            for pmc_id in pmc_ids[:max_results]:
                metadata = self._get_pmc_metadata(pmc_id)
                if metadata:
                    results.append(metadata)
                time.sleep(0.4)  # NCBI rate limit: 3 requests/second

            return results

        except Exception as e:
            print(f"Error searching PMC: {e}")
            return []

    def _get_pmc_metadata(self, pmc_id: str) -> Dict:
        """Get metadata for a PMC article."""
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

        params = {
            'db': 'pmc',
            'id': pmc_id,
            'retmode': 'json'
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

            result = data.get('result', {}).get(pmc_id, {})

            return {
                'pmc_id': pmc_id,
                'title': result.get('title', 'Unknown'),
                'journal': result.get('fulljournalname', 'Unknown'),
                'year': result.get('pubdate', 'Unknown')[:4],
                'doi': result.get('elocationid', ''),
                'pmid': result.get('articleids', [{}])[0].get('value', '')
            }

        except Exception as e:
            print(f"Error getting metadata for PMC{pmc_id}: {e}")
            return None

    def download_pmc_pdf(self, pmc_id: str, metadata: Dict) -> bool:
        """
        Download PDF from PMC.

        PMC PDFs are available at:
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{ID}/pdf/
        """
        pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf/"

        # Create safe filename
        journal_abbrev = metadata['journal'][:20].replace(' ', '_')
        year = metadata['year']
        filename = f"PMC{pmc_id}_{journal_abbrev}_{year}.pdf"
        output_path = self.output_dir / 'pmc' / filename

        print(f"Downloading PMC{pmc_id}: {metadata['title'][:50]}...")

        try:
            response = requests.get(pdf_url, timeout=30)

            # Check if we got HTML instead of PDF (some PMC articles don't have PDFs)
            if response.headers.get('content-type', '').startswith('text/html'):
                print(f"  -> No PDF available for PMC{pmc_id}")
                self.failed.append({
                    'pmc_id': pmc_id,
                    'reason': 'No PDF available',
                    'metadata': metadata
                })
                return False

            response.raise_for_status()

            # Save PDF
            with open(output_path, 'wb') as f:
                f.write(response.content)

            print(f"  -> Saved to {output_path}")

            self.downloaded.append({
                'pmc_id': pmc_id,
                'path': str(output_path),
                'metadata': metadata
            })

            return True

        except Exception as e:
            print(f"  -> Error downloading PMC{pmc_id}: {e}")
            self.failed.append({
                'pmc_id': pmc_id,
                'reason': str(e),
                'metadata': metadata
            })
            return False

    def download_batch(self, target_count: int = 40):
        """Download target number of PDFs."""
        print(f"\n{'='*70}")
        print(f"DOWNLOADING {target_count} K-M CURVE PDFs")
        print(f"{'='*70}\n")

        # Search PMC (request more to account for failures)
        papers = self.search_pmc_for_km_curves(max_results=target_count * 2)

        print(f"\nAttempting to download {len(papers)} papers...")
        print(f"Target: {target_count} successful downloads\n")

        for paper in papers:
            if len(self.downloaded) >= target_count:
                print(f"\n✓ Target reached: {target_count} PDFs downloaded")
                break

            self.download_pmc_pdf(paper['pmc_id'], paper)
            time.sleep(1)  # Be nice to NCBI servers

        self.generate_report()
        return len(self.downloaded)

    def generate_report(self):
        """Generate download report."""
        print(f"\n{'='*70}")
        print("DOWNLOAD SUMMARY")
        print(f"{'='*70}")
        print(f"Successfully downloaded: {len(self.downloaded)} PDFs")
        print(f"Failed downloads: {len(self.failed)}")
        print(f"{'='*70}\n")

        # Save metadata
        metadata_path = self.output_dir / 'download_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump({
                'downloaded': self.downloaded,
                'failed': self.failed,
                'summary': {
                    'total_downloaded': len(self.downloaded),
                    'total_failed': len(self.failed)
                }
            }, f, indent=2)

        print(f"Metadata saved to: {metadata_path}")

        # Print sample of downloaded papers
        if self.downloaded:
            print("\nSample of downloaded papers:")
            for paper in self.downloaded[:5]:
                print(f"  - {paper['metadata']['title'][:60]}...")
                print(f"    {paper['metadata']['journal']} ({paper['metadata']['year']})")
                print(f"    {paper['path']}\n")

    def add_manual_pdfs(self, pdf_paths: List[str]):
        """
        Add manually downloaded PDFs to the test collection.

        Useful for adding specific papers from NEJM, Lancet, etc.
        """
        print("\nAdding manually downloaded PDFs...")

        for pdf_path in pdf_paths:
            src = Path(pdf_path)
            if not src.exists():
                print(f"  Warning: {pdf_path} not found, skipping")
                continue

            # Copy to test_pdfs/other/
            dest = self.output_dir / 'other' / src.name
            import shutil
            shutil.copy2(src, dest)

            print(f"  Added: {src.name}")

            self.downloaded.append({
                'source': 'manual',
                'path': str(dest),
                'metadata': {'title': src.stem}
            })


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description='Download K-M curve PDFs for testing')
    parser.add_argument('--count', type=int, default=40, help='Number of PDFs to download')
    parser.add_argument('--output', default='test_pdfs', help='Output directory')

    args = parser.parse_args()

    downloader = KMPDFDownloader(output_dir=args.output)
    count = downloader.download_batch(target_count=args.count)

    print("\nDownload complete!")
    print(f"\nNext step: Run batch processor on downloaded PDFs:")
    print(f"  python batch_processor.py {args.output} batch_results")


if __name__ == "__main__":
    main()
