#!/usr/bin/env python3
"""
Download K-M curve PDFs from bioRxiv/medRxiv preprint servers.

These servers have 100% PDF availability (all preprints are PDFs).
Much better success rate than PMC.
"""
import requests
import time
import json
from pathlib import Path
from typing import List, Dict
import re


class BioRxivDownloader:
    """Downloader for bioRxiv/medRxiv preprints with K-M curves."""

    def __init__(self, output_dir: str = "test_pdfs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.output_dir / 'biorxiv').mkdir(exist_ok=True)
        (self.output_dir / 'medrxiv').mkdir(exist_ok=True)

        self.downloaded = []
        self.failed = []

        # bioRxiv API base
        self.biorxiv_api = "https://api.biorxiv.org/details/biorxiv"
        self.medrxiv_api = "https://api.biorxiv.org/details/medrxiv"

    def search_biorxiv(self, start_date: str = "2020-01-01", end_date: str = "2024-12-31", max_results: int = 100) -> List[Dict]:
        """
        Search bioRxiv for preprints in date range.

        bioRxiv API returns all preprints in date range, we'll filter client-side.
        """
        print(f"Searching bioRxiv preprints from {start_date} to {end_date}...")

        # bioRxiv API: /details/[server]/[start_date]/[end_date]/[cursor]
        url = f"{self.biorxiv_api}/{start_date}/{end_date}/0"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            papers = data.get('collection', [])
            print(f"  Found {len(papers)} bioRxiv preprints in date range")

            # Filter for oncology/cancer/survival keywords
            filtered = []
            for paper in papers:
                title = paper.get('title', '').lower()
                abstract = paper.get('abstract', '').lower()

                # Look for survival/Kaplan-Meier keywords
                if any(kw in title or kw in abstract for kw in [
                    'kaplan-meier', 'kaplan meier', 'survival', 'oncology',
                    'cancer', 'tumor', 'tumour', 'prognosis', 'time-to-event'
                ]):
                    filtered.append({
                        'doi': paper['doi'],
                        'title': paper['title'],
                        'authors': paper.get('authors', 'Unknown'),
                        'date': paper['date'],
                        'category': paper.get('category', 'Unknown'),
                        'server': 'biorxiv'
                    })

                    if len(filtered) >= max_results:
                        break

            print(f"  Filtered to {len(filtered)} oncology/survival papers")
            return filtered[:max_results]

        except Exception as e:
            print(f"Error searching bioRxiv: {e}")
            return []

    def search_medrxiv(self, start_date: str = "2020-01-01", end_date: str = "2024-12-31", max_results: int = 100) -> List[Dict]:
        """Search medRxiv for preprints."""
        print(f"Searching medRxiv preprints from {start_date} to {end_date}...")

        url = f"{self.medrxiv_api}/{start_date}/{end_date}/0"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            papers = data.get('collection', [])
            print(f"  Found {len(papers)} medRxiv preprints in date range")

            # Filter for oncology/survival keywords
            filtered = []
            for paper in papers:
                title = paper.get('title', '').lower()
                abstract = paper.get('abstract', '').lower()

                if any(kw in title or kw in abstract for kw in [
                    'kaplan-meier', 'kaplan meier', 'survival', 'oncology',
                    'cancer', 'tumor', 'tumour', 'prognosis', 'time-to-event'
                ]):
                    filtered.append({
                        'doi': paper['doi'],
                        'title': paper['title'],
                        'authors': paper.get('authors', 'Unknown'),
                        'date': paper['date'],
                        'category': paper.get('category', 'Unknown'),
                        'server': 'medrxiv'
                    })

                    if len(filtered) >= max_results:
                        break

            print(f"  Filtered to {len(filtered)} oncology/survival papers")
            return filtered[:max_results]

        except Exception as e:
            print(f"Error searching medRxiv: {e}")
            return []

    def download_pdf(self, paper: Dict) -> bool:
        """
        Download PDF from bioRxiv/medRxiv.

        URL format: https://www.biorxiv.org/content/[DOI].full.pdf
        """
        doi = paper['doi']
        server = paper['server']

        # Construct PDF URL
        pdf_url = f"https://www.{server}.org/content/{doi}.full.pdf"

        # Create safe filename
        # DOI format: 10.1101/2024.01.15.24301234
        doi_parts = doi.split('/')
        date_id = doi_parts[-1] if len(doi_parts) > 1 else doi

        filename = f"{server}_{date_id}.pdf"
        output_path = self.output_dir / server / filename

        # Truncate title for display
        title_display = paper['title'][:60] + "..." if len(paper['title']) > 60 else paper['title']
        print(f"Downloading {server} {date_id}: {title_display}")

        try:
            response = requests.get(pdf_url, timeout=60)
            response.raise_for_status()

            # Check if we got a PDF
            content_type = response.headers.get('content-type', '')
            if 'pdf' not in content_type.lower():
                print(f"  -> Warning: Content-Type is {content_type}, might not be PDF")

            # Save PDF
            with open(output_path, 'wb') as f:
                f.write(response.content)

            # Verify file size
            file_size = output_path.stat().st_size
            if file_size < 10000:  # Less than 10KB is suspicious
                print(f"  -> Warning: File size only {file_size} bytes, might be invalid")
                self.failed.append({
                    'paper': paper,
                    'reason': f'File too small ({file_size} bytes)'
                })
                output_path.unlink()  # Delete suspicious file
                return False

            print(f"  -> Saved to {output_path} ({file_size:,} bytes)")

            self.downloaded.append({
                'doi': doi,
                'path': str(output_path),
                'metadata': paper,
                'size_bytes': file_size
            })

            return True

        except Exception as e:
            print(f"  -> Error: {e}")
            self.failed.append({
                'paper': paper,
                'reason': str(e)
            })
            return False

    def download_batch(self, target_count: int = 40):
        """Download target number of PDFs from both servers."""
        print(f"\n{'='*70}")
        print(f"DOWNLOADING {target_count} K-M CURVE PDFs FROM BIORXIV/MEDRXIV")
        print(f"{'='*70}\n")

        # Search both servers
        # Split target: 60% from medRxiv (more medical), 40% from bioRxiv
        medrxiv_target = int(target_count * 0.6)
        biorxiv_target = target_count - medrxiv_target

        print(f"Target: {medrxiv_target} from medRxiv, {biorxiv_target} from bioRxiv\n")

        # Search medRxiv first (more medical content)
        medrxiv_papers = self.search_medrxiv(
            start_date="2020-01-01",
            end_date="2024-12-31",
            max_results=medrxiv_target * 2  # Search for 2x to account for failures
        )

        # Then bioRxiv
        biorxiv_papers = self.search_biorxiv(
            start_date="2020-01-01",
            end_date="2024-12-31",
            max_results=biorxiv_target * 2
        )

        # Combine
        all_papers = medrxiv_papers + biorxiv_papers
        print(f"\nTotal papers to download: {len(all_papers)}")
        print(f"Starting downloads...\n")

        # Download
        for paper in all_papers:
            if len(self.downloaded) >= target_count:
                print(f"\nTarget reached: {target_count} PDFs downloaded")
                break

            self.download_pdf(paper)
            time.sleep(1)  # Be polite to servers

        self.generate_report()
        return len(self.downloaded)

    def generate_report(self):
        """Generate download report."""
        print(f"\n{'='*70}")
        print("DOWNLOAD SUMMARY")
        print(f"{'='*70}")
        print(f"Successfully downloaded: {len(self.downloaded)} PDFs")
        print(f"Failed downloads: {len(self.failed)}")

        if self.downloaded:
            total_size = sum(d['size_bytes'] for d in self.downloaded)
            print(f"Total size: {total_size / 1024 / 1024:.1f} MB")

        print(f"{'='*70}\n")

        # Save metadata
        metadata_path = self.output_dir / 'biorxiv_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump({
                'downloaded': self.downloaded,
                'failed': self.failed,
                'summary': {
                    'total_downloaded': len(self.downloaded),
                    'total_failed': len(self.failed),
                    'total_size_mb': sum(d['size_bytes'] for d in self.downloaded) / 1024 / 1024 if self.downloaded else 0
                }
            }, f, indent=2)

        print(f"Metadata saved to: {metadata_path}")

        # Print sample
        if self.downloaded:
            print("\nSample of downloaded papers:")
            for paper in self.downloaded[:5]:
                meta = paper['metadata']
                print(f"  - [{meta['server'].upper()}] {meta['title'][:70]}...")
                print(f"    DOI: {meta['doi']}")
                print(f"    Date: {meta['date']}")
                print(f"    File: {paper['path']}\n")

        if self.failed:
            print(f"\nFailed downloads ({len(self.failed)}):")
            for fail in self.failed[:5]:
                print(f"  - {fail['paper']['title'][:60]}...")
                print(f"    Reason: {fail['reason']}\n")


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description='Download K-M curve PDFs from bioRxiv/medRxiv')
    parser.add_argument('--count', type=int, default=40, help='Number of PDFs to download')
    parser.add_argument('--output', default='test_pdfs', help='Output directory')

    args = parser.parse_args()

    downloader = BioRxivDownloader(output_dir=args.output)
    count = downloader.download_batch(target_count=args.count)

    print("\nDownload complete!")
    print(f"Downloaded {count} PDFs")
    print(f"\nNext step: Run batch processor on downloaded PDFs:")
    print(f"  python batch_processor.py {args.output} batch_results")


if __name__ == "__main__":
    main()
