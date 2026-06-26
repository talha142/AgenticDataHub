import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Any
from RAG.ingestion.selenium_base import SeleniumScraper


class ScientificDatabaseParser:
    """
    Parser for the 4 target scientific databases using real APIs and web scraping.
    
    Data Sources:
    - PhytoHub (phytohub.eu) — for dietary phytochemicals and food sources
    - CARD (card.mcmaster.ca) — for antibiotic resistance ontology
    """

    def __init__(self):
        self.scraper = SeleniumScraper(headless=True)

    # ─── PhytoHub (fixed real URLs) ───────────────────────────────────────

    def parse_phytohub(self, url: str) -> Dict[str, Any]:
        """
        Parses PhytoHub compound or search pages using Selenium.
        
        Supports two URL patterns:
        - Search: https://phytohub.eu/search/compounds?query=epigallocatechin
        - Entry:  https://phytohub.eu/entries/PHUB000265
        """
        try:
            print(f"    [PhytoHub] Fetching from {url}...")
            soup = self.scraper.get_page_source(url)

            if "/search/compounds" in url:
                return self._parse_phytohub_search(soup, url)
            elif "/entries/" in url:
                return self._parse_phytohub_entry(soup, url)
            else:
                print(f"    [PhytoHub] Unrecognized URL pattern: {url}")
                return {"source": "PhytoHub", "url": url, "text": ""}

        except Exception as e:
            print(f"    [PhytoHub] Error fetching {url}: {e}")
            return {"source": "PhytoHub", "url": url, "text": ""}

    def _parse_phytohub_search(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Parses a PhytoHub compound search results page."""
        compounds = []
        food_sources = []

        # Extract compound names (links to /entries/PHUBXXXXXX)
        for link in soup.select('a[href*="/entries/PHUB"]'):
            name = link.get_text(strip=True)
            if name and name not in compounds and name != "Show Entry":
                compounds.append(name)

        # Extract food sources (links to /food_sources/)
        for link in soup.select('a[href*="/food_sources/"]'):
            food = link.get_text(strip=True)
            if food and food not in food_sources:
                food_sources.append(food)

        # Extract query from URL for context
        query = ""
        if "query=" in url:
            query = url.split("query=")[1].split("&")[0]

        parts = []
        if query:
            parts.append(f"PhytoHub search results for '{query}'.")
        if compounds:
            parts.append(f"Compounds found: {', '.join(compounds[:10])}.")
        if food_sources:
            parts.append(f"Associated food sources: {', '.join(food_sources[:15])}.")

        full_text = "\n".join(parts) if parts else ""

        print(f"    [PhytoHub] Found {len(compounds)} compounds and {len(food_sources)} food sources")
        return {
            "source": "PhytoHub",
            "url": url,
            "text": full_text
        }

    def _parse_phytohub_entry(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Parses a PhytoHub entry detail page."""
        # Try to get compound name from the title tag
        title = soup.find("title")
        compound_name = ""
        if title:
            title_text = title.get_text(strip=True)
            if "Showing entry for" in title_text:
                compound_name = title_text.split("Showing entry for")[-1].strip()
            elif "PhytoHub" in title_text:
                compound_name = title_text.replace("PhytoHub:", "").strip()

        # Extract food sources
        food_sources = []
        for link in soup.select('a[href*="/food_sources/"]'):
            food = link.get_text(strip=True)
            if food and food not in food_sources and food.lower() != "show":
                food_sources.append(food)

        # Extract InChI or SMILES if present
        identifiers = []
        for code_block in soup.select("code, pre"):
            text = code_block.get_text(strip=True)
            if text.startswith("InChI="):
                identifiers.append(f"InChI: {text[:80]}...")
            elif len(text) > 10 and not text.startswith("<"):
                identifiers.append(f"SMILES: {text[:80]}")

        parts = []
        if compound_name:
            parts.append(f"Phytochemical: {compound_name}.")
        if food_sources:
            parts.append(f"Found in food sources: {', '.join(food_sources[:10])}.")
        if identifiers:
            parts.append(f"Chemical identifiers: {'; '.join(identifiers[:2])}.")

        full_text = "\n".join(parts) if parts else ""

        print(f"    [PhytoHub] Parsed entry: {compound_name or 'Unknown'}")
        return {
            "source": "PhytoHub",
            "url": url,
            "text": full_text
        }

    # ─── CARD (real URL, fixed parser) ────────────────────────────────────

    def parse_card(self, url: str) -> Dict[str, Any]:
        """
        Parses CARD (Comprehensive Antibiotic Resistance Database) ontology pages using Selenium.
        Example URL: https://card.mcmaster.ca/ontology/36026
        """
        try:
            print(f"    [CARD] Fetching from {url}...")
            soup = self.scraper.get_page_source(url)

            # Extract the ARO term name from the page title or headers
            aro_name = ""
            title = soup.find("title")
            if title:
                title_text = title.get_text(strip=True)
                if "|" in title_text:
                    aro_name = title_text.split("|")[0].strip()
                else:
                    aro_name = title_text

            if not aro_name or "CARD" in aro_name:
                h_tags = soup.find_all(['h1', 'h2'])
                for h in h_tags:
                    text = h.get_text(strip=True)
                    if len(text) > 3 and "ontology" not in text.lower():
                        aro_name = text
                        break

            if not aro_name or "CARD" in aro_name:
                download_link = soup.find("a", href=lambda h: h and "download" in h and "name=" in h)
                if download_link:
                    href = download_link.get("href", "")
                    if "name=" in href:
                        aro_name = href.split("name=")[-1].replace("+", " ").replace("%20", " ")

            description = ""
            paragraphs = soup.find_all('p')
            for p in paragraphs:
                p_text = p.get_text(strip=True)
                if len(p_text) > 50 and ("resistance" in p_text.lower() or "gene" in p_text.lower() or "protein" in p_text.lower() or "antibiotic" in p_text.lower()):
                    description = p_text
                    break

            mechanisms = []
            drug_classes = []
            related_terms = []
            for link in soup.select('a[href*="/ontology/"]'):
                text = link.get_text(strip=True)
                if text and text != "Show" and not text.startswith("OXA-") and len(text) > 3:
                    if "resistance" in text.lower() or "antibiotic" in text.lower():
                        mechanisms.append(text)
                    elif "lactam" in text.lower() or "penicillin" in text.lower():
                        drug_classes.append(text)
                    elif text not in related_terms:
                        related_terms.append(text)

            publications = []
            for link in soup.select('a[href*="pubmed"]'):
                pub_text = link.parent.get_text(strip=True) if link.parent else ""
                if pub_text and len(pub_text) > 20:
                    publications.append(pub_text[:150])

            parts = []
            if aro_name:
                parts.append(f"ARO Term: {aro_name}.")
            
            if "/ontology/" in url:
                aro_id = url.split("/ontology/")[-1]
                parts.append(f"ARO Accession: ARO:{aro_id}.")
                
            if description:
                parts.append(f"Description: {description}")

            if mechanisms:
                parts.append(f"Resistance mechanisms: {', '.join(mechanisms[:5])}.")
            if drug_classes:
                parts.append(f"Drug classes: {', '.join(drug_classes[:5])}.")
            if publications:
                parts.append(f"Key reference: {publications[0]}.")

            full_text = "\n".join(parts) if parts else ""

            if not full_text.strip():
                print(f"    [CARD] Warning: Parsed content is empty for {url}")

            print(f"    [CARD] Parsed ARO term: {aro_name or 'Unknown'}")
            return {
                "source": "CARD",
                "url": url,
                "text": full_text
            }

        except Exception as e:
            print(f"    [CARD] Error fetching {url}: {e}")
            return {"source": "CARD", "url": url, "text": ""}
    # ─── KEGG (REST API Wrapper) ──────────────────────────────────────────

    def parse_kegg(self, url: str) -> Dict[str, Any]:
        """
        Parses KEGG API data for pathways and compounds.
        Example URL: http://rest.kegg.jp/get/path:map00010
        """
        try:
            print(f"    [KEGG] Fetching from {url}...")
            # For simplicity, we use Selenium here to just grab the raw text, 
            # though requests would be faster for REST API.
            soup = self.scraper.get_page_source(url)
            text_content = soup.get_text()

            lines = text_content.split('\n')
            entry_id = ""
            name = ""
            description = ""
            
            for line in lines:
                if line.startswith("ENTRY"):
                    entry_id = line.split()[1] if len(line.split()) > 1 else ""
                elif line.startswith("NAME"):
                    name = line[12:].strip() if len(line) > 12 else line.split(maxsplit=1)[-1]
                elif line.startswith("DESCRIPTION"):
                    description = line[12:].strip() if len(line) > 12 else line.split(maxsplit=1)[-1]
            
            parts = []
            if entry_id:
                parts.append(f"KEGG Entry: {entry_id}.")
            if name:
                parts.append(f"Name: {name}.")
            if description:
                parts.append(f"Description: {description}.")
            
            full_text = " ".join(parts) if parts else text_content[:500] + "..."
            
            print(f"    [KEGG] Parsed entry: {entry_id or name or 'Unknown'}")
            return {
                "source": "KEGG",
                "url": url,
                "text": full_text
            }
        except Exception as e:
            print(f"    [KEGG] Error fetching {url}: {e}")
            return {"source": "KEGG", "url": url, "text": ""}
    # ─── NPBS (Web Scraper) ───────────────────────────────────────────────

    def parse_npbs(self, url: str) -> Dict[str, Any]:
        """Parses NPBS (Natural Products & Biological Sources) pages."""
        try:
            print(f"    [NPBS] Fetching from {url}...")
            soup = self.scraper.get_page_source(url)
            text_content = soup.get_text(separator=' ', strip=True)
            
            # Simple heuristic text extraction for NPBS
            parts = []
            if "Compound" in text_content or "compound" in text_content.lower():
                parts.append("Contains Compound information.")
            if "Source" in text_content or "source" in text_content.lower():
                parts.append("Contains Source Organism information.")
                
            full_text = " ".join(parts) + "\n\nExtracted Text:\n" + text_content[:1500]
            
            print(f"    [NPBS] Parsed NPBS data.")
            return {
                "source": "NPBS",
                "url": url,
                "text": full_text
            }
        except Exception as e:
            print(f"    [NPBS] Error fetching {url}: {e}")
            return {"source": "NPBS", "url": url, "text": ""}

    def close(self):
        """Closes the Selenium scraper."""
        try:
            self.scraper.close()
        except Exception:
            pass
        print("Webdriver Closed.")


if __name__ == "__main__":
    parser = ScientificDatabaseParser()

    print("=== Testing Real Parsers ===\n")

    print("--- PhytoHub Search ---")
    result = parser.parse_phytohub("https://phytohub.eu/search/compounds?query=epigallocatechin")
    print(f"Result: {result['text'][:200]}...\n")

    print("--- CARD Ontology ---")
    result = parser.parse_card("https://card.mcmaster.ca/ontology/36026")
    print(f"Result: {result['text'][:200]}...\n")

    parser.close()
