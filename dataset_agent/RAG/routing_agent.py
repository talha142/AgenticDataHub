import os
import re
from enum import Enum
from typing import Dict, Any

class DataSource(Enum):
    PHYTOHUB = "PhytoHub"
    CARD = "CARD"
    KEGG = "KEGG"
    NPBS = "NPBS"
    UNKNOWN = "Unknown"

class RoutingAgent:
    """
    Intelligent Routing Agent that analyzes a given input (query or URL)
    and determines the appropriate database and ingestion/retrieval strategy.
    
    Routes to:
    - PhytoHub (phytohub.eu) — for dietary phytochemicals and food sources
    - CARD (card.mcmaster.ca) — for antibiotic resistance ontology
    """
    
    def __init__(self):
        # Keyword patterns for each database (word-boundary aware)
        self.routing_rules = [
            # CARD — check first with strict word-boundary to avoid "cardamom" etc.
            {
                "pattern": re.compile(r'\bcard\b|mcmaster|antibiotic.?resistance|aro[:\s]|resistance.?gene', re.IGNORECASE),
                "source": DataSource.CARD,
                "strategy": "API_DOWNLOAD"
            },
            # PhytoHub — phytochemicals, dietary sources, metabolites
            {
                "pattern": re.compile(r'phytohub|phytochemical|food.?source|dietary|metabolite.?\d|polyphenol', re.IGNORECASE),
                "source": DataSource.PHYTOHUB,
                "strategy": "WEB_SCRAPE"
            },
            # KEGG — pathways, compounds, metabolic processes
            {
                "pattern": re.compile(r'\bkegg\b|pathway|metabolic.?process|citrate.?cycle|biosynthesis', re.IGNORECASE),
                "source": DataSource.KEGG,
                "strategy": "REST_API"
            },
            # NPBS — natural products, biological sources
            {
                "pattern": re.compile(r'\bnpbs\b|natural.?product|biological.?source', re.IGNORECASE),
                "source": DataSource.NPBS,
                "strategy": "WEB_SCRAPE"
            },
        ]

        # URL-based routing (direct match)
        self.url_mappings = {
            "phytohub.eu": DataSource.PHYTOHUB,
            "card.mcmaster.ca": DataSource.CARD,
            "rest.kegg.jp": DataSource.KEGG,
            "cstspace.cn/npbs": DataSource.NPBS,
        }

    def route_request(self, input_text: str) -> Dict[str, Any]:
        """
        Determines the data source and the strategy (API vs. Scrape).
        """
        source = DataSource.UNKNOWN
        strategy = "GENERAL_SEARCH"
        
        # Priority 1: URL-based routing
        for domain, db_source in self.url_mappings.items():
            if domain in input_text.lower():
                source = db_source
                strategy = "WEB_SCRAPE"
                break
        
        # Priority 2: Keyword-based routing
        if source == DataSource.UNKNOWN:
            for rule in self.routing_rules:
                if rule["pattern"].search(input_text):
                    source = rule["source"]
                    strategy = rule["strategy"]
                    break
        
        return {
            "source": source.value,
            "strategy": strategy,
            "input": input_text
        }

if __name__ == "__main__":
    agent = RoutingAgent()
    
    tests = [
        "Get resistance genes from CARD for E. coli",
        "What are the food sources for EGCG in PhytoHub?",
        "https://card.mcmaster.ca/ontology/36026",
        "What compounds are found in cardamom?",  # Should NOT match CARD
    ]
    
    for test in tests:
        result = agent.route_request(test)
        print(f"Query: {test[:60]:<60} -> Source: {result['source']:<10} Strategy: {result['strategy']}")
