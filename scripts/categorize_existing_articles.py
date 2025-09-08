#!/usr/bin/env python3
"""
Categorize existing BlockWireNews articles
This script reads all existing articles and updates their categories based on content analysis.
"""

import os
import re
from pathlib import Path

def determine_categories(content: str) -> list:
    """
    Determine appropriate categories based on article content.
    Returns a list of categories (can be multiple).
    Always includes "News" as a base category.
    """
    categories = ["News"]  # Base category for all articles
    
    # Convert to lowercase for analysis
    full_text = content.lower()
    
    # DeFi indicators
    defi_keywords = [
        'defi', 'decentralized finance', 'yield farming', 'liquidity pool',
        'amm', 'automated market maker', 'uniswap', 'aave', 'compound',
        'makerdao', 'curve', 'sushiswap', 'pancakeswap', 'dex',
        'decentralized exchange', 'lending protocol', 'borrowing protocol',
        'flash loan', 'impermanent loss', 'tvl', 'total value locked',
        'liquidity mining', 'yield aggregator', 'vault', 'staking rewards',
        'governance token', 'dao', 'decentralized autonomous', 'protocol',
        'swap', 'liquidity provider', 'yield', 'lending', 'borrowing'
    ]
    
    # Bitcoin indicators
    bitcoin_keywords = [
        'bitcoin', 'btc', 'satoshi', 'halving', 'mining', 'proof of work',
        'lightning network', 'taproot', 'segwit', 'bitcoin etf', 'grayscale gbtc',
        'michael saylor', 'microstrategy', 'bitcoin treasury', 'bitcoin mining',
        'hash rate', 'difficulty adjustment', 'bitcoin dominance', 'sat', 'sats',
        'bitcoin maximalist', 'digital gold', 'bitcoin adoption', 'bitcoin reserve',
        'bitcoin price', 'bitcoin futures', 'bitcoin options', 'btc price'
    ]
    
    # Web3 indicators
    web3_keywords = [
        'web3', 'nft', 'non-fungible', 'metaverse', 'gamefi', 'play to earn',
        'opensea', 'blur', 'ordinals', 'inscription', 'ipfs', 'arweave',
        'ens', 'ethereum name service', 'lens protocol', 'farcaster',
        'decentralized social', 'soulbound', 'poap', 'did', 'decentralized identity',
        'wallet connect', 'web3 gaming', 'on-chain', 'smart contract',
        'layer 2', 'polygon', 'arbitrum', 'optimism', 'zk rollup', 'zero knowledge',
        'blockchain gaming', 'crypto gaming', 'token gating', 'web3 social'
    ]
    
    # Special case: Ethereum-specific content (can be both DeFi and Web3)
    ethereum_keywords = ['ethereum', 'eth', 'ether', 'vitalik', 'ethereum foundation']
    
    # Check for DeFi
    defi_score = sum(1 for keyword in defi_keywords if keyword in full_text)
    if defi_score >= 2:  # Need at least 2 DeFi indicators
        categories.append("DeFi")
    
    # Check for Bitcoin (be more specific - needs strong Bitcoin focus)
    bitcoin_score = sum(2 if keyword in ['bitcoin', 'btc'] else 1 
                       for keyword in bitcoin_keywords if keyword in full_text)
    if bitcoin_score >= 3:  # Higher threshold for Bitcoin category
        categories.append("Bitcoin")
    
    # Check for Web3
    web3_score = sum(1 for keyword in web3_keywords if keyword in full_text)
    if web3_score >= 2:  # Need at least 2 Web3 indicators
        categories.append("Web3")
    
    # Special handling for Ethereum
    eth_score = sum(1 for keyword in ethereum_keywords if keyword in full_text)
    if eth_score >= 2:
        # Ethereum articles often fit into DeFi or Web3
        if "DeFi" not in categories and ('protocol' in full_text or 'staking' in full_text):
            categories.append("DeFi")
        if "Web3" not in categories and ('smart contract' in full_text or 'dapp' in full_text):
            categories.append("Web3")
    
    return categories


def update_article_categories(file_path: Path):
    """Update categories in a single article file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract front matter and body
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            front_matter = parts[1]
            body = parts[2]
            full_content = front_matter + body
            
            # Determine new categories
            new_categories = determine_categories(full_content)
            
            # Format categories for YAML
            categories_yaml = ', '.join([f'"{cat}"' for cat in new_categories])
            
            # Update categories line in front matter
            new_front_matter = re.sub(
                r'categories:\s*\[.*?\]',
                f'categories: [{categories_yaml}]',
                front_matter
            )
            
            # Only update if categories changed
            if new_front_matter != front_matter:
                # Reconstruct the file
                new_content = f"---{new_front_matter}---{body}"
                
                # Write back to file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                return True, new_categories
    
    return False, []


def main():
    """Process all articles in the content/news directory."""
    content_root = Path("content/news")
    
    if not content_root.exists():
        print(f"Error: {content_root} directory not found!")
        return
    
    updated_count = 0
    processed_count = 0
    
    # Track category distribution
    category_stats = {}
    
    # Find all index.md files
    for article_path in content_root.rglob("index.md"):
        processed_count += 1
        updated, categories = update_article_categories(article_path)
        
        if updated:
            updated_count += 1
            relative_path = article_path.relative_to(content_root)
            print(f"✓ Updated: {relative_path}")
            print(f"  Categories: {', '.join(categories)}")
        
        # Track statistics
        for cat in categories if categories else ["News"]:
            category_stats[cat] = category_stats.get(cat, 0) + 1
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Processed {processed_count} articles")
    print(f"Updated {updated_count} articles with new categories")
    print(f"\nCategory Distribution:")
    for cat, count in sorted(category_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count} articles")


if __name__ == "__main__":
    main()