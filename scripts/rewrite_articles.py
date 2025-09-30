#!/usr/bin/env python3
"""
Retroactive Article Rewriter - Updates existing articles to millennial voice
Safely rewrites published articles while preserving metadata and URLs

Usage:
  python rewrite_articles.py --dry-run           # Preview without changes
  python rewrite_articles.py --limit 5           # Rewrite only 5 articles
  python rewrite_articles.py --since 2024-01-01  # Only articles after date
  python rewrite_articles.py --backup            # Create backups before rewriting
"""

import os
import re
import json
import shutil
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import random

import yaml
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# ==== CONFIGURATION ====
REPO_ROOT = Path(__file__).resolve().parents[1]
CONTENT_ROOT = REPO_ROOT / "content" / "news"
BACKUP_ROOT = REPO_ROOT / "backups" / "articles"
REWRITE_LOG = REPO_ROOT / "data" / "rewrite_log.json"

# Rewrite prompt specifically for tone updates
REWRITE_SYSTEM_PROMPT = """
You're tasked with rewriting existing crypto news articles to have a more conversational, millennial-friendly tone.
Keep ALL facts, figures, dates, and sources exactly the same - only change the voice and style.
Make it sound like a millennial crypto enthusiast explaining news to a friend.
Use natural language, occasional humor when appropriate, and modern internet culture references.
The goal is to make the same information more engaging and relatable.
"""

REWRITE_USER_TEMPLATE = """
Rewrite this crypto article to be more conversational and millennial-friendly.

CRITICAL RULES:
1. Keep ALL facts, numbers, dates, and quotes EXACTLY the same
2. Maintain the same overall structure and information
3. Keep the same sources and attributions
4. Only change the tone, voice, and style
5. Make it conversational but still informative

Original article content:
---
{article_content}
---

Return a JSON object with these keys:
- "title": Rewritten title (60-70 chars, catchy but not clickbait)
- "description": SEO description (140-155 chars)
- "body": The rewritten article body

Style guidelines:
- Use "we" and "you" to create connection
- Include parentheticals for quick asides
- Use conversational transitions
- Acknowledge when things are confusing or absurd
- Break up long sentences
- Add appropriate cultural references
- Keep it real without being unprofessional

Respond ONLY with a fenced JSON block:
```json
{{ ... }}
```
"""

class ArticleRewriter:
    def __init__(self, dry_run=False, backup=True):
        self.dry_run = dry_run
        self.backup = backup
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.rewrite_log = self._load_rewrite_log()
        
    def _load_rewrite_log(self) -> Dict:
        """Load log of previously rewritten articles."""
        if REWRITE_LOG.exists():
            with open(REWRITE_LOG, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_rewrite_log(self):
        """Save rewrite log."""
        REWRITE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(REWRITE_LOG, 'w') as f:
            json.dump(self.rewrite_log, f, indent=2)
    
    def _parse_article(self, filepath: Path) -> Tuple[Dict, str]:
        """Parse article markdown file into frontmatter and content."""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split frontmatter and body
        parts = content.split('---', 2)
        if len(parts) < 3:
            raise ValueError(f"Invalid markdown format in {filepath}")
        
        frontmatter_str = parts[1].strip()
        body = parts[2].strip()
        
        # Parse frontmatter
        frontmatter = yaml.safe_load(frontmatter_str)
        
        return frontmatter, body
    
    def _create_backup(self, filepath: Path):
        """Create backup of article before rewriting."""
        if not self.backup:
            return
            
        backup_path = BACKUP_ROOT / filepath.relative_to(CONTENT_ROOT)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp to backup filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_path.parent / f"{backup_path.stem}_{timestamp}{backup_path.suffix}"
        
        shutil.copy2(filepath, backup_file)
        print(f"  📁 Backup created: {backup_file}")
    
    @retry(wait=wait_exponential(multiplier=2, min=4, max=60), stop=stop_after_attempt(3))
    def _call_llm(self, article_content: str) -> Dict:
        """Call OpenAI to rewrite article."""
        user_prompt = REWRITE_USER_TEMPLATE.format(article_content=article_content)
        
        response = self.client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.7,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        
        out = response.choices[0].message.content or ""
        
        # Parse JSON response
        m = re.search(r"```json(.*?)```", out, re.S | re.I)
        if m:
            return json.loads(m.group(1).strip())
        return json.loads(out.strip())
    
    def _should_rewrite(self, filepath: Path, frontmatter: Dict) -> bool:
        """Determine if article should be rewritten."""
        # Check if already rewritten
        article_id = str(filepath.relative_to(CONTENT_ROOT))
        if article_id in self.rewrite_log:
            print(f"  ⏭️  Already rewritten: {filepath.name}")
            return False
        
        # Check for markers that might indicate it's already in new voice
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().lower()
        
        # Look for millennial voice markers
        millennial_markers = [
            "here's the thing",
            "but wait",
            "crypto twitter",
            "not financial advice, obviously",
            "dyor",
            "(yes, this",
            "let's be real",
            "stay safe out there",
        ]
        
        marker_count = sum(1 for marker in millennial_markers if marker in content)
        if marker_count >= 3:
            print(f"  🎯 Already has millennial voice markers: {filepath.name}")
            return False
        
        return True
    
    def rewrite_article(self, filepath: Path) -> bool:
        """Rewrite a single article."""
        try:
            print(f"\n📝 Processing: {filepath.relative_to(CONTENT_ROOT)}")
            
            # Parse article
            frontmatter, body = self._parse_article(filepath)
            
            # Check if should rewrite
            if not self._should_rewrite(filepath, frontmatter):
                return False
            
            # Extract just the article content (remove sources, disclaimers, etc.)
            # This helps focus the rewrite on the actual content
            article_parts = body.split("### Sources")
            main_content = article_parts[0].strip()
            
            # Remove existing newsletter blocks, affiliate blocks, etc.
            main_content = re.sub(r'{{<.*?>}}', '', main_content)
            main_content = re.sub(r'---\n+', '', main_content)
            
            # Create backup
            if not self.dry_run:
                self._create_backup(filepath)
            
            # Rewrite with LLM
            print("  🤖 Calling LLM for rewrite...")
            rewritten = self._call_llm(main_content)
            
            # Update frontmatter
            if rewritten.get('title'):
                frontmatter['title'] = rewritten['title'].replace('"', "'")
            if rewritten.get('description'):
                frontmatter['description'] = rewritten['description'].replace('"', "'")
                if 'seo' in frontmatter:
                    frontmatter['seo']['meta_description'] = rewritten['description'].replace('"', "'")
            
            # Add rewrite timestamp
            frontmatter['last_rewrite'] = datetime.datetime.now().isoformat() + "Z"
            
            # Reconstruct article with new content
            new_body = rewritten['body']
            
            # Re-add sources if they existed
            if len(article_parts) > 1:
                sources_section = "### Sources" + article_parts[1]
                new_body = new_body.rstrip() + "\n\n" + sources_section
            
            # Add millennial-style newsletter block
            newsletter_block = """
{{< newsletter-inline >}}

📧 **Want crypto news that doesn't put you to sleep?** Get our weekly digest straight to your inbox. No spam, just the good stuff.

---

"""
            
            # Add disclaimer
            disclaimers = [
                "_Not financial advice, obviously. We're just here for the vibes and information. Do your own research!_",
                "_This is just news, not financial advice. DYOR and maybe don't bet the farm on magic internet money._",
                "_Quick reminder: This isn't financial advice. We're just keeping you in the loop. Stay safe out there!_",
            ]
            disclaimer = random.choice(disclaimers)
            
            # Find TL;DR section if it exists
            if "###" in new_body and "TL;DR" not in new_body and "bullets" in main_content:
                # Try to add a TL;DR section at the beginning
                new_body = "## The TL;DR 📝\n\n" + new_body
            
            # Construct final content
            final_body = f"{newsletter_block}{new_body}\n\n---\n\n{disclaimer}"
            
            # Reconstruct full markdown
            frontmatter_yaml = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)
            new_content = f"---\n{frontmatter_yaml}---\n\n{final_body}\n"
            
            if self.dry_run:
                print("  🔍 DRY RUN - Would rewrite:")
                print(f"     Old title: {frontmatter.get('title', 'N/A')}")
                print(f"     New title: {rewritten.get('title', 'N/A')}")
                print(f"     First 200 chars of new body: {new_body[:200]}...")
            else:
                # Write updated content
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                # Log the rewrite
                article_id = str(filepath.relative_to(CONTENT_ROOT))
                self.rewrite_log[article_id] = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "original_title": frontmatter.get('title'),
                    "new_title": rewritten.get('title'),
                }
                self._save_rewrite_log()
                
                print(f"  ✅ Successfully rewritten!")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Error rewriting {filepath}: {e}")
            return False
    
    def get_articles_to_rewrite(self, limit: Optional[int] = None, 
                                since_date: Optional[str] = None) -> List[Path]:
        """Get list of articles to rewrite based on criteria."""
        articles = []
        
        for year_dir in sorted(CONTENT_ROOT.iterdir(), reverse=True):
            if not year_dir.is_dir() or not year_dir.name.isdigit():
                continue
                
            for month_dir in sorted(year_dir.iterdir(), reverse=True):
                if not month_dir.is_dir() or not month_dir.name.isdigit():
                    continue
                
                for article_dir in sorted(month_dir.iterdir(), reverse=True):
                    if not article_dir.is_dir():
                        continue
                    
                    index_path = article_dir / "index.md"
                    if not index_path.exists():
                        continue
                    
                    # Check date filter
                    if since_date:
                        try:
                            frontmatter, _ = self._parse_article(index_path)
                            article_date = datetime.datetime.fromisoformat(
                                frontmatter.get('date', '').replace('Z', '+00:00')
                            )
                            filter_date = datetime.datetime.fromisoformat(since_date)
                            if article_date < filter_date:
                                continue
                        except:
                            continue
                    
                    articles.append(index_path)
                    
                    if limit and len(articles) >= limit:
                        return articles
        
        return articles

def main():
    parser = argparse.ArgumentParser(description='Rewrite existing articles with millennial voice')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Preview changes without modifying files')
    parser.add_argument('--limit', type=int, 
                       help='Limit number of articles to rewrite')
    parser.add_argument('--since', type=str, 
                       help='Only rewrite articles after this date (YYYY-MM-DD)')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip creating backups')
    parser.add_argument('--specific', type=str,
                       help='Rewrite specific article by path relative to content/news/')
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return
    
    # Initialize rewriter
    rewriter = ArticleRewriter(dry_run=args.dry_run, backup=not args.no_backup)
    
    if args.specific:
        # Rewrite specific article
        filepath = CONTENT_ROOT / args.specific / "index.md"
        if not filepath.exists():
            print(f"❌ Article not found: {filepath}")
            return
        
        success = rewriter.rewrite_article(filepath)
        print(f"\n{'✅ Complete!' if success else '❌ Failed'}")
    else:
        # Get articles to rewrite
        articles = rewriter.get_articles_to_rewrite(limit=args.limit, since_date=args.since)
        
        if not articles:
            print("No articles found to rewrite")
            return
        
        print(f"Found {len(articles)} articles to rewrite")
        if args.dry_run:
            print("🔍 DRY RUN MODE - No files will be modified")
        
        if input("\nProceed? (y/n): ").lower() != 'y':
            print("Cancelled")
            return
        
        # Process articles
        success_count = 0
        for i, article_path in enumerate(articles, 1):
            print(f"\n[{i}/{len(articles)}]")
            if rewriter.rewrite_article(article_path):
                success_count += 1
            
            # Rate limiting
            if not args.dry_run and i < len(articles):
                time.sleep(random.uniform(2, 4))
        
        # Summary
        print(f"\n{'='*50}")
        print(f"✅ Successfully rewrote: {success_count}/{len(articles)} articles")
        if not args.dry_run:
            print(f"📁 Backups saved to: {BACKUP_ROOT}")
            print(f"📊 Rewrite log: {REWRITE_LOG}")

if __name__ == "__main__":
    main()