"""
Graph verbalizer with proper token limits and sentence completion checks.
"""

import base64
import os
from typing import Optional, Dict, Union
from pathlib import Path
from enum import Enum
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import time
import random
import re

load_dotenv()


class DetailLevel(Enum):
    BRIEF = "brief"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


class UnifiedGraphVerbalizer:
    
    def __init__(self, api_key: Optional[str] = None):
        print("Initializing drug repurposing graph verbalizer...")
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("No API key provided. Set OPENAI_API_KEY in .env file")
        
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
        print(f"Using model: {self.model}")
        
        # Rate limit tracking
        self.last_api_call = 0
        self.min_delay_between_calls = 2  # Minimum 2 seconds between API calls
    
    def encode_image(self, image_path: str) -> str:
        print(f"Encoding image: {image_path}")
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
        print(f"Image encoded successfully ({len(encoded)} characters)")
        return encoded
    
    def wait_for_rate_limit(self):
        """Ensure minimum delay between API calls"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        
        if time_since_last_call < self.min_delay_between_calls:
            wait_time = self.min_delay_between_calls - time_since_last_call
            print(f"Rate limit protection: waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
    
    def check_incomplete_sentence(self, text: str) -> bool:
        """Check if text ends with an incomplete sentence"""
        text = text.strip()
        if not text:
            return False
        
        # Check for common sentence endings
        complete_endings = ['.', '!', '?', '."', '!"', '?"', '.]', '!]', '?]']
        for ending in complete_endings:
            if text.endswith(ending):
                return False
        
        # Check for incomplete patterns
        incomplete_patterns = [
            r'\b(is|was|are|were|has|have|had|will|would|can|could|may|might|must|shall|should)\s*$',
            r'\b(the|a|an|this|that|these|those)\s*$',
            r'\b(and|or|but|nor|for|yet|so)\s*$',
            r'\b(with|without|from|to|by|for|of|in|on|at|as)\s*$',
            r',\s*$',
            r':\s*$',
            r';\s*$',
            r'\b(which|that|who|whom|whose|where|when|why|how)\s*$'
        ]
        
        for pattern in incomplete_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Additional check: if last sentence doesn't end with punctuation
        last_sentence_start = max(
            text.rfind('. ') + 2 if '. ' in text else 0,
            text.rfind('! ') + 2 if '! ' in text else 0,
            text.rfind('? ') + 2 if '? ' in text else 0,
            0
        )
        last_part = text[last_sentence_start:].strip()
        
        # If the last part is very short and doesn't end with punctuation, likely incomplete
        if last_part and len(last_part) < 20 and not any(last_part.endswith(p) for p in ['.', '!', '?']):
            return True
            
        return False
    
    def contains_concluding_statement(self, text: str) -> bool:
        """Check if text contains overall/concluding statements"""
        concluding_patterns = [
            r'\b(overall|in conclusion|to conclude|in summary|to summarize|ultimately|finally)\b',
            r'\b(this (demonstrates|shows|illustrates|highlights|underscores|reveals) (the|that|how))\b',
            r'\b(these (connections|relationships|pathways|interactions) (demonstrate|show|illustrate))\b',
            r'\b(the (graph|network|diagram) (demonstrates|shows|illustrates|reveals))\b',
            r'\b(this (comprehensive|complex|intricate) (network|graph|system))\b'
        ]
        
        for pattern in concluding_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def verbalize(self,
                  image_path: str,
                  persona_file: Optional[str] = None,
                  drug_disease_pair: Optional[Dict] = None,
                  detail_level: Union[str, DetailLevel] = DetailLevel.STANDARD,
                  save_to_file: bool = True,
                  output_path: Optional[str] = None) -> str:
        """
        Verbalize a drug repurposing graph.
        """
        
        print("\n" + "="*50)
        mode = f"persona ({Path(persona_file).stem})" if persona_file else "generic"
        print(f"Starting {mode} verbalization...")
        
        if drug_disease_pair:
            print(f"Drug-disease prediction: {drug_disease_pair.get('drug', 'unknown')} → {drug_disease_pair.get('disease', 'unknown')}")
        
        base64_image = self.encode_image(image_path)
        
        if isinstance(detail_level, str):
            detail_level = DetailLevel(detail_level)
        
        # Context about the prediction
        context_info = ""
        if drug_disease_pair:
            drug = drug_disease_pair.get('drug', 'the drug')
            disease = drug_disease_pair.get('disease', 'the disease')
            context_info = f" The graph shows paths from {drug} to {disease}."
        
        # Enhanced prompts with stronger instructions
        base_prompts = {
            DetailLevel.BRIEF: f"""Write a brief, flowing description of this drug repurposing graph in 2-3 COMPLETE sentences.{context_info}
Describe the main relationships as natural, connected prose.
Write as flowing paragraphs only.

CRITICAL REQUIREMENTS:
- Do NOT describe the explanation in generic ways, make sure all sentences include a specific reference to the elements in the paths
- Every sentence MUST be grammatically complete with proper endings.
- Do NOT use bullet points, numbered lists, or dashes.
- Do NOT write partial or incomplete sentences
- Do NOT include overall statements like "This demonstrates" or "Overall"
- Do NOT write concluding or summary statements
- Focus ONLY on describing the specific connections you see
- Stop naturally after 2-3 complete sentences""",
            
            DetailLevel.STANDARD: f"""Write a natural, flowing description of the connections in this drug repurposing explanation.{context_info}
Cover the key relationships in 4-6 COMPLETE sentences that flow together naturally.
If there are grey colored nodes, mention these as ontological expansions (lowest common ancestors).
Write as connected prose, not as bullet points or lists.
Use transitional phrases to connect ideas smoothly.

CRITICAL REQUIREMENTS:
- Do NOT describe the explanation in generic ways, make sure all sentences include a specific reference to the elements in the paths
- Every sentence MUST be grammatically complete with proper endings
- Do NOT write partial or incomplete sentences
- Do NOT include overall statements like "This demonstrates" or "Overall"
- Do NOT write concluding or summary statements
- Focus ONLY on describing the specific connections you see
- Continue describing pathways and relationships without meta-commentary
- Stop naturally after covering the key relationships""",

            DetailLevel.COMPREHENSIVE: f"""Write a comprehensive, flowing description of all connections in this drug repurposing explanation.{context_info}
Include every relationship shown in connected, natural prose.
Write as flowing paragraphs with smooth transitions between ideas.
Never use bullet points, numbered lists, dashes, or any list format.
Connect all relationships into coherent, readable text.


CRITICAL REQUIREMENTS:
- Do NOT describe the explanation in generic ways, make sure all sentences include a specific reference to the elements in the paths
- Every sentence MUST be grammatically complete with proper endings
- Do NOT write partial or incomplete sentences
- Do NOT cut off mid-thought or mid-sentence
- Do NOT include overall statements like "This demonstrates" or "Overall"
- Do NOT write concluding or summary statements
- Focus ONLY on describing the specific connections you see
- Continue describing pathways until all major relationships are covered
- If running out of space, complete the current sentence properly before stopping"""
        }
        
        base_prompt = base_prompts[detail_level]
        
        # Add persona if provided
        if persona_file:
            print(f"Loading persona from: {persona_file}")
            with open(persona_file, "r", encoding="utf-8") as f:
                persona_text = f.read().strip()
            print(f"Persona loaded ({len(persona_text)} characters)")
            
            prompt = f"""Adopt this persona completely:
{persona_text}

{base_prompt}

You must embody this persona's specific priorities, standards, and interests. Write ONLY about what this persona would find valuable and relevant. Filter the entire description through this persona's lens - include details they would care about and completely omit information they would consider irrelevant or superficial.

Match this persona's preferred depth, focus, and style. Be selective and discriminating based on what matters to THIS specific viewpoint.

Never mention any names from the perspective description.

REMEMBER: Complete all sentences properly. No incomplete thoughts and  Do NOT describe the explanation in generic ways, make sure all sentences include a specific reference to the elements in the paths
"""
            
        else:
            print(f"Using detail level: {detail_level.value}")
            prompt = base_prompt
        
        # Adjusted parameters for better completion
        temperature = 0.3  # Slightly increased for more natural flow
        
        # RESTORED and INCREASED token limits to prevent cutoffs
        max_tokens = {
            DetailLevel.BRIEF: 120,         # Increased from 60
            DetailLevel.STANDARD: 300,      # Doubled from 150
            DetailLevel.COMPREHENSIVE: 500  # Doubled from 250
        }[detail_level]
        print(f"Making API call (max_tokens={max_tokens}, temp={temperature})...")
        
        # Enhanced retry logic
        max_retries = 5
        base_wait_time = 5
        
        for attempt in range(max_retries):
            try:
                # Wait before API call (rate limit protection)
                self.wait_for_rate_limit()
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url", 
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                        "detail": "high"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=45
                )
                
                # Update last API call time
                self.last_api_call = time.time()
                
                result = response.choices[0].message.content
                print("✅ API call successful")
                
                # Check for incomplete sentences or concluding statements
                if self.check_incomplete_sentence(result):
                    print("⚠️ Warning: Output appears to have incomplete sentences")
                    if attempt < 2:  # Try once more with higher token limit
                        print(f"Retrying with increased token limit...")
                        max_tokens = min(max_tokens + 100, 800)
                        time.sleep(2)
                        continue
                
                if self.contains_concluding_statement(result):
                    print("⚠️ Warning: Output contains concluding/overall statements")
                
                if save_to_file:
                    self.save_result(result, image_path, persona_file, drug_disease_pair, output_path)
                
                return result
                
            except Exception as e:
                error_str = str(e)
                print(f"❌ Attempt {attempt + 1} failed: {error_str}")
                
                # Check if it's a rate limit error
                if "rate_limit" in error_str.lower() or "429" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = base_wait_time * (2 ** attempt) + random.uniform(1, 3)
                        wait_time = min(wait_time, 120)
                        print(f"⏳ Rate limit hit - waiting {wait_time:.1f} seconds...")
                        time.sleep(wait_time)
                    else:
                        error_msg = f"Rate limit exceeded after {max_retries} attempts. Try again later."
                        print(error_msg)
                        return error_msg
                else:
                    if attempt < max_retries - 1:
                        wait_time = min(2 ** attempt, 10)
                        print(f"⏳ Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        error_msg = f"Failed after {max_retries} attempts: {error_str}"
                        print(error_msg)
                        if save_to_file:
                            self.save_result(error_msg, image_path, persona_file, drug_disease_pair, output_path)
                        return error_msg
    
    def save_result(self, result: str, image_path: str, persona_file: Optional[str], 
                    drug_disease_pair: Optional[Dict], output_path: Optional[str] = None):
        """Save the result to a text file."""
        
        if output_path:
            filename = output_path
            output_dir = Path(filename).parent
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_name = Path(image_path).stem
            
            if persona_file:
                persona_name = Path(persona_file).stem
                filename = f"{image_name}_{persona_name}_{timestamp}.txt"
            else:
                filename = f"{image_name}_generic_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("Drug Repurposing Graph Description\n")
            f.write("="*50 + "\n")
            f.write(f"Image: {image_path}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            if drug_disease_pair:
                f.write(f"Drug: {drug_disease_pair.get('drug', 'N/A')}\n")
                f.write(f"Disease: {drug_disease_pair.get('disease', 'N/A')}\n")
            
            if persona_file:
                f.write(f"Persona: {persona_file}\n")
            
            f.write(f"\n{'-'*50}\n\n")
            f.write(result)
            
            # Add warning if issues detected
            if self.check_incomplete_sentence(result):
                f.write("\n\n[WARNING: Output may contain incomplete sentences]")
            if self.contains_concluding_statement(result):
                f.write("\n[WARNING: Output contains concluding/overall statements]")
        
        print(f"💾 Result saved to: {filename}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Drug Repurposing Graph Verbalizer")
    parser.add_argument("--image", default="example.png", help="Path to graph image")
    parser.add_argument("--persona", help="Path to persona file (optional)")
    parser.add_argument("--drug", help="Drug name for context")
    parser.add_argument("--disease", help="Disease name for context")
    parser.add_argument("--detail", choices=["brief", "standard", "comprehensive"], 
                       default="standard", help="Detail level")
    parser.add_argument("--no-save", action="store_true", help="Don't save to file")
    parser.add_argument("--both", action="store_true", help="Run both generic and persona")
    parser.add_argument("--output", help="Custom output file path")
    
    args = parser.parse_args()
    
    print("Drug Repurposing Graph Verbalizer (Complete Sentences Fix)")
    print("-" * 60)
    
    verbalizer = UnifiedGraphVerbalizer()
    
    drug_disease = None
    if args.drug and args.disease:
        drug_disease = {"drug": args.drug, "disease": args.disease}
    
    # Run generic
    if not args.persona or args.both:
        print("\n### GENERIC VERBALIZATION ###")
        
        output_path = None
        if args.output and not args.both:
            output_path = args.output
        elif args.output and args.both:
            path = Path(args.output)
            output_path = str(path.parent / f"{path.stem}_generic{path.suffix}")
        
        generic_result = verbalizer.verbalize(
            image_path=args.image,
            persona_file=None,
            drug_disease_pair=drug_disease,
            detail_level=args.detail,
            save_to_file=not args.no_save,
            output_path=output_path
        )
        print(f"\nGeneric Result ({args.detail} level):")
        print(generic_result)
    
    # Run persona if provided
    if args.persona:
        print("\n### PERSONA VERBALIZATION ###")
        
        output_path = None
        if args.output and not args.both:
            output_path = args.output
        elif args.output and args.both:
            path = Path(args.output)
            persona_name = Path(args.persona).stem
            output_path = str(path.parent / f"{path.stem}_{persona_name}{path.suffix}")
        
        persona_result = verbalizer.verbalize(
            image_path=args.image,
            persona_file=args.persona,
            drug_disease_pair=drug_disease,
            detail_level=args.detail,
            save_to_file=not args.no_save,
            output_path=output_path
        )
        print(f"\nPersona Result ({args.detail} level):")
        print(persona_result)
    
    print("\n" + "="*50)
    print("Processing complete!")


if __name__ == "__main__":
    main()