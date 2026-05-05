#!/usr/bin/env python3
"""
create_personas.py - Generate synthetic personas from clustered user feedback
"""

import os
import json
import argparse
import re
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import openai
from openai import OpenAI
import pandas as pd
from pathlib import Path


@dataclass
class UserRecord:
    """Represents a single user's feedback record"""
    user_id: str
    background: str
    preferences: str


class PersonaGenerator:
    """Generates synthetic personas from clustered user feedback using LLMs"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """
        Initialize the persona generator
        
        Args:
            api_key: OpenAI API key (if None, reads from environment)
            model: Model to use (default: gpt-4o)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.prompt_template = self._load_prompt_template()
    
    def _load_prompt_template(self) -> str:
        """Load the persona synthesis prompt template"""
        return """Below are individual user feedback records from a group who share similar preferences. Each user has provided their background and detailed preferences.

Core Requirements:
- Analyze frequency patterns: Count how many users mention each theme
- Weight by user count: Prioritize themes mentioned by multiple users
- Flag weak signals: Mark any trait mentioned by <25% of users as potentially unreliable
- Create coherent character narratives: Write profiles as if describing a specific individual, not aggregated user research
- Validate with evidence: Support all persona traits with concrete user counts and percentages

Key Instruction:
Write the Profile section as if describing a real person's preferences and behaviors.
Never mention "users", "this group", or research language within the Profile.
Save all analytical language for the Evidence Summary.

User Data Format:
User ID: [user_id]
Background: [user_background]
Preferences: [full_preference_text]

Analysis Process:
Step 1: Frequency Count
- List each distinct preference theme
- Count how many users (and %) mention each theme
- Identify the top 3-4 most common themes (these become core traits)

Step 2: Evidence Quality Check
- Strong support: 40%+ of users
- Moderate support: 25-40%
- Weak signals: <25%

Step 3: Background Pattern Analysis
- Note background diversity without over-characterizing

Persona Structure:
Name: Descriptive name (avoid titles unless 60%+ of users explicitly hold those roles)
Tagline: One sentence capturing the strongest supported theme (40%+ users)
Profile:
- Lead paragraph: Core traits (40%+ support)
- Secondary paragraph: Additional values (25-40%)
- "Also values..." section: Traits in 15-25% range
- Background note: Brief mention of role diversity

Writing Guidelines:
Good: "Clara prefers direct pathways..."
Good: "She values concise summaries..."
Bad: "Users in this group prefer..."
Bad: "This persona represents users who..."

Evidence Summary:
- Core traits (40%+): List with counts and percentages
- Secondary traits (25-40%): List with counts
- Weak signals (<25%): List or omit
- Background composition: Summary
- Potential concerns: Biases, outliers, alignment issues

User Records:
{individual_user_records}

Output Format:
Name: [Descriptive name]
Tagline: [Strongest theme]
Profile: [Narrative description]
Evidence Summary: [Analytical details]"""
    
    def format_user_records(self, users: List[UserRecord]) -> str:
        """
        Format user records for inclusion in the prompt
        
        Args:
            users: List of UserRecord objects
            
        Returns:
            Formatted string of user records
        """
        formatted_records = []
        for user in users:
            record = f"""User ID: {user.user_id}
Background: {user.background}
Preferences: {user.preferences}
---"""
            formatted_records.append(record)
        
        return "\n".join(formatted_records)
    
    def generate_persona(self, user_records: List[UserRecord], 
                        temperature: float = 0.7,
                        max_tokens: int = 2000) -> Dict:
        """
        Generate a persona from user records
        
        Args:
            user_records: List of UserRecord objects
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response
            
        Returns:
            Dictionary containing persona details
        """
        # Format user records for the prompt
        formatted_records = self.format_user_records(user_records)
        
        # Prepare the prompt
        prompt = self.prompt_template.replace("{individual_user_records}", formatted_records)
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at synthesizing user research into coherent personas."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract the response
            persona_text = response.choices[0].message.content
            
            # Parse the response into structured format
            persona_dict = self._parse_persona_response(persona_text)
            persona_dict['raw_response'] = persona_text
            persona_dict['num_users'] = len(user_records)
            
            return persona_dict
            
        except Exception as e:
            raise Exception(f"Error generating persona: {str(e)}")
    
    def _parse_persona_response(self, response_text: str) -> Dict:
        """
        Parse the LLM response into structured persona data
        
        Args:
            response_text: Raw text response from LLM
            
        Returns:
            Dictionary with parsed persona components
        """
        persona = {
            'name': '',
            'tagline': '',
            'profile': '',
            'evidence_summary': ''
        }
        
        lines = response_text.strip().split('\n')
        current_section = None
        buffer = []
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('Name:'):
                if buffer and current_section:
                    persona[current_section] = '\n'.join(buffer).strip()
                persona['name'] = line.replace('Name:', '').strip()
                current_section = None
                buffer = []
            elif line.startswith('Tagline:'):
                if buffer and current_section:
                    persona[current_section] = '\n'.join(buffer).strip()
                persona['tagline'] = line.replace('Tagline:', '').strip()
                current_section = None
                buffer = []
            elif line.startswith('Profile:'):
                if buffer and current_section:
                    persona[current_section] = '\n'.join(buffer).strip()
                current_section = 'profile'
                buffer = []
            elif line.startswith('Evidence Summary:'):
                if buffer and current_section:
                    persona[current_section] = '\n'.join(buffer).strip()
                current_section = 'evidence_summary'
                buffer = []
            elif current_section and line:
                buffer.append(line)
        
        # Capture the last section
        if buffer and current_section:
            persona[current_section] = '\n'.join(buffer).strip()
        
        return persona
    
    def load_users_from_csv(self, csv_path: str) -> List[UserRecord]:
        """
        Load user records from a CSV file
        
        Expected columns: user_id, background, preferences
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            List of UserRecord objects
        """
        df = pd.read_csv(csv_path)
        
        required_columns = ['user_id', 'background', 'preferences']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        
        users = []
        for _, row in df.iterrows():
            users.append(UserRecord(
                user_id=str(row['user_id']),
                background=str(row['background']),
                preferences=str(row['preferences'])
            ))
        
        return users
    
    def load_users_from_aggregated_json(self, json_path: str, 
                                       persona_index: int = 0,
                                       default_backgrounds: Optional[List[str]] = None) -> List[UserRecord]:
        """
        Load user records from aggregated JSON format
        
        Args:
            json_path: Path to JSON file containing aggregated persona data
            persona_index: Index of persona to process (default: 0)
            default_backgrounds: List of backgrounds to assign to users (cycles through if provided)
            
        Returns:
            List of UserRecord objects parsed from combined comments
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list) or len(data) <= persona_index:
            raise ValueError(f"Invalid JSON format or persona_index {persona_index} out of range")
        
        persona_data = data[persona_index]
        combined_comments = persona_data.get('combined_comments', '')
        
        # Parse individual preferences from combined comments
        # Split by quotes and clean up
        raw_preferences = re.findall(r'"([^"]+)"', combined_comments)
        
        # Clean up preferences
        preferences = []
        for pref in raw_preferences:
            # Clean up encoding issues and whitespace
            cleaned = pref.strip()
            if cleaned and len(cleaned) > 10:  # Filter out very short fragments
                preferences.append(cleaned)
        
        # Default backgrounds if not provided
        if default_backgrounds is None:
            default_backgrounds = [
                "Life and Health Sciences",
                "CS/AI-heavy with Biomedical Experience",
                "Hybrid Computational-Biology",
                "Clinical Practice",
                "Research Science"
            ]
        
        # Create user records from preferences
        users = []
        for i, pref in enumerate(preferences):
            # Cycle through backgrounds
            background = default_backgrounds[i % len(default_backgrounds)]
            
            # Infer background from preference content if possible
            if 'clinician' in pref.lower():
                background = "Clinical Practice"
            elif 'researcher' in pref.lower():
                background = "Research Science"
            elif 'patient' in pref.lower():
                background = "Patient Advocacy"
            elif 'technical' in pref.lower() or 'code' in pref.lower():
                background = "CS/AI-heavy with Biomedical Experience"
            
            users.append(UserRecord(
                user_id=f"user_{i+1}",
                background=background,
                preferences=pref
            ))
        
        return users
    
    def process_all_personas_from_json(self, json_path: str, 
                                      output_dir: str = "personas",
                                      default_backgrounds: Optional[List[str]] = None) -> List[Dict]:
        """
        Process all personas from an aggregated JSON file
        
        Args:
            json_path: Path to JSON file containing aggregated persona data
            output_dir: Directory to save individual persona outputs
            default_backgrounds: List of backgrounds to assign to users
            
        Returns:
            List of generated persona dictionaries
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        personas = []
        for i, persona_data in enumerate(data):
            print(f"\nProcessing {persona_data.get('persona', f'Persona {i+1}')}...")
            
            # Load users for this persona
            users = self.load_users_from_aggregated_json(
                json_path, 
                persona_index=i,
                default_backgrounds=default_backgrounds
            )
            print(f"  Extracted {len(users)} user preferences")
            
            # Generate persona
            persona = self.generate_persona(users)
            persona['original_label'] = persona_data.get('persona', f'Persona {i+1}')
            personas.append(persona)
            
            # Save individual persona
            output_path = Path(output_dir) / f"persona_{i+1}.json"
            self.save_persona(persona, str(output_path))
            
            # Save markdown report
            md_path = Path(output_dir) / f"persona_{i+1}.md"
            md_report = self.generate_markdown_report(persona)
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(md_report)
            
            print(f"  Generated: {persona['name']}")
            print(f"  Saved to: {output_path}")
        
        # Save summary of all personas
        summary_path = Path(output_dir) / "all_personas_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(personas, f, indent=2, ensure_ascii=False)
        
        print(f"\nGenerated {len(personas)} personas")
        print(f"Summary saved to: {summary_path}")
        
        return personas
    
    def save_persona(self, persona: Dict, output_path: str):
        """
        Save persona to a JSON file
        
        Args:
            persona: Persona dictionary
            output_path: Output file path
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(persona, f, indent=2, ensure_ascii=False)
    
    def generate_markdown_report(self, persona: Dict) -> str:
        """
        Generate a markdown report from persona data
        
        Args:
            persona: Persona dictionary
            
        Returns:
            Markdown formatted report
        """
        report = f"""# Persona: {persona['name']}

## Tagline
*{persona['tagline']}*

## Profile
{persona['profile']}

## Evidence Summary
{persona['evidence_summary']}

---
*Generated from {persona.get('num_users', 'unknown')} user records*
*Original label: {persona.get('original_label', 'N/A')}*
"""
        return report


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Generate personas from clustered user feedback")
    parser.add_argument('input', help='Input file (CSV or JSON)')
    parser.add_argument('-o', '--output', help='Output file/directory', default='persona.json')
    parser.add_argument('-m', '--model', help='OpenAI model to use', default='4ogpt-')
    parser.add_argument('-t', '--temperature', type=float, help='Sampling temperature', default=0.7)
    parser.add_argument('--max-tokens', type=int, help='Max tokens in response', default=2000)
    parser.add_argument('--markdown', help='Also save as markdown report', action='store_true')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--json-persona-index', type=int, help='For JSON input: which persona to process', default=0)
    parser.add_argument('--process-all', help='Process all personas in JSON file', action='store_true')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = PersonaGenerator(api_key=args.api_key, model=args.model)
    
    # Determine input type
    input_path = Path(args.input)
    
    if input_path.suffix.lower() == '.json':
        # JSON input (aggregated format)
        if args.process_all:
            # Process all personas
            output_dir = args.output if args.output != 'persona.json' else 'personas'
            print(f"Processing all personas from {args.input}...")
            personas = generator.process_all_personas_from_json(
                args.input,
                output_dir=output_dir
            )
            print(f"\nAll personas processed and saved to {output_dir}/")
        else:
            # Process single persona
            print(f"Loading aggregated data from {args.input}...")
            users = generator.load_users_from_aggregated_json(
                args.input,
                persona_index=args.json_persona_index
            )
            print(f"Extracted {len(users)} user preferences from persona {args.json_persona_index}")
            
            # Generate persona
            print(f"Generating persona using {args.model}...")
            persona = generator.generate_persona(
                users,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )
            
            # Save results
            generator.save_persona(persona, args.output)
            print(f"Persona saved to {args.output}")
            
            # Optionally save markdown report
            if args.markdown:
                md_path = args.output.replace('.json', '.md')
                md_report = generator.generate_markdown_report(persona)
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(md_report)
                print(f"Markdown report saved to {md_path}")
            
            # Print summary
            print(f"\nPersona Generated: {persona['name']}")
            print(f"Tagline: {persona['tagline']}")
    
    else:
        # CSV input (original format)
        print(f"Loading user records from {args.input}...")
        users = generator.load_users_from_csv(args.input)
        print(f"Loaded {len(users)} user records")
        
        # Generate persona
        print(f"Generating persona using {args.model}...")
        persona = generator.generate_persona(
            users,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        
        # Save results
        generator.save_persona(persona, args.output)
        print(f"Persona saved to {args.output}")
        
        # Optionally save markdown report
        if args.markdown:
            md_path = args.output.replace('.json', '.md')
            md_report = generator.generate_markdown_report(persona)
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(md_report)
            print(f"Markdown report saved to {md_path}")
        
        # Print summary
        print(f"\nPersona Generated: {persona['name']}")
        print(f"Tagline: {persona['tagline']}")


if __name__ == "__main__":
    main()
    # This will create a 'personas' directory with all outputs
    #python create_personas.py persona_aggregates_drec.json --process-all

    # Or specify output directory
    #python create_personas.py persona_aggregates_dr_dti.json --process-all -o final_personas