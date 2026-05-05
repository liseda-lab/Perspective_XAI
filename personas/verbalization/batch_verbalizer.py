"""
batch_verbalizer.py
------------------
Batch processing script for drug repurposing graph verbalization.
Processes all folders with 3 images each and creates combined output files.

Expected image naming pattern:
- [drug]_[disease]_elena.png -> Elena persona (mechanistic focus)
- [drug]_[disease]_leo.png -> Leo persona (clinical focus) 
- [drug]_[disease]_rex.png -> Generic explanation (no persona)
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import argparse
from unified_graph_verbalizer import UnifiedGraphVerbalizer, DetailLevel


class BatchVerbalizer:
    
    def __init__(self):
        self.verbalizer = UnifiedGraphVerbalizer()
        self.persona_files = {
            'elena': 'elena.txt',
            'leo': 'leo.txt'
            # rex = generic (no persona file)
        }
    
    def parse_folder_name(self, folder_name: str) -> Tuple[str, str]:
        """Extract drug and disease from folder name like '1. Benzphetamine_obesit'"""
        # Remove the number prefix
        match = re.match(r'\d+\.\s*(.+)', folder_name)
        if match:
            name_part = match.group(1)
            # Split on underscore, assuming drug_disease format
            parts = name_part.split('_')
            if len(parts) >= 2:
                drug = parts[0]
                disease = '_'.join(parts[1:])  # In case disease has underscores
                return drug, disease
        
        # Fallback - use folder name as drug
        return folder_name, "unknown"
    
    def find_images_in_folder(self, folder_path: Path) -> List[Path]:
        """Find all PNG images in a folder"""
        return sorted(list(folder_path.glob('*.png')))
    
    def identify_persona_from_filename(self, image_path: Path) -> str:
        """Identify persona from image filename"""
        filename = image_path.stem.lower()
        
        if 'elena' in filename:
            return 'elena'
        elif 'leo' in filename:
            return 'leo'
        elif 'rex' in filename:
            return 'rex'  # rex = generic explanation
        else:
            # Try to guess from filename patterns
            for persona in self.persona_files.keys():
                if persona in filename:
                    return persona
            return 'generic'
    
    def process_folder(self, folder_path: Path, detail_level: DetailLevel = DetailLevel.STANDARD, 
                      output_dir: Path = None) -> str:
        """Process all images in a single folder"""
        
        print(f"\n{'='*60}")
        print(f"Processing folder: {folder_path.name}")
        print(f"{'='*60}")
        
        # Parse drug-disease from folder name
        drug, disease = self.parse_folder_name(folder_path.name)
        drug_disease_pair = {"drug": drug, "disease": disease}
        
        print(f"Identified: {drug} → {disease}")
        
        # Find images
        images = self.find_images_in_folder(folder_path)
        if not images:
            print(f"No PNG images found in {folder_path}")
            return ""
        
        print(f"Found {len(images)} images: {[img.name for img in images]}")
        
        # Prepare output file
        if output_dir is None:
            output_dir = folder_path
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"{folder_path.name}_combined_{timestamp}.txt"
        
        # Process each image and collect results
        results = []
        
        for image_path in images:
            print(f"\nProcessing: {image_path.name}")
            
            # Identify persona
            persona_key = self.identify_persona_from_filename(image_path)
            persona_file = None
            
            # Handle persona assignment
            if persona_key == 'rex':
                # rex = generic explanation (no persona file)
                persona_key = 'generic'
                persona_file = None
            elif persona_key in self.persona_files:
                # elena or leo with persona file
                persona_file = self.persona_files[persona_key]
                if not Path(persona_file).exists():
                    print(f"Warning: Persona file {persona_file} not found, using generic")
                    persona_file = None
                    persona_key = 'generic'
            else:
                # fallback to generic
                persona_key = 'generic'
                persona_file = None
            
            # Get verbalization (don't save individually)
            try:
                description = self.verbalizer.verbalize(
                    image_path=str(image_path),
                    persona_file=persona_file,
                    drug_disease_pair=drug_disease_pair,
                    detail_level=detail_level,
                    save_to_file=False  # We'll save everything together
                )
                
                results.append({
                    'image': image_path.name,
                    'persona': persona_key,
                    'persona_file': persona_file,
                    'description': description
                })
                
            except Exception as e:
                error_msg = f"Error processing {image_path.name}: {str(e)}"
                print(error_msg)
                results.append({
                    'image': image_path.name,
                    'persona': persona_key,
                    'persona_file': persona_file,
                    'description': error_msg
                })
        
        # Save combined results
        self.save_combined_results(output_file, folder_path.name, drug_disease_pair, 
                                  results, detail_level)
        
        print(f"\nFolder processing complete!")
        print(f"Combined output saved to: {output_file}")
        
        return str(output_file)
    
    def save_combined_results(self, output_file: Path, folder_name: str, 
                            drug_disease_pair: Dict, results: List[Dict], 
                            detail_level: DetailLevel):
        """Save all results to a single combined file"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Drug Repurposing Graph - Combined Verbalizations\n")
            f.write("=" * 60 + "\n")
            f.write(f"Folder: {folder_name}\n")
            f.write(f"Drug: {drug_disease_pair.get('drug', 'N/A')}\n")
            f.write(f"Disease: {drug_disease_pair.get('disease', 'N/A')}\n")
            f.write(f"Detail Level: {detail_level.value}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Images: {len(results)}\n")
            f.write("\n" + "=" * 60 + "\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"{i}. {result['image']}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Persona: {result['persona']}")
                if result['persona_file']:
                    f.write(f" ({result['persona_file']})")
                f.write("\n\n")
                
                f.write(result['description'])
                f.write("\n\n")
                
                if i < len(results):
                    f.write("~" * 40 + "\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("End of Combined Verbalizations\n")
    
    def process_all_folders(self, base_path: str = ".", detail_level: DetailLevel = DetailLevel.STANDARD,
                           output_dir: str = None, folder_pattern: str = None) -> List[str]:
        """Process all folders matching the pattern"""
        
        base_path = Path(base_path)
        output_dir = Path(output_dir) if output_dir else None
        
        if output_dir and not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find folders to process
        folders_to_process = []
        
        for item in base_path.iterdir():
            if not item.is_dir():
                continue
            
            # Skip system folders
            if item.name.startswith('.') or item.name.lower() in ['visualization', '__pycache__']:
                continue
            
            # Apply pattern filter if provided
            if folder_pattern and folder_pattern not in item.name:
                continue
            
            # Check if folder has PNG images
            png_files = list(item.glob('*.png'))
            if png_files:
                folders_to_process.append(item)
        
        if not folders_to_process:
            print("No folders with PNG images found to process!")
            return []
        
        print(f"Found {len(folders_to_process)} folders to process:")
        for folder in sorted(folders_to_process):
            png_count = len(list(folder.glob('*.png')))
            print(f"  - {folder.name} ({png_count} PNG files)")
        
        # Process each folder
        processed_files = []
        total_folders = len(folders_to_process)
        
        for i, folder in enumerate(sorted(folders_to_process), 1):
            print(f"\n{'#'*80}")
            print(f"BATCH PROGRESS: {i}/{total_folders}")
            print(f"{'#'*80}")
            
            try:
                output_file = self.process_folder(folder, detail_level, output_dir)
                processed_files.append(output_file)
            except Exception as e:
                error_msg = f"Failed to process folder {folder.name}: {str(e)}"
                print(f"ERROR: {error_msg}")
                continue
        
        return processed_files


def main():
    parser = argparse.ArgumentParser(description="Batch Drug Repurposing Graph Verbalizer")
    parser.add_argument("--base-path", default=".", help="Base directory to search for folders")
    parser.add_argument("--output-dir", help="Directory to save output files (default: same as input folders)")
    parser.add_argument("--detail", choices=["brief", "standard", "comprehensive"], 
                       default="standard", help="Detail level for all verbalizations")
    parser.add_argument("--pattern", help="Only process folders containing this pattern")
    parser.add_argument("--folder", help="Process only this specific folder")
    
    args = parser.parse_args()
    
    print("Batch Drug Repurposing Graph Verbalizer")
    print("=" * 60)
    print(f"Base path: {Path(args.base_path).absolute()}")
    print(f"Detail level: {args.detail}")
    if args.output_dir:
        print(f"Output directory: {args.output_dir}")
    if args.pattern:
        print(f"Folder pattern: {args.pattern}")
    print()
    
    batch_verbalizer = BatchVerbalizer()
    
    if args.folder:
        # Process single folder
        folder_path = Path(args.base_path) / args.folder
        if not folder_path.exists():
            print(f"Error: Folder {folder_path} does not exist!")
            return
        
        output_dir = Path(args.output_dir) if args.output_dir else None
        detail_level = DetailLevel(args.detail)
        
        output_file = batch_verbalizer.process_folder(folder_path, detail_level, output_dir)
        print(f"\nSingle folder processing complete!")
        print(f"Output saved to: {output_file}")
        
    else:
        # Process all folders
        detail_level = DetailLevel(args.detail)
        processed_files = batch_verbalizer.process_all_folders(
            base_path=args.base_path,
            detail_level=detail_level,
            output_dir=args.output_dir,
            folder_pattern=args.pattern
        )
        
        print(f"\n{'='*80}")
        print("BATCH PROCESSING COMPLETE!")
        print(f"{'='*80}")
        print(f"Successfully processed {len(processed_files)} folders")
        
        if processed_files:
            print("\nOutput files created:")
            for file_path in processed_files:
                print(f"  - {file_path}")
        print()


if __name__ == "__main__":
    main()

    # Example usage:
    # 
    # # Process all folders in current directory
    # python batch_verbalizer.py
    # 
    # # Process with comprehensive detail level
    # python batch_verbalizer.py --detail comprehensive
    # 
    # # Process and save to specific output directory
    # python batch_verbalizer.py --output-dir ./results
    # 
    # # Process only folders containing "obesity"
    # python batch_verbalizer.py --pattern obesity
    # 
    # # Process single folder
    # python batch_verbalizer.py --folder "1. Benzphetamine_obesit"
    #
    # Expected file naming in each folder:
    # - [drug]_[disease]_elena.png -> Elena persona (mechanistic)
    # - [drug]_[disease]_leo.png -> Leo persona (clinical)
    # - [drug]_[disease]_rex.png -> Generic explanation (no persona)