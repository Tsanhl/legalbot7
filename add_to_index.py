#!/usr/bin/env python3
"""
Add New Documents to Existing Index
====================================
This script allows you to add new documents to the existing RAG index
without re-indexing everything.

Usage:
    python add_to_index.py "path/to/new/folder"
    python add_to_index.py "Law resouces  copy 2/New PIL Materials"
    
The script will:
1. Index only the documents in the specified folder
2. Add them to the existing ChromaDB index
3. Keep all previously indexed documents intact
"""

import os
import sys
from rag_service import get_rag_service

def add_folder_to_index(folder_path: str):
    """Add documents from a specific folder to the existing index."""
    
    # Get absolute path
    if not os.path.isabs(folder_path):
        folder_path = os.path.join(os.getcwd(), folder_path)
    
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder not found: {folder_path}")
        print("\n📁 Available folders in 'Law resouces  copy 2':")
        law_resources = os.path.join(os.getcwd(), 'Law resouces  copy 2')
        if os.path.exists(law_resources):
            for item in os.listdir(law_resources):
                item_path = os.path.join(law_resources, item)
                if os.path.isdir(item_path):
                    print(f"   • {item}")
        return False
    
    if not os.path.isdir(folder_path):
        print(f"❌ Error: Path is not a folder: {folder_path}")
        return False
    
    print('='*60)
    print('📚 ADD NEW DOCUMENTS TO INDEX')
    print('='*60)
    print(f"\n📂 Folder to index: {folder_path}")
    
    # Count files
    pdf_count = 0
    docx_count = 0
    txt_count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.pdf'):
                pdf_count += 1
            elif file.endswith('.docx'):
                docx_count += 1
            elif file.endswith('.txt'):
                txt_count += 1
    
    total_files = pdf_count + docx_count + txt_count
    print(f"\n📊 Found {total_files} documents:")
    print(f"   • PDFs:  {pdf_count}")
    print(f"   • DOCX:  {docx_count}")
    print(f"   • TXT:   {txt_count}")
    
    if total_files == 0:
        print("\n⚠️  No documents found to index!")
        return False
    
    # Confirm
    print(f"\n🔄 Adding these documents to the existing index...")
    
    # Get RAG service (uses existing ChromaDB)
    rag = get_rag_service()
    
    # Get current stats
    try:
        current_count = rag.collection.count()
        print(f"📈 Current index size: {current_count} chunks")
    except:
        current_count = 0
    
    # Progress callback
    def progress(count, filename):
        if count % 10 == 0 or count == total_files:
            print(f'   ✓ Processed {count}/{total_files} files... ({os.path.basename(filename)})')
    
    # Index the folder
    try:
        # Skip BM25 rebuild here (can be very slow on large DBs); it will rebuild lazily on first search.
        stats = rag.index_documents(folder_path, progress_callback=progress, rebuild_bm25=False)
        
        # Get new stats
        try:
            new_count = rag.collection.count()
        except:
            new_count = current_count + stats.get('chunks', 0)
        
        print('\n' + '='*60)
        print('🎉 INDEXING COMPLETE!')
        print('='*60)
        print(f"\n📊 Results for this folder:")
        print(f"   • Documents processed: {stats['processed']}")
        print(f"   • New chunks added:    {stats['chunks']}")
        print(f"   • Errors:              {stats['errors']}")
        print(f"   • Skipped:             {stats['skipped']}")
        print(f"\n📈 Total index size: {new_count} chunks")
        print(f"   (Added {new_count - current_count} new chunks)")
        print('='*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Indexing Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def list_current_folders():
    """List available folders that can be indexed."""
    print('='*60)
    print('📁 AVAILABLE FOLDERS')
    print('='*60)
    
    law_resources = os.path.join(os.getcwd(), 'Law resouces  copy 2')
    if os.path.exists(law_resources):
        print(f"\nIn '{law_resources}':")
        for item in sorted(os.listdir(law_resources)):
            item_path = os.path.join(law_resources, item)
            if os.path.isdir(item_path):
                # Count files
                count = sum(1 for r, d, f in os.walk(item_path) 
                           for file in f if file.endswith(('.pdf', '.docx', '.txt')))
                print(f"   • {item} ({count} documents)")
    else:
        print(f"⚠️  'Law resouces  copy 2' folder not found")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python add_to_index.py <folder_path>")
        print("\nExamples:")
        print('  python add_to_index.py "Law resouces  copy 2/Competition Law copy"')
        print('  python add_to_index.py "Law resouces  copy 2/Private international law materials"')
        print('  python add_to_index.py "Law resouces  copy 2/New Folder"')
        print("\n")
        list_current_folders()
        sys.exit(1)
    
    folder_path = sys.argv[1]
    success = add_folder_to_index(folder_path)
    sys.exit(0 if success else 1)
