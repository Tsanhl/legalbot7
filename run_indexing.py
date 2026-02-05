
import os
import sys
from rag_service import get_rag_service

def run_indexing():
    print('🚀 Starting full document indexing with Dynamic Chunking...')
    
    # Get absolute path to resources
    cwd = os.getcwd()
    resources_path = os.path.join(cwd, 'Law resouces  copy 2')
    
    if not os.path.exists(resources_path):
        print(f"❌ Error: Resources path not found: {resources_path}")
        return

    rag = get_rag_service()
    
    # Check if files are real (LFS check)
    has_real_files = False
    for root, dirs, files in os.walk(resources_path):
        for file in files:
            if file.endswith('.pdf'):
                path = os.path.join(root, file)
                if os.path.getsize(path) > 1000:  # Larger than 1KB
                    has_real_files = True
                    break
        if has_real_files:
            break
            
    if not has_real_files:
        print("⚠️  WARNING: Files seem too small (git-lfs pointers?). Run 'git lfs pull' first.")
    
    print(f"📂 Indexing documents from: {resources_path}")
    
    def progress(count, filename):
        if count % 20 == 0:
            print(f'   Processed {count} files... ({filename})')

    try:
        stats = rag.index_documents(resources_path, progress_callback=progress)
        print('\n🎉 Indexing Complete!')
        print('='*50)
        print(f"📚 Total Documents: {stats['processed']}")
        print(f"🧩 Total Chunks:    {stats['chunks']}")
        print(f"⚠️  Errors:          {stats['errors']}")
        print(f"⏭️  Skipped:         {stats['skipped']}")
        print('-'*50)
        
        type_stats = stats.get('type_stats', {})
        print('📊 Document Type Breakdown:')
        print(f"   • PB (Problem-Based): {type_stats.get('pb', 0)}")
        print(f"   • General:            {type_stats.get('general', 0)}")
        print(f"   • Essay:              {type_stats.get('essay', 0)}")
        print(f"   • Long Essay:         {type_stats.get('long_essay', 0)}")
        print('='*50)
        
    except Exception as e:
        print(f"\n❌ Indexing Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_indexing()
