
import os
from rag_service import RAGService

# Initialize service
rag = RAGService()

# Try to find a PDF file to test
print("Searching for PDF files...")
pdf_file = None
for root, dirs, files in os.walk("Law resouces  copy 2"):
    for file in files:
        if file.endswith(".pdf"):
            pdf_file = os.path.join(root, file)
            break
    if pdf_file:
        break

if pdf_file:
    print(f"Testing extraction on: {pdf_file}")
    try:
        # Test extraction directly
        text = rag.extract_text(pdf_file)
        print(f"Extraction result length: {len(text)} characters")
        if len(text) > 0:
            print("Preview:", text[:100])
        else:
            print("Extraction returned empty string")
    except Exception as e:
        print(f"Extraction failed with error: {e}")
        import traceback
        traceback.print_exc()
else:
    print("No PDF files found to test")
