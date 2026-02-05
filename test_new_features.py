"""
Test script for the two new features:
1. detect_specific_para_improvement
2. should_use_google_search_grounding
"""

# Import the functions from gemini_service
import sys
sys.path.append('/Users/hltsang/Desktop/doc-ai copy (latest) 2 self use for essay backup 1')

from gemini_service import detect_specific_para_improvement, should_use_google_search_grounding

print("=" * 80)
print("TESTING FEATURE 1: Specific Paragraph Improvement Detection")
print("=" * 80)

# Test Case 1: Which paragraphs can be improved
test1 = "Can you tell me which paragraphs in my essay need improvement?"
result1 = detect_specific_para_improvement(test1)
print(f"\nTest 1: '{test1}'")
print(f"Result: {result1}")
assert result1['is_para_improvement'] == True
assert result1['improvement_type'] == 'specific_paras'
print("✅ PASSED")

# Test Case 2: Improve whole essay
test2 = "Improve my entire essay"
result2 = detect_specific_para_improvement(test2)
print(f"\nTest 2: '{test2}'")
print(f"Result: {result2}")
assert result2['is_para_improvement'] == True
assert result2['improvement_type'] == 'whole_essay'
print("✅ PASSED")

# Test Case 3: Improve specific paragraphs
test3 = "improve para 2 and para 4"
result3 = detect_specific_para_improvement(test3)
print(f"\nTest 3: '{test3}'")
print(f"Result: {result3}")
assert result3['is_para_improvement'] == True
assert result3['improvement_type'] == 'specific_paras'
assert 'para 2' in result3['which_paras']
assert 'para 4' in result3['which_paras']
print("✅ PASSED")

# Test Case 4: Not a paragraph improvement request
test4 = "Write an essay on contract law"
result4 = detect_specific_para_improvement(test4)
print(f"\nTest 4: '{test4}'")
print(f"Result: {result4}")
assert result4['is_para_improvement'] == False
print("✅ PASSED")

print("\n" + "=" * 80)
print("TESTING FEATURE 2: Google Search Grounding Detection")
print("=" * 80)

# Test Case 5: Essay request (should trigger Google Search)
test5 = "Write a 3000 word essay on AI regulation"
result5 = should_use_google_search_grounding(test5, rag_context="Some context")
print(f"\nTest 5: '{test5}'")
print(f"Result: {result5}")
assert result5['use_google_search'] == True
assert result5['enforce_oscola'] == True
print("✅ PASSED")

# Test Case 6: Recent case request
test6 = "What are the recent cases on data protection in 2025?"
result6 = should_use_google_search_grounding(test6, rag_context="Some context")
print(f"\nTest 6: '{test6}'")
print(f"Result: {result6}")
assert result6['use_google_search'] == True
print("✅ PASSED")

# Test Case 7: Insufficient RAG context
test7 = "What is vicarious liability?"
result7 = should_use_google_search_grounding(test7, rag_context="")  # Empty RAG
print(f"\nTest 7: '{test7}' (with empty RAG context)")
print(f"Result: {result7}")
assert result7['use_google_search'] == True
assert result7['reason'] == 'RAG context insufficient'
print("✅ PASSED")

# Test Case 8: RAG context sufficient + no special indicators
test8 = "What is the definition of consideration?"
long_rag = "This is sufficient RAG context. " * 50  # Make it > 500 chars
result8 = should_use_google_search_grounding(test8, rag_context=long_rag)
print(f"\nTest 8: '{test8}' (with sufficient RAG context)")
print(f"Result: {result8}")
assert result8['use_google_search'] == False  # Should not trigger
print("✅ PASSED")

print("\n" + "=" * 80)
print("ALL TESTS PASSED! ✅")
print("=" * 80)
print("\nThe new features are working correctly:")
print("1. ✅ Paragraph improvement detection (specific vs whole essay)")
print("2. ✅ Google Search grounding detection with OSCOLA enforcement")
print("\nYou can now use these features in the Streamlit app!")
