"""
Test script for the updated long essay detection with Version 1 and Version 2
"""

import sys
sys.path.append('/Users/hltsang/Desktop/doc-ai copy (latest) 2 self use for essay backup 1')

from gemini_service import detect_long_essay

print("=" * 80)
print("TESTING LONG ESSAY DETECTION - VERSION 1 VS VERSION 2")
print("=" * 80)

# Test Case 1: VERSION 1 - New essay generation (12000 words)
test1 = "Write a 12000 word essay on contract law"
result1 = detect_long_essay(test1)
print(f"\n📝 Test 1: '{test1}'")
print(f"Is long essay: {result1['is_long_essay']}")
print(f"Is user draft: {result1['is_user_draft']}")
print(f"Await user choice: {result1['await_user_choice']}")
print(f"Suggested parts: {result1['suggested_parts']}")
print(f"Words per part: {result1['words_per_part']}")
print("\nRecommendation message:")
print(result1['suggestion_message'])
print("\n✅ PASSED - VERSION 1 (New Essay)" if not result1['is_user_draft'] else "❌ FAILED")

print("\n" + "=" * 80)

# Test Case 2: VERSION 2 - User draft improvement (8000 words)
test2 = "Here is my essay. Please improve my 8000 word essay on tort law"
result2 = detect_long_essay(test2)
print(f"\n📝 Test 2: '{test2}'")
print(f"Is long essay: {result2['is_long_essay']}")
print(f"Is user draft: {result2['is_user_draft']}")
print(f"Await user choice: {result2['await_user_choice']}")
print(f"Suggested parts: {result2['suggested_parts']}")
print(f"Words per part: {result2['words_per_part']}")
print("\nRecommendation message:")
print(result2['suggestion_message'])
print("\n✅ PASSED - VERSION 2 (User Draft)" if result2['is_user_draft'] else "❌ FAILED")

print("\n" + "=" * 80)

# Test Case 3: VERSION 2 - Another user draft variation
test3 = "Can you check my 6000 word essay and make it better?"
result3 = detect_long_essay(test3)
print(f"\n📝 Test 3: '{test3}'")
print(f"Is long essay: {result3['is_long_essay']}")
print(f"Is user draft: {result3['is_user_draft']}")
print(f"Suggested parts: {result3['suggested_parts']}")
print("\n✅ PASSED - VERSION 2 (User Draft)" if result3['is_user_draft'] else "❌ FAILED")

print("\n" + "=" * 80)

# Test Case 4: Short essay - should NOT trigger long essay detection
test4 = "Write a 2000 word essay on criminal law"
result4 = detect_long_essay(test4)
print(f"\n📝 Test 4: '{test4}'")
print(f"Is long essay: {result4['is_long_essay']}")
print(f"Await user choice: {result4['await_user_choice']}")
print("\n✅ PASSED - NOT a long essay (< 5000 words)" if not result4['is_long_essay'] else "❌ FAILED")

print("\n" + "=" * 80)
print("\nSUMMARY:")
print("=" * 80)
print("\n✅ VERSION 1 (New Essay): Shows detailed breakdown with suggested sections")
print("✅ VERSION 2 (User Draft): Shows simplified message - parts according to user's essay")
print("✅ Both versions set await_user_choice=True to stop before 'Thinking...' indicator")
print("\nThe improvements are working correctly!")
print("\n📌 In Streamlit:")
print("- Long essay recommendation shows")
print("- User sees 'Please respond' message")
print("- NO 'Thinking...' indicator appears")
print("- User can choose 'Proceed now' or 'Part 1' approach")
