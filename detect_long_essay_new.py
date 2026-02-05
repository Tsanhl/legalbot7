def detect_long_essay(message: str) -> dict:
    """
    Detect if user is requesting a long essay that should be broken into parts.
    
    Returns:
        dict with:
        - 'is_long_essay': bool - whether the essay should be broken into parts
        - 'requested_words': int - the word count requested (0 if not detected)
        - 'suggested_parts': int - number of parts to break into
        - 'words_per_part': int - suggested words per part
        - 'suggestion_message': str - message to show to user
        - 'is_user_draft': bool - whether user is submitting their own essay for improvement
        - 'await_user_choice': bool - whether to wait for user to choose approach before proceeding
    """
    import re
    msg_lower = message.lower()
    
    result = {
        'is_long_essay': False,
        'requested_words': 0,
        'suggested_parts': 0,
        'words_per_part': 0,
        'suggestion_message': '',
        'is_user_draft': False,
        'await_user_choice': False
    }
    
    # Extract word count from message
    word_count_match = re.search(r'(\d{3,5})\s*words?', msg_lower)
    if not word_count_match:
        return result
    
    requested_words = int(word_count_match.group(1))
    result['requested_words'] = requested_words
    
    # Check if it's above the threshold
    LONG_ESSAY_THRESHOLD = 5000
    if requested_words >= LONG_ESSAY_THRESHOLD:
        result['is_long_essay'] = True
        result['await_user_choice'] = True  # Wait for user to choose approach before showing "Thinking..."
        
        # Calculate suggested parts (aim for 2500-3500 words per part)
        words_per_part = 3000  # Target per part
        suggested_parts = max(2, round(requested_words / words_per_part))
        actual_words_per_part = requested_words // suggested_parts
        
        result['suggested_parts'] = suggested_parts
        result['words_per_part'] = actual_words_per_part
        
        # Detect if user is submitting their own draft for improvement
        user_draft_indicators = [
            'here is my essay', 'here is my draft', 'my essay:', 'my draft:',
            'i wrote this', 'i have written', 'my attempt', 'my version',
            'please check my', 'please review my', 'please improve my',
            'can you check', 'can you review', 'is this correct',
            'here\'s what i have', 'this is what i wrote', 'my answer is',
            'below is my', 'following is my', 'attached is my',
            'improve my essay', 'improve this essay', 'better version of this'
        ]
        
        result['is_user_draft'] = any(indicator in msg_lower for indicator in user_draft_indicators)
        
        # VERSION 1: User asking AI to generate a new blank essay
        if not result['is_user_draft']:
            result['suggestion_message'] = f"""📝 **Long Essay Detected ({requested_words:,} words)**

For best results with essays over 5,000 words, I recommend breaking this into **{suggested_parts} parts**:

**Suggested Approach:**
1. Ask for Part 1 (~{actual_words_per_part:,} words) - Introduction + first {suggested_parts//2 + 1} sections
2. Then ask "Continue with Part 2" for the next sections
{f"3. Then ask 'Continue with Part 3' for the remaining sections" if suggested_parts >= 3 else ""}
{f"4. Finally ask 'Continue with Part 4 - Conclusion'" if suggested_parts >= 4 else ""}

**Why break into parts?**
- The AI has memory and will continue coherently
- Each part will hit its word count accurately
- No repetitive content across parts
- Better quality and depth in each section

**Or proceed now** and I'll write as much as I can (~3,500-4,000 words), then you can ask me to "Continue" for the rest."""
        
        # VERSION 2: User submitting their own essay for improvement
        else:
            result['suggestion_message'] = f"""📝 **Long Essay Improvement Detected ({requested_words:,} words)**

For best results with essays over 5,000 words, I recommend breaking this into **{suggested_parts} parts**:

**The parts will be according to your essay structure.**
**Total output will be {requested_words:,} words as you requested.**

**Why break into parts?**
- The AI has memory and will continue coherently
- Each part will hit its word count accurately
- Better quality and depth in each section
- Your essay structure will be preserved

**Or proceed now** and I'll improve as much as I can (~3,500-4,000 words), then you can ask me to "Continue" for the rest."""
    
    return result
