#!/usr/bin/env python3
"""
Test script to PROVE RAG is working
Run this after starting your Flask backend
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:5000"

def ask_question(question):
    """Ask a question and show RAG results"""
    print(f"\n{'='*60}")
    print(f"❓ ASKING: {question}")
    print('='*60)
    
    response = requests.post(
        f"{BASE_URL}/ask",
        json={"question": question},
        timeout=60
    )
    
    data = response.json()
    
    print(f"\n💬 ANSWER: {data['answer'][:200]}...")
    print(f"\n📊 STATS:")
    print(f"   - Total interactions: {data['total_interactions']}")
    print(f"   - Similar conversations found: {data['similar_count']}")
    
    if data.get('rag_details'):
        print(f"\n📚 RAG RETRIEVED:")
        for i, sim in enumerate(data['rag_details'], 1):
            print(f"   {i}. Q: {sim['question']}")
            print(f"      Similarity: {sim['similarity']:.3f}")
            print(f"      A: {sim['answer'][:80]}...\n")
    else:
        print("\n📚 RAG: No similar conversations found (this is the first time discussing this topic)")
    
    time.sleep(2)  # Wait between requests

def view_profile():
    """View student profile"""
    print(f"\n{'='*60}")
    print("👤 STUDENT PROFILE")
    print('='*60)
    
    response = requests.get(f"{BASE_URL}/student-profile")
    profile = response.json()
    
    print(json.dumps(profile, indent=2))

def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║          RAG PROOF TEST - AI TUTOR SYSTEM               ║
╚══════════════════════════════════════════════════════════╝

This script will prove RAG is working by:
1. Asking similar questions
2. Showing when RAG finds past conversations
3. Displaying similarity scores
""")

    # Test sequence
    tests = [
        "What is Python?",
        "Explain Python programming to me",  # Should match #1
        "What is machine learning?",
        "Tell me about Python syntax",  # Should match #1 and #2
        "More about machine learning algorithms",  # Should match #3
        "Hi, my name is Pranav",
        "What's my name?",  # Should match #6
    ]
    
    for question in tests:
        input("\nPress ENTER to ask next question...")
        ask_question(question)
    
    input("\nPress ENTER to view final profile...")
    view_profile()
    
    print("""
╔══════════════════════════════════════════════════════════╗
║                    TEST COMPLETE!                        ║
╚══════════════════════════════════════════════════════════╝

✅ If you saw:
   - Similarity scores > 0
   - Previous questions retrieved
   - Topics tracked in profile

Then RAG IS WORKING! 🎉

The bot remembers context through:
1. TF-IDF vectorization of all past Q&As
2. Cosine similarity matching
3. Retrieving top 3 similar conversations
4. Including them in AI prompt context
""")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure Flask backend is running on port 5000!")