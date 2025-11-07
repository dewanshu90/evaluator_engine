"""
OPTIMIZED AI Answer Evaluator - 75% Cost Reduction
Single API call instead of 4 parallel calls
Uses GPT-3.5-Turbo by default (10x cheaper than GPT-4)
"""

from typing import TypedDict, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import json
import os
from dotenv import load_dotenv

load_dotenv()


class EvaluationState(TypedDict):
    """State for evaluation"""
    question_id: str
    question_text: str
    correct_answer: str
    student_answer: str
    context: str
    difficulty: str
    max_score: int
    
    intent_analysis: Dict[str, Any]
    vocabulary_analysis: Dict[str, Any]
    spelling_analysis: Dict[str, Any]
    grammar_analysis: Dict[str, Any]
    
    final_score: float
    partial_scores: Dict[str, float]
    remarks: str
    suggestions: str
    evaluation_summary: Dict[str, Any]


class OptimizedAnswerEvaluator:
    """
    Optimized evaluator using SINGLE API call
    Cost: ~99% less than original (4 calls → 1 call)
    Speed: ~4x faster
    Model: GPT-5-mini by default (latest, cheapest, most efficient)
    """
    
    def __init__(self, model_name: str = "gpt-5-mini", temperature: float = 0.2):
        """
        Initialize evaluator
        
        Args:
            model_name: "gpt-5-mini" (default, cheapest), "gpt-4o-mini", "gpt-3.5-turbo", or "gpt-4"
            temperature: Lower = more consistent (Note: gpt-5-mini only supports default temperature=1.0)
        """
        # GPT-5-mini only supports default temperature (1.0)
        if "gpt-5" in model_name:
            temperature = 1.0
        
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.model_name = model_name
    
    def evaluate_answer(self, question_data: Dict[str, Any], student_answer: str) -> Dict[str, Any]:
        """
        Evaluate student answer with SINGLE API call
        
        Args:
            question_data: Question info from JSON
            student_answer: Student's text response
            
        Returns:
            Complete evaluation results
        """
        
        # Extract data
        question_id = question_data.get("question_id", "")
        question_text = question_data.get("question_text", "")
        correct_answer = question_data.get("correct_answer", {}).get("option_text", "")
        context = question_data.get("context", "")
        difficulty = question_data.get("difficulty", "Medium")
        max_score = question_data.get("max_score", 1)
        
        # Single comprehensive prompt
        prompt = f"""Evaluate this student's answer across 4 dimensions.

QUESTION: {question_text}
CORRECT ANSWER: {correct_answer}
STUDENT'S ANSWER: {student_answer}
DIFFICULTY: {difficulty}
CONTEXT: {context}

Evaluate on these 4 dimensions:

1. INTENT (Understanding): Did they grasp the main concept? (0-100)
2. VOCABULARY (Word Choice): Appropriate words for age? (0-100)
3. SPELLING (Accuracy): Consider phonetic attempts like "becaus"→"because" (0-100)
4. GRAMMAR (Structure): Sentence structure, tense, agreement (0-100)

Return ONLY valid JSON (no markdown, no explanation):
{{
    "intent": {{
        "score": 0-100,
        "understood": true/false,
        "concepts_right": ["list"],
        "concepts_missed": ["list"],
        "note": "brief"
    }},
    "vocabulary": {{
        "score": 0-100,
        "good_words": ["list"],
        "improve": ["suggestions"],
        "note": "brief"
    }},
    "spelling": {{
        "score": 0-100,
        "errors": [{{"word": "wrong", "correct": "right", "type": "phonetic/typo"}}],
        "phonetic_tries": ["list"],
        "note": "brief"
    }},
    "grammar": {{
        "score": 0-100,
        "errors": [{{"type": "error", "fix": "correction"}}],
        "strengths": ["list"],
        "note": "brief"
    }}
}}"""

        # SINGLE API CALL HERE (instead of 4)
        messages = [
            SystemMessage(content="You are an expert child literacy assessor. Return only valid JSON."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Parse response
        try:
            # Clean up response (remove markdown code blocks if present)
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            
            result = json.loads(content)
            
            # Extract scores
            intent_score = result.get("intent", {}).get("score", 0)
            vocab_score = result.get("vocabulary", {}).get("score", 0)
            spelling_score = result.get("spelling", {}).get("score", 0)
            grammar_score = result.get("grammar", {}).get("score", 0)
            
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback if JSON parsing fails
            result = {
                "intent": {"score": 50, "understood": True, "concepts_right": [], "concepts_missed": [], "note": "Parse error"},
                "vocabulary": {"score": 50, "good_words": [], "improve": [], "note": "Parse error"},
                "spelling": {"score": 50, "errors": [], "phonetic_tries": [], "note": "Parse error"},
                "grammar": {"score": 50, "errors": [], "strengths": [], "note": "Parse error"}
            }
            intent_score = vocab_score = spelling_score = grammar_score = 50
        
        # Calculate weights based on difficulty
        weights = self._get_weights(difficulty, context)
        
        # Calculate weighted average (0-100 scale)
        weighted_avg = (
            (intent_score * weights["intent"]) +
            (vocab_score * weights["vocabulary"]) +
            (spelling_score * weights["spelling"]) +
            (grammar_score * weights["grammar"])
        ) / 100
        
        # Scale to max_score
        final_score = (weighted_avg / 100) * max_score
        
        # Partial scores
        partial_scores = {
            "intent": round((intent_score / 100) * (weights["intent"] / 100) * max_score, 2),
            "vocabulary": round((vocab_score / 100) * (weights["vocabulary"] / 100) * max_score, 2),
            "spelling": round((spelling_score / 100) * (weights["spelling"] / 100) * max_score, 2),
            "grammar": round((grammar_score / 100) * (weights["grammar"] / 100) * max_score, 2)
        }
        
        # Generate feedback
        percentage = (final_score / max_score) * 100
        remarks = self._generate_remarks(percentage, result)
        suggestions = self._generate_suggestions(result)
        
        # Build evaluation summary
        return {
            "question_id": question_id,
            "final_score": round(final_score, 2),
            "max_score": max_score,
            "percentage": round(percentage, 1),
            "partial_scores": partial_scores,
            "intent_analysis": {
                "intent_match_score": intent_score,
                "understood_concept": result["intent"].get("understood", True),
                "key_concepts_identified": result["intent"].get("concepts_right", []),
                "key_concepts_missed": result["intent"].get("concepts_missed", []),
                "reasoning": result["intent"].get("note", "")
            },
            "vocabulary_analysis": {
                "vocabulary_score": vocab_score,
                "appropriate_words": result["vocabulary"].get("good_words", []),
                "suggested_improvements": result["vocabulary"].get("improve", []),
                "reasoning": result["vocabulary"].get("note", "")
            },
            "spelling_analysis": {
                "spelling_score": spelling_score,
                "misspelled_words": result["spelling"].get("errors", []),
                "phonetic_attempts": result["spelling"].get("phonetic_tries", []),
                "reasoning": result["spelling"].get("note", "")
            },
            "grammar_analysis": {
                "grammar_score": grammar_score,
                "errors": result["grammar"].get("errors", []),
                "strengths": result["grammar"].get("strengths", []),
                "reasoning": result["grammar"].get("note", "")
            },
            "remarks": remarks,
            "suggestions": suggestions
        }
    
    def _get_weights(self, difficulty: str, context: str) -> Dict[str, float]:
        """Determine scoring weights"""
        difficulty = difficulty.lower()
        context = context.lower()
        
        # Default weights
        weights = {
            "intent": 40,
            "vocabulary": 25,
            "spelling": 15,
            "grammar": 20
        }
        
        # Adjust by difficulty
        if difficulty == "easy":
            weights.update({"spelling": 20, "grammar": 25, "vocabulary": 20, "intent": 35})
        elif difficulty == "hard":
            weights.update({"intent": 50, "vocabulary": 25, "spelling": 10, "grammar": 15})
        
        # Adjust by context
        if "comprehension" in context or "reading" in context:
            weights.update({"intent": 50, "vocabulary": 25, "spelling": 10, "grammar": 15})
        elif "spelling" in context:
            weights.update({"spelling": 35, "intent": 30})
        elif "grammar" in context:
            weights.update({"grammar": 35, "intent": 30})
        
        return weights
    
    def _generate_remarks(self, percentage: float, result: Dict) -> str:
        """Generate encouraging remarks"""
        if percentage >= 90:
            tone = "Excellent work!"
        elif percentage >= 75:
            tone = "Great job!"
        elif percentage >= 60:
            tone = "Good effort!"
        elif percentage >= 40:
            tone = "Nice try!"
        else:
            tone = "Keep practicing!"
        
        remarks = [tone]
        
        if result["intent"].get("understood"):
            remarks.append("You understood the main idea.")
        
        if result["spelling"].get("phonetic_tries"):
            remarks.append("Good phonetic spelling attempts!")
        
        return " ".join(remarks)
    
    def _generate_suggestions(self, result: Dict) -> str:
        """Generate constructive suggestions"""
        suggestions = []
        
        # Intent
        missed = result["intent"].get("concepts_missed", [])
        if missed:
            suggestions.append(f"Include: {', '.join(missed[:2])}")
        
        # Vocabulary
        improve = result["vocabulary"].get("improve", [])
        if improve:
            suggestions.append(f"Word tip: {improve[0]}")
        
        # Spelling
        errors = result["spelling"].get("errors", [])
        if errors:
            err = errors[0]
            suggestions.append(f"Spelling: '{err.get('word')}' → '{err.get('correct')}'")
        
        # Grammar
        grammar_errors = result["grammar"].get("errors", [])
        if grammar_errors:
            suggestions.append(f"Grammar: {grammar_errors[0].get('fix', 'Check structure')}")
        
        return " | ".join(suggestions[:3]) if suggestions else "Keep up the good work!"


# Convenience function
def quick_evaluate(question_data: Dict[str, Any], student_answer: str, 
                   use_gpt4: bool = False) -> Dict[str, Any]:
    """
    Quick evaluation function
    
    Args:
        question_data: Question info
        student_answer: Student's answer
        use_gpt4: True for GPT-4 (expensive), False for GPT-5-mini (cheapest, default)
    
    Returns:
        Evaluation results
    """
    model = "gpt-4" if use_gpt4 else "gpt-5-mini"
    evaluator = OptimizedAnswerEvaluator(model_name=model)
    return evaluator.evaluate_answer(question_data, student_answer)
