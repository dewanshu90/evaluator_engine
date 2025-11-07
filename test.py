"""
Test Script for Student Responses Evaluation
Cost-optimized with single API call per answer
Default: GPT-3.5-Turbo for best cost/quality balance
"""

import json
from pathlib import Path
from evaluator import OptimizedAnswerEvaluator
from datetime import datetime


def test_with_optimized_evaluator(student_name=None):
    """Test student responses with optimized (cheap) evaluator
    
    Args:
        student_name: Optional. Specify student number (e.g., 1, "1", "Student_1", or "Student_1.json")
                     If None, evaluates all students
    """
    
    print("\n" + "="*80)
    print("  🚀 OPTIMIZED EVALUATION - LOW COST, HIGH SPEED")
    print("="*80 + "\n")
    
    # Load questions
    print("📚 Loading questions...")
    questions_file = r"c:\UpliftKidzAgent\questions\literacy_enriched.json"
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    questions_dict = {q['question_id']: q for q in questions}
    print(f"✅ Loaded {len(questions)} questions\n")
    
    # Text field questions
    text_field_question_ids = [
        'L8A', 'L8B', 'L8C', 'L8D',
        'L9A',
        'L11A', 'L11B', 'L11C', 'L11D'
    ]
    
    # Initialize evaluator with GPT-5-mini (cheapest!)
    print("🤖 Initializing OPTIMIZED evaluator...")
    print("   • Model: GPT-5-mini (latest, cheapest model)")
    print("   • Strategy: Single API call per answer")
    print("   • Expected cost: ~$0.05 for all 207 responses")
    print("   • Expected time: ~3.5 minutes\n")
    
    evaluator = OptimizedAnswerEvaluator(model_name="gpt-5-mini", temperature=0.2)
    
    # Load students
    student_dir = Path(r"c:\UpliftKidzAgent\student_answers")
    all_student_files = sorted(student_dir.glob("Student_*.json"))
    
    # Filter to specific student if requested
    if student_name:
        # Handle different input formats: 1, "1", "Student_1", "Student_1.json"
        if isinstance(student_name, int):
            student_name = f"Student_{student_name}.json"
        elif not student_name.endswith(".json"):
            if not student_name.startswith("Student_"):
                student_name = f"Student_{student_name}.json"
            else:
                student_name = f"{student_name}.json"
        
        # Find matching student file
        student_file_path = student_dir / student_name
        if student_file_path.exists():
            student_files = [student_file_path]
            print(f"👤 Testing single student: {student_name.replace('.json', '')}\n")
        else:
            print(f"❌ Student '{student_name}' not found!")
            print(f"Available students: {', '.join([f.stem for f in list(all_student_files)[:5]])}...\n")
            return
    else:
        student_files = all_student_files
        print(f"👥 Found {len(student_files)} students\n")
    
    print("="*80)
    print("  STARTING EVALUATION")
    print("="*80 + "\n")
    
    # Track progress
    total_evaluations = 0
    all_results = []
    
    start_time = datetime.now()
    
    # Process each student
    for student_file in student_files:
        with open(student_file, 'r', encoding='utf-8') as f:
            student_data = json.load(f)
        
        student_name = student_data['student_name']
        
        # Filter text responses
        text_responses = [
            ans for ans in student_data['answers']
            if ans['question_id'] in text_field_question_ids
        ]
        
        if not text_responses:
            continue
        
        print(f"\n📝 {student_name}: Evaluating {len(text_responses)} responses...")
        
        student_results = []
        student_total_score = 0
        student_max_score = 0
        
        for answer in text_responses:
            question_id = answer['question_id']
            student_answer = answer['response']
            
            question_data = questions_dict.get(question_id)
            if not question_data:
                continue
            
            # Evaluate (SINGLE API CALL)
            try:
                result = evaluator.evaluate_answer(question_data, student_answer)
                
                student_total_score += result['final_score']
                student_max_score += result['max_score']
                
                student_results.append({
                    'question_id': question_id,
                    'evaluation': result
                })
                
                total_evaluations += 1
                
                # Show progress
                print(f"   ✅ {question_id}: {result['final_score']:.2f}/{result['max_score']} ({result['percentage']:.1f}%)")
                
            except Exception as e:
                print(f"   ❌ {question_id}: Error - {e}")
                continue
        
        # Save student results
        if student_results:
            percentage = (student_total_score / student_max_score * 100) if student_max_score > 0 else 0
            
            student_evaluation = {
                'student_name': student_name,
                'evaluation_date': datetime.now().isoformat(),
                'total_score': round(student_total_score, 2),
                'max_score': student_max_score,
                'percentage': round(percentage, 1),
                'results': student_results
            }
            
            all_results.append(student_evaluation)
            
            print(f"   📊 Total: {student_total_score:.2f}/{student_max_score} ({percentage:.1f}%)")
    
    # Save all results
    output_dir = Path(r"c:\UpliftKidzAgent\evaluation_results_optimized")
    output_dir.mkdir(exist_ok=True)
    
    # Save individual student files
    for student_result in all_results:
        student_name = student_result['student_name']
        output_file = output_dir / f"{student_name}_evaluation.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(student_result, f, indent=2, ensure_ascii=False)
    
    # Generate summary report
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*80)
    print("  📊 EVALUATION COMPLETE!")
    print("="*80 + "\n")
    
    print(f"✅ Evaluated {total_evaluations} responses from {len(all_results)} students")
    print(f"⏱️  Total time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"💰 Estimated cost: ${(total_evaluations * 0.001):.2f}")
    print(f"📁 Results saved to: {output_dir}\n")
    
    # Generate text report
    report_file = output_dir / "evaluation_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("  OPTIMIZED STUDENT EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: GPT-3.5-Turbo (Optimized)\n")
        f.write(f"Total Students: {len(all_results)}\n")
        f.write(f"Total Evaluations: {total_evaluations}\n")
        f.write(f"Duration: {duration/60:.1f} minutes\n")
        f.write(f"Estimated Cost: ${(total_evaluations * 0.001):.2f}\n\n")
        
        f.write("="*80 + "\n")
        f.write("  STUDENT PERFORMANCE SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        for student_result in sorted(all_results, key=lambda x: x['percentage'], reverse=True):
            f.write(f"\n{student_result['student_name']}:\n")
            f.write(f"  Score: {student_result['total_score']:.2f}/{student_result['max_score']}\n")
            f.write(f"  Percentage: {student_result['percentage']:.1f}%\n")
    
    print(f"📄 Summary report: {report_file}\n")
    print("="*80 + "\n")


if __name__ == "__main__":
    print("\n💡 TIP: This uses GPT-5-mini by default (cheapest and most efficient!)")
    print("💡 USAGE:")
    print("   - All students:    python test.py")
    print("   - Single student:  python test.py 1  (or any number 1-23)\n")
    
    import sys
    
    # Check if student number is provided
    student_to_test = None
    if len(sys.argv) > 1:
        try:
            student_to_test = int(sys.argv[1])
        except ValueError:
            student_to_test = sys.argv[1]  # Could be "Student_1" format
    
    try:
        test_with_optimized_evaluator(student_to_test)
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user\n")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
