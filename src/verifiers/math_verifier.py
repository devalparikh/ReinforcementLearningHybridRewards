"""
Mathematical verification for RLVR.

This module implements a verifier that checks mathematical correctness
and reasoning in model outputs. It supports various mathematical operations
and can verify step-by-step solutions.
"""

import re
import time
import traceback
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import math
from decimal import Decimal, getcontext
import sympy
from sympy import symbols, solve, simplify, expand, factor, diff, integrate
from sympy.parsing.sympy_parser import parse_expr

from .base_verifier import BaseVerifier, VerificationOutput, VerificationResult


class MathVerifier(BaseVerifier):
    """
    Verifier that checks mathematical correctness and reasoning.
    
    This verifier can handle various mathematical operations including
    arithmetic, algebra, calculus, and step-by-step problem solving.
    """
    
    def __init__(
        self,
        name: str = "math_verifier",
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the math verifier.
        
        Args:
            name: Name of the verifier
            config: Configuration parameters
            logger: Logger instance
        """
        default_config = {
            "precision": 1e-10,  # Numerical precision for comparisons
            "max_steps": 50,  # Maximum number of solution steps to check
            "allowed_operations": [
                "arithmetic", "algebra", "calculus", "geometry", "statistics"
            ],
            "check_steps": True,  # Whether to verify intermediate steps
            "symbolic_computation": True,  # Enable symbolic computation
            "numerical_computation": True,  # Enable numerical computation
            "timeout": 30,  # Timeout for complex computations
            "decimal_precision": 28  # Decimal precision for calculations
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(
            name=name,
            description="Checks mathematical correctness and reasoning",
            config=default_config,
            logger=logger
        )
        
        # Set decimal precision
        getcontext().prec = self.config["decimal_precision"]
    
    def _validate_config(self) -> None:
        """Validate the verifier configuration."""
        if self.config["precision"] <= 0:
            raise ValueError("Precision must be positive")
        
        if self.config["max_steps"] <= 0:
            raise ValueError("Max steps must be positive")
        
        if self.config["timeout"] <= 0:
            raise ValueError("Timeout must be positive")
    
    def verify(
        self,
        instruction: str,
        model_output: str,
        expected_output: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> VerificationOutput:
        """
        Verify mathematical correctness and reasoning.
        
        Args:
            instruction: The instruction given to the model
            model_output: The mathematical solution from the model
            expected_output: Expected output (if available)
            context: Additional context for verification
            
        Returns:
            VerificationOutput: Structured verification result
        """
        start_time = time.time()
        
        try:
            # Extract mathematical expressions and solutions
            math_content = self._extract_math_content(model_output)
            if not math_content:
                return VerificationOutput(
                    result=VerificationResult.INCORRECT,
                    score=0.0,
                    details={"error": "No mathematical content found"},
                    error_message="No mathematical content found"
                )
            
            # Parse the problem from instruction
            problem_info = self._parse_problem(instruction, context)
            
            # Verify the solution
            verification_result = self._verify_solution(
                math_content, problem_info, expected_output, context
            )
            
            execution_time = time.time() - start_time
            
            return VerificationOutput(
                result=verification_result["result"],
                score=verification_result["score"],
                details={
                    "math_content": math_content,
                    "problem_info": problem_info,
                    "verification_details": verification_result["details"],
                    "execution_time": execution_time
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Mathematical verification failed: {e}")
            return VerificationOutput(
                result=VerificationResult.ERROR,
                score=0.0,
                details={"error": str(e), "traceback": traceback.format_exc()},
                error_message=str(e),
                execution_time=execution_time
            )
    
    def _extract_math_content(self, model_output: str) -> Dict[str, Any]:
        """
        Extract mathematical content from model output.
        
        Args:
            model_output: The output from the model
            
        Returns:
            Dictionary containing extracted mathematical content
        """
        content = {
            "final_answer": None,
            "steps": [],
            "expressions": [],
            "equations": [],
            "variables": set()
        }
        
        lines = model_output.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for final answer patterns
            if re.search(r'answer[:\s]*([^.\n]+)', line, re.IGNORECASE):
                match = re.search(r'answer[:\s]*([^.\n]+)', line, re.IGNORECASE)
                content["final_answer"] = match.group(1).strip()
            
            # Look for step-by-step solutions
            if re.match(r'^\d+[\.\)]?\s*', line):
                step_content = re.sub(r'^\d+[\.\)]?\s*', '', line)
                content["steps"].append(step_content)
            
            # Extract mathematical expressions
            expressions = self._extract_expressions(line)
            content["expressions"].extend(expressions)
            
            # Extract equations
            equations = self._extract_equations(line)
            content["equations"].extend(equations)
            
            # Extract variables
            variables = self._extract_variables(line)
            content["variables"].update(variables)
        
        # If no final answer found, try to extract from the last line
        if not content["final_answer"] and lines:
            last_line = lines[-1].strip()
            if self._is_numeric_expression(last_line):
                content["final_answer"] = last_line
        
        return content
    
    def _extract_expressions(self, text: str) -> List[str]:
        """Extract mathematical expressions from text."""
        # Pattern for mathematical expressions
        patterns = [
            r'[+-]?\d*\.?\d+',  # Numbers
            r'[a-zA-Z_][a-zA-Z0-9_]*',  # Variables
            r'[+\-*/^()=<>≤≥≠]',  # Operators
            r'sin|cos|tan|log|ln|exp|sqrt',  # Functions
        ]
        
        expressions = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            expressions.extend(matches)
        
        return expressions
    
    def _extract_equations(self, text: str) -> List[str]:
        """Extract equations from text."""
        # Look for patterns like "x = y" or "f(x) = expression"
        equation_pattern = r'([^=]+)\s*=\s*([^=]+)'
        matches = re.findall(equation_pattern, text)
        
        equations = []
        for left, right in matches:
            equation = f"{left.strip()} = {right.strip()}"
            equations.append(equation)
        
        return equations
    
    def _extract_variables(self, text: str) -> set:
        """Extract variable names from text."""
        # Pattern for variable names
        variable_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        matches = re.findall(variable_pattern, text)
        
        # Filter out common words and functions
        exclude_words = {
            'the', 'and', 'or', 'if', 'then', 'else', 'for', 'while',
            'sin', 'cos', 'tan', 'log', 'ln', 'exp', 'sqrt', 'abs',
            'min', 'max', 'sum', 'product', 'integral', 'derivative'
        }
        
        variables = set()
        for match in matches:
            if match.lower() not in exclude_words and len(match) > 1:
                variables.add(match)
        
        return variables
    
    def _is_numeric_expression(self, text: str) -> bool:
        """Check if text contains a numeric expression."""
        # Remove common words and check if remaining is mathematical
        cleaned = re.sub(r'\b(answer|result|solution|equals?|is)\b', '', text, flags=re.IGNORECASE)
        cleaned = cleaned.strip()
        
        # Check if it looks like a mathematical expression
        math_patterns = [
            r'^[+\-]?\d*\.?\d+$',  # Single number
            r'^[+\-]?\d*\.?\d+\s*[+\-*/^]\s*[+\-]?\d*\.?\d+$',  # Simple arithmetic
            r'^[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*[+\-]?\d*\.?\d+$',  # Variable assignment
        ]
        
        return any(re.match(pattern, cleaned) for pattern in math_patterns)
    
    def _parse_problem(self, instruction: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Parse the mathematical problem from the instruction.
        
        Args:
            instruction: The instruction containing the problem
            context: Additional context
            
        Returns:
            Dictionary containing problem information
        """
        problem_info = {
            "type": "unknown",
            "variables": set(),
            "equations": [],
            "constraints": [],
            "expected_format": "numeric"
        }
        
        # Determine problem type
        instruction_lower = instruction.lower()
        
        if any(word in instruction_lower for word in ['solve', 'equation', 'find x']):
            problem_info["type"] = "equation_solving"
        elif any(word in instruction_lower for word in ['calculate', 'compute', 'evaluate']):
            problem_info["type"] = "computation"
        elif any(word in instruction_lower for word in ['integrate', 'integral']):
            problem_info["type"] = "integration"
        elif any(word in instruction_lower for word in ['differentiate', 'derivative']):
            problem_info["type"] = "differentiation"
        elif any(word in instruction_lower for word in ['factor', 'expand', 'simplify']):
            problem_info["type"] = "algebraic_manipulation"
        
        # Extract variables and equations from instruction
        problem_info["variables"] = self._extract_variables(instruction)
        problem_info["equations"] = self._extract_equations(instruction)
        
        # Check for expected format
        if any(word in instruction_lower for word in ['decimal', 'fraction', 'percentage']):
            problem_info["expected_format"] = "decimal"
        elif any(word in instruction_lower for word in ['fraction', 'rational']):
            problem_info["expected_format"] = "fraction"
        
        return problem_info
    
    def _verify_solution(
        self,
        math_content: Dict[str, Any],
        problem_info: Dict[str, Any],
        expected_output: Optional[str],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Verify the mathematical solution.
        
        Args:
            math_content: Extracted mathematical content
            problem_info: Problem information
            expected_output: Expected output
            context: Additional context
            
        Returns:
            Verification result
        """
        score = 0.0
        details = {}
        
        # Check if final answer is provided
        if math_content["final_answer"]:
            final_answer = math_content["final_answer"]
            
            # Try to evaluate the final answer
            try:
                if self.config["symbolic_computation"]:
                    # Try symbolic evaluation
                    result = self._evaluate_symbolic(final_answer)
                    if result is not None:
                        details["symbolic_result"] = str(result)
                        score += 0.3
                
                if self.config["numerical_computation"]:
                    # Try numerical evaluation
                    result = self._evaluate_numeric(final_answer)
                    if result is not None:
                        details["numerical_result"] = result
                        score += 0.3
                        
            except Exception as e:
                details["evaluation_error"] = str(e)
        
        # Check step-by-step reasoning
        if self.config["check_steps"] and math_content["steps"]:
            step_score = self._verify_steps(math_content["steps"], problem_info)
            score += step_score * 0.4
            details["step_verification"] = step_score
        
        # Compare with expected output if provided
        if expected_output:
            comparison_result = self._compare_with_expected(
                math_content["final_answer"], expected_output
            )
            score = max(score, comparison_result["score"])
            details["expected_comparison"] = comparison_result
        
        # Determine result
        if score >= 0.8:
            result = VerificationResult.CORRECT
        elif score >= 0.5:
            result = VerificationResult.PARTIAL
        else:
            result = VerificationResult.INCORRECT
        
        return {
            "result": result,
            "score": score,
            "details": details
        }
    
    def _evaluate_symbolic(self, expression: str) -> Optional[sympy.Expr]:
        """Evaluate expression symbolically."""
        try:
            # Clean the expression
            cleaned = re.sub(r'[^\w\s+\-*/^()=.,]', '', expression)
            
            # Try to parse and simplify
            expr = parse_expr(cleaned)
            simplified = simplify(expr)
            
            return simplified
        except Exception:
            return None
    
    def _evaluate_numeric(self, expression: str) -> Optional[float]:
        """Evaluate expression numerically."""
        try:
            # Clean the expression
            cleaned = re.sub(r'[^\w\s+\-*/^()=.,]', '', expression)
            
            # Replace mathematical functions
            cleaned = re.sub(r'sin', 'math.sin', cleaned)
            cleaned = re.sub(r'cos', 'math.cos', cleaned)
            cleaned = re.sub(r'tan', 'math.tan', cleaned)
            cleaned = re.sub(r'log', 'math.log', cleaned)
            cleaned = re.sub(r'ln', 'math.log', cleaned)
            cleaned = re.sub(r'exp', 'math.exp', cleaned)
            cleaned = re.sub(r'sqrt', 'math.sqrt', cleaned)
            
            # Evaluate safely
            result = eval(cleaned, {"__builtins__": {}}, {"math": math})
            
            return float(result)
        except Exception:
            return None
    
    def _verify_steps(self, steps: List[str], problem_info: Dict[str, Any]) -> float:
        """
        Verify step-by-step reasoning.
        
        Args:
            steps: List of solution steps
            problem_info: Problem information
            
        Returns:
            Score for step verification
        """
        if not steps:
            return 0.0
        
        step_scores = []
        
        for i, step in enumerate(steps):
            step_score = 0.0
            
            # Check if step contains mathematical content
            if self._extract_expressions(step):
                step_score += 0.2
            
            # Check if step is logically connected to previous steps
            if i > 0 and self._check_step_connection(steps[i-1], step):
                step_score += 0.3
            
            # Check if step makes mathematical sense
            if self._check_mathematical_sense(step):
                step_score += 0.5
            
            step_scores.append(step_score)
        
        return sum(step_scores) / len(step_scores) if step_scores else 0.0
    
    def _check_step_connection(self, prev_step: str, current_step: str) -> bool:
        """Check if current step is logically connected to previous step."""
        # Extract variables and expressions from both steps
        prev_vars = self._extract_variables(prev_step)
        curr_vars = self._extract_variables(current_step)
        
        # Check for variable continuity
        if prev_vars and curr_vars:
            common_vars = prev_vars.intersection(curr_vars)
            if common_vars:
                return True
        
        # Check for expression continuity
        prev_exprs = self._extract_expressions(prev_step)
        curr_exprs = self._extract_expressions(current_step)
        
        if prev_exprs and curr_exprs:
            # Check if some expressions are related
            for prev_expr in prev_exprs:
                for curr_expr in curr_exprs:
                    if prev_expr in curr_expr or curr_expr in prev_expr:
                        return True
        
        return False
    
    def _check_mathematical_sense(self, step: str) -> bool:
        """Check if a step makes mathematical sense."""
        # Check for common mathematical operations
        math_indicators = [
            r'[+\-*/^=]',  # Mathematical operators
            r'\d+',  # Numbers
            r'[a-zA-Z_][a-zA-Z0-9_]*',  # Variables
            r'sin|cos|tan|log|ln|exp|sqrt',  # Functions
        ]
        
        has_math_content = any(re.search(pattern, step) for pattern in math_indicators)
        
        # Check for logical connectors
        logical_connectors = ['therefore', 'thus', 'hence', 'so', 'because', 'since']
        has_logic = any(connector in step.lower() for connector in logical_connectors)
        
        return has_math_content or has_logic
    
    def _compare_with_expected(
        self,
        actual_answer: Optional[str],
        expected_output: str
    ) -> Dict[str, Any]:
        """
        Compare actual answer with expected output.
        
        Args:
            actual_answer: The actual answer from the model
            expected_output: The expected output
            
        Returns:
            Comparison result
        """
        if not actual_answer:
            return {
                "score": 0.0,
                "match": False,
                "details": "No actual answer provided"
            }
        
        # Try exact match
        if actual_answer.strip() == expected_output.strip():
            return {
                "score": 1.0,
                "match": True,
                "details": "Exact match"
            }
        
        # Try numerical comparison
        try:
            actual_num = self._evaluate_numeric(actual_answer)
            expected_num = self._evaluate_numeric(expected_output)
            
            if actual_num is not None and expected_num is not None:
                if abs(actual_num - expected_num) < self.config["precision"]:
                    return {
                        "score": 1.0,
                        "match": True,
                        "details": "Numerical match"
                    }
        except Exception:
            pass
        
        # Try symbolic comparison
        try:
            actual_sym = self._evaluate_symbolic(actual_answer)
            expected_sym = self._evaluate_symbolic(expected_output)
            
            if actual_sym is not None and expected_sym is not None:
                if simplify(actual_sym - expected_sym) == 0:
                    return {
                        "score": 1.0,
                        "match": True,
                        "details": "Symbolic match"
                    }
        except Exception:
            pass
        
        return {
            "score": 0.0,
            "match": False,
            "details": "No match found"
        } 