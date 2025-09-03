"""
Logical reasoning verification for RLVR.

This module implements a verifier that checks logical consistency and
reasoning in model outputs. It can verify logical arguments, identify
fallacies, and check for coherent reasoning patterns.
"""

import re
import time
import traceback
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from collections import defaultdict

from .base_verifier import BaseVerifier, VerificationOutput, VerificationResult


class LogicVerifier(BaseVerifier):
    """
    Verifier that checks logical consistency and reasoning.
    
    This verifier can analyze logical arguments, identify fallacies,
    check for coherent reasoning patterns, and verify logical consistency.
    """
    
    def __init__(
        self,
        name: str = "logic_verifier",
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the logic verifier.
        
        Args:
            name: Name of the verifier
            config: Configuration parameters
            logger: Logger instance
        """
        default_config = {
            "check_fallacies": True,  # Check for logical fallacies
            "check_consistency": True,  # Check for logical consistency
            "check_coherence": True,  # Check for coherent reasoning
            "check_structure": True,  # Check argument structure
            "fallacy_patterns": {
                "ad_hominem": [
                    r"you're\s+(stupid|ignorant|biased)",
                    r"because\s+you\s+are\s+\w+",
                    r"attack.*person.*instead.*argument"
                ],
                "straw_man": [
                    r"misrepresent.*argument",
                    r"exaggerate.*position",
                    r"attack.*weak.*version"
                ],
                "false_dichotomy": [
                    r"either.*or.*nothing.*else",
                    r"only.*two.*options",
                    r"black.*white.*thinking"
                ],
                "appeal_to_authority": [
                    r"because.*expert.*said",
                    r"authority.*figure.*believes",
                    r"trust.*authority.*blindly"
                ],
                "circular_reasoning": [
                    r"because.*it.*is",
                    r"true.*because.*true",
                    r"assume.*conclusion"
                ]
            },
            "logical_connectors": [
                "therefore", "thus", "hence", "so", "because", "since",
                "if", "then", "implies", "leads to", "results in",
                "consequently", "as a result", "for this reason"
            ],
            "contradiction_indicators": [
                "but", "however", "nevertheless", "on the other hand",
                "in contrast", "despite", "although", "while"
            ]
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(
            name=name,
            description="Checks logical consistency and reasoning",
            config=default_config,
            logger=logger
        )
    
    def _validate_config(self) -> None:
        """Validate the verifier configuration."""
        if not isinstance(self.config["fallacy_patterns"], dict):
            raise ValueError("Fallacy patterns must be a dictionary")
        
        if not isinstance(self.config["logical_connectors"], list):
            raise ValueError("Logical connectors must be a list")
    
    def verify(
        self,
        instruction: str,
        model_output: str,
        expected_output: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> VerificationOutput:
        """
        Verify logical consistency and reasoning.
        
        Args:
            instruction: The instruction given to the model
            model_output: The logical argument from the model
            expected_output: Expected output (if available)
            context: Additional context for verification
            
        Returns:
            VerificationOutput: Structured verification result
        """
        start_time = time.time()
        
        try:
            # Extract logical content
            logic_content = self._extract_logic_content(model_output)
            if not logic_content:
                return VerificationOutput(
                    result=VerificationResult.INCORRECT,
                    score=0.0,
                    details={"error": "No logical content found"},
                    error_message="No logical content found"
                )
            
            # Parse the logical problem
            problem_info = self._parse_logic_problem(instruction, context)
            
            # Verify the logical reasoning
            verification_result = self._verify_logic(
                logic_content, problem_info, expected_output, context
            )
            
            execution_time = time.time() - start_time
            
            return VerificationOutput(
                result=verification_result["result"],
                score=verification_result["score"],
                details={
                    "logic_content": logic_content,
                    "problem_info": problem_info,
                    "verification_details": verification_result["details"],
                    "execution_time": execution_time
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Logical verification failed: {e}")
            return VerificationOutput(
                result=VerificationResult.ERROR,
                score=0.0,
                details={"error": str(e), "traceback": traceback.format_exc()},
                error_message=str(e),
                execution_time=execution_time
            )
    
    def _extract_logic_content(self, model_output: str) -> Dict[str, Any]:
        """
        Extract logical content from model output.
        
        Args:
            model_output: The output from the model
            
        Returns:
            Dictionary containing extracted logical content
        """
        content = {
            "premises": [],
            "conclusions": [],
            "arguments": [],
            "logical_connectors": [],
            "contradictions": [],
            "fallacies": [],
            "structure": "unknown"
        }
        
        sentences = self._split_into_sentences(model_output)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check for logical connectors
            connectors = self._find_logical_connectors(sentence)
            content["logical_connectors"].extend(connectors)
            
            # Identify premises and conclusions
            if self._is_premise(sentence):
                content["premises"].append(sentence)
            elif self._is_conclusion(sentence):
                content["conclusions"].append(sentence)
            
            # Check for contradictions
            if self._contains_contradiction(sentence):
                content["contradictions"].append(sentence)
            
            # Check for fallacies
            fallacies = self._detect_fallacies(sentence)
            content["fallacies"].extend(fallacies)
        
        # Determine argument structure
        content["structure"] = self._determine_argument_structure(content)
        
        # Extract arguments (premise-conclusion pairs)
        content["arguments"] = self._extract_arguments(content["premises"], content["conclusions"])
        
        return content
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - can be improved with NLP libraries
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _find_logical_connectors(self, sentence: str) -> List[str]:
        """Find logical connectors in a sentence."""
        connectors = []
        sentence_lower = sentence.lower()
        
        for connector in self.config["logical_connectors"]:
            if connector.lower() in sentence_lower:
                connectors.append(connector)
        
        return connectors
    
    def _is_premise(self, sentence: str) -> bool:
        """Check if a sentence is a premise."""
        # Look for indicators of premises
        premise_indicators = [
            r'because', r'since', r'given that', r'assuming',
            r'if.*then', r'when.*then', r'provided that'
        ]
        
        sentence_lower = sentence.lower()
        return any(re.search(pattern, sentence_lower) for pattern in premise_indicators)
    
    def _is_conclusion(self, sentence: str) -> bool:
        """Check if a sentence is a conclusion."""
        # Look for indicators of conclusions
        conclusion_indicators = [
            r'therefore', r'thus', r'hence', r'so',
            r'consequently', r'as a result', r'it follows that',
            r'we can conclude', r'this means'
        ]
        
        sentence_lower = sentence.lower()
        return any(re.search(pattern, sentence_lower) for pattern in conclusion_indicators)
    
    def _contains_contradiction(self, sentence: str) -> bool:
        """Check if a sentence contains contradictions."""
        contradiction_indicators = self.config["contradiction_indicators"]
        sentence_lower = sentence.lower()
        
        return any(indicator.lower() in sentence_lower for indicator in contradiction_indicators)
    
    def _detect_fallacies(self, sentence: str) -> List[Dict[str, Any]]:
        """Detect logical fallacies in a sentence."""
        fallacies = []
        sentence_lower = sentence.lower()
        
        for fallacy_type, patterns in self.config["fallacy_patterns"].items():
            for pattern in patterns:
                if re.search(pattern, sentence_lower, re.IGNORECASE):
                    fallacies.append({
                        "type": fallacy_type,
                        "pattern": pattern,
                        "sentence": sentence
                    })
        
        return fallacies
    
    def _determine_argument_structure(self, content: Dict[str, Any]) -> str:
        """Determine the structure of the argument."""
        premises_count = len(content["premises"])
        conclusions_count = len(content["conclusions"])
        connectors_count = len(content["logical_connectors"])
        
        if premises_count > 0 and conclusions_count > 0 and connectors_count > 0:
            return "deductive"
        elif premises_count > 0 and conclusions_count > 0:
            return "inductive"
        elif conclusions_count > 0:
            return "assertion"
        elif premises_count > 0:
            return "premise_only"
        else:
            return "unstructured"
    
    def _extract_arguments(self, premises: List[str], conclusions: List[str]) -> List[Dict[str, str]]:
        """Extract premise-conclusion argument pairs."""
        arguments = []
        
        # Simple pairing - can be improved with more sophisticated analysis
        for premise in premises:
            for conclusion in conclusions:
                arguments.append({
                    "premise": premise,
                    "conclusion": conclusion
                })
        
        return arguments
    
    def _parse_logic_problem(self, instruction: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Parse the logical problem from the instruction.
        
        Args:
            instruction: The instruction containing the problem
            context: Additional context
            
        Returns:
            Dictionary containing problem information
        """
        problem_info = {
            "type": "unknown",
            "expected_structure": "unknown",
            "constraints": [],
            "context": context or {}
        }
        
        instruction_lower = instruction.lower()
        
        # Determine problem type
        if any(word in instruction_lower for word in ['analyze', 'evaluate', 'assess']):
            problem_info["type"] = "analysis"
        elif any(word in instruction_lower for word in ['prove', 'demonstrate', 'show']):
            problem_info["type"] = "proof"
        elif any(word in instruction_lower for word in ['explain', 'justify', 'reason']):
            problem_info["type"] = "explanation"
        elif any(word in instruction_lower for word in ['compare', 'contrast']):
            problem_info["type"] = "comparison"
        
        # Determine expected structure
        if any(word in instruction_lower for word in ['deductive', 'logical', 'syllogism']):
            problem_info["expected_structure"] = "deductive"
        elif any(word in instruction_lower for word in ['inductive', 'generalization']):
            problem_info["expected_structure"] = "inductive"
        elif any(word in instruction_lower for word in ['analogy', 'comparison']):
            problem_info["expected_structure"] = "analogical"
        
        return problem_info
    
    def _verify_logic(
        self,
        logic_content: Dict[str, Any],
        problem_info: Dict[str, Any],
        expected_output: Optional[str],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Verify the logical reasoning.
        
        Args:
            logic_content: Extracted logical content
            problem_info: Problem information
            expected_output: Expected output
            context: Additional context
            
        Returns:
            Verification result
        """
        score = 0.0
        details = {}
        
        # Check for logical structure
        if self.config["check_structure"]:
            structure_score = self._check_argument_structure(logic_content, problem_info)
            score += structure_score * 0.3
            details["structure_score"] = structure_score
        
        # Check for logical consistency
        if self.config["check_consistency"]:
            consistency_score = self._check_logical_consistency(logic_content)
            score += consistency_score * 0.3
            details["consistency_score"] = consistency_score
        
        # Check for coherent reasoning
        if self.config["check_coherence"]:
            coherence_score = self._check_reasoning_coherence(logic_content)
            score += coherence_score * 0.2
            details["coherence_score"] = coherence_score
        
        # Check for fallacies
        if self.config["check_fallacies"]:
            fallacy_penalty = self._calculate_fallacy_penalty(logic_content)
            score -= fallacy_penalty
            details["fallacy_penalty"] = fallacy_penalty
            details["fallacies_found"] = logic_content["fallacies"]
        
        # Ensure score is within bounds
        score = max(0.0, min(1.0, score))
        
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
    
    def _check_argument_structure(self, logic_content: Dict[str, Any], problem_info: Dict[str, Any]) -> float:
        """Check if the argument has proper structure."""
        structure = logic_content["structure"]
        expected_structure = problem_info["expected_structure"]
        
        # Perfect match
        if structure == expected_structure and expected_structure != "unknown":
            return 1.0
        
        # Good structure even if not exactly expected
        if structure in ["deductive", "inductive"]:
            return 0.8
        
        # Basic structure
        if structure in ["assertion", "premise_only"]:
            return 0.5
        
        # No structure
        return 0.0
    
    def _check_logical_consistency(self, logic_content: Dict[str, Any]) -> float:
        """Check for logical consistency."""
        contradictions = logic_content["contradictions"]
        fallacies = logic_content["fallacies"]
        
        # Perfect consistency
        if not contradictions and not fallacies:
            return 1.0
        
        # Minor inconsistencies
        if len(contradictions) <= 1 and len(fallacies) <= 1:
            return 0.7
        
        # Moderate inconsistencies
        if len(contradictions) <= 2 and len(fallacies) <= 2:
            return 0.4
        
        # Major inconsistencies
        return 0.0
    
    def _check_reasoning_coherence(self, logic_content: Dict[str, Any]) -> float:
        """Check for coherent reasoning."""
        premises = logic_content["premises"]
        conclusions = logic_content["conclusions"]
        connectors = logic_content["logical_connectors"]
        arguments = logic_content["arguments"]
        
        # Check if there's a logical flow
        has_premises = len(premises) > 0
        has_conclusions = len(conclusions) > 0
        has_connectors = len(connectors) > 0
        has_arguments = len(arguments) > 0
        
        # Perfect coherence
        if has_premises and has_conclusions and has_connectors and has_arguments:
            return 1.0
        
        # Good coherence
        if (has_premises and has_conclusions) or (has_conclusions and has_connectors):
            return 0.8
        
        # Basic coherence
        if has_premises or has_conclusions:
            return 0.5
        
        # No coherence
        return 0.0
    
    def _calculate_fallacy_penalty(self, logic_content: Dict[str, Any]) -> float:
        """Calculate penalty for logical fallacies."""
        fallacies = logic_content["fallacies"]
        
        if not fallacies:
            return 0.0
        
        # Penalty based on number and severity of fallacies
        total_penalty = 0.0
        
        for fallacy in fallacies:
            fallacy_type = fallacy["type"]
            
            # Different penalties for different fallacy types
            if fallacy_type in ["circular_reasoning", "false_dichotomy"]:
                total_penalty += 0.3  # Severe fallacies
            elif fallacy_type in ["straw_man", "ad_hominem"]:
                total_penalty += 0.2  # Moderate fallacies
            else:
                total_penalty += 0.1  # Minor fallacies
        
        # Cap the penalty
        return min(0.5, total_penalty) 