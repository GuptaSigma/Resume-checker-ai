import re
import logging
import math
import string
import statistics
from datetime import datetime
from collections import Counter

# Try TextBlob; if not available, we fallback to neutral sentiment.
try:
    from textblob import TextBlob
    _HAS_TEXTBLOB = True
except ImportError:
    _HAS_TEXTBLOB = False


class AIDetector:
    def __init__(self):
        # Enhanced AI-cliché / corporate buzzwords (expanded for modern AI)
        self.ai_phrases = [
            # Classic AI buzzwords
            'leveraging cutting-edge', 'synergistic approach', 'paradigm shift',
            'best-in-class', 'game-changing', 'revolutionary', 'utilize',
            'facilitate', 'comprehensive understanding', 'extensive experience',
            'proven track record', 'dynamic professional', 'results-driven',
            'detail-oriented', 'strategic initiative', 'foster a culture',
            'holistic perspective', 'robust framework', 'streamline operations',
            'highly motivated', 'passionate', 'ready to contribute',
            'drive innovation', 'culture of innovation',
            # Modern AI patterns (GPT-4, Claude, Gemini style)
            'dedicated professional', 'committed to excellence', 'thrive in dynamic environments',
            'collaborative environment', 'cross-functional teams', 'stakeholder engagement',
            'actionable insights', 'data-driven solutions', 'scalable solutions',
            'end-to-end solutions', 'seamless integration', 'optimize processes',
            'enhance user experience', 'innovative solutions', 'strategic thinking',
            'problem-solving skills', 'analytical mindset', 'proactive approach',
            'continuous improvement', 'agile methodologies', 'deliver results',
            'exceed expectations', 'value-added', 'impactful solutions',
            'transformative impact', 'sustainable growth', 'operational excellence',
            # Additional patterns common in technical resumes
            'strong foundation', 'extracting insights', 'complex datasets', 'real-world applications',
            'hands-on experience', 'user-friendly', 'cross-browser compatibility', 
            'mobile-first design', 'visually appealing', 'responsive design',
            'aspiring', 'currently focused', 'developing predictive models'
        ]
        # Enhanced threshold - more conservative to reduce false positives
        self.ai_confidence_threshold = 70

        # Expanded stopword list (lightweight, deterministic)
        self.stopwords = {
            "the","is","in","and","of","a","to","for","with","on","at","by","an",
            "this","that","it","as","from","or","but","if","then","are","was","were",
            "be","been","being","so","because","into","about","not","no","up","down",
            "out","over","under","again","further","once","while","after","before",
            "more","most","less","least","you","your","we","our"
        }

    # --------- Helpers ---------
    def _count_syllables(self, word: str) -> int:
        word = word.lower()
        if not word:
            return 0
        vowels = "aeiouy"
        count = 1 if word[0] in vowels else 0
        for i in range(1, len(word)):
            if word[i] in vowels and word[i - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
            count += 1
        return max(1, count)

    def _split_sentences(self, text: str) -> list[str]:
        # Improved regex for better sentence splitting
        parts = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+', text)
        return [s.strip() for s in parts if s and s.strip()]

    def _words(self, text: str) -> list[str]:
        return re.findall(r'\b\w+\b', text.lower())

    # --------- Features ---------
    def _flesch_reading_ease(self, text: str) -> float:
        words = self._words(text)
        sents = self._split_sentences(text)
        if not words or not sents:
            return 50.0
        total_syllables = sum(self._count_syllables(w) for w in words)
        num_words, num_sents = len(words), len(sents)
        score = 206.835 - 1.015 * (num_words / num_sents) - 84.6 * (total_syllables / num_words)
        return max(0.0, min(100.0, score))

    def _burstiness(self, text: str) -> float:
        sents = self._split_sentences(text)
        if len(sents) < 2:
            return 0.5
        wc = [len(s.split()) for s in sents]
        mean = statistics.mean(wc)
        stdev = statistics.stdev(wc) if len(wc) > 1 else 0.0
        return min(1.0, stdev / mean if mean > 0 else 0.5)

    def _perplexity_proxy(self, text: str) -> float:
        # A simple proxy for perplexity
        common_words = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i'}
        unique_words = set(self._words(text))
        if not unique_words:
            return 50.0
        common_count = sum(1 for w in unique_words if w in common_words)
        return 100.0 - (common_count / len(unique_words)) * 50.0

    def _entropy_chars(self, text: str) -> float:
        # Calculates character-level entropy
        char_counts = Counter(c for c in text.lower() if 'a' <= c <= 'z')
        total_chars = sum(char_counts.values())
        if not total_chars:
            return 0.0
        return -sum((count / total_chars) * math.log2(count / total_chars) for count in char_counts.values())

    def _bullet_score(self, text: str) -> float:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return 0.0
        bullets = [ln for ln in lines if ln.startswith(("-", "•", "*", "—"))]
        bullet_ratio = len(bullets) / len(lines)
        avg_bullet_len = (sum(len(b) for b in bullets) / len(bullets)) if bullets else 0.0
        return round(min(1.0, bullet_ratio + (avg_bullet_len / 200.0)), 3)

    def _stopword_ratio(self, text: str) -> float:
        tokens = self._words(text)
        if not tokens:
            return 0.0
        stop_count = sum(1 for t in tokens if t in self.stopwords)
        return round(stop_count / len(tokens), 3)

    def _passive_voice_count(self, text: str) -> int:
        pattern = r'\b(is|are|was|were|be|been|being)\s+([a-zA-Z]+ed|made|given|taken|shown|known)\b'
        return len(re.findall(pattern, text.lower()))

    def _sentiment_neutrality(self, text: str) -> float:
        if _HAS_TEXTBLOB:
            try:
                polarity = TextBlob(text).sentiment.polarity
                return round(1.0 - abs(polarity), 3)
            except Exception:
                pass
        return 0.5

    def _ai_phrase_hits(self, text: str) -> int:
        lower_text = text.lower()
        return sum(1 for p in self.ai_phrases if p in lower_text)
    
    def _find_ai_phrases_with_positions(self, text: str) -> list[dict]:
        """Finds AI phrases and their positions."""
        found_phrases = []
        t_lower = text.lower()
        
        for phrase in self.ai_phrases:
            start_index = 0
            while (pos := t_lower.find(phrase, start_index)) != -1:
                original_phrase = text[pos:pos + len(phrase)]
                found_phrases.append({
                    'phrase': original_phrase,
                    'position': pos,
                    'length': len(phrase),
                    'ai_phrase_type': phrase
                })
                start_index = pos + 1
        
        return sorted(found_phrases, key=lambda x: x['position'])

    # --------- Suspicion shaping (direction-correct) ---------
    def _suspicion_vector(self, text: str) -> tuple[dict, dict, dict]:
        fre = self._flesch_reading_ease(text)
        burst = self._burstiness(text)
        pplx = self._perplexity_proxy(text)
        ent = self._entropy_chars(text)
        bullet = self._bullet_score(text)
        stopr = self._stopword_ratio(text)
        passive = self._passive_voice_count(text)
        neutral = self._sentiment_neutrality(text)
        ai_hits = self._ai_phrase_hits(text)

        suspicions = {
            'ai_phrases': min(1.0, ai_hits / 5.0),
            'neutrality': max(0.0, min(1.0, neutral)),
            'low_burst': max(0.0, min(1.0, 1.0 - burst)),
            'bullets_low': max(0.0, min(1.0, max(0.0, (0.25 - bullet) / 0.25))),
            'stop_dev': max(0.0, min(1.0, abs(stopr - 0.45) / 0.25)),
            'entropy_low': max(0.0, min(1.0, (4.0 - ent) / 0.8)),
            'read_dev': max(0.0, min(1.0, abs(fre - 60.0) / 40.0)),
            'pplx_low': max(0.0, min(1.0, (60.0 - pplx) / 60.0)),
            'passive': max(0.0, min(1.0, passive / 6.0))
        }

        weights = {
            'ai_phrases': 0.25,
            'neutrality': 0.16,
            'low_burst':  0.18,
            'bullets_low':0.10,
            'stop_dev':   0.14,
            'entropy_low':0.09,
            'read_dev':   0.05,
            'pplx_low':   0.02,
            'passive':    0.01
        }
        
        raw_features = {
            'readability': fre, 'burstiness': burst, 'perplexity': pplx,
            'entropy_score': ent, 'bullet_score': bullet, 'stopword_ratio': stopr,
            'passive_voice': passive, 'sentiment_neutrality': neutral, 'ai_phrases': ai_hits
        }
        return suspicions, weights, raw_features
    
    def _analyze_sentences_for_explainability(self, text: str) -> list[dict]:
        """Analyzes each sentence for AI-like patterns."""
        sentences = self._split_sentences(text)
        sentence_analysis = []
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            
            words = self._words(sentence)
            word_count = len(words)
            
            ai_phrases_in_sentence = [p for p in self.ai_phrases if p in sentence.lower()]
            
            stopword_ratio = len([w for w in words if w in self.stopwords]) / max(1, word_count)
            has_passive = self._passive_voice_count(sentence) > 0
            
            suspicion_factors = []
            if ai_phrases_in_sentence:
                suspicion_factors.append(f"Contains AI buzzwords: {', '.join(ai_phrases_in_sentence)}")
            if word_count > 25:
                suspicion_factors.append("Very long sentence")
            if stopword_ratio < 0.3:
                suspicion_factors.append("Unusually formal (low common words)")
            if has_passive:
                suspicion_factors.append("Contains passive voice")
            
            sentence_suspicion = min(1.0, len(suspicion_factors) * 0.3 + len(ai_phrases_in_sentence) * 0.4)
            
            sentence_analysis.append({
                'sentence_index': i,
                'sentence': sentence.strip(),
                'word_count': word_count,
                'ai_phrases': ai_phrases_in_sentence,
                'suspicion_score': round(sentence_suspicion, 3),
                'suspicion_factors': suspicion_factors,
                'is_suspicious': sentence_suspicion > 0.4
            })
        
        return sentence_analysis
    
    def _generate_explanation(self, suspicions: dict, weights: dict, ai_phrases_found: list[dict], sentence_analysis: list[dict]) -> list[str]:
        """Generates human-readable explanations."""
        explanations = []
        
        if suspicions['ai_phrases'] > 0.3:
            explanations.append(f"🚨 Found {len(ai_phrases_found)} AI buzzwords/phrases like: {', '.join([p['ai_phrase_type'] for p in ai_phrases_found[:3]])}{'...' if len(ai_phrases_found) > 3 else ''}")
        
        if suspicions['neutrality'] > 0.7:
            explanations.append("😐 Very neutral sentiment, common in AI-generated text.")
        
        if suspicions['low_burst'] > 0.6:
            explanations.append("📏 Sentences have very uniform length, a typical AI trait.")
        
        if suspicions['bullets_low'] > 0.7:
            explanations.append("📝 Lacks typical resume formatting like bullet points.")
        
        if suspicions['stop_dev'] > 0.6:
            explanations.append("🔤 Unusual ratio of common words, unlike typical human writing.")
        
        suspicious_sentences = [s for s in sentence_analysis if s['is_suspicious']]
        if suspicious_sentences:
            explanations.append(f"⚠️ Found {len(suspicious_sentences)} suspicious sentences with AI-like patterns.")
        
        return explanations

    def _map_s_to_percentage(self, S: float) -> float:
        # Maps weighted suspicion score to a percentage
        if S >= 0.65:
            return min(100.0, 88.0 + (S - 0.65) * 34.28) # 12.0 / 0.35
        elif S >= 0.45:
            return 60.0 + (S - 0.45) * 140.0 # 28.0 / 0.20
        elif S >= 0.30:
            return 35.0 + (S - 0.30) * 166.67 # 25.0 / 0.15
        else:
            return max(0.0, S * 116.67) # 35.0 / 0.30

    def _confidence_from_suspicions(self, suspicions: dict, weights: dict, ai_pct: float) -> float:
        # Enhanced confidence calculation
        strong_support = sum(w for k, w in weights.items() if suspicions[k] >= 0.5)
        very_strong_support = sum(w for k, w in weights.items() if suspicions[k] >= 0.7)
        
        ai_phrase_boost = min(25.0, suspicions['ai_phrases'] * 30.0)
        
        active_indicators = sum(1 for v in suspicions.values() if v > 0.4)
        multi_indicator_bonus = min(15.0, active_indicators * 2.0)
        
        vals = list(suspicions.values())
        mean_suspicion = statistics.mean(vals)
        variance = statistics.variance(vals)
        agreement_bonus = 15.0 * max(0.0, 1.0 - 2.0 * math.sqrt(variance))
        
        base_conf = 60.0 + (35.0 * strong_support) + (10.0 * very_strong_support)
        enhanced_conf = base_conf + ai_phrase_boost + multi_indicator_bonus + agreement_bonus
        
        margin_multiplier = 1.0
        if ai_pct >= 85.0:
            margin_multiplier = 1.1 + (ai_pct - 85.0) / 15.0 * 0.2
        elif ai_pct >= 70.0:
            margin_multiplier = 0.95 + (ai_pct - 70.0) / 15.0 * 0.15
        elif ai_pct <= 30.0:
            margin_multiplier = 1.0 + (30.0 - ai_pct) / 30.0 * 0.2
        
        final_confidence = enhanced_conf * margin_multiplier
        return float(max(0.0, min(99.0, final_confidence)))

    def _detect_hybrid_content(self, sentence_analysis: list[dict]) -> dict:
        """Detects if content is a mix of human and AI writing."""
        if len(sentence_analysis) < 3:
            return {'is_hybrid': False, 'confidence': 0.0, 'pattern': 'insufficient_data'}
        
        suspicion_scores = [s['suspicion_score'] for s in sentence_analysis]
        
        high_suspicion = [s > 0.6 for s in suspicion_scores]
        low_suspicion = [s < 0.3 for s in suspicion_scores]
        
        pattern_changes = sum(1 for i in range(1, len(high_suspicion)) if high_suspicion[i] != high_suspicion[i-1])
        
        has_both_extremes = any(high_suspicion) and any(low_suspicion)
        variance = statistics.variance(suspicion_scores)
        
        is_hybrid = has_both_extremes and (pattern_changes > len(suspicion_scores) * 0.3 or variance > 0.15)
        confidence = min(1.0, (pattern_changes / len(suspicion_scores)) + (variance * 2))
        
        pattern_type = 'mixed_editing' if is_hybrid else 'consistent'
        if pattern_changes > 3 and variance > 0.2:
            pattern_type = 'heavy_editing'
            
        return {
            'is_hybrid': is_hybrid,
            'confidence': round(confidence, 3),
            'pattern': pattern_type,
            'pattern_changes': pattern_changes,
            'variance': round(variance, 3)
        }
        
    def analyze_resume(self, text: str, candidate_name: str = 'Unknown', resume_filename: str = 'Unknown') -> dict:
        """Analyzes a resume for AI-generated content."""
        try:
            suspicions, weights, raw_feats = self._suspicion_vector(text)
            S = sum(suspicions[k] * weights[k] for k in weights)
            ai_percentage = round(self._map_s_to_percentage(S), 2)
            
            ai_confidence = round(self._confidence_from_suspicions(suspicions, weights, ai_percentage), 2)
            is_ai_generated = ai_percentage > self.ai_confidence_threshold
            
            ai_phrases_found = self._find_ai_phrases_with_positions(text)
            sentence_analysis = self._analyze_sentences_for_explainability(text)
            explanations = self._generate_explanation(suspicions, weights, ai_phrases_found, sentence_analysis)
            
            hybrid_analysis = self._detect_hybrid_content(sentence_analysis)

            return {
                'candidate_name': candidate_name,
                'resume_filename': resume_filename,
                'is_ai_generated': is_ai_generated,
                'ai_percentage': ai_percentage,
                'human_percentage': round(100.0 - ai_percentage, 2),
                'ai_confidence': ai_confidence,
                'features': {**raw_feats, 'suspicion': suspicions, 'weights': weights, 'S_weighted': round(S, 4)},
                'explainability': {
                    'ai_phrases_found': ai_phrases_found,
                    'sentence_analysis': sentence_analysis,
                    'explanations': explanations,
                    'suspicious_sentence_count': sum(1 for s in sentence_analysis if s['is_suspicious']),
                    'total_sentences': len(sentence_analysis)
                },
                'hybrid_analysis': hybrid_analysis,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            logging.error(f"Error in AI detection for {resume_filename}: {e}")
            return {'error': str(e)}

    def display_analysis(self, analysis_result: dict):
        """Displays analysis results in a user-friendly format."""
        if 'error' in analysis_result:
            print(f"\nAn error occurred: {analysis_result['error']}")
            return

        print(f"\n{'='*60}")
        print(f"AI DETECTION ANALYSIS: {analysis_result['candidate_name']}")
        print(f"{'='*60}")
        
        verdict = "🤖 AI-GENERATED" if analysis_result['is_ai_generated'] else "👤 HUMAN-WRITTEN"
        print(f"\n{verdict}")
        print(f"AI Likelihood: {analysis_result['ai_percentage']}%")
        print(f"Confidence: {analysis_result['ai_confidence']}%")
        
        if (exp := analysis_result.get('explainability')) and exp.get('explanations'):
            print("\n📋 REASONS FOR DETECTION:")
            for explanation in exp['explanations']:
                print(f"  • {explanation}")
        
        if (hybrid := analysis_result.get('hybrid_analysis')) and hybrid.get('is_hybrid'):
            print("\n🔄 HYBRID CONTENT DETECTED:")
            print(f"  Pattern: {hybrid['pattern']} (confidence: {hybrid['confidence']*100:.1f}%)")
            print("  This text appears to mix human and AI writing styles.")
        
        print(f"\n{'='*60}\n")


if __name__ == '__main__':
    detector = AIDetector()

    humanish_text = """
    Built a CRM from scratch with a small team; presented progress weekly; shipped v1 in 7 weeks.
    Bullet points and short lines summarize impact. Focused on users, reduced churn by 12%.
    Had fun debugging late-night issues. Coffee was essential!
    """
    
    modern_ai_text = """
    A dedicated professional with extensive experience in leveraging cutting-edge technologies to deliver impactful solutions. 
    I am committed to excellence and thrive in dynamic environments where I can engage with cross-functional teams. 
    My analytical mindset and proactive approach enable me to optimize processes and enhance user experience through data-driven solutions. 
    I am passionate about continuous improvement and delivering results that exceed expectations.
    """
    
    hybrid_text = """
    Software engineer with 5 years experience at Google and Meta. Love building scalable systems! 
    I am a dedicated professional with a proven track record of leveraging cutting-edge technologies to facilitate transformative impact. 
    Built the entire backend for our ML platform - was crazy complex but super rewarding. 
    My comprehensive understanding of modern methodologies enables me to deliver best-in-class solutions while fostering a culture of innovation.
    """

    print("\n" + "="*80)
    print("ENHANCED AI DETECTOR - EXPLAINABILITY DEMO")
    print("="*80)

    # Test human-like text
    print("\n[TEST 1: HUMAN-LIKE TEXT]")
    human_result = detector.analyze_resume(humanish_text, "John Smith", "human_resume.txt")
    detector.display_analysis(human_result)

    # Test modern AI text
    print("\n[TEST 2: MODERN AI TEXT]")
    ai_result = detector.analyze_resume(modern_ai_text, "AI Generated", "ai_resume.txt")
    detector.display_analysis(ai_result)

    # Test hybrid text
    print("\n[TEST 3: HYBRID HUMAN+AI TEXT]")
    hybrid_result = detector.analyze_resume(hybrid_text, "Mixed Content", "hybrid_resume.txt")
    detector.display_analysis(hybrid_result)
    
    print("\nDetection completed! The enhanced detector now provides:")
    print("✓ Detailed explanations of why text was flagged")
    print("✓ Identification of specific AI buzzwords and phrases")
    print("✓ Sentence-level analysis for suspicious patterns")
    print("✓ Detection of hybrid human+AI content")
    print("✓ Improved accuracy with refined weights and thresholds")