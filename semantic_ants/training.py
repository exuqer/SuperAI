"""Training module for SuperAI - handles learning with physics simulation"""
import time
import uuid
from typing import List, Dict, Any, Optional, Set

from .tokenizer import tokenize, split_sentences
from .physics import (
    WordState, PhysicsConfig, 
    place_new_words_around_center, run_simulation
)
from .database import (
    init_db, create_session, get_session, list_sessions,
    add_words, add_phrase, get_words, get_phrases,
    update_words_positions, get_session_stats, reset_session,
    add_training_stats
)


class TrainingManager:
    """Manages training sessions and learning process."""
    
    def __init__(self, config: Optional[PhysicsConfig] = None):
        self.config = config or PhysicsConfig()
        init_db()
        self._current_session: Optional[str] = None
    
    def create_session(self, name: str = "Обучение") -> str:
        """Create a new training session."""
        session_id = create_session(name)
        self._current_session = session_id
        return session_id
    
    def get_session(self, session_id: Optional[str] = None) -> Optional[Dict]:
        """Get session info."""
        sid = session_id or self._current_session
        if not sid:
            return None
        return get_session(sid)
    
    def set_session(self, session_id: str):
        """Set current session."""
        self._current_session = session_id
    
    def list_sessions(self) -> List[Dict]:
        """List all sessions."""
        return list_sessions()
    
    def delete_session(self, session_id: str):
        """Delete a session."""
        from .database import delete_session
        delete_session(session_id)
        if self._current_session == session_id:
            self._current_session = None
    
    def learn(self, text: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Train on text, update word space with physics simulation."""
        sid = session_id or self._current_session
        if not sid:
            sid = self.create_session()
        
        start_time = time.time()
        
        # Tokenize and split into sentences
        sentences = split_sentences(text)
        all_tokens = []
        sentence_tokens = []
        
        for sentence in sentences:
            tokens = tokenize(sentence)
            if tokens:
                sentence_tokens.append(tokens)
                all_tokens.extend(tokens)
        
        if not all_tokens:
            return {"success": False, "error": "No valid tokens found"}
        
        # Track unique words in this request
        unique_in_request: Set[str] = set(all_tokens)
        
        # Get existing words before this request
        existing_words_data = get_words(sid)
        existing_words_set = {w["word"] for w in existing_words_data}
        
        # Add words to database (increments mass for existing)
        word_counts = add_words(sid, list(unique_in_request))
        
        # Get all words with current state
        all_words_data = get_words(sid)
        
        # Create WordState objects
        word_states: Dict[str, WordState] = {}
        for wd in all_words_data:
            word_states[wd["word"]] = WordState(
                word=wd["word"],
                mass=wd["mass"],
                x=wd["x"],
                y=wd["y"],
            )
        
        # Identify new words in this request
        new_words_list = [word_states[w] for w in unique_in_request if w not in existing_words_set]
        existing_words_list = [word_states[w] for w in unique_in_request if w in existing_words_set]
        
        # Place new words around center of known words from the sentences
        # Use words from the sentences that already exist
        known_in_sentences = [ws for ws in existing_words_list]
        place_new_words_around_center(new_words_list, known_in_sentences, self.config)
        
        # Build phrase groups for physics (each sentence is a phrase group)
        phrase_groups: List[List[WordState]] = []
        for sent_tokens in sentence_tokens:
            group = [word_states[tok] for tok in sent_tokens if tok in word_states]
            if len(group) >= 2:
                phrase_groups.append(group)
        
        # Run physics simulation
        all_word_states = list(word_states.values())
        run_simulation(all_word_states, phrase_groups, self.config)
        
        # Update positions in database
        positions = {ws.word: (ws.x, ws.y) for ws in all_word_states}
        update_words_positions(sid, positions)
        
        # Add phrases (sentences) to database
        for sent_tokens in sentence_tokens:
            add_phrase(sid, sent_tokens)
        
        # Record training stats
        stats = get_session_stats(sid)
        add_training_stats(
            session_id=sid,
            epoch=1,  # Simplified - could track epoch properly
            tokens=stats["tokens"],
            edges=stats["edges"],
            phrases=stats["phrases"],
            loss=None
        )
        
        # Prepare response
        word_data = [
            {
                "word": ws.word,
                "mass": ws.mass,
                "x": ws.x,
                "y": ws.y,
            }
            for ws in all_word_states
        ]
        
        return {
            "success": True,
            "session_id": sid,
            "words": word_data,
            "stats": stats,
            "time_ms": int((time.time() - start_time) * 1000),
        }
    
    def get_space(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get current word space state."""
        sid = session_id or self._current_session
        if not sid:
            return {"words": [], "stats": {"tokens": 0, "total_tokens": 0, "phrases": 0, "edges": 0}}
        
        words_data = get_words(sid)
        word_data = [
            {
                "word": w["word"],
                "mass": w["mass"],
                "x": w["x"],
                "y": w["y"],
            }
            for w in words_data
        ]
        stats = get_session_stats(sid)
        
        return {
            "words": word_data,
            "stats": stats,
        }
    
    def reset_space(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Reset (clear) the word space."""
        sid = session_id or self._current_session
        if not sid:
            return {"success": False, "error": "No session"}
        
        reset_session(sid)
        return {"success": True, "words": [], "stats": {"tokens": 0, "total_tokens": 0, "phrases": 0, "edges": 0}}


# Global instance
_training_manager: Optional[TrainingManager] = None


def get_training_manager() -> TrainingManager:
    """Get or create global training manager."""
    global _training_manager
    if _training_manager is None:
        _training_manager = TrainingManager()
    return _training_manager