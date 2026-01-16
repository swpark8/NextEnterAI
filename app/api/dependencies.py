from app.services.resume_engine import resume_engine, MatchingEngine

def get_resume_engine() -> MatchingEngine:
    """
    Dependency injection for MatchingEngine singleton.
    """
    return resume_engine
