from app import db
from datetime import datetime

class AnalysisResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    company_name = db.Column(db.String(200), nullable=False)
    hr_name = db.Column(db.String(200), nullable=False)
    hr_email = db.Column(db.String(200), nullable=False)
    total_processed = db.Column(db.Integer, default=0)
    shortlisted_count = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<AnalysisResult {self.company_name}>'

class Candidate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey('analysis_result.id'), nullable=False)
    name = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(200))
    phone = db.Column(db.String(50))
    match_score = db.Column(db.Float, nullable=False)
    target_role = db.Column(db.String(200))
    experience_years = db.Column(db.Integer, default=0)
    education = db.Column(db.String(200))
    university = db.Column(db.String(200))
    is_shortlisted = db.Column(db.Boolean, default=False)
    is_ai_generated = db.Column(db.Boolean, default=False)
    resume_filename = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Candidate {self.name}>'
