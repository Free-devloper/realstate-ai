from datetime import datetime
from . import db


class Agent(db.Model):
    """Real estate agent model"""
    
    __tablename__ = 'agents'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    license_number = db.Column(db.String(50))
    company = db.Column(db.String(100))
    bio = db.Column(db.Text)
    image_url = db.Column(db.String(500))
    is_verified = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    properties = db.relationship('Property', back_populates='agent', lazy='select')
    
    def to_dict(self):
        """Convert agent to dictionary for API responses"""
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'phone': self.phone,
            'license_number': self.license_number,
            'company': self.company,
            'bio': self.bio,
            'image_url': self.image_url,
            'is_verified': self.is_verified,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class SearchHistory(db.Model):
    """Search history model for AI recommendations"""
    
    __tablename__ = 'search_history'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    search_query = db.Column(db.Text, nullable=False)
    filters = db.Column(db.Text)  # JSON string of filters
    results_count = db.Column(db.Integer, default=0)
    searched_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User', back_populates='search_history')
    
    def to_dict(self):
        """Convert search history to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'search_query': self.search_query,
            'filters': self.filters,
            'results_count': self.results_count,
            'searched_at': self.searched_at.isoformat() if self.searched_at else None
        }


class AIConversation(db.Model):
    """Persistent AI chat thread for a user and persona."""

    __tablename__ = 'ai_conversations'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    agent_id = db.Column(db.String(80), nullable=False)
    title = db.Column(db.String(200), nullable=False, default='New conversation')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_message_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', back_populates='ai_conversations')
    messages = db.relationship(
        'AIConversationMessage',
        back_populates='conversation',
        cascade='all, delete-orphan',
        order_by='AIConversationMessage.created_at.asc()',
        lazy='select',
    )

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'agent_id': self.agent_id,
            'preferred_agent_id': self.agent_id,
            'title': self.title,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'last_message_at': self.last_message_at.isoformat() if self.last_message_at else None,
            'message_count': len(self.messages),
        }


class AIConversationMessage(db.Model):
    """Single persisted AI chat message."""

    __tablename__ = 'ai_conversation_messages'

    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('ai_conversations.id'), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    content = db.Column(db.Text, nullable=False)
    source = db.Column(db.String(80))
    agent_id = db.Column(db.String(80))
    agent_name = db.Column(db.String(120))
    recommended_properties = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    conversation = db.relationship('AIConversation', back_populates='messages')

    def to_dict(self):
        return {
            'id': self.id,
            'conversation_id': self.conversation_id,
            'role': self.role,
            'content': self.content,
            'source': self.source,
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'recommended_properties': self.recommended_properties,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }
