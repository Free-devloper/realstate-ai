from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from . import db


class User(db.Model):
    """User model for buyers, sellers, and agents"""
    
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    user_type = db.Column(db.String(20), default='buyer')  # buyer, seller, agent
    phone = db.Column(db.String(20))
    first_name = db.Column(db.String(50))
    last_name = db.Column(db.String(50))
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    favorites = db.relationship('Favorite', back_populates='user', cascade='all, delete-orphan', lazy='select')
    search_history = db.relationship('SearchHistory', back_populates='user', cascade='all, delete-orphan', lazy='select')
    ai_conversations = db.relationship('AIConversation', back_populates='user', cascade='all, delete-orphan', lazy='select')
    
    def set_password(self, password):
        """Set password hash"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password against hash"""
        return check_password_hash(self.password_hash, password)
    
    def is_agent(self):
        """Check if user is an agent"""
        return self.user_type == 'agent'
    
    def is_buyer(self):
        """Check if user is a buyer"""
        return self.user_type == 'buyer'
    
    def is_seller(self):
        """Check if user is a seller"""
        return self.user_type == 'seller'
    
    def to_dict(self):
        """Convert user to dictionary for API responses"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'user_type': self.user_type,
            'phone': self.phone,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class Favorite(db.Model):
    """Favorite properties model"""
    
    __tablename__ = 'favorites'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    property_id = db.Column(db.Integer, db.ForeignKey('properties.id'), nullable=False)
    added_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User', back_populates='favorites')
    property = db.relationship('Property', back_populates='favorites')
    
    __table_args__ = (
        db.UniqueConstraint('user_id', 'property_id', name='unique_favorite'),
    )
    
    def to_dict(self):
        """Convert favorite to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'property_id': self.property_id,
            'added_at': self.added_at.isoformat() if self.added_at else None
        }
