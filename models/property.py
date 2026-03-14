from datetime import datetime
from . import db


class Property(db.Model):
    """Property listing model for real estate AI application"""
    
    __tablename__ = 'properties'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    price = db.Column(db.Float, nullable=False)
    location = db.Column(db.String(300), nullable=False)
    address = db.Column(db.String(300), nullable=False)
    bedrooms = db.Column(db.Integer, default=0)
    bathrooms = db.Column(db.Integer, default=0)
    square_feet = db.Column(db.Integer, default=0)
    property_type = db.Column(db.String(50), default='house')  # house, apartment, condo, townhouse
    status = db.Column(db.String(20), default='available')  # available, sold, pending
    listing_date = db.Column(db.DateTime, default=datetime.utcnow)
    image_url = db.Column(db.String(500))
    agent_id = db.Column(db.Integer, db.ForeignKey('agents.id'))
    
    # Relationships
    agent = db.relationship('Agent', back_populates='properties')
    favorites = db.relationship('Favorite', back_populates='property', cascade='all, delete-orphan', lazy='select')
    
    # Search indices
    search_text = db.Column(db.Text)
    
    def __repr__(self):
        return f'<Property {self.id}: {self.title} - ${self.price:,.2f}>'
    
    def to_dict(self):
        """Convert property to dictionary for API responses"""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'price': self.price,
            'location': self.location,
            'address': self.address,
            'bedrooms': self.bedrooms,
            'bathrooms': self.bathrooms,
            'square_feet': self.square_feet,
            'property_type': self.property_type,
            'status': self.status,
            'listing_date': self.listing_date.isoformat() if self.listing_date else None,
            'image_url': self.image_url,
            'ai_score': self.get_ai_score(),
            'agent': self.agent.to_dict() if self.agent else None
        }
    
    def get_ai_score(self):
        """Calculate AI score based on property features and market factors"""
        score = 0
        
        # Price factor (30% weight) - optimal range is $200K-$500K
        if 200000 <= self.price <= 500000:
            score += 30
        elif 500000 <= self.price <= 1000000:
            score += 25
        elif self.price > 1000000:
            score += 20
        else:
            score += 15
        
        # Bedrooms factor (20% weight) - more bedrooms = higher score
        if self.bedrooms >= 4:
            score += 25
        elif self.bedrooms >= 3:
            score += 20
        elif self.bedrooms >= 2:
            score += 15
        else:
            score += 10
        
        # Square feet factor (15% weight) - larger properties score higher
        if self.square_feet >= 2500:
            score += 25
        elif self.square_feet >= 2000:
            score += 20
        elif self.square_feet >= 1500:
            score += 15
        else:
            score += 10
        
        # Property type factor (15% weight)
        type_scores = {'house': 20, 'townhouse': 18, 'apartment': 15, 'condo': 12}
        score += type_scores.get(self.property_type, 10)
        
        # Bathroom factor (10% weight)
        if self.bathrooms >= 3:
            score += 20
        elif self.bathrooms >= 2:
            score += 15
        else:
            score += 10
        
        # Status factor (10% weight) - available properties score higher
        if self.status == 'available':
            score += 20
        elif self.status == 'pending':
            score += 10
        
        # Image factor (5% weight)
        if self.image_url:
            score += 10
        
        # Listing recency factor (5% weight) - newer listings score higher
        if self.listing_date and (datetime.utcnow() - self.listing_date).days < 30:
            score += 15
        elif self.listing_date and (datetime.utcnow() - self.listing_date).days < 60:
            score += 10
        elif self.listing_date:
            score += 5
        
        return min(score, 100)
