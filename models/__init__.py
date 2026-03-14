
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

from . import user
from . import agent
from . import property as property_model