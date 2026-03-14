import hashlib
import json
import os
import re
import time
from datetime import datetime, timedelta

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from sqlalchemy import inspect, or_, text
from sqlalchemy.exc import IntegrityError

from config import Config
from models import db
from models.agent import AIConversation, AIConversationMessage, Agent, SearchHistory
from models.property import Property
from models.user import Favorite, User
from services.agentic_graph import run_agentic_listing_flow
from services.ai_brain import get_ai_persona, list_ai_personas, select_chat_persona


PROPERTY_TYPES = {'house', 'apartment', 'condo', 'townhouse'}
PROPERTY_STATUSES = {'available', 'pending', 'sold'}


app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

AGENT_QUERY_CACHE = {}


def clean_text(value):
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    return value or None


def parse_optional_float(value, field_name):
    value = clean_text(value)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{field_name} must be a valid number.') from exc


def parse_optional_int(value, field_name):
    value = clean_text(value)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{field_name} must be a valid whole number.') from exc


def api_error(message, status_code=400, details=None):
    payload = {'error': message}
    if details:
        payload['details'] = details
    return jsonify(payload), status_code


def wants_json():
    return request.path.startswith('/api/')


def property_cover(property_obj):
    if getattr(property_obj, 'image_url', None):
        return property_obj.image_url
    palette = {
        'house': 'sunset-grove',
        'apartment': 'skyline-blue',
        'condo': 'amber-shore',
        'townhouse': 'terracotta',
    }
    return palette.get(getattr(property_obj, 'property_type', None), 'sunset-grove')


def serialize_property(property_obj, reason=None):
    payload = property_obj.to_dict()
    payload['cover_style'] = property_cover(property_obj)
    payload['lookup_key'] = f'local:{property_obj.id}'
    payload['listing_source'] = 'local_inventory'
    payload['source_label'] = 'Curated inventory'
    payload['is_external'] = False
    payload['can_favorite'] = True
    payload['listing_url'] = None
    if reason:
        payload['recommendation_reason'] = reason
    return payload


def serialize_listing_result(listing, reason=None):
    if isinstance(listing, Property):
        return serialize_property(listing, reason=reason)

    payload = dict(listing)
    payload.setdefault('cover_style', property_cover(type('Listing', (), payload)()))
    payload.setdefault('lookup_key', str(payload.get('id')))
    payload.setdefault('listing_source', 'live_market')
    payload.setdefault('source_label', 'Live market feed')
    payload.setdefault('is_external', True)
    payload.setdefault('can_favorite', False)
    payload.setdefault('listing_url', None)
    if reason:
        payload['recommendation_reason'] = reason
    return payload


def update_property_search_text(property_obj):
    property_obj.search_text = ' '.join(
        part for part in [
            property_obj.title,
            property_obj.description,
            property_obj.location,
            property_obj.address,
            property_obj.property_type,
            property_obj.status,
        ] if part
    ).lower()


def clone_json_value(value):
    return json.loads(json.dumps(value))


def normalize_history_for_cache(history):
    normalized = []
    for item in (history or [])[-6:]:
        if not isinstance(item, dict):
            continue
        role = clean_text(item.get('role'))
        content = clean_text(item.get('content'))
        if role and content:
            normalized.append({'role': role, 'content': content[:400]})
    return normalized


def build_agent_cache_key(mode, persona_id, message, filters=None, history=None):
    payload = {
        'mode': mode,
        'persona_id': persona_id,
        'message': clean_text(message) or '',
        'filters': {key: filters[key] for key in sorted(filters or {})},
        'history': normalize_history_for_cache(history),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def prune_agent_query_cache():
    now = time.time()
    expired_keys = [key for key, value in AGENT_QUERY_CACHE.items() if value.get('expires_at', 0) <= now]
    for key in expired_keys:
        AGENT_QUERY_CACHE.pop(key, None)

    max_entries = app.config.get('QUERY_CACHE_MAX_ENTRIES', 256)
    if len(AGENT_QUERY_CACHE) <= max_entries:
        return

    for key, _ in sorted(AGENT_QUERY_CACHE.items(), key=lambda item: item[1].get('created_at', 0))[:-max_entries]:
        AGENT_QUERY_CACHE.pop(key, None)


def get_cached_agent_flow(cache_key):
    if not app.config.get('QUERY_CACHE_ENABLED'):
        return None

    prune_agent_query_cache()
    entry = AGENT_QUERY_CACHE.get(cache_key)
    if not entry:
        return None
    return clone_json_value(entry['payload'])


def set_cached_agent_flow(cache_key, payload):
    if not app.config.get('QUERY_CACHE_ENABLED'):
        return

    prune_agent_query_cache()
    ttl = max(int(app.config.get('QUERY_CACHE_TTL', 180)), 1)
    AGENT_QUERY_CACHE[cache_key] = {
        'payload': clone_json_value(payload),
        'created_at': time.time(),
        'expires_at': time.time() + ttl,
    }
    prune_agent_query_cache()


def cacheable_flow_result(flow_result):
    return {
        'reply': flow_result.get('reply'),
        'source': flow_result.get('source', 'local_inventory'),
        'model_source': flow_result.get('model_source', 'local_fallback'),
        'remote_model': flow_result.get('remote_model'),
        'fallback_reason': flow_result.get('fallback_reason'),
        'retrieval_notes': flow_result.get('retrieval_notes') or [],
        'query_route': flow_result.get('query_route'),
        'route_reason': flow_result.get('route_reason'),
        'selected_results': [serialize_listing_result(item) for item in flow_result.get('selected_results') or []],
    }


def run_agentic_listing_flow_cached(persona, message='', filters=None, history=None, local_inventory=None, mode='chat'):
    cache_key = build_agent_cache_key(
        mode=mode,
        persona_id=persona['id'],
        message=message,
        filters=filters or {},
        history=history or [],
    )
    cached = get_cached_agent_flow(cache_key)
    if cached is not None:
        return cached, True

    flow_result = run_agentic_listing_flow(
        app.config,
        persona,
        message=message,
        history=history or [],
        filters=filters or {},
        local_inventory=local_inventory or [],
        mode=mode,
    )
    cacheable = cacheable_flow_result(flow_result)
    set_cached_agent_flow(cache_key, cacheable)
    return cacheable, False


def validate_property_type(property_type):
    if property_type and property_type not in PROPERTY_TYPES:
        raise ValueError(f'property_type must be one of: {", ".join(sorted(PROPERTY_TYPES))}.')


def validate_property_status(status):
    if status and status not in PROPERTY_STATUSES:
        raise ValueError(f'status must be one of: {", ".join(sorted(PROPERTY_STATUSES))}.')


def assign_property_fields(property_obj, data, partial=False):
    required_fields = ['title', 'description', 'price', 'location', 'address']
    for field in required_fields:
        if not partial and clean_text(data.get(field)) is None:
            raise ValueError(f'{field} is required.')
        if partial and field in data and clean_text(data.get(field)) is None:
            raise ValueError(f'{field} cannot be blank.')

    string_fields = ['title', 'description', 'location', 'address', 'image_url']
    for field in string_fields:
        if field in data:
            setattr(property_obj, field, clean_text(data.get(field)))

    numeric_fields = [
        ('price', parse_optional_float),
        ('bedrooms', parse_optional_int),
        ('bathrooms', parse_optional_int),
        ('square_feet', parse_optional_int),
        ('agent_id', parse_optional_int),
    ]
    for field, parser in numeric_fields:
        if field in data:
            value = parser(data.get(field), field)
            if value is not None:
                setattr(property_obj, field, value)
            elif field != 'price':
                setattr(property_obj, field, 0 if field in {'bedrooms', 'bathrooms', 'square_feet'} else None)

    if 'property_type' in data:
        property_type = clean_text(data.get('property_type'))
        validate_property_type(property_type)
        property_obj.property_type = property_type or 'house'

    if 'status' in data:
        status = clean_text(data.get('status'))
        validate_property_status(status)
        property_obj.status = status or 'available'

    if not partial:
        property_obj.property_type = property_obj.property_type or 'house'
        property_obj.status = property_obj.status or 'available'

    update_property_search_text(property_obj)


def apply_property_filters(query, params):
    query_text = clean_text(params.get('query'))
    location = clean_text(params.get('location'))
    min_price = parse_optional_float(params.get('min_price'), 'min_price')
    max_price = parse_optional_float(params.get('max_price'), 'max_price')
    min_bedrooms = parse_optional_int(params.get('min_bedrooms'), 'min_bedrooms')
    max_bedrooms = parse_optional_int(params.get('max_bedrooms'), 'max_bedrooms')
    property_type = clean_text(params.get('property_type'))
    status = clean_text(params.get('status'))

    validate_property_type(property_type)
    validate_property_status(status)

    if min_price is not None:
        query = query.filter(Property.price >= min_price)
    if max_price is not None:
        query = query.filter(Property.price <= max_price)
    if min_bedrooms is not None:
        query = query.filter(Property.bedrooms >= min_bedrooms)
    if max_bedrooms is not None:
        query = query.filter(Property.bedrooms <= max_bedrooms)
    if property_type:
        query = query.filter(Property.property_type == property_type)
    if status:
        query = query.filter(Property.status == status)
    if location:
        location_pattern = f'%{location}%'
        query = query.filter(
            or_(
                Property.location.ilike(location_pattern),
                Property.address.ilike(location_pattern),
            )
        )
    if query_text:
        pattern = f'%{query_text}%'
        query = query.filter(
            or_(
                Property.title.ilike(pattern),
                Property.description.ilike(pattern),
                Property.location.ilike(pattern),
                Property.address.ilike(pattern),
                Property.search_text.ilike(pattern),
            )
        )

    normalized = {
        'query': query_text,
        'location': location,
        'min_price': min_price,
        'max_price': max_price,
        'min_bedrooms': min_bedrooms,
        'max_bedrooms': max_bedrooms,
        'property_type': property_type,
        'status': status,
    }
    normalized = {key: value for key, value in normalized.items() if value not in (None, '')}
    return query, normalized


def rank_properties(properties, limit=None, exclude_ids=None):
    blocked = set(exclude_ids or [])
    ranked = sorted(
        [property_obj for property_obj in properties if property_obj.id not in blocked],
        key=lambda property_obj: (property_obj.get_ai_score(), property_obj.listing_date or datetime.min),
        reverse=True,
    )
    if limit is None:
        return ranked
    return ranked[:limit]


def top_properties(limit=10, exclude_ids=None):
    properties = Property.query.all()
    return rank_properties(properties, limit=limit, exclude_ids=exclude_ids)


def ai_runtime_status():
    huggingface_enabled = bool(app.config.get('HF_API_TOKEN')) and not bool(app.config.get('HF_DISABLED'))
    llama_cpp_enabled = bool(app.config.get('LLAMA_CPP_ENABLED'))
    remote_enabled = huggingface_enabled or llama_cpp_enabled
    live_listings_enabled = bool(app.config.get('RENTCAST_API_KEY'))

    if huggingface_enabled and llama_cpp_enabled:
        provider = 'huggingface -> llama_cpp'
        model = app.config.get('HF_MODEL')
    elif huggingface_enabled:
        provider = 'huggingface'
        model = app.config.get('HF_MODEL')
    elif llama_cpp_enabled:
        provider = 'llama_cpp'
        model = app.config.get('LLAMA_CPP_MODEL')
    else:
        provider = 'local_fallback'
        model = None

    return {
        'remote_enabled': remote_enabled,
        'provider': provider,
        'model': model,
        'fallback_provider': 'llama_cpp' if huggingface_enabled and llama_cpp_enabled else None,
        'fallback_model': app.config.get('LLAMA_CPP_MODEL') if llama_cpp_enabled else None,
        'streaming_supported': True,
        'query_cache_enabled': bool(app.config.get('QUERY_CACHE_ENABLED')),
        'live_listings_enabled': live_listings_enabled,
        'listings_provider': app.config.get('REAL_ESTATE_API_PROVIDER') if live_listings_enabled else 'local_inventory',
    }


def default_conversation_title():
    return 'New conversation'


def conversation_title_from_message(message):
    message = clean_text(message) or default_conversation_title()
    trimmed = ' '.join(message.split())
    if len(trimmed) <= 56:
        return trimmed
    return f'{trimmed[:53].rstrip()}...'


def serialize_conversation(conversation):
    payload = conversation.to_dict()
    last_message = conversation.messages[-1] if conversation.messages else None
    preferred_persona = get_ai_persona(conversation.agent_id)
    payload['last_preview'] = last_message.content[:120] if last_message else None
    payload['preferred_agent'] = preferred_persona
    return payload


def serialize_conversation_message(message_obj):
    payload = message_obj.to_dict()
    try:
        payload['recommended_properties'] = json.loads(message_obj.recommended_properties or '[]')
    except (TypeError, ValueError, json.JSONDecodeError):
        payload['recommended_properties'] = []

    if payload.get('role') == 'assistant' and not payload.get('agent_name'):
        fallback_persona = get_ai_persona(getattr(message_obj.conversation, 'agent_id', None)) if getattr(message_obj, 'conversation', None) else None
        if fallback_persona:
            payload['agent_id'] = payload.get('agent_id') or fallback_persona['id']
            payload['agent_name'] = fallback_persona['name']
    return payload


def _history_content_with_agent(role, content, agent_name=None):
    trimmed = clean_text(content)
    if not trimmed:
        return None
    if role == 'assistant' and agent_name:
        return f'[{agent_name}] {trimmed[:1200]}'
    return trimmed[:1200]


def build_history_from_messages(messages, limit=6):
    history = []
    for item in messages[-limit:]:
        role = clean_text(item.role)
        content = clean_text(item.content)
        agent_name = clean_text(getattr(item, 'agent_name', None))
        agent_id = clean_text(getattr(item, 'agent_id', None))
        if role in {'user', 'assistant'} and content:
            history.append({
                'role': role,
                'content': _history_content_with_agent(role, content, agent_name=agent_name),
                'agent_id': agent_id,
                'agent_name': agent_name,
            })
    return history


def get_user_or_404(user_id):
    return User.query.get_or_404(user_id)


def get_conversation_or_404(user_id, conversation_id):
    conversation = AIConversation.query.filter_by(id=conversation_id, user_id=user_id).first()
    if not conversation:
        return None
    return conversation


def property_reason_for_persona(persona_id):
    reasons = {
        'buyer-guide': 'Selected by the buyer guide for overall fit and livability',
        'investment-scout': 'Selected by the investment scout for pricing and upside balance',
        'neighborhood-navigator': 'Selected by the neighborhood navigator for lifestyle and location feel',
    }
    return reasons.get(persona_id, 'Selected by the AI concierge')


def overview_payload():
    properties = Property.query.all()
    prices = [property_obj.price for property_obj in properties]
    available_count = sum(1 for property_obj in properties if property_obj.status == 'available')
    return {
        'properties_count': len(properties),
        'available_count': available_count,
        'agents_count': Agent.query.count(),
        'average_price': round(sum(prices) / len(prices), 2) if prices else 0,
        'top_score': max((property_obj.get_ai_score() for property_obj in properties), default=0),
    }


def ensure_seed_data():
    if Agent.query.count() == 0:
        agents = [
            Agent(
                name='Ava Sterling',
                email='ava@realestateai.local',
                phone='(555) 210-1101',
                license_number='RSA-18821',
                company='Harbor & Hearth',
                bio='Luxury advisor focused on family-ready homes with calm neighborhoods and strong resale value.',
                is_verified=True,
            ),
            Agent(
                name='Noah Bennett',
                email='noah@realestateai.local',
                phone='(555) 210-1102',
                license_number='RSA-18822',
                company='Modern Nest Group',
                bio='Specializes in downtown condos, investment rentals, and first-time buyer coaching.',
                is_verified=True,
            ),
            Agent(
                name='Layla Chen',
                email='layla@realestateai.local',
                phone='(555) 210-1103',
                license_number='RSA-18823',
                company='Verdant Living',
                bio='Known for matching buyers with bright homes, walkable districts, and design-forward renovations.',
                is_verified=True,
            ),
        ]
        db.session.add_all(agents)
        db.session.flush()
    else:
        agents = Agent.query.order_by(Agent.id.asc()).all()

    if Property.query.count() > 0:
        return

    now = datetime.utcnow()
    sample_properties = [
        {
            'title': 'Sunlit Garden Residence',
            'description': 'A warm four-bedroom home with skylit living spaces, landscaped patios, and a chef-inspired kitchen.',
            'price': 485000,
            'location': 'Austin, Texas',
            'address': '1847 Willow Crest Lane',
            'bedrooms': 4,
            'bathrooms': 3,
            'square_feet': 2480,
            'property_type': 'house',
            'status': 'available',
            'listing_date': now - timedelta(days=4),
            'agent_id': agents[0].id,
        },
        {
            'title': 'Cityline Glass Condo',
            'description': 'Downtown condo with skyline views, concierge access, and a sleek open-plan layout.',
            'price': 629000,
            'location': 'Chicago, Illinois',
            'address': '88 Lakeshore Plaza, Unit 1704',
            'bedrooms': 2,
            'bathrooms': 2,
            'square_feet': 1460,
            'property_type': 'condo',
            'status': 'available',
            'listing_date': now - timedelta(days=10),
            'agent_id': agents[1].id,
        },
        {
            'title': 'Terrace Townhome Retreat',
            'description': 'Design-forward townhouse with rooftop entertaining space and flexible work-from-home rooms.',
            'price': 540000,
            'location': 'Denver, Colorado',
            'address': '612 Ember Terrace',
            'bedrooms': 3,
            'bathrooms': 3,
            'square_feet': 1980,
            'property_type': 'townhouse',
            'status': 'pending',
            'listing_date': now - timedelta(days=13),
            'agent_id': agents[2].id,
        },
        {
            'title': 'Marina Breeze Apartment',
            'description': 'Waterfront apartment with sunrise balconies, refined finishes, and resort-style amenities.',
            'price': 415000,
            'location': 'Miami, Florida',
            'address': '240 Harbor Walk, Unit 905',
            'bedrooms': 2,
            'bathrooms': 2,
            'square_feet': 1325,
            'property_type': 'apartment',
            'status': 'available',
            'listing_date': now - timedelta(days=2),
            'agent_id': agents[1].id,
        },
        {
            'title': 'Oaklight Family Estate',
            'description': 'Spacious suburban home with five bedrooms, poolside lounging, and a bright bonus room.',
            'price': 799000,
            'location': 'Nashville, Tennessee',
            'address': '29 Oaklight Drive',
            'bedrooms': 5,
            'bathrooms': 4,
            'square_feet': 3260,
            'property_type': 'house',
            'status': 'available',
            'listing_date': now - timedelta(days=7),
            'agent_id': agents[0].id,
        },
        {
            'title': 'Cedar Park Starter Home',
            'description': 'Comfortable starter home with updated baths, a shaded backyard, and quick commuter access.',
            'price': 289000,
            'location': 'Raleigh, North Carolina',
            'address': '407 Cedar Park Avenue',
            'bedrooms': 3,
            'bathrooms': 2,
            'square_feet': 1540,
            'property_type': 'house',
            'status': 'available',
            'listing_date': now - timedelta(days=16),
            'agent_id': agents[2].id,
        },
        {
            'title': 'Museum District Loft',
            'description': 'Polished loft with exposed brick, gallery walls, and a walkable urban location.',
            'price': 472000,
            'location': 'Houston, Texas',
            'address': '52 Gallery Square, Unit 11B',
            'bedrooms': 2,
            'bathrooms': 2,
            'square_feet': 1685,
            'property_type': 'apartment',
            'status': 'sold',
            'listing_date': now - timedelta(days=40),
            'agent_id': agents[1].id,
        },
        {
            'title': 'The Magnolia Court Condo',
            'description': 'Elegant corner condo with private study, club amenities, and soft natural light all day.',
            'price': 358000,
            'location': 'Charlotte, North Carolina',
            'address': '315 Magnolia Court, Unit 607',
            'bedrooms': 2,
            'bathrooms': 2,
            'square_feet': 1410,
            'property_type': 'condo',
            'status': 'available',
            'listing_date': now - timedelta(days=6),
            'agent_id': agents[2].id,
        },
    ]

    for payload in sample_properties:
        property_obj = Property(**payload)
        update_property_search_text(property_obj)
        db.session.add(property_obj)

    db.session.commit()


def ensure_ai_chat_schema():
    inspector = inspect(db.engine)
    if 'ai_conversation_messages' not in inspector.get_table_names():
        return

    message_columns = {column['name'] for column in inspector.get_columns('ai_conversation_messages')}
    if 'agent_id' not in message_columns:
        db.session.execute(text('ALTER TABLE ai_conversation_messages ADD COLUMN agent_id VARCHAR(80)'))
    if 'agent_name' not in message_columns:
        db.session.execute(text('ALTER TABLE ai_conversation_messages ADD COLUMN agent_name VARCHAR(120)'))
    db.session.commit()


with app.app_context():
    db.create_all()
    ensure_ai_chat_schema()
    ensure_seed_data()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/overview', methods=['GET'])
def get_overview():
    return jsonify(overview_payload())


@app.route('/api/ai/agents', methods=['GET'])
def get_ai_agents():
    return jsonify({
        'agents': list_ai_personas(),
        'status': ai_runtime_status(),
    })


@app.route('/api/users/<int:user_id>/ai/conversations', methods=['GET'])
def list_ai_conversations(user_id):
    get_user_or_404(user_id)
    query = AIConversation.query.filter_by(user_id=user_id)
    conversations = query.order_by(AIConversation.last_message_at.desc(), AIConversation.updated_at.desc()).all()
    return jsonify({
        'conversations': [serialize_conversation(conversation) for conversation in conversations],
    })


@app.route('/api/users/<int:user_id>/ai/conversations', methods=['POST'])
def create_ai_conversation(user_id):
    get_user_or_404(user_id)
    data = request.get_json(silent=True) or {}
    agent_id = clean_text(data.get('agent_id')) or 'buyer-guide'
    persona = get_ai_persona(agent_id)
    if not persona:
        return api_error('Unknown ai agent selected.', 400)

    title = clean_text(data.get('title')) or default_conversation_title()
    conversation = AIConversation(
        user_id=user_id,
        agent_id=persona['id'],
        title=title,
        last_message_at=datetime.utcnow(),
    )
    db.session.add(conversation)
    db.session.commit()
    return jsonify({
        'conversation': serialize_conversation(conversation),
        'messages': [],
    }), 201


@app.route('/api/users/<int:user_id>/ai/conversations/<int:conversation_id>', methods=['GET'])
def get_ai_conversation(user_id, conversation_id):
    get_user_or_404(user_id)
    conversation = get_conversation_or_404(user_id, conversation_id)
    if not conversation:
        return api_error('Conversation not found.', 404)
    return jsonify({
        'conversation': serialize_conversation(conversation),
        'messages': [serialize_conversation_message(message) for message in conversation.messages],
    })


def extract_request_history(data, conversation):
    if conversation is not None:
        return build_history_from_messages(conversation.messages, limit=8)

    history_messages = []
    for item in (data.get('history') or [])[-8:]:
        if not isinstance(item, dict):
            continue
        role = clean_text(item.get('role'))
        content = clean_text(item.get('content'))
        agent_id = clean_text(item.get('agent_id'))
        agent_name = clean_text(item.get('agent_name'))
        if not agent_name and agent_id:
            persona = get_ai_persona(agent_id)
            agent_name = persona['name'] if persona else None
        if role in {'user', 'assistant'} and content:
            history_messages.append({
                'role': role,
                'content': _history_content_with_agent(role, content, agent_name=agent_name),
                'agent_id': agent_id,
                'agent_name': agent_name,
            })
    return history_messages


def prepare_ai_chat_session(data):
    message = clean_text(data.get('message'))
    if not message:
        raise ValueError('message is required.')

    preferred_agent_id = clean_text(data.get('agent_id')) or 'buyer-guide'
    preferred_persona = get_ai_persona(preferred_agent_id)
    if not preferred_persona:
        raise ValueError('Unknown ai agent selected.')

    user_id = parse_optional_int(data.get('user_id'), 'user_id')
    conversation_id = parse_optional_int(data.get('conversation_id'), 'conversation_id')

    user = None
    conversation = None
    if user_id is not None:
        user = get_user_or_404(user_id)
        if conversation_id is not None:
            conversation = get_conversation_or_404(user.id, conversation_id)
            if not conversation:
                raise LookupError('Conversation not found.')
        else:
            conversation = AIConversation(
                user_id=user.id,
                agent_id=preferred_persona['id'],
                title=conversation_title_from_message(message),
                last_message_at=datetime.utcnow(),
            )
            db.session.add(conversation)
            db.session.flush()

    history_messages = extract_request_history(data, conversation)
    responder_persona = select_chat_persona(
        app.config,
        message,
        preferred_agent_id=preferred_persona['id'],
        history=history_messages,
    )

    return {
        'message': message,
        'persona': responder_persona,
        'preferred_persona': preferred_persona,
        'user': user,
        'conversation': conversation,
        'history_messages': history_messages,
    }


def run_ai_chat_session(chat_session):
    local_inventory = Property.query.order_by(Property.listing_date.desc()).all()
    flow_result, cache_hit = run_agentic_listing_flow_cached(
        persona=chat_session['persona'],
        message=chat_session['message'],
        history=chat_session['history_messages'],
        local_inventory=local_inventory,
        mode='chat',
    )

    recommended = []
    for item in flow_result.get('selected_results') or []:
        payload = dict(item)
        payload['recommendation_reason'] = property_reason_for_persona(chat_session['persona']['id'])
        recommended.append(payload)
    return flow_result, recommended, cache_hit


def persist_ai_chat_session(chat_session, flow_result, recommended):
    conversation = chat_session.get('conversation')
    if conversation is None:
        return None

    if conversation.title == default_conversation_title():
        conversation.title = conversation_title_from_message(chat_session['message'])
    conversation.agent_id = chat_session['preferred_persona']['id']
    conversation.last_message_at = datetime.utcnow()
    db.session.add(AIConversationMessage(
        conversation_id=conversation.id,
        role='user',
        content=chat_session['message'],
        source='user',
        recommended_properties='[]',
    ))
    db.session.add(AIConversationMessage(
        conversation_id=conversation.id,
        role='assistant',
        content=flow_result.get('reply') or '',
        source=flow_result.get('source', 'local_inventory'),
        agent_id=chat_session['persona']['id'],
        agent_name=chat_session['persona']['name'],
        recommended_properties=json.dumps(recommended),
    ))
    db.session.commit()
    return conversation


def build_ai_chat_payload(chat_session, flow_result, recommended, cache_hit=False):
    conversation = chat_session.get('conversation')
    return {
        'message': flow_result.get('reply'),
        'source': flow_result.get('model_source', 'local_fallback'),
        'remote_model': flow_result.get('remote_model'),
        'listings_source': flow_result.get('source', 'local_inventory'),
        'fallback_reason': flow_result.get('fallback_reason'),
        'retrieval_notes': flow_result.get('retrieval_notes') or [],
        'query_route': flow_result.get('query_route'),
        'route_reason': flow_result.get('route_reason'),
        'cache_hit': cache_hit,
        'agent': chat_session['persona'],
        'preferred_agent': chat_session['preferred_persona'],
        'status': ai_runtime_status(),
        'recommended_properties': recommended,
        'conversation': serialize_conversation(conversation) if conversation is not None else None,
    }


def process_ai_chat_payload(data):
    chat_session = prepare_ai_chat_session(data)
    flow_result, recommended, cache_hit = run_ai_chat_session(chat_session)
    conversation = persist_ai_chat_session(chat_session, flow_result, recommended)
    chat_session['conversation'] = conversation
    return build_ai_chat_payload(chat_session, flow_result, recommended, cache_hit=cache_hit)


def chunk_text_for_streaming(text, words_per_chunk=6):
    tokens = re.findall(r'\S+\s*', str(text or ''))
    if not tokens:
        return []

    chunks = []
    for index in range(0, len(tokens), words_per_chunk):
        chunk = ''.join(tokens[index:index + words_per_chunk]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def sse_event(event_name, payload):
    return f'event: {event_name}\ndata: {json.dumps(payload)}\n\n'


@app.route('/api/ai/chat', methods=['POST'])
def ai_chat():
    data = request.get_json(silent=True) or {}
    try:
        payload = process_ai_chat_payload(data)
    except ValueError as exc:
        return api_error(str(exc), 400)
    except LookupError as exc:
        return api_error(str(exc), 404)

    return jsonify(payload)


@app.route('/api/ai/chat/stream', methods=['POST'])
def ai_chat_stream():
    data = request.get_json(silent=True) or {}
    try:
        payload = process_ai_chat_payload(data)
    except ValueError as exc:
        return api_error(str(exc), 400)
    except LookupError as exc:
        return api_error(str(exc), 404)

    message = payload.get('message') or ''
    meta_payload = {key: value for key, value in payload.items() if key != 'message'}

    @stream_with_context
    def generate():
        yield sse_event('meta', meta_payload)
        for chunk in chunk_text_for_streaming(message):
            yield sse_event('delta', {'content': chunk})
            time.sleep(app.config.get('AI_STREAM_CHUNK_DELAY', 0.05))
        yield sse_event('done', payload)

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )


@app.route('/api/properties', methods=['GET'])
def get_properties():
    page = request.args.get('page', default=1, type=int)
    per_page = min(request.args.get('per_page', default=6, type=int), 18)

    try:
        query, applied_filters = apply_property_filters(Property.query, request.args)
    except ValueError as exc:
        return api_error(str(exc), 400)

    properties = query.order_by(Property.listing_date.desc()).paginate(page=page, per_page=per_page, error_out=False)
    return jsonify({
        'properties': [serialize_property(property_obj) for property_obj in properties.items],
        'applied_filters': applied_filters,
        'total': properties.total,
        'page': properties.page,
        'pages': properties.pages,
    })


@app.route('/api/properties/<int:property_id>', methods=['GET'])
def get_property(property_id):
    property_obj = Property.query.get_or_404(property_id)
    return jsonify(serialize_property(property_obj))


@app.route('/api/properties', methods=['POST'])
def create_property():
    data = request.get_json(silent=True) or {}
    if not data:
        return api_error('No data provided.', 400)

    property_obj = Property()
    try:
        assign_property_fields(property_obj, data, partial=False)
        db.session.add(property_obj)
        db.session.commit()
    except ValueError as exc:
        db.session.rollback()
        return api_error(str(exc), 400)
    except IntegrityError:
        db.session.rollback()
        return api_error('Property could not be created because related data was invalid.', 400)

    return jsonify(serialize_property(property_obj)), 201


@app.route('/api/properties/<int:property_id>', methods=['PUT'])
def update_property(property_id):
    property_obj = Property.query.get_or_404(property_id)
    data = request.get_json(silent=True) or {}
    if not data:
        return api_error('No data provided.', 400)

    try:
        assign_property_fields(property_obj, data, partial=True)
        db.session.commit()
    except ValueError as exc:
        db.session.rollback()
        return api_error(str(exc), 400)
    except IntegrityError:
        db.session.rollback()
        return api_error('Property could not be updated because related data was invalid.', 400)

    return jsonify(serialize_property(property_obj))


@app.route('/api/properties/<int:property_id>', methods=['DELETE'])
def delete_property(property_id):
    property_obj = Property.query.get_or_404(property_id)
    db.session.delete(property_obj)
    db.session.commit()
    return jsonify({'message': 'Property deleted successfully.'})


@app.route('/api/users/register', methods=['POST'])
def register_user():
    data = request.get_json(silent=True) or {}
    username = clean_text(data.get('username'))
    email = clean_text(data.get('email'))
    password = clean_text(data.get('password'))

    if not username or not email or not password:
        return api_error('username, email, and password are required.', 400)

    if User.query.filter_by(username=username).first():
        return api_error('Username already exists.', 400)
    if User.query.filter_by(email=email).first():
        return api_error('Email already exists.', 400)

    user_type = clean_text(data.get('user_type')) or 'buyer'
    if user_type not in {'buyer', 'seller', 'agent'}:
        return api_error('user_type must be buyer, seller, or agent.', 400)

    user = User(
        username=username,
        email=email,
        user_type=user_type,
        phone=clean_text(data.get('phone')),
        first_name=clean_text(data.get('first_name')),
        last_name=clean_text(data.get('last_name')),
    )
    user.set_password(password)

    db.session.add(user)
    db.session.commit()
    return jsonify(user.to_dict()), 201


@app.route('/api/users/login', methods=['POST'])
def login_user():
    data = request.get_json(silent=True) or {}
    username = clean_text(data.get('username'))
    password = clean_text(data.get('password'))

    if not username or not password:
        return api_error('Username and password are required.', 400)

    user = User.query.filter_by(username=username).first()
    if not user or not user.check_password(password):
        return api_error('Invalid credentials.', 401)

    return jsonify(user.to_dict())


@app.route('/api/users/<int:user_id>/favorites', methods=['POST'])
def add_favorite(user_id):
    user = User.query.get_or_404(user_id)
    data = request.get_json(silent=True) or {}

    try:
        property_id = parse_optional_int(data.get('property_id'), 'property_id')
    except ValueError as exc:
        return api_error(str(exc), 400)
    if property_id is None:
        return api_error('property_id is required.', 400)

    property_obj = Property.query.get_or_404(property_id)
    existing = Favorite.query.filter_by(user_id=user.id, property_id=property_obj.id).first()
    if existing:
        return api_error('Property already in favorites.', 400)

    favorite = Favorite(user_id=user.id, property_id=property_obj.id)
    db.session.add(favorite)
    db.session.commit()
    return jsonify(favorite.to_dict()), 201


@app.route('/api/users/<int:user_id>/favorites', methods=['GET'])
def get_favorites(user_id):
    user = User.query.get_or_404(user_id)
    results = [
        {
            'favorite': favorite.to_dict(),
            'property': serialize_property(favorite.property) if favorite.property else None,
        }
        for favorite in Favorite.query.filter_by(user_id=user.id).order_by(Favorite.added_at.desc()).all()
    ]
    return jsonify(results)


@app.route('/api/agents', methods=['GET'])
def get_agents():
    agents = Agent.query.order_by(Agent.created_at.desc()).all()
    return jsonify([agent.to_dict() for agent in agents])


@app.route('/api/agents/<int:agent_id>', methods=['GET'])
def get_agent(agent_id):
    agent = Agent.query.get_or_404(agent_id)
    return jsonify(agent.to_dict())


@app.route('/api/agents', methods=['POST'])
def create_agent():
    data = request.get_json(silent=True) or {}
    name = clean_text(data.get('name'))
    email = clean_text(data.get('email'))
    phone = clean_text(data.get('phone'))

    if not name or not email or not phone:
        return api_error('name, email, and phone are required.', 400)
    if Agent.query.filter_by(email=email).first():
        return api_error('An agent with this email already exists.', 400)

    agent = Agent(
        name=name,
        email=email,
        phone=phone,
        license_number=clean_text(data.get('license_number')),
        company=clean_text(data.get('company')),
        bio=clean_text(data.get('bio')),
        image_url=clean_text(data.get('image_url')),
        is_verified=bool(data.get('is_verified', False)),
    )

    db.session.add(agent)
    db.session.commit()
    return jsonify(agent.to_dict()), 201


@app.route('/api/search', methods=['POST'])
def search_properties():
    data = request.get_json(silent=True) or {}
    if not data:
        return api_error('No search data provided.', 400)

    try:
        query, applied_filters = apply_property_filters(Property.query, data)
    except ValueError as exc:
        return api_error(str(exc), 400)

    if not applied_filters:
        return api_error('Provide at least one search term or filter.', 400)

    agent_id = clean_text(data.get('agent_id')) or 'buyer-guide'
    persona = get_ai_persona(agent_id)
    if not persona:
        return api_error('Unknown ai agent selected.', 400)

    try:
        user_id = parse_optional_int(data.get('user_id'), 'user_id')
    except ValueError as exc:
        return api_error(str(exc), 400)

    flow_result, cache_hit = run_agentic_listing_flow_cached(
        persona=persona,
        message=clean_text(data.get('message')) or applied_filters.get('query') or applied_filters.get('location') or '',
        filters=applied_filters,
        local_inventory=query.order_by(Property.listing_date.desc()).all(),
        mode='search',
    )

    results = [dict(item) for item in flow_result.get('selected_results') or []]

    if user_id is not None:
        user = User.query.get_or_404(user_id)
        history = SearchHistory(
            user_id=user.id,
            search_query=applied_filters.get('query') or applied_filters.get('location') or applied_filters.get('property_type') or 'smart search',
            filters=json.dumps(applied_filters),
            results_count=len(results),
        )
        db.session.add(history)
        db.session.commit()

    return jsonify({
        'properties': results,
        'count': len(results),
        'applied_filters': applied_filters,
        'source': flow_result.get('source', 'local_inventory'),
        'fallback_reason': flow_result.get('fallback_reason'),
        'query_route': flow_result.get('query_route'),
        'route_reason': flow_result.get('route_reason'),
        'cache_hit': cache_hit,
        'agent_summary': flow_result.get('reply'),
    })


@app.route('/api/recommendations', methods=['GET'])
def get_recommendations():
    user_id = request.args.get('user_id', type=int)
    recommended = []
    seen_ids = set()

    if user_id:
        User.query.get_or_404(user_id)
        history_items = SearchHistory.query.filter_by(user_id=user_id).order_by(SearchHistory.searched_at.desc()).limit(5).all()
        for item in history_items:
            try:
                filters = json.loads(item.filters or '{}')
                query, _ = apply_property_filters(Property.query, filters)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue

            for property_obj in rank_properties(query.all(), limit=3, exclude_ids=seen_ids):
                recommended.append(serialize_property(property_obj, reason='Inspired by your recent searches'))
                seen_ids.add(property_obj.id)

    if len(recommended) < 8:
        for property_obj in top_properties(limit=8, exclude_ids=seen_ids):
            recommended.append(serialize_property(property_obj, reason='Trending among active buyers'))
            seen_ids.add(property_obj.id)

    return jsonify({'recommendations': recommended[:8]})


@app.route('/api/recommendations/favorites', methods=['GET'])
def get_recommendations_from_favorites():
    user_id = request.args.get('user_id', type=int)
    if not user_id:
        return api_error('user_id is required.', 400)

    User.query.get_or_404(user_id)
    favorites = Favorite.query.filter_by(user_id=user_id).all()
    if not favorites:
        return jsonify({'recommendations': []})

    favorite_properties = [favorite.property for favorite in favorites if favorite.property]
    if not favorite_properties:
        return jsonify({'recommendations': []})
    favorite_ids = {property_obj.id for property_obj in favorite_properties}

    type_counts = {}
    bedroom_counts = {}
    price_ranges = {'budget': 0, 'mid': 0, 'premium': 0}
    for property_obj in favorite_properties:
        type_counts[property_obj.property_type] = type_counts.get(property_obj.property_type, 0) + 1
        bedroom_counts[property_obj.bedrooms] = bedroom_counts.get(property_obj.bedrooms, 0) + 1
        if property_obj.price <= 300000:
            price_ranges['budget'] += 1
        elif property_obj.price <= 700000:
            price_ranges['mid'] += 1
        else:
            price_ranges['premium'] += 1

    preferred_type = max(type_counts, key=type_counts.get)
    preferred_bedrooms = max(bedroom_counts, key=bedroom_counts.get)
    preferred_price_range = max(price_ranges, key=price_ranges.get)

    recommendations = []
    for property_obj in Property.query.all():
        if property_obj.id in favorite_ids:
            continue

        score = property_obj.get_ai_score()
        if property_obj.property_type == preferred_type:
            score += 20
        if property_obj.bedrooms == preferred_bedrooms:
            score += 15
        if preferred_price_range == 'budget' and property_obj.price <= 300000:
            score += 15
        if preferred_price_range == 'mid' and 300000 < property_obj.price <= 700000:
            score += 15
        if preferred_price_range == 'premium' and property_obj.price > 700000:
            score += 15

        payload = serialize_property(property_obj, reason='Aligned with your saved favorites')
        payload['ai_score'] = min(score, 100)
        recommendations.append(payload)

    recommendations.sort(key=lambda item: item['ai_score'], reverse=True)
    return jsonify({'recommendations': recommendations[:10]})


@app.errorhandler(400)
def bad_request(error):
    if wants_json():
        return api_error('Bad request.', 400)
    return render_template('index.html'), 400


@app.errorhandler(404)
def not_found(error):
    if wants_json():
        return api_error('Not found.', 404)
    return render_template('index.html'), 404


@app.errorhandler(405)
def method_not_allowed(error):
    if wants_json():
        return api_error('Method not allowed.', 405)
    return render_template('index.html'), 405


@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    if wants_json():
        return api_error('Internal server error.', 500)
    return render_template('index.html'), 500


if __name__ == '__main__':
    app.run(
        debug=os.environ.get('FLASK_DEBUG') == '1',
        host='0.0.0.0',
        port=5000,
    )
