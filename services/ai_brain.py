import difflib
import json
import re
from collections import defaultdict
from datetime import datetime
from urllib import error, request

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


AI_PERSONAS = {
    'buyer-guide': {
        'id': 'buyer-guide',
        'name': 'Buyer Guide',
        'headline': 'Clarifies fit, tradeoffs, and next best homes for owner-occupiers.',
        'tone': 'warm, decisive, and practical',
        'goal': 'Help buyers narrow down homes based on space, price, and lifestyle fit.',
        'quick_prompts': [
            'Find me a family-friendly home under $550k.',
            'Which listing is best for a first-time buyer?',
            'Compare two and three bedroom options for me.',
        ],
    },
    'investment-scout': {
        'id': 'investment-scout',
        'name': 'Investment Scout',
        'headline': 'Focuses on upside, pricing spread, and portfolio-friendly listings.',
        'tone': 'analytical, concise, and ROI-minded',
        'goal': 'Help investors identify listings with strong price-to-space balance and resale potential.',
        'quick_prompts': [
            'Show me the best investment properties in the current inventory.',
            'Which condo looks strongest for resale value?',
            'What listings balance price and square footage best?',
        ],
    },
    'neighborhood-navigator': {
        'id': 'neighborhood-navigator',
        'name': 'Neighborhood Navigator',
        'headline': 'Frames locations, lifestyle mood, and the feel of each area.',
        'tone': 'observant, reassuring, and lifestyle-focused',
        'goal': 'Help buyers compare areas and understand which listing matches their day-to-day lifestyle.',
        'quick_prompts': [
            'Which city feels best for a calmer lifestyle?',
            'Find me a bright home in a walkable-feeling area.',
            'What areas in the current listings feel most urban versus residential?',
        ],
    },
}

NUMBER_WORDS = {
    'studio': 0,
    'zero': 0,
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
}

DEFAULT_HF_FALLBACK_MODELS = [
    'google/gemma-2-2b-it',
    'Qwen/Qwen2.5-7B-Instruct-1M',
    'deepseek-ai/DeepSeek-R1',
]


def list_ai_personas():
    return list(AI_PERSONAS.values())


def get_ai_persona(agent_id):
    return AI_PERSONAS.get(agent_id)


def normalize_agent_intent_text(message):
    text = str(message or '').strip().lower()
    if not text:
        return ''

    replacements = {
        'neighbourhood': 'neighborhood',
        'neighbourhoods': 'neighborhoods',
        'neighbour hood': 'neighborhood',
        'neighbor hood': 'neighborhood',
        'hows': 'how is',
        "how's": 'how is',
    }
    for source, target in replacements.items():
        text = text.replace(source, target)

    text = re.sub(r'\bneighbour\s+hoods?\b', lambda match: 'neighborhoods' if match.group(0).endswith('hoods') else 'neighborhood', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


AGENT_ROUTING_RULES = {
    'buyer-guide': {
        'aliases': {'buyer', 'buyer guide', 'buying', 'owner occupier', 'owner-occupier', 'starter home'},
        'keywords': {'buy', 'buyer', 'buyers', 'family', 'first', 'starter', 'owner', 'livability', 'mortgage'},
    },
    'investment-scout': {
        'aliases': {'investment scout', 'investment', 'investor', 'analyst', 'analysis'},
        'keywords': {'invest', 'investment', 'investor', 'roi', 'yield', 'rental', 'rent', 'income', 'cash', 'flow', 'returns', 'cap', 'rate', 'upside', 'portfolio'},
    },
    'neighborhood-navigator': {
        'aliases': {'neighborhood navigator', 'navigator', 'neighborhood', 'neighbourhood', 'area specialist'},
        'keywords': {'neighborhood', 'neighborhoods', 'neighbourhood', 'neighbourhoods', 'area', 'areas', 'walkable', 'walkability', 'commute', 'lifestyle', 'district', 'school', 'suburb', 'urban'},
    },
}


def _persona_name_variants(persona):
    name = str(persona.get('name') or '').strip().lower()
    variants = {name, persona['id']}
    variants.add(name.replace(' ', '-'))
    variants.add(name.replace(' ', ''))
    return {item for item in variants if item}


def _last_assistant_agent_from_history(history):
    for item in reversed(list(history or [])):
        if not isinstance(item, dict):
            continue
        if str(item.get('role') or '').strip().lower() != 'assistant':
            continue
        agent_id = str(item.get('agent_id') or '').strip()
        if agent_id in AI_PERSONAS:
            return agent_id
    return None


def _heuristic_chat_persona(message, preferred_agent_id='buyer-guide', history=None):
    lowered = normalize_agent_intent_text(message)
    preferred = get_ai_persona(preferred_agent_id) or get_ai_persona('buyer-guide')
    if not lowered:
        return preferred

    explicit_match = None
    explicit_score = -1
    scored = []
    for persona_id, persona in AI_PERSONAS.items():
        rules = AGENT_ROUTING_RULES.get(persona_id, {})
        aliases = set(rules.get('aliases') or set()) | _persona_name_variants(persona)
        alias_hits = sum(1 for alias in aliases if alias and alias in lowered)
        keyword_hits = sum(1 for keyword in rules.get('keywords') or set() if keyword in lowered)
        scored.append((persona_id, alias_hits, keyword_hits))
        if alias_hits > explicit_score:
            explicit_match = persona_id
            explicit_score = alias_hits

    if explicit_score > 0:
        return get_ai_persona(explicit_match)

    best_persona_id = None
    best_keyword_score = 0
    for persona_id, _alias_hits, keyword_hits in scored:
        if keyword_hits > best_keyword_score:
            best_persona_id = persona_id
            best_keyword_score = keyword_hits

    if best_persona_id and best_keyword_score > 0:
        return get_ai_persona(best_persona_id)

    last_agent_id = _last_assistant_agent_from_history(history)
    if last_agent_id:
        return get_ai_persona(last_agent_id)

    return preferred


def _coerce_agent_tool_call(tool_call):
    payload = _coerce_tool_call_args(tool_call)
    agent_id = str(payload.get('agent_id') or '').strip()
    if agent_id in AI_PERSONAS:
        return agent_id
    return None


def select_chat_persona_with_tools(config, message, preferred_agent_id='buyer-guide', history=None):
    if not config or not config.get('LLAMA_CPP_ENABLED') or not str(message or '').strip():
        return None

    llm = ChatOpenAI(
        model=str(config.get('LLAMA_CPP_MODEL') or 'llama.cpp'),
        base_url=_llama_cpp_base_url(config),
        api_key='llama-cpp',
        temperature=0,
        timeout=config.get('AI_REQUEST_TIMEOUT', 20),
        max_retries=0,
        disable_streaming='tool_calling',
    )

    preferred = get_ai_persona(preferred_agent_id) or get_ai_persona('buyer-guide')
    history_lines = []
    for item in list(history or [])[-6:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get('role') or '').strip().lower()
        content = str(item.get('content') or '').strip()
        agent_name = str(item.get('agent_name') or '').strip()
        if role in {'user', 'assistant'} and content:
            prefix = f'{role}: '
            if role == 'assistant' and agent_name:
                prefix = f'assistant ({agent_name}): '
            history_lines.append(prefix + content)

    persona_lines = []
    for persona in list_ai_personas():
        persona_lines.append(f"- {persona['id']}: {persona['headline']}")

    prompt_parts = [
        'Available agents:\n' + '\n'.join(persona_lines),
        f'Preferred agent: {preferred["id"]} ({preferred["name"]})',
    ]
    if history_lines:
        prompt_parts.append('Recent shared conversation:\n' + '\n'.join(history_lines))
    prompt_parts.append('Current user message:\n' + normalize_agent_intent_text(message))

    bound_model = llm.bind_tools([handoff_to_agent], tool_choice='required')
    response = bound_model.invoke([
        SystemMessage(content=(
            'Choose which specialist should answer the current turn. '
            'Use the recent shared conversation and the current message. '
            'Prefer Neighborhood Navigator for surroundings, neighborhood feel, walkability, commute, and lifestyle questions. '
            'Prefer Investment Scout for rent, ROI, yield, upside, and portfolio questions. '
            'Prefer Buyer Guide for affordability, family fit, first-time buyer questions, and owner-occupier tradeoffs. '
            'Always call the tool exactly once.'
        )),
        HumanMessage(content='\n\n'.join(prompt_parts)),
    ])

    tool_calls = getattr(response, 'tool_calls', None) or response.additional_kwargs.get('tool_calls', [])
    if not tool_calls:
        return None

    agent_id = _coerce_agent_tool_call(tool_calls[0])
    if not agent_id:
        return None
    return get_ai_persona(agent_id)


def select_chat_persona(config, message, preferred_agent_id='buyer-guide', history=None):
    try:
        llm_choice = select_chat_persona_with_tools(config, message, preferred_agent_id=preferred_agent_id, history=history)
    except Exception:
        llm_choice = None
    if llm_choice:
        return llm_choice
    return _heuristic_chat_persona(message, preferred_agent_id=preferred_agent_id, history=history)


def listing_value(listing, field_name, default=None):
    if isinstance(listing, dict):
        return listing.get(field_name, default)
    return getattr(listing, field_name, default)


def listing_ai_score(listing):
    if isinstance(listing, dict):
        return float(listing.get('ai_score') or 0)
    return float(listing.get_ai_score())


def listing_title(listing):
    return str(listing_value(listing, 'title', 'Untitled listing'))


def listing_location(listing):
    return str(listing_value(listing, 'location', 'Unknown location'))


def listing_price(listing):
    return float(listing_value(listing, 'price', 0) or 0)


def listing_bedrooms(listing):
    return int(listing_value(listing, 'bedrooms', 0) or 0)


def listing_bathrooms(listing):
    return int(listing_value(listing, 'bathrooms', 0) or 0)


def listing_square_feet(listing):
    return int(listing_value(listing, 'square_feet', 0) or 0)


def listing_property_type(listing):
    return str(listing_value(listing, 'property_type', 'house') or 'house')


def listing_status(listing):
    return str(listing_value(listing, 'status', 'available') or 'available')


def listing_listing_date(listing):
    value = listing_value(listing, 'listing_date')
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace('Z', '+00:00')).replace(tzinfo=None)
        except ValueError:
            return None
    return None


def build_market_snapshot(properties):
    if not properties:
        return 'No listings are currently available in the active search scope.'

    by_type = defaultdict(int)
    by_location = defaultdict(int)
    prices = [listing_price(property_obj) for property_obj in properties if listing_price(property_obj) > 0]
    average_price = sum(prices) / len(prices) if prices else 0
    for property_obj in properties:
        by_type[listing_property_type(property_obj)] += 1
        by_location[listing_location(property_obj)] += 1

    top_types = ', '.join(f'{kind}: {count}' for kind, count in sorted(by_type.items()))
    top_locations = ', '.join(
        f'{location}: {count}'
        for location, count in sorted(by_location.items(), key=lambda item: item[1], reverse=True)[:4]
    )
    return (
        f'Inventory count: {len(properties)}. '
        f'Average price: ${average_price:,.0f}. '
        f'Property types: {top_types}. '
        f'Locations represented: {top_locations}.'
    )


def property_fact_line(property_obj):
    return (
        f'{listing_title(property_obj)} in {listing_location(property_obj)} | '
        f'${listing_price(property_obj):,.0f} | {listing_bedrooms(property_obj)} bed / {listing_bathrooms(property_obj)} bath | '
        f'{listing_square_feet(property_obj):,} sq ft | {listing_property_type(property_obj)} | '
        f'status: {listing_status(property_obj)} | ai score: {listing_ai_score(property_obj):.0f}/100'
    )


def extract_budget(message):
    normalized = str(message or '').lower().replace(',', '')
    matches = re.findall(r'\$?\s*(\d+(?:\.\d+)?)\s*([km]?)', normalized)
    budgets = []
    for raw_amount, suffix in matches:
        try:
            amount = float(raw_amount)
        except ValueError:
            continue
        if suffix == 'k':
            amount *= 1_000
        elif suffix == 'm':
            amount *= 1_000_000
        if amount >= 50_000:
            budgets.append(amount)
    return budgets


def extract_bedrooms(message):
    match = re.search(r'(\d+)\s*(?:bed|beds|bedroom|bedrooms|br)\b', str(message or '').lower())
    if not match:
        return None
    return int(match.group(1))


def _parse_number_token(token):
    token = str(token or '').strip().lower()
    if token.isdigit():
        return int(token)
    return NUMBER_WORDS.get(token)


def extract_bedroom_targets(message):
    lowered = str(message or '').lower()
    targets = []

    exact_matches = re.findall(r'(\d+|studio|zero|one|two|three|four|five|six)\s*(?:bed|beds|bedroom|bedrooms|br)\b', lowered)
    for match in exact_matches:
        parsed = _parse_number_token(match)
        if parsed is not None:
            targets.append(parsed)

    grouped_matches = re.findall(
        r'((?:\d+|studio|zero|one|two|three|four|five|six)(?:\s*(?:,|and|or)\s*(?:\d+|studio|zero|one|two|three|four|five|six))+)[ ]+bed(?:room|rooms)?\b',
        lowered,
    )
    for group in grouped_matches:
        for token in re.split(r'\s*(?:,|and|or)\s*', group):
            parsed = _parse_number_token(token)
            if parsed is not None:
                targets.append(parsed)

    seen = set()
    ordered = []
    for target in targets:
        if target in seen:
            continue
        seen.add(target)
        ordered.append(target)
    return ordered


def is_comparison_request(message):
    lowered = str(message or '').lower()
    return any(keyword in lowered for keyword in {'compare', 'versus', 'vs', 'difference', 'options'})


def extract_property_type(message):
    lowered = str(message or '').lower()
    if 'townhouse' in lowered or 'townhome' in lowered:
        return 'townhouse'
    if 'condo' in lowered or 'condominium' in lowered:
        return 'condo'
    if 'apartment' in lowered or 'flat' in lowered or 'loft' in lowered:
        return 'apartment'
    if 'house' in lowered or 'home' in lowered:
        return 'house'
    return None


US_STATE_CODES = {
    'alabama': 'AL', 'alaska': 'AK', 'arizona': 'AZ', 'arkansas': 'AR', 'california': 'CA',
    'colorado': 'CO', 'connecticut': 'CT', 'delaware': 'DE', 'florida': 'FL', 'georgia': 'GA',
    'hawaii': 'HI', 'idaho': 'ID', 'illinois': 'IL', 'indiana': 'IN', 'iowa': 'IA', 'kansas': 'KS',
    'kentucky': 'KY', 'louisiana': 'LA', 'maine': 'ME', 'maryland': 'MD', 'massachusetts': 'MA',
    'michigan': 'MI', 'minnesota': 'MN', 'mississippi': 'MS', 'missouri': 'MO', 'montana': 'MT',
    'nebraska': 'NE', 'nevada': 'NV', 'new hampshire': 'NH', 'new jersey': 'NJ', 'new mexico': 'NM',
    'new york': 'NY', 'north carolina': 'NC', 'north dakota': 'ND', 'ohio': 'OH', 'oklahoma': 'OK',
    'oregon': 'OR', 'pennsylvania': 'PA', 'rhode island': 'RI', 'south carolina': 'SC',
    'south dakota': 'SD', 'tennessee': 'TN', 'texas': 'TX', 'utah': 'UT', 'vermont': 'VT',
    'virginia': 'VA', 'washington': 'WA', 'west virginia': 'WV', 'wisconsin': 'WI', 'wyoming': 'WY',
    'district of columbia': 'DC',
}

US_STATE_NAMES = list(US_STATE_CODES.keys())

LOCATION_ALIASES = {
    'newyork': 'New York',
    'new york': 'New York',
    'nyc': 'New York',
    'newjersey': 'New Jersey',
    'new jersy': 'New Jersey',
    'new jersey': 'New Jersey',
    'jersey': 'New Jersey',
    'losangeles': 'Los Angeles',
    'los angeles': 'Los Angeles',
    'sanfrancisco': 'San Francisco',
    'san francisco': 'San Francisco',
}


@tool('set_search_filters')
def set_search_filters(
    location: str = '',
    property_type: str = '',
    min_bedrooms: int | None = None,
    max_price: int | None = None,
):
    """Capture structured real-estate search filters for the current user request."""
    return json.dumps({
        'location': location,
        'property_type': property_type,
        'min_bedrooms': min_bedrooms,
        'max_price': max_price,
    })


@tool('handoff_to_agent')
def handoff_to_agent(agent_id: str, reason: str = ''):
    """Choose the best AI specialist to answer the current user turn."""
    return json.dumps({
        'agent_id': agent_id,
        'reason': reason,
    })


def _normalize_state_name(value):
    normalized = str(value or '').strip().lower()
    if not normalized:
        return None
    normalized = re.sub(r'\s+', ' ', normalized)
    if normalized in US_STATE_CODES:
        return ' '.join(word.capitalize() for word in normalized.split())
    close = difflib.get_close_matches(normalized, US_STATE_NAMES, n=1, cutoff=0.82)
    if close:
        return ' '.join(word.capitalize() for word in close[0].split())
    return None


def _normalize_location_fragment(fragment):
    value = str(fragment or '').strip().lower()
    if not value:
        return None

    value = re.sub(r'[^a-z0-9\s,-]', ' ', value)
    value = re.sub(r'\s+', ' ', value).strip(' ,.-')
    if not value:
        return None

    alias = LOCATION_ALIASES.get(value)
    if alias:
        return alias

    compact = value.replace(' ', '')
    alias = LOCATION_ALIASES.get(compact)
    if alias:
        return alias

    state_name = _normalize_state_name(value)
    if state_name:
        return state_name

    if ',' in value:
        parts = [_normalize_location_fragment(part) or part.strip().title() for part in value.split(',') if part.strip()]
        return ', '.join(parts)

    return ' '.join(word.capitalize() for word in value.split())


def _trim_location_candidate(fragment):
    value = str(fragment or '').strip().lower()
    if not value:
        return None

    value = re.sub(r'[\n\r\t]+', ' ', value)
    value = re.sub(r'\s+', ' ', value).strip(' ,.-')
    if not value:
        return None

    stop_pattern = (
        r'\b(?:'
        r'under|below|about|with|that|which|having|for|priced|price|budget|'
        r'max|min|minimum|at least|at most|'
        r'\d+\s*(?:k|m|dollars?)|'
        r'\d+\s*(?:bed|beds|bedroom|bedrooms|bath|baths|bathroom|bathrooms)|'
        r'house|home|homes|condo|condos|apartment|apartments|townhouse|townhouses|'
        r'townhome|townhomes|listing|listings|property|properties'
        r')\b'
    )
    match = re.search(stop_pattern, value)
    if match:
        value = value[:match.start()].strip(' ,.-')

    value = re.sub(r'^(?:me|something|anything)\s+', '', value).strip(' ,.-')
    return value or None


def _extract_location_parts(text):
    lowered = str(text or '').lower()
    if not lowered:
        return []

    matches = list(re.finditer(r'\b(?:in|near|around|within|from)\b', lowered))
    if not matches:
        return []

    parts = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(lowered)
        candidate = _trim_location_candidate(lowered[start:end])
        if candidate:
            parts.append(candidate)
    return parts


def extract_location_hint(message):
    text = str(message or '').strip()
    if not text:
        return None

    lowered = re.sub(r'\s+', ' ', text.lower()).strip()

    parts = [_normalize_location_fragment(part) for part in _extract_location_parts(lowered)]
    parts = [part for part in parts if part]
    if len(parts) >= 2:
        return ', '.join(parts[:2])
    if parts:
        return parts[0]

    direct_patterns = [
        r'\b(?:move to|buy in|homes in|find something in|find me something in|search in)\s+([a-z][a-z\s,-]{1,60})',
        r'\b(?:in|near|around|within|from)\s+([a-z][a-z\s,-]{1,60})',
    ]
    for pattern in direct_patterns:
        match = re.search(pattern, lowered)
        if not match:
            continue
        normalized = _normalize_location_fragment(_trim_location_candidate(match.group(1)))
        if normalized:
            return normalized
    return None


def _llama_cpp_base_url(config):
    url = str(config.get('LLAMA_CPP_URL') or '').rstrip('/')
    if url.endswith('/chat/completions'):
        return url.rsplit('/chat/completions', 1)[0]
    return url


def _coerce_tool_call_args(tool_call):
    if not tool_call:
        return {}

    if isinstance(tool_call, dict):
        raw_args = tool_call.get('args') or tool_call.get('arguments') or {}
    else:
        raw_args = getattr(tool_call, 'args', None) or getattr(tool_call, 'arguments', None) or {}

    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    return raw_args if isinstance(raw_args, dict) else {}


def infer_query_arguments_with_tools(config, message, history=None):
    if not config or not config.get('LLAMA_CPP_ENABLED') or not str(message or '').strip():
        return {}

    llm = ChatOpenAI(
        model=str(config.get('LLAMA_CPP_MODEL') or 'llama.cpp'),
        base_url=_llama_cpp_base_url(config),
        api_key='llama-cpp',
        temperature=0,
        timeout=config.get('AI_REQUEST_TIMEOUT', 20),
        max_retries=0,
        disable_streaming='tool_calling',
    )
    history_lines = []
    for item in list(history or [])[-4:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get('role') or '').strip().lower()
        content = str(item.get('content') or '').strip()
        if role in {'user', 'assistant'} and content:
            history_lines.append(f'{role}: {content}')

    prompt_parts = []
    if history_lines:
        prompt_parts.append('Recent conversation context:\n' + '\n'.join(history_lines))
    prompt_parts.append('Current user request:\n' + str(message or ''))

    bound_model = llm.bind_tools([set_search_filters], tool_choice='required')
    response = bound_model.invoke([
        SystemMessage(content=(
            'You extract real-estate search arguments. '
            'Always call the provided tool exactly once with the best available listing filters. '
            'Use recent conversation context to carry forward location or other constraints when the current message is a follow-up. '
            'Infer US city/state spelling when reasonable. Leave fields empty or null when unknown.'
        )),
        HumanMessage(content='\n\n'.join(prompt_parts)),
    ])
    tool_calls = getattr(response, 'tool_calls', None) or response.additional_kwargs.get('tool_calls', [])
    if not tool_calls:
        return {}

    args = _coerce_tool_call_args(tool_calls[0])
    filters = {}
    location = _normalize_location_fragment(args.get('location'))
    if location:
        filters['location'] = location

    property_type = extract_property_type(args.get('property_type')) if args.get('property_type') else None
    if property_type:
        filters['property_type'] = property_type

    try:
        if args.get('min_bedrooms') not in (None, ''):
            filters['min_bedrooms'] = int(args['min_bedrooms'])
    except (TypeError, ValueError):
        pass

    try:
        if args.get('max_price') not in (None, ''):
            filters['max_price'] = int(float(args['max_price']))
    except (TypeError, ValueError):
        pass

    return filters


def _infer_context_filters_from_history(history):
    filters = {}
    for item in reversed(list(history or [])):
        if not isinstance(item, dict) or str(item.get('role') or '').strip().lower() != 'user':
            continue

        content = str(item.get('content') or '').strip()
        if not content:
            continue

        if not filters.get('location'):
            location = extract_location_hint(content)
            if location:
                filters['location'] = location
        if not filters.get('max_price'):
            budgets = extract_budget(content)
            if budgets:
                filters['max_price'] = int(max(budgets))
        if not filters.get('min_bedrooms'):
            bedrooms = extract_bedrooms(content)
            if bedrooms is not None:
                filters['min_bedrooms'] = bedrooms
        if not filters.get('property_type'):
            property_type = extract_property_type(content)
            if property_type:
                filters['property_type'] = property_type

        if all(filters.get(key) is not None for key in ('location', 'max_price', 'min_bedrooms', 'property_type')):
            break

    return filters


def _extract_json_object(text):
    value = str(text or '').strip()
    if not value:
        return None
    fenced_match = re.search(r'\{.*\}', value, re.DOTALL)
    candidate = fenced_match.group(0) if fenced_match else value
    try:
        return json.loads(candidate)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def infer_query_arguments(config, message, existing_filters=None, history=None):
    filters = {key: value for key, value in dict(existing_filters or {}).items() if value not in (None, '', [])}
    locked_keys = set(filters)

    history_filters = _infer_context_filters_from_history(history)
    for key, value in history_filters.items():
        if value not in (None, '', []) and not filters.get(key):
            filters[key] = value

    try:
        tool_filters = infer_query_arguments_with_tools(config, message, history=history)
    except Exception:
        tool_filters = {}

    for key, value in tool_filters.items():
        if value not in (None, '', []) and key not in locked_keys:
            filters[key] = value

    location_hint = extract_location_hint(message)
    if location_hint and 'location' not in locked_keys:
        filters['location'] = location_hint

    budgets = extract_budget(message)
    if budgets and 'max_price' not in locked_keys:
        filters['max_price'] = int(max(budgets))

    bedrooms = extract_bedrooms(message)
    if bedrooms is not None and 'min_bedrooms' not in locked_keys:
        filters['min_bedrooms'] = bedrooms

    property_type = extract_property_type(message)
    if property_type and 'property_type' not in locked_keys:
        filters['property_type'] = property_type

    if filters.get('location') or not config or not config.get('LLAMA_CPP_ENABLED'):
        return filters

    try:
        parser_messages = [
            {
                'role': 'system',
                'content': (
                    'Extract real-estate search arguments from the user request. '
                    'Return only valid JSON with keys: location, property_type, min_bedrooms, max_price. '
                    'Use null for unknown values. Normalize location to "City" or "City, State" or "State".'
                ),
            },
            {'role': 'user', 'content': str(message or '')},
        ]
        parser_result = call_llama_cpp_chat(config, parser_messages)
        parsed = _extract_json_object(parser_result.get('content')) or {}
    except RuntimeError:
        return filters

    location = _normalize_location_fragment(parsed.get('location')) if isinstance(parsed, dict) else None
    if location and not filters.get('location'):
        filters['location'] = location

    property_type = extract_property_type(parsed.get('property_type')) if isinstance(parsed, dict) else None
    if property_type and not filters.get('property_type'):
        filters['property_type'] = property_type

    if isinstance(parsed, dict):
        try:
            if parsed.get('min_bedrooms') not in (None, '') and not filters.get('min_bedrooms'):
                filters['min_bedrooms'] = int(parsed['min_bedrooms'])
        except (TypeError, ValueError):
            pass
        try:
            if parsed.get('max_price') not in (None, '') and not filters.get('max_price'):
                filters['max_price'] = int(float(parsed['max_price']))
        except (TypeError, ValueError):
            pass

    return filters


REAL_ESTATE_KEYWORDS = {
    'home', 'house', 'listing', 'listings', 'property', 'properties', 'condo', 'apartment',
    'townhouse', 'townhome', 'bedroom', 'bedrooms', 'bathroom', 'bathrooms', 'budget',
    'price', 'prices', 'buy', 'buyer', 'buyers', 'invest', 'investment', 'neighborhood',
    'neighborhoods', 'neighbourhood', 'neighbourhoods', 'city', 'cities', 'rent', 'resale', 'upside', 'square', 'sq', 'feet',
}

GREETING_PATTERNS = [
    r'^\s*(hi|hello|hey|yo|hola)\b[!.?\s]*$',
    r'^\s*(good morning|good afternoon|good evening)\b[!.?\s]*$',
    r"^\s*(how are you|how are you doing|how's it going|whats up|what's up)\b[!.?\s]*$",
    r'^\s*(thanks|thank you)\b[!.?\s]*$',
]


def is_smalltalk_message(message):
    text = normalize_agent_intent_text(message)
    if not text:
        return False

    if any(re.match(pattern, text) for pattern in GREETING_PATTERNS):
        return True

    tokens = set(re.findall(r'[a-z]+', text))
    if not tokens:
        return False

    if tokens & REAL_ESTATE_KEYWORDS:
        return False

    smalltalk_tokens = {
        'hi', 'hello', 'hey', 'thanks', 'thank', 'you', 'yo',
        'how', 'are', 'doing', 'good', 'morning', 'afternoon',
        'evening', 'whats', 'what', 'up', 'going', 'it',
    }
    return len(tokens) <= 4 and tokens <= smalltalk_tokens


def build_smalltalk_reply(persona, user_message):
    lowered = normalize_agent_intent_text(user_message)
    if 'thank' in lowered:
        return "You're welcome. I can help compare listings, narrow a budget, or find the best next homes to review when you're ready."
    if 'how are you' in lowered or "how's it going" in lowered or 'whats up' in lowered or "what's up" in lowered:
        return (
            f"I'm doing well and ready to help. Ask {persona['name'].lower()} about a city, budget, bedrooms, "
            "or property type and I'll help you narrow the best matches."
        )
    return (
        f"Hello. I'm {persona['name']}, and I can help you compare homes, neighborhoods, budgets, and best-fit "
        "listings. Tell me a city, budget, or home type to get started."
    )


ADVISORY_PATTERNS = [
    'first-time buyer',
    'first time buyer',
    'what should i',
    'what matters most',
    'tips for',
    'advice for',
    'how do i',
    'how should i',
    'should i buy',
    'what do you recommend',
    'how is the neighborhood',
    'how is the neighbourhood',
    'tell me about the neighborhood',
    'rental income',
    'rental yield',
    'cash flow',
    'cap rate',
    'roi',
    'return on investment',
]

LISTING_TOOL_HINTS = {
    'listing', 'listings', 'property', 'properties', 'home', 'homes', 'house', 'houses',
    'condo', 'condos', 'apartment', 'apartments', 'townhouse', 'townhouses',
    'budget', 'bedroom', 'bedrooms', 'bathroom', 'bathrooms', 'price', 'prices',
    'city', 'neighborhood', 'neighborhoods', 'neighbourhood', 'neighbourhoods', 'compare', 'search', 'show', 'find',
    'which', 'best', 'under', 'over', 'within', 'fit', 'options',
    'rent', 'rental', 'income', 'yield', 'roi', 'cash', 'flow', 'cap', 'rate', 'returns',
}


def route_agent_query(message, filters=None, mode='chat'):
    normalized_filters = {key: value for key, value in dict(filters or {}).items() if value not in (None, '', [])}
    if mode == 'search':
        return {
            'query_route': 'tool_search',
            'use_tools': True,
            'route_reason': 'Explicit search mode requested property retrieval.',
        }

    if normalized_filters:
        return {
            'query_route': 'tool_search',
            'use_tools': True,
            'route_reason': 'Structured search filters were provided.',
        }

    if is_smalltalk_message(message):
        return {
            'query_route': 'smalltalk',
            'use_tools': False,
            'route_reason': 'Greeting or casual conversation detected.',
        }

    lowered = normalize_agent_intent_text(message)
    tokens = set(re.findall(r'[a-z]+', lowered))
    has_structured_hint = any([
        extract_location_hint(message),
        extract_budget(message),
        extract_bedrooms(message) is not None,
        extract_property_type(message),
        extract_bedroom_targets(message),
    ])
    has_tool_language = bool(tokens & LISTING_TOOL_HINTS)
    has_advisory_language = any(pattern in lowered for pattern in ADVISORY_PATTERNS)
    neighborhood_advisory = ('neighborhood' in lowered or 'neighbourhood' in lowered) and not has_structured_hint

    if neighborhood_advisory:
        return {
            'query_route': 'advisory',
            'use_tools': False,
            'route_reason': 'Neighborhood guidance was requested without a specific listing or city filter.',
        }

    if has_structured_hint or has_tool_language:
        return {
            'query_route': 'tool_search',
            'use_tools': True,
            'route_reason': 'The message includes property-search intent or listing constraints.',
        }

    if has_advisory_language:
        return {
            'query_route': 'advisory',
            'use_tools': False,
            'route_reason': 'General real-estate advice requested without listing constraints.',
        }

    return {
        'query_route': 'conversation',
        'use_tools': False,
        'route_reason': 'Conversational message detected, so listing tools were skipped.',
    }


def build_no_tool_reply(persona, user_message, query_route):
    lowered = normalize_agent_intent_text(user_message)
    if query_route == 'advisory':
        if persona['id'] == 'investment-scout':
            if any(term in lowered for term in {'rental income', 'rental yield', 'cash flow', 'cap rate', 'roi'}):
                return (
                    'I can help estimate rental upside, but I need a specific city, listing, or budget range to ground it. '
                    'Share a location or ask about one of the current listings and I will break down likely rent potential, price fit, and tradeoffs.'
                )
            return (
                'For investment analysis, start with price-to-rent fit, vacancy resilience, renovation exposure, and resale liquidity. '
                'If you share a city, budget, or target listing, I can switch into search mode and compare the strongest options.'
            )
        if persona['id'] == 'neighborhood-navigator':
            return (
                'I can help decode neighborhood feel, walkability, commute rhythm, and lifestyle fit. '
                'Share a city, listing, or area and I will compare what the neighborhood is likely to feel like day to day.'
            )
        if 'first-time buyer' in lowered or 'first time buyer' in lowered:
            return (
                'For a first-time buyer, focus on monthly payment comfort, neighborhood fit, required repairs, '
                'and resale resilience before stretching for extra space. If you share a city, budget, and bedroom '
                'range, I can switch into listing mode and narrow the best options.'
            )
        return (
            f"{persona['name']} can help with strategy first. Share what tradeoff matters most, like budget, space, "
            "location, or resale confidence, and I can either talk it through or switch into listing search mode."
        )

    if persona['id'] == 'investment-scout':
        return (
            'I can help with rental income, yield, and resale potential. Share a city, budget, or listing and I will '
            'pull the strongest options to analyze.'
        )
    if persona['id'] == 'neighborhood-navigator':
        return (
            'I can help compare neighborhoods, lifestyle fit, and the feel of different areas. Share a city or listing '
            'and I will narrow it down.'
        )
    return (
        f"I'm here to help. Ask {persona['name'].lower()} for neighborhood guidance, first-time buyer advice, or "
        "share a city and budget when you want me to pull matching listings."
    )


def rank_listing_for_persona(property_obj, persona_id):
    score = listing_ai_score(property_obj)
    property_type = listing_property_type(property_obj)
    price = listing_price(property_obj)
    square_feet = listing_square_feet(property_obj)
    bedrooms = listing_bedrooms(property_obj)
    status = listing_status(property_obj)
    location = listing_location(property_obj)
    image_url = listing_value(property_obj, 'image_url')

    if persona_id == 'buyer-guide':
        score += 12 if bedrooms >= 3 else 0
        score += 8 if property_type == 'house' else 0
        score += 5 if status == 'available' else 0
    elif persona_id == 'investment-scout':
        score += 10 if property_type in {'condo', 'apartment', 'townhouse'} else 0
        score += 8 if 0 < price <= 650000 else 0
        score += min(square_feet / 150, 12) if square_feet else 0
    elif persona_id == 'neighborhood-navigator':
        score += 12 if location.split(',')[0] else 0
        score += 6 if image_url else 0
        score += 6 if status == 'available' else 0
    return score


def choose_properties_for_message(properties, message, persona_id, limit=3):
    if not properties:
        return []

    lowered = str(message or '').lower()
    budgets = extract_budget(message)
    bedrooms = extract_bedrooms(message)
    bedroom_targets = extract_bedroom_targets(message)
    requested_type = extract_property_type(message)
    filtered = list(properties)

    if budgets:
        ceiling = max(budgets)
        filtered = [property_obj for property_obj in filtered if listing_price(property_obj) <= ceiling]

    if bedrooms is not None:
        filtered = [property_obj for property_obj in filtered if listing_bedrooms(property_obj) >= bedrooms]

    if bedroom_targets and is_comparison_request(message):
        exact_bedroom_matches = [
            property_obj for property_obj in filtered
            if listing_bedrooms(property_obj) in set(bedroom_targets)
        ]
        if exact_bedroom_matches:
            filtered = exact_bedroom_matches

    if requested_type:
        typed = [property_obj for property_obj in filtered if listing_property_type(property_obj) == requested_type]
        if typed:
            filtered = typed

    location_hits = [
        property_obj for property_obj in filtered
        if listing_location(property_obj).lower() in lowered or any(
            part.strip().lower() in lowered for part in listing_location(property_obj).split(',')
        )
    ]
    if location_hits:
        filtered = location_hits

    if not filtered:
        filtered = list(properties)

    ranked = sorted(
        filtered,
        key=lambda property_obj: (
            rank_listing_for_persona(property_obj, persona_id),
            listing_listing_date(property_obj) or datetime.min,
        ),
        reverse=True,
    )
    return ranked[:limit]


def build_system_prompt(persona, properties, market_snapshot):
    listing_lines = '\n'.join(f'- {property_fact_line(property_obj)}' for property_obj in properties)
    return (
        'You are an AI real-estate concierge on a property website. '
        f'Your role is {persona["name"]}. '
        f'Your tone should be {persona["tone"]}. '
        f'Primary goal: {persona["goal"]} '
        'Use only the listing and market context provided. '
        'Do not invent neighborhoods, schools, HOA rules, rental yields, or legal facts. '
        'When you recommend homes, mention the exact property title. '
        'If the user asks for something beyond the available inventory, explain the limitation clearly and suggest the closest fit. '
        'Keep responses between 90 and 180 words when possible.\n\n'
        f'Market snapshot: {market_snapshot}\n'
        f'Listings in focus:\n{listing_lines}'
    )


def _parse_hf_model_candidates(config):
    configured = str(config.get('HF_MODEL') or '').strip()
    configured_fallbacks = [
        item.strip()
        for item in str(config.get('HF_MODEL_FALLBACKS') or '').split(',')
        if item.strip()
    ]

    ordered = []
    seen = set()
    for candidate in [configured, *configured_fallbacks, *DEFAULT_HF_FALLBACK_MODELS]:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        ordered.append(candidate)
    return ordered


def _build_hf_model_ref(model_name, provider):
    model_name = str(model_name or '').strip()
    provider = str(provider or '').strip()
    if not model_name:
        raise RuntimeError('HF_MODEL is not configured.')
    if ':' in model_name:
        return model_name
    if provider and provider != 'auto':
        return f'{model_name}:{provider}'
    return model_name


def _decode_chat_completion(body):
    choices = body.get('choices') or []
    if not choices:
        raise RuntimeError('No completion choices were returned by the remote model.')

    message = choices[0].get('message', {})
    content = message.get('content', '')
    if isinstance(content, list):
        text_parts = [part.get('text', '') for part in content if isinstance(part, dict)]
        content = '\n'.join(part for part in text_parts if part)
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError('The remote model returned an empty message.')
    return content.strip()


def _post_json_chat_request(url, payload, headers, timeout, provider_name):
    raw_request = request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers=headers,
        method='POST',
    )

    try:
        with request.urlopen(raw_request, timeout=timeout) as response:
            return json.loads(response.read().decode('utf-8'))
    except error.HTTPError as exc:
        details = exc.read().decode('utf-8', errors='ignore')
        raise RuntimeError(f'{provider_name} returned {exc.code}: {details}') from exc
    except error.URLError as exc:
        raise RuntimeError(f'{provider_name} request failed: {exc.reason}') from exc


def _call_huggingface_chat_once(config, messages, model_name):
    token = config.get('HF_API_TOKEN')
    if not token:
        raise RuntimeError('HF_API_TOKEN is not configured.')

    provider = config.get('HF_PROVIDER')
    payload = {
        'model': _build_hf_model_ref(model_name, provider),
        'messages': messages,
        'temperature': 0.6,
        'max_tokens': 320,
    }

    body = _post_json_chat_request(
        config.get('HF_API_URL'),
        payload,
        {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
        },
        config.get('AI_REQUEST_TIMEOUT', 20),
        'Hugging Face',
    )

    return {
        'content': _decode_chat_completion(body),
        'model': payload['model'],
    }


def call_huggingface_chat(config, messages):
    failures = []
    for model_name in _parse_hf_model_candidates(config):
        try:
            return _call_huggingface_chat_once(config, messages, model_name)
        except RuntimeError as exc:
            failures.append(f'{model_name}: {exc}')

    if failures:
        raise RuntimeError(' | '.join(failures))
    raise RuntimeError('No Hugging Face model candidates are configured.')


def call_llama_cpp_chat(config, messages):
    if not config.get('LLAMA_CPP_ENABLED'):
        raise RuntimeError('llama.cpp fallback is disabled.')

    payload = {
        'messages': messages,
        'temperature': 0.4,
        'max_tokens': 320,
        'stream': False,
        'cache_prompt': True,
    }

    model_name = str(config.get('LLAMA_CPP_MODEL') or '').strip()
    if model_name:
        payload['model'] = model_name

    body = _post_json_chat_request(
        config.get('LLAMA_CPP_URL'),
        payload,
        {'Content-Type': 'application/json'},
        config.get('AI_REQUEST_TIMEOUT', 20),
        'llama.cpp',
    )

    response_model = body.get('model') or model_name or 'llama.cpp'
    return {
        'content': _decode_chat_completion(body),
        'model': response_model,
    }


def call_remote_chat(config, messages):
    failures = []
    huggingface_enabled = bool(config.get('HF_API_TOKEN')) and not bool(config.get('HF_DISABLED'))

    if huggingface_enabled:
        try:
            result = call_huggingface_chat(config, messages)
            result['source'] = 'huggingface'
            return result
        except RuntimeError as exc:
            failures.append(f'Hugging Face: {exc}')

    if config.get('LLAMA_CPP_ENABLED'):
        try:
            result = call_llama_cpp_chat(config, messages)
            result['source'] = 'llama_cpp'
            result['upstream_failures'] = failures
            return result
        except RuntimeError as exc:
            failures.append(f'llama.cpp: {exc}')

    if failures:
        raise RuntimeError(' | '.join(failures))
    raise RuntimeError('No remote AI providers are configured.')


def build_local_fallback_reply(persona, user_message, matched_properties, market_snapshot):
    intro = {
        'buyer-guide': 'Here is the clearest buyer-focused read on your request.',
        'investment-scout': 'Here is the strongest investment-oriented view from the active search results.',
        'neighborhood-navigator': 'Here is the lifestyle and location-focused read from the active search results.',
    }[persona['id']]

    if not matched_properties:
        return (
            f'{intro} I do not have a direct listing match for "{user_message}" right now. '
            f'{market_snapshot} Try widening the location, bedroom count, or budget so I can point you to the closest options.'
        )

    bedroom_targets = extract_bedroom_targets(user_message)
    if persona['id'] == 'buyer-guide' and bedroom_targets and is_comparison_request(user_message):
        grouped = {}
        for property_obj in matched_properties:
            grouped.setdefault(listing_bedrooms(property_obj), []).append(property_obj)

        comparison_lines = []
        for target in bedroom_targets[:3]:
            options = grouped.get(target, [])
            if not options:
                comparison_lines.append(f'I do not currently have a strong {target}-bedroom option in this search scope.')
                continue

            best = sorted(
                options,
                key=lambda property_obj: (listing_ai_score(property_obj), -listing_price(property_obj)),
                reverse=True,
            )[:2]
            line = ', '.join(
                f'{listing_title(property_obj)} at ${listing_price(property_obj):,.0f}'
                for property_obj in best
            )
            comparison_lines.append(f'For {target}-bedroom options, look at {line}.')

        return (
            f'{intro} {market_snapshot} '
            + ' '.join(comparison_lines)
            + ' If you want, I can narrow this comparison by city, budget, or property type.'
        )

    lines = []
    for property_obj in matched_properties:
        if persona['id'] == 'investment-scout':
            lines.append(
                f'{listing_title(property_obj)} stands out at ${listing_price(property_obj):,.0f} with '
                f'{listing_square_feet(property_obj):,} sq ft and an AI score of {listing_ai_score(property_obj):.0f}/100.'
            )
        elif persona['id'] == 'neighborhood-navigator':
            lines.append(
                f'{listing_title(property_obj)} gives a {listing_location(property_obj)} vibe, with '
                f'{listing_bedrooms(property_obj)} bedrooms and a {listing_property_type(property_obj)} layout that feels easy to picture day to day.'
            )
        else:
            lines.append(
                f'{listing_title(property_obj)} is a strong fit at ${listing_price(property_obj):,.0f}, with '
                f'{listing_bedrooms(property_obj)} bedrooms, {listing_bathrooms(property_obj)} bathrooms, and {listing_square_feet(property_obj):,} sq ft.'
            )

    return (
        f'{intro} {market_snapshot} '
        + ' '.join(lines)
        + ' If you want, I can narrow this further by budget, property type, or city.'
    )
