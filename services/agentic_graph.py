from typing import Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from services.ai_brain import (
    build_local_fallback_reply,
    build_market_snapshot,
    build_no_tool_reply,
    build_smalltalk_reply,
    build_system_prompt,
    call_remote_chat,
    choose_properties_for_message,
    infer_query_arguments,
    route_agent_query,
    listing_ai_score,
    listing_bedrooms,
    listing_location,
    listing_price,
    listing_property_type,
    listing_status,
    listing_title,
    rank_listing_for_persona,
)
from services.realtime_listings import search_live_listings


class AgenticSearchState(TypedDict, total=False):
    config: Any
    mode: str
    persona: dict[str, Any]
    message: str
    history: list[dict[str, str]]
    filters: dict[str, Any]
    local_inventory: list[Any]
    local_results: list[Any]
    live_results: list[dict[str, Any]]
    selected_results: list[Any]
    source: str
    model_source: str
    remote_model: str | None
    fallback_reason: str | None
    retrieval_notes: list[str]
    query_route: str
    route_reason: str | None
    use_tools: bool
    reply: str | None
    market_snapshot: str


def _clean_text(value):
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _copy_filters(filters):
    return {key: value for key, value in dict(filters or {}).items() if value not in (None, '')}


def _apply_message_hints(config, filters, message, history=None):
    enriched = infer_query_arguments(config, message, existing_filters=_copy_filters(filters), history=history)
    if not enriched.get('query') and message:
        enriched['query'] = message
    return enriched


def _matches_filters(listing, filters):
    location = _clean_text(filters.get('location'))
    query = _clean_text(filters.get('query'))
    property_type = _clean_text(filters.get('property_type'))
    status = _clean_text(filters.get('status'))
    min_price = filters.get('min_price')
    max_price = filters.get('max_price')
    min_bedrooms = filters.get('min_bedrooms')
    max_bedrooms = filters.get('max_bedrooms')

    if location:
        haystacks = [
            listing_location(listing).lower(),
            str(getattr(listing, 'address', '') if not isinstance(listing, dict) else listing.get('address', '')).lower(),
        ]
        if location.lower() not in ' '.join(haystacks):
            return False

    if query:
        haystack = ' '.join([
            listing_title(listing).lower(),
            listing_location(listing).lower(),
            str(getattr(listing, 'description', '') if not isinstance(listing, dict) else listing.get('description', '')).lower(),
            str(getattr(listing, 'address', '') if not isinstance(listing, dict) else listing.get('address', '')).lower(),
        ])
        if query.lower() not in haystack and not any(token and token in haystack for token in query.lower().split()):
            return False

    if property_type and listing_property_type(listing) != property_type:
        return False
    if status and listing_status(listing) != status:
        return False
    if min_price not in (None, '') and listing_price(listing) < float(min_price):
        return False
    if max_price not in (None, '') and listing_price(listing) > float(max_price):
        return False
    if min_bedrooms not in (None, '') and listing_bedrooms(listing) < int(min_bedrooms):
        return False
    if max_bedrooms not in (None, '') and listing_bedrooms(listing) > int(max_bedrooms):
        return False
    return True


def _build_search_summary(persona, results, source, filters):
    if not results:
        return 'I could not find a close match in the current search scope. Try widening the location, budget, or home type.'

    titles = ', '.join(listing_title(item) for item in results[:3])
    if source == 'live_market':
        prefix = f'{persona["name"]} found live market matches'
    else:
        prefix = f'{persona["name"]} found strong matches from the local inventory'

    location = _clean_text(filters.get('location'))
    if location:
        prefix += f' for {location}'
    return f'{prefix}: {titles}.'


def _parse_request_node(state: AgenticSearchState):
    message = _clean_text(state.get('message')) or ''
    incoming_filters = _copy_filters(state.get('filters') or {})
    route = route_agent_query(message, filters=incoming_filters, mode=state.get('mode') or 'chat')
    filters = _apply_message_hints(state.get('config'), incoming_filters, message, history=state.get('history')) if route.get('use_tools') else incoming_filters
    notes = []
    if route.get('use_tools') and not filters.get('location'):
        notes.append('No explicit location was found, so live provider retrieval may fall back to local inventory.')
    return {
        'filters': filters,
        'retrieval_notes': notes,
        'query_route': route.get('query_route', 'tool_search'),
        'route_reason': route.get('route_reason'),
        'use_tools': route.get('use_tools', True),
    }


def _retrieve_live_results_node(state: AgenticSearchState):
    if not state.get('use_tools', True):
        return {'live_results': [], 'fallback_reason': None}

    filters = state.get('filters') or {}
    if not filters.get('location'):
        return {
            'live_results': [],
            'fallback_reason': 'Add a city, ZIP code, or neighborhood to search live listings.',
        }

    try:
        live_results = search_live_listings(state['config'], filters, limit=12)
    except RuntimeError as exc:
        return {'live_results': [], 'fallback_reason': str(exc)}

    if not live_results:
        return {
            'live_results': [],
            'fallback_reason': 'The live market feed returned no listings for that search.',
        }
    return {'live_results': live_results}


def _retrieve_local_results_node(state: AgenticSearchState):
    if not state.get('use_tools', True):
        return {'local_results': []}

    filters = state.get('filters') or {}
    local_inventory = list(state.get('local_inventory') or [])
    local_results = [item for item in local_inventory if _matches_filters(item, filters)]
    if not local_results and state.get('message'):
        local_results = choose_properties_for_message(
            local_inventory,
            state.get('message') or '',
            state['persona']['id'],
            limit=12,
        )
    return {'local_results': local_results}


def _rank_results_node(state: AgenticSearchState):
    if not state.get('use_tools', True):
        return {
            'selected_results': [],
            'source': 'conversation',
            'market_snapshot': build_market_snapshot([]),
        }

    live_results = list(state.get('live_results') or [])
    local_results = list(state.get('local_results') or [])
    candidates = live_results or local_results
    source = 'live_market' if live_results else 'local_inventory'

    if not candidates:
        return {'selected_results': [], 'source': source}

    message = state.get('message') or ''
    persona_id = state['persona']['id']
    if message:
        prioritized = choose_properties_for_message(candidates, message, persona_id, limit=6)
    else:
        prioritized = sorted(
            candidates,
            key=lambda item: (rank_listing_for_persona(item, persona_id), listing_ai_score(item)),
            reverse=True,
        )[:6]

    if not prioritized:
        prioritized = sorted(
            candidates,
            key=lambda item: (rank_listing_for_persona(item, persona_id), listing_ai_score(item)),
            reverse=True,
        )[:6]

    return {
        'selected_results': prioritized,
        'source': source,
        'market_snapshot': build_market_snapshot(candidates),
    }


def _compose_response_node(state: AgenticSearchState):
    persona = state['persona']
    results = list(state.get('selected_results') or [])
    market_snapshot = state.get('market_snapshot') or build_market_snapshot(results)
    source = state.get('source') or 'local_inventory'
    message = state.get('message') or 'find homes that fit my search'

    if state.get('mode') == 'search':
        return {
            'reply': _build_search_summary(persona, results, source, state.get('filters') or {}),
        }

    if state.get('query_route') == 'smalltalk':
        return {
            'reply': build_smalltalk_reply(persona, message),
            'fallback_reason': None,
            'model_source': 'smalltalk',
            'remote_model': None,
            'query_route': state.get('query_route'),
            'route_reason': state.get('route_reason'),
        }

    if not state.get('use_tools', True):
        return {
            'reply': build_no_tool_reply(persona, message, state.get('query_route') or 'conversation'),
            'fallback_reason': None,
            'model_source': 'query_router',
            'remote_model': None,
            'query_route': state.get('query_route'),
            'route_reason': state.get('route_reason'),
        }

    if not results:
        reply = build_local_fallback_reply(persona, message, [], market_snapshot)
        return {
            'reply': reply,
            'source': source,
            'model_source': 'local_fallback',
            'query_route': state.get('query_route'),
            'route_reason': state.get('route_reason'),
        }

    system_prompt = build_system_prompt(persona, results, market_snapshot)
    history_messages = list(state.get('history') or [])[-6:]
    model_messages = [
        {'role': 'system', 'content': system_prompt},
        *history_messages,
        {'role': 'user', 'content': message},
    ]

    fallback_reason = state.get('fallback_reason')
    try:
        model_result = call_remote_chat(state['config'], model_messages)
        reply = model_result.get('content')
        model_source = model_result.get('source', 'huggingface')
        remote_model = model_result.get('model')
    except RuntimeError as exc:
        fallback_reason = str(exc)
        reply = build_local_fallback_reply(persona, message, results, market_snapshot)
        model_source = 'local_fallback'
        remote_model = None

    return {
        'reply': reply,
        'fallback_reason': fallback_reason,
        'model_source': model_source,
        'remote_model': remote_model,
        'query_route': state.get('query_route'),
        'route_reason': state.get('route_reason'),
    }


def _build_agentic_graph():
    graph = StateGraph(AgenticSearchState)
    graph.add_node('parse_request', _parse_request_node)
    graph.add_node('retrieve_live', _retrieve_live_results_node)
    graph.add_node('retrieve_local', _retrieve_local_results_node)
    graph.add_node('rank_results', _rank_results_node)
    graph.add_node('compose_response', _compose_response_node)
    graph.add_edge(START, 'parse_request')
    graph.add_edge('parse_request', 'retrieve_live')
    graph.add_edge('retrieve_live', 'retrieve_local')
    graph.add_edge('retrieve_local', 'rank_results')
    graph.add_edge('rank_results', 'compose_response')
    graph.add_edge('compose_response', END)
    return graph.compile()


AGENTIC_LISTING_GRAPH = _build_agentic_graph()


def run_agentic_listing_flow(config, persona, message='', filters=None, history=None, local_inventory=None, mode='chat'):
    state = {
        'config': config,
        'mode': mode,
        'persona': persona,
        'message': message or '',
        'history': history or [],
        'filters': filters or {},
        'local_inventory': local_inventory or [],
        'fallback_reason': None,
    }
    return AGENTIC_LISTING_GRAPH.invoke(state)
