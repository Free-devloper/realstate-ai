from datetime import datetime, timezone
import io
import json
from urllib.error import HTTPError
import services.ai_brain as ai_brain
import services.realtime_listings as realtime_listings


LIVE_SAMPLE_LISTING = {
    'id': 'external:rentcast:listing-1',
    'lookup_key': 'external:rentcast:listing-1',
    'title': '742 Evergreen Terrace',
    'description': 'Live provider listing used in tests.',
    'price': 515000,
    'location': 'Austin, Texas',
    'address': '742 Evergreen Terrace',
    'bedrooms': 3,
    'bathrooms': 2,
    'square_feet': 1820,
    'property_type': 'house',
    'status': 'available',
    'listing_date': '2026-03-14T00:00:00',
    'image_url': None,
    'ai_score': 88,
    'cover_style': 'sunset-grove',
    'listing_source': 'live_market',
    'source_label': 'Live market feed',
    'is_external': True,
    'can_favorite': False,
    'listing_url': 'https://example.com/listing/742-evergreen',
}


def parse_sse_events(raw_text):
    events = []
    for block in str(raw_text or '').strip().split('\n\n'):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        event_name = 'message'
        data_lines = []
        for line in lines:
            if line.startswith('event:'):
                event_name = line.split(':', 1)[1].strip()
            elif line.startswith('data:'):
                data_lines.append(line.split(':', 1)[1].strip())
        if data_lines:
            events.append((event_name, json.loads('\n'.join(data_lines))))
    return events


def test_infer_query_arguments_handles_joined_and_misspelled_new_york_new_jersey():
    filters = ai_brain.infer_query_arguments({'LLAMA_CPP_ENABLED': False}, 'Find something for me in newyork in new jersy')

    assert filters['location'] == 'New York, New Jersey'


def test_build_rentcast_params_normalizes_city_and_state_names():
    params = realtime_listings.build_rentcast_params({'location': 'New York, New Jersey'}, limit=12)

    assert params['city'] == 'New York'
    assert params['state'] == 'NJ'


def test_infer_query_arguments_uses_history_for_follow_up_budget(monkeypatch):
    class FakeBoundModel:
        def invoke(self, messages):
            class FakeResponse:
                tool_calls = [{
                    'args': {
                        'location': 'sacramento',
                        'max_price': 200000,
                    }
                }]
                additional_kwargs = {}

            return FakeResponse()

    class FakeChatOpenAI:
        def __init__(self, *args, **kwargs):
            pass

        def bind_tools(self, tools, tool_choice='required'):
            return FakeBoundModel()

    monkeypatch.setattr(ai_brain, 'ChatOpenAI', FakeChatOpenAI)

    filters = ai_brain.infer_query_arguments(
        {
            'LLAMA_CPP_ENABLED': True,
            'LLAMA_CPP_URL': 'http://127.0.0.1:8080/v1/chat/completions',
            'LLAMA_CPP_MODEL': 'qwen3.5-9b',
            'AI_REQUEST_TIMEOUT': 20,
        },
        'Find a home in 200k budget',
        history=[{'role': 'user', 'content': 'Find a home suitable in sacramento'}],
    )

    assert filters['location'] == 'Sacramento'
    assert filters['max_price'] == 200000


def test_infer_query_arguments_prefers_langchain_tool_calls(monkeypatch):
    class FakeBoundModel:
        def invoke(self, messages):
            class FakeResponse:
                tool_calls = [{
                    'args': {
                        'location': 'phoenix, arizona',
                        'property_type': 'condo',
                        'min_bedrooms': 2,
                        'max_price': 550000,
                    }
                }]
                additional_kwargs = {}

            return FakeResponse()

    class FakeChatOpenAI:
        def __init__(self, *args, **kwargs):
            pass

        def bind_tools(self, tools, tool_choice='required'):
            return FakeBoundModel()

    monkeypatch.setattr(ai_brain, 'ChatOpenAI', FakeChatOpenAI)

    filters = ai_brain.infer_query_arguments(
        {
            'LLAMA_CPP_ENABLED': True,
            'LLAMA_CPP_URL': 'http://127.0.0.1:8080/v1/chat/completions',
            'LLAMA_CPP_MODEL': 'qwen3.5-9b',
            'AI_REQUEST_TIMEOUT': 20,
        },
        'Show me something nice.',
    )

    assert filters['location'] == 'Phoenix, Arizona'
    assert filters['property_type'] == 'condo'
    assert filters['min_bedrooms'] == 2
    assert filters['max_price'] == 550000


def test_huggingface_chat_uses_provider_suffix_and_falls_back_models(monkeypatch):
    import services.ai_brain as ai_brain

    seen_models = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def read(self):
            return self.payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(raw_request, timeout=20):
        payload = json.loads(raw_request.data.decode('utf-8'))
        seen_models.append(payload['model'])
        if payload['model'] == 'bad-model:hf-inference':
            raise HTTPError(
                raw_request.full_url,
                400,
                'Bad Request',
                hdrs=None,
                fp=io.BytesIO(b'{"error":"model_not_supported"}'),
            )
        return FakeResponse(b'{"choices":[{"message":{"content":"Remote answer"}}]}')

    monkeypatch.setattr(ai_brain.request, 'urlopen', fake_urlopen)

    result = ai_brain.call_huggingface_chat({
        'HF_API_TOKEN': 'token',
        'HF_API_URL': 'https://router.huggingface.co/v1/chat/completions',
        'HF_MODEL': 'bad-model',
        'HF_MODEL_FALLBACKS': 'google/gemma-2-2b-it',
        'HF_PROVIDER': 'hf-inference',
        'AI_REQUEST_TIMEOUT': 20,
    }, [{'role': 'user', 'content': 'hello'}])

    assert result['content'] == 'Remote answer'
    assert result['model'] == 'google/gemma-2-2b-it:hf-inference'
    assert seen_models == ['bad-model:hf-inference', 'google/gemma-2-2b-it:hf-inference']


def test_remote_chat_skips_huggingface_when_disabled(monkeypatch):
    import services.ai_brain as ai_brain

    def fail_if_called(config, messages):
        raise AssertionError('Hugging Face should have been skipped')

    def fake_llama_cpp(config, messages):
        return {
            'content': 'llama only',
            'model': 'qwen3.5-9b',
        }

    monkeypatch.setattr(ai_brain, 'call_huggingface_chat', fail_if_called)
    monkeypatch.setattr(ai_brain, 'call_llama_cpp_chat', fake_llama_cpp)

    result = ai_brain.call_remote_chat({
        'HF_API_TOKEN': 'token',
        'HF_DISABLED': True,
        'LLAMA_CPP_ENABLED': True,
    }, [{'role': 'user', 'content': 'hello'}])

    assert result['source'] == 'llama_cpp'
    assert result['content'] == 'llama only'
    assert result['upstream_failures'] == []


def test_remote_chat_falls_back_to_llama_cpp(monkeypatch):
    import services.ai_brain as ai_brain

    def fake_huggingface(config, messages):
        raise RuntimeError('provider rejected request')

    def fake_llama_cpp(config, messages):
        return {
            'content': 'Local llama.cpp response',
            'model': 'qwen3.5-9b',
        }

    monkeypatch.setattr(ai_brain, 'call_huggingface_chat', fake_huggingface)
    monkeypatch.setattr(ai_brain, 'call_llama_cpp_chat', fake_llama_cpp)

    result = ai_brain.call_remote_chat({
        'HF_API_TOKEN': 'token',
        'LLAMA_CPP_ENABLED': True,
    }, [{'role': 'user', 'content': 'hello'}])

    assert result['source'] == 'llama_cpp'
    assert result['model'] == 'qwen3.5-9b'
    assert result['content'] == 'Local llama.cpp response'
    assert result['upstream_failures'] == ['Hugging Face: provider rejected request']


def test_index_page_renders(client):
    response = client.get('/')

    assert response.status_code == 200
    assert b'Real Estate AI' in response.data


def test_overview_endpoint_returns_seeded_metrics(client):
    response = client.get('/api/overview')
    payload = response.get_json()

    assert response.status_code == 200
    assert payload['properties_count'] >= 8
    assert payload['agents_count'] >= 3
    assert payload['available_count'] >= 1


def test_ai_agents_endpoint_returns_personas(client):
    response = client.get('/api/ai/agents')
    payload = response.get_json()

    assert response.status_code == 200
    assert len(payload['agents']) >= 3
    assert payload['status']['provider'] == 'local_fallback'
    assert payload['status']['live_listings_enabled'] is False


def test_ai_chat_returns_fallback_response_and_recommendations(client):
    response = client.post('/api/ai/chat', json={
        'agent_id': 'buyer-guide',
        'message': 'Find me a home in Austin with at least 3 bedrooms under 600k.',
        'history': [],
    })
    payload = response.get_json()

    assert response.status_code == 200
    assert payload['source'] == 'local_fallback'
    assert payload['agent']['id'] == 'buyer-guide'
    assert payload['recommended_properties']
    assert payload['recommended_properties'][0]['lookup_key'].startswith('local:')
    assert payload['listings_source'] == 'local_inventory'
    assert 'Austin' in payload['message'] or 'home' in payload['message'].lower()



def test_ai_chat_stream_returns_sse_events(client):
    response = client.post('/api/ai/chat/stream', json={
        'agent_id': 'buyer-guide',
        'message': 'Find me a home in Austin with at least 3 bedrooms under 600k.',
        'history': [],
    }, buffered=True)

    assert response.status_code == 200
    assert response.mimetype == 'text/event-stream'

    events = parse_sse_events(response.get_data(as_text=True))
    event_names = [name for name, _ in events]

    assert event_names[0] == 'meta'
    assert 'delta' in event_names
    assert event_names[-1] == 'done'
    assert events[-1][1]['source'] == 'local_fallback'
    assert events[-1][1]['status']['streaming_supported'] is True
    assert events[-1][1]['message']


def test_ai_chat_greeting_does_not_trigger_listing_recommendations(client):
    response = client.post('/api/ai/chat', json={
        'agent_id': 'buyer-guide',
        'message': 'hello',
        'history': [],
    })
    payload = response.get_json()

    assert response.status_code == 200
    assert payload['source'] == 'smalltalk'
    assert payload['recommended_properties'] == []
    assert payload['retrieval_notes'] == []
    assert 'hello' in payload['message'].lower() or 'help' in payload['message'].lower()


def test_neighborhood_prompt_switches_to_navigator_even_with_misspelling(client):
    response = client.post('/api/ai/chat', json={
        'agent_id': 'buyer-guide',
        'message': 'hows the neighbour hood',
        'history': [],
    })
    payload = response.get_json()

    assert response.status_code == 200
    assert payload['agent']['id'] == 'neighborhood-navigator'
    assert payload['source'] == 'query_router'
    assert payload['query_route'] == 'advisory'
    assert payload['recommended_properties'] == []
    assert 'neighborhood' in payload['message'].lower() or 'walkability' in payload['message'].lower()


def test_ai_chat_advisory_route_skips_listing_tools(client):
    response = client.post('/api/ai/chat', json={
        'agent_id': 'buyer-guide',
        'message': 'What should a first-time buyer prioritize?',
        'history': [],
    })
    payload = response.get_json()

    assert response.status_code == 200
    assert payload['source'] == 'query_router'
    assert payload['query_route'] == 'advisory'
    assert payload['recommended_properties'] == []
    assert 'first-time buyer' in payload['message'].lower() or 'budget' in payload['message'].lower()


def test_ai_chat_reuses_query_cache(client, app_module, monkeypatch):
    app_module.AGENT_QUERY_CACHE.clear()
    call_count = {'value': 0}

    def fake_flow(config, persona, message='', filters=None, history=None, local_inventory=None, mode='chat'):
        call_count['value'] += 1
        return {
            'reply': 'Cached chat answer',
            'source': 'conversation',
            'model_source': 'query_router',
            'remote_model': None,
            'fallback_reason': None,
            'retrieval_notes': [],
            'query_route': 'conversation',
            'route_reason': 'Conversational message detected, so listing tools were skipped.',
            'selected_results': [],
        }

    monkeypatch.setattr(app_module, 'run_agentic_listing_flow', fake_flow)

    first = client.post('/api/ai/chat', json={
        'agent_id': 'buyer-guide',
        'message': 'Can you help me think through my options?',
        'history': [],
    }).get_json()
    second = client.post('/api/ai/chat', json={
        'agent_id': 'buyer-guide',
        'message': 'Can you help me think through my options?',
        'history': [],
    }).get_json()

    assert call_count['value'] == 1
    assert first['cache_hit'] is False
    assert second['cache_hit'] is True


def test_search_reuses_query_cache(client, app_module, monkeypatch):
    app_module.AGENT_QUERY_CACHE.clear()
    call_count = {'value': 0}

    def fake_flow(config, persona, message='', filters=None, history=None, local_inventory=None, mode='search'):
        call_count['value'] += 1
        return {
            'reply': 'Buyer Guide found strong matches from the local inventory: Sunlit Garden Residence.',
            'source': 'local_inventory',
            'model_source': 'local_fallback',
            'remote_model': None,
            'fallback_reason': None,
            'retrieval_notes': [],
            'query_route': 'tool_search',
            'route_reason': 'Structured search filters were provided.',
            'selected_results': [dict(LIVE_SAMPLE_LISTING)],
        }

    monkeypatch.setattr(app_module, 'run_agentic_listing_flow', fake_flow)

    first = client.post('/api/search', json={
        'location': 'Austin',
        'min_bedrooms': 2,
    }).get_json()
    second = client.post('/api/search', json={
        'location': 'Austin',
        'min_bedrooms': 2,
    }).get_json()

    assert call_count['value'] == 1
    assert first['cache_hit'] is False
    assert second['cache_hit'] is True


def test_ai_chat_comparison_prompt_stays_scoped_to_requested_bedrooms(client):
    response = client.post('/api/ai/chat', json={
        'agent_id': 'buyer-guide',
        'message': 'Compare two and three bedroom options for me.',
        'history': [],
    })
    payload = response.get_json()

    assert response.status_code == 200
    assert payload['recommended_properties']
    assert all(item['bedrooms'] in {2, 3} for item in payload['recommended_properties'])
    assert 'For 2-bedroom options' in payload['message']
    assert 'For 3-bedroom options' in payload['message']

def test_ai_chat_investment_rental_income_question_uses_tools(client, app_module, monkeypatch):
    import services.agentic_graph as agentic_graph

    app_module.AGENT_QUERY_CACHE.clear()

    def fake_live_search(config, filters, limit=12):
        assert filters['location'] == 'Sacramento'
        return [dict(LIVE_SAMPLE_LISTING, location='Sacramento, CA', title='Sacramento Income Candidate')]

    monkeypatch.setattr(agentic_graph, 'search_live_listings', fake_live_search)

    response = client.post('/api/ai/chat', json={
        'agent_id': 'investment-scout',
        'message': 'what would be the rental income?',
        'history': [
            {'role': 'user', 'content': 'Find a home suitable in sacramento'},
            {'role': 'assistant', 'content': 'I found several options in Sacramento.'},
        ],
    })
    payload = response.get_json()

    assert response.status_code == 200
    assert payload['query_route'] == 'tool_search'
    assert payload['listings_source'] == 'live_market'
    assert payload['recommended_properties']


def test_ai_chat_follow_up_uses_prior_city_context(client, app_module, monkeypatch):
    import services.agentic_graph as agentic_graph

    app_module.AGENT_QUERY_CACHE.clear()

    def fake_live_search(config, filters, limit=12):
        assert filters['location'] == 'Sacramento'
        assert filters['max_price'] == 200000
        return [dict(LIVE_SAMPLE_LISTING, location='Sacramento, CA', title='Budget Sacramento Home')]

    monkeypatch.setattr(agentic_graph, 'search_live_listings', fake_live_search)

    response = client.post('/api/ai/chat', json={
        'agent_id': 'buyer-guide',
        'message': 'Find a home in 200k budget',
        'history': [
            {'role': 'user', 'content': 'Find a home suitable in sacramento'},
            {'role': 'assistant', 'content': 'I found several options in Sacramento.'},
        ],
    })
    payload = response.get_json()

    assert response.status_code == 200
    assert payload['listings_source'] == 'live_market'
    assert payload['recommended_properties'][0]['is_external'] is True


def test_ai_chat_uses_live_listings_when_provider_returns_results(client, app_module, monkeypatch):
    import services.agentic_graph as agentic_graph

    app_module.AGENT_QUERY_CACHE.clear()

    def fake_live_search(config, filters, limit=12):
        assert filters['location'] == 'Austin'
        return [dict(LIVE_SAMPLE_LISTING)]

    monkeypatch.setattr(agentic_graph, 'search_live_listings', fake_live_search)

    response = client.post('/api/ai/chat', json={
        'agent_id': 'buyer-guide',
        'message': 'Find me a house in Austin under 600k.',
        'history': [],
    })
    payload = response.get_json()

    assert response.status_code == 200
    assert payload['source'] in {'huggingface', 'local_fallback'}
    assert payload['recommended_properties'][0]['is_external'] is True
    assert payload['recommended_properties'][0]['lookup_key'].startswith('external:rentcast:')
    assert payload['listings_source'] == 'live_market'



def test_ai_conversation_memory_persists_for_logged_in_user(client, app_module):
    unique = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')
    register_response = client.post('/api/users/register', json={
        'username': f'chat_user_{unique}',
        'email': f'chat_user_{unique}@example.com',
        'password': 'secret123',
        'user_type': 'buyer',
    })
    user_id = register_response.get_json()['id']

    first_response = client.post('/api/ai/chat', json={
        'agent_id': 'buyer-guide',
        'message': 'Find me a family home in Austin.',
        'user_id': user_id,
    })
    first_payload = first_response.get_json()

    assert first_response.status_code == 200
    assert first_payload['conversation']['agent_id'] == 'buyer-guide'

    conversation_id = first_payload['conversation']['id']
    second_response = client.post('/api/ai/chat', json={
        'agent_id': 'buyer-guide',
        'message': 'Keep the budget under 600k and prefer at least 3 bedrooms.',
        'user_id': user_id,
        'conversation_id': conversation_id,
    })
    second_payload = second_response.get_json()

    assert second_response.status_code == 200
    assert second_payload['conversation']['id'] == conversation_id

    conversation_response = client.get(f'/api/users/{user_id}/ai/conversations/{conversation_id}')
    conversation_payload = conversation_response.get_json()

    assert conversation_response.status_code == 200
    assert len(conversation_payload['messages']) == 4
    assert conversation_payload['messages'][0]['role'] == 'user'
    assert conversation_payload['messages'][-1]['role'] == 'assistant'


def test_select_chat_persona_prefers_llm_handoff_for_surroundings_prompt(monkeypatch):
    class FakeBoundModel:
        def invoke(self, messages):
            class FakeResponse:
                tool_calls = [{'args': {'agent_id': 'neighborhood-navigator', 'reason': 'surroundings and neighborhood context'}}]
                additional_kwargs = {}

            return FakeResponse()

    class FakeChatOpenAI:
        def __init__(self, *args, **kwargs):
            pass

        def bind_tools(self, tools, tool_choice='required'):
            return FakeBoundModel()

    monkeypatch.setattr(ai_brain, 'ChatOpenAI', FakeChatOpenAI)

    persona = ai_brain.select_chat_persona(
        {
            'LLAMA_CPP_ENABLED': True,
            'LLAMA_CPP_URL': 'http://127.0.0.1:8080/v1/chat/completions',
            'LLAMA_CPP_MODEL': 'qwen3.5-9b',
            'AI_REQUEST_TIMEOUT': 20,
        },
        'Tell me about the surroundings',
        preferred_agent_id='investment-scout',
        history=[
            {'role': 'user', 'content': 'Find me a good investment property in Sacramento'},
            {'role': 'assistant', 'content': '[Investment Scout] I found some strong Sacramento options.', 'agent_id': 'investment-scout', 'agent_name': 'Investment Scout'},
        ],
    )

    assert persona['id'] == 'neighborhood-navigator'


def test_ai_chat_can_switch_responder_agents_within_one_conversation(client):
    unique = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')
    register_response = client.post('/api/users/register', json={
        'username': f'mixed_agents_{unique}',
        'email': f'mixed_agents_{unique}@example.com',
        'password': 'secret123',
        'user_type': 'buyer',
    })
    user_id = register_response.get_json()['id']

    first_response = client.post('/api/ai/chat', json={
        'agent_id': 'buyer-guide',
        'message': 'Find a home suitable in Sacramento.',
        'user_id': user_id,
    })
    first_payload = first_response.get_json()
    conversation_id = first_payload['conversation']['id']

    second_response = client.post('/api/ai/chat', json={
        'agent_id': 'buyer-guide',
        'message': 'What would be the rental income?',
        'user_id': user_id,
        'conversation_id': conversation_id,
    })
    second_payload = second_response.get_json()

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert second_payload['conversation']['id'] == conversation_id
    assert second_payload['agent']['id'] == 'investment-scout'

    conversation_response = client.get(f'/api/users/{user_id}/ai/conversations/{conversation_id}')
    messages = conversation_response.get_json()['messages']

    assert any(message.get('agent_name') == 'Buyer Guide' for message in messages if message['role'] == 'assistant')
    assert any(message.get('agent_name') == 'Investment Scout' for message in messages if message['role'] == 'assistant')


def test_ai_can_create_and_list_multiple_conversations_per_agent(client):
    unique = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')
    register_response = client.post('/api/users/register', json={
        'username': f'threads_user_{unique}',
        'email': f'threads_user_{unique}@example.com',
        'password': 'secret123',
        'user_type': 'buyer',
    })
    user_id = register_response.get_json()['id']

    first_conversation = client.post(f'/api/users/{user_id}/ai/conversations', json={'agent_id': 'buyer-guide'}).get_json()
    second_conversation = client.post(f'/api/users/{user_id}/ai/conversations', json={'agent_id': 'buyer-guide'}).get_json()
    list_response = client.get(f'/api/users/{user_id}/ai/conversations?agent_id=buyer-guide')
    list_payload = list_response.get_json()

    assert first_conversation['conversation']['id'] != second_conversation['conversation']['id']
    assert list_response.status_code == 200
    assert len(list_payload['conversations']) == 2

def test_properties_filter_by_type_and_location(client):
    response = client.get('/api/properties?property_type=house&location=Austin')
    payload = response.get_json()

    assert response.status_code == 200
    assert payload['properties']
    assert all(item['property_type'] == 'house' for item in payload['properties'])
    assert all('Austin' in item['location'] for item in payload['properties'])


def test_search_records_history_and_uses_live_results_when_available(client, app_module, monkeypatch):
    import services.agentic_graph as agentic_graph

    unique = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')
    register_response = client.post('/api/users/register', json={
        'username': f'searcher_{unique}',
        'email': f'searcher_{unique}@example.com',
        'password': 'secret123',
        'user_type': 'buyer',
    })
    user_id = register_response.get_json()['id']

    def fake_live_search(config, filters, limit=12):
        assert filters['location'] == 'Austin'
        return [dict(LIVE_SAMPLE_LISTING)]

    monkeypatch.setattr(agentic_graph, 'search_live_listings', fake_live_search)

    search_response = client.post('/api/search', json={
        'location': 'Austin',
        'min_bedrooms': 2,
        'user_id': user_id,
    })
    search_payload = search_response.get_json()

    assert search_response.status_code == 200
    assert search_payload['count'] == 1
    assert search_payload['source'] == 'live_market'
    assert search_payload['properties'][0]['is_external'] is True
    assert search_payload['agent_summary']

    with app_module.app.app_context():
        history = app_module.SearchHistory.query.filter_by(user_id=user_id).all()
        assert len(history) == 1
        assert history[0].results_count == search_payload['count']


def test_register_login_and_favorite_flow(client):
    unique = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')
    username = f'pytest_user_{unique}'
    email = f'{username}@example.com'

    register_response = client.post('/api/users/register', json={
        'username': username,
        'email': email,
        'password': 'secret123',
        'user_type': 'buyer',
    })
    register_payload = register_response.get_json()

    assert register_response.status_code == 201
    assert register_payload['username'] == username

    login_response = client.post('/api/users/login', json={
        'username': username,
        'password': 'secret123',
    })
    login_payload = login_response.get_json()

    assert login_response.status_code == 200
    assert login_payload['email'] == email

    property_id = client.get('/api/properties').get_json()['properties'][0]['id']
    favorite_response = client.post(
        f"/api/users/{login_payload['id']}/favorites",
        json={'property_id': property_id},
    )

    assert favorite_response.status_code == 201

    favorites_response = client.get(f"/api/users/{login_payload['id']}/favorites")
    favorites_payload = favorites_response.get_json()

    assert favorites_response.status_code == 200
    assert len(favorites_payload) == 1
    assert favorites_payload[0]['property']['id'] == property_id


def test_property_crud_flow(client):
    create_response = client.post('/api/properties', json={
        'title': 'Pytest Listing',
        'description': 'Created during automated testing',
        'price': 350000,
        'location': 'Test City',
        'address': '42 Test Lane',
        'bedrooms': 3,
        'bathrooms': 2,
        'square_feet': 1600,
        'property_type': 'house',
        'status': 'available',
    })
    create_payload = create_response.get_json()

    assert create_response.status_code == 201
    assert create_payload['title'] == 'Pytest Listing'

    property_id = create_payload['id']
    update_response = client.put(f'/api/properties/{property_id}', json={
        'status': 'pending',
        'price': 365000,
    })
    update_payload = update_response.get_json()

    assert update_response.status_code == 200
    assert update_payload['status'] == 'pending'
    assert update_payload['price'] == 365000.0

    delete_response = client.delete(f'/api/properties/{property_id}')

    assert delete_response.status_code == 200
    assert delete_response.get_json()['message'] == 'Property deleted successfully.'


def test_validation_errors_return_json(client):
    unique = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')
    register_response = client.post('/api/users/register', json={
        'username': f'validator_{unique}',
        'email': f'validator_{unique}@example.com',
        'password': 'secret123',
        'user_type': 'buyer',
    })
    user_id = register_response.get_json()['id']

    bad_search = client.post('/api/search', json={})
    bad_filter = client.get('/api/properties?property_type=castle')
    bad_favorite = client.post(f'/api/users/{user_id}/favorites', json={})
    bad_ai_message = client.post('/api/ai/chat', json={'agent_id': 'buyer-guide'})
    bad_ai_agent = client.post('/api/ai/chat', json={'agent_id': 'unknown-agent', 'message': 'hello'})
    bad_search_agent = client.post('/api/search', json={'location': 'Austin', 'agent_id': 'unknown-agent'})

    assert bad_search.status_code == 400
    assert bad_search.get_json()['error'] == 'No search data provided.'
    assert bad_filter.status_code == 400
    assert 'property_type must be one of' in bad_filter.get_json()['error']
    assert bad_favorite.status_code == 400
    assert bad_favorite.get_json()['error'] == 'property_id is required.'
    assert bad_ai_message.status_code == 400
    assert bad_ai_message.get_json()['error'] == 'message is required.'
    assert bad_ai_agent.status_code == 400
    assert bad_ai_agent.get_json()['error'] == 'Unknown ai agent selected.'
    assert bad_search_agent.status_code == 400
    assert bad_search_agent.get_json()['error'] == 'Unknown ai agent selected.'
