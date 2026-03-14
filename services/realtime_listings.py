import hashlib
import json
import re
from datetime import datetime
from urllib import error, parse, request


PROPERTY_TYPE_ALIASES = {
    'single_family': 'house',
    'single-family': 'house',
    'single family': 'house',
    'house': 'house',
    'townhouse': 'townhouse',
    'townhome': 'townhouse',
    'condo': 'condo',
    'condominium': 'condo',
    'apartment': 'apartment',
    'multi_family': 'apartment',
    'multi-family': 'apartment',
    'loft': 'apartment',
}


STATE_ALIASES = {
    'new york': 'NY',
    'ny': 'NY',
    'new jersey': 'NJ',
    'nj': 'NJ',
    'california': 'CA',
    'ca': 'CA',
    'texas': 'TX',
    'tx': 'TX',
    'florida': 'FL',
    'fl': 'FL',
    'north carolina': 'NC',
    'nc': 'NC',
    'tennessee': 'TN',
    'tn': 'TN',
    'colorado': 'CO',
    'co': 'CO',
    'illinois': 'IL',
    'il': 'IL',
}


def _clean_text(value):
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _cover_style(property_type):
    palette = {
        'house': 'sunset-grove',
        'apartment': 'skyline-blue',
        'condo': 'amber-shore',
        'townhouse': 'terracotta',
    }
    return palette.get(property_type, 'sunset-grove')


def _safe_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_property_type(value):
    normalized = _clean_text(value)
    if not normalized:
        return 'house'
    normalized = normalized.lower().replace('/', ' ').replace('-', ' ').replace('_', ' ')
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return PROPERTY_TYPE_ALIASES.get(normalized, 'house')


def _normalize_state(value):
    normalized = _clean_text(value)
    if not normalized:
        return None
    normalized = re.sub(r'\s+', ' ', normalized.strip().lower())
    return STATE_ALIASES.get(normalized, normalized.upper())


def _parse_datetime(value):
    if not value:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    text = str(value)
    try:
        return datetime.fromisoformat(text.replace('Z', '+00:00')).isoformat()
    except ValueError:
        return None


def _build_lookup_key(provider, raw_id, address, location):
    if raw_id is None:
        digest = hashlib.sha1(f'{provider}:{address}:{location}'.encode('utf-8')).hexdigest()[:16]
        raw_id = digest
    return f'external:{provider}:{raw_id}'


def _build_location(raw):
    city = _clean_text(raw.get('city'))
    state = _clean_text(raw.get('state'))
    zip_code = _clean_text(raw.get('zipCode') or raw.get('postalCode'))
    parts = [part for part in [city, state] if part]
    if parts:
        location = ', '.join(parts)
        if zip_code:
            return f'{location} {zip_code}'
        return location
    county = _clean_text(raw.get('county'))
    return county or 'Live market result'


def compute_external_ai_score(listing):
    score = 40
    price = _safe_float(listing.get('price'))
    bedrooms = _safe_int(listing.get('bedrooms'))
    bathrooms = _safe_int(listing.get('bathrooms'))
    square_feet = _safe_int(listing.get('square_feet'))
    status = _clean_text(listing.get('status')) or 'available'
    property_type = _normalize_property_type(listing.get('property_type'))
    if 200000 <= price <= 700000:
        score += 18
    elif price > 0:
        score += 10
    if bedrooms >= 4:
        score += 14
    elif bedrooms >= 2:
        score += 10
    if bathrooms >= 2:
        score += 8
    if square_feet >= 2200:
        score += 10
    elif square_feet >= 1400:
        score += 6
    if status == 'available':
        score += 6
    if property_type == 'house':
        score += 4
    if listing.get('image_url'):
        score += 4
    return min(score, 100)


def normalize_rentcast_listing(raw):
    address = _clean_text(raw.get('formattedAddress') or raw.get('addressLine1') or raw.get('address')) or 'Address unavailable'
    location = _build_location(raw)
    property_type = _normalize_property_type(raw.get('propertyType') or raw.get('propertySubType'))
    lookup_key = _build_lookup_key(
        'rentcast',
        raw.get('id') or raw.get('listingId') or raw.get('mlsNumber'),
        address,
        location,
    )
    price = _safe_float(raw.get('price') or raw.get('listPrice'))
    bedrooms = _safe_int(raw.get('bedrooms') or raw.get('beds') or raw.get('bedroomsTotal'))
    bathrooms = _safe_int(raw.get('bathrooms') or raw.get('baths') or raw.get('bathroomsTotal'))
    square_feet = _safe_int(raw.get('squareFootage') or raw.get('livingArea') or raw.get('squareFeet'))
    image_url = None
    photos = raw.get('photos') or []
    if isinstance(photos, list) and photos:
        image_url = photos[0]
    image_url = image_url or _clean_text(raw.get('imgSrc') or raw.get('photo') or raw.get('photoUrl'))
    listing_url = _clean_text(raw.get('listingUrl') or raw.get('url'))
    description = _clean_text(raw.get('description'))
    title = _clean_text(raw.get('formattedAddress'))
    if not title:
        title = f'{bedrooms or 0} bed {property_type} in {location}'

    listing = {
        'id': lookup_key,
        'lookup_key': lookup_key,
        'title': title,
        'description': description or 'Live listing retrieved from the external market feed.',
        'price': price,
        'location': location,
        'address': address,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'square_feet': square_feet,
        'property_type': property_type,
        'status': 'available',
        'listing_date': _parse_datetime(raw.get('listedDate') or raw.get('lastSeenDate')),
        'image_url': image_url,
        'agent': None,
        'listing_source': 'live_market',
        'source_label': 'Live market feed',
        'provider': 'rentcast',
        'cover_style': _cover_style(property_type),
        'is_external': True,
        'can_favorite': False,
        'listing_url': listing_url,
        'city': _clean_text(raw.get('city')),
        'state': _clean_text(raw.get('state')),
        'zip_code': _clean_text(raw.get('zipCode') or raw.get('postalCode')),
        'data_source': 'rentcast',
    }
    listing['ai_score'] = compute_external_ai_score(listing)
    return listing


def build_rentcast_params(filters, limit=12):
    params = {'limit': max(1, min(int(limit or 12), 40))}
    location = _clean_text(filters.get('location'))
    if location:
        if re.fullmatch(r'\d{5}', location):
            params['zipCode'] = location
        else:
            parts = [part.strip() for part in location.split(',') if part.strip()]
            if len(parts) >= 2:
                params['city'] = parts[0]
                params['state'] = _normalize_state(parts[1])
            else:
                state = _normalize_state(location)
                if state in STATE_ALIASES.values():
                    params['state'] = state
                else:
                    params['city'] = location

    property_type = _clean_text(filters.get('property_type'))
    if property_type:
        params['propertyType'] = property_type

    min_price = filters.get('min_price')
    max_price = filters.get('max_price')
    min_bedrooms = filters.get('min_bedrooms')
    max_bedrooms = filters.get('max_bedrooms')
    status = _clean_text(filters.get('status'))

    if min_price not in (None, ''):
        params['minPrice'] = int(float(min_price))
    if max_price not in (None, ''):
        params['maxPrice'] = int(float(max_price))
    if min_bedrooms not in (None, ''):
        params['bedrooms'] = int(min_bedrooms)
    if max_bedrooms not in (None, ''):
        params['maxBedrooms'] = int(max_bedrooms)
    if status and status != 'available':
        params['status'] = status
    return params


def search_rentcast_listings(config, filters, limit=12):
    api_key = config.get('RENTCAST_API_KEY')
    if not api_key:
        raise RuntimeError('RENTCAST_API_KEY is not configured.')

    location = _clean_text(filters.get('location'))
    if not location:
        raise RuntimeError('A location is required to search the live market feed.')

    base_url = str(config.get('RENTCAST_BASE_URL') or 'https://api.rentcast.io/v1').rstrip('/')
    params = build_rentcast_params(filters, limit=limit)
    query = parse.urlencode(params, doseq=True)
    url = f'{base_url}/listings/sale?{query}'
    raw_request = request.Request(
        url,
        headers={
            'Accept': 'application/json',
            'X-Api-Key': api_key,
            'User-Agent': config.get('APP_PUBLIC_NAME', 'RealEstateAI/1.0'),
        },
        method='GET',
    )

    try:
        with request.urlopen(raw_request, timeout=config.get('AI_REQUEST_TIMEOUT', 20)) as response:
            payload = json.loads(response.read().decode('utf-8'))
    except error.HTTPError as exc:
        details = exc.read().decode('utf-8', errors='ignore')
        raise RuntimeError(f'Live listings provider returned {exc.code}: {details}') from exc
    except error.URLError as exc:
        raise RuntimeError(f'Live listings provider request failed: {exc.reason}') from exc

    listings = payload if isinstance(payload, list) else payload.get('results') or payload.get('listings') or []
    if not isinstance(listings, list):
        raise RuntimeError('Live listings provider returned an unexpected response shape.')

    return [normalize_rentcast_listing(item) for item in listings if isinstance(item, dict)]


def search_live_listings(config, filters, limit=12):
    provider = str(config.get('REAL_ESTATE_API_PROVIDER') or 'rentcast').lower()
    if provider != 'rentcast':
        raise RuntimeError(f'Unsupported real estate API provider: {provider}.')
    return search_rentcast_listings(config, filters, limit=limit)
