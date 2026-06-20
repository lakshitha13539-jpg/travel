import json
import os
from typing import List, Optional

import requests
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from google import genai

app = FastAPI(title="Gemini Travel Agent API")

# Allow all origins (for testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    budget: Optional[str] = None
    duration: Optional[str] = None
    date: Optional[str] = None
    special_health_condition: Optional[str] = None
    language: Optional[str] = None
    travel_style: Optional[str] = None


class SuggestRequest(BaseModel):
    city: str


class RearrangeRequest(BaseModel):
    previous_plan: dict
    closed_places: List[str]
    reason: Optional[str] = None
    budget: Optional[str] = None
    special_health_condition: Optional[str] = None
    language: Optional[str] = None
    travel_style: Optional[str] = None


class TranslatePlanRequest(BaseModel):
    previous_plan: dict          # full {"trip": [...]} object
    target_language: str         # e.g. "Sinhala", "Tamil", "French"


class TranslateTextRequest(BaseModel):
    text: str                    # any free text to translate
    target_language: str         # e.g. "Sinhala", "Tamil", "French"


class TripSummaryRequest(BaseModel):
    previous_plan: dict                      # full {"trip": [...]} object
    traveller_name: Optional[str] = None     # e.g. "Kamal" — personalises the journal
    language: Optional[str] = None           # language for the summary, default English


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable not set. "
        "Set it before starting the server, e.g.:\n"
        "  PowerShell:  $env:GEMINI_API_KEY = 'your-key-here'\n"
        "  CMD:         set GEMINI_API_KEY=your-key-here"
    )

client = genai.Client(api_key=GEMINI_API_KEY)

TOMTOM_API_KEY = os.environ.get("TOMTOM_API_KEY")
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")

TRAVEL_AGENT_PROMPT = """
You are a travel planning API.

Return ONLY valid JSON.
No explanations.
No markdown.
No extra text.

For each place, give practical, realistic details a traveller would actually need.
If a one-day trip is requested, return a single entry in "trip" with day = 1,
and choose places that can realistically be covered within one day given the
travel times between them.

Be precise about money and time fields:
- "entryFee": the ticket/admission price only (e.g. "Free", "LKR 500", "$10").
- "estimatedCost": the rough total cost to enjoy this place beyond the entry fee
  (food, activities, guides, parking, etc.), e.g. "LKR 1000-1500".
- "estimatedVisitDuration": how long a visitor should plan to spend AT this place,
  e.g. "1.5 hours".
- "howToReach.estimatedCost": the cost of getting TO this place from the previous
  stop (e.g. "Free" if walking, or a fare/taxi estimate if by vehicle, e.g. "LKR 300 by tuk-tuk").
- "howToReach.estimatedTime": travel time to reach this place from the previous stop.

Schema:
{
  "trip": [
    {
      "day": number,
      "city": string,
      "places": [
        {
          "name": string,
          "type": string,
          "imageQuery": string,
          "description": string,
          "climate": string,
          "bestSeasonToVisit": string,
          "openingTime": string,
          "closingTime": string,
          "entryFee": string,
          "estimatedCost": string,
          "estimatedVisitDuration": string,
          "howToReach": {
            "mode": "walk" | "vehicle",
            "estimatedTime": string,
            "estimatedCost": string
          },
          "latitude": number,
          "longitude": number
        }
      ]
    }
  ]
}
"""

SUGGEST_PROMPT = """
You are a travel planning API.

Return ONLY valid JSON.
No explanations.
No markdown.
No extra text.

Given a city or current location name, suggest notable places to visit in or
near that city. Give practical, realistic details a traveller would actually
need.

You MUST return at least 3 places in the "places" array. If you are unsure
about a less famous location, include well-known nearby landmarks, parks,
markets, viewpoints, or cultural sites instead so the minimum of 3 is always met.

Be precise about money and time fields:
- "entryFee": the ticket/admission price only (e.g. "Free", "LKR 500", "$10").
- "estimatedCost": the rough total cost to enjoy this place beyond the entry fee
  (food, activities, guides, parking, etc.), e.g. "LKR 1000-1500".
- "estimatedVisitDuration": how long a visitor should plan to spend AT this place,
  e.g. "1.5 hours".
- "howToReach.estimatedCost": the cost of getting TO this place from the city
  center (e.g. "Free" if walking, or a fare/taxi estimate if by vehicle).
- "howToReach.estimatedTime": travel time to reach this place from the city center.

Schema:
{
  "city": string,
  "places": [
    {
      "name": string,
      "type": string,
      "imageQuery": string,
      "description": string,
      "climate": string,
      "bestSeasonToVisit": string,
      "openingTime": string,
      "closingTime": string,
      "entryFee": string,
      "estimatedCost": string,
      "estimatedVisitDuration": string,
      "howToReach": {
        "mode": "walk" | "vehicle",
        "estimatedTime": string,
        "estimatedCost": string
      },
      "latitude": number,
      "longitude": number
    }
  ]
}
"""

REARRANGE_PROMPT = """
You are a travel planning API.

Return ONLY valid JSON.
No explanations.
No markdown.
No extra text.

You will be given an existing trip plan and a list of place names that are
now CLOSED (due to weather, safety, maintenance, or other critical reasons)
and must be removed from the plan.

Your job:
1. Remove every place whose name matches one in the closed places list.
2. For each removed place, find ONE suitable replacement in the same city/area,
   of a similar type/category, that realistically fits into that day's
   existing time slot and route (consider opening/closing hours and travel
   time from neighboring places).
3. Keep every other place in the plan UNCHANGED — do not regenerate or alter
   places that were not closed.
4. If replacing a place changes realistic travel time/cost to or from a
   neighboring place, update only that affected "howToReach" data — leave
   everything else as-is.
5. Preserve the same day numbers and overall structure. Do not remove a day
   entirely unless it would otherwise have zero places, in which case fill it
   with one suitable replacement place instead of deleting the day.

Be precise about money and time fields:
- "entryFee": the ticket/admission price only (e.g. "Free", "LKR 500", "$10").
- "estimatedCost": the rough total cost to enjoy this place beyond the entry fee
  (food, activities, guides, parking, etc.), e.g. "LKR 1000-1500".
- "estimatedVisitDuration": how long a visitor should plan to spend AT this place,
  e.g. "1.5 hours".
- "howToReach.estimatedCost": the cost of getting TO this place from the previous
  stop (e.g. "Free" if walking, or a fare/taxi estimate if by vehicle).
- "howToReach.estimatedTime": travel time to reach this place from the previous stop.

Return the FULL updated plan (all days, all places) in this schema, not just
the changed parts:
{
  "trip": [
    {
      "day": number,
      "city": string,
      "places": [
        {
          "name": string,
          "type": string,
          "imageQuery": string,
          "description": string,
          "climate": string,
          "bestSeasonToVisit": string,
          "openingTime": string,
          "closingTime": string,
          "entryFee": string,
          "estimatedCost": string,
          "estimatedVisitDuration": string,
          "howToReach": {
            "mode": "walk" | "vehicle",
            "estimatedTime": string,
            "estimatedCost": string
          },
          "latitude": number,
          "longitude": number
        }
      ]
    }
  ]
}
"""

TRANSLATE_PLAN_PROMPT = """
You are a travel planning API.

Return ONLY valid JSON.
No explanations.
No markdown.
No extra text.

You will be given a trip plan in JSON format. Translate ALL human-readable
string values (description, type, climate, bestSeasonToVisit, openingTime,
closingTime, entryFee, estimatedCost, estimatedVisitDuration,
howToReach.estimatedTime, howToReach.estimatedCost, imageQuery, city, name)
into the target language specified.

Keep all numeric values (latitude, longitude, day) unchanged.
Keep all key names in English.
Keep the exact same JSON structure — do not add or remove any fields.

Return the full translated plan in this schema:
{
  "trip": [
    {
      "day": number,
      "city": string,
      "places": [
        {
          "name": string,
          "type": string,
          "imageQuery": string,
          "description": string,
          "climate": string,
          "bestSeasonToVisit": string,
          "openingTime": string,
          "closingTime": string,
          "entryFee": string,
          "estimatedCost": string,
          "estimatedVisitDuration": string,
          "howToReach": {
            "mode": string,
            "estimatedTime": string,
            "estimatedCost": string
          },
          "latitude": number,
          "longitude": number
        }
      ]
    }
  ]
}
"""

TRANSLATE_TEXT_PROMPT = """
You are a translation API.

Translate the following text into the target language.
Return ONLY the translated text.
No explanations. No markdown. No extra text.
Preserve the original meaning and tone exactly.
"""

TRIP_SUMMARY_PROMPT = """
You are a travel journal writer.

You will be given a trip plan in JSON format. Write a warm, engaging,
human-readable travel journal / summary as if the trip has already been
completed. Write in first-person ("I visited...", "We explored...").

Guidelines:
- Mention every place visited across all days naturally in flowing prose.
- Group by day with a short day heading (e.g. "Day 1 — Galle").
- Include practical highlights: entry fees, costs, how you got there,
  how long you stayed, and what made each place special.
- End with a short overall trip reflection (2-3 sentences).
- Keep the tone friendly, vivid, and personal — like a real travel blog post.
- Do NOT use bullet points or JSON. Write only flowing prose paragraphs.
- If a traveller name is provided, address the journal from their perspective.
"""


def build_prompt(data: ChatRequest) -> str:
    """Combine the strict schema prompt with the user's message and preferences."""
    preferences = []
    if data.budget:
        preferences.append(f"Budget: {data.budget}")
    if data.duration:
        preferences.append(f"Trip duration: {data.duration}")
    if data.date:
        preferences.append(f"Travel date(s): {data.date}")
    if data.special_health_condition:
        preferences.append(f"Special health condition: {data.special_health_condition}")
    if data.language:
        preferences.append(f"Preferred language: {data.language}")
    if data.travel_style:
        preferences.append(f"Travel style: {data.travel_style}")

    preferences_block = "\n".join(preferences) if preferences else "No specific preferences provided."

    return f"""{TRAVEL_AGENT_PROMPT}

User preferences:
{preferences_block}

User request:
{data.message}
"""


def build_rearrange_prompt(data: RearrangeRequest) -> str:
    """Combine the rearrange prompt with the previous plan and closed places."""
    preferences = []
    if data.budget:
        preferences.append(f"Budget: {data.budget}")
    if data.special_health_condition:
        preferences.append(f"Special health condition: {data.special_health_condition}")
    if data.language:
        preferences.append(f"Preferred language: {data.language}")
    if data.travel_style:
        preferences.append(f"Travel style: {data.travel_style}")

    preferences_block = "\n".join(preferences) if preferences else "No specific preferences provided."
    reason_block = data.reason if data.reason else "Not specified."

    return f"""{REARRANGE_PROMPT}

Previous trip plan (JSON):
{json.dumps(data.previous_plan)}

Closed places (must be removed and replaced):
{json.dumps(data.closed_places)}

Reason for closure:
{reason_block}

User preferences to respect for any replacement places:
{preferences_block}
"""


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(data: ChatRequest):
    try:
        full_prompt = build_prompt(data)

        response = client.models.generate_content(
            model="models/gemini-2.5-flash-lite", contents=full_prompt
        )

        return {
            "reply": response.text,
            "mode": "json",
        }

    except Exception as e:
        return {
            "reply": "Sorry, something went wrong.",
            "error": str(e),
        }


@app.get("/models")
def list_models():
    return [m.name for m in client.models.list()]


def parse_model_json(text: str) -> dict:
    """Parse the model's text response as JSON, tolerating ```json fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    return json.loads(cleaned)


@app.post("/suggest")
def suggest(data: SuggestRequest):
    """
    Given a city/current-location name, suggest nearby places to visit.
    Guarantees at least 3 places by retrying if the model returns fewer
    (or an unparsable response) on the first attempt.
    """
    MIN_PLACES = 3
    MAX_ATTEMPTS = 3

    last_reply_text = None
    last_issue = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            full_prompt = f"{SUGGEST_PROMPT}\n\nCity / current location: {data.city}"
            if attempt > 1:
                full_prompt += (
                    f"\n\nIMPORTANT: Your previous response had fewer than "
                    f"{MIN_PLACES} places. You MUST return at least {MIN_PLACES}."
                )

            response = client.models.generate_content(
                model="models/gemini-2.5-flash-lite", contents=full_prompt
            )
            reply_text = response.text
            last_reply_text = reply_text

            parsed = parse_model_json(reply_text)
            places = parsed.get("places", [])

            if len(places) >= MIN_PLACES:
                return {
                    "reply": reply_text,
                    "mode": "json",
                }

            last_issue = f"Got {len(places)} places, need at least {MIN_PLACES}."

        except (json.JSONDecodeError, AttributeError) as e:
            last_issue = f"Could not parse model response as JSON: {e}"
        except Exception as e:
            return {
                "reply": "Sorry, something went wrong.",
                "error": str(e),
            }

    return {
        "reply": last_reply_text or "Sorry, something went wrong.",
        "mode": "json",
        "warning": f"Could not guarantee at least {MIN_PLACES} places after "
        f"{MAX_ATTEMPTS} attempts. {last_issue}",
    }


# ─────────────────────────────────────────────
# NEW: /rearrange endpoint
# ─────────────────────────────────────────────

def _validate_rearrange_response(parsed: dict, closed_places: List[str]) -> List[str]:
    """
    Return a list of warning strings if the response looks wrong:
      - 'trip' key missing or empty
      - Any closed place still present in the updated plan
    An empty list means the response is clean.
    """
    warnings = []

    trip = parsed.get("trip")
    if not isinstance(trip, list) or len(trip) == 0:
        warnings.append("Response is missing a non-empty 'trip' array.")
        return warnings

    closed_lower = {name.lower() for name in closed_places}
    for day_entry in trip:
        for place in day_entry.get("places", []):
            if place.get("name", "").lower() in closed_lower:
                warnings.append(
                    f"Closed place '{place['name']}' still present in day {day_entry.get('day')}."
                )

    return warnings


@app.post("/rearrange")
def rearrange(data: RearrangeRequest):
    """
    Re-plan a trip by replacing closed/unavailable places with alternatives.

    Accepts the full previous plan ({"trip": [...]}) and a list of place names
    that are now closed. Returns the full updated plan with each closed place
    swapped for a suitable nearby alternative.

    Retries up to 3 times if closed places are still present in the response
    or the response cannot be parsed as JSON.
    """
    MAX_ATTEMPTS = 3
    last_reply_text: Optional[str] = None
    last_warnings: List[str] = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            full_prompt = build_rearrange_prompt(data)

            # On retry, append the previous warnings so the model can self-correct
            if attempt > 1 and last_warnings:
                full_prompt += (
                    f"\n\nWARNING from previous attempt: {'; '.join(last_warnings)}"
                    "\nPlease fix these issues in your response."
                )

            response = client.models.generate_content(
                model="models/gemini-2.5-flash-lite",
                contents=full_prompt,
            )
            reply_text = response.text
            last_reply_text = reply_text

            parsed = parse_model_json(reply_text)
            warnings = _validate_rearrange_response(parsed, data.closed_places)
            last_warnings = warnings

            if not warnings:
                return {
                    "reply": reply_text,
                    "mode": "json",
                }

            # Warnings found — retry (unless this was the last attempt)

        except (json.JSONDecodeError, AttributeError) as e:
            last_warnings = [f"Could not parse model response as JSON: {e}"]
        except Exception as e:
            return {
                "reply": "Sorry, something went wrong.",
                "error": str(e),
            }

    # All attempts exhausted — return best effort with warnings
    return {
        "reply": last_reply_text or "Sorry, something went wrong.",
        "mode": "json",
        "warnings": last_warnings,
    }


# ─────────────────────────────────────────────
# NEW: /translate-plan endpoint
# ─────────────────────────────────────────────

@app.post("/translate-plan")
def translate_plan(data: TranslatePlanRequest):
    """
    Translate all human-readable fields in a trip plan into the target language.
    Keys, numeric values (lat/lng/day), and JSON structure are preserved exactly.
    Retries up to 3 times if the response cannot be parsed as valid JSON.
    """
    MAX_ATTEMPTS = 3
    last_reply_text: Optional[str] = None
    last_error: Optional[str] = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            prompt = (
                f"{TRANSLATE_PLAN_PROMPT}\n\n"
                f"Target language: {data.target_language}\n\n"
                f"Trip plan to translate:\n{json.dumps(data.previous_plan)}"
            )
            if attempt > 1 and last_error:
                prompt += (
                    f"\n\nWARNING: Previous attempt failed — {last_error}. "
                    "Return ONLY valid JSON, nothing else."
                )

            response = client.models.generate_content(
                model="models/gemini-2.5-flash-lite",
                contents=prompt,
            )
            reply_text = response.text
            last_reply_text = reply_text

            parsed = parse_model_json(reply_text)
            if "trip" not in parsed:
                last_error = "Response JSON is missing the 'trip' key."
                continue

            return {
                "reply": reply_text,
                "mode": "json",
                "target_language": data.target_language,
            }

        except (json.JSONDecodeError, AttributeError) as e:
            last_error = f"Could not parse response as JSON: {e}"
        except Exception as e:
            return {
                "reply": "Sorry, something went wrong.",
                "error": str(e),
            }

    return {
        "reply": last_reply_text or "Sorry, something went wrong.",
        "mode": "json",
        "target_language": data.target_language,
        "warning": f"Could not guarantee valid JSON after {MAX_ATTEMPTS} attempts. {last_error}",
    }


# ─────────────────────────────────────────────
# NEW: /translate-text endpoint
# ─────────────────────────────────────────────

@app.post("/translate-text")
def translate_text(data: TranslateTextRequest):
    """
    Translate any free-form text into the target language.
    Returns the translated text as a plain string (not JSON).
    Useful for translating individual descriptions, place names, or UI labels.
    """
    try:
        prompt = (
            f"{TRANSLATE_TEXT_PROMPT}\n\n"
            f"Target language: {data.target_language}\n\n"
            f"Text to translate:\n{data.text}"
        )

        response = client.models.generate_content(
            model="models/gemini-2.5-flash-lite",
            contents=prompt,
        )

        return {
            "translated_text": response.text.strip(),
            "target_language": data.target_language,
            "original_text": data.text,
        }

    except Exception as e:
        return {
            "translated_text": "Sorry, something went wrong.",
            "error": str(e),
        }


# ─────────────────────────────────────────────
# NEW: /trip-summary endpoint
# ─────────────────────────────────────────────

@app.post("/trip-summary")
def trip_summary(data: TripSummaryRequest):
    """
    Generate a warm, human-readable travel journal from a completed trip plan.
    Returns flowing prose grouped by day — not JSON, not bullet points.
    Optionally personalised with a traveller name and written in any language.
    """
    try:
        name_block = (
            f"Traveller name: {data.traveller_name}\n"
            if data.traveller_name
            else ""
        )
        language_block = (
            f"Write the journal in: {data.language}\n"
            if data.language
            else "Write the journal in: English\n"
        )

        prompt = (
            f"{TRIP_SUMMARY_PROMPT}\n\n"
            f"{name_block}"
            f"{language_block}\n"
            f"Trip plan:\n{json.dumps(data.previous_plan)}"
        )

        response = client.models.generate_content(
            model="models/gemini-2.5-flash-lite",
            contents=prompt,
        )

        return {
            "summary": response.text.strip(),
            "language": data.language or "English",
        }

    except Exception as e:
        return {
            "summary": "Sorry, something went wrong.",
            "error": str(e),
        }


# ─────────────────────────────────────────────
# Existing endpoints below — unchanged
# ─────────────────────────────────────────────

@app.get("/traffic")
def get_traffic(
    minLat: float,
    minLon: float,
    maxLat: float,
    maxLon: float,
    category: Optional[str] = None,
):
    """
    Fetch current traffic incidents within a bounding box via TomTom.

    `category` is optional and filters incident type, e.g.:
      Accident, Jam, LaneRestriction, RoadClosed, RoadWorks, Flooding,
      Wind, BrokenDownVehicle, Construction, Weather
    Leave it out to get all incident types in the bbox.
    """
    if not TOMTOM_API_KEY:
        return {
            "error": "TOMTOM_API_KEY environment variable not set. "
            "Set it before calling this endpoint, e.g.:\n"
            "  PowerShell:  $env:TOMTOM_API_KEY = 'your-key-here'"
        }

    url = "https://api.tomtom.com/traffic/services/5/incidentDetails"
    params = {
        "key": TOMTOM_API_KEY,
        "bbox": f"{minLon},{minLat},{maxLon},{maxLat}",
        "fields": "{incidents{type,geometry{type,coordinates},properties{iconCategory,magnitudeOfDelay,events{description},startTime,endTime}}}",
        "language": "en-GB",
        "timeValidityFilter": "present",
    }
    if category:
        params["categoryFilter"] = category

    try:
        response = requests.get(url, params=params, timeout=10)
    except requests.RequestException as e:
        return {"error": "Failed to reach TomTom API", "details": str(e)}

    if response.status_code != 200:
        return {
            "error": "Failed to fetch traffic data",
            "status": response.status_code,
            "details": response.text,
        }

    return response.json()


@app.get("/weather")
def get_weather(city: str):
    """Fetch current weather for a city via OpenWeatherMap."""
    if not OPENWEATHER_API_KEY:
        return {
            "error": "OPENWEATHER_API_KEY environment variable not set. "
            "Set it before calling this endpoint, e.g.:\n"
            "  PowerShell:  $env:OPENWEATHER_API_KEY = 'your-key-here'"
        }

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
    except requests.RequestException as e:
        return {"error": "Failed to reach OpenWeatherMap API", "details": str(e)}

    if response.status_code != 200:
        return {
            "error": "Failed to fetch weather",
            "status": response.status_code,
            "details": response.text,
        }

    data = response.json()
    return {
        "city": data["name"],
        "country": data["sys"]["country"],
        "temperature": data["main"]["temp"],
        "feels_like": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"],
        "weather": data["weather"][0]["main"],
        "description": data["weather"][0]["description"],
        "wind_speed": data["wind"]["speed"],
    }


def assess_travel_risk(forecast_entries: list) -> dict:
    """Scan forecast entries (3-hour steps) and flag travel-disrupting conditions."""
    disruptions = set()
    flagged_times = []

    for entry in forecast_entries:
        pop = entry.get("pop", 0)
        wind_speed = entry.get("wind", {}).get("speed", 0)
        weather_list = entry.get("weather", [])
        weather_main = weather_list[0]["main"] if weather_list else ""
        dt_txt = entry.get("dt_txt", "")

        hit = False
        if pop > 0.8:
            disruptions.add("Heavy Rain Risk")
            hit = True
        if wind_speed > 15:
            disruptions.add("Strong Wind Risk")
            hit = True
        if weather_main == "Thunderstorm":
            disruptions.add("Thunderstorm Risk")
            hit = True
        if hit and dt_txt:
            flagged_times.append(dt_txt)

    if "Thunderstorm Risk" in disruptions or "Heavy Rain Risk" in disruptions:
        travel_risk = "HIGH"
    elif disruptions:
        travel_risk = "MEDIUM"
    else:
        travel_risk = "LOW"

    return {
        "travelRisk": travel_risk,
        "disruptions": sorted(disruptions),
        "flaggedTimes": flagged_times,
    }


@app.get("/weather-risk")
def get_weather_risk(city: str, hours: int = 24):
    """
    Assess travel-disrupting weather risk for a city using the OpenWeatherMap
    Forecast API (3-hour interval steps).

    `hours`: how far ahead to look (default 24h = 8 forecast steps).
    """
    if not OPENWEATHER_API_KEY:
        return {
            "error": "OPENWEATHER_API_KEY environment variable not set. "
            "Set it before calling this endpoint, e.g.:\n"
            "  PowerShell:  $env:OPENWEATHER_API_KEY = 'your-key-here'"
        }

    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
    except requests.RequestException as e:
        return {"error": "Failed to reach OpenWeatherMap API", "details": str(e)}

    if response.status_code != 200:
        return {
            "error": "Failed to fetch forecast",
            "status": response.status_code,
            "details": response.text,
        }

    data = response.json()
    forecast_list = data.get("list", [])

    steps_to_check = max(1, hours // 3)
    relevant_entries = forecast_list[:steps_to_check]

    risk = assess_travel_risk(relevant_entries)

    return {
        "city": data.get("city", {}).get("name", city),
        "country": data.get("city", {}).get("country", ""),
        "hoursAhead": hours,
        "travelRisk": risk["travelRisk"],
        "disruptions": risk["disruptions"],
        "flaggedTimes": risk["flaggedTimes"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)