import requests


def requestCall():
    key = "b58c0a99-b240-4f7c-8d85-87525e266e40"
    secret = "I87dtwaNCUSmNQcCSZcP6g=="
    fromNumber = "+447520651004"
    to = "+919260927430"
    locale = "en-US"
    url = "https://calling.api.sinch.com/calling/v1/callouts"
    payload = {
        "method": "ttsCallout",
        "ttsCallout": {
            "cli": fromNumber,
            "destination": {
                "type": "number",
                "endpoint": to
            },
            "locale": locale,
            "text": "Fire Alert Fire Alert Fire Alert, details are shared on email"
        }
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(
        url, json=payload, headers=headers, auth=(key, secret))
