import requests


def notify_app():
    requests.post("https://ntfy.sh/willweste_surveillance",
                  data="Activity Detected",
                  headers={
                      "Title": "A person has been detected at the door",
                      "Priority": "urgent",
                      "Tags": "warning"
                  })
