from twilio.rest import Client
import keys

client = Client(keys.account_sid, keys.auth_token)

message = client.messages.create(
    body="This is a test message",
    from_=keys.twilio_number,
    to=keys.target_number
)
