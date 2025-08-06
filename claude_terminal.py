import anthropic
import os

client = anthropic.Anthropic(api_key=os.environ[9"])

message = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=1000,
    temperature=0.7,
    messages=[
        {"role": "user", "content": "Hola Claude, ¿puedes contarme un chiste?"}
    ]
)

print(message.content[0].text)
