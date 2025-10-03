# %% [markdown]
# # OpenAI Client
# This section explores random data generation.

# %%
import os
api_key=os.getenv("OPEN_API_KEY")
if not api_key:
    api_key=input("Enter your OpenAI API key: ").strip()
    os.environ["OPENAI_API_KEY"] = api_key
    if not api_key:
        raise EnvironmentError("No API key provided")

# %%
from openai import OpenAI
client=OpenAI(api_key=api_key)
response=client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content": "Can you print 10 random words?"}]
)
print(response.choices[0].message.content)
# %%