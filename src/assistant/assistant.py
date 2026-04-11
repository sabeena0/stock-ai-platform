from openai import OpenAI
import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def generate_explanation(ticker, prediction):

    trend = "upward 📈" if prediction == 1 else "downward 📉"

    prompt = f"""
    You are a financial assistant.

    Explain this stock prediction in simple terms:

    Stock: {ticker.upper()}
    Predicted Trend: {trend}

    Also mention risk in a simple way.
    Keep it short and beginner-friendly.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful financial assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content
    except Exception:
        # ✅ fallback response
        return f"The stock {ticker.upper()} is expected to move in a {trend} trend based on recent patterns. Market conditions may vary, so risk is involved."
    