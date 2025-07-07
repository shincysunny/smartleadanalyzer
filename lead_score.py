# lead_score.py

#import openai
import joblib
from sklearn.exceptions import NotFittedError
import os
import pandas as pd

# Load OpenAI API key from environment variable
#openai.api_key = os.getenv("OPENAI_API_KEY")

# GPT-powered scoring function
#def gpt_score_description(description):
    #if pd.isna(description) or description.strip() == "":
        #return 0, "No description", "Ignore"

#    prompt = (
 #       f"Rate this company (1–10) for lead generation value, based on this description:\n"
  #      f"\"{description}\"\n\n"
   #     f"Then explain the score in 1 sentence.\n"
    #    f"Format:\nScore: <number>\nReason: <1-line explanation>"
    #)

    #try:
     #   response = openai.ChatCompletion.create(
      #      model="gpt-3.5-turbo",
       #     messages=[
        #        {"role": "user", "content": prompt}
            #],
         #   max_tokens=60,
          #  temperature=0.5
     #   )

    #     text = response['choices'][0]['message']['content']
    #     lines = text.strip().split("\n")
    #     score = 5
    #     reason = "No reason found"
    #     for line in lines:
    #         if line.lower().startswith("score:"):
    #             score = int(line.split(":")[1].strip())
    #         elif line.lower().startswith("reason:"):
    #             reason = line.split(":", 1)[1].strip()

    #     if score >= 8:
    #         tag = "Hot"
    #     elif score >= 5:
    #         tag = "Research"
    #     else:
    #         tag = "Ignore"

    #     return score, reason, tag

    # except Exception as e:
    #     print("GPT ERROR:", e)
    #     return 5, "GPT error", "Research"

# Email checker function
def check_email(email):
    if pd.isna(email) or email.strip() == "":
        return "Missing"
    elif any(x in email.lower() for x in ['no-email', '.xyz', '@test.', '@example.']):
        return "Suspicious"
    else:
        return "Valid"
def predict_lead_tag_ml(df):
    try:
        model = joblib.load("lead_classifier.joblib")
    except FileNotFoundError:
        print("Model file not found. Run ml_model.py first.")
        df["ML Tag"] = "Unknown"
        return df
    except NotFittedError:
        print("Model not fitted.")
        df["ML Tag"] = "Unknown"
        return df

    descriptions = df["Description"].fillna("")
    predicted_tags = model.predict(descriptions)
    df["ML Tag"] = predicted_tags
    return df

# Main analyzer function
def analyze_leads(df):
    # cores = []
    # reasons = []
    # tags = []
    email_statuses = []

    for index, row in df.iterrows():
        #desc = row.get("Description", "")
        email = row.get("Email", "")

        #score, reason, tag = gpt_score_description(desc)
        email_status = check_email(email)

        # scores.append(score)
        # reasons.append(reason)
        # tags.append(tag)
        email_statuses.append(email_status)

    # df["GPT Score"] = scores
    # df["Score Reason"] = reasons
    # df["Lead Tag"] = tags
    df["Email Status"] = email_statuses
    df = predict_lead_tag_ml(df)


    return df
