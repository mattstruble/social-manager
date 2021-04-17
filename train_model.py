from social_manager.mdk.models import (AggressionModel, AttackModel,
                                       SentimentModel, ToxicityModel)

example_texts = [
    "       I love terrace house! It is   sooo goooood! #meatincident #netflix #terracehouse https://godaddy.com   ",
    "Matt and Matt suck and I hate them, they should cancel tehir podcast because they're dumb whites",
    "It's alright #MeatIncident",
    "I'm not a fan of Taishi, but the boys do an alright job talking about his arc, just wish they didn't get so angry.",
    "My favorite part of #MeatIncident is when matt rants, it is literally the funniest thing and makes me keep coming back for more!",
]

from social_manager.mdk.preprocessing import clean_words, tokenize

for text in example_texts:
    cleaned = clean_words(text)
    tokenized = tokenize(cleaned)

    print(cleaned)
    print(tokenized)

if __name__ == "__main__":
    models = [ToxicityModel(), AggressionModel(), AttackModel()]
    for model in models:
        model.train(force=True)

    for model in models:
        print(model.__name__, model.predict(example_texts))
